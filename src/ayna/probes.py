"""Ground-truth-free probes: tier 0 (outputs only) and tier 1 (dynamics callable).

These are ayna's own additions to the CtD metric set. Each returns a plain
dict of evidence; :mod:`ayna.detectors` turns evidence into findings.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from .bundle import Bundle
from .metrics import bits_per_spike, vw_r2


# ------------------------------------------------------------------ helpers
def lagged(arr: np.ndarray, n_lags: int) -> np.ndarray:
    """(R, T, F) -> (R, T, F * n_lags) with lags 0..n_lags-1, zero-padded per trial."""
    R, T, F = arr.shape
    out = np.zeros((R, T, F * n_lags), dtype=float)
    for lag in range(n_lags):
        out[:, lag:, lag * F:(lag + 1) * F] = arr[:, : T - lag, :]
    return out


def poisson_glm_predict(X_fit, Y_fit, X_eval, l2: float = 1e-3, iters: int = 25):
    """Independent Poisson GLMs (log link, ridge) for every column of ``Y_fit``, by Newton's method."""
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-12
    Xf = np.c_[np.ones(len(X_fit)), (X_fit - mu) / sd]
    Xe = np.c_[np.ones(len(X_eval)), (X_eval - mu) / sd]
    P_ = Xf.shape[1]
    reg = l2 * len(Xf) * np.eye(P_)
    reg[0, 0] = 0.0
    pred = np.empty((len(X_eval), Y_fit.shape[1]))
    for j in range(Y_fit.shape[1]):
        y = Y_fit[:, j].astype(float)
        w = np.zeros(P_)
        w[0] = np.log(max(y.mean(), 1e-6))
        for _ in range(iters):
            lam = np.exp(np.clip(Xf @ w, -20, 20))
            grad = Xf.T @ (y - lam) - reg @ w
            hess = (Xf * lam[:, None]).T @ Xf + reg
            step = np.linalg.solve(hess, grad)
            w += step
            if np.max(np.abs(step)) < 1e-6:
                break
        pred[:, j] = np.exp(np.clip(Xe @ w, -20, 20))
    return pred


# --------------------------------------------------- reconstruction probes
def baseline_co_bps(b: Bundle, sigmas=(1.0, 2.0, 4.0), n_pcs: int = 10, seed: int = 0) -> dict:
    """co-BPS of a deliberately simple reference model.

    Smoothed, square-rooted held-in spikes -> PCA -> one Poisson GLM per
    held-out neuron. The smoothing width is chosen on a split *inside* the
    training trials. A dynamics model that cannot beat this has not captured
    the shared structure the data plainly contain.
    """
    hi, ho = b.heldin, b.heldout
    feats_all = {}
    for s in sigmas:
        sm = np.sqrt(gaussian_filter1d(b.spikes[:, :, hi].astype(float), s, axis=1))
        feats_all[s] = sm

    train_trials = np.where(b.trials("train"))[0]
    rng = np.random.default_rng(seed)
    inner_val = np.zeros(b.n_trials, bool)
    inner_val[rng.choice(train_trials, max(1, len(train_trials) // 4), replace=False)] = True
    inner_train = b.trials("train") & ~inner_val

    def fit_eval(feat, fit_trials, eval_trials):
        m_fit = b.bin_mask("all") & fit_trials[:, None]
        m_eval = b.bin_mask("all") & eval_trials[:, None]
        X_fit, X_eval = feat[m_fit], feat[m_eval]
        k = min(n_pcs, X_fit.shape[1])
        pca = PCA(k).fit(X_fit)
        X_fit, X_eval = pca.transform(X_fit), pca.transform(X_eval)
        Y_fit = b.spikes[:, :, ho][m_fit]
        Y_eval = b.spikes[:, :, ho][m_eval]
        pred = poisson_glm_predict(X_fit, Y_fit, X_eval)
        return bits_per_spike(pred, Y_eval)

    scores = {s: fit_eval(feats_all[s], inner_train, inner_val) for s in sigmas}
    best = max(scores, key=lambda s: -np.inf if np.isnan(scores[s]) else scores[s])
    value = fit_eval(feats_all[best], b.trials("train"), b.trials("val"))
    return {"baseline_co_bps": value, "baseline_sigma_bins": best}


def residual_structure(b: Bundle, sigma: float = 2.0, n_null: int = 50, seed: int = 0) -> dict:
    """Is there shared, slow structure left in the residuals?

    Pearson residuals ``(y - rate) / sqrt(rate)`` on validation trials are
    smoothed in time. If the model captured every shared latent, what is left
    is independent Poisson noise and the top eigenvalue of the residual
    covariance looks like the null made by circularly shifting each neuron's
    residuals independently. Missed latents leave a shared component.
    """
    rng = np.random.default_rng(seed)
    trials = b.trials("val")
    y = b.spikes[trials].astype(float)
    lam = b.rates[trials]
    res = (y - lam) / np.sqrt(lam)
    if b.trial_lens is not None:
        valid = (np.arange(b.n_bins)[None, :] < b.trial_lens[trials][:, None])[..., None]
        res = res * valid
    res = gaussian_filter1d(res, sigma, axis=1)
    keep = res.reshape(-1, res.shape[2]).std(axis=0) > 0
    res = res[:, :, keep]

    def top_eig(r):
        X = r.reshape(-1, r.shape[2])
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
        return ev[-1] / X.shape[1]

    observed = top_eig(res)
    null = []
    T = res.shape[1]
    for _ in range(n_null):
        shifted = np.empty_like(res)
        for j in range(res.shape[2]):
            shifts = rng.integers(0, T, size=res.shape[0])
            idx = (np.arange(T)[None, :] + shifts[:, None]) % T
            shifted[:, :, j] = np.take_along_axis(res[:, :, j], idx, axis=1)
        null.append(top_eig(shifted))
    null = np.asarray(null)
    ratio = observed / np.quantile(null, 0.95)

    # where: when in the trial, and which neurons, carry the shared residual
    X = res.reshape(-1, res.shape[2])
    pc = PCA(1).fit(X)
    score = pc.transform(X).reshape(res.shape[0], res.shape[1])
    time_profile = np.mean(score**2, axis=0)
    loadings = np.abs(pc.components_[0])
    neuron_ids = np.where(keep)[0]
    return {
        "residual_top_eig": float(observed),
        "residual_null_q95": float(np.quantile(null, 0.95)),
        "residual_ratio": float(ratio),
        "residual_peak_bins": _peak_window(time_profile),
        "residual_top_neurons": [int(n) for n in neuron_ids[np.argsort(loadings)[::-1][:5]]],
    }


def _peak_window(profile: np.ndarray, frac: float = 0.5) -> list:
    """Contiguous window around the peak where the profile exceeds ``frac`` of it."""
    p = int(np.argmax(profile))
    thr = frac * profile[p]
    lo = p
    while lo > 0 and profile[lo - 1] >= thr:
        lo -= 1
    hi = p
    while hi < len(profile) - 1 and profile[hi + 1] >= thr:
        hi += 1
    return [lo, hi]


# ------------------------------------------------------------ input probes
def input_leakage(b: Bundle) -> dict:
    """How well do the inferred latents, at the same instant, predict the inferred inputs?

    External inputs should arrive from outside the circuit. When a model's
    inferred inputs can be read off its own current state, the controller is
    likely carrying dynamics the generator should own -- the pattern behind
    CtD Fig 6B, where badly inferred inputs "resembled the state of the network".

    Caveat: in closed-loop tasks (e.g. sensory feedback of the model's own
    output) true inputs legitimately depend on state.
    """
    Z_tr, Z_va = b.flat("latents", "train"), b.flat("latents", "val")
    U_tr, U_va = b.flat("inputs", "train"), b.flat("inputs", "val")
    model = LinearRegression().fit(Z_tr, U_tr)
    pred = model.predict(Z_va)
    per_dim = np.atleast_1d(r2_score(U_va, pred, multioutput="raw_values"))
    # temporal smoothness: lag-1 autocorrelation (pulses ~0, state-like ~1)
    U = b.inputs - b.inputs.mean(axis=(0, 1), keepdims=True)
    ac = np.sum(U[:, 1:] * U[:, :-1], axis=(0, 1)) / (np.sum(U**2, axis=(0, 1)) + 1e-12)
    flat = U.reshape(-1, U.shape[2])
    kurt = np.mean(flat**4, 0) / (np.mean(flat**2, 0) ** 2 + 1e-12) - 3
    return {
        "input_leakage_r2": vw_r2(U_va, pred),
        "input_leakage_per_dim": [float(x) for x in per_dim],
        "input_autocorr_lag1": [float(x) for x in ac],
        "input_excess_kurtosis": [float(x) for x in kurt],
    }


def input_inherited(b: Bundle, n_lags: int = 10, alpha: float = 1.0) -> dict:
    """Share of latent variance that is a linear filter of the last ``n_lags`` input bins.

    When inputs are supplied and rich (CtD's RandomTarget), latents can look
    right because they copy the inputs. That part says nothing about the
    model's intrinsic dynamics.
    """
    X = lagged(b.inputs.astype(float), n_lags)
    X_tr, X_va = b.flat(X, "train"), b.flat(X, "val")
    Z_tr, Z_va = b.flat("latents", "train"), b.flat("latents", "val")
    model = Ridge(alpha=alpha).fit(X_tr, Z_tr)
    out = {"input_filter_r2": vw_r2(Z_va, model.predict(X_va)), "input_filter_lags": n_lags}
    if b.true_latents is not None:
        Zt_tr, Zt_va = b.flat("true_latents", "train"), b.flat("true_latents", "val")
        out["true_input_filter_r2"] = vw_r2(Zt_va, Ridge(alpha=alpha).fit(X_tr, Zt_tr).predict(X_va))
    return out


# --------------------------------------------------------- tier 1 probes
def autonomous_rollout(b: Bundle, horizon: int = 10, max_starts: int = 4000, seed: int = 0) -> dict:
    """Can the model's own dynamics hold the state without its inputs?

    From inferred states on validation trials, roll the dynamics ``horizon``
    steps with inputs (a) as inferred and (b) clamped to zero, and compare
    both to simply holding the starting state. If holding the state predicts
    the future far better than the model's own autonomous rollout, the
    persistence in the data is being produced by the inputs, not the dynamics.
    """
    rng = np.random.default_rng(seed)
    f = b.dynamics
    Z = b.latents
    U = b.inputs if b.inputs is not None else np.zeros(Z.shape[:2] + (1,))
    trials = np.where(b.trials("val"))[0]
    starts = []
    for r in trials:
        last = (b.trial_lens[r] if b.trial_lens is not None else b.n_bins) - horizon
        for t in range(0, max(last, 0)):
            starts.append((r, t))
    if not starts:
        return {}
    starts = np.asarray(starts)
    if len(starts) > max_starts:
        starts = starts[rng.choice(len(starts), max_starts, replace=False)]
    r, t = starts[:, 0], starts[:, 1]
    z_full = Z[r, t].copy()
    z_auto = Z[r, t].copy()
    zero = np.zeros_like(U[r, t])
    for k in range(horizon):
        z_full = np.asarray(f(z_full, U[r, t + k]))
        z_auto = np.asarray(f(z_auto, zero))
    target = Z[r, t + horizon]
    hold = Z[r, t]
    return {
        "rollout_horizon": horizon,
        "rollout_r2_full": vw_r2(target, z_full),
        "rollout_r2_autonomous": vw_r2(target, z_auto),
        "rollout_r2_hold": vw_r2(target, hold),
    }


def zero_input_fixed_points(b: Bundle, n_starts: int = 256, steps: int = 400, lr: float = 0.05,
                            tol: float = 1e-6, seed: int = 0) -> dict:
    """Fixed points of ``z -> f(z, 0)``, found by minimising ``q = |f(z,0) - z|^2 / 2``.

    Batched finite-difference gradients keep this framework-free. Stability
    uses the eigenvalues of the discrete-time Jacobian (stable if all |λ| < 1).
    """
    rng = np.random.default_rng(seed)
    f = b.dynamics
    Zall = b.flat("latents", "val")
    U_dim = b.inputs.shape[2] if b.inputs is not None else 1
    z = Zall[rng.choice(len(Zall), min(n_starts, len(Zall)), replace=False)].astype(float)
    z = z + 0.05 * Zall.std(0) * rng.standard_normal(z.shape)
    zero = np.zeros((len(z), U_dim))
    D = z.shape[1]
    scale = Zall.std() + 1e-12
    h = 1e-4 * scale

    def q(zz):
        d = np.asarray(f(zz, zero)) - zz
        return 0.5 * np.sum(d * d, axis=1)

    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for i in range(1, steps + 1):
        base = q(z)
        grad = np.empty_like(z)
        for d in range(D):
            e = np.zeros(D)
            e[d] = h
            grad[:, d] = (q(z + e) - base) / h
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        z = z - lr * scale * (m / (1 - 0.9**i)) / (np.sqrt(v / (1 - 0.999**i)) + 1e-12)
    qs = q(z) / scale**2
    found = z[qs < tol]
    uniq = []
    for p in found:
        if all(np.linalg.norm(p - u) > 0.05 * scale * np.sqrt(D) for u in uniq):
            uniq.append(p)
    n_stable = 0
    for p in uniq:
        J = np.empty((D, D))
        f0 = np.asarray(f(p[None], np.zeros((1, U_dim))))[0]
        for d in range(D):
            e = np.zeros(D)
            e[d] = h
            J[:, d] = (np.asarray(f((p + e)[None], np.zeros((1, U_dim))))[0] - f0) / h
        if np.max(np.abs(np.linalg.eigvals(J))) < 1:
            n_stable += 1
    return {"fixed_points": len(uniq), "stable_fixed_points": n_stable}


def occupied_clusters(b: Bundle, max_k: int = 12, seed: int = 0) -> int:
    """Rough count of distinct regions the latent state dwells in (for FP comparison)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    Z = b.flat("latents", "val")
    Z = Z[rng.choice(len(Z), min(3000, len(Z)), replace=False)]
    best_k, best_s = 1, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(k, n_init=3, random_state=seed).fit_predict(Z)
        s = silhouette_score(Z, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k if best_s > 0.5 else 1
