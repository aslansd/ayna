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


def poisson_glm_predict(X_fit, Y_fit, X_eval, l2: float = 1e-3, iters: int = 25,
                        offset_fit=None, offset_eval=None):
    """Independent Poisson GLMs (log link, ridge) for every column of ``Y_fit``, by Newton's method.

    ``offset_fit`` / ``offset_eval`` (same shape as ``Y``) are fixed log-rate
    offsets: the GLM then learns only what the features add on top of them.
    """
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-12
    Xf = np.c_[np.ones(len(X_fit)), (X_fit - mu) / sd]
    Xe = np.c_[np.ones(len(X_eval)), (X_eval - mu) / sd]
    P_ = Xf.shape[1]
    reg = l2 * len(Xf) * np.eye(P_)
    reg[0, 0] = 0.0
    pred = np.empty((len(X_eval), Y_fit.shape[1]))
    for j in range(Y_fit.shape[1]):
        y = Y_fit[:, j].astype(float)
        o_f = 0.0 if offset_fit is None else offset_fit[:, j]
        o_e = 0.0 if offset_eval is None else offset_eval[:, j]
        w = np.zeros(P_)
        if offset_fit is None:
            w[0] = np.log(max(y.mean(), 1e-6))
        for _ in range(iters):
            lam = np.exp(np.clip(o_f + Xf @ w, -20, 20))
            grad = Xf.T @ (y - lam) - reg @ w
            hess = (Xf * lam[:, None]).T @ Xf + reg
            step = np.linalg.solve(hess, grad)
            w += step
            if np.max(np.abs(step)) < 1e-6:
                break
        pred[:, j] = np.exp(np.clip(o_e + Xe @ w, -20, 20))
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


def recoverable_information(b: Bundle, sigma: float = 2.0, n_pcs: int = 10, l2: float = 1e-2,
                            seed: int = 0) -> dict:
    """Information about one set of neurons, present in another set's spikes, that the model left unused.

    For each target neuron, a Poisson GLM takes the model's own log-rate as a
    fixed offset and adds smoothed, square-rooted spikes of *source* neurons as
    features. The gain in validation bits per spike over the model alone is
    information the data carry and the model missed. Poisson noise is
    independent across neurons, so source spikes cannot predict target noise.

    The same fit with source features taken from *other trials* (shuffled
    trial identity, same time structure) is subtracted, so per-neuron gain or
    time-locked corrections do not count as recoverable information.

    Source/target: held-in -> held-out when held-out neurons exist; otherwise
    two random halves of the neurons, both directions, averaged.
    """
    rng = np.random.default_rng(seed)
    N = b.n_neurons
    if b.heldout is not None:
        pairs = [(np.where(b.heldin)[0], np.where(b.heldout)[0])]
        split = "held-in -> held-out"
    else:
        perm = rng.permutation(N)
        a, c = np.sort(perm[: N // 2]), np.sort(perm[N // 2:])
        pairs = [(a, c), (c, a)]
        split = "neuron halves, both directions"
    tr, va = b.bin_mask("train"), b.bin_mask("val")
    shuffle = np.arange(b.n_trials)
    trn, van = np.where(b.trials("train"))[0], np.where(b.trials("val"))[0]
    shuffle[trn] = rng.permutation(trn)
    shuffle[van] = rng.permutation(van)
    trial_of_sample = np.nonzero(va)[0]
    gains, controls, model_bps, n_spikes = [], [], [], []
    per_trial_net = np.zeros(b.n_trials)
    for src, tgt in pairs:
        feat = np.sqrt(gaussian_filter1d(b.spikes[:, :, src].astype(float), sigma, axis=1))
        Y_tr = b.spikes[:, :, tgt][tr].astype(float)
        Y_va = b.spikes[:, :, tgt][va].astype(float)
        O_tr = np.log(np.maximum(b.rates[:, :, tgt][tr], 1e-9))
        O_va = np.log(np.maximum(b.rates[:, :, tgt][va], 1e-9))
        base = np.exp(O_va)
        ll_base = np.sum(Y_va * O_va - base, axis=1)
        nsp = Y_va.sum()
        out = []
        for f in (feat, feat[shuffle]):
            k = min(n_pcs, len(src))
            pca = PCA(k).fit(f[tr][:: max(1, int(tr.sum()) // 50000)])
            pred = poisson_glm_predict(pca.transform(f[tr]), Y_tr, pca.transform(f[va]), l2=l2,
                                       offset_fit=O_tr, offset_eval=O_va)
            out.append(np.sum(Y_va * np.log(pred) - pred, axis=1) - ll_base)   # per-sample LL gain
        gains.append(out[0].sum() / (nsp * np.log(2)))
        controls.append(out[1].sum() / (nsp * np.log(2)))
        per_trial_net += np.bincount(trial_of_sample, weights=out[0] - out[1], minlength=b.n_trials)
        model_bps.append(bits_per_spike(base, Y_va))
        n_spikes.append(nsp)
    w = np.asarray(n_spikes) / np.sum(n_spikes)
    gain = float(np.dot(w, gains))
    control = float(np.dot(w, controls))
    mbps = float(np.dot(w, model_bps))
    net = gain - control
    # paired across validation trials: is the net gain distinguishable from zero?
    d = per_trial_net[b.trials("val")]
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else 0.0
    return {
        "recoverable_bits_per_spike": net,
        "recoverable_gain_raw": gain,
        "recoverable_gain_shuffled": control,
        "recoverable_reference_bps": mbps,
        "recoverable_fraction": net / mbps if mbps > 0 else float("nan"),
        "recoverable_z": z,
        "recoverable_split": split,
    }


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
def input_leakage(b: Bundle, lag: int = 1, var: float = 0.99, max_dims: int = 20) -> dict:
    """How well does the latent state *before* each input predict that input?

    A genuine external input arrives from outside the circuit, so the state
    that precedes it should say little about it. When a model offloads its
    dynamics to inferred inputs, the inputs become a function of the state --
    the pattern behind CtD Fig 6B, where badly inferred inputs "resembled the
    state of the network".

    The state at ``t - lag`` is used because most models update
    ``z_t = f(z_{t-1}, u_t)``: the state at ``t`` already contains the input,
    and with enough latent dimensions it can be decoded back from it (on
    CtD's LFADS models, same-instant decoding gives R² 0.88-0.99 whether or
    not the inputs are right). Latents are reduced to the PCs explaining
    ``var`` of their variance (at most ``max_dims``) to keep the regression
    honest in high dimensions; fit on training trials, scored on validation.

    Caveat: in closed-loop tasks (sensory feedback of the model's own output)
    true inputs legitimately depend on the preceding state.
    """
    Z, U = b.latents, b.inputs
    Zp = np.zeros_like(Z)
    Zp[:, lag:] = Z[:, : Z.shape[1] - lag]
    tr, va = b.bin_mask("train").copy(), b.bin_mask("val").copy()
    tr[:, :lag] = False
    va[:, :lag] = False
    sub = max(1, int(tr.sum()) // 50000)
    pca = PCA().fit(Zp[tr][::sub])
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var)) + 1
    k = max(1, min(k, max_dims, Z.shape[2]))
    pca = PCA(k).fit(Zp[tr][::sub])
    model = Ridge(alpha=1.0).fit(pca.transform(Zp[tr]), U[tr])
    pred = model.predict(pca.transform(Zp[va]))
    per_dim = np.atleast_1d(r2_score(U[va], pred, multioutput="raw_values"))
    # same-instant decoding, kept as evidence only
    same = LinearRegression().fit(b.flat("latents", "train"), b.flat("inputs", "train"))
    same_r2 = vw_r2(b.flat("inputs", "val"), same.predict(b.flat("latents", "val")))
    Uc = U - U.mean(axis=(0, 1), keepdims=True)
    ac = np.sum(Uc[:, 1:] * Uc[:, :-1], axis=(0, 1)) / (np.sum(Uc**2, axis=(0, 1)) + 1e-12)
    return {
        "input_leakage_r2": vw_r2(U[va], pred),
        "input_leakage_per_dim": [float(x) for x in per_dim],
        "input_leakage_lag": lag,
        "input_leakage_pcs": k,
        "input_leakage_same_instant_r2": same_r2,
        "input_autocorr_lag1": [float(x) for x in ac],
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
                            tol: float = 1e-6, seed: int = 0, max_dims: int = 32) -> dict:
    """Fixed points of ``z -> f(z, 0)``, found by minimising ``q = |f(z,0) - z|^2 / 2``.

    Gradients come from the dynamics itself when it offers them: a callable
    with a ``vjp(z, u, v)`` attribute (vector-Jacobian product, ``J^T v``) and
    a ``jacobian(z, u)`` attribute, as the CtD wrappers provide through torch
    autograd. Otherwise batched finite differences are used, which cost one
    call per latent dimension and are skipped above ``max_dims``.
    Stability uses the eigenvalues of the discrete-time Jacobian (stable if
    all |λ| < 1). ``tol`` is on q relative to the latent variance.
    """
    rng = np.random.default_rng(seed)
    f = b.dynamics
    exact = hasattr(f, "vjp")
    if not exact and b.n_latents > max_dims:
        return {"fixed_points": None, "fixed_points_note":
                f"not attempted: {b.n_latents} latent dims exceeds {max_dims} for the finite-difference "
                "search, and the dynamics callable offers no vjp() for exact gradients"}
    Zall = b.flat("latents", "val").astype(float)
    U_dim = b.inputs.shape[2] if b.inputs is not None else 1
    z = Zall[rng.choice(len(Zall), min(n_starts, len(Zall)), replace=False)]
    z = z + 0.05 * Zall.std(0) * rng.standard_normal(z.shape)
    zero = np.zeros((len(z), U_dim))
    D = z.shape[1]
    scale = Zall.std() + 1e-12
    h = 1e-4 * scale

    def resid(zz, uu):
        return np.asarray(f(zz, uu), dtype=float) - zz

    def q(zz):
        d = resid(zz, zero[: len(zz)])
        return 0.5 * np.sum(d * d, axis=1)

    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for i in range(1, steps + 1):
        if exact:
            d = resid(z, zero)
            grad = np.asarray(f.vjp(z, zero, d), dtype=float) - d      # (J - I)^T d
        else:
            base = q(z)
            grad = np.empty_like(z)
            for k in range(D):
                e = np.zeros(D)
                e[k] = h
                grad[:, k] = (q(z + e) - base) / h
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
    u0 = np.zeros((1, U_dim))
    for p in uniq:
        if hasattr(f, "jacobian"):
            J = np.asarray(f.jacobian(p[None], u0), dtype=float).reshape(D, D)
        else:
            J = np.empty((D, D))
            f0 = np.asarray(f(p[None], u0))[0]
            for k in range(D):
                e = np.zeros(D)
                e[k] = h
                J[:, k] = (np.asarray(f((p + e)[None], u0))[0] - f0) / h
        if np.max(np.abs(np.linalg.eigvals(J))) < 1:
            n_stable += 1
    return {"fixed_points": len(uniq), "stable_fixed_points": n_stable,
            "fixed_point_search": "exact gradients" if exact else "finite differences",
            "fixed_point_min_q": float(np.min(qs))}


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
