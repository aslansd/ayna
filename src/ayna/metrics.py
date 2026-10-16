"""Numpy ports of the Computation-through-Dynamics Toolkit metrics.

Each function mirrors the definition in Versteeg et al. (2026), PLOS Comput
Biol 22(10): e1014755, and the reference implementation in
``ctd/comparison/metrics.py``. Arrays are ``(samples, features)``; use
:meth:`ayna.bundle.Bundle.flat` to get there from ``(trials, bins, features)``.

Regression direction matters and is spelled out in every name:

* State R²  -- true latents -> inferred latents (fails when features are *invented*)
* Recovery  -- inferred latents -> true latents (fails when features are *missed*)
* Input R²  -- inferred inputs -> true inputs
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

LOG2 = np.log(2.0)


def vw_r2(target, prediction) -> float:
    """Variance-weighted R² across feature dimensions."""
    return float(r2_score(target, prediction, multioutput="variance_weighted"))


def affine_r2(source_train, target_train, source_val, target_val, alpha: float = 0.0) -> float:
    """Fit ``source -> target`` on train, score variance-weighted R² on val."""
    model = Ridge(alpha=alpha) if alpha else LinearRegression()
    model.fit(source_train, target_train)
    return vw_r2(target_val, model.predict(source_val))


def affine_r2_per_dim(source_train, target_train, source_val, target_val) -> np.ndarray:
    model = LinearRegression().fit(source_train, target_train)
    return np.atleast_1d(r2_score(target_val, model.predict(source_val), multioutput="raw_values"))


# ----------------------------------------------------------------- tier 2
def rate_r2(true_rates, rates) -> float:
    """Rate R² (eq. 1): aligned, no fitted map, variance-weighted over neurons."""
    return vw_r2(true_rates, rates)


def state_r2(true_train, inf_train, true_val, inf_val) -> float:
    """State R² (eq. 2): how much of the inferred latents the truth explains."""
    return affine_r2(true_train, inf_train, true_val, inf_val)


def recovery_r2(true_train, inf_train, true_val, inf_val) -> float:
    """How much of the true latents the inferred latents explain (missed features)."""
    return affine_r2(inf_train, true_train, inf_val, true_val)


def input_r2(true_train, inf_train, true_val, inf_val) -> float:
    """Input R² (eq. 3): inferred inputs -> true inputs."""
    return affine_r2(inf_train, true_train, inf_val, true_val)


# ----------------------------------------------------------------- tier 0
def poisson_ll(rates, spikes) -> float:
    rates = np.maximum(rates, 1e-9)
    return float(np.sum(spikes * np.log(rates) - rates))


def bits_per_spike(rates, spikes) -> float:
    """Bits per spike against each neuron's mean rate (NLB convention).

    With held-out neurons on held-out trials this is co-BPS.
    """
    n_spikes = float(np.sum(spikes))
    if n_spikes == 0:
        return float("nan")
    null = np.broadcast_to(np.mean(spikes, axis=0, keepdims=True), spikes.shape)
    return (poisson_ll(rates, spikes) - poisson_ll(null, spikes)) / (n_spikes * LOG2)


def _truncated_pinv(W, variance_threshold):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    share = S**2 / np.sum(S**2)
    keep = min(int(np.searchsorted(np.cumsum(share), 1 - variance_threshold)) + 1, len(S))
    return Vt[:keep].T @ np.diag(1.0 / S[:keep]) @ U[:, :keep].T, keep


def cycle_consistency(lat_train, rates_train, lat_val, variance_threshold: float = 0.01):
    """Linear cycle-consistency, following ``get_cycle_consistency`` in CtD.

    Fits an affine readout from latent PCs to log-rate PCs, regenerates the
    latents through a singular-value-thresholded pseudo-inverse, and scores the
    variance-weighted R² between latents and their regenerated versions.

    Returns ``(r2, per_pc_r2, pc_variance_share, kept_components)`` so callers
    can say *which* latent directions are not reflected in the rates.
    """
    pca_lat = PCA().fit(lat_train)
    z_train = pca_lat.transform(lat_train)
    z_val = pca_lat.transform(lat_val)
    pca_rate = PCA().fit(np.log(rates_train))
    n_train = pca_rate.transform(np.log(rates_train))
    readout = LinearRegression().fit(z_train, n_train)
    preds = readout.predict(z_val)
    pinv, kept = _truncated_pinv(readout.coef_, variance_threshold)
    z_back = (preds - readout.intercept_) @ pinv.T
    r2 = vw_r2(z_val, z_back)
    per_pc = r2_score(z_val, z_back, multioutput="raw_values")
    return float(r2), np.asarray(per_pc), pca_lat.explained_variance_ratio_, kept


def readout_linearity(lat_train, rates_train, lat_val, rates_val) -> float:
    """R² of an affine map from latents to log-rates on validation data.

    The linear metrics assume the canonical linear-exponential link. When this
    is low, they can mislead.
    """
    return affine_r2(lat_train, np.log(rates_train), lat_val, np.log(rates_val))


def readout_null_directions(lat_train, rates_train, rel_gain: float = 0.01):
    """Latent PCs whose readout gain is below ``rel_gain`` of the largest.

    Returns ``(null_pcs, gains, pc_variance_share)``. A direction the readout
    ignores is where an invented feature can live without costing
    reconstruction.
    """
    pca_lat = PCA().fit(lat_train)
    z_train = pca_lat.transform(lat_train)
    readout = LinearRegression().fit(z_train, np.log(rates_train))
    share = pca_lat.explained_variance_ratio_
    # sensitivity of log-rates to a unit move along each latent PC (PCs share
    # the latent units, so sensitivities are comparable)
    gains = np.linalg.norm(readout.coef_, axis=0)
    live = share > 1e-6
    top = gains[live].max() if live.any() and gains[live].max() > 0 else 1.0
    gains = np.where(live, gains / top, np.nan)
    null = np.where(live & (gains < rel_gain))[0]
    return null, gains, share


def nonlinear_cycle_consistency(
    lat_train, rates_train, lat_val, hidden: int = 64, max_samples: int = 20000, seed: int = 0
) -> float:
    """Cycle-consistency with learned nonlinear maps (cf. CtD ``nl_cycle_con``).

    Latents -> log-rates and log-rates -> latents are both small MLPs
    (scikit-learn), so arbitrary invertible readouts are not penalised.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    if len(lat_train) > max_samples:
        idx = rng.choice(len(lat_train), max_samples, replace=False)
        lat_train, rates_train = lat_train[idx], rates_train[idx]
    sz = StandardScaler().fit(lat_train)
    sn = StandardScaler().fit(np.log(rates_train))
    z_tr, n_tr = sz.transform(lat_train), sn.transform(np.log(rates_train))
    kw = dict(hidden_layer_sizes=(hidden, hidden), max_iter=300, early_stopping=True,
              random_state=seed, tol=1e-4)
    fwd = MLPRegressor(**kw).fit(z_tr, n_tr)
    bwd = MLPRegressor(**kw).fit(fwd.predict(z_tr), z_tr)
    z_val = sz.transform(lat_val)
    z_back = bwd.predict(fwd.predict(z_val))
    if z_back.ndim == 1:
        z_back = z_back[:, None]
    return vw_r2(z_val, z_back)
