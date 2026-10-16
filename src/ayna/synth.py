"""Synthetic ground truth and simulated model outputs with injected faults.

Nothing here trains a model. A known system (a 3-bit flip-flop, or an
input-driven system with no memory) produces spikes; then ayna writes the
outputs a fitted data-driven model *would* produce if it had a specific
failure, at a chosen severity. Because every fault is injected on purpose,
``ayna validate`` can measure what the detectors catch and what they miss.

The flip-flop follows the 3BFF task in CtD: each bit is a bistable unit,
``z <- (1 - dt) z + dt tanh(w z + g u)``, flipped by brief signed pulses.
Rates are linear-exponential in the latents and spikes are Poisson, matching
the canonical CtD observation model unless ``nonlinear_readout`` is asked for.

Faults (severity in [0, 1]):

``underfit``            the model misses that fraction of one bit's modulation
``feature_invention``   invented, readout-silent latent activity; severity is
                        its share of latent variance
``misattribution``      inferred inputs become state-like and the dynamics leak;
                        severity mixes clean and offloaded solutions
``input_driven``        a different ground truth: latents are a short filter of
                        rich supplied inputs (CtD's RandomTarget warning)
``nonlinear_readout``   a different ground truth: log-rates are a nonlinear
                        function of the latents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .bundle import Bundle

FAULTS = ("underfit", "feature_invention", "misattribution", "input_driven", "nonlinear_readout")
DEFAULT_SEVERITY = {
    "underfit": 1.0,
    "feature_invention": 0.3,
    "misattribution": 0.8,
    "input_driven": 1.0,
    "nonlinear_readout": 1.0,
}

FF_DT, FF_W, FF_G = 0.4, 2.0, 3.0


def flipflop_step(z, u):
    return (1 - FF_DT) * z + FF_DT * np.tanh(FF_W * z + FF_G * u)


@dataclass
class Truth:
    kind: str
    z: np.ndarray          # (R, T, B) true latents
    u: np.ndarray          # (R, T, U) true inputs
    rates: np.ndarray      # (R, T, N)
    spikes: np.ndarray     # (R, T, N)
    log_rate: Callable     # latents (..., B) -> log rates (..., N)
    step: Optional[Callable]
    heldout: np.ndarray
    val: np.ndarray


def _smooth_noise(rng, shape, sigma):
    x = gaussian_filter1d(rng.standard_normal(shape), sigma, axis=1)
    return x / (x.std() + 1e-12)


def make_truth(kind: str = "flipflop", n_trials: int = 200, n_bins: int = 100,
               n_neurons: int = 60, n_heldout: int = 15, nonlinear: bool = False,
               seed: int = 0) -> Truth:
    rng = np.random.default_rng(seed)
    R, T, N = n_trials, n_bins, n_neurons
    if kind == "flipflop":
        B = 3
        u = np.zeros((R, T, B))
        events = rng.random((R, T, B)) < 0.035
        signs = rng.choice([-1.0, 1.0], size=(R, T, B))
        for lag in range(3):
            u[:, lag:] += (events * signs)[:, : T - lag]
        u = np.clip(u, -1, 1) + 0.05 * rng.standard_normal((R, T, B))
        z = np.zeros((R, T, B))
        state = rng.choice([-0.96, 0.96], size=(R, B))
        for t in range(T):
            state = flipflop_step(state, u[:, t]) + 0.02 * rng.standard_normal((R, B))
            z[:, t] = state
        step = flipflop_step
    elif kind == "input_driven":
        B = 3
        u = _smooth_noise(rng, (R, T, 3), 4.0)
        A = rng.standard_normal((B, 3)) / np.sqrt(3)
        drive = np.tanh(u @ A.T)
        z = np.zeros((R, T, B))
        state = np.zeros((R, B))
        for t in range(T):
            state = 0.7 * state + 0.3 * drive[:, t] + 0.02 * rng.standard_normal((R, B))
            z[:, t] = state
        z = z / z.std()
        step = None
    else:
        raise ValueError(f"unknown ground truth kind: {kind}")

    C = rng.standard_normal((N, B)) * 0.8
    base = np.log(rng.uniform(0.05, 0.3, size=N))
    if nonlinear:
        V = rng.standard_normal((N, B))

        def log_rate(zz):
            return base + 0.9 * np.tanh(2.5 * zz @ C.T) + 0.6 * (zz @ V.T) ** 2 / B - 0.3
    else:
        def log_rate(zz):
            return base + zz @ C.T

    rates = np.exp(log_rate(z))
    spikes = rng.poisson(rates)
    heldout = np.zeros(N, bool)
    heldout[rng.choice(N, n_heldout, replace=False)] = True
    val = np.zeros(R, bool)
    val[rng.choice(R, max(1, R // 5), replace=False)] = True
    return Truth(kind, z, u, rates, spikes, log_rate, step, heldout, val)


def _orthonormal(rng, n, k):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q[:, :k], q[:, k:]


def make_model_outputs(truth: Truth, latent_dim: int = 8, faults: Optional[dict] = None,
                       inputs_kind: Optional[str] = "inferred", rate_noise: float = 0.12,
                       latent_noise: float = 0.05, seed: int = 0, name: str = "") -> Bundle:
    """Simulate a fitted model's outputs on ``truth`` with the given faults."""
    faults = dict(faults or {})
    rng = np.random.default_rng(seed + 1000)
    R, T, B = truth.z.shape
    D = max(latent_dim, 1)
    z = truth.z.copy()

    # --- underfit: the model loses part (or all) of one true latent ---------
    s_under = float(faults.get("underfit", 0.0))
    lost_bit = int(rng.integers(B))
    if s_under > 0:
        z[..., lost_bit] = (1 - s_under) * z[..., lost_bit] + s_under * z[..., lost_bit].mean()
    # a model with fewer latents than the truth can only keep D of them
    kept = np.sort(rng.choice(B, min(D, B), replace=False))
    K = len(kept)
    z_mean = truth.z.mean(axis=(0, 1))

    # --- embed the kept latents into D dims ---------------------------------
    n_inv = 2 if D >= K + 2 else 0
    Mq, perp = _orthonormal(rng, D, K)
    M = Mq * rng.uniform(0.8, 1.5, size=K)                     # (D, K)
    lat = z[..., kept] @ M.T
    inv_dirs = perp[:, :n_inv]
    other = perp[:, n_inv:]
    lat = lat + latent_noise * lat.std() * _smooth_noise(rng, (R, T, D), 2.0)

    # --- readout: true readout through M, plus generic gain on the rest ------
    M_pinv = np.linalg.pinv(M)
    W_perp = rng.standard_normal((truth.rates.shape[2], D)) * 0.4

    def log_rate_hat(latents):
        full = np.broadcast_to(z_mean, latents.shape[:-1] + (B,)).copy()
        full[..., kept] = latents @ M_pinv.T                   # back to true coords
        rest = (latents @ other) @ other.T                     # generic directions
        return truth.log_rate(full) + rest @ W_perp.T

    # --- feature invention: readout-silent latent activity ------------------
    s_inv = float(faults.get("feature_invention", 0.0))
    if s_inv > 0 and n_inv:
        walk = np.zeros((R, T, n_inv))
        x = rng.standard_normal((R, n_inv))
        for t in range(T):
            x = 0.97 * x + 0.25 * rng.standard_normal((R, n_inv))
            walk[:, t] = x
        walk = walk / walk.std()
        share = min(s_inv, 0.95)
        target_var = share / (1 - share) * lat.var(axis=(0, 1)).sum()
        walk = walk * np.sqrt(target_var / n_inv)
        lat = lat + walk @ inv_dirs.T
    log_r = log_rate_hat(lat)
    log_r = log_r + rate_noise * _smooth_noise(rng, log_r.shape, 3.0)
    rates = np.exp(log_r)

    # --- inputs and dynamics ------------------------------------------------
    inputs, dynamics, meta_inputs = None, None, {}
    s_mis = float(faults.get("misattribution", 0.0))
    if inputs_kind == "supplied":
        inputs = truth.u.copy()
        meta_inputs = {"inputs_kind": "supplied"}
    elif inputs_kind == "inferred":
        Uh = truth.u.shape[2]
        A = np.linalg.qr(rng.standard_normal((Uh, Uh)))[0]
        clean_u = truth.u @ A.T + 0.1 * rng.standard_normal(truth.u.shape)
        # offloaded solution: a leaky generator z' = 0.5 z + G u, with inputs
        # chosen to carry whatever the leak loses -- so they track the state
        G = rng.standard_normal((D, Uh))
        G_pinv = np.linalg.pinv(G)
        nxt_lat = np.concatenate([lat[:, 1:], lat[:, -1:]], axis=1)
        state_u = (nxt_lat - 0.5 * lat) @ G_pinv.T
        u_scale = clean_u.std() / (state_u.std() + 1e-12)
        state_u = state_u * u_scale
        inputs = (1 - s_mis) * clean_u + s_mis * state_u
        meta_inputs = {"inputs_kind": "inferred"}
        if truth.step is not None and K == B:
            A_inv = A.T
            s_lost = s_under

            def dynamics(zh, uh, _m=s_mis):
                core = truth.step(zh @ M_pinv.T, uh @ A_inv.T)
                if s_lost > 0:  # the model's dynamics lack what its latents lack
                    core[..., lost_bit] = (1 - s_lost) * core[..., lost_bit] + s_lost * z_mean[lost_bit]
                nxt = core @ M.T
                nxt = nxt + 0.8 * (zh @ other) @ other.T + 0.97 * (zh @ inv_dirs) @ inv_dirs.T
                return (1 - _m) * nxt + _m * (0.5 * zh + (uh / u_scale) @ G.T)

    meta = {
        "name": name or "synthetic",
        "model": "simulated",
        "bin_ms": 10,
        "synth": {"truth": truth.kind, "faults": faults, "latent_dim": D, "seed": seed},
        **meta_inputs,
    }
    return Bundle(
        spikes=truth.spikes, rates=rates, latents=lat, heldout=truth.heldout, val=truth.val,
        inputs=inputs, true_latents=truth.z, true_rates=truth.rates, true_inputs=truth.u,
        meta=meta, dynamics=dynamics,
    )


def parse_faults(items) -> dict:
    """``["underfit=0.5", "feature_invention"]`` -> ``{"underfit": 0.5, "feature_invention": 0.3}``."""
    out = {}
    for item in items or []:
        name, _, value = str(item).partition("=")
        name = name.strip()
        if name not in FAULTS:
            raise ValueError(f"unknown fault '{name}'; choose from {', '.join(FAULTS)}")
        out[name] = float(value) if value else DEFAULT_SEVERITY[name]
    return out


def synth(name: str = "synthetic", faults=None, latent_dim: int = 8, inputs_kind: Optional[str] = None,
          n_trials: int = 200, n_bins: int = 100, n_neurons: int = 60, n_heldout: int = 15,
          rate_noise: float = 0.12, seed: int = 0) -> Bundle:
    """One synthetic bundle with the requested faults injected."""
    faults = parse_faults(faults) if isinstance(faults, (list, tuple)) else dict(faults or {})
    kind = "input_driven" if faults.get("input_driven") else "flipflop"
    if inputs_kind is None:
        inputs_kind = "supplied" if kind == "input_driven" else "inferred"
    truth = make_truth(kind, n_trials, n_bins, n_neurons, n_heldout,
                       nonlinear=bool(faults.get("nonlinear_readout")), seed=seed)
    return make_model_outputs(truth, latent_dim, faults, inputs_kind, rate_noise, seed=seed, name=name)


def synth_sweep(sizes=(2, 3, 5, 8, 16, 32, 64), seeds=(0, 1, 2), **kw) -> list:
    """A latent-size sweep shaped like CtD Fig 5.

    Too few latents underfit; too many invent features. Larger models also get
    slightly less rate noise, so the invented-feature models *win* on co-BPS --
    the trap a single headline metric walks into.
    """
    truth = make_truth("flipflop", seed=kw.pop("truth_seed", 0), **kw)
    out = []
    for size in sizes:
        invention = {16: 0.1, 32: 0.25, 64: 0.4}.get(size, 0.0 if size <= 8 else 0.4)
        noise = {2: 0.15, 3: 0.14, 5: 0.13, 8: 0.12, 16: 0.11, 32: 0.10, 64: 0.09}.get(size, 0.1)
        for s in seeds:
            faults = {"feature_invention": invention} if invention else {}
            b = make_model_outputs(truth, size, faults, inputs_kind=None, rate_noise=noise,
                                   seed=1000 * size + s, name=f"node{size}_s{s}")
            b.meta["latent_dim"] = size
            out.append(b)
    return out
