"""The ayna bundle: everything a fitted dynamics model produced, as plain arrays.

A bundle is one ``.npz`` file. ayna never imports the framework that trained
the model; whoever trained it exports these arrays once, and every check runs
on numpy from there.

Shapes use R trials, T time bins, N neurons, D inferred latent dimensions and
U input dimensions. Rates are expected counts **per bin**, the same unit as
the spikes.

Required
    spikes        (R, T, N)  observed spike counts
    rates         (R, T, N)  model-predicted rates (held-in and held-out neurons)
    latents       (R, T, D)  model-inferred latent states

Optional
    heldout       (N,)  bool  neurons the model never saw as input (enables co-BPS)
    val           (R,)  bool  validation trials (default: the last 20%)
    trial_lens    (R,)  int   valid bins per trial, for padded data
    inputs        (R, T, U)   external inputs; ``meta["inputs_kind"]`` says
                              whether they were ``supplied`` to the model or
                              ``inferred`` by it
    true_latents  (R, T, Dt)  ground truth (synthetic data only) -- tier 2
    true_rates    (R, T, N)
    true_inputs   (R, T, Ut)  the inputs the teacher received (CtD's Input R² target)
    true_inputs_noiseless (R, T, Ut)  the same inputs before input noise, when the
                              simulator adds noise (used for ayna's misattribution truth)
    meta          JSON string: name, model, bin_ms, inputs_kind, labels, ...

A ``dynamics`` callable ``f(z, u) -> z_next`` (arrays ``(M, D)``, ``(M, U)``)
can be attached in Python; it cannot be stored in an ``.npz`` and unlocks the
tier-1 probes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ARRAY_KEYS = (
    "spikes",
    "rates",
    "latents",
    "heldout",
    "val",
    "trial_lens",
    "inputs",
    "true_latents",
    "true_rates",
    "true_inputs",
    "true_inputs_noiseless",
)

RATE_FLOOR = 1e-9


class BundleError(ValueError):
    """The bundle cannot be checked as given."""


@dataclass
class Bundle:
    spikes: np.ndarray
    rates: np.ndarray
    latents: np.ndarray
    heldout: Optional[np.ndarray] = None
    val: Optional[np.ndarray] = None
    trial_lens: Optional[np.ndarray] = None
    inputs: Optional[np.ndarray] = None
    true_latents: Optional[np.ndarray] = None
    true_rates: Optional[np.ndarray] = None
    true_inputs: Optional[np.ndarray] = None
    true_inputs_noiseless: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    dynamics: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    source: str = ""
    notes: list = field(default_factory=list)

    def __post_init__(self):
        for key in ARRAY_KEYS:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, np.asarray(value))
        self._validate()

    # ------------------------------------------------------------------ shape
    @property
    def n_trials(self) -> int:
        return self.spikes.shape[0]

    @property
    def n_bins(self) -> int:
        return self.spikes.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.spikes.shape[2]

    @property
    def n_latents(self) -> int:
        return self.latents.shape[2]

    @property
    def inputs_kind(self) -> Optional[str]:
        if self.inputs is None:
            return None
        return self.meta.get("inputs_kind", "inferred")

    @property
    def has_ground_truth(self) -> bool:
        return self.true_latents is not None or self.true_rates is not None

    @property
    def name(self) -> str:
        return self.meta.get("name") or Path(self.source).stem or "bundle"

    # ------------------------------------------------------------- validation
    def _validate(self):
        for key in ("spikes", "rates", "latents"):
            arr = getattr(self, key)
            if arr is None:
                raise BundleError(f"'{key}' is required")
            if arr.ndim != 3:
                raise BundleError(
                    f"'{key}' must be (trials, bins, features); got shape {arr.shape}"
                )
        R, T, N = self.spikes.shape
        if self.rates.shape != (R, T, N):
            raise BundleError(
                f"'rates' must match 'spikes' {self.spikes.shape}; got {self.rates.shape}"
            )
        for key in ("latents", "inputs", "true_latents", "true_inputs", "true_inputs_noiseless"):
            arr = getattr(self, key)
            if arr is None:
                continue
            if arr.ndim != 3 or arr.shape[:2] != (R, T):
                raise BundleError(
                    f"'{key}' must be ({R}, {T}, features); got {arr.shape}"
                )
        if self.true_rates is not None and self.true_rates.shape != (R, T, N):
            raise BundleError(f"'true_rates' must be {(R, T, N)}; got {self.true_rates.shape}")

        if np.any(self.spikes < 0):
            raise BundleError("'spikes' contains negative counts")
        if not np.all(np.isfinite(self.rates)):
            raise BundleError("'rates' contains NaN or inf")
        if np.any(self.rates <= 0):
            n_bad = int(np.sum(self.rates <= 0))
            self.rates = np.maximum(self.rates, RATE_FLOOR)
            self.notes.append(
                f"{n_bad} non-positive predicted rates were floored at {RATE_FLOOR:g}"
            )
        if not np.all(np.isfinite(self.latents)):
            raise BundleError("'latents' contains NaN or inf")

        if self.heldout is not None:
            self.heldout = self.heldout.astype(bool).reshape(-1)
            if self.heldout.shape != (N,):
                raise BundleError(f"'heldout' must have {N} entries")
            if self.heldout.all():
                raise BundleError("every neuron is marked held-out")
        if self.val is None:
            n_val = max(1, int(round(0.2 * R)))
            self.val = np.zeros(R, bool)
            self.val[R - n_val:] = True
            self.notes.append(f"no 'val' split given; used the last {n_val} of {R} trials")
        else:
            self.val = self.val.astype(bool).reshape(-1)
            if self.val.shape != (R,):
                raise BundleError(f"'val' must have {R} entries")
        if self.val.all() or not self.val.any():
            raise BundleError("'val' must mark some, but not all, trials")
        if self.trial_lens is not None:
            self.trial_lens = self.trial_lens.astype(int).reshape(-1)
            if self.trial_lens.shape != (R,) or self.trial_lens.max() > T or self.trial_lens.min() < 1:
                raise BundleError("'trial_lens' must be R integers in [1, T]")
        kind = self.meta.get("inputs_kind")
        if kind is not None and kind not in ("supplied", "inferred"):
            raise BundleError("meta['inputs_kind'] must be 'supplied' or 'inferred'")

    # ------------------------------------------------------------- selection
    def trials(self, split: str) -> np.ndarray:
        if split == "train":
            return ~self.val
        if split == "val":
            return self.val.copy()
        if split == "all":
            return np.ones(self.n_trials, bool)
        raise ValueError(split)

    def bin_mask(self, split: str = "all") -> np.ndarray:
        """(R, T) boolean mask of valid bins in the chosen trials."""
        mask = np.zeros((self.n_trials, self.n_bins), bool)
        mask[self.trials(split)] = True
        if self.trial_lens is not None:
            mask &= np.arange(self.n_bins)[None, :] < self.trial_lens[:, None]
        return mask

    def flat(self, key_or_array, split: str = "all", neurons=None) -> np.ndarray:
        """Flatten (R, T, F) to (samples, F) over valid bins of a split."""
        arr = getattr(self, key_or_array) if isinstance(key_or_array, str) else key_or_array
        if arr is None:
            return None
        arr = arr[self.bin_mask(split)]
        if neurons is not None:
            arr = arr[:, neurons]
        return arr

    @property
    def heldin(self) -> Optional[np.ndarray]:
        return None if self.heldout is None else ~self.heldout

    # ------------------------------------------------------------------- I/O
    def save(self, path) -> Path:
        path = Path(path)
        payload = {k: getattr(self, k) for k in ARRAY_KEYS if getattr(self, k) is not None}
        payload["meta"] = np.array(json.dumps(self.meta, sort_keys=True))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    def summary(self) -> str:
        lens = ""
        if self.trial_lens is not None:
            lens = f" (variable length, {int(self.trial_lens.min())}-{int(self.trial_lens.max())} bins)"
        parts = [
            f"{self.n_trials} trials x {self.n_bins} bins{lens}",
            f"{self.n_neurons} neurons"
            + (f" ({int(self.heldout.sum())} held-out)" if self.heldout is not None else ""),
            f"{self.n_latents} latents",
        ]
        if self.inputs is not None:
            parts.append(f"{self.inputs.shape[2]} {self.inputs_kind} inputs")
        if self.has_ground_truth:
            parts.append("ground truth")
        if self.dynamics is not None:
            parts.append("dynamics callable")
        return ", ".join(parts)


def load_bundle(path, dynamics=None) -> Bundle:
    """Read a bundle written by :meth:`Bundle.save` or by any exporter."""
    path = Path(path)
    if not path.exists():
        raise BundleError(f"no such file: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ARRAY_KEYS if k in data.files}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    missing = [k for k in ("spikes", "rates", "latents") if k not in arrays]
    if missing:
        raise BundleError(f"{path.name} lacks required array(s): {', '.join(missing)}")
    return Bundle(**arrays, meta=meta, dynamics=dynamics, source=str(path))
