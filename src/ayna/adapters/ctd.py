"""Export a Computation-through-Dynamics Toolkit run to an ayna bundle.

Run this inside the CtD environment (torch, Lightning, ...). ayna itself never
imports torch; this module only touches CtD objects through the getters the
toolkit's own ``Comparison.compute_metrics`` uses, so it duck-types and can
be tested without CtD installed.

Assumptions, all taken from CtD's own comparison code:

* ``get_model_outputs(phase)`` returns ``(rates, latents)``, rates in spikes/bin;
* ``get_spiking(phase)`` returns all neurons, held-in first, held-out last;
* the held-in count is the width of the spiking the model receives,
  ``get_model_inputs(phase)[0]``;
* the task-trained analysis returns trials in the same order as the
  data-trained one for each phase.

Status: written against CtD 1.0.2 source; not yet run on a live CtD install
(the pretrained models are Git-LFS objects). Please report mismatches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..bundle import Bundle

PHASES = ("train", "val")


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _get(obj, method, phase):
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        return _np(fn(phase=phase))
    except (NotImplementedError, AttributeError, TypeError):
        return None


def from_ctd(dd, tt=None, name: Optional[str] = None, inputs_kind: Optional[str] = None,
             model_type: Optional[str] = None) -> Bundle:
    """Build a :class:`~ayna.bundle.Bundle` from CtD analysis objects.

    ``dd`` is an ``Analysis_DD`` (SAE or LFADS); ``tt`` an optional
    ``Analysis_TT`` supplying ground-truth latents and inputs.
    """
    model_type = model_type or getattr(dd, "model_type", "")
    parts = {k: [] for k in ("spikes", "rates", "latents", "inputs", "true_rates",
                             "true_latents", "true_inputs", "trial_lens")}
    n_heldin = None
    n_val = 0
    if inputs_kind is None:
        inputs_kind = "inferred" if model_type == "LFADS" and hasattr(dd, "get_inferred_inputs") else "supplied"

    for phase in PHASES:
        rates, latents = dd.get_model_outputs(phase=phase)
        rates, latents = _np(rates), _np(latents)
        spikes = _get(dd, "get_spiking", phase)
        model_in = dd.get_model_inputs(phase=phase)
        heldin_spikes, supplied = _np(model_in[0]), _np(model_in[1])
        n_heldin = heldin_spikes.shape[-1]
        if inputs_kind == "inferred":
            inputs = _get(dd, "get_inferred_inputs", phase)
        else:
            inputs = supplied
        parts["spikes"].append(spikes)
        parts["rates"].append(rates)
        parts["latents"].append(latents)
        parts["inputs"].append(inputs)
        parts["true_rates"].append(_get(dd, "get_true_rates", phase))
        lens = _get(dd, "get_trial_lens", phase)
        parts["trial_lens"].append(None if lens is None else lens.reshape(-1).astype(int))
        if tt is not None:
            parts["true_latents"].append(_get(tt, "get_latents", phase))
            parts["true_inputs"].append(_get(tt, "get_inputs", phase))
        if phase == "val":
            n_val = rates.shape[0]

    def cat(key):
        items = parts[key]
        if not items or any(i is None for i in items):
            return None
        return np.concatenate(items, axis=0)

    spikes = cat("spikes")
    N = spikes.shape[-1]
    heldout = np.arange(N) >= n_heldin if n_heldin is not None and n_heldin < N else None
    R = spikes.shape[0]
    val = np.zeros(R, bool)
    val[R - n_val:] = True
    trial_lens = cat("trial_lens")
    if trial_lens is not None and np.all(trial_lens >= spikes.shape[1]):
        trial_lens = None
    inputs = cat("inputs")
    meta = {
        "name": name or getattr(dd, "run_name", "ctd_run"),
        "model": model_type or "unknown",
        "source": "ctd",
        "exporter": "ayna.adapters.ctd",
    }
    if inputs is not None:
        meta["inputs_kind"] = inputs_kind
    if tt is not None:
        meta["tt_run"] = getattr(tt, "run_name", "")
    return Bundle(spikes=spikes, rates=cat("rates"), latents=cat("latents"), heldout=heldout, val=val,
                  trial_lens=trial_lens, inputs=inputs, true_rates=cat("true_rates"),
                  true_latents=cat("true_latents"), true_inputs=cat("true_inputs"), meta=meta)


def export_run(dd_dir: str, out_path: str, model_type: str, tt_dir: Optional[str] = None,
               inputs_kind: Optional[str] = None) -> str:
    """Load a saved CtD run from disk (needs CtD importable) and write a bundle."""
    try:
        from ctd.comparison.analysis.dd.dd import Analysis_DD
        from ctd.comparison.analysis.tt.tt import Analysis_TT
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise ImportError(
            "export-ctd must run in an environment where the Computation-through-Dynamics "
            "Toolkit is installed (`pip install -e .` in its repository)."
        ) from exc
    dd_dir = dd_dir.rstrip("/") + "/"
    dd = Analysis_DD.create(run_name=dd_dir.rstrip("/").split("/")[-1], filepath=dd_dir, model_type=model_type)
    tt = None
    if tt_dir:
        tt_dir = tt_dir.rstrip("/") + "/"
        tt = Analysis_TT(run_name=tt_dir.rstrip("/").split("/")[-1], filepath=tt_dir)
    bundle = from_ctd(dd, tt, inputs_kind=inputs_kind, model_type=model_type)
    return str(bundle.save(out_path))
