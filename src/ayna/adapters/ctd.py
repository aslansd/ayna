"""Export a Computation-through-Dynamics Toolkit run to an ayna bundle.

Run this inside the CtD environment (torch, Lightning, ...). ayna itself never
imports torch at module level; this module touches CtD objects only through
the getters the toolkit's own ``Comparison.compute_metrics`` uses, so it
duck-types and is unit-tested without CtD installed.

Conventions, taken from CtD's own comparison code and checked against the
pretrained walkthrough models (CtD 1.0.2, 3BFF NODE and LFADS sweeps):

* ``get_model_outputs(phase)`` returns ``(rates, latents)``, rates in spikes/bin;
* ``get_spiking(phase)`` returns all neurons, held-in first, held-out last;
* the held-in count is the width of the spiking the model receives,
  ``get_model_inputs(phase)[0]``;
* the task-trained analysis returns trials in the same order as the
  data-trained one for each phase;
* ``get_trial_lens`` returns zeros for fixed-length datasets (and raises for
  some LFADS datamodules), so non-positive or unavailable lengths are treated
  as "all trials full length";
* models are put in ``eval()`` mode before export: CtD leaves them in
  training mode, where dropout makes repeated exports differ.

Everything runs under ``torch.no_grad()``: CtD's getters run the full training
split through the model in one pass, which needs tens of GB with autograd on.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..bundle import Bundle

PHASES = ("train", "val")


def _no_grad():
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.no_grad()


def _np(x, dtype=None):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if dtype is not None and np.issubdtype(x.dtype, np.floating):
        x = x.astype(dtype, copy=False)
    return x


def _get(obj, method, phase, dtype=np.float32):
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        return _np(fn(phase=phase), dtype)
    except Exception:  # noqa: BLE001 - optional getters; some CtD versions raise for unused fields
        return None


# ------------------------------------------------------------------ dynamics
def dynamics_from_ctd(dd, model_type: Optional[str] = None, latent_dim: Optional[int] = None,
                      input_dim: Optional[int] = None) -> tuple[Optional[Callable], str]:
    """Wrap a CtD data-trained model's dynamics as ``f(z, u) -> z_next`` on numpy arrays.

    Returns ``(callable or None, explanation)``.

    * SAE / NODE: ``model.decoder.cell(u, z)`` (``MLPCell``: z + 0.1 * vf([z, u])).
    * LFADS: the generator cell ``gen_cell([u, ext], z)``; ``u`` is the
      controller output (inferred inputs). External inputs, if the model has
      any, are held at zero, so tier-1 rollouts treat them as absent.
    """
    try:
        import torch
    except ImportError:
        return None, "torch is not importable"
    model_type = model_type or getattr(dd, "model_type", "")
    try:
        cell = dd.get_dynamics_model()
    except Exception as exc:  # noqa: BLE001
        return None, f"get_dynamics_model() failed: {exc}"
    if cell is None:
        return None, "model exposes no dynamics cell"
    try:
        param = next(cell.parameters())
        device, dtype = param.device, param.dtype
    except StopIteration:
        device, dtype = "cpu", torch.float32
    ext_dim = 0
    if model_type == "LFADS":
        hp = getattr(getattr(dd, "model", None), "hparams", None)
        ext_dim = int(getattr(hp, "ext_input_dim", 0) or 0) if hp is not None else 0
        hidden = getattr(cell, "hidden_size", None)
        if hidden is not None and latent_dim is not None and hidden != latent_dim:
            return None, (f"LFADS latents have {latent_dim} dims but the generator cell has {hidden}; "
                          "the exported latents are not the generator state")

    def f(z, u):
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z), dtype=dtype, device=device)
            ut = torch.as_tensor(np.asarray(u), dtype=dtype, device=device)
            if ext_dim:
                ut = torch.cat([ut, torch.zeros(ut.shape[0], ext_dim, dtype=dtype, device=device)], dim=1)
            return cell(ut, zt).cpu().numpy()

    note = f"{type(cell).__name__} from {model_type or 'model'}"
    if ext_dim:
        note += f" ({ext_dim} external input dims held at zero)"
    return f, note


# -------------------------------------------------------------------- bundle
def from_ctd(dd, tt=None, name: Optional[str] = None, inputs_kind: Optional[str] = None,
             model_type: Optional[str] = None, with_dynamics: bool = True) -> Bundle:
    """Build a :class:`~ayna.bundle.Bundle` from CtD analysis objects.

    ``dd`` is an ``Analysis_DD`` (SAE or LFADS); ``tt`` an optional
    ``Analysis_TT`` supplying ground-truth latents and inputs. With
    ``with_dynamics`` the model's dynamics cell is attached for tier-1 probes.
    """
    model_type = model_type or getattr(dd, "model_type", "")
    parts = {k: [] for k in ("spikes", "rates", "latents", "inputs", "true_rates",
                             "true_latents", "true_inputs", "trial_lens")}
    n_heldin = None
    n_val = 0
    if inputs_kind is None:
        inputs_kind = "inferred" if model_type == "LFADS" and hasattr(dd, "get_inferred_inputs") else "supplied"

    # CtD's analysis objects leave models in training mode, so dropout would make every
    # export slightly different. Inference belongs in eval mode.
    for obj in (getattr(dd, "model", None), getattr(getattr(tt, "wrapper", None), "model", None)):
        if hasattr(obj, "eval"):
            obj.eval()
    with _no_grad():
        for phase in PHASES:
            rates, latents = dd.get_model_outputs(phase=phase)
            rates, latents = _np(rates, np.float32), _np(latents, np.float32)
            spikes = _get(dd, "get_spiking", phase, dtype=None)
            model_in = dd.get_model_inputs(phase=phase)
            heldin_spikes, supplied = _np(model_in[0]), _np(model_in[1], np.float32)
            n_heldin = heldin_spikes.shape[-1]
            inputs = _get(dd, "get_inferred_inputs", phase) if inputs_kind == "inferred" else supplied
            parts["spikes"].append(None if spikes is None else spikes.astype(np.int32))
            parts["rates"].append(rates)
            parts["latents"].append(latents)
            parts["inputs"].append(inputs)
            parts["true_rates"].append(_get(dd, "get_true_rates", phase))
            lens = _get(dd, "get_trial_lens", phase, dtype=None)
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
    R, T, N = spikes.shape
    heldout = np.arange(N) >= n_heldin if n_heldin is not None and n_heldin < N else None
    val = np.zeros(R, bool)
    val[R - n_val:] = True
    notes = []
    trial_lens = cat("trial_lens")
    if trial_lens is not None:
        if np.any(trial_lens <= 0) or np.any(trial_lens > T):
            if np.any(trial_lens > 0):
                notes.append("CtD trial lengths outside [1, T] were ignored; all trials treated as full length")
            trial_lens = None
        elif np.all(trial_lens >= T):
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
        meta["tt_latents_note"] = "teacher latents as returned by CtD get_latents (noisy teacher; seed fixed by ayna)"
    hp = getattr(getattr(dd, "model", None), "hparams", None)
    if model_type == "LFADS" and hp is not None and getattr(hp, "fac_dim", None):
        # generator state -> factors -> rates: cycle-consistency is comparative only
        meta["readout_bottleneck"] = int(hp.fac_dim)
        meta["latents_are"] = "LFADS generator state"
    dynamics = None
    if with_dynamics:
        dynamics, why = dynamics_from_ctd(dd, model_type, latent_dim=cat("latents").shape[-1],
                                          input_dim=None if inputs is None else inputs.shape[-1])
        meta["dynamics"] = why
    b = Bundle(spikes=spikes, rates=cat("rates"), latents=cat("latents"), heldout=heldout, val=val,
               trial_lens=trial_lens, inputs=inputs, true_rates=cat("true_rates"),
               true_latents=cat("true_latents"), true_inputs=cat("true_inputs"), meta=meta,
               dynamics=dynamics)
    b.notes.extend(notes)
    return b


def _require_ctd():
    try:
        from ctd.comparison.analysis.dd.dd import Analysis_DD
        from ctd.comparison.analysis.tt.tt import Analysis_TT
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise ImportError(
            "This needs the Computation-through-Dynamics Toolkit importable "
            "(`pip install -e .` in its repository) and its pickles' dependencies "
            "(torch, pytorch-lightning, ray, wandb). See INSTALL.md."
        ) from exc
    return Analysis_DD, Analysis_TT


def check_lfs(run_dir) -> None:
    """Raise a clear error if a run's pickles are Git LFS pointer stubs rather than real files."""
    stubs = []
    for p in sorted(Path(run_dir).glob("*.pkl")):
        with open(p, "rb") as fh:
            if fh.read(40).startswith(b"version https://git-lfs"):
                stubs.append(p.name)
    if stubs:
        raise FileNotFoundError(
            f"{run_dir}: {', '.join(stubs)} are Git LFS pointer files, not models. Install git-lfs and run "
            "`git lfs pull --include=\"<that folder>/**\"` in the CtD repository (see INSTALL.md).")


def load_run(dd_dir: str, model_type: str, tt_dir: Optional[str] = None, inputs_kind: Optional[str] = None,
             with_dynamics: bool = True, name: Optional[str] = None, seed: Optional[int] = 0) -> Bundle:
    """Load a saved CtD data-trained run (and optionally its task-trained teacher) as a bundle.

    ``seed`` seeds torch and numpy before export: CtD's noisy task-trained
    teachers draw fresh noise each time their latents are requested, so
    unseeded exports give slightly different ground truth on every run.
    """
    if seed is not None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass
    check_lfs(dd_dir)
    if tt_dir:
        check_lfs(tt_dir)
    Analysis_DD, Analysis_TT = _require_ctd()
    dd_dir = str(dd_dir).rstrip("/") + "/"
    run_name = name or Path(dd_dir).name
    dd = Analysis_DD.create(run_name=run_name, filepath=dd_dir, model_type=model_type)
    tt = None
    if tt_dir:
        tt_dir = str(tt_dir).rstrip("/") + "/"
        tt = Analysis_TT(run_name=Path(tt_dir).name, filepath=tt_dir)
    b = from_ctd(dd, tt, name=run_name, inputs_kind=inputs_kind, model_type=model_type,
                 with_dynamics=with_dynamics)
    return b


def export_run(dd_dir: str, out_path: str, model_type: str, tt_dir: Optional[str] = None,
               inputs_kind: Optional[str] = None) -> str:
    """Load a saved CtD run from disk and write a bundle (the dynamics cell is not stored)."""
    b = load_run(dd_dir, model_type, tt_dir, inputs_kind, with_dynamics=False)
    return str(b.save(out_path))


def find_runs(sweep_dir: str) -> list:
    """Run directories (containing model.pkl and datamodule.pkl) under ``sweep_dir``."""
    root = Path(sweep_dir)
    if (root / "model.pkl").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/model.pkl") if (p.parent / "datamodule.pkl").exists())


# ------------------------------------------------------------------- worker
def check_run(run_dir, model_type: str, tt_dir=None, out_dir="ctd_reports", save_bundle: bool = False,
              with_dynamics: bool = True, overrides: Optional[dict] = None) -> dict:
    """Load one CtD run, scan it, and write ``report.*`` plus ``select_row.json`` under ``out_dir/<run>``.

    ``ayna ctd`` calls this in a fresh subprocess per run: CtD's datamodules
    hold the whole dataset in memory and are not reliably freed, so several
    runs in one process can exhaust RAM on a laptop.
    """
    import json

    from ..scan import Thresholds, scan
    from ..select import select_row

    b = load_run(run_dir, model_type, tt_dir, with_dynamics=with_dynamics)
    th = Thresholds().update(overrides or {})
    report = scan(b, th)
    run_out = Path(out_dir) / b.name
    report.save(run_out)
    row = select_row(b)
    (run_out / "select_row.json").write_text(json.dumps(row, default=float))
    if save_bundle:
        b.save(run_out / "bundle.npz")
    return {"name": b.name, "out": str(run_out), "flags": sorted(report.flagged()),
            "truth": sorted(report.ground_truth_flags())}


def _main(argv=None) -> int:
    import argparse
    import json
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="python -m ayna.adapters.ctd", description="scan one CtD run (worker)")
    p.add_argument("run_dir")
    p.add_argument("model_type", choices=["SAE", "LFADS"])
    p.add_argument("--tt")
    p.add_argument("--out", default="ctd_reports")
    p.add_argument("--save-bundle", action="store_true")
    p.add_argument("--no-dynamics", action="store_true")
    p.add_argument("--overrides", default="{}", help="JSON dict of threshold overrides")
    a = p.parse_args(argv)
    res = check_run(a.run_dir, a.model_type, a.tt, a.out, a.save_bundle, not a.no_dynamics, json.loads(a.overrides))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
