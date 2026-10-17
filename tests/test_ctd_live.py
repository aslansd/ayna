"""Live tests against the Computation-through-Dynamics Toolkit's pretrained 3BFF models.

Skipped unless both hold:
  * CtD is importable (run inside the CtD environment, see INSTALL.md);
  * AYNA_CTD_WALKTHROUGH points to CtD's ``examples/walkthrough_models`` with the
    Git LFS files pulled (tt_3bff, node_sweep, inputinf_sweep).

Each model is loaded in its own subprocess through ``ayna ctd``, as in normal use.
Expect several minutes and ~6 GB of RAM per model at peak.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ctd")
ROOT = os.environ.get("AYNA_CTD_WALKTHROUGH")
pytestmark = pytest.mark.skipif(not ROOT, reason="set AYNA_CTD_WALKTHROUGH to CtD's examples/walkthrough_models")


def _run_ctd(tmp_path, model_type, *runs):
    root = Path(ROOT)
    cmd = [sys.executable, "-m", "ayna.cli", "ctd", *[str(root / r) for r in runs], "--model-type", model_type,
           "--tt", str(root / "tt_3bff"), "--out", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    reports = {p.parent.name: json.loads(p.read_text()) for p in tmp_path.glob("*/report.json")}
    return reports, proc.stdout


def _flags(report, tier_max=1):
    return {c["id"] for c in report["checks"] if c["status"] == "flagged" and c["tier"] <= tier_max}


def test_node_sweep_ends(tmp_path):
    reports, out = _run_ctd(tmp_path, "SAE",
                            "node_sweep/prefix=tt_3bff_latent_size=3_max_epochs=1000_seed=0",
                            "node_sweep/prefix=tt_3bff_latent_size=8_max_epochs=1000_seed=0",
                            "node_sweep/prefix=tt_3bff_latent_size=64_max_epochs=1000_seed=0")
    by_size = {int(k.split("latent_size=")[1].split("_")[0]): v for k, v in reports.items()}
    assert "UNDERFITTING" in _flags(by_size[3])
    assert _flags(by_size[8]) - {"READOUT_NULLSPACE"} == set()
    assert "FEATURE_INVENTION" in _flags(by_size[64])
    assert by_size[64]["metrics"]["state_r2"] < 0.7          # ground truth agrees
    assert "latent_size=8" in out.split("<- recommended")[0].splitlines()[-1]
    assert "RECONSTRUCTION_TRAP" in out


def test_lfads_select_skips_misattributed(tmp_path):
    reports, out = _run_ctd(tmp_path, "LFADS",
                            "inputinf_sweep/kl_co_scale=0_prefix=20241004_NBFF_NoisyGRU_Final_df=5_seed=0",
                            "inputinf_sweep/kl_co_scale=1.00E-04_prefix=20241004_NBFF_NoisyGRU_Final_df=5_seed=0")
    kl0 = next(v for k, v in reports.items() if k.startswith("kl_co_scale=0_"))
    kl4 = next(v for k, v in reports.items() if "1.00E-04" in k)
    assert "DYNAMICAL_MISATTRIBUTION" in _flags(kl0)
    assert "DYNAMICAL_MISATTRIBUTION" not in _flags(kl4)
    assert kl4["metrics"]["input_r2"] > kl0["metrics"]["input_r2"]   # ground truth agrees
    assert "1.00E-04" in out.split("<- recommended")[0].splitlines()[-1]


def test_co_bps_matches_ctd_on_true_heldout():
    import torch
    from ctd.comparison.analysis.dd.dd import Analysis_DD
    from ctd.comparison.metrics import get_bps

    from ayna.metrics import bits_per_spike

    torch.set_grad_enabled(False)
    run = str(Path(ROOT) / "node_sweep/prefix=tt_3bff_latent_size=8_max_epochs=1000_seed=0") + "/"
    dd = Analysis_DD.create(run_name="node8", filepath=run, model_type="SAE")
    n_heldin = dd.get_model_inputs(phase="val")[0].shape[-1]
    rates, _ = dd.get_model_outputs(phase="val")
    spikes = dd.get_spiking(phase="val")
    ctd_value = get_bps(rates[..., n_heldin:].numpy(), spikes[..., n_heldin:])
    r = rates[..., n_heldin:].numpy().reshape(-1, spikes.shape[-1] - n_heldin)
    y = spikes[..., n_heldin:].numpy().reshape(r.shape)
    assert abs(bits_per_spike(r, y) - ctd_value) < 1e-4
