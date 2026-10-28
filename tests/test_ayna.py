import json

import numpy as np
import pytest

from ayna import Bundle, BundleError, Thresholds, load_bundle, scan, select
from ayna import metrics as M
from ayna.cli import main
from ayna.synth import parse_faults, synth, synth_sweep
from ayna.validate import validate_synthetic

SMALL = dict(n_trials=120, n_neurons=40, n_heldout=10)


def flags(report, tier_max=1):
    return report.flagged(tier_max)


# ------------------------------------------------------------------ bundle
def test_bundle_roundtrip(tmp_path):
    b = synth("rt", [], seed=1, **SMALL)
    path = b.save(tmp_path / "rt.npz")
    b2 = load_bundle(path)
    assert b2.meta["name"] == "rt"
    np.testing.assert_allclose(b2.rates, b.rates)
    assert b2.heldout.sum() == 10
    assert b2.inputs_kind == "inferred"
    assert b2.dynamics is None  # callables do not survive .npz


def test_bundle_validation():
    R, T, N = 10, 20, 5
    ok = dict(spikes=np.ones((R, T, N)), rates=np.ones((R, T, N)), latents=np.zeros((R, T, 2)))
    with pytest.raises(BundleError):
        Bundle(**{**ok, "rates": np.ones((R, T, N + 1))})
    with pytest.raises(BundleError):
        Bundle(**{**ok, "spikes": -np.ones((R, T, N))})
    with pytest.raises(BundleError):
        Bundle(**ok, heldout=np.ones(N, bool))
    b = Bundle(**{**ok, "rates": np.zeros((R, T, N))})
    assert b.rates.min() > 0 and any("floored" in n for n in b.notes)
    assert b.val.sum() == 2 and any("last 2" in n for n in b.notes)


def test_trial_lens_mask():
    R, T, N = 4, 10, 3
    b = Bundle(spikes=np.ones((R, T, N)), rates=np.ones((R, T, N)), latents=np.zeros((R, T, 2)),
               trial_lens=np.array([10, 5, 7, 3]), val=np.array([0, 0, 1, 1], bool))
    assert b.flat("spikes", "train").shape[0] == 15
    assert b.flat("spikes", "val").shape[0] == 10


# ----------------------------------------------------------------- metrics
def test_bits_per_spike_signs():
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.05, 1.0, size=(5000, 4))
    spikes = rng.poisson(rates)
    assert M.bits_per_spike(rates, spikes) > 0.05
    null = np.broadcast_to(spikes.mean(0), spikes.shape)
    assert abs(M.bits_per_spike(null, spikes)) < 1e-9


def test_cycle_consistency_detects_silent_dimension():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4000, 3))
    C = rng.standard_normal((20, 3))
    rates = np.exp(0.3 * z @ C.T - 1)
    cc, *_ = M.cycle_consistency(z[:3000], rates[:3000], z[3000:])
    assert cc > 0.99
    z_inv = np.c_[z, 2 * rng.standard_normal((4000, 1))]  # silent in the rates
    cc2, per_pc, share, _ = M.cycle_consistency(z_inv[:3000], rates[:3000], z_inv[3000:])
    assert cc2 < 0.8
    assert per_pc[0] < 0.5 and share[0] > 0.3  # the silent dim is the top PC


def test_state_r2_direction():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2000, 2))
    inferred = np.c_[z, rng.standard_normal((2000, 1))]  # one invented dim
    tr, va = slice(0, 1500), slice(1500, None)
    assert M.state_r2(z[tr], inferred[tr], z[va], inferred[va]) < 0.8      # invented feature
    assert M.recovery_r2(z[tr], inferred[tr], z[va], inferred[va]) > 0.99  # nothing missed


# --------------------------------------------------------------- detectors
def test_clean_model_is_clean():
    r = scan(synth("clean", [], seed=3, **SMALL))
    assert flags(r) == set()
    assert r.ground_truth_flags() == set()
    assert not r.has_errors


@pytest.mark.parametrize("fault,detector", [
    ("underfit=1.0", "UNDERFITTING"),
    ("feature_invention=0.4", "FEATURE_INVENTION"),
    ("misattribution=0.9", "DYNAMICAL_MISATTRIBUTION"),
    ("input_driven", "INPUT_INHERITED_STRUCTURE"),
    ("nonlinear_readout", "NONLINEAR_READOUT"),
])
def test_injected_fault_is_found(fault, detector):
    b = synth("f", [fault], seed=4, **SMALL)
    r = scan(b)
    assert detector in flags(r), r.render_text()
    others = flags(r) - {detector, "READOUT_NULLSPACE"}
    assert not others, f"unexpected extra flags {others}"


def test_where_names_the_invented_pcs():
    r = scan(synth("inv", ["feature_invention=0.4"], seed=5, **SMALL))
    f = next(f for f in r.findings if f.id == "FEATURE_INVENTION" and f.tier == 0)
    assert "latent PCs" in f.where


def test_tier1_rollout_runs_with_dynamics():
    b = synth("mis", ["misattribution=0.9"], seed=6, **SMALL)
    r = scan(b)
    assert any(c.id == "DYNAMICAL_MISATTRIBUTION" and c.tier == 1 and c.status == "flagged" for c in r.checks)
    assert "stable_fixed_points" in next(f for f in r.findings
                                        if f.id == "DYNAMICAL_MISATTRIBUTION" and f.tier < 2).evidence


def test_not_assessed_is_explicit():
    b = synth("x", [], seed=7, **SMALL)
    b.heldout, b.inputs, b.true_latents, b.true_rates, b.true_inputs = None, None, None, None, None
    r = scan(b)  # the attached dynamics expect inputs; its failure must be reported, not raised
    assert any(c.id == "DYNAMICAL_MISATTRIBUTION" and c.tier == 1 and "raised" in c.reason for c in r.checks)
    na = {(c.id, c.tier) for c in r.checks if c.status == "not_assessed"}
    assert ("UNDERFITTING/co-BPS", 0) in na
    assert ("DYNAMICAL_MISATTRIBUTION", 0) in na
    assert ("FEATURE_INVENTION", 2) in na
    assert "not assessed" in r.render_text()


def test_thresholds_override():
    th = Thresholds().update({"leakage_r2": "0.99", "input_lags": "4"})
    assert th.leakage_r2 == 0.99 and th.input_lags == 4
    with pytest.raises(KeyError):
        Thresholds().update({"nope": 1})


# ------------------------------------------------------------------ select
def test_select_finds_reconstruction_trap():
    rep = select(synth_sweep(sizes=(2, 5, 8, 32, 64), seeds=(0,), n_trials=120, n_neurons=40, n_heldout=10))
    assert rep.knee in ("node5_s0", "node8_s0")
    assert rep.best_reconstruction in ("node32_s0", "node64_s0")
    assert [f.id for f in rep.findings] == ["RECONSTRUCTION_TRAP"]


# ---------------------------------------------------------------- validate
def test_validate_small_corpus():
    val = validate_synthetic(n_per_fault=1, seed=1)
    table = val.table()
    assert ("FEATURE_INVENTION", 0) in table
    assert "precision" in val.render_text()


# --------------------------------------------------------------------- cli
def test_cli_synth_scan(tmp_path, capsys):
    assert main(["synth", "s", "--faults", "feature_invention=0.6", "--trials", "120", "--neurons", "40",
                 "--out", str(tmp_path)]) == 0
    code = main(["scan", str(tmp_path / "s.npz"), "--json"])
    json.loads(capsys.readouterr().out.split("\n", 2)[-1])
    assert code == 1  # severe invention is an error -> non-zero exit for CI gating
    main(["scan", str(tmp_path / "s.npz"), "--out", str(tmp_path / "rep")])
    data = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert any(f["id"] == "FEATURE_INVENTION" for f in data["findings"])


def test_parse_faults():
    assert parse_faults(["underfit=0.5", "misattribution"]) == {"underfit": 0.5, "misattribution": 0.8}
    with pytest.raises(ValueError):
        parse_faults(["bogus"])


# ------------------------------------------------------------- ctd adapter
class _FakeDD:
    """Duck-typed stand-in for a CtD Analysis_DD_LFADS."""

    model_type = "LFADS"
    run_name = "fake_lfads"

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.d = {}
        for phase, R in (("train", 16), ("val", 4)):
            self.d[phase] = dict(
                rates=rng.uniform(0.1, 1, (R, 30, 12)), latents=rng.standard_normal((R, 30, 5)),
                spikes=rng.poisson(0.5, (R, 30, 12)), inputs=rng.standard_normal((R, 30, 3)),
                inferred=rng.standard_normal((R, 30, 3)), true_rates=rng.uniform(0.1, 1, (R, 30, 12)))

    def get_model_outputs(self, phase):
        return self.d[phase]["rates"], self.d[phase]["latents"]

    def get_spiking(self, phase):
        return self.d[phase]["spikes"]

    def get_model_inputs(self, phase):
        return self.d[phase]["spikes"][..., :9], self.d[phase]["inputs"]

    def get_inferred_inputs(self, phase):
        return self.d[phase]["inferred"]

    def get_true_rates(self, phase):
        return self.d[phase]["true_rates"]

    def get_trial_lens(self, phase):
        return np.full(len(self.d[phase]["rates"]), 30)


class _FakeTT:
    run_name = "fake_tt"

    def get_latents(self, phase):
        return np.zeros(({"train": 16, "val": 4}[phase], 30, 64))

    def get_inputs(self, phase):
        return np.zeros(({"train": 16, "val": 4}[phase], 30, 3))


def test_ctd_adapter_duck_typed():
    from ayna.adapters.ctd import from_ctd

    b = from_ctd(_FakeDD(), _FakeTT())
    assert b.spikes.shape == (20, 30, 12)
    assert b.heldout.sum() == 3 and b.heldout[-3:].all()
    assert b.val.sum() == 4 and b.val[-4:].all()
    assert b.inputs_kind == "inferred"
    assert b.true_latents.shape == (20, 30, 64)
    assert b.trial_lens is None  # all full length


# ================================================================ 0.2 additions
def test_recoverable_information_separates_underfit():
    from ayna import probes as P

    clean = P.recoverable_information(synth("c", [], seed=11, **SMALL))
    under = P.recoverable_information(synth("u", ["underfit=0.8"], seed=11, **SMALL))
    th = Thresholds()

    def flagged(r):
        return r["recoverable_fraction"] > th.underfit_recoverable and r["recoverable_z"] >= th.underfit_recoverable_z

    assert not flagged(clean)  # small recording: fraction can exceed 0.5% by chance, z cannot
    assert flagged(under)
    assert under["recoverable_split"] == "held-in -> held-out"


def test_recoverable_information_without_heldout_uses_neuron_split():
    from ayna import probes as P

    b = synth("u", ["underfit=0.8"], seed=12, **SMALL)
    b.heldout = None
    rec = P.recoverable_information(b)
    assert rec["recoverable_split"].startswith("neuron halves")
    assert rec["recoverable_fraction"] > 0.005


def test_leakage_uses_preceding_state():
    """Latents that contain the current input must not look like misattribution."""
    from ayna import probes as P

    b = synth("c", [], seed=13, **SMALL)
    b.latents = np.concatenate([b.latents, 3.0 * b.inputs], axis=2)  # z_t carries u_t (as in LFADS)
    leak = P.input_leakage(b)
    assert leak["input_leakage_same_instant_r2"] > 0.9
    assert leak["input_leakage_r2"] < 0.5


def test_bottleneck_makes_invention_comparative():
    b = synth("lf", ["feature_invention=0.4"], seed=14, **SMALL)
    b.meta["readout_bottleneck"] = 3
    r = scan(b)
    assert ("FEATURE_INVENTION", 0) in {(c.id, c.tier) for c in r.checks if c.status == "not_assessed"}
    f = next(f for f in r.findings if f.id == "FEATURE_INVENTION" and f.tier == 0)
    assert f.severity == "info" and "bottleneck" in f.what


def test_select_never_recommends_leaky_models():
    from ayna.select import select_from_rows

    rows = [
        {"name": "kl0", "recon": 0.130, "cycle": 0.47, "recon_metric": "val_bps", "input_leakage_r2": 0.89},
        {"name": "kl4", "recon": 0.125, "cycle": 0.39, "recon_metric": "val_bps", "input_leakage_r2": 0.25},
        {"name": "kl3", "recon": 0.120, "cycle": 0.41, "recon_metric": "val_bps", "input_leakage_r2": 0.20},
        {"name": "kl2", "recon": 0.118, "cycle": 0.29, "recon_metric": "val_bps", "input_leakage_r2": 0.35},
    ]
    rep = select_from_rows(rows)
    assert rep.knee == "kl4"
    assert rep.excluded == ["kl0"]
    assert any(f.id == "MISATTRIBUTION_EXCLUDED" for f in rep.findings)


def test_select_labels_show_the_varying_part():
    from ayna.select import _distinct

    names = [f"prefix=tt_3bff_latent_size={k}_max_epochs=1000_seed=0" for k in (3, 8, 64)]
    assert _distinct(names)[names[1]] == "tt_3bff_latent_size=8"


def test_ctd_adapter_tolerates_ctd_quirks():
    from ayna.adapters.ctd import from_ctd

    class Quirky(_FakeDD):
        evaluated = False

        class model:  # noqa: N801 - mimics dd.model
            class hparams:
                fac_dim = 2

            @classmethod
            def eval(cls):
                Quirky.evaluated = True

        def get_trial_lens(self, phase):  # CtD returns zeros for fixed-length data
            return np.zeros(len(self.d[phase]["rates"]))

        def get_true_rates(self, phase):  # some CtD datamodules raise here
            raise IndexError("list index out of range")

    b = from_ctd(Quirky(), _FakeTT(), with_dynamics=False)
    assert b.trial_lens is None
    assert b.true_rates is None
    assert Quirky.evaluated
    assert b.meta["readout_bottleneck"] == 2


def test_lfs_pointer_stubs_get_a_clear_error(tmp_path):
    from ayna.adapters.ctd import check_lfs

    (tmp_path / "model.pkl").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        check_lfs(tmp_path)


# ================================================================ 0.3 additions
def test_fixed_points_use_exact_gradients_in_high_dimension():
    from ayna import probes as P

    rng = np.random.default_rng(0)
    D = 40
    A = 0.5 * np.linalg.qr(rng.standard_normal((D, D)))[0]   # stable linear map, one fixed point at 0
    R, T = 30, 20
    b = Bundle(spikes=rng.poisson(1.0, (R, T, 5)), rates=np.ones((R, T, 5)),
               latents=rng.standard_normal((R, T, D)), inputs=np.zeros((R, T, 2)),
               meta={"inputs_kind": "inferred"})

    def f(z, u):
        return z @ A.T

    b.dynamics = f
    assert P.zero_input_fixed_points(b, n_starts=16)["fixed_points"] is None   # FD refuses at D=40
    f.vjp = lambda z, u, v: v @ A
    f.jacobian = lambda z, u: A
    fp = P.zero_input_fixed_points(b, n_starts=16, steps=300)
    assert fp["fixed_point_search"] == "exact gradients"
    assert fp["fixed_points"] == 1 and fp["stable_fixed_points"] == 1


def test_noiseless_inputs_drive_misattribution_truth():
    b = synth("m", ["misattribution=0.2"], seed=21, **SMALL)
    b.true_inputs_noiseless = b.true_inputs.copy()
    b.true_inputs = b.true_inputs + 5.0 * np.random.default_rng(0).standard_normal(b.true_inputs.shape)
    r = scan(b)
    assert r.metrics["input_r2"] < 0.5 <= r.metrics["input_r2_noiseless"]
    assert "DYNAMICAL_MISATTRIBUTION" not in r.ground_truth_flags()   # judged on the noiseless inputs


def test_ctd_adapter_trims_teacher_to_shrunk_dataset():
    from ayna.adapters.ctd import from_ctd

    class BigTT(_FakeTT):
        def get_latents(self, phase):
            return np.arange({"train": 20, "val": 6}[phase] * 30 * 4, dtype=float).reshape(-1, 30, 4)

        def get_inputs(self, phase):
            return np.zeros(({"train": 20, "val": 6}[phase], 30, 3))

    b = from_ctd(_FakeDD(), BigTT(), with_dynamics=False)
    assert b.true_latents.shape == (20, 30, 4)          # 16 train + 4 val, the model's trials
    np.testing.assert_array_equal(b.true_latents[16], BigTT().get_latents("val")[0])
    assert any("teacher trimmed" in n for n in b.notes)


def test_ctd_adapter_drops_misaligned_teacher():
    from ayna.adapters.ctd import from_ctd

    class ShortTT(_FakeTT):
        def get_latents(self, phase):
            return np.zeros(({"train": 16, "val": 4}[phase], 25, 4))   # wrong number of bins

    b = from_ctd(_FakeDD(), ShortTT(), with_dynamics=False)
    assert b.true_latents is None
    assert any("ground truth dropped" in n for n in b.notes)


def test_rollout_probe_does_not_judge_supplied_inputs():
    """Losing the state without *supplied* inputs is real input dependence (closed-loop tasks)."""
    b = synth("s", [], seed=31, inputs_kind="supplied", **SMALL)

    def leaky(z, u):            # dynamics that need their inputs to hold the state
        return 0.3 * z

    b.dynamics = leaky
    r = scan(b)
    chk = next(c for c in r.checks if c.id == "DYNAMICAL_MISATTRIBUTION" and c.tier == 1)
    assert chk.status == "not_assessed" and "supplied" in chk.reason
    assert "DYNAMICAL_MISATTRIBUTION" not in r.flagged()


def test_ctd_adapter_applies_trim_window_to_noiseless_inputs():
    from ayna.adapters.ctd import from_ctd

    class TrimTT(_FakeTT):
        trim_inds = (3, 32)     # CtD convention: slice [trim[0]-1 : trim[1]] -> 30 bins

        def get_true_inputs(self, phase):
            return np.ones(({"train": 16, "val": 4}[phase], 35, 3))

    b = from_ctd(_FakeDD(), TrimTT(), with_dynamics=False)
    assert b.true_inputs_noiseless is not None and b.true_inputs_noiseless.shape[1] == 30
