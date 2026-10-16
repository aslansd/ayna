"""Turn evidence into findings.

Every detector reports at the highest tier its inputs allow and records a
:class:`~ayna.report.Check` either way, so a report always says what was
checked, what was clean, and what could not be assessed and why.

Thresholds are defaults calibrated on ayna's own synthetic faults
(``ayna validate``), not on CtD's trained models or on recordings. Treat them
as starting points; every one can be overridden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from . import __version__
from . import metrics as M
from . import probes as P
from .bundle import Bundle
from .report import Check, Finding, Report


@dataclass
class Thresholds:
    underfit_residual_ratio: float = 1.25   # residual top eigenvalue / null 95th percentile
    underfit_baseline_margin: float = 0.0   # co-BPS must beat the reference model by this
    invention_cycle_warning: float = 0.90
    invention_cycle_error: float = 0.70
    invention_pc_r2: float = 0.50           # latent PCs below this are "not in the rates"
    min_pc_share: float = 0.01              # ignore latent PCs with less variance than this
    nullspace_rel_gain: float = 0.01
    readout_linearity: float = 0.90
    leakage_r2: float = 0.50
    rollout_gap: float = 0.25               # hold-state R² minus autonomous-rollout R²
    rollout_hold_min: float = 0.30          # only meaningful when the state persists
    rollout_horizon: int = 10
    inherited_warning: float = 0.70
    inherited_info: float = 0.40
    input_lags: int = 10
    gt_rate_r2: float = 0.90
    gt_recovery_r2: float = 0.90
    gt_state_r2: float = 0.90
    gt_input_r2: float = 0.50
    gt_inherited_r2: float = 0.70

    def update(self, overrides: dict) -> "Thresholds":
        names = {f.name: f.type for f in fields(self)}
        for key, value in overrides.items():
            if key not in names:
                raise KeyError(f"unknown threshold '{key}'; known: {', '.join(names)}")
            setattr(self, key, int(value) if names[key] in (int, "int") else float(value))
        return self


def scan(b: Bundle, thresholds: Thresholds | None = None, nonlinear_fallback: bool = True,
         seed: int = 0) -> Report:
    """Run every detector the bundle supports and return a :class:`Report`."""
    th = thresholds or Thresholds()
    metrics, findings, checks = {}, [], []
    ctx = _Context(b, th, metrics, findings, checks, seed)
    ctx.readout()
    ctx.underfitting()
    ctx.invention(nonlinear_fallback)
    ctx.misattribution()
    ctx.inherited()
    if b.has_ground_truth:
        ctx.ground_truth()
    else:
        for det in ("UNDERFITTING", "FEATURE_INVENTION", "DYNAMICAL_MISATTRIBUTION"):
            checks.append(Check(det, 2, "not_assessed", "no ground truth in bundle (expected for real data)"))
    return Report(
        name=b.name, summary=b.summary(), metrics=metrics, findings=findings, checks=checks,
        notes=list(b.notes), thresholds=asdict(th), version=__version__, source=b.source,
    )


class _Context:
    def __init__(self, b, th, metrics, findings, checks, seed):
        self.b, self.th, self.m, self.f, self.c, self.seed = b, th, metrics, findings, checks, seed
        self.lat_tr, self.lat_va = b.flat("latents", "train"), b.flat("latents", "val")
        self.rate_tr, self.rate_va = b.flat("rates", "train"), b.flat("rates", "val")
        self.linear_ok = True

    def check(self, det, tier, flagged, reason=""):
        self.c.append(Check(det, tier, "flagged" if flagged else "clean", reason))

    def skip(self, det, tier, reason):
        self.c.append(Check(det, tier, "not_assessed", reason))

    # ------------------------------------------------------------ readout
    def readout(self):
        b, th = self.b, self.th
        lin = M.readout_linearity(self.lat_tr, self.rate_tr, self.lat_va, self.rate_va)
        self.m["readout_linearity"] = lin
        self.linear_ok = lin >= th.readout_linearity
        if not self.linear_ok:
            self.f.append(Finding(
                "NONLINEAR_READOUT", "info", 0,
                what=(f"An affine map from latents explains only {lin:.0%} of log-rate variance on "
                      "validation trials. The linear metrics (cycle-consistency, readout null space) "
                      "assume the linear-exponential link and can mislead here; ayna switched to the "
                      "nonlinear cycle-consistency."),
                fix="Read FEATURE_INVENTION from the nonlinear version; treat linear numbers as indicative only.",
                evidence={"readout_linearity": lin}))
        self.check("NONLINEAR_READOUT", 0, not self.linear_ok)

    # --------------------------------------------------------- underfitting
    def underfitting(self):
        b, th = self.b, self.th
        signals, evidence, where = [], {}, []
        res = P.residual_structure(b, seed=self.seed)
        self.m["residual_ratio"] = res["residual_ratio"]
        evidence.update(res)
        if res["residual_ratio"] > th.underfit_residual_ratio:
            lo, hi = res["residual_peak_bins"]
            signals.append(
                f"validation residuals share a slow component {res['residual_ratio']:.2f}x larger than "
                f"chance (95th percentile of shuffled residuals)")
            where.append(f"bins {lo}-{hi} of the trial; neurons {', '.join(map(str, res['residual_top_neurons']))} load most")
        if b.heldout is not None:
            cobps = M.bits_per_spike(b.flat("rates", "val", b.heldout), b.flat("spikes", "val", b.heldout))
            base = P.baseline_co_bps(b, seed=self.seed)
            self.m["co_bps"] = cobps
            self.m["baseline_co_bps"] = base["baseline_co_bps"]
            evidence.update(base, co_bps=cobps)
            if cobps < base["baseline_co_bps"] + th.underfit_baseline_margin:
                signals.append(
                    f"co-BPS {cobps:.3f} does not beat a smoothed-spikes Poisson GLM "
                    f"({base['baseline_co_bps']:.3f})")
        else:
            self.skip("UNDERFITTING/co-BPS", 0, "no held-out neurons marked, so co-BPS and the baseline comparison cannot run")
        flagged = bool(signals)
        if flagged:
            sev = "error" if len(signals) > 1 else "warning"
            self.f.append(Finding(
                "UNDERFITTING", sev, 0,
                what=("The model misses shared structure that the spikes contain: " + "; ".join(signals) + "."),
                where="; ".join(where),
                fix=("Increase capacity (latent size, generator units) or relax regularisation. If a "
                     "larger model only helps by inventing features, compare with `ayna select`."),
                evidence=evidence))
        self.check("UNDERFITTING", 0, flagged)

    # ---------------------------------------------------- feature invention
    def invention(self, nonlinear_fallback):
        b, th = self.b, self.th
        cc, per_pc, share, kept = M.cycle_consistency(self.lat_tr, self.rate_tr, self.lat_va)
        self.m["cycle_consistency"] = cc
        null, gains, _ = M.readout_null_directions(self.lat_tr, self.rate_tr, th.nullspace_rel_gain)
        method = "linear"
        score = cc
        if not self.linear_ok:
            if nonlinear_fallback:
                score = M.nonlinear_cycle_consistency(self.lat_tr, self.rate_tr, self.lat_va, seed=self.seed)
                self.m["cycle_consistency_nonlinear"] = score
                method = "nonlinear"
            else:
                self.skip("FEATURE_INVENTION", 0, "readout is nonlinear and the nonlinear fallback was disabled")
                return
        bad = [i for i in range(len(per_pc)) if per_pc[i] < th.invention_pc_r2 and share[i] >= th.min_pc_share]
        bad_share = float(np.sum(share[bad])) if bad else 0.0
        flagged = score < th.invention_cycle_warning
        if flagged:
            sev = "error" if score < th.invention_cycle_error else "warning"
            where = (f"latent PCs {', '.join(str(i + 1) for i in bad)} ({bad_share:.0%} of latent variance)"
                     if bad and method == "linear" else "")
            self.f.append(Finding(
                "FEATURE_INVENTION", sev, 0,
                what=(f"{method.capitalize()} cycle-consistency is {score:.2f}: part of the inferred latent "
                      "activity cannot be regenerated from the model's own predicted rates. Latent structure "
                      "the rates do not reflect costs nothing in reconstruction, so it is unconstrained by "
                      "data -- a likely invented feature (CtD Fig 1G)."),
                where=where,
                fix=("Do not interpret those dimensions. Reduce latent size or regularise toward the "
                     "simplest model that keeps co-BPS (`ayna select` on a sweep)."),
                evidence={"cycle_consistency": score, "method": method,
                          "per_pc_r2": [float(x) for x in per_pc],
                          "pc_variance_share": [float(x) for x in share],
                          "readout_rank_kept": int(kept)}))
        self.check("FEATURE_INVENTION", 0, flagged)

        # readout null space: the mechanism, reported on its own when harmless so far
        null_share = float(np.sum(share[null])) if len(null) else 0.0
        if len(null) and not flagged and self.linear_ok:
            self.f.append(Finding(
                "READOUT_NULLSPACE", "info", 0,
                what=(f"{len(null)} latent direction(s) barely affect predicted rates (gain under "
                      f"{th.nullspace_rel_gain:.0%} of the strongest). They hold {null_share:.1%} of latent "
                      "variance now, so this is harmless today, but cycle-consistency cannot see "
                      "features that grow there."),
                where=f"latent PCs {', '.join(str(i + 1) for i in null)}",
                evidence={"null_pcs": [int(i) for i in null], "null_variance_share": null_share}))
        self.check("READOUT_NULLSPACE", 0, bool(len(null)))

    # ------------------------------------------------------- misattribution
    def misattribution(self):
        b, th = self.b, self.th
        det = "DYNAMICAL_MISATTRIBUTION"
        signals, evidence = [], {}
        if b.inputs is not None and b.inputs_kind == "inferred":
            leak = P.input_leakage(b)
            self.m["input_leakage_r2"] = leak["input_leakage_r2"]
            evidence.update(leak)
            hit = leak["input_leakage_r2"] > th.leakage_r2
            if hit:
                dims = [i for i, v in enumerate(leak["input_leakage_per_dim"]) if v > th.leakage_r2]
                signals.append(("leakage", f"the latent state at the same instant predicts the inferred inputs "
                                f"(R² {leak['input_leakage_r2']:.2f}; input dims {', '.join(map(str, dims))})"))
            self.check(det, 0, hit)
        elif b.inputs is None:
            self.skip(det, 0, "bundle has no inputs; models without inferred inputs cannot misattribute to them")
        else:
            self.skip(det, 0, "inputs were supplied to the model, not inferred")

        if b.dynamics is not None:
            try:
                roll = P.autonomous_rollout(b, horizon=th.rollout_horizon, seed=self.seed)
            except Exception as exc:  # user code: report, never crash the scan
                roll = None
                self.skip(det, 1, f"dynamics callable raised {type(exc).__name__}: {exc}")
            if roll:
                self.m["rollout_r2_autonomous"] = roll["rollout_r2_autonomous"]
                self.m["rollout_r2_hold"] = roll["rollout_r2_hold"]
                evidence.update(roll)
                gap = roll["rollout_r2_hold"] - roll["rollout_r2_autonomous"]
                hit = roll["rollout_r2_hold"] >= th.rollout_hold_min and gap > th.rollout_gap
                if hit:
                    signals.append(("rollout", f"with inputs clamped to zero the model's own dynamics lose the "
                                    f"state: {th.rollout_horizon}-step R² {roll['rollout_r2_autonomous']:.2f} vs "
                                    f"{roll['rollout_r2_hold']:.2f} for simply holding it"))
                    fp = P.zero_input_fixed_points(b, seed=self.seed)
                    evidence.update(fp)
                    evidence["occupied_regions"] = P.occupied_clusters(b, seed=self.seed)
                self.check(det, 1, hit)
        else:
            self.skip(det, 1, "no dynamics callable attached (Python API only: load_bundle(path, dynamics=f))")

        if signals:
            sev = "error" if len(signals) > 1 else "warning"
            extra = ""
            if "stable_fixed_points" in evidence:
                extra = (f" Zero-input dynamics have {evidence['stable_fixed_points']} stable fixed point(s) "
                         f"while the latent state dwells in about {evidence['occupied_regions']} region(s).")
            self.f.append(Finding(
                det, sev, max(1 if s[0] == "rollout" else 0 for s in signals),
                what=("The inferred inputs appear to carry dynamics the model's generator should own: "
                      + "; ".join(s[1] for s in signals) + "." + extra
                      + " Reconstruction can be perfect in this failure mode (CtD Fig 1H, 6)."),
                fix=("Strengthen the input prior or penalty (e.g. LFADS controller KL) until inferred inputs "
                     "are sparse, and re-check. In closed-loop tasks, inputs may legitimately track state; "
                     "judge with task knowledge."),
                evidence=evidence, experimental=True))

    # ------------------------------------------------------ inherited input
    def inherited(self):
        b, th = self.b, self.th
        det = "INPUT_INHERITED_STRUCTURE"
        if b.inputs is None or b.inputs_kind != "supplied":
            self.skip(det, 0, "needs inputs supplied to the model")
            return
        res = P.input_inherited(b, n_lags=th.input_lags)
        r2 = res["input_filter_r2"]
        self.m["input_filter_r2"] = r2
        sev = "warning" if r2 >= th.inherited_warning else "info" if r2 >= th.inherited_info else None
        if sev:
            self.f.append(Finding(
                det, sev, 0,
                what=(f"{r2:.0%} of latent variance is a linear filter of the last {th.input_lags} bins of the "
                      "supplied inputs. That part would look right for any model that copies its inputs, so "
                      "good-looking latents say little about intrinsic dynamics here (CtD's RandomTarget case)."),
                fix=("Judge the model on what the inputs cannot explain: residual latents, held-out conditions, "
                     "or perturbations. Compare model classes on the input-unexplained part."),
                evidence=res))
        self.check(det, 0, bool(sev))

    # --------------------------------------------------------- ground truth
    def ground_truth(self):
        b, th = self.b, self.th
        if b.true_rates is not None:
            r = M.rate_r2(b.flat("true_rates", "val"), self.rate_va)
            self.m["rate_r2"] = r
        else:
            r = None
        if b.true_latents is not None:
            zt_tr, zt_va = b.flat("true_latents", "train"), b.flat("true_latents", "val")
            s = M.state_r2(zt_tr, self.lat_tr, zt_va, self.lat_va)
            rec = M.recovery_r2(zt_tr, self.lat_tr, zt_va, self.lat_va)
            self.m["state_r2"], self.m["recovery_r2"] = s, rec
            under = (r is not None and r < th.gt_rate_r2) or rec < th.gt_recovery_r2
            self._gt("UNDERFITTING", under,
                     f"Rate R² {_f(r)}, latent recovery R² {rec:.3f} (flag if < {th.gt_rate_r2} / {th.gt_recovery_r2})")
            self._gt("FEATURE_INVENTION", s < th.gt_state_r2,
                     f"State R² {s:.3f} (flag if < {th.gt_state_r2})")
        else:
            self.skip("FEATURE_INVENTION", 2, "no true latents")
        if b.inputs is not None and b.inputs_kind == "inferred" and b.true_inputs is not None:
            ir = M.input_r2(b.flat("true_inputs", "train"), b.flat("inputs", "train"),
                            b.flat("true_inputs", "val"), b.flat("inputs", "val"))
            self.m["input_r2"] = ir
            self._gt("DYNAMICAL_MISATTRIBUTION", ir < th.gt_input_r2, f"Input R² {ir:.3f} (flag if < {th.gt_input_r2})")
        else:
            self.skip("DYNAMICAL_MISATTRIBUTION", 2, "needs inferred inputs and true inputs")
        if b.inputs is not None and b.inputs_kind == "supplied" and b.true_latents is not None:
            res = P.input_inherited(b, n_lags=th.input_lags)
            tr2 = res.get("true_input_filter_r2", float("nan"))
            self.m["true_input_filter_r2"] = tr2
            self._gt("INPUT_INHERITED_STRUCTURE", tr2 >= th.gt_inherited_r2,
                     f"true latents are {tr2:.0%} explained by recent inputs")

    def _gt(self, det, flagged, text):
        self.check(det, 2, flagged)
        if flagged:
            self.f.append(Finding(det, "error", 2, what="confirmed: " + text))
        else:
            self.f.append(Finding(det, "info", 2, what="not present: " + text))


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"
