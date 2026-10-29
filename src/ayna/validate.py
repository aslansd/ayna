"""Did the detectors predict what ground truth says?

``validate`` scans bundles that carry ground truth and compares each
ground-truth-free verdict (tier 0 and tier 1) with the tier-2 verdict on the
same bundle. It runs on ayna's synthetic corpus out of the box, and on any
directory of bundles exported from CtD or another simulator.

The synthetic corpus spans fault severities from barely-there to blatant, so
some disagreements are expected near the thresholds. That is the point: the
numbers say how far to trust each detector, and where.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scan import Thresholds, scan
from .synth import make_model_outputs, make_truth

DETECTORS = ("UNDERFITTING", "FEATURE_INVENTION", "DYNAMICAL_MISATTRIBUTION",
             "INPUT_INHERITED_STRUCTURE", "NONLINEAR_READOUT")

# Sweep-relative truth (pre-registered in 0.3): a fault is present if its truth metric is more than
# RELATIVE_MARGIN below the best model in the same sweep. First available metric wins.
RELATIVE_MARGIN = 0.05
RELATIVE_TRUTH = {
    "UNDERFITTING": ("rate_r2",),
    "FEATURE_INVENTION": ("state_r2",),
    "DYNAMICAL_MISATTRIBUTION": ("input_r2_noiseless", "input_r2"),
}

# (ground-truth-free metric, ground-truth metric, expected sign of the rank correlation)
PAIRS = (
    ("UNDERFITTING", "recoverable_fraction", "rate_r2", -1),
    ("FEATURE_INVENTION", "cycle_consistency", "state_r2", +1),
    ("DYNAMICAL_MISATTRIBUTION", "input_leakage_r2", "input_r2", -1),
    ("DYNAMICAL_MISATTRIBUTION", "rollout_gap", "input_r2", -1),
    ("DYNAMICAL_MISATTRIBUTION", "input_leakage_r2", "input_r2_noiseless", -1),
    ("DYNAMICAL_MISATTRIBUTION", "rollout_gap", "input_r2_noiseless", -1),
)


@dataclass
class Validation:
    cases: list = field(default_factory=list)   # one dict per bundle
    relative_margin: float | None = RELATIVE_MARGIN   # None: not a sweep (e.g. the synthetic corpus)

    def relative_truth(self) -> dict:
        """Sweep-relative truth: present if the truth metric is more than ``relative_margin`` below the
        best model in this set (see RELATIVE_TRUTH). Absolute truth lines from synthetic data do not
        transfer to tasks whose best achievable value is lower (noisy teachers, sparse spiking).

        Pre-registered in 0.3, before the 3BFF-reseeded, MultiTask and RandomTarget-LFADS sweeps
        were scanned. Returns ``{case index: {detector: bool}}``.
        """
        out = {i: {} for i in range(len(self.cases))}
        if self.relative_margin is None:
            return out
        for det, candidates in RELATIVE_TRUTH.items():
            for metric in candidates:
                vals = {i: c["metrics"].get(metric) for i, c in enumerate(self.cases)}
                vals = {i: v for i, v in vals.items() if v is not None and np.isfinite(v)}
                if len(vals) >= 2:
                    best = max(vals.values())
                    for i, v in vals.items():
                        out[i][det] = bool(v < best - self.relative_margin)
                    break
        return out

    def table(self, relative: bool = False) -> dict:
        """Confusion counts per (detector, tier), against absolute or sweep-relative truth."""
        out = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        rel = self.relative_truth() if relative else None
        for idx, case in enumerate(self.cases):
            for det in DETECTORS:
                truth = rel[idx].get(det) if relative else case["truth"].get(det)
                if truth is None:
                    continue
                for tier, pred in case["pred"].get(det, {}).items():
                    cell = out[(det, tier)]
                    key = ("tp" if pred else "fn") if truth else ("fp" if pred else "tn")
                    cell[key] += 1
        return dict(out)

    def correlations(self, min_cases: int = 4) -> list:
        """Spearman rank agreement between each ground-truth-free metric and its ground truth.

        Binary verdicts depend on where the truth line is drawn; rank agreement
        does not, which matters on real models where every value sits near it.
        """
        from scipy.stats import spearmanr

        out = []
        for det, x, y, sign in PAIRS:
            pts = [(c["metrics"][x], c["metrics"][y]) for c in self.cases
                   if x in c["metrics"] and y in c["metrics"]
                   and np.isfinite(c["metrics"][x]) and np.isfinite(c["metrics"][y])]
            if len(pts) < min_cases:
                continue
            rho = float(spearmanr(*zip(*pts)).statistic)
            out.append({"detector": det, "metric": x, "truth": y, "n": len(pts), "spearman": rho,
                        "expected_sign": sign, "agrees": bool(np.sign(rho) == sign)})
        return out

    def render_text(self) -> str:
        lines = [f"{len(self.cases)} bundles with ground truth", "",
                 f"{'detector':<27}{'tier':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'precision':>11}{'recall':>8}"]
        for (det, tier), c in sorted(self.table().items()):
            prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
            rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
            lines.append(f"{det:<27}{tier:>5}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}{c['tn']:>5}"
                         f"{_p(prec):>11}{_p(rec):>8}")
        if self.relative_margin is not None and any(self.relative_truth().values()) and len(self.cases) >= 2:
            lines += ["", f"against sweep-relative truth (more than {self.relative_margin} below the best model here)",
                      f"{'detector':<27}{'tier':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'precision':>11}{'recall':>8}"]
            for (det, tier), c in sorted(self.table(relative=True).items()):
                prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
                rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
                lines.append(f"{det:<27}{tier:>5}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}{c['tn']:>5}"
                             f"{_p(prec):>11}{_p(rec):>8}")
        corr = self.correlations()
        if corr:
            lines += ["", "rank agreement with ground truth (Spearman)",
                      f"{'metric':<24}{'truth':<20}{'n':>4}{'rho':>8}  expected"]
            for c in corr:
                lines.append(f"{c['metric']:<24}{c['truth']:<20}{c['n']:>4}{c['spearman']:>8.2f}  "
                             f"{'negative' if c['expected_sign'] < 0 else 'positive'}"
                             f"{'' if c['agrees'] else '  <- disagrees'}")
        misses = [c for c in self.cases if c["disagreements"]]
        if misses:
            lines += ["", "disagreements"]
            for c in misses:
                lines.append(f"  {c['name']:<34} {'; '.join(c['disagreements'])}")
        return "\n".join(lines) + "\n"

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table = {f"{d}|tier{t}": c for (d, t), c in self.table().items()}
        rel_table = {f"{d}|tier{t}": c for (d, t), c in self.table(relative=True).items()}
        (out / "validation.json").write_text(json.dumps(
            {"table": table, "relative_margin": self.relative_margin, "relative_table": rel_table,
             "correlations": self.correlations(), "cases": self.cases}, indent=2, default=float))
        (out / "validation.txt").write_text(self.render_text())
        return out


def _p(x):
    return "  n/a" if np.isnan(x) else f"{x:.2f}"


def _case_from_report(name, report, labels=None):
    flagged = {}
    for chk in report.checks:
        if chk.status == "not_assessed":
            continue
        flagged.setdefault(chk.id, {})[chk.tier] = chk.status == "flagged"
    truth = {det: tiers.pop(2) for det, tiers in flagged.items() if 2 in tiers}
    if labels:
        truth.update(labels)
    pred = {det: tiers for det, tiers in flagged.items() if det in DETECTORS and tiers}
    disagreements = []
    for det, tiers in pred.items():
        if det not in truth:
            continue
        for tier, p in tiers.items():
            if p != truth[det]:
                disagreements.append(f"{det} tier {tier}: {'flagged' if p else 'missed'}"
                                     f" (truth: {'present' if truth[det] else 'absent'})")
    return {"name": name, "truth": truth, "pred": pred, "disagreements": disagreements,
            "metrics": {k: float(v) for k, v in report.metrics.items() if isinstance(v, (int, float, np.floating))}}


def case_from_report_json(path) -> dict:
    """A validation case from a saved ``report.json`` (e.g. written by ``ayna ctd`` workers)."""
    from .report import Check

    data = json.loads(Path(path).read_text())

    class _R:  # the two attributes _case_from_report reads
        checks = [Check(**c) for c in data["checks"]]
        metrics = data["metrics"]

    return _case_from_report(data["name"], _R())


def corpus_specs(n_per_fault: int = 6, seed: int = 0) -> list:
    """Randomised synthetic cases spanning severities, noise and recording sizes."""
    rng = np.random.default_rng(seed)
    specs = []

    def add(kind, faults, inputs_kind, i):
        specs.append(dict(
            kind=kind, faults=faults, inputs_kind=inputs_kind,
            seed=int(rng.integers(1_000_000)),
            n_trials=int(rng.choice([120, 200, 300])),
            n_neurons=int(rng.choice([40, 60, 90])),
            rate_noise=float(rng.uniform(0.06, 0.2)),
            latent_dim=int(rng.choice([6, 8, 12])),
            name=f"{'+'.join(faults) or 'clean'}_{inputs_kind or 'noinput'}_{i}",
        ))

    def sev(lo, hi, i):
        """Stratified severity: evenly spread over [lo, hi] with jitter, so every
        corpus covers the whole range instead of whatever a few draws give."""
        width = (hi - lo) / n_per_fault
        return round(float(lo + width * (i + rng.uniform(0.1, 0.9))), 2)

    for i in range(n_per_fault):
        add("flipflop", {}, "inferred", i)
        add("flipflop", {}, None, i)
        add("flipflop", {}, "supplied", i)
        add("flipflop", {"underfit": sev(0.1, 1.0, i)}, "inferred", i)
        add("flipflop", {"feature_invention": sev(0.02, 0.5, i)}, None, i)
        add("flipflop", {"misattribution": sev(0.1, 1.0, i)}, "inferred", i)
        add("input_driven", {"input_driven": 1.0}, "supplied", i)
        add("flipflop", {"nonlinear_readout": 1.0}, None, i)
        if i % 2 == 0:
            add("flipflop", {"underfit": sev(0.3, 1.0, i), "feature_invention": sev(0.1, 0.4, i)}, "inferred", i)
    return specs


def validate_synthetic(n_per_fault: int = 6, seed: int = 0, thresholds: Thresholds | None = None,
                       progress=None) -> Validation:
    val = Validation(relative_margin=None)   # a corpus of unrelated systems, not one sweep
    specs = corpus_specs(n_per_fault, seed)
    for k, spec in enumerate(specs):
        truth = make_truth(spec["kind"], spec["n_trials"], 100, spec["n_neurons"], max(8, spec["n_neurons"] // 4),
                           nonlinear=bool(spec["faults"].get("nonlinear_readout")), seed=spec["seed"])
        b = make_model_outputs(truth, spec["latent_dim"], spec["faults"], spec["inputs_kind"],
                               spec["rate_noise"], seed=spec["seed"], name=spec["name"])
        report = scan(b, thresholds)
        labels = {"NONLINEAR_READOUT": bool(spec["faults"].get("nonlinear_readout"))}
        case = _case_from_report(spec["name"], report, labels)
        case["spec"] = spec
        val.cases.append(case)
        if progress:
            progress(k + 1, len(specs), spec["name"])
    return val


def validate_bundles(bundles, thresholds: Thresholds | None = None, progress=None) -> Validation:
    val = Validation()
    bundles = list(bundles)
    for k, b in enumerate(bundles):
        if not b.has_ground_truth:
            continue
        val.cases.append(_case_from_report(b.name, scan(b, thresholds)))
        if progress:
            progress(k + 1, len(bundles), b.name)
    return val
