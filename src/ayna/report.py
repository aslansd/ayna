"""Findings and reports: what was found, where, how sure, and what was not checked."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SEVERITIES = ("error", "warning", "info")
TIER_NAMES = {0: "outputs only", 1: "dynamics callable", 2: "ground truth"}


@dataclass
class Finding:
    id: str
    severity: str
    tier: int
    what: str
    where: str = ""
    fix: str = ""
    evidence: dict = field(default_factory=dict)
    experimental: bool = False


@dataclass
class Check:
    """Outcome of one detector at one tier: flagged, clean, or not assessed (with why)."""

    id: str
    tier: int
    status: str  # "flagged" | "clean" | "not_assessed"
    reason: str = ""


@dataclass
class Report:
    name: str
    summary: str
    metrics: dict
    findings: list
    checks: list
    notes: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    version: str = ""
    source: str = ""

    # ------------------------------------------------------------- queries
    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" and f.tier < 2 for f in self.findings)

    def flagged(self, tier_max: int = 1) -> set:
        return {c.id for c in self.checks if c.status == "flagged" and c.tier <= tier_max}

    def ground_truth_flags(self) -> set:
        return {c.id for c in self.checks if c.status == "flagged" and c.tier == 2}

    def to_dict(self) -> dict:
        return {
            "ayna_version": self.version,
            "name": self.name,
            "source": self.source,
            "summary": self.summary,
            "metrics": self.metrics,
            "findings": [asdict(f) for f in self.findings],
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
            "thresholds": self.thresholds,
        }

    def __getitem__(self, key):  # report["findings"], like raftaar
        return self.to_dict()[key]

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(self.to_dict(), indent=2, default=_jsonable))
        (out / "report.md").write_text(self.render_markdown())
        (out / "report.txt").write_text(self.render_text())
        return out

    # ------------------------------------------------------------ rendering
    def render_text(self, color: bool = False) -> str:
        lines = [f"ayna {self.version} -- {self.name}", f"  {self.summary}", ""]
        lines.append("metrics")
        for key, value in self.metrics.items():
            lines.append(f"  {key:<28} {_fmt(value)}")
        lines.append("")
        model_findings = [f for f in self.findings if f.tier < 2]
        gt_findings = [f for f in self.findings if f.tier == 2]
        if not model_findings:
            lines.append("no findings from the model's outputs")
        for f in _sorted(model_findings):
            tag = f.severity.upper() + (" (experimental)" if f.experimental else "")
            lines.append(f"{tag:<9} {f.id}  [tier {f.tier}: {TIER_NAMES[f.tier]}]")
            if f.where:
                lines.append(f"          where: {f.where}")
            lines.append(_wrap(f.what, "          "))
            if f.fix:
                lines.append(_wrap("fix: " + f.fix, "          "))
            lines.append("")
        if gt_findings:
            lines.append("ground truth says")
            for f in _sorted(gt_findings):
                lines.append(f"  {f.id:<26} {f.what}")
            lines.append("")
        not_assessed = [c for c in self.checks if c.status == "not_assessed"]
        if not_assessed:
            lines.append("not assessed (a check that did not run is not a clean result)")
            for c in not_assessed:
                lines.append(f"  {c.id:<26} tier {c.tier}: {c.reason}")
            lines.append("")
        clean = sorted({c.id for c in self.checks if c.status == "clean" and c.tier < 2})
        if clean:
            lines.append("checked and clean: " + ", ".join(clean))
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines).rstrip() + "\n"

    def render_markdown(self) -> str:
        md = [f"# ayna report: {self.name}", "", f"*{self.summary}*", "", "## Metrics", "",
              "| metric | value |", "|---|---|"]
        md += [f"| {k} | {_fmt(v)} |" for k, v in self.metrics.items()]
        md += ["", "## Findings", ""]
        model_findings = [f for f in self.findings if f.tier < 2]
        if not model_findings:
            md.append("No findings from the model's outputs.")
        for f in _sorted(model_findings):
            exp = " *(experimental)*" if f.experimental else ""
            md += [f"### {f.severity.upper()} `{f.id}`{exp} -- tier {f.tier}", ""]
            if f.where:
                md.append(f"**Where:** {f.where}  ")
            md.append(f.what)
            if f.fix:
                md += ["", f"**Fix:** {f.fix}"]
            md.append("")
        gt = [f for f in self.findings if f.tier == 2]
        if gt:
            md += ["## Ground truth", ""] + [f"- `{f.id}`: {f.what}" for f in gt] + [""]
        na = [c for c in self.checks if c.status == "not_assessed"]
        if na:
            md += ["## Not assessed", ""] + [f"- `{c.id}` (tier {c.tier}): {c.reason}" for c in na] + [""]
        if self.notes:
            md += ["## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(md) + "\n"


def _sorted(findings):
    return sorted(findings, key=lambda f: (SEVERITIES.index(f.severity), f.id))


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _wrap(text: str, indent: str, width: int = 88) -> str:
    import textwrap

    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _jsonable(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)
