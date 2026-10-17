"""Model selection across a sweep without a single headline number.

CtD's discussion warns that ranking by reconstruction alone invites
Goodhart's law: on PhaseCodedMemory, Rate R² rose while State R² fell. ``select``
puts every model on the reconstruction (co-BPS) x simplicity (cycle-consistency)
plane, keeps the Pareto front, and recommends the knee. When the best-co-BPS
model is not the knee and pays for its edge in simplicity, it reports
``RECONSTRUCTION_TRAP``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from .bundle import Bundle
from .report import Finding


@dataclass
class SelectReport:
    rows: list
    front: list
    knee: str
    best_reconstruction: str
    findings: list = field(default_factory=list)
    recon_metric: str = "co_bps"
    excluded: list = field(default_factory=list)

    def to_dict(self):
        return {
            "rows": self.rows, "pareto_front": self.front, "recommended": self.knee,
            "best_reconstruction": self.best_reconstruction, "reconstruction_metric": self.recon_metric,
            "findings": [f.__dict__ for f in self.findings], "excluded_misattribution": self.excluded,
        }

    def render_text(self) -> str:
        import textwrap

        has_gt = any("state_r2" in r for r in self.rows)
        has_leak = any("input_leakage_r2" in r for r in self.rows)
        label = _distinct([r["name"] for r in self.rows])
        w = max(5, min(max(len(v) for v in label.values()), 44))
        head = f"{'model':<{w}}{'latents':>8}{self.recon_metric:>10}{'cycle':>8}"
        head += f"{'leak':>7}" if has_leak else ""
        head += f"{'stateR2':>9}{'rateR2':>8}" if has_gt else ""
        lines = [head, "-" * len(head)]
        for r in sorted(self.rows, key=lambda r: (r.get("latent_dim") or 0, r["name"])):
            if r["name"] == self.knee:
                mark = " <- recommended"
            elif r["name"] in self.excluded:
                mark = " x misattribution"
            elif r["name"] in self.front:
                mark = " *front"
            else:
                mark = ""
            line = f"{_short(label[r['name']], w):<{w}}{str(r.get('latent_dim', '')):>8}{r['recon']:>10.4f}{r['cycle']:>8.3f}"
            if has_leak:
                line += f"{r.get('input_leakage_r2', float('nan')):>7.2f}"
            if has_gt:
                line += f"{r.get('state_r2', float('nan')):>9.3f}{r.get('rate_r2', float('nan')):>8.3f}"
            lines.append(line + mark)
        lines.append("")
        for f in self.findings:
            lines.append(f"{f.severity.upper()} {f.id}")
            lines.append(textwrap.fill(f.what, 88, initial_indent="  ", subsequent_indent="  "))
        if not any(f.id == "RECONSTRUCTION_TRAP" for f in self.findings):
            lines.append(f"The best-reconstructing model ({self.best_reconstruction}) is also the recommended one."
                         if self.best_reconstruction == self.knee else
                         f"Recommended: {self.knee}.")
        return "\n".join(lines) + "\n"

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "select.json").write_text(json.dumps(self.to_dict(), indent=2, default=float))
        (out / "select.txt").write_text(self.render_text())
        return out

    def plot(self, path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=150)
        x = [r["cycle"] for r in self.rows]
        y = [r["recon"] for r in self.rows]
        ax.scatter(x, y, s=22, color="#9aa3ad", label="model", zorder=2)
        fr = sorted([r for r in self.rows if r["name"] in self.front], key=lambda r: r["cycle"])
        ax.plot([r["cycle"] for r in fr], [r["recon"] for r in fr], color="#3b6ea8", lw=1.5,
                marker="o", ms=5, label="Pareto front", zorder=3)
        for name, color, label in ((self.knee, "#2e8b57", "recommended"),
                                   (self.best_reconstruction, "#c0504d", f"best {self.recon_metric}")):
            r = next(r for r in self.rows if r["name"] == name)
            ax.scatter([r["cycle"]], [r["recon"]], s=90, facecolors="none", edgecolors=color, lw=2,
                       label=f"{label}: {name}", zorder=4)
        ax.set_xlabel("simplicity: cycle-consistency")
        ax.set_ylabel(f"reconstruction: {self.recon_metric}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path


def _distinct(names: list) -> dict:
    """Drop the prefix and suffix every name shares, keeping the key of a trailing ``key=value``.

    ``prefix=tt_3bff_latent_size=8_max_epochs=1000_seed=0`` -> ``tt_3bff_latent_size=8``.
    """
    import os

    if len(names) < 2:
        return {n: n for n in names}
    pre = os.path.commonprefix(names)
    suf = os.path.commonprefix([n[::-1] for n in names])[::-1]
    if "=" in pre:
        cut = pre.rfind("=")
        while cut > 0 and (pre[cut - 1].isalnum() or pre[cut - 1] == "_"):
            cut -= 1
        pre = pre[:cut]
    else:
        pre = ""
    out = {}
    for n in names:
        mid = n[len(pre): len(n) - len(suf) if suf else len(n)]
        out[n] = mid.strip("_") or n
    if len(set(out.values())) < len(names):
        return {n: n for n in names}
    return out


def _short(name: str, width: int) -> str:
    if len(name) <= width:
        return name
    keep = width - 1
    return name[: keep // 2] + "~" + name[len(name) - (keep - keep // 2):]


def _pareto(points):
    pts = np.asarray(points)
    front = []
    for i, p in enumerate(pts):
        dominated = np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1))
        if not dominated:
            front.append(i)
    return front


def select_row(b: Bundle, use_heldout: bool = True) -> dict:
    """The per-model numbers ``select`` needs (computed once, so large models can be scanned one by one)."""
    lat_tr, lat_va = b.flat("latents", "train"), b.flat("latents", "val")
    if use_heldout and b.heldout is not None:
        recon, metric = M.bits_per_spike(b.flat("rates", "val", b.heldout), b.flat("spikes", "val", b.heldout)), "co_bps"
    else:
        recon, metric = M.bits_per_spike(b.flat("rates", "val"), b.flat("spikes", "val")), "val_bps"
    cyc, _, _, _ = M.cycle_consistency(lat_tr, b.flat("rates", "train"), lat_va)
    row = {"name": b.name, "latent_dim": b.meta.get("latent_dim", b.n_latents), "recon": float(recon),
           "recon_metric": metric, "cycle": float(cyc)}
    if b.true_latents is not None:
        zt_tr, zt_va = b.flat("true_latents", "train"), b.flat("true_latents", "val")
        row["state_r2"] = M.state_r2(zt_tr, lat_tr, zt_va, lat_va)
    if b.true_rates is not None:
        row["rate_r2"] = M.rate_r2(b.flat("true_rates", "val"), b.flat("rates", "val"))
    if b.inputs is not None and b.inputs_kind == "inferred":
        from .probes import input_leakage

        row["input_leakage_r2"] = float(input_leakage(b)["input_leakage_r2"])
    return row


def select(bundles, cycle_gap: float = 0.1) -> SelectReport:
    """Compare a sweep of models fitted to the same data."""
    bundles = list(bundles)
    if len(bundles) < 2:
        raise ValueError("select needs at least two models")
    use_heldout = all(b.heldout is not None for b in bundles)
    return select_from_rows([select_row(b, use_heldout) for b in bundles], cycle_gap)


def select_from_rows(rows: list, cycle_gap: float = 0.1, leakage_r2: float = 0.5) -> SelectReport:
    """Pareto front, knee and RECONSTRUCTION_TRAP from precomputed rows (see :func:`select_row`).

    Models whose inferred inputs leak from the preceding state (``input_leakage_r2``
    above ``leakage_r2``) stay in the table but are never recommended: an
    autoencoder that offloads its dynamics to inputs reconstructs *better*, so
    reconstruction cannot be the judge for them.
    """
    rows = [dict(r) for r in rows]
    if len(rows) < 2:
        raise ValueError("select needs at least two models")
    metrics = {r.get("recon_metric", "co_bps") for r in rows}
    if len(metrics) > 1:
        raise ValueError(f"rows mix reconstruction metrics {sorted(metrics)}; compare like with like")
    use_heldout = metrics == {"co_bps"}
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        for i, r in enumerate(rows):
            r["name"] = f"{r['name']}#{i}"
    recon = np.array([r["recon"] for r in rows])
    cyc = np.array([r["cycle"] for r in rows])
    front_idx = _pareto(np.c_[recon, cyc])

    def norm(x):
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 0 else np.ones_like(x)

    nr, nc = norm(recon), norm(cyc)
    excluded = [i for i, r in enumerate(rows) if r.get("input_leakage_r2", 0.0) > leakage_r2]
    eligible = [i for i in range(len(rows)) if i not in excluded]
    if not eligible:
        eligible, excluded_note = list(range(len(rows))), True
    else:
        excluded_note = False
    elig_front = _pareto(np.c_[recon[eligible], cyc[eligible]])
    elig_front = [eligible[i] for i in elig_front]
    knee_i = min(elig_front, key=lambda i: (1 - nr[i]) ** 2 + (1 - nc[i]) ** 2)
    best_i = int(np.argmax(recon))
    rep = SelectReport(rows=rows, front=[rows[i]["name"] for i in front_idx], knee=rows[knee_i]["name"],
                       best_reconstruction=rows[best_i]["name"],
                       recon_metric="co_bps" if use_heldout else "val_bps",
                       excluded=[rows[i]["name"] for i in excluded])
    if excluded and not excluded_note:
        names_x = ", ".join(rows[i]["name"] for i in excluded)
        rep.findings.append(Finding(
            "MISATTRIBUTION_EXCLUDED", "warning", 0,
            what=(f"{len(excluded)} model(s) were not considered for recommendation because their inferred "
                  f"inputs are predictable from the preceding latent state (lag-1 leakage > {leakage_r2}): "
                  f"{names_x}. Offloading dynamics to inputs improves reconstruction, so these models "
                  "would otherwise win on it."),
            fix="Inspect them with `ayna scan`; raise the input penalty (e.g. LFADS controller KL)."))
    elif excluded_note:
        rep.findings.append(Finding(
            "MISATTRIBUTION_EXCLUDED", "warning", 0,
            what="Every model's inferred inputs leak from the preceding state; the recommendation ignores "
                 "this and should not be trusted until one model passes."))
    if best_i != knee_i and best_i not in excluded and cyc[knee_i] - cyc[best_i] > cycle_gap:
        best, knee = rows[best_i], rows[knee_i]
        text = (f"The best-reconstructing model {best['name']} gains {best['recon'] - knee['recon']:+.4f} "
                f"{rep.recon_metric} over {knee['name']} but its cycle-consistency is {best['cycle']:.2f} "
                f"vs {knee['cycle']:.2f}. Ranking by reconstruction alone would pick a model whose extra "
                "latent structure the data do not constrain.")
        if "state_r2" in best:
            text += (f" Ground truth agrees: State R² {best['state_r2']:.2f} vs {knee['state_r2']:.2f}.")
        rep.findings.append(Finding("RECONSTRUCTION_TRAP", "warning", 0, what=text,
                                    where=f"{best['name']} vs {knee['name']}",
                                    fix=f"Prefer {knee['name']} or another model on the Pareto front."))
    return rep
