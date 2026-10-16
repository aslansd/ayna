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

    def to_dict(self):
        return {
            "rows": self.rows, "pareto_front": self.front, "recommended": self.knee,
            "best_reconstruction": self.best_reconstruction, "reconstruction_metric": self.recon_metric,
            "findings": [f.__dict__ for f in self.findings],
        }

    def render_text(self) -> str:
        has_gt = any("state_r2" in r for r in self.rows)
        head = f"{'model':<22}{'latents':>8}{self.recon_metric:>10}{'cycle':>8}"
        head += f"{'stateR2':>9}{'rateR2':>8}" if has_gt else ""
        lines = [head, "-" * len(head)]
        for r in sorted(self.rows, key=lambda r: (r.get("latent_dim") or 0, r["name"])):
            mark = " <- recommended" if r["name"] == self.knee else (" *front" if r["name"] in self.front else "")
            line = f"{r['name']:<22}{str(r.get('latent_dim', '')):>8}{r['recon']:>10.4f}{r['cycle']:>8.3f}"
            if has_gt:
                line += f"{r.get('state_r2', float('nan')):>9.3f}{r.get('rate_r2', float('nan')):>8.3f}"
            lines.append(line + mark)
        lines.append("")
        for f in self.findings:
            lines.append(f"{f.severity.upper()} {f.id}")
            import textwrap

            lines.append(textwrap.fill(f.what, 88, initial_indent="  ", subsequent_indent="  "))
        if not self.findings:
            lines.append(f"The best-reconstructing model ({self.best_reconstruction}) is also the recommended one.")
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


def _pareto(points):
    pts = np.asarray(points)
    front = []
    for i, p in enumerate(pts):
        dominated = np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1))
        if not dominated:
            front.append(i)
    return front


def select(bundles, cycle_gap: float = 0.1) -> SelectReport:
    """Compare a sweep of models fitted to the same data."""
    bundles = list(bundles)
    if len(bundles) < 2:
        raise ValueError("select needs at least two models")
    rows = []
    use_heldout = all(b.heldout is not None for b in bundles)
    for b in bundles:
        lat_tr, lat_va = b.flat("latents", "train"), b.flat("latents", "val")
        if use_heldout:
            recon = M.bits_per_spike(b.flat("rates", "val", b.heldout), b.flat("spikes", "val", b.heldout))
        else:
            recon = M.bits_per_spike(b.flat("rates", "val"), b.flat("spikes", "val"))
        cyc, _, _, _ = M.cycle_consistency(lat_tr, b.flat("rates", "train"), lat_va)
        row = {"name": b.name, "latent_dim": b.meta.get("latent_dim", b.n_latents), "recon": float(recon),
               "cycle": float(cyc)}
        if b.true_latents is not None:
            zt_tr, zt_va = b.flat("true_latents", "train"), b.flat("true_latents", "val")
            row["state_r2"] = M.state_r2(zt_tr, lat_tr, zt_va, lat_va)
        if b.true_rates is not None:
            row["rate_r2"] = M.rate_r2(b.flat("true_rates", "val"), b.flat("rates", "val"))
        rows.append(row)

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
    knee_i = min(front_idx, key=lambda i: (1 - nr[i]) ** 2 + (1 - nc[i]) ** 2)
    best_i = int(np.argmax(recon))
    rep = SelectReport(rows=rows, front=[rows[i]["name"] for i in front_idx], knee=rows[knee_i]["name"],
                       best_reconstruction=rows[best_i]["name"],
                       recon_metric="co_bps" if use_heldout else "val_bps")
    if best_i != knee_i and cyc[knee_i] - cyc[best_i] > cycle_gap:
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
