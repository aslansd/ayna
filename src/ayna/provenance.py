"""Optional provenance through daftar (``pip install "ayna[provenance]"``).

Each tracked scan records the bundle's hash, the thresholds, ayna's version
and environment, and the headline metrics, so two scans can be diffed later
with ``daftar diff``.
"""

from __future__ import annotations

from contextlib import contextmanager


def available() -> bool:
    try:
        import daftar  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def tracked(label: str, params: dict, inputs=(), seed: int | None = None):
    """Yield a daftar run, or ``None`` when daftar is not installed."""
    if not available():
        yield None
        return
    import daftar

    with daftar.track(label, params=params, seed=seed, tags=["ayna"]) as run:
        for path in inputs:
            run.add_input(path)
        yield run


def log_report(run, report, out_dir=None):
    if run is None:
        return
    for key, value in report.metrics.items():
        if isinstance(value, (int, float)):
            run.log_result(f"metrics.{key}", float(value))
    run.log_result("findings", sorted(f"{f.severity}:{f.id}" for f in report.findings if f.tier < 2))
    if out_dir is not None:
        from pathlib import Path

        for name in ("report.json", "report.md"):
            path = Path(out_dir) / name
            if path.exists():
                run.add_output(path)
