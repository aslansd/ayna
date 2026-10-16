"""ayna command line.

    ayna synth NAME [--faults underfit=0.5 feature_invention ...] [--sweep]
    ayna scan BUNDLE [--out DIR] [--json] [--set name=value ...] [--daftar]
    ayna select BUNDLE... | DIR [--out DIR] [--plot]
    ayna validate [DIR] [--n 6] [--out DIR]
    ayna info BUNDLE
    ayna export-ctd DD_RUN_DIR OUT.npz --model-type SAE|LFADS [--tt TT_RUN_DIR]
    ayna doctor

``scan`` exits 1 when an error-level finding is raised from the model's
outputs, so it can gate a pipeline or CI job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _bundles_from(paths):
    from .bundle import load_bundle

    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        raise SystemExit("no .npz bundles found")
    return [load_bundle(f) for f in files]


def _overrides(items):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects name=value, got '{item}'")
        out[key.strip()] = value
    return out


def cmd_synth(args):
    from .synth import synth, synth_sweep

    out = Path(args.out)
    if args.sweep:
        target = out / args.name
        bundles = synth_sweep(n_trials=args.trials, n_neurons=args.neurons, truth_seed=args.seed)
        for b in bundles:
            b.save(target / f"{b.name}.npz")
        print(f"wrote {len(bundles)} bundles (latent sizes 2-64, simulated) to {target}/")
        return 0
    b = synth(args.name, args.faults, latent_dim=args.latent_dim,
              inputs_kind=None if args.inputs == "auto" else args.inputs,
              n_trials=args.trials, n_neurons=args.neurons, seed=args.seed)
    path = b.save(out / f"{args.name}.npz")
    faults = b.meta["synth"]["faults"] or "none"
    print(f"wrote {path}  ({b.summary()}; injected faults: {faults})")
    print("note: a .npz cannot hold the dynamics callable, so tier-1 probes need the Python API")
    return 0


def cmd_scan(args):
    from .bundle import load_bundle
    from .scan import Thresholds, scan
    from . import provenance

    b = load_bundle(args.bundle)
    th = Thresholds().update(_overrides(args.set))
    if args.daftar and not provenance.available():
        print("daftar is not installed; run `pip install \"ayna[provenance]\"`", file=sys.stderr)
    with provenance.tracked("ayna scan", params={"bundle": str(args.bundle), **th.__dict__},
                            inputs=[args.bundle] if args.daftar else []) if args.daftar else _null() as run:
        report = scan(b, th, nonlinear_fallback=not args.no_nonlinear)
        if args.out:
            report.save(args.out)
        if run is not None:
            provenance.log_report(run, report, args.out)
    if args.json:
        from .report import _jsonable

        print(json.dumps(report.to_dict(), indent=2, default=_jsonable))
    else:
        print(report.render_text(), end="")
        if args.out:
            print(f"\nreport written to {args.out}/report.{{txt,md,json}}")
    return 1 if report.has_errors else 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def cmd_select(args):
    from .select import select

    rep = select(_bundles_from(args.bundles))
    print(rep.render_text(), end="")
    if args.out:
        rep.save(args.out)
        if args.plot:
            try:
                rep.plot(Path(args.out) / "select.png")
            except ImportError:
                print("plot needs matplotlib: pip install \"ayna[plot]\"", file=sys.stderr)
        print(f"\nwritten to {args.out}/")
    return 0


def cmd_validate(args):
    from .scan import Thresholds
    from .validate import validate_bundles, validate_synthetic

    th = Thresholds().update(_overrides(args.set))

    def progress(i, n, name):
        print(f"\r  [{i}/{n}] {name:<40}", end="", file=sys.stderr, flush=True)

    if args.bundles:
        val = validate_bundles(_bundles_from(args.bundles), th, progress)
        if not val.cases:
            raise SystemExit("none of these bundles carries ground truth; nothing to validate against")
    else:
        val = validate_synthetic(args.n, args.seed, th, progress)
    print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    print(val.render_text(), end="")
    if not args.bundles:
        print("\ncaveat: these faults were injected by ayna's own simulator, so this measures whether the "
              "detectors work as designed -- not whether they transfer to trained models or recordings.")
    if args.out:
        val.save(args.out)
    return 0


def cmd_info(args):
    from .bundle import load_bundle

    b = load_bundle(args.bundle)
    print(f"{args.bundle}: {b.summary()}")
    for k, v in sorted(b.meta.items()):
        print(f"  {k}: {v}")
    for n in b.notes:
        print(f"  note: {n}")
    return 0


def cmd_export_ctd(args):
    from .adapters.ctd import export_run

    path = export_run(args.dd_dir, args.out, args.model_type, args.tt, args.inputs)
    print(f"wrote {path}")
    return 0


def cmd_doctor(args):
    import platform

    print(f"ayna {__version__} on Python {platform.python_version()}")
    for mod, why in (("numpy", "core"), ("scipy", "core"), ("sklearn", "core"),
                     ("matplotlib", "extra [plot]: select --plot"),
                     ("daftar", "extra [provenance]: scan --daftar"),
                     ("ctd", "export-ctd (install the CtD repository itself)"),
                     ("torch", "needed by CtD, never by ayna")):
        try:
            m = __import__(mod)
            print(f"  ok       {mod:<11} {getattr(m, '__version__', '')}  ({why})")
        except ImportError:
            print(f"  missing  {mod:<11} ({why})")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="ayna", description="Know whether your inferred dynamics are real, before you interpret them.")
    p.add_argument("--version", action="version", version=f"ayna {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic bundle with injected faults")
    s.add_argument("name")
    s.add_argument("--faults", nargs="*", default=[], metavar="FAULT[=SEVERITY]",
                   help="underfit, feature_invention, misattribution, input_driven, nonlinear_readout")
    s.add_argument("--sweep", action="store_true", help="write a simulated latent-size sweep instead")
    s.add_argument("--latent-dim", type=int, default=8)
    s.add_argument("--inputs", choices=["auto", "inferred", "supplied", "none"], default="auto")
    s.add_argument("--trials", type=int, default=200)
    s.add_argument("--neurons", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="datasets")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("scan", help="diagnose one fitted model")
    s.add_argument("bundle")
    s.add_argument("--out", help="write report.{txt,md,json} here")
    s.add_argument("--json", action="store_true")
    s.add_argument("--set", nargs="*", metavar="NAME=VALUE", help="override thresholds")
    s.add_argument("--no-nonlinear", action="store_true", help="skip the nonlinear cycle-consistency fallback")
    s.add_argument("--daftar", action="store_true", help="record the scan with daftar")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("select", help="compare a sweep of models fitted to the same data")
    s.add_argument("bundles", nargs="+")
    s.add_argument("--out")
    s.add_argument("--plot", action="store_true")
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("validate", help="check detectors against ground truth")
    s.add_argument("bundles", nargs="*", help="bundles or directories with ground truth (default: synthetic corpus)")
    s.add_argument("--n", type=int, default=6, help="synthetic cases per fault type")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--set", nargs="*", metavar="NAME=VALUE")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("info", help="describe a bundle")
    s.add_argument("bundle")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("export-ctd", help="export a CtD run to a bundle (run inside the CtD environment)")
    s.add_argument("dd_dir")
    s.add_argument("out")
    s.add_argument("--model-type", required=True, choices=["SAE", "LFADS"])
    s.add_argument("--tt", help="task-trained run directory, for ground truth")
    s.add_argument("--inputs", choices=["supplied", "inferred"])
    s.set_defaults(fn=cmd_export_ctd)

    s = sub.add_parser("doctor", help="report versions and optional extras")
    s.set_defaults(fn=cmd_doctor)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .bundle import BundleError

    try:
        return args.fn(args)
    except BundleError as exc:
        print(f"ayna: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
