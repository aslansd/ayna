# Changelog

## 0.1.0 — 2026-10-16

First release.

- Bundle format (`.npz`) with validation, train/val splits, padded trials.
- Numpy ports of CtD metrics: Rate R², State R², Input R², co-BPS, linear
  cycle-consistency; plus latent recovery R², readout linearity, readout null
  directions and a scikit-learn nonlinear cycle-consistency.
- Detectors: `UNDERFITTING`, `FEATURE_INVENTION`, `READOUT_NULLSPACE`,
  `NONLINEAR_READOUT`, `DYNAMICAL_MISATTRIBUTION` (experimental; tier 0 input
  leakage, tier 1 autonomous rollout + zero-input fixed points),
  `INPUT_INHERITED_STRUCTURE`, and `RECONSTRUCTION_TRAP` via `select`.
- Synthetic ground truth (3-bit flip-flop, input-driven system, nonlinear
  readout) with injected faults; simulated latent-size sweep.
- `validate` against ground truth, on the synthetic corpus or exported bundles.
- CtD exporter (duck-typed; not yet run against a live CtD install).
- Optional daftar provenance for scans.
