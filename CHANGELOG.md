# Changelog

## 0.2.0 — 2026-10-17

First contact with trained models. ayna was run on the 12 pretrained 3BFF
models that ship with the Computation-through-Dynamics Toolkit (6 NODE, 6
LFADS) and scored against the task-trained teacher. Three 0.1 detectors that
passed on synthetic data failed there, and were replaced. See VALIDATION.md.

### Changed
- **`UNDERFITTING`** now rests on *recoverable information*: a Poisson GLM
  with the model's log-rate as a fixed offset, plus other neurons' smoothed
  spikes, scored on validation trials, minus the same fit on trial-shuffled
  features. It must exceed 0.5% of the model's own bits/spike **and** be
  significant across trials (z ≥ 3). It works without held-out neurons by
  splitting neurons in half. The 0.1 residual-structure test fired on every
  trained NODE model and is kept as evidence only.
- **Input leakage** (`DYNAMICAL_MISATTRIBUTION`, tier 0) predicts each
  inferred input from the latent state *one step before* it, reduced to
  ≤ 20 PCs. Same-instant decoding was trivially high (0.88–0.99) on every
  LFADS model, because the state already contains the input.
- **`FEATURE_INVENTION`** is comparative only (info, not assessed in
  validation) when `meta["readout_bottleneck"]` is smaller than the latent
  dimension, e.g. LFADS generator state → factors.
- **`select`** never recommends a model whose inferred inputs leak
  (`MISATTRIBUTION_EXCLUDED`), shows leakage in its table, and labels models
  by the part of their name that varies. Split into `select_row` and
  `select_from_rows` so large models can be scanned one at a time.
- The fixed-point search is skipped above 32 latent dimensions, with a reason.
- `validate` reports Spearman rank agreement between each ground-truth-free
  metric and its ground truth next to the binary tables.
- The rate truth line for `UNDERFITTING` is 0.90 (it was 0.85 in the first
  0.1 draft).

### Added
- `ayna ctd SWEEP... --model-type SAE|LFADS --tt TEACHER`: scan, select and
  validate CtD runs in place, one subprocess per run.
- Tier-1 dynamics wrappers for CtD NODE (`MLPCell`) and LFADS (generator
  cell) models, attached automatically.
- CtD adapter robustness:
  - exports under `torch.no_grad()`, which previously ran out of memory;
  - calls `model.eval()`, since CtD leaves dropout on;
  - seeds torch and numpy, since the noisy teacher resamples its noise;
  - ignores zero or unavailable trial lengths;
  - tolerates optional getters that raise;
  - records the LFADS factor bottleneck;
  - gives a clear error for Git LFS pointer stubs.
- `tests/test_ctd_live.py`: live tests against CtD's pretrained models
  (skipped unless the CtD environment and models are present).
- `validation/`: raw outputs and per-model metrics behind VALIDATION.md.
- Docs: INSTALL.md, TESTING.md, PUBLISHING.md, ROADMAP.md, VALIDATION.md.

### Found
- CtD's `Comparison.compute_metrics` computes co-BPS from column
  `get_inputs()[0].shape[-1]` (3 on 3BFF) instead of the held-in count (50),
  so 47 held-in neurons count as held-out (0.1262 vs 0.1132 on NODE-8).

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
