# Changelog

## 0.3.0 — 2026-10-29

Held-out evidence. ayna was run on 15 models it had never seen: CtD NODE and
LFADS models trained here on RandomTarget, MultiTask and a regenerated 3BFF
dataset. No detector threshold was changed after seeing them. What held and
what failed is in VALIDATION.md, and the failures drive the 0.4 roadmap.

### Added
- **Sweep-relative truth** in `validate` and `ayna ctd`: a fault is present
  if its truth metric is more than 0.05 below the best model in the sweep.
  It is reported next to the absolute truth table. It was pre-registered
  (commit `b4e48f4`) before the 3BFF, MultiTask and LFADS held-out sweeps
  were scanned.
- **Noiseless input truth.** Bundles can carry `true_inputs_noiseless`. The
  CtD adapter exports the teacher's noiseless inputs (applying CtD's trim
  window), and the misattribution truth uses them, because CtD's Input R²
  against noisy inputs is capped near 0.2 on 3BFF. `input_r2` is still
  reported for comparison.
- **Exact-gradient fixed points.** A dynamics callable may offer `vjp(z, u, v)`
  and `jacobian(z, u)`. The CtD wrappers provide both through autograd, so
  the zero-input fixed-point search runs above 32 dimensions.
- **CtD adapter:**
  - task-specific teacher classes (`Analysis_TT_MultiTask`,
    `Analysis_TT_RandomTarget`);
  - trims the teacher to a prefix-shrunk dataset, with a note;
  - drops misaligned ground truth instead of failing, with a note.
- Bundle summaries show variable trial lengths.
- `experiments/ctd_tasks/`: scripts that generate CtD datasets, shrink them,
  train NODE and LFADS models on a CPU, and repair pickles, plus the exact
  grid used.

### Fixed
- **The tier-1 rollout probe judged supplied inputs.** A state that depends
  on *supplied* inputs (sensory feedback in a closed-loop task) is genuine
  input dependence, not misattribution. The probe now runs only for inferred
  inputs. This was found on RandomTarget, the first held-out task.

### Found (not fixed in 0.3; see ROADMAP.md)
- In closed-loop tasks, tier-0 lag-1 leakage flags well-inferred inputs
  (RandomTarget LFADS, KL = 0: leakage 0.88, noiseless Input R² 0.78).
- Recoverable information lacks power at low spike counts (RandomTarget,
  z ≈ 1.8).
- Cycle-consistency cannot see invention the readout can see (RandomTarget
  NODE-10, MultiTask NODE-20).
- Info-level `INPUT_INHERITED_STRUCTURE` findings count as flagged in
  validation, while the truth line uses the warning threshold.
- CtD's LFADS code is incompatible with PyTorch ≥ 2.7 for *training*
  (`ReduceLROnPlateau(verbose=...)`).

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
