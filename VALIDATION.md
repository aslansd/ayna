# Validation

How far to trust each detector, and where it breaks. Raw outputs for every
table here are in [`validation/`](validation/), and [TESTING.md](TESTING.md)
gives the commands that regenerate them.

There are two kinds of evidence, and they answer different questions:

- **Synthetic corpus** (`ayna validate`). Do the detectors find faults that
  were deliberately injected? ayna's own simulator injects them and the same
  author wrote the detectors, so this tests design, not transfer.
- **Trained models** (`ayna ctd`). Do the detectors agree with ground truth on
  models that were actually trained? These are CtD's 12 pretrained 3BFF
  models, scored against the task-trained teacher's latents, rates and inputs.

## 1. Trained models: CtD 3BFF walkthrough

CtD 1.0.2 (commit `051e11e`), `examples/walkthrough_models/`:

- `tt_3bff`: the task-trained NoisyGRU teacher, which supplies ground truth;
- `node_sweep`: NODE sequential autoencoders, latent size 3 to 64, supplied
  inputs, 50 held-in and 10 held-out neurons;
- `inputinf_sweep`: LFADS with inferred inputs, controller KL scale 0 to 1e-2,
  50 neurons and no held-out neurons.

One seed per configuration, CPU only, exports seeded.

### NODE sweep: supplied inputs

| latent size | co-BPS | recoverable info | cycle-con | ayna flags | Rate R² | State R² | ground truth flags |
|---|---|---|---|---|---|---|---|
| 3 | 0.1043 | **0.98%** | 1.00 | `UNDERFITTING` | 0.848 | 0.962 | `UNDERFITTING` |
| 5 | 0.1122 | **0.51%** | 1.00 | `UNDERFITTING` | 0.891 | 0.973 | `UNDERFITTING` |
| 8 | 0.1131 | 0.31% | 1.00 | — | 0.908 | 0.959 | — |
| 16 | 0.1141 | 0.23% | **0.87** | `FEATURE_INVENTION` | 0.909 | 0.851 | `FEATURE_INVENTION` |
| 32 | 0.1139 | 0.22% | **0.63** | `FEATURE_INVENTION` | 0.910 | 0.648 | `FEATURE_INVENTION` |
| 64 | 0.1155 | 0.21% | **0.61** | `FEATURE_INVENTION` | 0.920 | 0.562 | `FEATURE_INVENTION` |

- **Binary agreement: 12 of 12** (2 detectors × 6 models).
- **Rank agreement:** recoverable information vs Rate R², Spearman −1.00;
  cycle-consistency vs State R², +0.94.
- **`select`** recommends NODE-8. It reports NODE-64 as the reconstruction
  trap: +0.0024 co-BPS, but cycle-consistency 0.61 vs 1.00, and ground truth
  agrees (State R² 0.56 vs 0.96). This is CtD Fig 5 recovered without ground
  truth.
- **Caveat: NODE-5 is flagged with almost no margin.** Its recoverable
  information is 0.51% against a 0.5% threshold, and the ground-truth Rate
  R² is 0.891 against a 0.90 line. Both lines were drawn before these models
  were seen. Neither has any precision near the boundary.

### LFADS sweep: inferred inputs

| KL scale | lag-1 leakage | rollout gap | cycle-con | ayna flags | Input R² | Rate R² | State R² |
|---|---|---|---|---|---|---|---|
| 0 | **0.90** | **0.45** | 0.47 | `DYNAMICAL_MISATTRIBUTION` | 0.036 | 0.895 | 0.909 |
| 1e-6 | **0.89** | **0.42** | 0.46 | `DYNAMICAL_MISATTRIBUTION` | 0.039 | 0.896 | 0.908 |
| 1e-5 | **0.75** | 0.15 | 0.47 | `DYNAMICAL_MISATTRIBUTION` | 0.084 | 0.896 | 0.900 |
| 1e-4 | 0.27 | −0.01 | 0.39 | — | **0.206** | 0.892 | 0.893 |
| 1e-3 | 0.21 | 0.20 | 0.41 | — | 0.114 | 0.881 | 0.897 |
| 1e-2 | 0.36 | **1.38** | 0.29 | (rollout only) | 0.002 | 0.865 | 0.861 |

**What agrees:**

- **Ayna's leakage flag and CtD's own reading coincide.** ayna flags the three
  lowest-KL models, which are the ones CtD's paper describes as offloading
  their dynamics to the controller (Fig 6). It leaves the KL = 1e-4 model
  unflagged; that model has the best Input R² and is the one the paper
  highlights as correct.
- **`select` recommends KL = 1e-4.** Ranked by reconstruction alone it would
  have chosen KL = 1e-5, because misattributed autoencoders reconstruct
  better. 0.2 therefore excludes leaky models from the recommendation.
- **Rank agreement:**
  - rollout gap vs Input R², Spearman −0.94;
  - recoverable information vs Rate R², −1.00;
  - cycle-consistency vs State R², +0.94.

**What does not:**

- **Binary tables against the synthetic truth lines.** Every LFADS Input R²
  is below 0.21 because CtD's true 3BFF inputs are noisy. So the 0.5 line
  calls all six models misattributed, and ayna "misses" three. Every Rate R²
  is between 0.86 and 0.90, so the 0.90 line calls all six underfit, and
  ayna's recoverable-information values (0.11–0.40%) stay under its 0.5%
  threshold. The ranks are right and the lines are not. Real-model truth
  lines are on the [roadmap](ROADMAP.md).
- **Lag-1 leakage vs Input R² is only −0.60.** The outlier is KL = 1e-2.
  Its inputs are suppressed rather than offloaded (Input R² 0.002), which is
  a different failure. Leakage is moderate there (0.36) and the rollout gap
  is extreme (1.38): its own dynamics cannot hold the state.
- **Cycle-consistency is low for every LFADS model (0.29–0.47).** It
  still ranks them correctly (+0.94). LFADS reads its 128-dimensional
  generator state through a low-dimensional factor layer, so most latent
  directions are silent by design. CtD reports the same range. ayna 0.2
  therefore reports cycle-consistency as comparative only for models with a
  declared readout bottleneck.
- **The zero-input fixed-point search is not attempted.** With 128 latent
  dimensions, the finite-difference search is skipped and the report says so.

### What 0.1 got wrong on these models

0.2 exists because of this run. On the same 12 models, the 0.1 detectors:

- flagged `UNDERFITTING` on all six NODE models, including the clean NODE-8,
  because shared residual structure exists at every size. Recoverable
  information replaced it: it asks whether other neurons' spikes still
  improve predictions, against a trial-shuffled control;
- gave input leakage of 0.88–0.99 on every LFADS model, because the
  same-instant generator state already contains the input. Leakage now uses
  the preceding state;
- flagged `FEATURE_INVENTION` on every LFADS model through absolute
  cycle-consistency thresholds;
- would have recommended a misattributed LFADS model in `select`.

### A bug found in CtD along the way

`ctd/comparison/comparison.py` computes co-BPS on
`inf_rates_val[..., n_input_neurons:]`, where
`n_input_neurons = self.analyses[i].get_inputs()[0].shape[-1]`. For
data-trained analyses, `get_inputs()` returns the *external task inputs*, so
on 3BFF the slice starts at column 3, not 50. That counts 47 held-in neurons
as held-out.

On the NODE-8 model, CtD's code gives 0.1262. The true held-out neurons give
0.1132. ayna's co-BPS matches the latter to within 1e-4, and
`tests/test_ctd_live.py` checks this. Using
`get_model_inputs(phase)[0].shape[-1]`, the held-in spike count, fixes it.

## 2. Synthetic corpus

`ayna validate --n 10 --seed 7` gives 85 bundles. The cases are 3-bit
flip-flop and input-driven systems with one or two faults, injected at
stratified severities, with varied noise, trial counts and neuron counts.
Thresholds were set on seed 0. Seed 7 was not used to set any threshold.

| detector | tier | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|---|
| `FEATURE_INVENTION` | 0 | 14 | 0 | 0 | 71 | 1.00 | 1.00 |
| `DYNAMICAL_MISATTRIBUTION` | 0 | 5 | 0 | 0 | 30 | 1.00 | 1.00 |
| `DYNAMICAL_MISATTRIBUTION` | 1 | 5 | 0 | 0 | 30 | 1.00 | 1.00 |
| `INPUT_INHERITED_STRUCTURE` | 0 | 10 | 0 | 0 | 10 | 1.00 | 1.00 |
| `NONLINEAR_READOUT` | 0 | 10 | 0 | 0 | 75 | 1.00 | 1.00 |
| `UNDERFITTING` | 0 | 13 | 2 | 2 | 68 | 0.87 | 0.87 |

The `UNDERFITTING` disagreements:

- **The two "false positives"** are injected underfits so mild that Rate R²
  stays above 0.90.
- **The two misses** are clean models on a nonlinear readout, where the
  nonlinearity amplifies rate noise below the truth line without leaving
  recoverable structure.

On the calibration corpus (seed 0), the one disagreement is a misattribution
case with Input R² 0.54 and leakage 0.53, right at both lines.

Rank agreement on the synthetic corpus is weaker than on the real sweeps:
recoverable information vs Rate R², −0.53; cycle-consistency vs State R²,
+0.49. That is expected. Most synthetic cases are clean for any given
detector, so the metric sits near its floor and ties dominate the ranks.

## How the thresholds were set

| threshold | value | set on |
|---|---|---|
| `underfit_recoverable` | 0.5% of the model's bits/spike | synthetic seed 0: clean ≤ 0.40%, injected underfits ≥ 0.97% |
| `underfit_recoverable_z` | 3.0 | added after a small clean synthetic recording reached 0.56% by chance (z = 2.0) |
| `invention_cycle_warning` | 0.90 | synthetic, 0.1 |
| `leakage_r2` | 0.50 | synthetic seed 0, re-checked after moving to the lag-1 state |
| `rollout_gap` | 0.25 | synthetic, 0.1 |

**These 12 models are not a clean held-out test.** The recoverable-information
probe, the lag-1 leakage and the bottleneck rule were all designed *after*
0.1 failed on them. When 0.5% was read off the synthetic gap (0.40% to
0.97%), the real NODE values were already known (0.98%, 0.49% at the time,
0.31%, ...). No number was fitted to them, but the choice was not blind
either. NODE-5 sitting right at the line is the place this matters most. A
clean test needs models these detectors have never seen: new CtD tasks and
new seeds (see [ROADMAP.md](ROADMAP.md)).
