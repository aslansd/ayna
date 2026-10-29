# Validation

How far to trust each detector, and where it breaks. Raw outputs for every
table here are in [`validation/`](validation/), and [TESTING.md](TESTING.md)
gives the commands that regenerate them.

There are three kinds of evidence, and they answer different questions:

- **Synthetic corpus** (`ayna validate`). Do the detectors find faults that
  were deliberately injected? ayna's own simulator injects them and the same
  author wrote the detectors, so this tests design, not transfer.
- **Held-out trained models** (0.3, `ayna ctd`). Do the detectors hold on
  models, tasks and seeds they were not designed on? These are 15 models
  trained for this purpose (section 1).
- **CtD's pretrained models** (0.2, `ayna ctd`). Do the detectors agree with
  ground truth on models that were actually trained? These are CtD's 12
  pretrained 3BFF models, scored against the task-trained teacher. The 0.2
  detectors were redesigned on them, so they are not a held-out test
  (section 2).

## 1. Held-out models (0.3)

These are 15 models ayna's detectors had never seen. They were trained in
this project on a 2-core CPU from CtD's pretrained teachers
([experiments/ctd_tasks](experiments/ctd_tasks/README.md)), on three tasks:

- RandomTarget, closed-loop arm control;
- MultiTask, 15 cognitive subtasks;
- 3BFF, with its dataset regenerated and a new training seed.

**Ground rules:**

- No detector threshold was changed after these results were seen.
- One scoping bug was fixed: the tier-1 rollout probe judged supplied
  inputs (see below).
- The sweep-relative truth line was committed (`b4e48f4`) before the 3BFF,
  MultiTask and LFADS sweeps were scanned. RandomTarget NODE had been seen
  before the commit.

**Caveats:**

- **These are not CtD's models.** Batch size 25 and far fewer epochs than
  CtD's GPU runs make many of them undertrained, especially MultiTask at 25
  epochs. That makes them real underfitting cases, but they are not
  reproductions.
- **Each sweep has 3–6 models.** A rank correlation of ±1 over 3 models is
  easy to get by chance.

Two truth columns appear below:

- **absolute**: the synthetic lines (Rate R² < 0.90, State R² < 0.90,
  noiseless Input R² < 0.50);
- **relative**: more than 0.05 below the best model in the same sweep.

### RandomTarget NODE: supplied, closed-loop inputs (6 models)

| model | co-BPS | baseline | recoverable (z) | cycle-con | input filter | ayna flags | Rate R² | State R² | truth: absolute / relative |
|---|---|---|---|---|---|---|---|---|---|
| latent 3, seed 0 | 0.017 | 0.014 | 1.8% (1.8) | 1.00 | 0.93 | inherited | 0.725 | 0.962 | underfit, inherited / underfit |
| latent 3, seed 1 | 0.016 | 0.014 | 2.1% (2.0) | 1.00 | 0.94 | inherited | 0.729 | 0.964 | underfit, inherited / underfit |
| latent 10, seed 0 | 0.017 | 0.014 | 1.1% (1.8) | 1.00 | 0.86 | inherited | 0.864 | 0.883 | underfit, invention, inherited / invention |
| latent 10, seed 1 | 0.018 | 0.014 | −0.2% (−0.3) | 1.00 | 0.84 | inherited | 0.873 | 0.868 | underfit, invention, inherited / invention |
| latent 64, seed 0 | 0.014 | 0.014 | 0.6% (0.8) | **0.75** | 0.30 | invention, underfit | 0.653 | 0.315 | all three / underfit, invention |
| latent 64, seed 1 | 0.013 | 0.014 | 0.7% (1.2) | **0.75** | 0.29 | invention, underfit | 0.651 | 0.303 | all three / underfit, invention |

**What held:**

- **`INPUT_INHERITED_STRUCTURE` is correct on 4 of 6** in its first test on
  trained models. 94% of the teacher's latent variance is a 10-bin filter of
  its inputs. The two misses are the NODE-64s, whose latents are mostly
  invented, so the inputs explain less of them.
- **`FEATURE_INVENTION` catches both NODE-64s**, where State R² is 0.30–0.32.
  Rank agreement of cycle-consistency with State R² is +0.85.
- **`select` recommends NODE-10, seed 1.** It has the best Rate R² in the
  sweep.

**What failed:**

- **Underfitting from recoverable information.** RandomTarget's spike counts
  are tiny: co-BPS is about 0.017, against about 0.11 on 3BFF. So the
  per-trial test never reaches z = 3, even at a 2% fraction, and recoverable
  information does not rank this sweep (Spearman −0.09). The NODE-64s were
  caught only by the baseline comparison.
- **Invention the readout can see.** Under relative truth, both NODE-10s
  invent (State R² 0.87–0.88 against 0.96 for NODE-3), but their
  cycle-consistency is 1.00. Cycle-consistency only sees latent structure
  the rates ignore, the limit CtD's paper itself states.

### 3BFF NODE: regenerated dataset, seed 1 (3 models)

| model | co-BPS | recoverable (z) | cycle-con | ayna flags | Rate R² | State R² | truth: absolute / relative |
|---|---|---|---|---|---|---|---|
| latent 3 | 0.102 | 1.1% (17) | 1.00 | underfit | 0.827 | 0.951 | underfit / underfit |
| latent 8 | 0.112 | 0.3% (8) | 0.95 | — | 0.897 | 0.931 | underfit / — |
| latent 64 | 0.108 | 0.6% (11) | **0.70** | invention, underfit | 0.862 | 0.622 | underfit, invention / invention |

- **`select` recommends NODE-8, as on CtD's own sweep.**
- **Invention is right on all three.**
- **Recoverable information ranks the three exactly as Rate R² does.**
- **Disagreements:**
  - NODE-8 sits just under the absolute line (0.897 vs 0.90) and is not
    flagged;
  - NODE-64 is flagged underfit at 0.6%, but under relative truth it is
    within 0.05 of the best (0.862 vs 0.897).

### MultiTask NODE: 25 epochs (3 models)

| model | co-BPS | baseline | recoverable (z) | cycle-con | input filter | ayna flags | Rate R² | State R² | truth: absolute / relative |
|---|---|---|---|---|---|---|---|---|---|
| latent 5 | 0.068 | 0.072 | 17% (13) | 1.00 | 0.60 | underfit, inherited (info) | 0.377 | 0.863 | underfit, invention / underfit |
| latent 20 | 0.100 | 0.072 | 2.1% (9) | 0.95 | 0.42 | underfit, inherited (info) | 0.553 | 0.746 | underfit, invention / underfit, invention |
| latent 64 | 0.110 | 0.072 | 1.9% (9) | **0.81** | 0.35 | invention, underfit | 0.623 | 0.697 | underfit, invention / invention |

- **`UNDERFITTING` catches all three**, which is right: every model is badly
  undertrained.
- **`select` reports the reconstruction trap.** NODE-64 has the best co-BPS
  but cycle-consistency 0.81, against 0.95 for NODE-20.
- **NODE-20's invention is missed.** Its cycle-consistency of 0.95 is just
  above the line.
- **The two `INPUT_INHERITED_STRUCTURE` "flags" are info-level findings**
  (filter R² 0.42 and 0.60, below the 0.70 warning line). Validation counts
  info as flagged while the truth uses 0.70. That is a definitional
  mismatch, recorded for 0.4 and not changed here.

### RandomTarget LFADS: inferred inputs, controller KL sweep (3 models)

The latents are the generator state read through a 64-factor bottleneck,
so `FEATURE_INVENTION` is comparative only.

| KL scale | lag-1 leakage | rollout gap | ayna flags | Input R² (noiseless) | CtD Input R² | Rate R² |
|---|---|---|---|---|---|---|
| 0 | **0.88** | 0.22 | misattribution (tier 0), underfit | 0.779 | 0.718 | 0.836 |
| 1e-4 | **0.64** | −0.54 | misattribution (tier 0) | 0.462 | 0.424 | 0.825 |
| 1e-2 | **0.66** | −0.33 | misattribution (tier 0), underfit | 0.649 | 0.596 | 0.708 |

- **Tier-0 leakage flags all three models.** Precision is 0.33 against
  absolute truth and 0.67 against relative truth. Ground truth calls KL = 0
  clean (0.78).
- **The tier-1 rollout flags none of them.** It misses KL = 1e-4, the one
  model clearly misattributed (0.46).
- **`select` refuses to trust itself.** Every model leaks, so it says the
  recommendation (KL = 1e-4, the best co-BPS) "should not be trusted until
  one model passes". On this sweep that is the right answer from a tool that
  cannot tell closed-loop feedback from misattribution.

**The closed-loop question from 0.2's roadmap now has an answer.** On KL = 0,
lag-1 leakage is 0.88, yet that model infers its inputs *well* (noiseless
Input R² 0.78). RandomTarget's true inputs include sensory feedback that
depends on the network's own state, so they are predictable from the
preceding state whether or not the model misattributes. Tier-0 leakage is
not trustworthy in closed-loop tasks. The tier-1 rollout probe stayed
clean on that model. The 0.4 proposal is a declared `closed_loop` flag that
makes tier-0 leakage informational.

Unlike 3BFF, more KL made input inference *worse* here (0.78 → 0.46). The
3BFF finding "low KL ⇒ misattribution" does not carry over as a rule.


### Held-out scorecard

Counts are summed over the four held-out sweeps (15 models), shown as
TP / FP / FN / TN.

| detector | tier | absolute truth | relative truth | verdict |
|---|---|---|---|---|
| `FEATURE_INVENTION` | 0 | 4 / 0 / 4 / 4 | 4 / 0 / 3 / 5 | Never wrong when it fires. Misses invention that the readout can see. |
| `UNDERFITTING` | 0 | 9 / 0 / 6 / 0 | 6 / 3 / 2 / 4 | Reliable where spikes are plentiful (3BFF, MultiTask). Underpowered on RandomTarget. |
| `INPUT_INHERITED_STRUCTURE` | 0 | 4 / 2 / 2 / 4 | — | Correct on the closed-loop task it was built for. Info-level counting causes the false positives. |
| `DYNAMICAL_MISATTRIBUTION` | 0 | 1 / 2 / 0 / 0 | 2 / 1 / 0 / 0 | Not usable in closed-loop tasks without a flag. |
| `DYNAMICAL_MISATTRIBUTION` | 1 | 0 / 0 / 1 / 2 | 0 / 0 / 2 / 1 | Quiet, and misses the one misattributed model. |
| `select` | — | recommends the best-truth model on 3BFF and RandomTarget NODE | | reports the trap on MultiTask; declines to trust itself on RandomTarget LFADS |

Every number in this section is in [`validation/heldout/`](validation/heldout/).

## 2. CtD's pretrained 3BFF walkthrough models (0.2)

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

**Rescored in 0.3 against noiseless inputs.** 0.2's roadmap blamed input
noise for the low Input R² values here. It is only part of the cause:

- Scored against the teacher's noiseless inputs, the best model reaches only
  0.27 (KL = 1e-4), up from 0.21. KL = 1e-3 rises most (0.11 → 0.22).
- The binary table against the 0.5 line is unchanged: 3 TP, 3 FN.
- Against sweep-relative truth, the three low-KL models ayna flags are all
  correct. The two it leaves alone (KL = 1e-3, 1e-2) fall just over the line
  (0.215 and 0.004 against 0.223). That gives 3 TP, 2 FN and 1 TN.

These models were seen while designing 0.2, so this is not held-out
evidence.

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

## 3. Synthetic corpus

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
