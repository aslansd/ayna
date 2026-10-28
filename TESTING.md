# Testing ayna

There are three layers. Each catches a different kind of mistake.

| layer | what it is | needs | catches |
|---|---|---|---|
| unit and integration tests | `tests/test_ayna.py` | ayna + pytest | broken code, broken contracts |
| synthetic validation | `ayna validate` | ayna | detectors that stopped finding injected faults |
| live CtD tests and validation | `tests/test_ctd_live.py`, `ayna ctd` | CtD environment + pretrained models | adapter breakage, detectors that fail on *trained* models |

The third layer is the only one that exercises real trained models. In 0.2
it found three detectors that passed the first two layers and still failed on
real models (see [ROADMAP.md](ROADMAP.md)). Run it whenever a detector or
threshold changes.

## 1. The core suite

```bash
pip install -e ".[dev]"
pytest -q
```

34 tests, under a minute to a few minutes on a laptop CPU. They need no
optional dependency and must all pass in any environment. `test_ctd_live.py`
is collected too and shows as 1 skipped module when CtD is not importable.
That is expected.

What the core suite pins down:

- **Bundle contract.** Round-trip through `.npz`, shape and value
  validation, the default validation split, padded trials, and the note
  recorded when rates are floored.
- **Metric ports.** Bits/spike sign and null, cycle-consistency finding a
  readout-silent dimension, and the regression direction of State R²
  (true → inferred) versus recovery (inferred → true).
- **Detectors.** A clean model stays clean. Each injected fault raises its
  own finding and no other. The "where" field names the invented PCs. Tier-1
  probes run when dynamics are attached, and a crashing callable is
  reported, never raised. Checks that cannot run are listed as not
  assessed.
- **0.2 behaviour learned on real models:**
  - recoverable information separates underfit from clean, and does not
    fire on small clean recordings, thanks to the z-test;
  - the neuron-split fallback works without held-out neurons;
  - leakage uses the *preceding* state, so latents that contain the
    current input are not mistaken for misattribution;
  - a readout bottleneck makes cycle-consistency comparative only;
  - `select` never recommends a leaky model;
  - `select` labels show the part of the name that varies.
- **0.3 behaviour learned on held-out models:**
  - fixed points use exact gradients (`vjp`, `jacobian`) when the dynamics
    offer them, so the search runs above 32 dimensions;
  - the misattribution truth uses noiseless inputs when the bundle has them;
  - the rollout probe never judges *supplied* inputs. Found on RandomTarget,
    where the state legitimately depends on sensory feedback.
- **CtD adapter, without CtD.** Duck-typed fakes cover the getters, held-out
  ordering, the train/val split, all-zero trial lengths, getters that
  raise, `eval()` being called, the LFADS bottleneck, the error for Git
  LFS pointer stubs, trimming the teacher to a prefix-shrunk dataset,
  dropping misaligned ground truth, and applying CtD's trim window to the
  noiseless inputs.
- **CLI.** `synth` → `scan` with JSON output and the exit code for CI.

## 2. Synthetic validation

```bash
ayna validate --n 10 --seed 7 --out reports/val-seed7     # test corpus
ayna validate --n 6  --seed 0 --out reports/val-seed0     # calibration corpus
```

Seed 0 is the corpus thresholds were set on. Seed 7 was not used to set any
threshold. Quote seed 7. The output has a confusion table per detector and
tier, rank agreement between each ground-truth-free metric and its ground
truth, and the list of individual disagreements. The disagreements are the
useful part.

## 3. Live CtD tests and validation

Set up the CtD environment and pull the walkthrough models first
([INSTALL.md](INSTALL.md)). Then, inside that environment:

```bash
export HOME_DIR=/path/to/ComputationThroughDynamicsToolkit/
export AYNA_CTD_WALKTHROUGH=$HOME_DIR/examples/walkthrough_models
pytest tests/test_ctd_live.py -v
```

Three tests, about 5 minutes, about 6 GB RAM at peak:

- **NODE sweep ends.** NODE-3 is flagged `UNDERFITTING`, NODE-8 is clean,
  NODE-64 is flagged `FEATURE_INVENTION` (ground truth State R² < 0.7).
  `select` recommends NODE-8 and reports the reconstruction trap.
- **LFADS.** KL = 0 is flagged `DYNAMICAL_MISATTRIBUTION` and KL = 1e-4 is
  not. Ground truth agrees on Input R². `select` recommends KL = 1e-4.
- **co-BPS parity.** ayna's bits/spike equals CtD's `get_bps` on the true
  held-out neurons to within 1e-4.

For the full tables quoted in [VALIDATION.md](VALIDATION.md):

```bash
W=$AYNA_CTD_WALKTHROUGH
ayna ctd $W/node_sweep     --model-type SAE   --tt $W/tt_3bff --out reports/node
ayna ctd $W/inputinf_sweep --model-type LFADS --tt $W/tt_3bff --out reports/lfads
```

Each writes per-model `report.{txt,md,json}`, `select.{txt,json}`,
`validation.{txt,json}` and `summary.txt`. Exports are seeded, so
repeated runs give identical numbers.

## 4. Held-out models (0.3)

The models behind VALIDATION.md's held-out section are trained, not
downloaded. See [experiments/ctd_tasks/README.md](experiments/ctd_tasks/README.md).
Training takes about 7 hours on a 2-core CPU. Scanning afterwards takes a few
minutes per task:

```bash
T=$HOME_DIR/content/trained_models/task-trained
ayna ctd runs/RandomTarget       --model-type SAE   --tt $T/tt_RandomTarget --out reports/rt
ayna ctd runs/RandomTarget_LFADS --model-type LFADS --tt $T/tt_RandomTarget --out reports/rt_lfads
ayna ctd runs/NBFF               --model-type SAE   --tt $T/tt_3bff         --out reports/nbff
ayna ctd runs/MultiTask          --model-type SAE   --tt $T/tt_MultiTask    --out reports/mt
```

## Before a release

1. `pytest -q` is green.
2. `pytest tests/test_ctd_live.py` is green in the CtD environment.
3. If detectors or thresholds changed, regenerate the validation tables and
   update README.md and VALIDATION.md.
4. Build the wheel, install it in a fresh virtualenv, and run the README
   quickstart there (see [PUBLISHING.md](PUBLISHING.md)).
