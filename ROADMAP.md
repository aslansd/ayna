# Roadmap

ayna's question stays the same: *are this model's inferred dynamics real?*
Each release should make the answer more trustworthy, not just longer.

## Done

### 0.1 — the checklist
Bundle format. Numpy ports of the CtD metrics. Detectors for underfitting,
feature invention, dynamical misattribution, input-inherited structure and
nonlinear readout, plus `select` for sweeps. A synthetic corpus with injected
faults and `validate`. A daftar provenance extra.

### 0.2 — first contact with trained models
- Ran ayna on CtD's pretrained 3BFF models: 6 NODE autoencoders across latent
  sizes and 6 LFADS models across controller KL penalties, all against the
  task-trained teacher as ground truth. See [VALIDATION.md](VALIDATION.md).
- That run exposed three detectors that held up on synthetic data and failed
  on trained models. All three were replaced:
  - **Underfitting.** The residual-structure test fired on every NODE model.
    Its replacement measures information still recoverable from other
    neurons' spikes, against a trial-shuffled control, with a per-trial
    significance test.
  - **Input leakage.** Decoding inputs from the same-instant state is trivial
    when the state already contains the input. The probe now uses the
    preceding state.
  - **Cycle-consistency.** It is not comparable across architectures with a
    readout bottleneck such as LFADS factors. It is now reported as
    comparative only for those models.
- `select` no longer recommends models flagged for misattribution.
- `ayna ctd`: scan and validate a CtD sweep in place, one subprocess per run.
- Tier-1 dynamics wrappers for CtD's NODE and LFADS cells.
- Found and reproduced a co-BPS slicing bug in CtD's comparison code
  (reported as CtD #16, fixed upstream in PR #18).

### 0.3 — held-out evidence
- **Models ayna had never seen.** Trained on a 2-core CPU from CtD's
  pretrained teachers:
  - RandomTarget NODE (3 sizes × 2 seeds) and LFADS (3 controller-KL
    scales);
  - reseeded 3BFF NODE (3 sizes);
  - MultiTask NODE (3 sizes).

  No detector threshold was changed after seeing them. Results and every
  miss are in [VALIDATION.md](VALIDATION.md).
- **Sweep-relative truth, pre-registered** (git commit `b4e48f4`) before the
  3BFF, MultiTask and LFADS sweeps were scanned.
- **Noiseless Input R²** as the misattribution truth when the teacher
  provides it.
- **Exact-gradient fixed points.** Dynamics callables can offer `vjp` and
  `jacobian`, and the CtD wrappers do, so the search runs in any dimension.
- **A bug found by the held-out set.** The tier-1 rollout probe judged
  *supplied* inputs. It is now limited to inferred inputs.
- **CtD adapter:** prefix-shrunk datasets, task-specific teacher classes,
  CtD's trim window applied to the noiseless inputs.

### 0.4 — fixes for the 0.3 held-out failures, tested on fresh models

- **Pre-registered** (commit `1a75502`) before 12 fresh NODE models were
  trained: RandomTarget and 3BFF, 3 sizes, seeds 2 and 3. Two of the four
  criteria failed. [VALIDATION.md](VALIDATION.md) section 1 has the details.
- **Cross-fitted recoverable information.** RandomTarget NODE-3 went from
  z ≈ 1.8 to z ≈ 9.5.
- **`SEED_INCONSISTENCY`**: cross-seed affine latent consistency, the first
  check that sees invention the readout uses.
- **`closed_loop` declaration** for tier-0 input leakage.
- `INPUT_INHERITED_STRUCTURE` is counted at warning level only.
- `ayna ctd --resume`.

## Next: 0.5 — what the 0.4 test says to fix

As before, a fix tuned on these models needs another fresh set to be
believed.

- **A misattribution detector for closed-loop tasks.** With `closed_loop`
  declared, tier 0 is informational and tier 1 is quiet, so RandomTarget
  LFADS KL = 1e-4 (noiseless Input R² 0.46) goes unflagged. Candidates:
  - leakage measured against a model trained without inferred inputs on
    the same data, which separates "the feedback is predictable" from
    "the dynamics moved into the inputs";
  - a rollout that feeds the model's own predicted feedback back in.
- **Underfitting relative to the sweep.** The per-model test answers "is
  anything left in the spikes?". On 3BFF it says yes for every model, which
  is true, but five of the six are within 0.05 of the sweep's best Rate R².
  `select` should report how much more each model leaves than the best
  model, alongside the absolute test.
- **Underfitting the spikes cannot reveal.** RandomTarget NODE-64 seed 2
  has Rate R² 0.66 but only 0.1% recoverable information. Its errors are
  not predictable from other neurons. The baseline comparison catches
  seed 3 only by 0.0001 bits/spike.
- **Should `select` exclude seed-inconsistent models?** It recommended
  RandomTarget NODE-10 seed 2, which is flagged seed-inconsistent. Deciding
  that needs a test written before the data.
- **Seed consistency with more seeds.** Two seeds give one comparison per
  model. With three or more, the median would be more stable than the
  best, which matters for NODE-10 seed 3 at 0.909.
- **PhaseCodedMemory and ChaoticDelayedMatching.** CtD's simulator needs
  more than about 4 GB to generate their datasets even with chunked teacher
  calls, more than this machine had alongside training. On a bigger machine,
  `gen_data.py` should just work.
- **Real recordings.** An NWB / Neural Latents Benchmark reader, with
  MC_Maze as the worked example. The DANDI archive has not been reachable
  from the build environment, so this needs a local copy of the data.

## Later

- **`ayna design`** — given N neurons, T trials and a firing rate, can any
  model recover the dynamics at all? This is the miras question, asked of
  neural recordings.
- **Spiking ground truth.** CtD's teachers are rate networks. Spiking
  teachers built with jaxley (via tarjuman) would test whether the detectors
  survive when spikes come from membrane dynamics rather than a Poisson
  observation model.
- **Temporal metrics** from the dynamical-systems-reconstruction literature,
  such as power-spectrum agreement, as CtD's discussion proposes.

## Open questions

- **Noise-driven residuals.** Should a residual that no deterministic model
  can remove count as underfitting? On CtD's noisy teacher, every NODE size
  leaves about 0.2% recoverable information.
- **What should truth be on real-model sweeps?** Absolute lines from
  synthetic data do not transfer. 0.3 pre-registered a sweep-relative line
  (0.05 below the best model). Small sweeps make it brittle, because one
  model's value moves the line.
