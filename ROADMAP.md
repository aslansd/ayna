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
- Found and reproduced a co-BPS slicing bug in CtD's comparison code.

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

## Next: 0.4 — what the held-out set says to fix

Each item below is a failure seen in 0.3. A fix tuned on these same models
needs another fresh set to be believed.

- **Underfitting at low spike counts.** On RandomTarget, recoverable
  information was large (1–2%), but the per-trial test never reached z = 3
  (z ≈ 1.8). Options: pool across neurons before testing, or use the
  baseline comparison alone when spikes are sparse.
- **Invention the readout can see.** Cycle-consistency only detects latent
  structure the rates ignore. RandomTarget NODE-10 and MultiTask NODE-20
  have State R² 0.75–0.88 with cycle-consistency ≥ 0.95. Candidates: seed
  consistency, which needs seeds, and comparison against a simpler model's
  latents.
- **Info-level findings count as "flagged" in validation.** For
  `INPUT_INHERITED_STRUCTURE`, that produced two MultiTask false positives
  at filter R² 0.42–0.60, below the 0.70 warning line that the truth uses.
- **Closed-loop tasks and input leakage.** The 0.2 open question is now
  answered. On RandomTarget LFADS, KL = 0, lag-1 leakage is 0.88 and flags
  misattribution, yet the inputs are well inferred (Input R² 0.78 against
  noiseless inputs). True sensory feedback depends on the state. The tier-1
  rollout stayed clean. Proposal: a declared `meta["closed_loop"]` that
  makes tier-0 leakage informational and lets tier 1 decide.
- **More seeds.** One or two per configuration is not enough for
  run-to-run spread.
- **PhaseCodedMemory and ChaoticDelayedMatching.** CtD's simulator needs
  more than about 4 GB to generate their datasets even with chunked teacher
  calls, more than this machine had alongside training. On a bigger machine,
  `gen_data.py` should just work.
- **Real recordings.** An NWB / Neural Latents Benchmark reader, with
  MC_Maze as the worked example. The DANDI archive was not reachable from
  the 0.3 build environment, so this needs a local copy of the data.

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
