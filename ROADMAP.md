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

## Next: 0.3 — breadth of evidence

The 0.2 real-model evidence comes from one task (3BFF), two model classes, one
seed each and twelve models in total. That is enough to catch broken
detectors, and not enough to trust thresholds.

- **More CtD tasks.** CtD ships pretrained task-trained teachers for
  MultiTask, RandomTarget, PhaseCodedMemory and ChaoticDelayedMatching, but no
  data-trained sweeps on them. Generate small NODE and LFADS sweeps on each.
  They need a GPU to train, but ayna scans them on a CPU. Then run
  leave-one-task-out tests of every threshold. RandomTarget is the real test
  for `INPUT_INHERITED_STRUCTURE`.
- **Several seeds per configuration.** Run-to-run spread is the natural unit
  for "significantly worse", in `select` and in the truth lines.
- **Truth lines for real models.** On CtD's LFADS models every Rate R² sits
  between 0.86 and 0.90 and every Input R² below 0.2, because the true inputs
  are noisy. The synthetic truth lines (0.90 and 0.50) do not transfer. Options:
  score Input R² against noiseless inputs, or define truth relative to the
  best model in a sweep. For now, VALIDATION.md reports rank agreement next to
  every binary table.
- **Real recordings.** A bundle reader for NWB / Neural Latents Benchmark
  data, with a worked example on MC_Maze. There is no ground truth there; the
  claim is only "stable, and quiet when unsure".
- **Fixed points in high dimension.** The finite-difference search is skipped
  above 32 latent dimensions. Use autograd when the dynamics are torch.

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

- **Closed-loop tasks.** In RandomTarget, sensory inputs depend on the
  model's own output, so true inputs *are* predictable from the preceding
  state. How should input leakage be read there?
- **Noise-driven residuals.** Should a residual that no deterministic model
  can remove count as underfitting? On CtD's noisy teacher, every NODE size
  leaves about 0.2% recoverable information.
