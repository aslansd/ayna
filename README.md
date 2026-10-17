# ayna

**آینه / ayna** — *mirror.* Persian *āyeneh*; *ayna* in Turkish and Azerbaijani.

**Know whether your inferred dynamics are real, before you interpret them.**

ayna reads the outputs of a fitted data-driven dynamics model (LFADS, a
sequential autoencoder, a neural ODE, anything that turns spikes into latents
and rates) and reports *which failure mode it has, and where*. It does not
train anything. It runs on a CPU in seconds.

```bash
pip install ayna
```

Python 3.10+. Apache 2.0. Three dependencies: numpy, scipy, scikit-learn. No
torch, no GPU, no downloads.

[Install](INSTALL.md) · [Testing](TESTING.md) · [Validation](VALIDATION.md) ·
[Roadmap](ROADMAP.md) · [Publishing](PUBLISHING.md) · [Changelog](CHANGELOG.md)

---

## The failure that matters

A model can reconstruct held-out neurons almost perfectly and still give you
the wrong dynamics. The Computation-through-Dynamics Toolkit (CtD; Versteeg et
al., *PLOS Comput Biol* 2026) names three ways this happens:

- **Underfitting** — the model misses latent structure that is really there.
- **Feature invention** — the model adds latent structure that is not there.
  The readout ignores it, so it costs nothing in reconstruction, and models
  that invent features often *win* on co-BPS.
- **Dynamical misattribution** — the inferred inputs do the work the dynamics
  should do. Latents and rates look right. Only the inputs, and the dynamics
  left behind, are wrong.

CtD measures all three when ground truth is known. On a real recording there
is no ground truth, and reconstruction alone points the wrong way. ayna runs
every check your outputs allow, says which ones it could not run, and gets
sharper when you give it more.

## What it found on CtD's own trained models

ayna 0.2 was run on the 12 pretrained 3BFF models that ship with CtD, using
no ground truth for its verdicts. It was then scored against the
task-trained teacher.

- **NODE latent-size sweep.** Every verdict agrees with ground truth (12 of
  12):
  - NODE-3 and NODE-5 are flagged `UNDERFITTING`;
  - NODE-16, -32 and -64 are flagged `FEATURE_INVENTION`;
  - NODE-8 is clean.

  `select` recommends NODE-8 and reports that the best-co-BPS model (NODE-64)
  is a trap. Ground truth agrees: State R² 0.56 vs 0.96.
- **LFADS controller-KL sweep.** ayna flags `DYNAMICAL_MISATTRIBUTION` on the
  three lowest-KL models, which CtD's paper describes as offloading dynamics
  to inputs. `select` recommends KL = 1e-4, the model with the best Input R².
  Ranked by reconstruction alone, it would have picked a misattributed model.
- **Where it falls short.** On LFADS, every model sits close to the synthetic
  truth lines, so the binary tables disagree even where the ranks agree
  (Spearman −0.94 to −1.00). One seed, one task: this is evidence that the
  detectors are not broken, not a calibration.
- **Along the way:** CtD's own co-BPS code includes 47 held-in neurons in its
  "held-out" set. Details are in [VALIDATION.md](VALIDATION.md).

---

## Quickstart

```bash
# 1. A model output with known, deliberately injected faults
ayna synth field --faults feature_invention misattribution=0.8

# 2. Diagnose it -- no ground truth used for the verdicts
ayna scan datasets/field.npz --out reports/field

# 3. A latent-size sweep: is the best co-BPS model the one to trust?
ayna synth sweep --sweep
ayna select datasets/sweep --out reports/sweep --plot

# 4. How often are the detectors right? (compares against ground truth)
ayna validate
```

From Python:

```python
from ayna import load_bundle, scan

report = scan(load_bundle("run.npz"))
for f in report["findings"]:
    if f["tier"] < 2:                 # tier 2 = ground-truth confirmations
        print(f["severity"], f["id"], f["where"])
```

`scan` exits with status 1 when a finding from the model's outputs is an
error, so it can gate a pipeline.

### On a CtD sweep, in place

Inside the CtD environment (see [INSTALL.md](INSTALL.md)):

```bash
W=$HOME_DIR/examples/walkthrough_models
ayna ctd $W/node_sweep --model-type SAE --tt $W/tt_3bff --out reports/node
```

Each run is loaded in its own subprocess, scanned with its real dynamics
attached (tier 1), compared across the sweep, and validated against the
teacher.

---

## Three tiers of evidence

| Tier | You provide | Unlocks |
|---|---|---|
| **0** | spikes, predicted rates, latents (+ held-out neurons, + inferred or supplied inputs) | every detector marked tier 0 |
| **1** | a dynamics callable `f(z, u) -> z_next` (Python API, or automatic with `ayna ctd`) | autonomous-rollout and zero-input fixed-point probes |
| **2** | true latents / rates / inputs (synthetic data only) | CtD's Rate R², State R², Input R², and validation |

## The detectors

| Finding | Question it answers | Tier |
|---|---|---|
| `UNDERFITTING` | Do other neurons' spikes still improve the model's predictions (recoverable information, net of a trial-shuffled control, significant across trials)? Does co-BPS fail to beat a smoothed-spikes GLM? | 0 |
| `FEATURE_INVENTION` | Can the latents be regenerated from the model's own rates (cycle-consistency)? Reports *which latent PCs* cannot. Comparative only for models with a readout bottleneck (LFADS factors). | 0 |
| `READOUT_NULLSPACE` | Are there latent directions the readout ignores, where invention could hide later? | 0 |
| `NONLINEAR_READOUT` | Is the latent→log-rate map far from linear? If so, a nonlinear cycle-consistency replaces the linear one. | 0 |
| `DYNAMICAL_MISATTRIBUTION` *(experimental)* | Does the state *before* each inferred input predict it (lag-1 leakage)? With inputs clamped to zero, do the model's own dynamics lose a state that simply holding would keep? | 0 / 1 |
| `INPUT_INHERITED_STRUCTURE` | With supplied inputs, how much of the latent variance is just a short filter of the inputs? | 0 |
| `RECONSTRUCTION_TRAP` | In a sweep, is the best-reconstructing model off the reconstruction × simplicity knee? (`ayna select`) | 0 |
| `MISATTRIBUTION_EXCLUDED` | In a sweep, which models must not be recommended however well they reconstruct? (`ayna select`) | 0 |

The misattribution probes are ayna's own. CtD lists a ground-truth-free
input-accuracy metric as future work. In closed-loop tasks, where sensory
feedback depends on the model's output, true inputs can legitimately depend
on the preceding state, so read that finding with the task in mind.

---

## The bundle

One `.npz` per fitted model. Rates are expected counts **per bin**.

| key | shape | |
|---|---|---|
| `spikes` | (R, T, N) | required |
| `rates` | (R, T, N) | required: predicted, held-in and held-out |
| `latents` | (R, T, D) | required |
| `heldout` | (N,) bool | neurons never given to the model (enables co-BPS) |
| `val` | (R,) bool | validation trials (default: last 20%) |
| `trial_lens` | (R,) int | for padded trials |
| `inputs` | (R, T, U) | with `meta["inputs_kind"]` = `supplied` or `inferred` |
| `true_latents`, `true_rates`, `true_inputs` | | ground truth (tier 2) |
| `meta` | JSON string | name, model, bin size, `readout_bottleneck`, anything |

```python
from ayna import Bundle
Bundle(spikes=y, rates=r, latents=z, heldout=ho, inputs=u_hat,
       meta={"inputs_kind": "inferred", "name": "lfads_run7", "readout_bottleneck": 8}).save("run7.npz")

report = scan(load_bundle("run7.npz", dynamics=lambda z, u: my_model.step(z, u)))   # tier 1
```

## Provenance

```bash
pip install "ayna[provenance]"
ayna scan run.npz --out reports/run --daftar
```

Each scan becomes a [daftar](https://pypi.org/project/daftar/) run that
records the bundle hash, the thresholds, the environment and the headline
metrics.

## Thresholds

Every threshold has a name and can be overridden
(`ayna scan run.npz --set leakage_r2=0.6`). The defaults were set on ayna's
synthetic calibration corpus. [VALIDATION.md](VALIDATION.md) records how each
one was chosen and how it behaved on trained models. Before trusting a
single scan's verdict, compare models of the same class on the same data
with `ayna select`.

## What ayna does not do

- It does not train, fit or tune a model.
- It cannot prove a model's dynamics are right. It can only rule out specific
  ways they are wrong.
- No DSA or Lyapunov comparison yet.
- Its linear metrics assume the linear-exponential observation model.
  `NONLINEAR_READOUT` tells you when that fails.

## Citation

ayna ports metrics and failure-mode definitions from:

> Versteeg C, McCart JD, Ostrow M, et al. (2026) Computation-through-Dynamics
> Toolkit: Simulated datasets and quality metrics for dynamical models of neural
> activity. *PLOS Comput Biol* 22(10): e1014755.
> https://doi.org/10.1371/journal.pcbi.1014755

If you use ayna, please cite that paper.

## Related

Part of a small family of pre-flight tools by the same author:
[daftar](https://pypi.org/project/daftar/) (provenance),
[raftaar](https://pypi.org/project/raftaar/) (robot demonstration datasets),
[miras](https://pypi.org/project/miras/) (identifiability of cultural-evolution
designs), [tarjuman](https://pypi.org/project/tarjuman/) (NeuroML → Jaxley) and
[baghban](https://pypi.org/project/baghban/) (Bonsai workflows).

Research prototype, alpha. Apache 2.0.
