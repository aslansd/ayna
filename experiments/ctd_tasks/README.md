# Held-out evidence: models ayna had never seen

These scripts produce the models behind the "held-out" section of
[VALIDATION.md](../../VALIDATION.md). They are not part of the ayna package.
They run in the CtD environment ([INSTALL.md](../../INSTALL.md)) and train
models, which ayna itself never does.

The question: do the 0.2 detectors, designed on CtD's 3BFF walkthrough
models, hold on models they were not designed on? That means other tasks,
other seeds, and models trained under a CPU budget. No detector threshold
was changed after these results were seen. The sweep-relative truth used
alongside the absolute one was committed (`b4e48f4`) before the 3BFF,
MultiTask and LFADS sweeps were scanned.

## What was run

| sweep | model | sizes / settings | seeds | epochs | wall time each* |
|---|---|---|---|---|---|
| RandomTarget | NODE, supplied inputs | latent 3, 10, 64 | 0, 1 | 150 | ~24 min |
| 3BFF (data regenerated from the pretrained teacher) | NODE, supplied inputs | latent 3, 8, 64 | 1 | 100 | ~40 min |
| MultiTask (first 1600 / 400 trials) | NODE, supplied inputs | latent 5, 20, 64 | 0 | 25 | ~13 min |
| RandomTarget | LFADS, inferred inputs | controller KL 0, 1e-4, 1e-2 | 0 | 150 | ~85 min |

\* On 2 CPU cores, batch size 25.

## Steps

```bash
export HOME_DIR=/path/to/ComputationThroughDynamicsToolkit/
# pull pretrained/ with git lfs first (about 1.5 GB)

python gen_data.py NBFF            # simulate each task's spikes from its teacher
python gen_data.py MultiTask
python gen_data.py RandomTarget
python shrink_dataset.py tt_MultiTask 1600 400

./run_sweep.sh runs                # NODE grid (about 5 h)
./run_lfads.sh runs                # LFADS grid (about 4.5 h)

T=$HOME_DIR/content/trained_models/task-trained
ayna ctd runs/RandomTarget       --model-type SAE   --tt $T/tt_RandomTarget --out reports/rt
ayna ctd runs/RandomTarget_LFADS --model-type LFADS --tt $T/tt_RandomTarget --out reports/rt_lfads
ayna ctd runs/NBFF               --model-type SAE   --tt $T/tt_3bff         --out reports/nbff
ayna ctd runs/MultiTask          --model-type SAE   --tt $T/tt_MultiTask    --out reports/mt
```

## Differences from CtD's own training

These choices fit a 2-core CPU with 8 GB of RAM. Each is recorded in the run
folder names or below:

- **Batch size 25 instead of 250.** The samples per epoch are the same, but
  there are 10x the gradient steps. With 250, 60 epochs is about 200
  updates and leaves a NODE-8 badly underfit (co-BPS 0.03 vs 0.11 fully
  trained). With 25 it reaches 0.107.
- **Far fewer epochs than CtD** (200–1000 on a GPU). Many of these models
  are undertrained, especially MultiTask at 25 epochs. That makes them
  genuine underfitting cases. It does not make them reproductions of CtD's
  models.
- **MultiTask shrunk to its first 1600 train / 400 valid trials**
  (`shrink_dataset.py`), because the full 7500 trials and the teacher do not
  fit in 8 GB together. CtD splits trials in order, so a prefix stays aligned
  with the teacher, and ayna's adapter trims the teacher to match and records
  that in the bundle notes.
- **Chunked teacher calls in `gen_data.py`.** Outputs are identical to CtD's
  `gen_datasets.py`.
- **No Ray, no WandB, no plotting callbacks.** `train_dd.py` calls CtD's
  `train_PTL` directly with the same model, datamodule and trainer configs.
- **A PyTorch compatibility patch.** CtD's LFADS passes `verbose=` to
  `ReduceLROnPlateau`, which PyTorch ≥ 2.7 rejects. `train_dd.py` drops the
  argument. The first LFADS runs used a subclass for this, which leaked into
  their pickles; `fix_pickles.py` repairs them.

## Not included

**PhaseCodedMemory and ChaoticDelayedMatching.** CtD's simulator needed
more than about 4 GB to generate their datasets even with chunked teacher
calls, which is more than this machine had alongside training. On a bigger
machine, `python gen_data.py PhaseCodedMemory` and the same training steps
should work unchanged.
