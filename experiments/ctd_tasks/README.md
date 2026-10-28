# Held-out evidence: CtD tasks ayna has never seen

These scripts produce the models behind the "held-out" section of
[VALIDATION.md](../../VALIDATION.md). They are not part of the ayna package.
They run in the CtD environment ([INSTALL.md](../../INSTALL.md)) and train
models, which ayna itself never does.

The question is whether the 0.2 detectors, designed on CtD's 3BFF
walkthrough models, hold on models they were not designed on:

- different tasks: MultiTask, RandomTarget, and 3BFF data generated afresh;
- different seeds;
- models trained here, under a CPU budget.

No threshold is changed after looking at these results. If a detector fails
here, it fails in VALIDATION.md.

## Steps

```bash
export HOME_DIR=/path/to/ComputationThroughDynamicsToolkit/

# 1. teachers: pull pretrained/ with git lfs (about 1.5 GB), then simulate each task's spikes
python gen_data.py NBFF
python gen_data.py MultiTask
python gen_data.py RandomTarget

# 2. NODE autoencoders on CPU: TASK LATENT_SIZE SEED MAX_EPOCHS OUT [BATCH_SIZE]
python train_node.py RandomTarget 3 0 100 runs 25

# 3. scan, compare and validate against each task's teacher
ayna ctd runs/RandomTarget --model-type SAE \
    --tt $HOME_DIR/content/trained_models/task-trained/tt_RandomTarget --out reports/RandomTarget
```

`run_sweep.sh` runs the whole grid used in VALIDATION.md.

## Differences from CtD's own training

These choices fit a 2-core CPU and 8 GB of RAM. They are recorded in every
run folder's name:

- **Batch size 25 instead of 250.** The samples per epoch are the same, but
  there are 10x more gradient steps. With 250, 60 epochs is about 200
  updates and leaves a NODE-8 badly underfit (co-BPS 0.03 vs 0.11 fully
  trained).
- **Fewer epochs.** See `run_sweep.sh`. CtD uses 200-1000 on a GPU.
- **No Ray, no WandB, no plotting callbacks.** The script calls CtD's
  `train_PTL` directly with the same model, datamodule and trainer configs.
- **Chunked teacher calls in `gen_data.py`.** These run the teacher over
  64 trials at a time so simulation fits in memory. The outputs are
  identical to CtD's.

PhaseCodedMemory and ChaoticDelayedMatching are not included: see
VALIDATION.md for why.
