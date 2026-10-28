"""Train one CtD data-driven model on CPU, the way CtD's run_data_training.py does, minus Ray and WandB.

NODE sequential autoencoders with supplied inputs by default; set MODEL=LFADS
for LFADS with inferred inputs (then LATENT_SIZE is ignored and KL_CO sets the
controller-output KL scale).

Run inside the CtD environment with HOME_DIR set to the CtD repository and the
task's dataset generated (gen_data.py). Saves model.pkl + datamodule.pkl under
OUT/<task>/latent_size=<D>_seed=<S>_max_epochs=<E>/ -- a run folder `ayna ctd` reads.

    python train_node.py TASK LATENT_SIZE SEED MAX_EPOCHS OUT [BATCH_SIZE]

CtD trains with batch size 250 (about 3 updates per epoch on 800 training
trials) for 500-1000 epochs on a GPU. On a CPU budget the same number of
samples per epoch with batch size 25 gives 10x the updates; BATCH_SIZE is
recorded in the run folder name so the change is never hidden.
"""

import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
HOME = Path(os.environ["HOME_DIR"])
os.chdir(HOME / "ctd" / "data_modeling")       # CtD resolves config paths from here
from ctd.data_modeling.train_PTL import train_PTL  # noqa: E402

PREFIX = {"NBFF": "tt_3bff", "MultiTask": "tt_MultiTask", "RandomTarget": "tt_RandomTarget",
          "PhaseCodedMemory": "tt_PhaseCodedMemory", "ChaoticDelayedMatching": "tt_ChaoticDelayedMatching"}


def main(task, latent, seed, epochs, out, batch=None):
    cfg = Path("configs")
    lfads = os.environ.get("MODEL", "NODE") == "LFADS"
    if lfads:
        config_dict = {
            "model": cfg / f"models/LFADS/{task}/{task}_LFADS_infer.yaml",
            "datamodule": cfg / f"datamodules/LFADS/data_{task}_infer.yaml",
            "trainer": cfg / f"trainers/LFADS/trainer_{task}.yaml",
        }
        kl = float(os.environ.get("KL_CO", "1e-5"))
        overrides = {"model.kl_co_scale": kl, "trainer.max_epochs": int(epochs), "params.seed": int(seed)}
    else:
        config_dict = {
            "model": cfg / f"models/SAE/{task}/{task}_NODE.yaml",
            "datamodule": cfg / f"datamodules/SAE/data_{task}.yaml",
            "trainer": cfg / f"trainers/SAE/trainer_{task}.yaml",
        }
        overrides = {
            "model.latent_size": int(latent),
            "trainer.max_epochs": int(epochs),
            "datamodule.num_workers": 0,
            "params.seed": int(seed),
        }
    if batch is not None:
        overrides["datamodule.batch_size"] = int(batch)
    tmp = Path(out) / "_staging"
    t0 = time.time()
    train_PTL(overrides, config_dict, {"trained_models": str(tmp)}, run_tag=task)
    produced = next((tmp / task).iterdir())
    if lfads:
        tag = f"kl_co_scale={os.environ.get('KL_CO', '1e-5')}_seed={seed}_max_epochs={epochs}"
    else:
        tag = f"latent_size={latent}_seed={seed}_max_epochs={epochs}"
    tag += f"_batch={batch}" if batch else ""
    dest = Path(out) / (f"{task}_LFADS" if lfads else task) / tag
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"trained {task} {'LFADS' if lfads else 'NODE'} {tag} in {time.time() - t0:.0f}s -> {dest}")


if __name__ == "__main__":
    a = sys.argv[1:]
    # run_sweep.sh passes OUT then BATCH after the four positional values
    main(*a[:6])
