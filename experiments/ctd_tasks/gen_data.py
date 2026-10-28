"""Simulate CtD's spiking datasets from its pretrained task-trained teachers.

Run inside the CtD environment with HOME_DIR set to the CtD repository.
Writes HOME_DIR/content/datasets/dd/<prefix>/<subfolder>/*.h5 and copies each
teacher to HOME_DIR/content/trained_models/task-trained/<prefix>/ -- the
layout CtD's data-training datamodules expect. Same subfolders and seed as
CtD's examples/gen_datasets.py, one task per call so memory stays bounded.

    python gen_data.py NBFF|MultiTask|RandomTarget|PhaseCodedMemory|ChaoticDelayedMatching
"""

import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch  # noqa: E402

torch.set_grad_enabled(False)
from ctd.comparison.analysis.tt.tt import Analysis_TT  # noqa: E402

HOME = Path(os.environ["HOME_DIR"])
TASKS = {
    "NBFF": ("tt_3bff", "20241017_NBFF_NoisyGRU_NewFinal",
             "max_epochs=500 n_samples=1000 latent_size=64 seed=0 learning_rate=0.001"),
    "MultiTask": ("tt_MultiTask", "20241113_MultiTask_NoisyGRU_Final2", "max_epochs=500 seed=0"),
    "RandomTarget": ("tt_RandomTarget", "20241113_RandomTarget_NoisyGRU_Final2",
                     "max_epochs=2000 latent_size=128 l2_wt=5e-05 proprioception_delay=0.02 "
                     "vision_delay=0.05 n_samples=1100 n_samples=1100 seed=0 learning_rate=0.005"),
    "PhaseCodedMemory": ("tt_PhaseCodedMemory", "PCM_NoisyGRU_Final",
                         "max_epochs=50_learning_rate=1.00E-03_weight_decay=1.00E-08_"
                         "latent_size=512_post_stim_l2_weight=20"),
    "ChaoticDelayedMatching": ("tt_ChaoticDelayedMatching", "20260320_ChaoticDelayedMatching_Final",
                               "max_epochs=100_recurrent_gain=2.20E+00_batch_size=256_seed=0"),
}


def teacher_class(task):
    if task == "MultiTask":
        from ctd.comparison.analysis.tt.tasks.tt_MultiTask import Analysis_TT_MultiTask
        return Analysis_TT_MultiTask
    if task == "RandomTarget":
        from ctd.comparison.analysis.tt.tasks.tt_RandomTarget import Analysis_TT_RandomTarget
        return Analysis_TT_RandomTarget
    return Analysis_TT


class ChunkedTeacher:
    """Run the teacher over trials in chunks and concatenate its outputs.

    CtD's simulator calls the teacher once on every trial; for PhaseCodedMemory
    (512 latents) and ChaoticDelayedMatching that needs more than 6 GB. Outputs
    are identical, only the memory peak changes.
    """

    def __init__(self, model, chunk=64):
        self._model, self._chunk = model, chunk

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, ics, inputs, inputs_to_env=None, **kw):
        outs = []
        for i in range(0, ics.shape[0], self._chunk):
            sl = slice(i, i + self._chunk)
            ite = None if inputs_to_env is None else inputs_to_env[sl]
            outs.append(self._model(ics[sl], inputs[sl], inputs_to_env=ite, **kw))
        merged = {}
        for key, val in outs[0].items():
            if torch.is_tensor(val) and val.dim() > 0 and val.shape[0] == min(self._chunk, ics.shape[0]):
                merged[key] = torch.cat([o[key] for o in outs], dim=0)
            else:
                merged[key] = val
        return merged


def main(task):
    prefix, folder, subfolder = TASKS[task]
    src = HOME / "pretrained" / folder
    dst = HOME / "content" / "trained_models" / "task-trained" / prefix
    if not dst.exists():
        shutil.copytree(src, dst)
    torch.manual_seed(0)
    tt = teacher_class(task)(run_name=prefix, filepath=str(dst) + os.sep)
    tt.wrapper = ChunkedTeacher(tt.wrapper)
    tt.simulate_neural_data(subfolder=subfolder, dataset_path=HOME / "content" / "datasets" / "dd")
    print("simulated", task, "->", HOME / "content" / "datasets" / "dd" / prefix)


if __name__ == "__main__":
    main(sys.argv[1])
