"""Keep the first N train and M valid trials of a generated CtD dataset, in place.

MultiTask's dataset (7500 trials x 320 bins) plus its teacher does not fit in
8 GB of RAM in one process. CtD splits simulated trials in order (first 80%
train, last 20% valid), and the teacher analysis indexes them the same way, so
keeping a prefix of each split keeps data and teacher aligned. ayna's CtD
adapter then trims the teacher to the data-trained model's trial counts and
records that it did.

    python shrink_dataset.py PREFIX N_TRAIN N_VALID      # e.g. tt_MultiTask 1600 400
"""

import os
import sys
from pathlib import Path

import h5py

root = Path(os.environ["HOME_DIR"]) / "content" / "datasets" / "dd" / sys.argv[1]
n = {"train": int(sys.argv[2]), "valid": int(sys.argv[3])}
for path in root.rglob("*.h5"):
    tmp = path.with_suffix(".shrunk.h5")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for key in src:
            split = key.split("_")[0]
            data = src[key][: n[split]] if split in n else src[key][()]
            dst.create_dataset(key, data=data)
        dst.attrs.update(dict(src.attrs))
        dst.attrs["ayna_shrunk"] = f"first {n['train']} train / {n['valid']} valid trials"
    os.replace(tmp, path)
    print("shrunk", path)
