"""Re-save LFADS runs trained with an early version of train_dd.py.

That version subclassed ReduceLROnPlateau as ``__main__._PlateauCompat``, so the
pickled model referenced a class that only existed in the training process.
This loads the pickle with that name mapped to torch's own ReduceLROnPlateau
and writes it again, so it loads anywhere.

    python fix_pickles.py RUN_DIR [RUN_DIR ...]
"""

import pickle
import sys

import torch


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "_PlateauCompat":
            self.mapped = getattr(self, "mapped", 0) + 1
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        return super().find_class(module, name)


for run in sys.argv[1:]:
    for name in ("model.pkl", "datamodule.pkl"):   # the datamodule keeps a trainer reference
        path = f"{run.rstrip('/')}/{name}"
        with open(path, "rb") as fh:
            if b"_PlateauCompat" not in fh.read():
                continue
        with open(path, "rb") as fh:
            up = _Unpickler(fh)
            obj = up.load()
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"{path}: {getattr(up, 'mapped', 0)} reference(s) mapped to torch's ReduceLROnPlateau")
