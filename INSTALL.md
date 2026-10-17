# Installing ayna

## The package

```bash
pip install ayna
```

Python 3.10 or newer. The core needs three packages: numpy, scipy and
scikit-learn. It never imports torch.

Optional extras:

| extra | adds | for |
|---|---|---|
| `ayna[plot]` | matplotlib | `ayna select --plot` |
| `ayna[provenance]` | daftar | `ayna scan --daftar` |
| `ayna[dev]` | pytest, matplotlib | running the test suite |
| `ayna[all]` | matplotlib, daftar | everything except development tools |

Check what is available with:

```bash
ayna doctor
```

### From source

```bash
git clone https://github.com/aslansd/ayna.git
cd ayna
pip install -e ".[dev]"
pytest -q
```

## Working with the Computation-through-Dynamics Toolkit

Most of ayna needs nothing from CtD: give it a bundle (`.npz`) and it runs.
Two commands, `ayna ctd` and `ayna export-ctd`, read CtD's own saved runs.
Those two run **inside an environment where CtD can unpickle its models**.
That environment is heavier than ayna's, so keep it separate.

These steps were tested on Linux (Python 3.13, CPU only, 8 GB RAM) with
CtD 1.0.2 and its pretrained 3BFF walkthrough models.

### 1. Get CtD and its pretrained models

The pretrained models are stored with Git LFS. Without it, the `.pkl` files
are 130-byte pointer stubs. ayna detects stubs and says so instead of failing
inside `pickle`.

```bash
git lfs install                      # once per machine
git clone https://github.com/snel-repo/ComputationThroughDynamicsToolkit.git
cd ComputationThroughDynamicsToolkit

# only what ayna's validation uses (about 5.6 GB):
git lfs pull --include="examples/walkthrough_models/tt_3bff/**"
git lfs pull --include="examples/walkthrough_models/node_sweep/**"
git lfs pull --include="examples/walkthrough_models/inputinf_sweep/**"
```

If `git lfs pull` is not possible (for example behind a proxy that blocks
LFS), the same files are served over plain HTTPS at
`https://media.githubusercontent.com/media/snel-repo/ComputationThroughDynamicsToolkit/main/<path>`.
Each pointer stub contains the file's `sha256`, so check every download
against it.

### 2. Make an environment that can load the pickles

CtD's datamodule pickles reference `ray` and `wandb` classes, so those must be
importable even though nothing is trained or logged. A minimal working set:

```bash
python -m venv ctd-env && source ctd-env/bin/activate
pip install torch                                     # CPU is enough
pip install pytorch-lightning torchmetrics h5py gymnasium hydra-core omegaconf \
            python-dotenv matplotlib scikit-learn scipy plotly geotorch "ray[tune]" wandb
pip install --no-deps motornet==0.2.0                 # imported by CtD's task code
pip install --no-deps -e /path/to/ComputationThroughDynamicsToolkit
pip install ayna
```

CtD's own `pip install -e .` also works if you prefer its full requirement set.
It additionally installs JAX, DSA and OpenCV, which ayna does not need.

### 3. Run

```bash
export HOME_DIR=/path/to/ComputationThroughDynamicsToolkit/
W=$HOME_DIR/examples/walkthrough_models

ayna ctd $W/node_sweep     --model-type SAE   --tt $W/tt_3bff --out reports/node
ayna ctd $W/inputinf_sweep --model-type LFADS --tt $W/tt_3bff --out reports/lfads
```

`ayna ctd` loads each run in a fresh subprocess. CtD's datamodules hold the
whole dataset in memory and are not reliably freed, so loading several runs
in one process ran out of 8 GB. One run peaks at about 6 GB.

### Things that went wrong, and what ayna now does about them

| symptom | cause | handling |
|---|---|---|
| `UnpicklingError` / tiny `.pkl` files | Git LFS stubs | ayna names the stub files and the `git lfs pull` to run |
| process killed while exporting | CtD's getters run the whole split through the model with autograd on | ayna exports under `torch.no_grad()` |
| `ModuleNotFoundError: ray` / `wandb` | referenced by the pickles | install them (step 2) |
| trial lengths all zero, or `IndexError` in `get_trial_lens` | CtD quirk for fixed-length data | ayna treats trials as full length |
| numbers differ slightly between exports | CtD leaves models in training mode, so dropout is active | ayna calls `model.eval()` first |
