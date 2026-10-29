#!/bin/bash
# The NODE grid behind VALIDATION.md's held-out section. Sequential: 2 CPU cores, 8 GB RAM.
# Usage: HOME_DIR=/path/to/CtD/ ./run_sweep.sh OUT_DIR
set -u
OUT=${1:-runs}
HERE=$(cd "$(dirname "$0")" && pwd)
train() { python "$HERE/train_dd.py" "$@" "$OUT" 25 2>&1 | grep -E "^trained|Error" ; }
# task          latent seed epochs
for d in 3 10 64; do for s in 0 1; do
  [ -d "$OUT/RandomTarget/latent_size=${d}_seed=${s}_max_epochs=150_batch=25" ] || train RandomTarget $d $s 150; done; done
for d in 3 8 64; do
  [ -d "$OUT/NBFF/latent_size=${d}_seed=1_max_epochs=100_batch=25" ] || train NBFF $d 1 100; done
for d in 5 20 64; do
  [ -d "$OUT/MultiTask/latent_size=${d}_seed=0_max_epochs=25_batch=25" ] || train MultiTask $d 0 25; done
echo SWEEP_DONE
