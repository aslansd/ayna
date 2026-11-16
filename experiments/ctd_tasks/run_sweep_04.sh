#!/bin/bash
# 0.4 fresh held-out grid: seeds 2 and 3, never trained or scanned before 0.4's
# detector changes were committed. Sequential: 2 CPU cores, 8 GB RAM.
# Usage: HOME_DIR=/path/to/CtD/ ./run_sweep_04.sh OUT_DIR
set -u
OUT=${1:-runs04}
HERE=$(cd "$(dirname "$0")" && pwd)
train() { python "$HERE/train_dd.py" "$@" "$OUT" 25 2>&1 | grep -E "^trained|Error" ; }
for s in 2 3; do for d in 3 10 64; do
  [ -d "$OUT/RandomTarget/latent_size=${d}_seed=${s}_max_epochs=150_batch=25" ] || train RandomTarget $d $s 150; done; done
for s in 2 3; do for d in 3 8 64; do
  [ -d "$OUT/NBFF/latent_size=${d}_seed=${s}_max_epochs=100_batch=25" ] || train NBFF $d $s 100; done; done
echo SWEEP04_DONE
