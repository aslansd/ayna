#!/bin/bash
# LFADS with inferred inputs on RandomTarget (closed loop: true inputs include sensory feedback
# that depends on the network's own output). Controller KL scale sweep, one seed.
# Usage: HOME_DIR=/path/to/CtD/ ./run_lfads.sh OUT_DIR
set -u
OUT=${1:-runs}
HERE=$(cd "$(dirname "$0")" && pwd)
for kl in 0 1e-4 1e-2; do
  [ -d "$OUT/RandomTarget_LFADS/kl_co_scale=${kl}_seed=0_max_epochs=150_batch=25" ] || \
    MODEL=LFADS KL_CO=$kl python "$HERE/train_dd.py" RandomTarget 0 0 150 "$OUT" 25 2>&1 | grep -E "^trained|Error"
done
echo LFADS_DONE
