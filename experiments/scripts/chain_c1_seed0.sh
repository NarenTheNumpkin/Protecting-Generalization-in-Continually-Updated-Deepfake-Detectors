#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# C1-A: finish seed 0 only. Waits for the seq seed-0 run that is already in progress.
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "[s]tream.py --method seq --seed 0 --tag c1"; do sleep 20; done
echo "seq seed 0 finished $(date -Is) results=$( [ -f runs/c1/seq_seed0/results.json ] && grep -c panel runs/c1/seq_seed0/results.json )"
INIT=${GEND_INIT_CKPT:-$GEND_ROOT/work/runs/phase0/phase0-retrain-GenD_CLIP-FFval-seed0/checkpoints/best_mAP.ckpt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1 RADIAL_GPU_FRACTION=0.9
for m in realkd featkd; do
  ${PYTHON:-python} stream.py --method $m --seed 0 --tag c1 --init $INIT --eval-role test --panel > runs_c1_${m}_seed0.log 2>&1
  echo "$m seed 0 rc=$? $(date -Is)"
done
echo C1_SEED0_DONE
