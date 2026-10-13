#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1 RADIAL_GPU_FRACTION=0.9
echo "start $(date -Is)"
for s in 0 1 2; do
  for m in realkd radial_lwf; do
    ${PYTHON:-python} stream.py --method $m --seed $s --tag e1 > runs_e1_${m}_seed$s.log 2>&1
    echo "$m seed $s rc=$? $(date -Is)"
  done
done
echo E1_DONE
