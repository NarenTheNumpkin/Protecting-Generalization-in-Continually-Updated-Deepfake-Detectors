#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Gate 1 rerun under splits_v1.1 (Amendment A1), seeds 0-2, otherwise identical to gate1.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1
echo "start $(date -Is)"
for s in 0 1 2; do
  ${PYTHON:-python} stream.py --method seq --seed $s --tag gate1_v1.1 > runs_gate1_v11_seq_seed$s.log 2>&1
  echo "seed $s rc=$? $(date -Is)"
done
echo GATE1_DONE
