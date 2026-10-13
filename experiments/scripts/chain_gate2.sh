#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Gate 2 (Amendment A2): 6 methods x 3 seeds under splits_v1.1, defaults from dev_protocol_v1.md.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1
echo "start $(date -Is)"
for s in 0 1 2; do
  for m in radial sdc featkd lwf ewc lnmerge; do
    ${PYTHON:-python} stream.py --method $m --seed $s --tag gate2 > runs_gate2_${m}_seed$s.log 2>&1
    echo "$m seed $s rc=$? $(date -Is)"
  done
done
echo GATE2_DONE
