#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Gate 1 (registered in registration/dev_protocol_v1.md): naive sequential LN fine-tuning, seeds 0-2.
# Starts after all development-pool preprocessing has finished (frees the detector GPU sessions).
G=$GEND_ROOT/work
until grep -q PREPROCESS_DONE $G/preprocess_radial_pools.log && grep -q PREPROCESS_DONE $G/preprocess_dfd_masked.log; do sleep 60; done
echo "gate1 start $(date -Is)"
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 2; do
  ${PYTHON:-python} stream.py --method seq --seed $s --tag gate1 > runs_gate1_seq_seed$s.log 2>&1
  echo "seed $s rc=$? $(date -Is)"
done
echo GATE1_DONE
