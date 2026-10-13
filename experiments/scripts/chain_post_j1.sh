#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
cd "$(dirname "$0")/.."
until grep -q J1_DONE chain_j1.log; do sleep 120; done
export RADIAL_SPLITS=splits_v1.1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON:-python} analyze_j1.py > j1_analysis.txt 2>&1; echo "analyze_j1 rc=$? $(date -Is)"
${PYTHON:-python} analysis_f.py > analysis_f.log 2>&1; echo "analysis_f rc=$? $(date -Is)"
echo POST_DONE
