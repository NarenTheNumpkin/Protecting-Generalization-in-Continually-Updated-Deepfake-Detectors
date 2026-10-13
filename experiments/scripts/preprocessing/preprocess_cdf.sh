#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Phase 0: GenD README preprocessing settings applied to the official CDF-v2 518-video test list.
set -u
cd $GEND_ROOT
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
for src in Celeb-real YouTube-real Celeb-synthesis; do
  ${PYTHON:-python} detector.py -i work/cdf_test_src/$src -m at_least -n 32 \
    -o work/datasets/CDFv2 --det_thres 0.1 -s 1.3 --target_size none --num_workers 8
done
echo PREPROCESS_DONE
