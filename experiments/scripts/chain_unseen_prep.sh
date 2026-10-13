#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Wait for both unseen-panel downloads, then preprocess (no model is run on these data here).
D=$LOG_DIR
until grep -q "^done" $D/wdf_test_download.log && grep -q "^done" $D/dff_download.log; do
  grep -q Traceback $D/wdf_test_download.log $D/dff_download.log && { echo DOWNLOAD_FAILED; exit 1; }; sleep 60; done
G=$GEND_ROOT; NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
echo "prep start $(date -Is)"; ${PYTHON:-python} preprocess_unseen.py; echo "PREP_DONE rc=$? $(date -Is)"
