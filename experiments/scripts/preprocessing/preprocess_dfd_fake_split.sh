#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# DFD fakes: same detector.py settings as preprocess_dfd.sh, 4 processes over disjoint quarters.
cd $GEND_ROOT
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
for k in 0 1 2 3; do
  ${PYTHON:-python} detector.py -i work/dfd_split/$k/fake -m at_least -n 32 \
    -o work/datasets/DFD --det_thres 0.1 -s 1.3 --target_size none --num_workers 6 --skip_processed_videos \
    > work/preprocess_dfd_fake_$k.log 2>&1 &
done
wait
echo PREPROCESS_DONE
