#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# RADIAL development pools: DFDC-P train split (aligned), CDF fit + dev subsample, FaceShifter dev probe.
cd $GEND_ROOT
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
${PYTHON:-python} work/preprocess_dfdcp.py --split train > work/preprocess_dfdcp_train.log 2>&1 &
for s in Celeb-real Celeb-synthesis YouTube-real; do
  ${PYTHON:-python} detector.py -i work/cdf_train_src/$s -m at_least -n 32 -o work/datasets/CDFv2_trainpool \
    --det_thres 0.1 -s 1.3 --target_size none --num_workers 4 --skip_processed_videos > work/preprocess_cdf_trainpool_$s.log 2>&1 &
done
${PYTHON:-python} detector.py -i work/fsh_src/FSh -m at_least -n 32 -o work/datasets/FF \
  --det_thres 0.1 -s 1.3 --target_size none --num_workers 4 --skip_processed_videos > work/preprocess_fsh.log 2>&1 &
wait
echo PREPROCESS_DONE
