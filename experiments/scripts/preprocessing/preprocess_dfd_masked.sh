#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# DFD re-crop, step 1: fakes with mask-guided face selection (official DFD masks, losslessly remuxed so
# that frame counts are readable). Reals: no masks exist and fakes are sub-clips of the reals (not
# frame-aligned), so reals keep the largest-face crops for now (linked from work/datasets/DFD/real).
set -u
cd $GEND_ROOT
B=$RAW_DATA/FaceForensics_official
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
mkdir -p work/datasets/DFD_masked
ln -sfn $(pwd)/work/datasets/DFD/real work/datasets/DFD_masked/real
for k in 0 1 2 3; do
  ${PYTHON:-python} detector.py -i work/dfd_split/$k/fake --mask_folder $B/manipulated_sequences/DeepFakeDetection/masks/videos_remux \
    -m at_least -n 32 -o work/datasets/DFD_masked --det_thres 0.1 -s 1.3 --target_size none --num_workers 4 \
    --skip_processed_videos > work/preprocess_dfd_masked_fake_$k.log 2>&1 &
done
wait
echo PREPROCESS_DONE
