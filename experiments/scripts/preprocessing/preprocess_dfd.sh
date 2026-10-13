#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Phase 0: GenD README preprocessing for DFD (DeepFakeDetection, FF++ v3 c23), no masks (test-time setting).
# Symlink trees give GenD's <dataset>/<source>/<video>/<frame> layout: DFD/real, DFD/fake.
set -u
cd $GEND_ROOT
B=$RAW_DATA/FaceForensics_official
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
mkdir -p work/dfd_src/real work/dfd_src/fake
for f in $B/original_sequences/actors/c23/videos/*.mp4; do ln -sf $f work/dfd_src/real/; done
for f in $B/manipulated_sequences/DeepFakeDetection/c23/videos/*.mp4; do ln -sf $f work/dfd_src/fake/; done
echo "real=$(ls work/dfd_src/real | wc -l) fake=$(ls work/dfd_src/fake | wc -l)"
for src in ${1:-real fake}; do
  ${PYTHON:-python} detector.py -i work/dfd_src/$src -m at_least -n 32 \
    -o work/datasets/DFD --det_thres 0.1 -s 1.3 --target_size none --num_workers ${WORKERS:-8} --skip_processed_videos
done
echo PREPROCESS_DONE
