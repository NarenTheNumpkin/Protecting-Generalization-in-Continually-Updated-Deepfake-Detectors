#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
cd $GEND_ROOT
M=$RAW_DATA/FaceForensics_official/manipulated_sequences/DeepFakeDetection/masks/videos
until grep -q "^listed" $LOG_DIR/parallel_dfd_masks.log; do sleep 30; done
echo "masks: $(ls $M | grep -c 'mp4$') $(date -Is)"
work/preprocess_dfd_masked.sh
echo "chain end $(date -Is)"
