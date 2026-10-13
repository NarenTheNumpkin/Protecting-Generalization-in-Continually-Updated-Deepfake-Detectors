#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
cd $GEND_ROOT
O=$RAW_DATA/FaceForensics_official/manipulated_sequences
masks_ready() { for m in Deepfakes Face2Face FaceSwap NeuralTextures; do [ "$(ls $O/$m/masks/videos 2>/dev/null | grep -c 'mp4$')" -ge 1000 ] || return 1; done; }
until grep -q PREPROCESS_DONE work/preprocess_dfd_fake_split.log && masks_ready; do sleep 30; done
echo "chain start $(date -Is)"
work/preprocess_ffpp.sh
echo "chain end $(date -Is)"
