#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# FF++ c23 preprocessing for GenD retraining, README settings. Fakes use mask-guided face
# selection (official FF++ v3 masks); originals have no masks. FS/NT videos come from the local
# mirror, verified byte-identical to official files on sampled videos.
set -u
cd $GEND_ROOT
O=$RAW_DATA/FaceForensics_official
M=$RAW_DATA/FaceForensics++/FaceForensics++_C23
NV=$(${PYTHON:-python} -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
declare -A VID=( [DF]=$O/manipulated_sequences/Deepfakes/c23/videos [F2F]=$O/manipulated_sequences/Face2Face/c23/videos
                 [FS]=$M/FaceSwap [NT]=$M/NeuralTextures [real]=$O/original_sequences/youtube/c23/videos )
declare -A MASK=( [DF]=$O/manipulated_sequences/Deepfakes/masks/videos [F2F]=$O/manipulated_sequences/Face2Face/masks/videos
                  [FS]=$O/manipulated_sequences/FaceSwap/masks/videos [NT]=$O/manipulated_sequences/NeuralTextures/masks/videos )
for s in DF F2F FS NT real; do
  mkdir -p work/ff_src/$s
  for f in ${VID[$s]}/*.mp4; do ln -sf $f work/ff_src/$s/; done
  echo "$s videos=$(ls work/ff_src/$s | wc -l) masks=$( [ $s = real ] && echo - || ls ${MASK[$s]} | grep -c mp4 )"
done
for s in DF F2F FS NT real; do
  mask_arg=""; [ $s != real ] && mask_arg="--mask_folder ${MASK[$s]}"
  ${PYTHON:-python} detector.py -i work/ff_src/$s $mask_arg -m at_least -n 32 \
    -o work/datasets/FF --det_thres 0.1 -s 1.3 --target_size none --num_workers 8 --skip_processed_videos \
    > work/preprocess_ffpp_$s.log 2>&1 &
done
wait
echo PREPROCESS_DONE
