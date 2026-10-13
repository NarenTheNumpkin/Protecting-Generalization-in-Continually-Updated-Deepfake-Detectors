#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Lossless remux (-c copy) of DFD mask videos: the official files are fragmented MP4s without frame-count
# metadata, which imageio/pyav improps reads as 0 frames. Pixels verified identical on a sample.
FF=$(${PYTHON:-python} -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
M=$RAW_DATA/FaceForensics_official/manipulated_sequences/DeepFakeDetection/masks
mkdir -p $M/videos_remux
ls $M/videos | grep 'mp4$' | xargs -P 8 -I{} sh -c "[ -f $M/videos_remux/{} ] || $FF -loglevel error -y -i $M/videos/{} -c copy -map 0 -movflags faststart $M/videos_remux/{}"
echo "remuxed $(ls $M/videos_remux | grep -c 'mp4$')"
