#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
cd "$(dirname "$0")/.."
until grep -q E2_DONE chain_e2.log; do sleep 30; done
${PYTHON:-python} choose_mstar.py || exit 1
M=$(python3 -c "import json;print(json.load(open('registration/C1_choice.json'))['M_star'])")
INIT=${GEND_INIT_CKPT:-$GEND_ROOT/work/runs/phase0/phase0-retrain-GenD_CLIP-FFval-seed0/checkpoints/best_mAP.ckpt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1 RADIAL_GPU_FRACTION=0.9
echo "C1 start M*=$M $(date -Is)"
${PYTHON:-python} stream.py --method none --seed 0 --tag c1 --init $INIT --eval-role test --panel > runs_c1_none.log 2>&1; echo "none rc=$? $(date -Is)"
for s in 0 1 2; do
  for m in $M lwf seq realkd featkd; do
    ${PYTHON:-python} stream.py --method $m --seed $s --tag c1 --init $INIT --eval-role test --panel > runs_c1_${m}_seed$s.log 2>&1
    echo "$m seed $s rc=$? $(date -Is)"
  done
done
echo C1_DONE
