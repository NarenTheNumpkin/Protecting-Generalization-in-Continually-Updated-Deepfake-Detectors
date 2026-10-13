#!/bin/bash
# Locations (override as needed): GEND_ROOT = GenD clone, RAW_DATA = raw downloads, LOG_DIR = download logs, PYTHON = interpreter
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${GEND_ROOT:=$REPO_ROOT/third_party/GenD}"; : "${RAW_DATA:=$REPO_ROOT/data/raw}"; : "${LOG_DIR:=$REPO_ROOT/logs}"
# Journal extension J1 (registration/journal_J1.md): all blocks in priority order, seed 0. Continues on failure.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RADIAL_SPLITS=splits_v1.1 RADIAL_GPU_FRACTION=0.9
INIT=${GEND_INIT_CKPT:-$GEND_ROOT/work/runs/phase0/phase0-retrain-GenD_CLIP-FFval-seed0/checkpoints/best_mAP.ckpt}
run() { local tag=$1; local name=$2; shift 2
  if [ -f runs/$tag/$name/results.json ] && grep -q '"panel"' runs/$tag/$name/results.json; then echo "skip $tag/$name (done)"; return; fi
  ${PYTHON:-python} stream.py --seed 0 --tag $tag --run-name $name --eval-role test --panel "$@" > logs_j1/${tag}_${name}.log 2>&1
  echo "$tag/$name rc=$? $(date -Is)"; }
mkdir -p logs_j1
echo "J1 start $(date -Is)"
# Block A + M (S1, clean init)
for m in featkd_lwf head ewc lnmerge replay joint fakekd randkd; do run j1_s1 ${m}_seed0 --method $m --init $INIT; done
echo BLOCK_AM_DONE
# Block O (task orders)
for o in "S2:FF,DFDCP,CDF,DFD" "S3:FF,DFD,DFDCP,CDF"; do
  for m in realkd_lwf lwf seq realkd featkd featkd_lwf; do run j1_${o%%:*} ${m}_seed0 --method $m --init $INIT --stream ${o#*:}; done
done
echo BLOCK_O_DONE
# Block B (PE backbone, released init)
for m in none realkd_lwf lwf seq realkd featkd featkd_lwf; do run j1_pe ${m}_seed0 --method $m --init yermandy/GenD_PE_L; done
echo BLOCK_B_DONE
# Block L (sensitivity, light eval)
for l in 0.3 3 10; do
  run j1_lam realkd_lam${l} --method realkd --lam $l --init $INIT --eval-light
  run j1_lam featkd_lam${l} --method featkd --lam $l --init $INIT --eval-light
done
for gl in "0.3:1" "3:1" "1:0.3" "1:3"; do
  run j1_lam realkd_lwf_g${gl%%:*}_l${gl#*:} --method realkd_lwf --lam ${gl%%:*} --lam-lwf ${gl#*:} --init $INIT --eval-light
done
echo BLOCK_L_DONE
# Block S (data regime)
for k in 5 10; do for m in realkd_lwf lwf seq; do run j1_shots ${m}_shots${k} --method $m --shots $k --init $INIT; done; done
echo J1_DONE
