#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CAND_DIR="${CAND_DIR:-checkpoints_frozen/obstacle_ft80k_candidate}"
MASTER_CKPT="${MASTER_CKPT:-$CAND_DIR/ema_model_current.pth}"
DATA_ROOT="${DATA_ROOT:-/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd}"
DEVICE="${DEVICE:-cuda}"

FIXED_SEEDS="${FIXED_SEEDS:-0 1 2}"
FIXED_N_CASES="${FIXED_N_CASES:-10}"
FIXED_N_SAMPLES="${FIXED_N_SAMPLES:-64}"

RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
RANDOM_N_CASES="${RANDOM_N_CASES:-100}"
RANDOM_N_SAMPLES="${RANDOM_N_SAMPLES:-64}"

if [[ ! -f "$MASTER_CKPT" ]]; then
  echo "[ERR] candidate ckpt not found: $MASTER_CKPT" >&2
  exit 1
fi

echo "[CAND] ckpt=$MASTER_CKPT"
echo "[CAND] fixed_seeds=$FIXED_SEEDS fixed_cases=$FIXED_N_CASES fixed_samples=$FIXED_N_SAMPLES"
echo "[CAND] random_seeds=$RANDOM_SEEDS random_cases=$RANDOM_N_CASES random_samples=$RANDOM_N_SAMPLES"

for s in $FIXED_SEEDS; do
  echo "[STEP] fixed obstacle regression, seed=$s"
  SEED="$s" \
  MASTER_CKPT="$MASTER_CKPT" \
  DATA_ROOT="$DATA_ROOT" \
  DEVICE="$DEVICE" \
  N_CASES="$FIXED_N_CASES" \
  N_SAMPLES="$FIXED_N_SAMPLES" \
  bash scripts/eval/run_obstacle_regression_min.sh
done

echo "[STEP] random obstacle regression"
MASTER_CKPT="$MASTER_CKPT" \
DATA_ROOT="$DATA_ROOT" \
DEVICE="$DEVICE" \
SEEDS="$RANDOM_SEEDS" \
N_CASES="$RANDOM_N_CASES" \
N_SAMPLES="$RANDOM_N_SAMPLES" \
STRICT=1 \
bash scripts/eval/run_obstacle_regression_random.sh

echo "[DONE] candidate release validation finished."
