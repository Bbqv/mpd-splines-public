#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BASE_CKPT="${BASE_CKPT:-checkpoints_frozen/0313_220238/base_44k.pth}"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[ERR] BASE_CKPT not found: $BASE_CKPT" >&2
  exit 1
fi

SHA_OUT="${BASE_CKPT}.sha256"
sha256sum "$BASE_CKPT" > "$SHA_OUT"
echo "[CKPT] sha256 -> $SHA_OUT"

# Resume
export RESUME_CKPT="${RESUME_CKPT:-$BASE_CKPT}"
export RESUME_OPT="${RESUME_OPT:-0}"

# Training schedule
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-80000}"
export LOG_EVERY="${LOG_EVERY:-100}"
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

# Planner-like losses (keep existing defaults)
export PL_SOFT_CAP="${PL_SOFT_CAP:-0}"
export PL_USE_PRED_SMOOTH="${PL_USE_PRED_SMOOTH:-0}"

# Obstacle-aware fine-tune (new)
export OBST_TRAIN_ENABLE="${OBST_TRAIN_ENABLE:-1}"
export OBST_W="${OBST_W:-0.20}"
export OBST_N_SPHERES="${OBST_N_SPHERES:-2}"
export OBST_MARGIN="${OBST_MARGIN:-0.05}"
export OBST_R_MIN="${OBST_R_MIN:-0.12}"
export OBST_R_MAX="${OBST_R_MAX:-0.24}"
export OBST_XYZ_MIN="${OBST_XYZ_MIN:--0.40,-0.40,0.80}"
export OBST_XYZ_MAX="${OBST_XYZ_MAX:-0.40,0.40,1.80}"
export OBST_TG_MASK_K="${OBST_TG_MASK_K:-8}"

echo "[RUN] RESUME_CKPT=$RESUME_CKPT"
echo "[RUN] NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"
echo "[RUN] OBST_TRAIN_ENABLE=$OBST_TRAIN_ENABLE OBST_W=$OBST_W OBST_N_SPHERES=$OBST_N_SPHERES"

python -u tools/train_eagle_3pt_plannerlike.py
