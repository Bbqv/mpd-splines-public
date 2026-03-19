#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MASTER_CKPT="${MASTER_CKPT:-checkpoints_frozen/0313_220238/base_44k.pth}"
DATA_ROOT="${DATA_ROOT:-/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd}"
DEVICE="${DEVICE:-cuda}"
N_CASES="${N_CASES:-10}"
N_SAMPLES="${N_SAMPLES:-64}"
SEED="${SEED:-0}"

run_one () {
  TAG="$1"
  S1="$2"
  S2="$3"

  OUT="eval_regmin_${TAG}_seed${SEED}_$(date +%m%d_%H%M%S)"
  mkdir -p "$OUT"
  echo "[RUN] $OUT"

  python -u scripts/eval/eval_eagle_grasp.py \
    --dataset_file_merged "$DATA_ROOT" \
    --ckpt "$MASTER_CKPT" --device "$DEVICE" \
    --n_cases "$N_CASES" --n_samples "$N_SAMPLES" \
    --save_dir "$OUT" \
    --H 144 --mode endpoints_and_mid_hard --ctx_mode orig --t_g 72 --seed "$SEED" \
    --obst_enable \
    --obst_sphere="$S1" \
    --obst_sphere="$S2" \
    --obst_alpha 0.0 --obst_t_start_guide 0 \
    --obst_margin 0.10 --obst_max_push 0.0 \
    --obst_project_enable \
    --obst_proj_iters 80 --obst_proj_lr 0.005 \
    --obst_proj_w_data 40 --obst_proj_w_a 30 --obst_proj_w_j 8 --obst_proj_w_obst 80 \
    --obst_proj_margin 0.05 --obst_proj_max_grad_value 0.05 --obst_proj_max_delta 0.03 \
    --obst_proj_adaptive_enable \
    --obst_proj_adaptive_rounds 2 \
    --obst_proj_adaptive_clearance 0.0 \
    --obst_proj_adaptive_iters_mult 2.0 \
    --obst_proj_adaptive_w_obst_mult 2.0 \
    --obst_proj_adaptive_max_delta_mult 2.0 \
    --obst_proj_adaptive_max_grad_mult 1.5 \
    2>&1 | tee "$OUT/eval.log"

  python - "$OUT/eval.log" <<'PY'
import re, sys
log = open(sys.argv[1], "r", errors="ignore").read()
ms = re.search(r"\[METRIC\] succ@2cm\(all3\): ([0-9.]+)", log)
mc = re.search(r"\[OBST\] clr_min\(m\): mean=([-0-9.eE]+), p50=([-0-9.eE]+), p90=([-0-9.eE]+), min=([-0-9.eE]+)", log)
if not ms or not mc:
    raise SystemExit("[FAIL] missing succ/clr summary in log")
succ = float(ms.group(1))
clr_min = float(mc.group(4))
if succ < 1.0:
    raise SystemExit(f"[FAIL] succ={succ:.4f} < 1.0")
if clr_min < 0.0:
    raise SystemExit(f"[FAIL] clr_min={clr_min:.6f} < 0")
print(f"[PASS] succ={succ:.4f}, clr_min={clr_min:.6f}")
PY
}

# t1
run_one "t1" "-0.10,-0.20,1.20,0.25" "0.00,-0.18,1.45,0.20"
# t2
run_one "t2" "-0.20,-0.25,1.10,0.22" "0.12,-0.12,1.35,0.18"
# t3 (feasible version)
run_one "t3r016" "-0.05,-0.28,1.00,0.20" "0.08,-0.10,1.55,0.16"

echo "[DONE] minimal obstacle regression passed."
