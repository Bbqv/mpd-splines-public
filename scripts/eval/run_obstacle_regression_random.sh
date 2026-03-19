#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MASTER_CKPT="${MASTER_CKPT:-checkpoints_frozen/0313_220238/base_44k.pth}"
DATA_ROOT="${DATA_ROOT:-/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd}"
DEVICE="${DEVICE:-cuda}"
N_CASES="${N_CASES:-10}"
N_SAMPLES="${N_SAMPLES:-64}"
SEEDS="${SEEDS:-0 1 2}"

RANDOM_N="${RANDOM_N:-2}"
RANDOM_XYZ_MIN="${RANDOM_XYZ_MIN:--0.40,-0.40,0.80}"
RANDOM_XYZ_MAX="${RANDOM_XYZ_MAX:-0.40,0.40,1.80}"
RANDOM_R_MIN="${RANDOM_R_MIN:-0.12}"
RANDOM_R_MAX="${RANDOM_R_MAX:-0.24}"
RANDOM_ANCHOR_CLEARANCE="${RANDOM_ANCHOR_CLEARANCE:-0.0}"
RANDOM_AVOID_OVERLAP="${RANDOM_AVOID_OVERLAP:-0}"
RANDOM_OVERLAP_MARGIN="${RANDOM_OVERLAP_MARGIN:-0.0}"

STRICT="${STRICT:-1}"

OUTS=()

for SEED in $SEEDS; do
  OUT="eval_regrand_seed${SEED}_$(date +%m%d_%H%M%S)"
  OUTS+=("$OUT")
  mkdir -p "$OUT"
  echo "[RUN] $OUT"

  CMD=(
    python -u scripts/eval/eval_eagle_grasp.py
    --dataset_file_merged "$DATA_ROOT"
    --ckpt "$MASTER_CKPT" --device "$DEVICE"
    --n_cases "$N_CASES" --n_samples "$N_SAMPLES"
    --save_dir "$OUT"
    --H 144 --mode endpoints_and_mid_hard --ctx_mode orig --t_g 72 --seed "$SEED"
    --obst_enable
    --obst_random_enable --obst_random_n "$RANDOM_N"
    --obst_random_xyz_min="$RANDOM_XYZ_MIN"
    --obst_random_xyz_max="$RANDOM_XYZ_MAX"
    --obst_random_r_min "$RANDOM_R_MIN" --obst_random_r_max "$RANDOM_R_MAX"
    --obst_random_anchor_clearance "$RANDOM_ANCHOR_CLEARANCE"
    --obst_random_overlap_margin "$RANDOM_OVERLAP_MARGIN"
    --obst_project_enable
    --obst_proj_adaptive_enable
  )

  if [[ "$RANDOM_AVOID_OVERLAP" == "1" ]]; then
    CMD+=(--obst_random_avoid_overlap)
  fi

  "${CMD[@]}" 2>&1 | tee "$OUT/eval.log"
done

python - "${OUTS[@]}" "$STRICT" <<'PY'
import re
import sys
import numpy as np

if len(sys.argv) < 3:
    raise SystemExit("[ERR] expected at least one run dir and STRICT flag")

run_dirs = sys.argv[1:-1]
strict = int(sys.argv[-1])

pat_succ = re.compile(r"\[METRIC\] succ@2cm\(all3\): ([0-9.]+)")
pat_uj = re.compile(r"\[SMOOTH\] uav_j_mean: mean=([-0-9.eE]+)")
pat_clr = re.compile(r"\[OBST\] clr_min\(m\): mean=([-0-9.eE]+), p50=([-0-9.eE]+), p90=([-0-9.eE]+), min=([-0-9.eE]+)")
pat_anchor = re.compile(r"\[OBST\] hard_anchor_feasible_cases: ([0-9]+)/([0-9]+)")

rows = []
for d in run_dirs:
    p = f"{d}/eval.log"
    txt = open(p, "r", errors="ignore").read()
    m1 = pat_succ.search(txt)
    m2 = pat_uj.search(txt)
    m3 = pat_clr.search(txt)
    m4 = pat_anchor.search(txt)
    if not (m1 and m2 and m3 and m4):
        raise SystemExit(f"[FAIL] missing summary lines: {p}")
    succ = float(m1.group(1))
    uj = float(m2.group(1))
    clr_mean = float(m3.group(1))
    clr_min = float(m3.group(4))
    n_ok = int(m4.group(1))
    n_all = int(m4.group(2))
    rows.append((d, succ, uj, clr_mean, clr_min, n_ok, n_all))
    print(f"[RUN_SUMMARY] {d} succ={succ:.4f} uav_j={uj:.6f} clr_mean={clr_mean:.6f} clr_min={clr_min:.6f} anchors={n_ok}/{n_all}")

arr = np.asarray([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
print("\n[AGG] over seeds")
print(f"[AGG] succ mean={arr[:,0].mean():.4f}")
if arr.shape[0] > 1:
    print(f"[AGG] uav_j_mean mean/std={arr[:,1].mean():.6f}/{arr[:,1].std(ddof=1):.6f}")
    print(f"[AGG] clr_mean  mean/std={arr[:,2].mean():.6f}/{arr[:,2].std(ddof=1):.6f}")
    print(f"[AGG] clr_min   mean/std={arr[:,3].mean():.6f}/{arr[:,3].std(ddof=1):.6f}")
else:
    print(f"[AGG] uav_j_mean={arr[:,1].mean():.6f}")
    print(f"[AGG] clr_mean={arr[:,2].mean():.6f}")
    print(f"[AGG] clr_min={arr[:,3].mean():.6f}")

if strict:
    for d, succ, _, _, clr_min, n_ok, n_all in rows:
        if succ < 1.0:
            raise SystemExit(f"[FAIL] {d}: succ={succ:.4f} < 1.0")
        if clr_min < 0.0:
            raise SystemExit(f"[FAIL] {d}: clr_min={clr_min:.6f} < 0")
        if n_ok != n_all:
            raise SystemExit(f"[FAIL] {d}: hard anchors feasible {n_ok}/{n_all}")
    print("[PASS] strict checks passed")
PY

echo "[DONE] random obstacle regression completed."
