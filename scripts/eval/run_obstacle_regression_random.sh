#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"

OBST_PRESET="${OBST_PRESET:-scripts/eval/obstacle_preset.json}"
MASTER_CKPT="${MASTER_CKPT:-checkpoints_frozen/0313_220238/base_44k.pth}"
DATA_ROOT="${DATA_ROOT:-/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd}"
DEVICE="${DEVICE:-cuda}"
N_CASES="${N_CASES:-10}"
CASE_START="${CASE_START:-0}"
N_SAMPLES="${N_SAMPLES:-64}"
SEEDS="${SEEDS:-0 1 2}"

# New: force a visible avoidance setup with only 4 REAL obstacles.
OBST_MODE="${OBST_MODE:-corridor_block4}"
OBST_TOTAL_N="${OBST_TOTAL_N:-4}"
OBST_SIZE_SCALE="${OBST_SIZE_SCALE:-4.0}"
OBST_CORRIDOR_RADIUS="${OBST_CORRIDOR_RADIUS:-0.4}"
OBST_CORRIDOR_RESAMPLE_MAX="${OBST_CORRIDOR_RESAMPLE_MAX:-50}"
OBST_EFFECT_CHECK="${OBST_EFFECT_CHECK:-0}"

RANDOM_N_EXPLICIT=0
if [[ -n "${RANDOM_N+x}" ]]; then
  RANDOM_N_EXPLICIT=1
fi
RANDOM_BOX_N_EXPLICIT=0
if [[ -n "${RANDOM_BOX_N+x}" ]]; then
  RANDOM_BOX_N_EXPLICIT=1
fi

# Keep random budget small by default; corridor_block4 will override to exactly total_n anyway.
RANDOM_N="${RANDOM_N:-4}"
RANDOM_XYZ_MIN="${RANDOM_XYZ_MIN:--0.40,-0.40,0.80}"
RANDOM_XYZ_MAX="${RANDOM_XYZ_MAX:-0.40,0.40,1.80}"
RANDOM_R_MIN="${RANDOM_R_MIN:-0.06}"
RANDOM_R_MAX="${RANDOM_R_MAX:-0.18}"
RANDOM_ANCHOR_CLEARANCE="${RANDOM_ANCHOR_CLEARANCE:-0.15}"
RANDOM_AVOID_OVERLAP="${RANDOM_AVOID_OVERLAP:-1}"
RANDOM_BOX_AVOID_OVERLAP="${RANDOM_BOX_AVOID_OVERLAP:-$RANDOM_AVOID_OVERLAP}"
RANDOM_OVERLAP_MARGIN="${RANDOM_OVERLAP_MARGIN:-0.01}"
RANDOM_SAMPLING_MODE="${RANDOM_SAMPLING_MODE:-stratified}"
RANDOM_RADIUS_MODE="${RANDOM_RADIUS_MODE:-mixed}"
RANDOM_CENTER_MIN_DIST="${RANDOM_CENTER_MIN_DIST:-0.20}"
RANDOM_MAX_TRIES="${RANDOM_MAX_TRIES:-10000}"
SCENE_BOUNDS_MODE="${SCENE_BOUNDS_MODE:-mix}"
SCENE_PAD_XY="${SCENE_PAD_XY:-0.7}"
SCENE_PAD_Z="${SCENE_PAD_Z:-0.4}"
SCENE_MIX_GLOBAL_RATIO="${SCENE_MIX_GLOBAL_RATIO:-0.9}"
GLOBAL_BOUNDS_MODE="${GLOBAL_BOUNDS_MODE:-auto}"
GLOBAL_PAD_XY="${GLOBAL_PAD_XY:-1.0}"
GLOBAL_PAD_Z="${GLOBAL_PAD_Z:-0.6}"
WORKSPACE_MIN="${WORKSPACE_MIN:-}"
WORKSPACE_MAX="${WORKSPACE_MAX:-}"
# Default: disable boxes so REAL obstacle count stays exactly 4.
RANDOM_BOX_ENABLE="${RANDOM_BOX_ENABLE:-0}"
RANDOM_BOX_N="${RANDOM_BOX_N:-0}"
RANDOM_N_MIN="${RANDOM_N_MIN:-0}"
RANDOM_N_MAX="${RANDOM_N_MAX:-256}"
RANDOM_BOX_N_MIN="${RANDOM_BOX_N_MIN:-0}"
RANDOM_BOX_N_MAX="${RANDOM_BOX_N_MAX:-256}"
DENSITY_SPH_PER_M3="${DENSITY_SPH_PER_M3:-0}"
DENSITY_BOX_PER_M3="${DENSITY_BOX_PER_M3:-0}"
RANDOM_BOX_HALF_MIN="${RANDOM_BOX_HALF_MIN:-0.04,0.04,0.05}"
RANDOM_BOX_HALF_MAX="${RANDOM_BOX_HALF_MAX:-0.10,0.10,0.18}"
RANDOM_BOX_MAX_TRIES="${RANDOM_BOX_MAX_TRIES:-10000}"
RANDOM_BOX_SAMPLING_MODE="${RANDOM_BOX_SAMPLING_MODE:-stratified}"
RANDOM_BOX_SIZE_MODE="${RANDOM_BOX_SIZE_MODE:-mixed}"
RANDOM_BOX_CENTER_MIN_DIST="${RANDOM_BOX_CENTER_MIN_DIST:-0.20}"
RANDOM_BOX_PROXY_RADIUS_SCALE="${RANDOM_BOX_PROXY_RADIUS_SCALE:-0.65}"
RANDOM_BOX_PROXY_MIN_RADIUS="${RANDOM_BOX_PROXY_MIN_RADIUS:-0.03}"
RANDOM_BOX_PROXY_MAX_PER_BOX="${RANDOM_BOX_PROXY_MAX_PER_BOX:-2}"
ANCHOR_CLEARANCE_THR="${ANCHOR_CLEARANCE_THR:-0.03}"
ANCHOR_RESAMPLE_MAX="${ANCHOR_RESAMPLE_MAX:-50}"
DISABLE_ANCHOR_RESAMPLE="${DISABLE_ANCHOR_RESAMPLE:-0}"

STRICT="${STRICT:-1}"
STRICT_CLR_MIN="${STRICT_CLR_MIN:-0.03}"
SELECT_MIN_CLEARANCE="${SELECT_MIN_CLEARANCE:-0.03}"
SAFE_MARGIN="${SAFE_MARGIN:-0.05}"
SELECT_W_GRASP="${SELECT_W_GRASP:-1.0}"
SELECT_W_UAV_J="${SELECT_W_UAV_J:-0.3}"
SELECT_W_UAV_A="${SELECT_W_UAV_A:-0.1}"
SELECT_W_PATH_LEN="${SELECT_W_PATH_LEN:-0.6}"
SELECT_W_TURN="${SELECT_W_TURN:-0.25}"
SELECT_W_STRAIGHT="${SELECT_W_STRAIGHT:-0.15}"
SELECT_MAX_PATH_LEN_RATIO="${SELECT_MAX_PATH_LEN_RATIO:-2.6}"
PROJ_W_LEN="${PROJ_W_LEN:-20.0}"
PROJ_W_REF="${PROJ_W_REF:-10.0}"
PROJ_W_LEN_RATIO="${PROJ_W_LEN_RATIO:-12.0}"
PROJ_MAX_LEN_RATIO="${PROJ_MAX_LEN_RATIO:-2.0}"
PROJ_W_DATA="${PROJ_W_DATA:-0.05}"
PROJ_W_V="${PROJ_W_V:-50.0}"
PROJ_W_A="${PROJ_W_A:-200.0}"
PROJ_W_J="${PROJ_W_J:-50.0}"
PROJ_W_UAV_BACK="${PROJ_W_UAV_BACK:-200.0}"
PROJ_W_UAV_TURN="${PROJ_W_UAV_TURN:-200.0}"
PROJ_TURN_THETA_MAX_DEG="${PROJ_TURN_THETA_MAX_DEG:-25.0}"
PROJ_W_UAV_CURV="${PROJ_W_UAV_CURV:-0.0}"
PROJ_W_UAV_CURV_HARD="${PROJ_W_UAV_CURV_HARD:-800.0}"
PROJ_UAV_CURV_D2_MAX="${PROJ_UAV_CURV_D2_MAX:-0.05}"
PROJ_PROG_EPS="${PROJ_PROG_EPS:-0.002}"
PROJ_ITERS="${PROJ_ITERS:-400}"
PROJ_MAX_DELTA="${PROJ_MAX_DELTA:-1.20}"

OUTS=()

for SEED in $SEEDS; do
  OUT="eval_regrand_seed${SEED}_$(date +%m%d_%H%M%S)"
  OUTS+=("$OUT")
  mkdir -p "$OUT"
  echo "[RUN] $OUT"

  CMD=(
    "$PYTHON" -u scripts/eval/eval_eagle_grasp.py
    --dataset_file_merged "$DATA_ROOT"
    --ckpt "$MASTER_CKPT" --device "$DEVICE"
    --obst_preset "$OBST_PRESET"
    --n_cases "$N_CASES" --n_samples "$N_SAMPLES"
    --case_start "$CASE_START"
    --save_dir "$OUT"
    --H 144 --mode endpoints_and_mid_hard --ctx_mode orig --t_g 72 --seed "$SEED"
    --obst_enable
    --obst_mode "$OBST_MODE"
    --obst_total_n "$OBST_TOTAL_N"
    --obst_size_scale "$OBST_SIZE_SCALE"
    --obst_corridor_radius "$OBST_CORRIDOR_RADIUS"
    --obst_corridor_resample_max "$OBST_CORRIDOR_RESAMPLE_MAX"
    --obst_random_enable
    --obst_random_n_min "$RANDOM_N_MIN"
    --obst_random_n_max "$RANDOM_N_MAX"
    --obst_density_sph_per_m3 "$DENSITY_SPH_PER_M3"
    --obst_scene_bounds_mode "$SCENE_BOUNDS_MODE"
    --obst_scene_pad_xy "$SCENE_PAD_XY"
    --obst_scene_pad_z "$SCENE_PAD_Z"
    --obst_scene_mix_global_ratio "$SCENE_MIX_GLOBAL_RATIO"
    --obst_global_bounds_mode "$GLOBAL_BOUNDS_MODE"
    --obst_global_pad_xy "$GLOBAL_PAD_XY"
    --obst_global_pad_z "$GLOBAL_PAD_Z"
    --obst_random_avoid_overlap "$RANDOM_AVOID_OVERLAP"
    --obst_random_box_avoid_overlap "$RANDOM_BOX_AVOID_OVERLAP"
    --obst_random_xyz_min="$RANDOM_XYZ_MIN"
    --obst_random_xyz_max="$RANDOM_XYZ_MAX"
    --obst_random_r_min "$RANDOM_R_MIN" --obst_random_r_max "$RANDOM_R_MAX"
    --obst_random_anchor_clearance "$RANDOM_ANCHOR_CLEARANCE"
    --obst_anchor_clearance_thr "$ANCHOR_CLEARANCE_THR"
    --obst_anchor_resample_max "$ANCHOR_RESAMPLE_MAX"
    --obst_random_overlap_margin "$RANDOM_OVERLAP_MARGIN"
    --obst_random_max_tries "$RANDOM_MAX_TRIES"
    --obst_random_sampling_mode "$RANDOM_SAMPLING_MODE"
    --obst_random_radius_mode "$RANDOM_RADIUS_MODE"
    --obst_random_center_min_dist "$RANDOM_CENTER_MIN_DIST"
    --obst_random_boxes_n_min "$RANDOM_BOX_N_MIN"
    --obst_random_boxes_n_max "$RANDOM_BOX_N_MAX"
    --obst_density_box_per_m3 "$DENSITY_BOX_PER_M3"
    --obst_random_box_half_min="$RANDOM_BOX_HALF_MIN"
    --obst_random_box_half_max="$RANDOM_BOX_HALF_MAX"
    --obst_random_box_max_tries "$RANDOM_BOX_MAX_TRIES"
    --obst_random_box_sampling_mode "$RANDOM_BOX_SAMPLING_MODE"
    --obst_random_box_size_mode "$RANDOM_BOX_SIZE_MODE"
    --obst_random_box_center_min_dist "$RANDOM_BOX_CENTER_MIN_DIST"
    --obst_random_box_proxy_radius_scale "$RANDOM_BOX_PROXY_RADIUS_SCALE"
    --obst_random_box_proxy_min_radius "$RANDOM_BOX_PROXY_MIN_RADIUS"
    --obst_random_box_proxy_max_per_box "$RANDOM_BOX_PROXY_MAX_PER_BOX"
    --obst_select_min_clearance "$SELECT_MIN_CLEARANCE"
    --obst_safe_margin "$SAFE_MARGIN"
    --obst_select_w_grasp "$SELECT_W_GRASP"
    --obst_select_w_j "$SELECT_W_UAV_J"
    --obst_select_w_a "$SELECT_W_UAV_A"
    --obst_select_w_len "$SELECT_W_PATH_LEN"
    --obst_select_w_turn "$SELECT_W_TURN"
    --obst_select_w_straight "$SELECT_W_STRAIGHT"
    --obst_select_max_len_ratio "$SELECT_MAX_PATH_LEN_RATIO"
	    --obst_project_enable
	    --obst_proj_adaptive_enable
	    --obst_proj_iters "$PROJ_ITERS"
	    --obst_proj_w_data "$PROJ_W_DATA"
	    --obst_proj_w_v "$PROJ_W_V"
	    --obst_proj_w_a "$PROJ_W_A"
	    --obst_proj_w_j "$PROJ_W_J"
	    --obst_proj_w_len "$PROJ_W_LEN"
	    --obst_proj_w_ref "$PROJ_W_REF"
	    --obst_proj_w_len_ratio "$PROJ_W_LEN_RATIO"
	    --obst_proj_max_len_ratio "$PROJ_MAX_LEN_RATIO"
	    --obst_proj_w_uav_back "$PROJ_W_UAV_BACK"
	    --obst_proj_w_uav_turn "$PROJ_W_UAV_TURN"
	    --obst_proj_turn_theta_max_deg "$PROJ_TURN_THETA_MAX_DEG"
	    --obst_proj_w_uav_curv "$PROJ_W_UAV_CURV"
	    --obst_proj_w_uav_curv_hard "$PROJ_W_UAV_CURV_HARD"
	    --obst_proj_uav_curv_d2_max "$PROJ_UAV_CURV_D2_MAX"
	    --obst_proj_prog_eps "$PROJ_PROG_EPS"
	    --obst_proj_max_delta "$PROJ_MAX_DELTA"
	  )

  if [[ -n "$WORKSPACE_MIN" ]]; then
    CMD+=(--obst_workspace_min="$WORKSPACE_MIN")
  fi
  if [[ -n "$WORKSPACE_MAX" ]]; then
    CMD+=(--obst_workspace_max="$WORKSPACE_MAX")
  fi
  if [[ "$DISABLE_ANCHOR_RESAMPLE" == "1" ]]; then
    CMD+=(--obst_disable_anchor_resample)
  fi
  if [[ "$RANDOM_N_EXPLICIT" == "1" ]]; then
    CMD+=(--obst_random_n "$RANDOM_N")
  fi
  if [[ "$RANDOM_BOX_ENABLE" == "1" ]]; then
    CMD+=(--obst_random_boxes_enable)
    if [[ "$RANDOM_BOX_N_EXPLICIT" == "1" ]]; then
      CMD+=(--obst_random_boxes_n "$RANDOM_BOX_N")
    fi
  fi
  if [[ "$OBST_EFFECT_CHECK" == "1" ]]; then
    CMD+=(--obst_effect_check)
  fi

  "${CMD[@]}" 2>&1 | tee "$OUT/eval.log"
done

"$PYTHON" - "${OUTS[@]}" "$STRICT" "$STRICT_CLR_MIN" <<'PY'
import re
import sys
import numpy as np

if len(sys.argv) < 4:
    raise SystemExit("[ERR] expected at least one run dir, STRICT flag and STRICT_CLR_MIN")

run_dirs = sys.argv[1:-2]
strict = int(sys.argv[-2])
strict_clr_min = float(sys.argv[-1])

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
        if clr_min < strict_clr_min:
            raise SystemExit(f"[FAIL] {d}: clr_min={clr_min:.6f} < {strict_clr_min:.6f}")
        if n_ok != n_all:
            raise SystemExit(f"[FAIL] {d}: hard anchors feasible {n_ok}/{n_all}")
    print(f"[PASS] strict checks passed (clr_min >= {strict_clr_min:.6f})")
PY

echo "[DONE] random obstacle regression completed."
