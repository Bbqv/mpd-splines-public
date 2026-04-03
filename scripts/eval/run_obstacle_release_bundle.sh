#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints_frozen/0313_220238/base_44k.pth}"
CAND_CKPT="${CAND_CKPT:-checkpoints_frozen/obstacle_ft80k_candidate/ema_model_current.pth}"
DATA_ROOT="${DATA_ROOT:-/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd}"
DEVICE="${DEVICE:-cuda}"

RUN_BASE="${RUN_BASE:-1}"
RUN_CAND="${RUN_CAND:-1}"

FIXED_SEEDS="${FIXED_SEEDS:-0 1 2}"
FIXED_N_CASES="${FIXED_N_CASES:-10}"
FIXED_N_SAMPLES="${FIXED_N_SAMPLES:-64}"

RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
RANDOM_N_CASES="${RANDOM_N_CASES:-100}"
RANDOM_N_SAMPLES="${RANDOM_N_SAMPLES:-64}"

RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)}"
OUT_DIR="release_obstacle_bundle_${RUN_ID}"
mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.tsv"

echo -e "model\tfamily\tseed\ttag\tlog_dir\tckpt" > "$MANIFEST"

die_if_missing() {
  local p="$1"
  local name="$2"
  if [[ ! -f "$p" ]]; then
    echo "[ERR] missing $name: $p" >&2
    exit 1
  fi
}

append_fixed_rows() {
  local model="$1"
  local seed="$2"
  local ckpt="$3"
  local d
  for tag in t1 t2 t3r016; do
    d="$(ls -dt eval_regmin_${tag}_seed${seed}_* 2>/dev/null | head -1 || true)"
    if [[ -z "$d" ]]; then
      echo "[ERR] missing fixed log dir for tag=$tag seed=$seed model=$model" >&2
      exit 1
    fi
    echo -e "${model}\tfixed\t${seed}\t${tag}\t${d}\t${ckpt}" >> "$MANIFEST"
  done
}

append_random_rows() {
  local model="$1"
  local ckpt="$2"
  local d
  for seed in $RANDOM_SEEDS; do
    d="$(ls -dt eval_regrand_seed${seed}_* 2>/dev/null | head -1 || true)"
    if [[ -z "$d" ]]; then
      echo "[ERR] missing random log dir for seed=$seed model=$model" >&2
      exit 1
    fi
    echo -e "${model}\trandom\t${seed}\tall\t${d}\t${ckpt}" >> "$MANIFEST"
  done
}

run_model() {
  local model="$1"
  local ckpt="$2"

  echo "[MODEL] $model ckpt=$ckpt"

  for seed in $FIXED_SEEDS; do
    echo "[RUN] $model fixed seed=$seed"
    SEED="$seed" \
    MASTER_CKPT="$ckpt" \
    DATA_ROOT="$DATA_ROOT" \
    DEVICE="$DEVICE" \
    N_CASES="$FIXED_N_CASES" \
    N_SAMPLES="$FIXED_N_SAMPLES" \
    bash scripts/eval/run_obstacle_regression_min.sh

    append_fixed_rows "$model" "$seed" "$ckpt"
  done

  echo "[RUN] $model random seeds=[$RANDOM_SEEDS]"
  MASTER_CKPT="$ckpt" \
  DATA_ROOT="$DATA_ROOT" \
  DEVICE="$DEVICE" \
  SEEDS="$RANDOM_SEEDS" \
  N_CASES="$RANDOM_N_CASES" \
  N_SAMPLES="$RANDOM_N_SAMPLES" \
  STRICT=1 \
  bash scripts/eval/run_obstacle_regression_random.sh

  append_random_rows "$model" "$ckpt"
}

die_if_missing "$BASE_CKPT" "BASE_CKPT"
die_if_missing "$CAND_CKPT" "CAND_CKPT"

if [[ "$RUN_BASE" == "1" ]]; then
  run_model "base_44k" "$BASE_CKPT"
fi

if [[ "$RUN_CAND" == "1" ]]; then
  run_model "obstacle_ft80k_candidate" "$CAND_CKPT"
fi

python - "$MANIFEST" "$OUT_DIR" <<'PY'
import csv
import datetime
import os
import re
import statistics
import sys
from collections import defaultdict

manifest = sys.argv[1]
out_dir = sys.argv[2]

pat_succ = re.compile(r"\[METRIC\] succ@2cm\(all3\): ([0-9.]+)")
pat_uj = re.compile(r"\[SMOOTH\] uav_j_mean: mean=([-0-9.eE]+)")
pat_clr = re.compile(r"\[OBST\] clr_min\(m\): mean=([-0-9.eE]+), p50=([-0-9.eE]+), p90=([-0-9.eE]+), min=([-0-9.eE]+)")
pat_anchor = re.compile(r"\[OBST\] hard_anchor_feasible_cases: ([0-9]+)/([0-9]+)")

rows = []
with open(manifest, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        log = os.path.join(r["log_dir"], "eval.log")
        txt = open(log, "r", errors="ignore").read()
        m1 = pat_succ.search(txt)
        m2 = pat_uj.search(txt)
        m3 = pat_clr.search(txt)
        m4 = pat_anchor.search(txt)
        if not (m1 and m2 and m3 and m4):
            raise RuntimeError(f"missing summary line in {log}")
        rows.append(
            dict(
                model=r["model"],
                family=r["family"],
                seed=int(r["seed"]),
                tag=r["tag"],
                log_dir=r["log_dir"],
                ckpt=r["ckpt"],
                succ=float(m1.group(1)),
                uav_j=float(m2.group(1)),
                clr_mean=float(m3.group(1)),
                clr_min=float(m3.group(4)),
                anchors_ok=int(m4.group(1)),
                anchors_all=int(m4.group(2)),
            )
        )

if not rows:
    raise RuntimeError("no rows parsed from manifest")

def agg(items):
    def mean(k):
        return sum(x[k] for x in items) / len(items)

    def std(k):
        vals = [x[k] for x in items]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    return dict(
        n=len(items),
        succ_mean=mean("succ"),
        uav_j_mean=mean("uav_j"),
        uav_j_std=std("uav_j"),
        clr_mean_mean=mean("clr_mean"),
        clr_mean_std=std("clr_mean"),
        clr_min_mean=mean("clr_min"),
        clr_min_std=std("clr_min"),
        clr_min_worst=min(x["clr_min"] for x in items),
    )

by_model_family = defaultdict(list)
for r in rows:
    by_model_family[(r["model"], r["family"])].append(r)

models = sorted({r["model"] for r in rows})
families = ["fixed", "random"]

report = os.path.join(out_dir, "comparison.md")
with open(report, "w") as f:
    f.write("# Obstacle Release Bundle Comparison\n\n")
    f.write(f"- Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- Manifest: `{manifest}`\n\n")

    for fam in families:
        f.write(f"## {fam.capitalize()}\n\n")
        f.write("| model | n_runs | succ mean | uav_j mean/std | clr_mean mean/std | clr_min mean/std | clr_min worst |\\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\\n")
        fam_aggs = {}
        for m in models:
            items = by_model_family.get((m, fam), [])
            if not items:
                continue
            a = agg(items)
            fam_aggs[m] = a
            f.write(
                f"| {m} | {a['n']} | {a['succ_mean']:.4f} | {a['uav_j_mean']:.6f}/{a['uav_j_std']:.6f} | "
                f"{a['clr_mean_mean']:.6f}/{a['clr_mean_std']:.6f} | {a['clr_min_mean']:.6f}/{a['clr_min_std']:.6f} | {a['clr_min_worst']:.6f} |\\n"
            )

        if "base_44k" in fam_aggs and "obstacle_ft80k_candidate" in fam_aggs:
            b = fam_aggs["base_44k"]
            c = fam_aggs["obstacle_ft80k_candidate"]
            f.write("\nDelta (candidate - base):\\n")
            f.write(f"- succ mean: `{c['succ_mean'] - b['succ_mean']:+.4f}`\\n")
            f.write(f"- uav_j mean: `{c['uav_j_mean'] - b['uav_j_mean']:+.6f}` (negative is smoother)\\n")
            f.write(f"- clr_mean mean: `{c['clr_mean_mean'] - b['clr_mean_mean']:+.6f}`\\n")
            f.write(f"- clr_min worst: `{c['clr_min_worst'] - b['clr_min_worst']:+.6f}`\\n")
        f.write("\n")

print(report)
PY

echo "[DONE] bundle run finished: $OUT_DIR"
