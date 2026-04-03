#!/usr/bin/env python3
# tools/select_best_sample_v3.py
import argparse
import numpy as np

def percentile(x, p):
    return np.percentile(np.asarray(x), p)

def path_length_xyz(xyz):
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return float(d.sum()), d

def speed_cv(step_dists, eps=1e-9):
    mu = float(np.mean(step_dists))
    sd = float(np.std(step_dists))
    return sd / (mu + eps)

def finite_diff_metrics(xyz, dt=1.0):
    # simple discrete derivatives on xyz
    v = np.diff(xyz, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    a_norm = np.linalg.norm(a, axis=1)
    j_norm = np.linalg.norm(j, axis=1)
    return float(a_norm.mean()) if len(a_norm) else 0.0, float(j_norm.mean()) if len(j_norm) else 0.0

def point_line_distances(points, p0, p1, eps=1e-9):
    # distance from point to infinite line through p0->p1
    v = p1 - p0
    vv = float(np.dot(v, v)) + eps
    w = points - p0
    proj = (w @ v) / vv  # (H,)
    closest = p0[None, :] + proj[:, None] * v[None, :]
    d = np.linalg.norm(points - closest, axis=1)
    return d

def point_segment_distances(points, p0, p1, eps=1e-9):
    # distance from point to finite segment p0->p1
    v = p1 - p0
    vv = float(np.dot(v, v)) + eps
    w = points - p0
    proj = (w @ v) / vv
    proj = np.clip(proj, 0.0, 1.0)
    closest = p0[None, :] + proj[:, None] * v[None, :]
    d = np.linalg.norm(points - closest, axis=1)
    return d

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # expected: cp (S,H,9), q_start, q_goal, q_grasp, best_idx...
    cp = data["cp"]
    q_start = data["q_start"]
    q_goal = data["q_goal"]
    q_grasp = data["q_grasp"] if "q_grasp" in data else None
    best_idx = int(data["best_idx"]) if "best_idx" in data else None
    return data, cp, q_start, q_goal, q_grasp, best_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)

    # weights
    ap.add_argument("--w_len", type=float, default=5.0)
    ap.add_argument("--w_straight", type=float, default=10.0)
    ap.add_argument("--w_acc", type=float, default=1.0)
    ap.add_argument("--w_jerk", type=float, default=1.0)
    ap.add_argument("--w_spdcv", type=float, default=5.0)
    ap.add_argument("--w_len_ratio_over", type=float, default=0.0,
                    help="penalty on max(0, len/chord - len_ratio_ref)")

    # straightness stats
    ap.add_argument("--straight_p", type=float, default=90.0, help="percentile for straightness cost, e.g. 90")
    ap.add_argument("--line_mode", type=str, choices=["segment", "line"], default="segment",
                    help="straightness distance to segment or infinite line")
    ap.add_argument("--len_ratio_ref", type=float, default=1.25,
                    help="acceptable len/chord ratio before over-penalty")
    ap.add_argument("--max_len_ratio", type=float, default=0.0,
                    help="hard filter: drop sample if len/chord > this value (<=0 disables)")
    ap.add_argument("--max_spdcv", type=float, default=0.0,
                    help="hard filter: drop sample if spd_cv > this value (<=0 disables)")

    args = ap.parse_args()
    data, cp, q_start, q_goal, q_grasp, best_idx = load_npz(args.npz)

    S, H, D = cp.shape
    assert D >= 3, "cp must include xyz in first 3 dims"
    start_xyz = cp[0, 0, :3]  # or q_start? but cp already matches hard constraint
    goal_xyz  = cp[0, -1, :3]

    rows = []
    filtered_out = 0
    for sid in range(S):
        xyz = cp[sid, :, :3]

        L, step = path_length_xyz(xyz)
        spdcv = speed_cv(step)

        acc_m, jerk_m = finite_diff_metrics(xyz, dt=args.dt)

        # straightness
        if args.line_mode == "segment":
            dline = point_segment_distances(xyz, start_xyz, goal_xyz)
        else:
            dline = point_line_distances(xyz, start_xyz, goal_xyz)
        straight = float(percentile(dline, args.straight_p))
        chord = float(np.linalg.norm(goal_xyz - start_xyz))
        len_ratio = L / max(chord, 1e-9)
        len_ratio_over = max(0.0, len_ratio - args.len_ratio_ref)

        if args.max_len_ratio > 0.0 and len_ratio > args.max_len_ratio:
            filtered_out += 1
            continue
        if args.max_spdcv > 0.0 and spdcv > args.max_spdcv:
            filtered_out += 1
            continue

        cost = (
            args.w_len * L
            + args.w_straight * straight
            + args.w_acc * acc_m
            + args.w_jerk * jerk_m
            + args.w_spdcv * spdcv
            + args.w_len_ratio_over * len_ratio_over
        )
        rows.append((cost, sid, L, straight, acc_m, jerk_m, spdcv, len_ratio, len_ratio_over))

    if len(rows) == 0:
        raise RuntimeError(
            f"All samples filtered out (S={S}, filtered={filtered_out}). "
            f"Relax --max_len_ratio / --max_spdcv."
        )

    rows.sort(key=lambda x: x[0])
    best = rows[0]
    if filtered_out > 0:
        print(f"[FILTER] dropped {filtered_out}/{S} samples by hard thresholds")

    print(f"BEST sid={best[1]}")
    print(f"cost={best[0]:.6f}  len={best[2]:.4f}  straight_p{args.straight_p:g}={best[3]:.4f}  "
          f"acc={best[4]:.4f}  jerk={best[5]:.4f}  spd_cv={best[6]:.4f}  "
          f"len_ratio={best[7]:.4f}")
    if best_idx is not None:
        print(f"(npz best_idx={best_idx})")

    print("\nTOP:")
    for r in rows[:args.topk]:
        cost, sid, L, straight, acc_m, jerk_m, spdcv, len_ratio, len_ratio_over = r
        print(
            f"sid={sid:3d} cost={cost:.6f} len={L:.4f} straight={straight:.4f} "
            f"acc={acc_m:.4f} jerk={jerk_m:.4f} spd_cv={spdcv:.4f} "
            f"len_ratio={len_ratio:.4f}"
        )

if __name__ == "__main__":
    main()
