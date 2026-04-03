#!/usr/bin/env python3
# tools/postprocess_arclen_smooth.py

import argparse
import numpy as np
import matplotlib.pyplot as plt


def path_length_xyz(xyz):
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return float(d.sum()), d


def speed_cv(step_dists, eps=1e-9):
    mu = float(np.mean(step_dists))
    sd = float(np.std(step_dists))
    return sd / (mu + eps)


def finite_diff_metrics(xyz, dt=1.0):
    v = np.diff(xyz, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    a_norm = np.linalg.norm(a, axis=1)
    j_norm = np.linalg.norm(j, axis=1)
    return float(a_norm.mean()) if len(a_norm) else 0.0, float(j_norm.mean()) if len(j_norm) else 0.0


def point_line_distances(points, p0, p1, eps=1e-9):
    v = p1 - p0
    vv = float(np.dot(v, v)) + eps
    w = points - p0
    proj = (w @ v) / vv
    closest = p0[None, :] + proj[:, None] * v[None, :]
    d = np.linalg.norm(points - closest, axis=1)
    return d


def straightness_p(points, p0, p1, p=90):
    d = point_line_distances(points, p0, p1)
    return float(np.percentile(d, p))


def unwrap_yaw(yaw):
    return np.unwrap(yaw)


def wrap_yaw(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Piecewise arc-length reparam
# -----------------------------
def arclen_reparam_segment(traj_seg, xyz_idx=(0, 1, 2), yaw_idx=None):
    """
    traj_seg: (T,D) -> uniform arclength in xyz.
    """
    T, D = traj_seg.shape
    if T <= 2:
        return traj_seg.copy()

    xyz = traj_seg[:, list(xyz_idx)]
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return traj_seg.copy()

    u = s / total
    u_new = np.linspace(0.0, 1.0, T)

    out = np.zeros_like(traj_seg)

    # xyz interp
    for di in xyz_idx:
        out[:, di] = np.interp(u_new, u, traj_seg[:, di])

    # optional yaw scalar interp (ONLY if you truly have yaw dim)
    if yaw_idx is not None and 0 <= yaw_idx < D:
        yw = unwrap_yaw(traj_seg[:, yaw_idx])
        out[:, yaw_idx] = wrap_yaw(np.interp(u_new, u, yw))

    # remaining dims linear interp
    for di in range(D):
        if di in xyz_idx:
            continue
        if yaw_idx is not None and di == yaw_idx:
            continue
        out[:, di] = np.interp(u_new, u, traj_seg[:, di])

    return out


def arclen_reparam_piecewise(traj, tg, xyz_idx=(0, 1, 2), yaw_idx=None):
    """
    Respect hard anchors at 0/tg/H-1 by reparameterizing two segments:
      [0..tg] and [tg..H-1]
    """
    H, D = traj.shape
    tg = int(np.clip(tg, 1, H - 2))

    seg1 = traj[: tg + 1].copy()
    seg2 = traj[tg:].copy()

    seg1_r = arclen_reparam_segment(seg1, xyz_idx=xyz_idx, yaw_idx=yaw_idx)
    seg2_r = arclen_reparam_segment(seg2, xyz_idx=xyz_idx, yaw_idx=yaw_idx)

    out = np.vstack([seg1_r[:-1], seg2_r])

    # enforce exact anchors (all dims)
    out[0] = traj[0]
    out[tg] = traj[tg]
    out[-1] = traj[-1]
    return out


# -----------------------------
# Two-segment straight reference
# -----------------------------
def make_piecewise_straight_ref(x0, x_tg, xT, tg, H):
    """
    Returns xyz_ref: (H,3)
    Segment 1: t=0..tg lerp x0->x_tg
    Segment 2: t=tg..H-1 lerp x_tg->xT
    """
    tg = int(np.clip(tg, 1, H - 2))
    ref = np.zeros((H, 3), dtype=float)

    # seg1
    for t in range(0, tg + 1):
        a = t / float(tg)
        ref[t] = (1.0 - a) * x0 + a * x_tg

    # seg2
    denom = float((H - 1) - tg)
    for t in range(tg, H):
        a = (t - tg) / denom
        ref[t] = (1.0 - a) * x_tg + a * xT

    # enforce exact
    ref[0] = x0
    ref[tg] = x_tg
    ref[-1] = xT
    return ref


# -----------------------------
# Constrained smoothing (exact)
# -----------------------------
def build_D1(H):
    D = np.zeros((H - 1, H))
    for i in range(H - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


def build_D2(H):
    D = np.zeros((H - 2, H))
    for i in range(H - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def build_D3(H):
    D = np.zeros((H - 3, H))
    for i in range(H - 3):
        D[i, i] = -1.0
        D[i, i + 1] = 3.0
        D[i, i + 2] = -3.0
        D[i, i + 3] = 1.0
    return D


def constrained_smooth_1d(x0, fixed_idx, fixed_val,
                          w_data=10.0, w_v=1.0, w_a=10.0, w_j=1.0,
                          w_ref=0.0, x_ref=None, w_ref_vec=None):
    """
    Solve:
      min_x w_data||x-x0||^2
           + w_v||D1 x||^2 + w_a||D2 x||^2 + w_j||D3 x||^2
           + w_ref||x-x_ref||^2
      s.t. x[i]=fixed_val for i in fixed_idx

    Exact constraints by eliminating fixed variables.
    """
    H = x0.shape[0]
    I = np.eye(H)
    D1 = build_D1(H) if H >= 2 else np.zeros((0, H))
    D2 = build_D2(H) if H >= 3 else np.zeros((0, H))
    D3 = build_D3(H) if H >= 4 else np.zeros((0, H))

    A = w_data * (I.T @ I)
    b = w_data * x0

    if D1.shape[0]:
        A += w_v * (D1.T @ D1)
    if D2.shape[0]:
        A += w_a * (D2.T @ D2)
    if D3.shape[0]:
        A += w_j * (D3.T @ D3)

    if x_ref is not None:
        if w_ref_vec is not None:
            wr = np.asarray(w_ref_vec, dtype=float).reshape(H)
            wr = np.clip(wr, 0.0, None)
            A += np.diag(wr)
            b += wr * x_ref
        elif w_ref > 0.0:
            A += w_ref * (I.T @ I)
            b += w_ref * x_ref

    fixed_idx = np.array(sorted(set(int(i) for i in fixed_idx)), dtype=int)
    fixed_val = np.array(fixed_val, dtype=float)

    free_mask = np.ones(H, dtype=bool)
    free_mask[fixed_idx] = False
    free_idx = np.where(free_mask)[0]

    x = np.zeros(H, dtype=float)
    x[fixed_idx] = fixed_val

    A_ff = A[np.ix_(free_idx, free_idx)]
    A_fc = A[np.ix_(free_idx, fixed_idx)]
    rhs = b[free_idx] - A_fc @ x[fixed_idx]

    A_ff = A_ff + 1e-9 * np.eye(A_ff.shape[0])
    sol = np.linalg.solve(A_ff, rhs)
    x[free_idx] = sol
    return x


def constrained_smooth(traj, fixed_points, smooth_dims,
                       w_data=10.0, w_v=1.0, w_a=10.0, w_j=1.0,
                       w_ref=0.0, x_ref_full=None, ref_dims=None, w_ref_vec=None):
    """
    traj: (H,D)
    fixed_points: list of (idx, value_vec(D))
    smooth_dims: dims to smooth
    x_ref_full: optional (H,D) reference trajectory (only used on ref_dims)
    ref_dims: list/set of dims that use reference term
    """
    H, D = traj.shape
    out = traj.copy()

    fixed_idx = [i for i, _ in fixed_points]
    fixed_vals = {i: v for i, v in fixed_points}

    ref_dims = set(ref_dims) if ref_dims is not None else set()

    for d in smooth_dims:
        x0 = traj[:, d]
        fval = np.array([fixed_vals[i][d] for i in fixed_idx], dtype=float)

        use_ref = (x_ref_full is not None) and (d in ref_dims) and (w_ref > 0.0)
        xr = x_ref_full[:, d] if use_ref else None
        wr = w_ref if use_ref else 0.0
        wr_vec = w_ref_vec if use_ref else None

        out[:, d] = constrained_smooth_1d(
            x0,
            fixed_idx=fixed_idx,
            fixed_val=fval,
            w_data=w_data, w_v=w_v, w_a=w_a, w_j=w_j,
            w_ref=wr, x_ref=xr, w_ref_vec=wr_vec
        )
    return out


def plot_compare_xyz(raw, rep, sm, out_png):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(raw[:, 0], raw[:, 1], raw[:, 2], label="raw")
    ax.plot(rep[:, 0], rep[:, 1], rep[:, 2], label="arclen(piecewise)")
    ax.plot(sm[:, 0], sm[:, 1], sm[:, 2], label="post")
    ax.scatter([raw[0, 0]], [raw[0, 1]], [raw[0, 2]], s=30, label="start")
    ax.scatter([raw[-1, 0]], [raw[-1, 1]], [raw[-1, 2]], s=30, label="goal")
    ax.set_title("XYZ 3D")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax2 = fig.add_subplot(122)

    def stepdist(xyz):
        return np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    ax2.plot(stepdist(raw[:, :3]), label="raw step")
    ax2.plot(stepdist(rep[:, :3]), label="arclen step")
    ax2.plot(stepdist(sm[:, :3]), label="post step")
    ax2.set_title("Step distance per timestep")
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--tg", type=int, required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_png", required=True)

    # IMPORTANT: your cp seems [x,y,z, qx,qy,qz,qw, arm0, arm1]
    ap.add_argument("--yaw_idx", type=int, default=None,
                    help="set only if you truly have yaw as scalar dim; quaternion -> leave None")

    ap.add_argument("--smooth_dims", type=str, default="0,1,2,7,8",
                    help="dims to smooth; default smooth xyz+arm, keep quat")

    ap.add_argument("--w_data", type=float, default=20.0)
    ap.add_argument("--w_v", type=float, default=5.0)
    ap.add_argument("--w_a", type=float, default=50.0)
    ap.add_argument("--w_j", type=float, default=5.0)

    # shortest-path / straight preference
    ap.add_argument("--w_line", type=float, default=0.0,
                    help="pull xyz toward piecewise straight (start->tg->goal). 0 disables.")
    ap.add_argument("--w_line_tg_mask", type=int, default=0,
                    help="disable line pull in [tg-K, tg+K] to avoid sharp corner at grasp")

    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    cp = data["cp"]  # (S,H,9)
    traj_raw = cp[args.sid]  # (H,D)

    H, D = traj_raw.shape
    tg = int(np.clip(args.tg, 1, H - 2))

    fixed_points = [
        (0, traj_raw[0].copy()),
        (tg, traj_raw[tg].copy()),
        (H - 1, traj_raw[H - 1].copy()),
    ]

    # Step 1: piecewise arc-length reparam (fix tg spike + uniform step)
    traj_rep = arclen_reparam_piecewise(traj_raw, tg=tg, xyz_idx=(0, 1, 2), yaw_idx=args.yaw_idx)

    # Build piecewise straight reference for xyz (start->tg->goal) in same length H
    x0 = traj_raw[0, :3]
    xg = traj_raw[tg, :3]
    xT = traj_raw[-1, :3]
    xyz_ref = make_piecewise_straight_ref(x0, xg, xT, tg=tg, H=H)

    # pack ref into (H,D) for passing
    ref_full = np.zeros((H, D), dtype=float)
    ref_full[:, 0:3] = xyz_ref
    w_line_vec = np.full(H, float(args.w_line), dtype=float)
    if args.w_line_tg_mask > 0:
        k = int(max(0, args.w_line_tg_mask))
        lo = max(0, tg - k)
        hi = min(H - 1, tg + k)
        w_line_vec[lo:hi + 1] = 0.0

    # Step 2: constrained smoothing (+ optional straight pull on xyz)
    smooth_dims = [int(x) for x in args.smooth_dims.split(",") if x.strip() != ""]
    traj_sm = constrained_smooth(
        traj_rep,
        fixed_points=fixed_points,
        smooth_dims=smooth_dims,
        w_data=args.w_data, w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
        w_ref=args.w_line, x_ref_full=ref_full, ref_dims=[0, 1, 2], w_ref_vec=w_line_vec
    )

    # Enforce exact anchors after smoothing (all dims)
    for idx, val in fixed_points:
        traj_rep[idx] = val.copy()
        traj_sm[idx] = val.copy()

    # diagnostics
    start_xyz = traj_raw[0, :3]
    goal_xyz = traj_raw[-1, :3]

    def report(name, tr):
        L, step = path_length_xyz(tr[:, :3])
        spdcv = speed_cv(step)
        acc_m, jerk_m = finite_diff_metrics(tr[:, :3])
        straight = straightness_p(tr[:, :3], start_xyz, goal_xyz, p=90)
        print(f"{name:>14s}: len={L:.4f}  straight_p90={straight:.4f}  acc={acc_m:.4f}  jerk={jerk_m:.4f}  spd_cv={spdcv:.4f}")

    report("raw", traj_raw)
    report("arclen_pw", traj_rep)
    report("post", traj_sm)

    np.savez(
        args.out_npz,
        traj_raw=traj_raw,
        traj_arclen=traj_rep,
        traj_post=traj_sm,
        q_start=data["q_start"] if "q_start" in data else traj_raw[0].copy(),
        q_goal=data["q_goal"] if "q_goal" in data else traj_raw[-1].copy(),
        q_grasp=data["q_grasp"] if "q_grasp" in data else None,
        sid=args.sid,
        tg=tg,
        source_npz=args.npz,
        w_line=args.w_line,
        w_line_tg_mask=int(args.w_line_tg_mask),
    )

    plot_compare_xyz(traj_raw, traj_rep, traj_sm, args.out_png)
    print(f"[OK] wrote {args.out_npz} and {args.out_png}")


if __name__ == "__main__":
    main()
