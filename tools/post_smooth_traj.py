import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_D2(H):
    # (H-2) x H
    D2 = np.zeros((H - 2, H), dtype=np.float64)
    for i in range(H - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def build_D1(H):
    # (H-1) x H
    D1 = np.zeros((H - 1, H), dtype=np.float64)
    for i in range(H - 1):
        D1[i, i] = -1.0
        D1[i, i + 1] = 1.0
    return D1


def smooth_1d(x, fixed_idx, lam_acc=1.0, lam_vel=0.1, w_fix=1e6):
    """
    minimize: lam_acc ||D2 x||^2 + lam_vel ||D1 x||^2 + w_fix * sum_i (x[i]-x0[i])^2
    via linear system solve.
    """
    H = x.shape[0]
    if H < 3:
        return x.copy()

    D2 = build_D2(H)
    D1 = build_D1(H)

    A = lam_acc * (D2.T @ D2) + lam_vel * (D1.T @ D1)
    b = np.zeros((H,), dtype=np.float64)

    # hard keep by penalty
    for i in fixed_idx:
        A[i, i] += w_fix
        b[i] += w_fix * x[i]

    # small ridge for stability
    A += 1e-9 * np.eye(H)

    y = np.linalg.solve(A, b)
    return y


def _unwrap_yaw(yaw):
    return np.unwrap(yaw)


def _wrap_yaw(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def arclen_reparam_segment(traj_seg, xyz_idx=(0, 1, 2), yaw_idx=None):
    """
    traj_seg: (T, D)
    return: (T, D) reparameterized to uniform arclength in xyz.
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

    # yaw unwrap/interp/wrap if needed
    if yaw_idx is not None and 0 <= yaw_idx < D:
        yw = _unwrap_yaw(traj_seg[:, yaw_idx])
        out[:, yaw_idx] = _wrap_yaw(np.interp(u_new, u, yw))

    # other dims linear interp (including quat/arm etc.)
    for di in range(D):
        if di in xyz_idx:
            continue
        if yaw_idx is not None and di == yaw_idx:
            continue
        out[:, di] = np.interp(u_new, u, traj_seg[:, di])

    return out


def arclen_reparam_piecewise(traj, tg, xyz_idx=(0, 1, 2), yaw_idx=None):
    """
    Keep traj[0], traj[tg], traj[-1] fixed by reparameterizing two segments:
      [0..tg] and [tg..H-1]
    """
    H, D = traj.shape
    tg = int(np.clip(tg, 1, H - 2))

    seg1 = traj[: tg + 1].copy()   # length tg+1
    seg2 = traj[tg:].copy()        # length H-tg

    seg1_r = arclen_reparam_segment(seg1, xyz_idx=xyz_idx, yaw_idx=yaw_idx)
    seg2_r = arclen_reparam_segment(seg2, xyz_idx=xyz_idx, yaw_idx=yaw_idx)

    # stitch (avoid duplicating tg point)
    out = np.vstack([seg1_r[:-1], seg2_r])

    # enforce anchors exactly
    out[0] = traj[0]
    out[tg] = traj[tg]
    out[-1] = traj[-1]
    return out


def step_dist(xyz):
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--tg", type=int, default=72, help="t_grasp index (dataset typical 72)")
    ap.add_argument("--out_npz", default="case_post.npz")
    ap.add_argument("--out_png", default="case_post.png")
    ap.add_argument("--lam_acc", type=float, default=1.0)
    ap.add_argument("--lam_vel", type=float, default=0.1)
    ap.add_argument("--w_fix", type=float, default=1e6)
    ap.add_argument("--yaw_idx", type=int, default=None,
                    help="if you store yaw as a scalar dim, set it (e.g. 3). If quaternion (3:7), keep None.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    cp = d["cp"]                 # [S,H,9]
    q_start = d["q_start"]
    q_goal = d["q_goal"]
    q_grasp = d["q_grasp"]

    traj_raw = cp[args.sid].copy()  # [H,9]
    H = traj_raw.shape[0]
    tg = int(np.clip(args.tg, 1, H - 2))
    fixed = [0, tg, H - 1]

    # ---------------------------
    # Step 1) piecewise arc-length reparam on full traj
    # (quat/arm also interpolated, but anchors stay exact)
    # ---------------------------
    traj_arclen = arclen_reparam_piecewise(
        traj_raw, tg=tg, xyz_idx=(0, 1, 2), yaw_idx=args.yaw_idx
    )

    # ---------------------------
    # Step 2) constrained smoothing on xyz + arm(7:9)
    # keep quat(3:7) untouched to avoid quaternion issues
    # ---------------------------
    traj_post = traj_arclen.copy()

    # smooth xyz
    xyz = traj_arclen[:, :3]
    xyz_s = np.zeros_like(xyz)
    for k in range(3):
        xyz_s[:, k] = smooth_1d(
            xyz[:, k], fixed,
            lam_acc=args.lam_acc, lam_vel=args.lam_vel, w_fix=args.w_fix
        )
    traj_post[:, :3] = xyz_s

    # smooth arm(7:9)
    arm = traj_arclen[:, 7:9]
    arm_s = np.zeros_like(arm)
    for k in range(2):
        arm_s[:, k] = smooth_1d(
            arm[:, k], fixed,
            lam_acc=args.lam_acc, lam_vel=args.lam_vel, w_fix=args.w_fix
        )
    traj_post[:, 7:9] = arm_s

    # ---------------------------
    # Enforce exact anchors (xyz + arm)
    # ---------------------------
    for idx in fixed:
        traj_post[idx, :3] = traj_raw[idx, :3]
        traj_post[idx, 7:9] = traj_raw[idx, 7:9]
        # quat remains from arclen interp; but anchor should also match raw if you want strict:
        traj_post[idx, 3:7] = traj_raw[idx, 3:7]

    # ---------------------------
    # Save
    # ---------------------------
    np.savez(
        args.out_npz,
        traj_raw=traj_raw,
        traj_arclen=traj_arclen,
        traj_post=traj_post,
        q_start=q_start,
        q_goal=q_goal,
        q_grasp=q_grasp,
        sid=np.array([args.sid]),
        tg=np.array([tg]),
    )

    # ---------------------------
    # Plot compare:
    # left: 3D xyz path
    # right: step distance curve
    # ---------------------------
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(traj_raw[:, 0], traj_raw[:, 1], traj_raw[:, 2], label="raw")
    ax.plot(traj_arclen[:, 0], traj_arclen[:, 1], traj_arclen[:, 2], label="arclen(piecewise)")
    ax.plot(traj_post[:, 0], traj_post[:, 1], traj_post[:, 2], label="arclen+smooth")
    ax.scatter([q_start[0]], [q_start[1]], [q_start[2]], marker="o", label="start")
    ax.scatter([q_goal[0]], [q_goal[1]], [q_goal[2]], marker="^", label="goal")
    ax.scatter([q_grasp[0]], [q_grasp[1]], [q_grasp[2]], marker="x", label="grasp(ee target)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"sid={args.sid}, tg={tg}")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(step_dist(traj_raw[:, :3]), label="raw step")
    ax2.plot(step_dist(traj_arclen[:, :3]), label="arclen step")
    ax2.plot(step_dist(traj_post[:, :3]), label="arclen+smooth step")
    ax2.set_title("Step distance per timestep")
    ax2.set_xlabel("t")
    ax2.set_ylabel("||x[t+1]-x[t]||")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("[OK] wrote", args.out_npz, "and", args.out_png)


if __name__ == "__main__":
    main()