import argparse
import numpy as np
import matplotlib.pyplot as plt


def draw_box(ax, mn, mx):
    # 画轴对齐包围盒线框
    x0,y0,z0 = mn
    x1,y1,z1 = mx
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=float)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i,j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="eval生成的 case_xxxx_samples.npz")
    ap.add_argument("--out", default="traj_vis.png")
    ap.add_argument("--sample", type=int, default=-1, help="-1表示用best_idx，否则用指定sample id")
    ap.add_argument("--room_mode", type=str, default="auto", choices=["auto","none"], help="auto=用数据min/max画盒子")
    ap.add_argument("--pad", type=float, default=0.05, help="auto盒子边界外扩比例（相对range）")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    cp = d["cp"]                 # [S,H,9]
    q_start = d["q_start"]       # [9]
    q_goal = d["q_goal"]         # [9]
    q_grasp = d["q_grasp"]       # [3]
    best_idx = int(d["best_idx"][0]) if "best_idx" in d else 0

    sid = best_idx if args.sample < 0 else args.sample
    sid = max(0, min(sid, cp.shape[0]-1))

    traj = cp[sid]               # [H,9]
    xyz = traj[:, :3]

    # 自动估计房间边界：用所有sample的xyz范围（更稳）
    if args.room_mode == "auto":
        xyz_all = cp[:, :, :3].reshape(-1, 3)
        mn = xyz_all.min(axis=0)
        mx = xyz_all.max(axis=0)
        rng = np.maximum(mx - mn, 1e-6)
        mn = mn - args.pad * rng
        mx = mx + args.pad * rng
    else:
        mn = mx = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2])
    ax.scatter([q_start[0]], [q_start[1]], [q_start[2]], marker="o")
    ax.scatter([q_goal[0]], [q_goal[1]], [q_goal[2]], marker="^")
    ax.scatter([q_grasp[0]], [q_grasp[1]], [q_grasp[2]], marker="x")

    if mn is not None:
        draw_box(ax, mn, mx)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{args.npz} | sample={sid} (best_idx={best_idx})")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()