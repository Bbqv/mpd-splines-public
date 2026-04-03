import argparse
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

URDF_PATH = "/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf"
EE_FRAME = "gripper_link"

def quat_norm_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="traj_uav_ee.png")
    ap.add_argument("--sample", type=int, default=-1, help="-1 use best_idx")
    ap.add_argument("--stride", type=int, default=1, help="FK stride (1=every frame)")
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
    xyz_uav = traj[:, :3]

    # pinocchio FK
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)

    ee_list = []
    for t in range(0, traj.shape[0], args.stride):
        q9 = traj[t].astype(np.float64).copy()
        q9[3:7] = quat_norm_xyzw(q9[3:7])
        pin.forwardKinematics(model, data, q9)
        pin.updateFramePlacements(model, data)
        ee_list.append(data.oMf[fid].translation.copy())
    xyz_ee = np.stack(ee_list, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # UAV path
    ax.plot(xyz_uav[:,0], xyz_uav[:,1], xyz_uav[:,2], label="UAV")

    # EE path
    ax.plot(xyz_ee[:,0], xyz_ee[:,1], xyz_ee[:,2], label="EE")

    # markers
    ax.scatter([q_start[0]], [q_start[1]], [q_start[2]], marker="o", label="start(uav)")
    ax.scatter([q_goal[0]], [q_goal[1]], [q_goal[2]], marker="^", label="goal(uav)")
    ax.scatter([q_grasp[0]], [q_grasp[1]], [q_grasp[2]], marker="x", label="grasp(ee target)")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"sample={sid} (best_idx={best_idx})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()