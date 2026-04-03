import glob
import numpy as np
import pinocchio as pin

ACCEPTED = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed/accepted"
URDF_PATH = "/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf"
EE_FRAME = "gripper_link"

def quat_norm_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n

def main():
    files = sorted(glob.glob(ACCEPTED + "/*.npz"))[:50]
    if len(files) == 0:
        raise RuntimeError(f"No npz found under {ACCEPTED}")

    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)

    def fk(q9):
        q9 = np.asarray(q9, dtype=np.float64).copy()
        q9[3:7] = quat_norm_xyzw(q9[3:7])
        pin.forwardKinematics(model, data, q9)
        pin.updateFramePlacements(model, data)
        return data.oMf[fid].translation.copy()

    def ik(q_init, p_target, iters=400, tol=2e-2, step=0.5, lam=1e-2):
        q = np.asarray(q_init, dtype=np.float64).copy()
        I3 = np.eye(3)
        for _ in range(iters):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            err = np.asarray(p_target, dtype=np.float64) - data.oMf[fid].translation
            if np.linalg.norm(err) < tol:
                break
            J6 = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jpos = J6[:3, :]
            A = Jpos @ Jpos.T + (lam * lam) * I3
            dq = Jpos.T @ np.linalg.solve(A, err)
            q = pin.integrate(model, q, step * dq)
        e = float(np.linalg.norm(np.asarray(p_target) - fk(q)))
        return e

    errs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        q_start = np.asarray(d["q_start"], dtype=np.float64)
        grasp = np.asarray(d["q_grasp"], dtype=np.float64)
        errs.append(ik(q_start, grasp))

    errs = np.asarray(errs, dtype=np.float64)
    ok = int(np.sum(errs <= 0.02))
    print("dryrun50 err min/median/p90/max:",
          float(errs.min()),
          float(np.median(errs)),
          float(np.percentile(errs, 90)),
          float(errs.max()))
    print("dryrun50 ok@2cm:", ok, "/", int(errs.shape[0]))

if __name__ == "__main__":
    main()