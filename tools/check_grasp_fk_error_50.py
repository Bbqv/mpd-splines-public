import glob, os, numpy as np
import pinocchio as pin

ROOT = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd/accepted"
URDF_PATH = "/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf"
EE_FRAME = "gripper_link"
N = 50

model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()
fid = model.getFrameId(EE_FRAME)

def quat_norm_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0,0,0,1.0], dtype=np.float64)
    return q / n

def fk_ee(q9):
    q9 = np.asarray(q9, dtype=np.float64).copy()
    q9[3:7] = quat_norm_xyzw(q9[3:7])
    pin.forwardKinematics(model, data, q9)
    pin.updateFramePlacements(model, data)
    return data.oMf[fid].translation.copy()

fs = sorted(glob.glob(os.path.join(ROOT, "traj_*.npz")))[:N]
assert len(fs) > 0, f"no npz in {ROOT}"

errs_qgs = []
errs_mid = []
bad = 0

for f in fs:
    d = np.load(f, allow_pickle=False)
    qgs = d["q_grasp_state"].astype(np.float64)
    grasp = d["q_grasp"].astype(np.float64)
    tg = int(d["t_grasp"][0]) if d["t_grasp"].ndim else int(d["t_grasp"])
    ee_traj = d["ee_traj"].astype(np.float64)

    ee_qgs = fk_ee(qgs)
    e1 = np.linalg.norm(ee_qgs - grasp)
    e2 = np.linalg.norm(ee_traj[tg] - grasp)

    errs_qgs.append(e1)
    errs_mid.append(e2)
    if e1 > 0.02:
        bad += 1

errs_qgs = np.array(errs_qgs)
errs_mid = np.array(errs_mid)

def stats(x):
    return float(x.mean()), float(np.percentile(x,50)), float(np.percentile(x,90)), float(x.max())

m,p50,p90,mx = stats(errs_qgs)
m2,p502,p902,mx2 = stats(errs_mid)

print("[CHECK] FK(q_grasp_state)->grasp (m): mean=%.4f p50=%.4f p90=%.4f max=%.4f" % (m,p50,p90,mx))
print("[CHECK] ee_traj[t_grasp]->grasp (m): mean=%.4f p50=%.4f p90=%.4f max=%.4f" % (m2,p502,p902,mx2))
print("[CHECK] count FK err > 2cm:", bad, "/", len(fs))