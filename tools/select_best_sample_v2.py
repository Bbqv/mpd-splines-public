import argparse
import numpy as np

def path_length(xyz):
    d = xyz[1:] - xyz[:-1]
    return float(np.linalg.norm(d, axis=1).sum())

def mean_acc(xyz):
    if xyz.shape[0] < 3: return 0.0
    a = xyz[2:] - 2*xyz[1:-1] + xyz[:-2]
    return float(np.linalg.norm(a, axis=1).mean())

def mean_jerk(xyz):
    if xyz.shape[0] < 4: return 0.0
    j = xyz[3:] - 3*xyz[2:-1] + 3*xyz[1:-2] - xyz[:-3]
    return float(np.linalg.norm(j, axis=1).mean())

def speed_variation(xyz):
    d = xyz[1:] - xyz[:-1]
    v = np.linalg.norm(d, axis=1)
    if v.mean() < 1e-9: return 0.0
    return float(v.std() / (v.mean() + 1e-9))  # 变异系数：越小越“匀速”

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--w_len", type=float, default=1.0)
    ap.add_argument("--w_acc", type=float, default=2.0)
    ap.add_argument("--w_jerk", type=float, default=1.0)
    ap.add_argument("--w_spd", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    cp = d["cp"]  # [S,H,9]
    S = cp.shape[0]

    rows = []
    for s in range(S):
        xyz = cp[s, :, :3]
        L = path_length(xyz)
        A = mean_acc(xyz)
        J = mean_jerk(xyz)
        SV = speed_variation(xyz)
        C = args.w_len*L + args.w_acc*A + args.w_jerk*J + args.w_spd*SV
        rows.append((C, s, L, A, J, SV))

    rows.sort(key=lambda x: x[0])
    best = rows[0]
    print(f"[BEST] sid={best[1]}  total={best[0]:.6f}  len={best[2]:.6f}  acc={best[3]:.6f}  jerk={best[4]:.6f}  spd_cv={best[5]:.6f}")

    print("\n[TOP]")
    for i,(C,s,L,A,J,SV) in enumerate(rows[:args.topk]):
        print(f"{i:02d} sid={s:3d}  total={C:.6f}  len={L:.6f}  acc={A:.6f}  jerk={J:.6f}  spd_cv={SV:.6f}")

if __name__ == "__main__":
    main()