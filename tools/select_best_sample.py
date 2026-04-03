import argparse
import numpy as np

def path_length(xyz):
    d = xyz[1:] - xyz[:-1]
    return float(np.linalg.norm(d, axis=1).sum())

def smoothness_cost(xyz):
    # 用二阶差分/三阶差分做一个简单cost（和eval指标同方向）
    if xyz.shape[0] < 4:
        return 0.0
    a = xyz[2:] - 2*xyz[1:-1] + xyz[:-2]
    j = xyz[3:] - 3*xyz[2:-1] + 3*xyz[1:-2] - xyz[:-3]
    a_m = float(np.linalg.norm(a, axis=1).mean())
    j_m = float(np.linalg.norm(j, axis=1).mean())
    return a_m + 0.5 * j_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--w_len", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=1.0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    cp = d["cp"]  # [S,H,9]
    S = cp.shape[0]

    costs = []
    lens = []
    smos = []
    for s in range(S):
        xyz = cp[s, :, :3]
        L = path_length(xyz)
        Sm = smoothness_cost(xyz)
        C = args.w_len * L + args.w_smooth * Sm
        costs.append(C); lens.append(L); smos.append(Sm)

    best = int(np.argmin(costs))
    print("[BEST] sid =", best)
    print("[BEST] length =", lens[best])
    print("[BEST] smooth_cost =", smos[best])
    print("[BEST] total =", costs[best])

    # 也顺便输出 top5 方便你看差异
    order = np.argsort(costs)[:5]
    print("\n[TOP5]")
    for i in order:
        print(f"sid={int(i):3d}  total={costs[i]:.6f}  len={lens[i]:.6f}  smooth={smos[i]:.6f}")

if __name__ == "__main__":
    main()