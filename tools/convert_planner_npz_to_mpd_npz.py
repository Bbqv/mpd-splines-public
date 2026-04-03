import os, glob
import numpy as np

SRC = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed/accepted"
DST = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd/accepted"
H = 144

def resample(arr, H):
    # arr: (T, D)
    T = arr.shape[0]
    xs = np.linspace(0, 1, T)
    xt = np.linspace(0, 1, H)
    out = np.zeros((H, arr.shape[1]), dtype=np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(xt, xs, arr[:, j])
    return out

def main():
    os.makedirs(DST, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(SRC, "*.npz")))
    print("found", len(fs), "src npz")
    if not fs:
        return

    for i, f in enumerate(fs):
        d = np.load(f, allow_pickle=False)
        states = np.asarray(d["states"], dtype=np.float64)      # (T,17)
        ee = np.asarray(d["ee_positions"], dtype=np.float64)    # (T,3)
        traj9 = states[:, :9].astype(np.float32)                # (T,9)

        T = traj9.shape[0]
        tg_src = T // 2

        # 重采样到 H
        traj9_H = resample(traj9, H)        # (H,9)
        ee_H = resample(ee.astype(np.float32), H)  # (H,3)

        tg = H // 2
        out = {
            "traj": traj9_H,                         # (H,9)
            "q_start": traj9_H[0].copy(),            # (9,)
            "q_goal": traj9_H[-1].copy(),            # (9,)
            "t_grasp": np.array([tg], dtype=np.int64),
            "q_grasp_state": traj9_H[tg].copy(),     # (9,)
            "q_grasp": ee_H[tg].copy(),              # (3,) 作为 grasp 点
            "ee_traj": ee_H,                          # (H,3) 可选，调试用
            "src_file": np.array([os.path.basename(f)]),
        }

        out_path = os.path.join(DST, os.path.basename(f))
        np.savez(out_path, **out)

        if (i+1) % 500 == 0:
            print("converted", i+1, "latest", out_path)

    print("done. dst:", DST)

if __name__ == "__main__":
    main()