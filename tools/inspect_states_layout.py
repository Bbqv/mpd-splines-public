import glob, numpy as np

ACCEPTED="/home/yongxin/wpj/dataset_room_4x4x2_relaxed/accepted"

def main():
    f = sorted(glob.glob(ACCEPTED+"/*.npz"))[0]
    d = np.load(f, allow_pickle=False)
    states = d["states"]  # (T,17)
    ee = d["ee_positions"]  # (T,3)
    print("file:", f)
    print("states shape:", states.shape, "dtype:", states.dtype)
    print("ee_positions shape:", ee.shape, "dtype:", ee.dtype)

    # 打印每一维的 min/max（前10维先看）
    mins = states.min(axis=0)
    maxs = states.max(axis=0)
    print("\nstates dim stats (idx: min .. max):")
    for i in range(states.shape[1]):
        print(f"  {i:02d}: {mins[i]: .4f} .. {maxs[i]: .4f}")

    # 打印头尾两帧 states（看是不是 start/goal）
    print("\nstate[0]:", states[0])
    print("state[-1]:", states[-1])
    print("\nee[0]:", ee[0])
    print("ee[-1]:", ee[-1])

if __name__ == "__main__":
    main()