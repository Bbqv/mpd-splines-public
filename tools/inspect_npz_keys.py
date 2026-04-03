import glob
import numpy as np

ACCEPTED = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed/accepted"

def safe_keys(npz_path):
    # allow_pickle=False 避免碰到 s500_config 这种 pickle 对象
    d = np.load(npz_path, allow_pickle=False)
    return list(d.keys())

def safe_get(d, k):
    try:
        v = d[k]
        return v
    except Exception as e:
        return None

def main():
    fs = sorted(glob.glob(ACCEPTED + "/*.npz"))
    print("found", len(fs), "npz")
    if not fs:
        return

    # 前3个文件：只打印非 object 的 array 信息
    for f in fs[:3]:
        d = np.load(f, allow_pickle=False)
        print("\n==", f)
        print("keys:", list(d.keys()))
        for k in d.keys():
            v = safe_get(d, k)
            if v is None:
                print(f"  {k:20s} <SKIP: cannot load without pickle>")
                continue
            if hasattr(v, "shape"):
                print(f"  {k:20s} shape={v.shape} dtype={v.dtype}")

    # key 频率统计（前50）
    key_count = {}
    for f in fs[:50]:
        d = np.load(f, allow_pickle=False)
        for k in d.keys():
            key_count[k] = key_count.get(k, 0) + 1

    print("\n--- key frequency in first50 (allow_pickle=False) ---")
    for k, c in sorted(key_count.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k:25s} {c}/50")

if __name__ == "__main__":
    main()