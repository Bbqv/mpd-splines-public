import os, glob, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="directory containing many .npz (e.g. .../accepted)")
    ap.add_argument("--out", required=True, help="output merged npz path")
    ap.add_argument("--limit", type=int, default=0, help="0=all, else limit number of files for quick test")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    assert len(files) > 0, f"No npz found in {args.in_dir}"
    print("[INFO] merging", len(files), "files")

    merged = {}
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        for k in d.files:
            v = d[k]
            merged.setdefault(k, []).append(v)
        if (fi+1) % 200 == 0:
            print("[INFO] loaded", fi+1)

    out = {}
    for k, vs in merged.items():
        # Most keys should be arrays with first dim = N. We concatenate on axis 0.
        try:
            out[k] = np.concatenate(vs, axis=0)
        except Exception:
            # Fallback: store as object array if shapes differ
            out[k] = np.array(vs, dtype=object)

        print("[KEY]", k, "->", out[k].shape if hasattr(out[k], "shape") else type(out[k]))

    np.savez(args.out, **out)
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
