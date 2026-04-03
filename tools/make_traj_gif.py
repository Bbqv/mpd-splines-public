import argparse
import glob
import math
import os
import time
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

try:
    from scipy.interpolate import splprep, splev
    _HAVE_SCIPY = True
except Exception:
    splprep = None
    splev = None
    _HAVE_SCIPY = False


def _short_exc(e: Exception, max_len: int = 220) -> str:
    msg = f"{type(e).__name__}: {str(e)}"
    msg = msg.replace("\n", " ").replace("\r", " ")
    msg = " ".join(msg.split())
    if len(msg) > int(max_len):
        msg = msg[: int(max_len) - 3] + "..."
    return msg


def _obst_hash_sha1(obst_spheres, obst_boxes) -> str:
    """
    Stable obstacle hash to verify 'eval saved obstacles' == 'viz loaded obstacles'.
    Must match scripts/eval/eval_eagle_grasp.py::_obst_hash_sha1.
    """
    sph = np.asarray(obst_spheres, dtype=np.float32).reshape(-1, 4)
    box = np.asarray(obst_boxes, dtype=np.float32).reshape(-1, 6)
    h = hashlib.sha1()
    h.update(sph.tobytes(order="C"))
    h.update(box.tobytes(order="C"))
    return h.hexdigest()


def _dedup_consecutive_xyz(xyz: np.ndarray, eps: float = 1e-6):
    """
    Remove consecutive near-duplicate points (distance < eps).
    Returns:
      xyz_dedup: (m,3)
      raw_to_dedup: (H,) mapping raw index -> dedup index
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    H = int(xyz.shape[0])
    if H == 0:
        return xyz, np.zeros((0,), dtype=np.int64)
    eps = float(max(0.0, eps))
    out = [xyz[0]]
    raw_to_dedup = np.zeros((H,), dtype=np.int64)
    j = 0
    for i in range(1, H):
        if eps > 0.0 and float(np.linalg.norm(xyz[i] - out[-1])) < eps:
            raw_to_dedup[i] = j
            continue
        out.append(xyz[i])
        j += 1
        raw_to_dedup[i] = j
    return np.asarray(out, dtype=np.float64), raw_to_dedup


def _upsample_seg_mask_from_idx(seg_mask_raw: np.ndarray, idx_map: np.ndarray, n_up: int):
    """
    Upsample a raw segment mask (length H-1) onto the upsampled trajectory segments (length n_up-1),
    using the raw->upsample index mapping idx_map (length H).
    """
    seg_mask_raw = np.asarray(seg_mask_raw, dtype=bool).reshape(-1)
    idx_map = np.asarray(idx_map, dtype=np.int64).reshape(-1)
    n_up = int(n_up)
    if n_up < 2:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((n_up - 1,), dtype=bool)
    if seg_mask_raw.size == 0 or idx_map.size < 2:
        return out
    H = int(idx_map.size)
    n_raw_seg = min(int(seg_mask_raw.size), H - 1)
    for i in range(n_raw_seg):
        if not bool(seg_mask_raw[i]):
            continue
        a = int(idx_map[i])
        b = int(idx_map[i + 1])
        if b <= a:
            continue
        a = max(0, min(a, n_up - 1))
        b = max(0, min(b, n_up - 1))
        if b > a:
            out[a:b] = True
    return out


def _upsample_seg_mask(seg_mask, up: int):
    seg_mask = np.asarray(seg_mask, dtype=bool).reshape(-1)
    u = int(max(1, up))
    if u == 1 or seg_mask.size == 0:
        return seg_mask
    return np.repeat(seg_mask, repeats=u, axis=0)


def _upsample_xyz_for_plot(
    xyz: np.ndarray,
    up: int,
    use_spline: bool = True,
    smooth="auto",
    debug: bool = False,
):
    """
    Upsample xyz for nicer visualization only.
    Returns:
      xyz_up: (H_up, 3)
      idx_map: (H,) mapping from original waypoint index -> xyz_up index
      info: dict (debug metadata)
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    up = int(max(1, up))
    H_raw = int(xyz.shape[0])
    if H_raw < 2 or up == 1:
        info = dict(
            H_raw=H_raw,
            m_dedup=H_raw,
            method="raw",
            spline_ok=False,
            k=0,
            s=0.0,
            smooth_spec=str(smooth),
            have_scipy=bool(_HAVE_SCIPY),
        )
        return xyz, np.arange(H_raw, dtype=np.int64), info

    # Robust smoothing needs consecutive-point deduplication; otherwise splprep can fail or degrade.
    dedup_eps = 1e-6
    xyz_dedup, raw_to_dedup = _dedup_consecutive_xyz(xyz, eps=dedup_eps)
    m_dedup = int(xyz_dedup.shape[0])
    n_drop = int(H_raw - m_dedup)

    # Chord-length (arc-length proxy) parameterization u in [0,1].
    if m_dedup < 2:
        n_up = int((H_raw - 1) * up + 1)
        xyz_up = np.repeat(xyz[:1], repeats=n_up, axis=0)
        idx_map = np.zeros((H_raw,), dtype=np.int64)
        info = dict(
            H_raw=H_raw,
            m_dedup=m_dedup,
            dedup_drop=n_drop,
            method="degenerate",
            spline_ok=False,
            k=0,
            s=0.0,
            smooth_spec=str(smooth),
            have_scipy=bool(_HAVE_SCIPY),
        )
        return xyz_up, idx_map, info

    du = np.linalg.norm(xyz_dedup[1:] - xyz_dedup[:-1], axis=1).astype(np.float64)
    L = float(np.sum(du))
    if (not np.isfinite(L)) or L <= 1e-12:
        n_up = int((H_raw - 1) * up + 1)
        xyz_up = np.repeat(xyz_dedup[:1], repeats=n_up, axis=0)
        idx_map = np.zeros((H_raw,), dtype=np.int64)
        info = dict(
            H_raw=H_raw,
            m_dedup=m_dedup,
            dedup_drop=n_drop,
            method="degenerate",
            spline_ok=False,
            k=0,
            s=0.0,
            smooth_spec=str(smooth),
            have_scipy=bool(_HAVE_SCIPY),
        )
        return xyz_up, idx_map, info
    u_dedup = np.concatenate([[0.0], np.cumsum(du) / L], axis=0).astype(np.float64)
    u_dedup[0] = 0.0
    u_dedup[-1] = 1.0

    n_up = int((H_raw - 1) * up + 1)
    t_raw = np.arange(H_raw, dtype=np.float64)
    t_up = np.linspace(0.0, float(H_raw - 1), n_up, dtype=np.float64)

    # Map raw time -> chord-length u for stable fitting, then evaluate the curve on t_up.
    # This keeps the raw->upsample index mapping simple: idx_map[t] == t * up.
    u_raw = u_dedup[np.asarray(raw_to_dedup, dtype=np.int64)]
    u_eval = np.interp(t_up, t_raw, u_raw).astype(np.float64)

    idx_map = (np.arange(H_raw, dtype=np.int64) * int(up)).astype(np.int64)
    idx_map = np.clip(idx_map, 0, int(n_up - 1))

    # Parse smoothing parameter: allow "auto" or a numeric coefficient.
    smooth_spec = str(smooth).strip().lower()
    smooth_mode = "interp"
    s_val = 0.0
    step = float(np.median(du)) if du.size > 0 else 0.0
    if (not np.isfinite(step)) or step <= 1e-12:
        step = float(np.mean(du)) if du.size > 0 else 0.0
    scale2 = float(max(step * step, 1e-12))
    if smooth_spec in ("", "0", "0.0", "none", "off", "interp"):
        s_val = 0.0
        smooth_mode = "interp"
    elif smooth_spec in ("auto", "a"):
        # Default: moderate smoothing scaled by point count and step scale.
        coeff = 1.0
        s_val = float(coeff) * float(m_dedup) * scale2
        smooth_mode = "auto"
    elif smooth_spec.startswith("abs:"):
        try:
            s_val = float(max(0.0, float(smooth_spec.split(":", 1)[1])))
            smooth_mode = "abs"
        except Exception:
            s_val = 0.0
            smooth_mode = "bad"
    else:
        try:
            coeff = float(smooth_spec)
            if coeff <= 0.0:
                s_val = 0.0
                smooth_mode = "interp"
            else:
                s_val = float(coeff) * float(m_dedup) * scale2
                smooth_mode = "coef"
        except Exception:
            s_val = 0.0
            smooth_mode = "bad"

    def _catmullrom_chain(points: np.ndarray, n_out: int, alpha: float = 0.5):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        m = int(points.shape[0])
        n_out = int(max(2, n_out))
        if m <= 1:
            return np.repeat(points[:1], repeats=n_out, axis=0)
        if m == 2:
            a = points[0]
            b = points[1]
            t = np.linspace(0.0, 1.0, n_out, dtype=np.float64)[:, None]
            return (1.0 - t) * a[None, :] + t * b[None, :]

        def tj(ti, pi, pj):
            d = float(np.linalg.norm(pj - pi))
            d = max(d, 1e-12)
            return ti + (d ** float(alpha))

        def seg(p0, p1, p2, p3, n_seg: int):
            n_seg = int(max(2, n_seg))
            t0 = 0.0
            t1 = tj(t0, p0, p1)
            t2 = tj(t1, p1, p2)
            t3 = tj(t2, p2, p3)
            t = np.linspace(t1, t2, n_seg, dtype=np.float64)[:, None]

            def lerp(pa, pb, ta, tb):
                den = float(max(1e-12, tb - ta))
                wa = (tb - t) / den
                wb = (t - ta) / den
                return wa * pa[None, :] + wb * pb[None, :]

            a1 = lerp(p0, p1, t0, t1)
            a2 = lerp(p1, p2, t1, t2)
            a3 = lerp(p2, p3, t2, t3)
            b1 = lerp(a1, a2, t0, t2)
            b2 = lerp(a2, a3, t1, t3)
            c = lerp(b1, b2, t1, t2)
            return c

        n_per = int(max(6, math.ceil(float(n_out) / float(m - 1))))
        chunks = []
        for i in range(m - 1):
            p0 = points[i - 1] if i - 1 >= 0 else points[0]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2] if (i + 2) < m else points[-1]
            s = seg(p0, p1, p2, p3, n_per)
            if i < (m - 2):
                chunks.append(s[:-1])
            else:
                chunks.append(s)
        q = np.concatenate(chunks, axis=0)
        if q.shape[0] == n_out:
            return q
        # Resample by chord length to hit exact n_out.
        dd = np.linalg.norm(q[1:] - q[:-1], axis=1)
        ss = np.concatenate([[0.0], np.cumsum(dd)], axis=0)
        total = float(ss[-1])
        if (not np.isfinite(total)) or total <= 1e-12:
            return np.repeat(q[:1], repeats=n_out, axis=0)
        ss = ss / total
        t_new = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
        out = np.stack([np.interp(t_new, ss, q[:, k]) for k in range(3)], axis=1).astype(np.float64)
        return out

    def _resample_by_chord(q: np.ndarray, u_target: np.ndarray):
        q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
        u_target = np.asarray(u_target, dtype=np.float64).reshape(-1)
        if q.shape[0] < 2 or u_target.size == 0:
            return np.repeat(q[:1], repeats=int(max(1, u_target.size)), axis=0)
        dd = np.linalg.norm(q[1:] - q[:-1], axis=1)
        ss = np.concatenate([[0.0], np.cumsum(dd)], axis=0)
        total = float(ss[-1])
        if (not np.isfinite(total)) or total <= 1e-12:
            return np.repeat(q[:1], repeats=int(max(1, u_target.size)), axis=0)
        ss = ss / total
        u_target = np.clip(u_target, 0.0, 1.0)
        out = np.stack([np.interp(u_target, ss, q[:, k]) for k in range(3)], axis=1).astype(np.float64)
        return out

    method = "linear"
    spline_ok = False
    k = int(min(3, m_dedup - 1))
    fallback_reason = ""

    xyz_up = None
    if bool(use_spline) and bool(_HAVE_SCIPY) and (splprep is not None) and (splev is not None) and m_dedup >= 2 and k >= 1:
        try:
            tck, _ = splprep(
                [xyz_dedup[:, 0], xyz_dedup[:, 1], xyz_dedup[:, 2]],
                u=u_dedup,
                s=float(s_val),
                k=int(k),
            )
            x, y, z = splev(u_eval, tck)
            xyz_up = np.stack([x, y, z], axis=1).astype(np.float64)
            method = "bspline"
            spline_ok = True
        except Exception as e:
            fallback_reason = _short_exc(e)
            print(f"[PLOT_SPLINE_FALLBACK] method=bspline reason={fallback_reason}")
            xyz_up = None

    if xyz_up is None and bool(use_spline):
        try:
            n_dense = int(max(n_up, (m_dedup - 1) * int(up) * 6 + 1))
            q_dense = _catmullrom_chain(xyz_dedup, n_out=n_dense, alpha=0.5)
            xyz_up = _resample_by_chord(q_dense, u_eval)
            method = "catmullrom"
            spline_ok = True
        except Exception as e:
            fallback_reason = _short_exc(e)
            print(f"[PLOT_SPLINE_FALLBACK] method=catmullrom reason={fallback_reason}")
            xyz_up = None

    if xyz_up is None:
        # Final fallback: chord-length linear interpolation (still avoids raw-index parameterization issues).
        xyz_up = np.stack([np.interp(u_eval, u_dedup, xyz_dedup[:, k]) for k in range(3)], axis=1).astype(np.float64)
        method = "linear"
        spline_ok = False

    # Keep endpoints exact for visualization sanity.
    xyz_up[0] = xyz_dedup[0]
    xyz_up[-1] = xyz_dedup[-1]

    info = dict(
        H_raw=H_raw,
        m_dedup=m_dedup,
        dedup_drop=n_drop,
        dedup_eps=float(dedup_eps),
        H_up=int(xyz_up.shape[0]),
        up=int(up),
        method=str(method),
        spline_ok=bool(spline_ok),
        k=int(k),
        s=float(s_val),
        smooth_spec=str(smooth),
        smooth_mode=str(smooth_mode),
        have_scipy=bool(_HAVE_SCIPY),
    )
    if debug:
        print(
            "[PLOT_CURVE] "
            f"raw={H_raw} dedup={m_dedup} drop={n_drop} eps={dedup_eps:g} "
            f"up={up} H_up={int(xyz_up.shape[0])} "
            f"method={method} spline_ok={bool(spline_ok)} k={k} "
            f"smooth_spec='{smooth_spec}' mode={smooth_mode} s={float(s_val):.6g}"
        )
    return xyz_up, idx_map, info


def _cumulative_arclen_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    H = int(xyz.shape[0])
    if H <= 0:
        return np.zeros((0,), dtype=np.float64)
    if H == 1:
        return np.zeros((1,), dtype=np.float64)
    ds = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1).astype(np.float64)
    s = np.concatenate([[0.0], np.cumsum(ds)], axis=0).astype(np.float64)
    return s


def _make_playback_frames_arclen(xyz_draw: np.ndarray, n_frames: int, td_start: int = 0):
    """
    Return a nondecreasing list of draw indices (td) so that each frame advances
    by approximately uniform arc length along xyz_draw.
    """
    xyz_draw = np.asarray(xyz_draw, dtype=np.float64).reshape(-1, 3)
    H_up = int(xyz_draw.shape[0])
    n_frames = int(max(1, n_frames))
    if H_up <= 0:
        return [0] * n_frames
    if n_frames == 1:
        return [H_up - 1]

    td_start = int(np.clip(int(td_start), 0, H_up - 1))
    s = _cumulative_arclen_xyz(xyz_draw)
    s_end = float(s[-1]) if s.size else 0.0
    if (not np.isfinite(s_end)) or s_end <= 1e-12:
        return [H_up - 1] * n_frames
    s0 = float(s[td_start]) if s.size else 0.0
    if (not np.isfinite(s0)) or (s_end - s0) <= 1e-12:
        return [H_up - 1] * n_frames

    targets = np.linspace(s0, s_end, n_frames, dtype=np.float64)
    td = np.searchsorted(s, targets, side="left").astype(np.int64)
    td = np.clip(td, 0, H_up - 1)
    td[-1] = H_up - 1
    for i in range(1, int(td.size)):
        if td[i] < td[i - 1]:
            td[i] = td[i - 1]
    return td.astype(int).tolist()


def draw_box(ax, mn, mx):
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            linewidth=1.0
        )


def draw_sphere(
    ax,
    center,
    radius,
    color="tab:red",
    alpha=0.12,
    edgecolor="#7f003f",
    n_u=18,
    n_v=14,
):
    cx, cy, cz = center
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)
    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=0.25,
        antialiased=True,
        shade=True,
    )


def draw_aabb(
    ax,
    center,
    half_ext,
    color="#c2185b",
    alpha=0.22,
    edgecolor="#5a003e",
):
    c = np.asarray(center, dtype=float).reshape(3)
    h = np.asarray(half_ext, dtype=float).reshape(3)
    mn = c - h
    sz = 2.0 * h
    ax.bar3d(
        [float(mn[0])],
        [float(mn[1])],
        [float(mn[2])],
        [float(sz[0])],
        [float(sz[1])],
        [float(sz[2])],
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=0.5,
        shade=True,
    )


def _resolve_npz_arg(npz_arg, extras):
    candidates = []
    if npz_arg:
        candidates.append(npz_arg)
    for x in extras:
        s = str(x).strip()
        if s:
            candidates.append(s)
    if len(candidates) == 0:
        return ""

    expanded = []
    for c in candidates:
        g = glob.glob(c)
        if len(g) == 0:
            expanded.append(c)
        else:
            expanded.extend(g)
    uniq = []
    seen = set()
    for p in expanded:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    exist = [p for p in uniq if os.path.exists(p)]
    if len(exist) == 0:
        return uniq[0]
    if len(exist) == 1:
        return exist[0]
    # If multiple paths are provided/expanded, pick newest by mtime.
    best = sorted(exist, key=lambda p: os.path.getmtime(p), reverse=True)[0]
    print(f"[WARN] multiple npz candidates found ({len(exist)}), using newest: {best}")
    return best


def _compute_clearance_curves(xyz, obst_spheres):
    """
    Returns:
      point_sd: (T,) min signed distance to obstacle surface at waypoints
      seg_sd:   (T-1,) min signed distance to obstacle surface over each segment
    """
    T = int(xyz.shape[0])
    if len(obst_spheres) == 0:
        return (
            np.full((T,), np.inf, dtype=float),
            np.full((max(0, T - 1),), np.inf, dtype=float),
        )

    cc = np.asarray([[float(x), float(y), float(z)] for x, y, z, _ in obst_spheres], dtype=float)
    rr = np.asarray([float(r) for _, _, _, r in obst_spheres], dtype=float)

    # Waypoint-wise signed distance to sphere surface.
    diff = xyz[:, None, :] - cc[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) - rr[None, :]
    point_sd = np.min(dist, axis=1)

    # Segment-wise signed distance is kept for diagnostics only.
    if T < 2:
        seg_sd = np.full((0,), np.inf, dtype=float)
    else:
        p0 = xyz[:-1, :]
        p1 = xyz[1:, :]
        v = p1 - p0
        vv = np.sum(v * v, axis=1)
        eps = 1e-12
        seg_all = []
        for j in range(cc.shape[0]):
            c = cc[j : j + 1, :]
            w = c - p0
            t = np.zeros_like(vv)
            nz = vv > eps
            t[nz] = np.sum(w[nz] * v[nz], axis=1) / vv[nz]
            t = np.clip(t, 0.0, 1.0)
            closest = p0 + t[:, None] * v
            d = np.linalg.norm(closest - c, axis=1) - rr[j]
            seg_all.append(d)
        seg_sd = np.min(np.stack(seg_all, axis=1), axis=1)

    return point_sd, seg_sd


def _segments_to_polyline_xyz(xyz_prefix, seg_mask_prefix):
    pts = []
    seg_mask_prefix = np.asarray(seg_mask_prefix, dtype=bool).reshape(-1)
    for i in np.where(seg_mask_prefix)[0]:
        ii = int(i)
        pts.append(xyz_prefix[ii])
        pts.append(xyz_prefix[ii + 1])
        pts.append(np.array([np.nan, np.nan, np.nan], dtype=float))
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


def _pointmask_to_segmask(mask_prefix):
    m = np.asarray(mask_prefix, dtype=bool).reshape(-1)
    if m.size < 2:
        return np.zeros((0,), dtype=bool)
    return m[:-1] & m[1:]


def _save_clearance_plot(path, clearance, safe_margin):
    fig = plt.figure(figsize=(7.2, 2.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    t = np.arange(clearance.shape[0], dtype=int)
    ax.plot(t, clearance, color="tab:blue", linewidth=1.8, label="clearance (surface - uav_radius)")
    ax.axhline(0.0, color="tab:red", linewidth=1.2, linestyle="--", label="collision boundary")
    if safe_margin > 0.0:
        ax.axhline(float(safe_margin), color="darkorange", linewidth=1.2, linestyle="--", label="safe margin")
    ax.set_xlabel("t")
    ax.set_ylabel("clearance (m)")
    ax.set_title(
        f"Min Clearance vs Time  (d_min={float(np.min(clearance)):.4f} m)"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] wrote", path)


def _mask_intervals(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    out = []
    s = int(idx[0])
    p = int(idx[0])
    for i in idx[1:]:
        ii = int(i)
        if ii == p + 1:
            p = ii
            continue
        out.append((s, p))
        s = ii
        p = ii
    out.append((s, p))
    return out


def _compute_dynamics_metrics(xyz: np.ndarray, dt: float = 1.0):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    dt = float(max(1e-8, dt))
    if xyz.shape[0] < 2:
        return dict(
            speed=np.zeros((0,), dtype=np.float64),
            acc=np.zeros((0,), dtype=np.float64),
            jerk=np.zeros((0,), dtype=np.float64),
            dv=np.zeros((0,), dtype=np.float64),
        )
    vel = (xyz[1:] - xyz[:-1]) / dt
    speed = np.linalg.norm(vel, axis=1)
    if vel.shape[0] >= 2:
        acc_v = (vel[1:] - vel[:-1]) / dt
        acc = np.linalg.norm(acc_v, axis=1)
        dv = np.abs(speed[1:] - speed[:-1])
    else:
        acc = np.zeros((0,), dtype=np.float64)
        dv = np.zeros((0,), dtype=np.float64)
        acc_v = np.zeros((0, 3), dtype=np.float64)
    if acc_v.shape[0] >= 2:
        jerk_v = (acc_v[1:] - acc_v[:-1]) / dt
        jerk = np.linalg.norm(jerk_v, axis=1)
    else:
        jerk = np.zeros((0,), dtype=np.float64)
    return dict(speed=speed, acc=acc, jerk=jerk, dv=dv)


def _save_dynamics_plot(path, dyn, topk_idx):
    fig = plt.figure(figsize=(8.0, 5.4), constrained_layout=True)
    axs = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
    names = [("speed", "||dp/dt||"), ("acc", "||d2p/dt2||"), ("jerk", "||d3p/dt3||")]
    for ax, (k, ylab) in zip(axs, names):
        y = np.asarray(dyn.get(k, []), dtype=np.float64)
        x = np.arange(y.shape[0], dtype=int)
        ax.plot(x, y, linewidth=1.6)
        if y.shape[0] > 0:
            im = int(np.argmax(y))
            ax.scatter([im], [y[im]], s=24, c="crimson", zorder=3)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.2)
    # Mark top abrupt-change segments (based on |delta speed|)
    dv = np.asarray(dyn.get("dv", []), dtype=np.float64)
    if dv.shape[0] > 0:
        for i in topk_idx:
            ii = int(i)
            axs[0].axvspan(ii, ii + 1, color="orange", alpha=0.14)
    axs[-1].set_xlabel("t")
    axs[0].set_title("Dynamics Along Trajectory")
    fig.savefig(path, dpi=150)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] wrote", path)


def _parse_views(spec):
    views = []
    for part in str(spec).split(";"):
        s = part.strip()
        if not s:
            continue
        items = [t.strip() for t in s.split(",")]
        if len(items) != 2:
            raise SystemExit(f"Error: bad view spec '{s}', expected 'elev,azim'.")
        try:
            elev = float(items[0])
            azim = float(items[1])
        except Exception:
            raise SystemExit(f"Error: bad numeric view spec '{s}', expected 'elev,azim'.")
        views.append((elev, azim))
    if len(views) == 0:
        raise SystemExit("Error: --views resolved to empty list.")
    return views


def _parse_sphere_specs(specs):
    out = []
    for s in specs:
        ss = s.strip()
        if not ss:
            continue
        items = [t.strip() for t in ss.split(",")]
        if len(items) != 4:
            raise SystemExit(f"Error: bad --obst_sphere '{s}', expected 'x,y,z,r'.")
        try:
            x, y, z, r = [float(t) for t in items]
        except Exception:
            raise SystemExit(f"Error: bad numeric --obst_sphere '{s}', expected 'x,y,z,r'.")
        if r <= 0.0:
            raise SystemExit(f"Error: obstacle radius must be > 0, got {r}.")
        out.append((x, y, z, r))
    return out


def _default_view_labels(n_views):
    base = ["Main 3D", "Side View", "Front View", "Rear-High View", "Aux View"]
    if n_views <= len(base):
        return base[:n_views]
    out = base[:]
    while len(out) < n_views:
        out.append(f"View {len(out) + 1}")
    return out


def main():
    ap = argparse.ArgumentParser()
    # original mode (samples)
    ap.add_argument("--npz", default="", help="case_XXXX_samples.npz (contains cp). Required unless --traj_npz is used.")
    ap.add_argument("--sample", type=int, default=-1, help="-1 use best_idx (only for --npz mode)")

    # new mode (postprocessed single traj)
    ap.add_argument("--traj_npz", default="", help="npz containing traj_post/traj_raw/traj_arclen (e.g. eval_out/case0000_sid8_post.npz)")
    ap.add_argument("--traj_key", default="traj_post", choices=["traj_post", "traj_raw", "traj_arclen"],
                    help="which trajectory to load from --traj_npz")

    ap.add_argument("--out", default="", help="输出文件名；留空则自动生成不覆盖的名字")
    ap.add_argument("--stride", type=int, default=2, help="frame stride along time")
    ap.add_argument(
        "--play_param",
        choices=["index", "arclen", "time"],
        default="index",
        help=(
            "Animation playback parameterization. "
            "'index' plays by raw waypoint index (current behavior). "
            "'arclen' plays by approximately uniform arc length along the drawn UAV path. "
            "'time' plays by traj_time (if present in NPZ), so you can see slow-down near obstacles."
        ),
    )
    ap.add_argument("--plot_up", type=int, default=12,
                    help="Visualization-only: upsample trajectory points for smoother curve. 1 disables.")
    ap.add_argument("--plot_show_raw_polyline", dest="plot_show_raw_polyline", action="store_true",
                    help="Overlay raw polyline (thin, transparent) to diagnose spline overshoot.")
    ap.add_argument("--no_plot_show_raw_polyline", dest="plot_show_raw_polyline", action="store_false",
                    help="Disable raw polyline overlay.")
    # Default off: raw polyline can be visually confusing (esp. in 2D projections).
    ap.set_defaults(plot_show_raw_polyline=False)
    ap.add_argument("--plot_raw_alpha", type=float, default=0.25,
                    help="Alpha for raw polyline overlay when --plot_show_raw_polyline is enabled.")
    ap.add_argument("--plot_spline", dest="plot_spline", action="store_true",
                    help="Visualization-only: enable robust curve upsampling (bspline/catmullrom).")
    ap.add_argument("--no_plot_spline", dest="plot_spline", action="store_false",
                    help="Disable spline interpolation; use linear interpolation when upsampling trajectory.")
    ap.add_argument(
        "--plot_smooth",
        default="auto",
        help=(
            "Visualization-only: spline smoothing. "
            "Use 'auto' (default), '0' for exact interpolation, "
            "a numeric coefficient (scaled by m*median_step^2), "
            "or 'abs:<s>' for absolute splprep s."
        ),
    )
    ap.add_argument("--plot_debug", action="store_true",
                    help="Print debug info for plot upsampling and frame mapping (raw->upsample).")
    ap.add_argument("--draw_exec_setpoints", action="store_true",
                    help="Overlay executable uniform-dt setpoints from NPZ (traj_exec_xyz), if present.")
    ap.add_argument("--exec_color", default="#17becf",
                    help="Color for exec setpoints overlay (traj_exec_xyz).")
    ap.add_argument("--exec_alpha", type=float, default=0.25,
                    help="Alpha for exec setpoints overlay.")
    ap.add_argument("--exec_ms", type=float, default=1.8,
                    help="Marker size (points) for exec setpoints overlay.")
    ap.add_argument("--exec_every", type=int, default=5,
                    help="Subsample exec setpoints by this factor when plotting to avoid clutter.")
    ap.add_argument("--room", action="store_true", help="draw sanity box as data min/max")
    ap.add_argument("--elev", type=float, default=18.0, help="fixed camera elevation")
    ap.add_argument("--azim", type=float, default=30.0, help="fixed camera azimuth")
    ap.add_argument("--view_preset", default="baseline44k", choices=["none", "baseline44k"],
                    help="Preset for main camera. baseline44k is a close match to base visualization style.")
    ap.add_argument("--multiview", action="store_true", help="Render multiple fixed viewpoints in one GIF.")
    ap.add_argument("--singleview", dest="multiview", action="store_false",
                    help="Force single 3D view (optionally with top-down).")
    ap.add_argument("--views", default="22,-60;18,30;18,120",
                    help="Semicolon-separated 'elev,azim' pairs for multiview.")
    ap.add_argument("--topdown", dest="topdown", action="store_true",
                    help="Add XY top-down panel with map-style obstacle projection.")
    ap.add_argument("--no_topdown", dest="topdown", action="store_false",
                    help="Disable top-down panel.")
    ap.add_argument("--draw_obstacles", dest="draw_obstacles", action="store_true",
                    help="Draw obstacle spheres from npz obst_spheres (or --obst_sphere).")
    ap.add_argument("--hide_obstacles", dest="draw_obstacles", action="store_false",
                    help="Disable obstacle sphere drawing.")
    ap.add_argument("--obst_sphere", action="append", default=[],
                    help="extra obstacle sphere 'x,y,z,r' (can repeat).")
    ap.add_argument("--obst_alpha", type=float, default=0.14, help="Obstacle sphere alpha.")
    ap.add_argument("--obst_edge_color", default="#8b005d", help="Obstacle edge color.")
    ap.add_argument("--obst_color", default="tab:red", help="Obstacle sphere color.")
    ap.add_argument("--obst_vis_radius_scale", type=float, default=0.85,
                    help="Visualization-only radius scale for obstacle drawing.")
    ap.add_argument("--draw_boxes", dest="draw_boxes", action="store_true",
                    help="Draw AABB obstacles from npz obst_boxes (if available).")
    ap.add_argument("--hide_boxes", dest="draw_boxes", action="store_false",
                    help="Disable AABB box drawing.")
    ap.add_argument("--box_color", default="#c2185b", help="Obstacle box color.")
    ap.add_argument("--box_edge_color", default="#5a003e", help="Obstacle box edge color.")
    ap.add_argument("--box_alpha", type=float, default=0.22, help="Obstacle box alpha.")
    ap.add_argument("--show_box_sphere_proxy", action="store_true",
                    help="Also draw proxy spheres that were generated from boxes.")
    ap.add_argument("--collision_threshold", type=float, default=0.0,
                    help="Clearance threshold for collision highlighting (meters).")
    ap.add_argument("--collision_color", default="crimson",
                    help="Color for trajectory segments/points under collision threshold.")
    ap.add_argument("--unsafe_color", default="darkorange",
                    help="Color for trajectory segments/points below safe margin (but not collision).")
    ap.add_argument("--uav_radius", type=float, default=0.0,
                    help="UAV safety radius (meters), used for collision/clearance metrics.")
    ap.add_argument("--safe_margin", type=float, default=0.0,
                    help="Extra safety margin beyond UAV radius (meters).")

    ap.add_argument("--draw_ee_path", dest="draw_ee_path", action="store_true",
                    help="Draw end-effector (EE) trajectory in addition to UAV path.")
    ap.add_argument("--no_draw_ee_path", dest="draw_ee_path", action="store_false",
                    help="Disable EE trajectory drawing.")
    ap.add_argument("--ee_debug", action="store_true",
                    help="Print EE extraction debug: chosen source, keys, and uav/ee xyz ranges.")
    ap.add_argument("--ee_affect_limits", action="store_true",
                    help="Include EE path in axis limits. Default: EE does NOT affect axis limits.")
    ap.add_argument("--ee_urdf", default="/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf",
                    help="URDF for Pinocchio FK (EE path). Used only when EE is not present in the NPZ.")
    ap.add_argument("--ee_frame", default="gripper_link", help="EE frame name for Pinocchio FK.")
    ap.add_argument("--ee_color", default="#2ca02c",
                    help="EE path color. Default matches Matplotlib's 'tab:green'.")
    ap.add_argument("--ee_alpha", type=float, default=0.95, help="EE path alpha.")
    ap.add_argument("--ee_lw", type=float, default=2.2, help="EE path line width.")
    ap.add_argument("--ee_ls", default="--", help="EE path line style (e.g. '--').")

    ap.add_argument("--show_collision", dest="show_collision", action="store_true",
                    help="Highlight trajectory points where clearance <= collision_threshold.")
    ap.add_argument("--hide_collision", dest="show_collision", action="store_false",
                    help="Disable collision highlighting.")
    ap.add_argument("--clearance_png", default="",
                    help="If set, save min-clearance-vs-time plot PNG to this path.")
    ap.add_argument("--dyn_png", default="",
                    help="If set, save speed/acc/jerk diagnostics PNG to this path.")
    ap.add_argument("--state_png", default="",
                    help="If set, also save a full-state debug PNG (pos/vel/att/omega/clearance/grasp).")
    ap.add_argument("--dump_state", action="store_true",
                    help="Convenience: auto-save state PNG to /tmp/<out_base>_state.png.")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="Time step for dynamic metrics (speed/acc/jerk).")
    ap.add_argument("--abrupt_topk", type=int, default=2,
                    help="Report top-k abrupt segments by |delta speed|.")
    ap.add_argument("--show_argmin", dest="show_argmin", action="store_true",
                    help="Highlight the argmin_t point of clearance in all views.")
    ap.add_argument("--hide_argmin", dest="show_argmin", action="store_false",
                    help="Disable argmin_t marker.")
    ap.add_argument("--status_level", default="brief", choices=["none", "brief", "full"],
                    help="Overlay verbosity: none/brief/full. Keep titles compact and move details to status box.")
    ap.add_argument("--metrics_txt", default="",
                    help="Optional path to write full safety/dynamics metrics as plain text.")
    ap.add_argument("--tick_fontsize", type=float, default=7.0,
                    help="Axis tick label fontsize for all panels.")
    ap.add_argument("--axis_label_fontsize", type=float, default=9.0,
                    help="Axis label fontsize for all panels.")
    ap.add_argument("--fps", type=int, default=20)
    ap.set_defaults(draw_obstacles=True, draw_boxes=True, topdown=True, show_collision=True, show_argmin=False,
                    plot_spline=True, multiview=True, draw_ee_path=True)
    args, extra = ap.parse_known_args()

    # Prefer a CJK-capable font so Chinese titles render correctly (if available).
    # Noto Sans CJK is commonly present on Linux images; fall back silently otherwise.
    try:
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    bad_flags = [x for x in extra if str(x).startswith("-")]
    if len(bad_flags) > 0:
        raise SystemExit(f"Error: unrecognized arguments: {' '.join(bad_flags)}")

    # Robust handling: allow unquoted wildcard expansion with multiple NPZ args.
    if not args.traj_npz:
        args.npz = _resolve_npz_arg(args.npz, extra)
    elif len(extra) > 0:
        print(f"[WARN] ignoring extra positional args in --traj_npz mode: {extra}")

    use_traj_npz = bool(args.traj_npz)
    if (not use_traj_npz) and (not args.npz):
        raise SystemExit("Error: must provide --npz (samples) OR --traj_npz (postprocessed traj).")

    # --------------------------
    # Load data
    # --------------------------
    sid = 0
    best_idx = 0
    best_idx_selection = 0
    best_idx_grasp = 0
    post_proj_applied = -1
    saved_obst_hash = ""
    meta_txt = ""
    obst_xy_box = None

    obst_spheres = []
    obst_sphere_from_box = np.zeros((0,), dtype=np.int64)
    obst_boxes = []
    traj_time_raw = None
    if use_traj_npz:
        d = np.load(args.traj_npz, allow_pickle=True)
        if args.traj_key not in d:
            raise SystemExit(f"Error: {args.traj_key} not found in {args.traj_npz}. Keys={list(d.keys())}")

        traj = d[args.traj_key]   # (H, D)
        uav_xyz_raw = traj[:, :3]
        H = uav_xyz_raw.shape[0]
        if "traj_time" in d:
            try:
                tt = np.asarray(d["traj_time"], dtype=np.float64).reshape(-1)
                if tt.shape[0] == H:
                    traj_time_raw = tt
            except Exception:
                traj_time_raw = None

        # optional meta
        sid = int(d["sid"]) if "sid" in d else 0
        tg = int(d["tg"]) if "tg" in d else None
        src = str(d["source_npz"]) if "source_npz" in d else ""
        meta_txt = f"{args.traj_key}"
        if tg is not None:
            meta_txt += f" tg={tg}"
        if src:
            meta_txt += f" src={src.split('/')[-1]}"

        # markers: try from post npz first
        q_start = d["q_start"] if "q_start" in d else None
        q_goal  = d["q_goal"]  if "q_goal"  in d else None
        q_grasp = d["q_grasp"] if "q_grasp" in d else None

        # FALLBACK: load markers from source_npz (original samples)
        if ("source_npz" in d) and (q_grasp is None or q_start is None or q_goal is None):
            try:
                src_path = str(d["source_npz"])
                ds = np.load(src_path, allow_pickle=True)
                if q_start is None and "q_start" in ds:
                    q_start = ds["q_start"]
                if q_goal is None and "q_goal" in ds:
                    q_goal = ds["q_goal"]
                if q_grasp is None and "q_grasp" in ds:
                    q_grasp = ds["q_grasp"]
                if "obst_spheres" in ds:
                    obst_spheres = np.asarray(ds["obst_spheres"], dtype=float).reshape(-1, 4).tolist()
                if "obst_sphere_from_box" in ds:
                    obst_sphere_from_box = np.asarray(ds["obst_sphere_from_box"], dtype=np.int64).reshape(-1)
                if "obst_boxes" in ds:
                    obst_boxes = np.asarray(ds["obst_boxes"], dtype=float).reshape(-1, 6).tolist()
                if ("obst_random_xyz_min" in ds) and ("obst_random_xyz_max" in ds):
                    mn2 = np.asarray(ds["obst_random_xyz_min"], dtype=float).reshape(3)
                    mx2 = np.asarray(ds["obst_random_xyz_max"], dtype=float).reshape(3)
                    obst_xy_box = (mn2[:2], mx2[:2])
            except Exception as e:
                print("[WARN] failed to load markers from source_npz:", e)

        if "obst_spheres" in d:
            obst_spheres = np.asarray(d["obst_spheres"], dtype=float).reshape(-1, 4).tolist()
        if "obst_sphere_from_box" in d:
            obst_sphere_from_box = np.asarray(d["obst_sphere_from_box"], dtype=np.int64).reshape(-1)
        if "obst_boxes" in d:
            obst_boxes = np.asarray(d["obst_boxes"], dtype=float).reshape(-1, 6).tolist()
        if ("obst_random_xyz_min" in d) and ("obst_random_xyz_max" in d):
            mn2 = np.asarray(d["obst_random_xyz_min"], dtype=float).reshape(3)
            mx2 = np.asarray(d["obst_random_xyz_max"], dtype=float).reshape(3)
            obst_xy_box = (mn2[:2], mx2[:2])

        # auto output name
        if not args.out:
            ts = int(time.time())
            base = args.traj_npz.split("/")[-1].replace(".npz", "")
            args.out = f"{base}_{args.traj_key}_stride{args.stride}_{ts}.gif"

        # room box (optional): use this traj only
        if args.room:
            mn = uav_xyz_raw.min(axis=0)
            mx = uav_xyz_raw.max(axis=0)
        else:
            mn = mx = None

    else:
        d = np.load(args.npz, allow_pickle=True)
        cp = d["cp"]              # [S,H,9]
        q_start = d["q_start"]    # [9]
        q_start = d["q_start"]    # [9]
        q_goal = d["q_goal"]      # [9]
        q_grasp = d["q_grasp"]    # [3]
        if "obst_spheres" in d:
            obst_spheres = np.asarray(d["obst_spheres"], dtype=float).reshape(-1, 4).tolist()
        if "obst_sphere_from_box" in d:
            obst_sphere_from_box = np.asarray(d["obst_sphere_from_box"], dtype=np.int64).reshape(-1)
        if "obst_boxes" in d:
            obst_boxes = np.asarray(d["obst_boxes"], dtype=float).reshape(-1, 6).tolist()
        if ("obst_random_xyz_min" in d) and ("obst_random_xyz_max" in d):
            mn2 = np.asarray(d["obst_random_xyz_min"], dtype=float).reshape(3)
            mx2 = np.asarray(d["obst_random_xyz_max"], dtype=float).reshape(3)
            obst_xy_box = (mn2[:2], mx2[:2])
        best_idx = int(d["best_idx"][0]) if "best_idx" in d else 0
        best_idx_selection = int(d["best_idx_selection"][0]) if "best_idx_selection" in d else int(best_idx)
        best_idx_grasp = int(d["best_idx_grasp"][0]) if "best_idx_grasp" in d else int(best_idx)
        if "post_proj_applied" in d:
            try:
                post_proj_applied = int(np.asarray(d["post_proj_applied"]).reshape(-1)[0])
            except Exception:
                post_proj_applied = -1
        if "obst_hash" in d:
            try:
                saved_obst_hash = str(np.asarray(d["obst_hash"]).reshape(-1)[0])
            except Exception:
                saved_obst_hash = ""

        sid0 = best_idx_selection if args.sample < 0 else args.sample
        sid = int(sid0)
        sid = max(0, min(sid, cp.shape[0] - 1))
        traj = cp[sid]            # [H,9]
        uav_xyz_raw = traj[:, :3]
        H = uav_xyz_raw.shape[0]
        if "traj_time" in d:
            try:
                tt = np.asarray(d["traj_time"], dtype=np.float64)
                if tt.ndim == 1 and tt.shape[0] == H:
                    traj_time_raw = tt.reshape(-1)
                elif tt.ndim == 2 and tt.shape[0] == int(cp.shape[0]) and tt.shape[1] == H:
                    traj_time_raw = tt[sid].reshape(-1)
            except Exception:
                traj_time_raw = None

        if not args.out:
            ts = int(time.time())
            base = args.npz.split("/")[-1].replace(".npz", "")
            args.out = f"{base}_sid{sid}_stride{args.stride}_{ts}.gif"

        if args.room:
            xyz_all = cp[:, :, :3].reshape(-1, 3)
            mn = xyz_all.min(axis=0)
            mx = xyz_all.max(axis=0)
        else:
            mn = mx = None

    print(
        "[DRAW_META] "
        f"traj_key={str(args.traj_key) if use_traj_npz else 'traj_post'} "
        f"sid={int(sid)} best_idx={int(best_idx)} best_idx_selection={int(best_idx_selection)} best_idx_grasp={int(best_idx_grasp)} "
        f"post_proj_applied={int(post_proj_applied)} obst_hash={saved_obst_hash if saved_obst_hash else '<none>'} "
        f"H={int(H)}"
    )

    if args.plot_debug:
        print(
            "[DRAW] "
            f"use_traj_npz={int(use_traj_npz)} "
            f"traj_key={str(args.traj_key) if use_traj_npz else 'traj_post'} "
            f"sid={int(sid)} best_idx={int(best_idx)} best_idx_selection={int(best_idx_selection)} best_idx_grasp={int(best_idx_grasp)} "
            f"post_proj_applied={int(post_proj_applied)} obst_hash={saved_obst_hash if saved_obst_hash else '<none>'} "
            f"H={int(H)} plot_up={int(args.plot_up)} play_param={str(args.play_param)}"
        )
        if traj_time_raw is None:
            print(f"[TIME] sid={int(sid)} traj_time=NONE")
        else:
            tt = np.asarray(traj_time_raw, dtype=np.float64).reshape(-1)
            if tt.size >= 2:
                dtt = np.diff(tt)
                mono_strict = bool(np.all(dtt > 0.0))
                mono_nd = bool(np.all(dtt >= 0.0))
            else:
                mono_strict = True
                mono_nd = True
            t_end = float(tt[-1] - tt[0]) if tt.size else 0.0
            print(
                "[TIME] "
                f"sid={int(sid)} "
                f"shape={tuple(tt.shape)} "
                f"t_end={t_end:.6f} "
                f"monotonic_strict={int(mono_strict)} "
                f"monotonic_nondecr={int(mono_nd)}"
            )

    # --------------------------
    # Optional: EE trajectory for plotting
    # --------------------------
    ee_xyz_raw = None
    ee_source = ""
    if bool(getattr(args, "draw_ee_path", True)):
        ee_dbg = bool(getattr(args, "ee_debug", False) or bool(getattr(args, "plot_debug", False)))

        def _resample_xyz_to_H(xyz_in: np.ndarray, H_out: int):
            xyz_in = np.asarray(xyz_in, dtype=np.float64)
            if xyz_in.ndim == 1:
                xyz_in = xyz_in.reshape(-1, 3)
            else:
                xyz_in = xyz_in.reshape(-1, xyz_in.shape[-1])
            if xyz_in.shape[1] < 3:
                raise ValueError(f"bad xyz shape: {xyz_in.shape}")
            xyz_in = xyz_in[:, :3]
            H_out = int(H_out)
            if xyz_in.shape[0] == H_out:
                return xyz_in
            if xyz_in.shape[0] <= 1 or H_out <= 1:
                return np.repeat(xyz_in[:1], repeats=max(1, H_out), axis=0)
            t_old = np.linspace(0.0, 1.0, int(xyz_in.shape[0]), dtype=np.float64)
            t_new = np.linspace(0.0, 1.0, H_out, dtype=np.float64)
            return np.stack([np.interp(t_new, t_old, xyz_in[:, k]) for k in range(3)], axis=1).astype(np.float64)

        # 1) Prefer explicit EE position keys if present.
        if isinstance(d, np.lib.npyio.NpzFile):
            for k in [
                "ee_positions_resampled",
                "ee_positions",
                "ee_traj",
                "ee_xyz",
                "traj_ee",
                "ee_path",
                "ee_pos",
                "xyz_ee",
            ]:
                if k not in d:
                    continue
                try:
                    ee_xyz_raw = _resample_xyz_to_H(d[k], H)
                    ee_source = f"npz:{k}"
                    break
                except Exception as e:
                    print(f"[EE_PLOT] bad key '{k}': {_short_exc(e)}")
                    ee_xyz_raw = None
                    ee_source = ""

        # 2) FK via pinocchio for EagleGrasp state: [x,y,z,qx,qy,qz,qw,j1,j2]
        if ee_xyz_raw is None:
            try:
                import pinocchio as pin  # type: ignore

                URDF_PATH = str(getattr(args, "ee_urdf", "")).strip()
                EE_FRAME = str(getattr(args, "ee_frame", "gripper_link")).strip()

                if not os.path.exists(URDF_PATH):
                    raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

                def _quat_norm_xyzw(q):
                    q = np.asarray(q, dtype=np.float64).reshape(4)
                    n = float(np.linalg.norm(q))
                    if (not np.isfinite(n)) or n < 1e-12:
                        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    return q / n

                q_all = np.asarray(traj, dtype=np.float64).reshape(H, -1)
                if q_all.shape[1] < 9:
                    raise ValueError(f"traj has too few dims for pinocchio FK: {q_all.shape}")
                q_all = q_all[:, :9]

                model_fk = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
                data_fk = model_fk.createData()
                fid = model_fk.getFrameId(EE_FRAME)
                ee_list = []
                for t in range(H):
                    q9 = q_all[t].copy()
                    q9[3:7] = _quat_norm_xyzw(q9[3:7])
                    pin.forwardKinematics(model_fk, data_fk, q9)
                    pin.updateFramePlacements(model_fk, data_fk)
                    ee_list.append(np.asarray(data_fk.oMf[fid].translation, dtype=np.float64).copy())
                ee_xyz_raw = np.stack(ee_list, axis=0).astype(np.float64)
                ee_source = f"pinocchio:{EE_FRAME}"
            except Exception as e:
                ee_xyz_raw = None
                ee_source = ""
                if ee_dbg:
                    print(f"[EE_PLOT] pinocchio FK failed: {_short_exc(e)}")

        if ee_xyz_raw is None:
            if ee_dbg and isinstance(d, np.lib.npyio.NpzFile):
                print("[NPZ_KEYS]", list(d.files))
            print("[EE_PLOT] not found; skip drawing ee path.")
        else:
            if ee_dbg:
                uav_mn = np.asarray(uav_xyz_raw, dtype=float).reshape(-1, 3).min(axis=0)
                uav_mx = np.asarray(uav_xyz_raw, dtype=float).reshape(-1, 3).max(axis=0)
                ee_mn = np.asarray(ee_xyz_raw, dtype=float).reshape(-1, 3).min(axis=0)
                ee_mx = np.asarray(ee_xyz_raw, dtype=float).reshape(-1, 3).max(axis=0)
                print(
                    "[EE_PLOT] "
                    f"source={ee_source} "
                    f"ee_xyz_raw_shape={tuple(ee_xyz_raw.shape)} "
                    f"uav_min={uav_mn.round(6).tolist()} uav_max={uav_mx.round(6).tolist()} "
                    f"ee_min={ee_mn.round(6).tolist()} ee_max={ee_mx.round(6).tolist()} "
                    f"ee_affect_limits={int(bool(getattr(args, 'ee_affect_limits', False)))}"
                )
            else:
                print(f"[EE_PLOT] source={ee_source} ee_xyz_raw_shape={tuple(ee_xyz_raw.shape)}")

    # Hash check BEFORE adding any extra CLI spheres (those are visualization-only).
    if isinstance(d, np.lib.npyio.NpzFile):
        saved_hash = None
        if "obst_hash" in d:
            try:
                saved_hash = str(np.asarray(d["obst_hash"]).reshape(-1)[0])
            except Exception:
                saved_hash = None
        try:
            computed_hash = _obst_hash_sha1(obst_spheres, obst_boxes)
            if saved_hash:
                print(f"[OBST_HASH] saved={saved_hash} computed={computed_hash} match={int(saved_hash == computed_hash)}")
            else:
                print(f"[OBST_HASH] saved=<none> computed={computed_hash}")
        except Exception as e:
            print(f"[OBST_HASH] failed to compute hash: {_short_exc(e)}")

    extra_spheres = _parse_sphere_specs(args.obst_sphere)
    obst_spheres.extend(extra_spheres)
    if len(extra_spheres) > 0:
        if obst_sphere_from_box.size == 0:
            obst_sphere_from_box = np.zeros((len(extra_spheres),), dtype=np.int64)
        else:
            obst_sphere_from_box = np.concatenate(
                [obst_sphere_from_box.reshape(-1), np.zeros((len(extra_spheres),), dtype=np.int64)],
                axis=0,
            )
    if obst_sphere_from_box.size != len(obst_spheres):
        obst_sphere_from_box = np.zeros((len(obst_spheres),), dtype=np.int64)

    draw_sphere_mask = np.ones((len(obst_spheres),), dtype=bool)
    if args.draw_boxes and len(obst_boxes) > 0 and (not args.show_box_sphere_proxy):
        draw_sphere_mask = (obst_sphere_from_box.reshape(-1) == 0)
    # Debug-friendly summary: this script does NOT resample/reconstruct obstacles; it visualizes NPZ obstacles as-is.
    n_sph_total = int(len(obst_spheres))
    n_sph_draw = int(np.sum(draw_sphere_mask)) if draw_sphere_mask.size > 0 else 0
    n_box_total = int(len(obst_boxes))
    print(
        "[PLOT_OBST] "
        f"spheres_total={n_sph_total} spheres_draw={n_sph_draw} "
        f"boxes_total={n_box_total} "
        f"draw_obstacles={int(bool(args.draw_obstacles))} draw_boxes={int(bool(args.draw_boxes))} "
        f"show_box_sphere_proxy={int(bool(args.show_box_sphere_proxy))}"
    )
    # If not explicitly set, inherit safety settings from eval npz metadata when available.
    if float(args.uav_radius) <= 0.0 and isinstance(d, np.lib.npyio.NpzFile) and ("obst_uav_radius" in d):
        try:
            args.uav_radius = float(np.asarray(d["obst_uav_radius"], dtype=float).reshape(-1)[0])
        except Exception:
            pass
    if float(args.safe_margin) <= 0.0 and isinstance(d, np.lib.npyio.NpzFile):
        try:
            # Prefer the actual selection-safe margin used by eval (closest to "unsafe highlight" meaning).
            if "obst_safe_margin" in d:
                args.safe_margin = float(np.asarray(d["obst_safe_margin"], dtype=float).reshape(-1)[0])
            elif "obst_proj_margin" in d:
                args.safe_margin = float(np.asarray(d["obst_proj_margin"], dtype=float).reshape(-1)[0])
            elif "obst_margin" in d:
                args.safe_margin = float(np.asarray(d["obst_margin"], dtype=float).reshape(-1)[0])
        except Exception:
            pass

    plot_up = int(max(1, args.plot_up))
    if bool(args.plot_spline) and (not _HAVE_SCIPY):
        print("[WARN] scipy not available; bspline disabled, will use catmullrom/linear fallback for plotting.")
    uav_xyz_draw, _uav_idx_map_draw, _uav_plot_info = _upsample_xyz_for_plot(
        uav_xyz_raw,
        up=plot_up,
        use_spline=bool(args.plot_spline),
        smooth=str(args.plot_smooth),
        debug=bool(args.plot_debug),
    )

    # Executable setpoints overlay (uniform-dt in time). This helps visually verify the speed profile:
    # denser points imply slower motion (smaller ds per control step).
    exec_xyz_plot = None
    exec_dt = None
    if bool(getattr(args, "draw_exec_setpoints", False)) and isinstance(d, np.lib.npyio.NpzFile) and ("traj_exec_xyz" in d):
        try:
            ex_all = np.asarray(d["traj_exec_xyz"], dtype=np.float64)
            if ex_all.ndim == 2:
                ex = ex_all.reshape(-1, 3)
            else:
                ex = ex_all[int(sid)]
                ex = np.asarray(ex, dtype=np.float64).reshape(-1, 3)
            if "traj_exec_len" in d and ex_all.ndim >= 3:
                n_ex = int(np.asarray(d["traj_exec_len"], dtype=np.int64).reshape(-1)[int(sid)])
                n_ex = int(max(0, min(int(ex.shape[0]), n_ex)))
                if n_ex > 0:
                    ex = ex[:n_ex]
            every = int(max(1, int(getattr(args, "exec_every", 5))))
            exec_xyz_plot = ex[::every].copy()
            if "traj_exec_dt" in d:
                exec_dt = float(np.asarray(d["traj_exec_dt"], dtype=float).reshape(-1)[0])
            if args.plot_debug:
                print(
                    "[EXEC_PLOT] "
                    f"sid={int(sid)} "
                    f"n_exec={int(ex.shape[0])} "
                    f"every={every} plotted={int(exec_xyz_plot.shape[0])} "
                    f"dt_ctrl={exec_dt if exec_dt is not None else float('nan'):.6g}"
                )
        except Exception as e:
            exec_xyz_plot = None
            if args.plot_debug:
                print(f"[WARN] exec setpoints overlay failed: {_short_exc(e)}")
    ee_xyz_draw = None
    if bool(getattr(args, "draw_ee_path", True)) and (ee_xyz_raw is not None):
        ee_xyz_draw, _ee_idx_map_draw, _ee_plot_info = _upsample_xyz_for_plot(
            ee_xyz_raw,
            up=plot_up,
            use_spline=bool(args.plot_spline),
            smooth=str(args.plot_smooth),
            debug=bool(args.plot_debug),
        )
    sd_surface_pt, sd_surface_seg = _compute_clearance_curves(uav_xyz_raw, obst_spheres)
    uav_r = float(max(0.0, args.uav_radius))
    safe_margin = float(max(0.0, args.safe_margin))
    clearance = sd_surface_pt - uav_r
    # Keep old behavior: point-wise collision/unsafe masks.
    coll_mask = clearance <= float(args.collision_threshold)
    unsafe_mask = (clearance < safe_margin) & (~coll_mask)
    d_safe = uav_r + safe_margin
    min_sd = float(np.min(sd_surface_pt))
    min_clr = float(np.min(clearance))
    coll_intervals = _mask_intervals(coll_mask)
    unsafe_intervals = _mask_intervals(unsafe_mask)
    coll_len = int(sum((b - a + 1) for a, b in coll_intervals))
    unsafe_len = int(sum((b - a + 1) for a, b in unsafe_intervals))
    print(
        f"[SAFE] uav_radius={uav_r:.4f} safe_margin={safe_margin:.4f} d_safe={d_safe:.4f} "
        f"collision_any={bool(np.any(coll_mask))} "
        f"min_signed_distance={min_sd:.6f} min_clearance={min_clr:.6f} "
        f"coll_pts={int(np.sum(coll_mask))} unsafe_pts={int(np.sum(unsafe_mask))} "
        f"coll_len={coll_len} unsafe_len={unsafe_len}"
    )
    if len(coll_intervals) > 0:
        print(f"[SAFE] collision_intervals={coll_intervals}")
    if len(unsafe_intervals) > 0:
        print(f"[SAFE] unsafe_intervals={unsafe_intervals}")
    has_collision = bool(np.any(coll_mask))
    has_unsafe = bool(np.any(unsafe_mask))
    argmin_t = int(np.argmin(clearance))
    argmin_xyz = uav_xyz_raw[argmin_t]
    print(
        f"[SAFE_BOOL] has_collision={has_collision} has_unsafe={has_unsafe} "
        f"argmin_t={argmin_t} "
        f"min_signed_dist={min_sd:.6f} min_clearance={min_clr:.6f}"
    )

    dyn = _compute_dynamics_metrics(uav_xyz_raw, dt=float(args.dt))
    sp = np.asarray(dyn["speed"], dtype=np.float64)
    ac = np.asarray(dyn["acc"], dtype=np.float64)
    jk = np.asarray(dyn["jerk"], dtype=np.float64)
    dv = np.asarray(dyn["dv"], dtype=np.float64)
    if sp.size > 0:
        sp_max_i = int(np.argmax(sp))
        print(f"[DYN] speed mean/max={float(np.mean(sp)):.6f}/{float(np.max(sp)):.6f} at t={sp_max_i}")
    if ac.size > 0:
        ac_max_i = int(np.argmax(ac)) + 1
        print(f"[DYN] acc   mean/max={float(np.mean(ac)):.6f}/{float(np.max(ac)):.6f} at t={ac_max_i}")
    if jk.size > 0:
        jk_max_i = int(np.argmax(jk)) + 2
        print(f"[DYN] jerk  mean/max={float(np.mean(jk)):.6f}/{float(np.max(jk)):.6f} at t={jk_max_i}")
    abrupt_idx = np.array([], dtype=np.int64)
    if dv.size > 0:
        k = int(max(0, args.abrupt_topk))
        if k > 0:
            order = np.argsort(-dv)
            abrupt_idx = order[: min(k, order.size)].astype(np.int64)
            segs = [(int(i), int(i + 1), float(dv[int(i)])) for i in abrupt_idx]
            print(f"[DYN] abrupt_segments_by_dv={segs}")
    else:
        segs = []

    if args.metrics_txt:
        with open(args.metrics_txt, "w", encoding="utf-8") as f:
            f.write(f"sample={sid} best_idx={best_idx} H={H}\n")
            f.write(
                f"has_collision={has_collision} has_unsafe={has_unsafe} argmin_t={argmin_t} "
                f"uav_radius={uav_r:.6f} safe_margin={safe_margin:.6f} d_safe={d_safe:.6f}\n"
            )
            f.write(
                f"min_signed_distance={min_sd:.6f} min_clearance={min_clr:.6f} "
                f"unsafe_len={unsafe_len} coll_len={coll_len}\n"
            )
            f.write("collision_condition=(point_clearance <= collision_threshold), "
                    "point_clearance = dist(point,sphere_surface) - uav_radius\n")
            f.write(f"unsafe_intervals={unsafe_intervals}\n")
            f.write(f"collision_intervals={coll_intervals}\n")
            if sp.size > 0:
                f.write(f"speed_mean={float(np.mean(sp)):.6f} speed_max={float(np.max(sp)):.6f}\n")
            if ac.size > 0:
                f.write(f"acc_mean={float(np.mean(ac)):.6f} acc_max={float(np.max(ac)):.6f}\n")
            if jk.size > 0:
                f.write(f"jerk_mean={float(np.mean(jk)):.6f} jerk_max={float(np.max(jk)):.6f}\n")
            f.write(f"abrupt_segments_by_dv={segs}\n")
        print("[OK] wrote", args.metrics_txt)

    if args.multiview:
        views = _parse_views(args.views)
    else:
        if args.view_preset == "baseline44k":
            views = [(22.0, -60.0)]
        else:
            views = [(float(args.elev), float(args.azim))]

    # --------------------------
    # Plot setup
    # --------------------------
    n_views = len(views)
    n_panels = n_views + (1 if args.topdown else 0)

    # Fixed multiview layout (2x2): Front / Left / Top / Main.
    use_fixed_multiview_layout = bool(args.multiview and args.topdown and n_views == 3 and n_panels == 4)
    if use_fixed_multiview_layout:
        # If user didn't customize --views (kept the default string), use the canonical engineering layout:
        #   Front: (18,120), Left: (18,30), Main: (22,-60).
        DEFAULT_VIEWS_STR = "22,-60;18,30;18,120"
        if str(args.views).strip() == DEFAULT_VIEWS_STR:
            views = [(18.0, 120.0), (18.0, 30.0), (22.0, -60.0)]
        # If user DID provide --views explicitly, interpret them as [Front, Left, Main] order.
        # Apply view preset to the Main (isometric) panel, not the Front panel.
        if args.view_preset == "baseline44k" and len(views) >= 3:
            views[2] = (22.0, -60.0)
    else:
        # Backward compatible behavior: for generic multiview grids, preset applies to the first 3D axis.
        if args.multiview and args.view_preset == "baseline44k" and len(views) > 0:
            views[0] = (22.0, -60.0)
    if n_panels == 1:
        fig = plt.figure(figsize=(6.8, 5.2), constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14)
    else:
        n_cols = 2
        n_rows = int(math.ceil(n_panels / float(n_cols)))
        fig = plt.figure(figsize=(6.8 * n_cols, 5.0 * n_rows), constrained_layout=False)
        fig.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.14, wspace=0.12, hspace=0.22)

    axes = []
    ax_top = None
    ax_main = None  # used for legend placement in fixed multiview
    if use_fixed_multiview_layout:
        # 2x2 fixed: (1) Front, (2) Left, (3) Top-Down, (4) Main isometric.
        ax_front = fig.add_subplot(2, 2, 1)
        ax_left = fig.add_subplot(2, 2, 2)
        ax_top = fig.add_subplot(2, 2, 3)
        ax_main = fig.add_subplot(2, 2, 4, projection="3d")
        axes = [ax_main]
    else:
        if n_panels == 1:
            axes.append(fig.add_subplot(111, projection="3d"))
        else:
            n_cols = 2
            n_rows = int(math.ceil(n_panels / float(n_cols)))
            for i in range(n_views):
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, projection="3d"))
        if args.topdown:
            if n_panels == 1:
                ax_top = fig.add_subplot(111)
            else:
                n_cols = 2
                n_rows = int(math.ceil(n_panels / float(n_cols)))
                ax_top = fig.add_subplot(n_rows, n_cols, n_views + 1)

    # stable limits (include obstacles so views are spatially faithful)
    pad = 0.05
    mn_lim = uav_xyz_raw.min(axis=0)
    mx_lim = uav_xyz_raw.max(axis=0)
    # By default, EE does NOT affect axis limits (to avoid confusing "obstacles moved" impressions when EE is mis-read).
    if bool(getattr(args, "ee_affect_limits", False)) and (ee_xyz_raw is not None):
        try:
            mn_lim = np.minimum(mn_lim, np.asarray(ee_xyz_raw, dtype=float).reshape(-1, 3).min(axis=0))
            mx_lim = np.maximum(mx_lim, np.asarray(ee_xyz_raw, dtype=float).reshape(-1, 3).max(axis=0))
        except Exception:
            pass
    if args.draw_obstacles and len(obst_spheres) > 0:
        sph_draw = [obst_spheres[i] for i in range(len(obst_spheres)) if draw_sphere_mask[i]]
        if len(sph_draw) > 0:
            cc = np.asarray([[float(x), float(y), float(z)] for x, y, z, _ in sph_draw], dtype=float)
            rr = np.asarray([float(r) for _, _, _, r in sph_draw], dtype=float).reshape(-1, 1)
            mn_obs = np.min(cc - rr, axis=0)
            mx_obs = np.max(cc + rr, axis=0)
            mn_lim = np.minimum(mn_lim, mn_obs)
            mx_lim = np.maximum(mx_lim, mx_obs)
    if args.draw_boxes and len(obst_boxes) > 0:
        bb = np.asarray(obst_boxes, dtype=float).reshape(-1, 6)
        c = bb[:, :3]
        h = bb[:, 3:]
        mn_box = np.min(c - h, axis=0)
        mx_box = np.max(c + h, axis=0)
        mn_lim = np.minimum(mn_lim, mn_box)
        mx_lim = np.maximum(mx_lim, mx_box)
    rng = np.maximum(mx_lim - mn_lim, 1e-6)
    mn_lim = mn_lim - pad * rng
    mx_lim = mx_lim + pad * rng
    # Use cube limits for 3D readability (avoid skewed/stretchy perspective across panels).
    ctr = 0.5 * (mn_lim + mx_lim)
    half = 0.5 * float(np.max(mx_lim - mn_lim))
    half = max(1e-3, 1.04 * half)
    mn_cube = ctr - half
    mx_cube = ctr + half
    raw_lines = []
    lines = []
    ee_lines = []
    coll_lines = []
    coll_points = []
    unsafe_lines = []
    unsafe_points = []
    views_3d = list(views)
    if use_fixed_multiview_layout:
        # Only the isometric/Main panel is a 3D axis now.
        views_3d = [views[2]]
    if use_fixed_multiview_layout:
        view_labels = ["主视角 (Main 3D)"]
    else:
        view_labels = _default_view_labels(n_views)
    rx = max(1e-6, float(mx_cube[0] - mn_cube[0]))
    ry = max(1e-6, float(mx_cube[1] - mn_cube[1]))
    rz = max(1e-6, float(mx_cube[2] - mn_cube[2]))
    rvis_scale = float(max(0.05, args.obst_vis_radius_scale))
    for ax_i, (ev, az) in zip(axes, views_3d):
        ax_i.view_init(elev=ev, azim=az)
        if hasattr(ax_i, "set_proj_type"):
            try:
                ax_i.set_proj_type("ortho")
            except Exception:
                pass

        # markers
        if q_start is not None:
            ax_i.scatter([q_start[0]], [q_start[1]], [q_start[2]], marker="o", label="start(uav)")
        else:
            ax_i.scatter([uav_xyz_raw[0, 0]], [uav_xyz_raw[0, 1]], [uav_xyz_raw[0, 2]], marker="o", label="start(uav)")


        if q_goal is not None:
            ax_i.scatter([q_goal[0]], [q_goal[1]], [q_goal[2]], marker="^", label="goal(uav)")
        else:
            ax_i.scatter([uav_xyz_raw[-1, 0]], [uav_xyz_raw[-1, 1]], [uav_xyz_raw[-1, 2]], marker="^", label="goal(uav)")

        if q_grasp is not None:
            ax_i.scatter([q_grasp[0]], [q_grasp[1]], [q_grasp[2]], marker="x", label="grasp(ee target)")
        if args.show_argmin:
            ax_i.scatter(
                [argmin_xyz[0]],
                [argmin_xyz[1]],
                [argmin_xyz[2]],
                marker="*",
                s=90,
                c="gold",
                edgecolors="black",
                linewidths=0.8,
                label=f"argmin_t={argmin_t}",
            )

        if args.draw_obstacles and len(obst_spheres) > 0:
            for i, (ox, oy, oz, rr) in enumerate(obst_spheres):
                if not draw_sphere_mask[i]:
                    continue
                draw_sphere(
                    ax_i,
                    center=(float(ox), float(oy), float(oz)),
                    radius=float(rr) * rvis_scale,
                    color=str(args.obst_color),
                    alpha=float(args.obst_alpha),
                    edgecolor=str(args.obst_edge_color),
                )
        if args.draw_boxes and len(obst_boxes) > 0:
            for cx, cy, cz, hx, hy, hz in obst_boxes:
                draw_aabb(
                    ax_i,
                    center=(float(cx), float(cy), float(cz)),
                    half_ext=(float(hx), float(hy), float(hz)),
                    color=str(args.box_color),
                    alpha=float(args.box_alpha),
                    edgecolor=str(args.box_edge_color),
                )

        # Optional: show executable uniform-dt setpoints as tiny dots (static overlay).
        # This makes "slow regions" appear denser (smaller ds per control step).
        if exec_xyz_plot is not None and exec_xyz_plot.shape[0] > 0:
            ax_i.plot(
                exec_xyz_plot[:, 0],
                exec_xyz_plot[:, 1],
                exec_xyz_plot[:, 2],
                linestyle="None",
                marker=".",
                markersize=float(args.exec_ms),
                color=str(args.exec_color),
                alpha=float(np.clip(float(args.exec_alpha), 0.0, 1.0)),
                label="_nolegend_",
                zorder=12,
            )

        raw_line_i = None
        if bool(args.plot_show_raw_polyline):
            (raw_line_i,) = ax_i.plot(
                [],
                [],
                [],
                label="raw polyline",
                linewidth=1.0,
                alpha=float(np.clip(float(args.plot_raw_alpha), 0.0, 1.0)),
                color="#9aa0a6",
                zorder=20,
            )
            try:
                raw_line_i.set_solid_capstyle("round")
                raw_line_i.set_solid_joinstyle("round")
            except Exception:
                pass
        raw_lines.append(raw_line_i)

        (line_i,) = ax_i.plot([], [], [], label="UAV path", linewidth=2.0, zorder=25)
        try:
            line_i.set_solid_capstyle("round")
            line_i.set_solid_joinstyle("round")
        except Exception:
            pass
        lines.append(line_i)

        ee_line_i = None
        if bool(getattr(args, "draw_ee_path", True)) and (ee_xyz_draw is not None):
            (ee_line_i,) = ax_i.plot(
                [],
                [],
                [],
                label="EE path",
                linewidth=float(args.ee_lw),
                alpha=float(np.clip(float(args.ee_alpha), 0.0, 1.0)),
                color=str(args.ee_color),
                linestyle=str(args.ee_ls),
                zorder=24,
            )
            try:
                ee_line_i.set_solid_capstyle("round")
                ee_line_i.set_solid_joinstyle("round")
            except Exception:
                pass
        ee_lines.append(ee_line_i)

        if args.show_collision:
            (cline_i,) = ax_i.plot([], [], [], color=str(args.collision_color), linewidth=4.2, label="collision segment", zorder=40)
            cpts_i = ax_i.scatter([], [], [], s=64, color=str(args.collision_color), marker="o",
                                  edgecolors="black", linewidths=0.4, depthshade=False, zorder=41)
            (uline_i,) = ax_i.plot([], [], [], color=str(args.unsafe_color), linewidth=4.0, label="unsafe segment", zorder=36)
            upts_i = ax_i.scatter([], [], [], s=58, color=str(args.unsafe_color), marker="o",
                                  edgecolors="black", linewidths=0.35, depthshade=False, zorder=37)
        else:
            (cline_i,) = ax_i.plot([], [], [], color=str(args.collision_color), linewidth=2.2)
            cpts_i = ax_i.scatter([], [], [], s=12, color=str(args.collision_color), marker="o", alpha=0.0)
            (uline_i,) = ax_i.plot([], [], [], color=str(args.unsafe_color), linewidth=2.2)
            upts_i = ax_i.scatter([], [], [], s=12, color=str(args.unsafe_color), marker="o", alpha=0.0)
        for _ln in (cline_i, uline_i):
            try:
                _ln.set_solid_capstyle("round")
                _ln.set_solid_joinstyle("round")
            except Exception:
                pass
        coll_lines.append(cline_i)
        coll_points.append(cpts_i)
        unsafe_lines.append(uline_i)
        unsafe_points.append(upts_i)

        ax_i.set_xlim(mn_cube[0], mx_cube[0])
        ax_i.set_ylim(mn_cube[1], mx_cube[1])
        ax_i.set_zlim(mn_cube[2], mx_cube[2])
        ax_i.set_xlabel("x", fontsize=float(args.axis_label_fontsize))
        ax_i.set_ylabel("y", fontsize=float(args.axis_label_fontsize))
        ax_i.set_zlabel("z", fontsize=float(args.axis_label_fontsize))
        ax_i.tick_params(axis="x", labelsize=float(args.tick_fontsize), pad=1.0)
        ax_i.tick_params(axis="y", labelsize=float(args.tick_fontsize), pad=1.0)
        ax_i.tick_params(axis="z", labelsize=float(args.tick_fontsize), pad=1.0)
        ax_i.set_box_aspect((rx, ry, rz))

        if mn is not None:
            draw_box(ax_i, mn, mx)

    # --------------------------
    # Fixed multiview: 2D orthographic projections (Front X-Z, Left Y-Z)
    # --------------------------
    raw_line_front = None
    line_front = None
    ee_line_front = None
    coll_front_line = None
    coll_front_pts = None
    unsafe_front_line = None
    unsafe_front_pts = None

    raw_line_left = None
    line_left = None
    ee_line_left = None
    coll_left_line = None
    coll_left_pts = None
    unsafe_left_line = None
    unsafe_left_pts = None

    if use_fixed_multiview_layout:
        # Front: X-Z (look along +Y). Left: Y-Z (look along +X).
        for ax_2d, (ix, iy), title, xlabel, ylabel in [
            (ax_front, (0, 2), "正视图 (Front, X-Z)", "x", "z"),
            (ax_left, (1, 2), "左视图 (Left, Y-Z)", "y", "z"),
        ]:
            ax_2d.set_title(title, fontsize=10)
            ax_2d.set_aspect("equal", adjustable="box")
            ax_2d.set_xlabel(xlabel, fontsize=float(args.axis_label_fontsize))
            ax_2d.set_ylabel(ylabel, fontsize=float(args.axis_label_fontsize))
            ax_2d.tick_params(axis="both", labelsize=float(args.tick_fontsize), pad=1.0)
            ax_2d.grid(alpha=0.16)

            # Stable square limits from the 3D cube (keeps panels aligned).
            sx = float(mx_cube[ix] - mn_cube[ix])
            sy = float(mx_cube[iy] - mn_cube[iy])
            span = max(max(1e-6, sx), max(1e-6, sy))
            cx = 0.5 * float(mn_cube[ix] + mx_cube[ix])
            cy = 0.5 * float(mn_cube[iy] + mx_cube[iy])
            half2 = 0.52 * span
            ax_2d.set_xlim(cx - half2, cx + half2)
            ax_2d.set_ylim(cy - half2, cy + half2)

            # Obstacles as projected circles/rectangles (readable 2D engineering view).
            if args.draw_obstacles and len(obst_spheres) > 0:
                for i, (ox, oy, oz, rr) in enumerate(obst_spheres):
                    if not draw_sphere_mask[i]:
                        continue
                    cc = [float(ox), float(oy), float(oz)]
                    circ = Circle(
                        (cc[ix], cc[iy]),
                        radius=float(rr) * rvis_scale,
                        facecolor=str(args.obst_color),
                        edgecolor=str(args.obst_edge_color),
                        alpha=min(0.70, max(0.20, float(args.obst_alpha) + 0.16)),
                        linewidth=1.1,
                    )
                    ax_2d.add_patch(circ)
            if args.draw_boxes and len(obst_boxes) > 0:
                for cx3, cy3, cz3, hx, hy, hz in obst_boxes:
                    cc = [float(cx3), float(cy3), float(cz3)]
                    hh = [float(hx), float(hy), float(hz)]
                    rect_box = Rectangle(
                        (cc[ix] - hh[ix], cc[iy] - hh[iy]),
                        width=float(max(1e-6, 2.0 * hh[ix])),
                        height=float(max(1e-6, 2.0 * hh[iy])),
                        facecolor=str(args.box_color),
                        edgecolor=str(args.box_edge_color),
                        alpha=min(0.78, max(0.18, float(args.box_alpha) + 0.08)),
                        linewidth=1.1,
                    )
                    ax_2d.add_patch(rect_box)

            # Markers (projected)
            if q_start is not None:
                ax_2d.scatter([q_start[ix]], [q_start[iy]], marker="o", s=28, label="start(uav)")
            else:
                ax_2d.scatter([uav_xyz_raw[0, ix]], [uav_xyz_raw[0, iy]], marker="o", s=28, label="start(uav)")
            if q_goal is not None:
                ax_2d.scatter([q_goal[ix]], [q_goal[iy]], marker="^", s=34, label="goal(uav)")
            else:
                ax_2d.scatter([uav_xyz_raw[-1, ix]], [uav_xyz_raw[-1, iy]], marker="^", s=34, label="goal(uav)")
            if q_grasp is not None:
                ax_2d.scatter([q_grasp[ix]], [q_grasp[iy]], marker="x", s=34, label="grasp(ee)")
            if args.show_argmin:
                ax_2d.scatter(
                    [argmin_xyz[ix]],
                    [argmin_xyz[iy]],
                    marker="*",
                    s=95,
                    c="gold",
                    edgecolors="black",
                    linewidths=0.8,
                    label=f"argmin_t={argmin_t}",
                )

            # Executable setpoints overlay (static).
            if exec_xyz_plot is not None and exec_xyz_plot.shape[0] > 0:
                ax_2d.plot(
                    exec_xyz_plot[:, ix],
                    exec_xyz_plot[:, iy],
                    linestyle="None",
                    marker=".",
                    markersize=float(args.exec_ms),
                    color=str(args.exec_color),
                    alpha=float(np.clip(float(args.exec_alpha), 0.0, 1.0)),
                    label="_nolegend_",
                    zorder=12,
                )

        # Lines (Front)
        if bool(args.plot_show_raw_polyline):
            (raw_line_front,) = ax_front.plot(
                [],
                [],
                linewidth=1.0,
                alpha=float(np.clip(float(args.plot_raw_alpha), 0.0, 1.0)),
                color="#9aa0a6",
                label="raw polyline",
                zorder=18,
            )
            try:
                raw_line_front.set_solid_capstyle("round")
                raw_line_front.set_solid_joinstyle("round")
            except Exception:
                pass
        (line_front,) = ax_front.plot([], [], linewidth=2.0, label="UAV path", zorder=25)
        try:
            line_front.set_solid_capstyle("round")
            line_front.set_solid_joinstyle("round")
        except Exception:
            pass
        if bool(getattr(args, "draw_ee_path", True)) and (ee_xyz_draw is not None):
            (ee_line_front,) = ax_front.plot(
                [],
                [],
                linewidth=float(args.ee_lw),
                alpha=float(np.clip(float(args.ee_alpha), 0.0, 1.0)),
                color=str(args.ee_color),
                linestyle=str(args.ee_ls),
                label="EE path",
                zorder=24,
            )
            try:
                ee_line_front.set_solid_capstyle("round")
                ee_line_front.set_solid_joinstyle("round")
            except Exception:
                pass
        if args.show_collision:
            (coll_front_line,) = ax_front.plot([], [], linewidth=4.0, color=str(args.collision_color), label="collision segment", zorder=40)
            coll_front_pts = ax_front.scatter([], [], s=52, color=str(args.collision_color), marker="o",
                                              edgecolors="black", linewidths=0.4, zorder=41)
            (unsafe_front_line,) = ax_front.plot([], [], linewidth=3.8, color=str(args.unsafe_color), label="unsafe segment", zorder=36)
            unsafe_front_pts = ax_front.scatter([], [], s=48, color=str(args.unsafe_color), marker="o",
                                                edgecolors="black", linewidths=0.35, zorder=37)
        else:
            (coll_front_line,) = ax_front.plot([], [], linewidth=2.2, color=str(args.collision_color))
            coll_front_pts = ax_front.scatter([], [], s=12, color=str(args.collision_color), marker="o", alpha=0.0)
            (unsafe_front_line,) = ax_front.plot([], [], linewidth=2.2, color=str(args.unsafe_color))
            unsafe_front_pts = ax_front.scatter([], [], s=12, color=str(args.unsafe_color), marker="o", alpha=0.0)
        for _ln in (coll_front_line, unsafe_front_line):
            try:
                _ln.set_solid_capstyle("round")
                _ln.set_solid_joinstyle("round")
            except Exception:
                pass

        # Lines (Left)
        if bool(args.plot_show_raw_polyline):
            (raw_line_left,) = ax_left.plot(
                [],
                [],
                linewidth=1.0,
                alpha=float(np.clip(float(args.plot_raw_alpha), 0.0, 1.0)),
                color="#9aa0a6",
                label="raw polyline",
                zorder=18,
            )
            try:
                raw_line_left.set_solid_capstyle("round")
                raw_line_left.set_solid_joinstyle("round")
            except Exception:
                pass
        (line_left,) = ax_left.plot([], [], linewidth=2.0, label="UAV path", zorder=25)
        try:
            line_left.set_solid_capstyle("round")
            line_left.set_solid_joinstyle("round")
        except Exception:
            pass
        if bool(getattr(args, "draw_ee_path", True)) and (ee_xyz_draw is not None):
            (ee_line_left,) = ax_left.plot(
                [],
                [],
                linewidth=float(args.ee_lw),
                alpha=float(np.clip(float(args.ee_alpha), 0.0, 1.0)),
                color=str(args.ee_color),
                linestyle=str(args.ee_ls),
                label="EE path",
                zorder=24,
            )
            try:
                ee_line_left.set_solid_capstyle("round")
                ee_line_left.set_solid_joinstyle("round")
            except Exception:
                pass
        if args.show_collision:
            (coll_left_line,) = ax_left.plot([], [], linewidth=4.0, color=str(args.collision_color), label="collision segment", zorder=40)
            coll_left_pts = ax_left.scatter([], [], s=52, color=str(args.collision_color), marker="o",
                                            edgecolors="black", linewidths=0.4, zorder=41)
            (unsafe_left_line,) = ax_left.plot([], [], linewidth=3.8, color=str(args.unsafe_color), label="unsafe segment", zorder=36)
            unsafe_left_pts = ax_left.scatter([], [], s=48, color=str(args.unsafe_color), marker="o",
                                              edgecolors="black", linewidths=0.35, zorder=37)
        else:
            (coll_left_line,) = ax_left.plot([], [], linewidth=2.2, color=str(args.collision_color))
            coll_left_pts = ax_left.scatter([], [], s=12, color=str(args.collision_color), marker="o", alpha=0.0)
            (unsafe_left_line,) = ax_left.plot([], [], linewidth=2.2, color=str(args.unsafe_color))
            unsafe_left_pts = ax_left.scatter([], [], s=12, color=str(args.unsafe_color), marker="o", alpha=0.0)
        for _ln in (coll_left_line, unsafe_left_line):
            try:
                _ln.set_solid_capstyle("round")
                _ln.set_solid_joinstyle("round")
            except Exception:
                pass

    line_top = None
    raw_line_top = None
    ee_line_top = None
    coll_top_line = None
    coll_top_pts = None
    unsafe_top_line = None
    unsafe_top_pts = None
    if ax_top is not None:
        # Map-style XY boundaries
        if obst_xy_box is not None:
            bmn, bmx = obst_xy_box
            bmn = np.asarray(bmn, dtype=float).reshape(2)
            bmx = np.asarray(bmx, dtype=float).reshape(2)
        else:
            bmn = np.asarray(mn_lim[:2], dtype=float)
            bmx = np.asarray(mx_lim[:2], dtype=float)

        # Obstacles as XY circles (paper-like map view)
        if args.draw_obstacles and len(obst_spheres) > 0:
            for i, (ox, oy, _, rr) in enumerate(obst_spheres):
                if not draw_sphere_mask[i]:
                    continue
                circ = Circle(
                    (float(ox), float(oy)),
                    radius=float(rr) * rvis_scale,
                    facecolor=str(args.obst_color),
                    edgecolor=str(args.obst_edge_color),
                    alpha=min(0.70, max(0.20, float(args.obst_alpha) + 0.16)),
                    linewidth=1.1,
                )
                ax_top.add_patch(circ)
        if args.draw_boxes and len(obst_boxes) > 0:
            for cx, cy, _, hx, hy, _ in obst_boxes:
                rect_box = Rectangle(
                    (float(cx - hx), float(cy - hy)),
                    width=float(max(1e-6, 2.0 * hx)),
                    height=float(max(1e-6, 2.0 * hy)),
                    facecolor=str(args.box_color),
                    edgecolor=str(args.box_edge_color),
                    alpha=min(0.78, max(0.18, float(args.box_alpha) + 0.08)),
                    linewidth=1.1,
                )
                ax_top.add_patch(rect_box)

        rect = Rectangle(
            (float(bmn[0]), float(bmn[1])),
            width=float(max(1e-6, bmx[0] - bmn[0])),
            height=float(max(1e-6, bmx[1] - bmn[1])),
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
        ax_top.add_patch(rect)

        if q_start is not None:
            ax_top.scatter([q_start[0]], [q_start[1]], marker="o", s=28, label="start(uav)")
        else:
            ax_top.scatter([uav_xyz_raw[0, 0]], [uav_xyz_raw[0, 1]], marker="o", s=28, label="start(uav)")
        if q_goal is not None:
            ax_top.scatter([q_goal[0]], [q_goal[1]], marker="^", s=34, label="goal(uav)")
        else:
            ax_top.scatter([uav_xyz_raw[-1, 0]], [uav_xyz_raw[-1, 1]], marker="^", s=34, label="goal(uav)")
        if q_grasp is not None:
            ax_top.scatter([q_grasp[0]], [q_grasp[1]], marker="x", s=34, label="grasp(ee)")
        if args.show_argmin:
            ax_top.scatter(
                [argmin_xyz[0]],
                [argmin_xyz[1]],
                marker="*",
                s=95,
                c="gold",
                edgecolors="black",
                linewidths=0.8,
                label=f"argmin_t={argmin_t}",
            )

        if exec_xyz_plot is not None and exec_xyz_plot.shape[0] > 0:
            ax_top.plot(
                exec_xyz_plot[:, 0],
                exec_xyz_plot[:, 1],
                linestyle="None",
                marker=".",
                markersize=float(args.exec_ms),
                color=str(args.exec_color),
                alpha=float(np.clip(float(args.exec_alpha), 0.0, 1.0)),
                label="exec setpoints",
                zorder=12,
            )

        if bool(args.plot_show_raw_polyline):
            (raw_line_top,) = ax_top.plot(
                [],
                [],
                linewidth=1.0,
                alpha=float(np.clip(float(args.plot_raw_alpha), 0.0, 1.0)),
                color="#9aa0a6",
                label="raw polyline",
                zorder=18,
            )
            try:
                raw_line_top.set_solid_capstyle("round")
                raw_line_top.set_solid_joinstyle("round")
            except Exception:
                pass
        (line_top,) = ax_top.plot([], [], linewidth=2.0, label="UAV path")
        try:
            line_top.set_solid_capstyle("round")
            line_top.set_solid_joinstyle("round")
        except Exception:
            pass
        if bool(getattr(args, "draw_ee_path", True)) and (ee_xyz_draw is not None):
            (ee_line_top,) = ax_top.plot(
                [],
                [],
                linewidth=float(args.ee_lw),
                alpha=float(np.clip(float(args.ee_alpha), 0.0, 1.0)),
                color=str(args.ee_color),
                linestyle=str(args.ee_ls),
                label="EE path",
                zorder=24,
            )
            try:
                ee_line_top.set_solid_capstyle("round")
                ee_line_top.set_solid_joinstyle("round")
            except Exception:
                pass
        if args.show_collision:
            (coll_top_line,) = ax_top.plot([], [], linewidth=4.0, color=str(args.collision_color), label="collision segment")
            coll_top_pts = ax_top.scatter([], [], s=52, color=str(args.collision_color), marker="o",
                                          edgecolors="black", linewidths=0.4)
            (unsafe_top_line,) = ax_top.plot([], [], linewidth=3.8, color=str(args.unsafe_color), label="unsafe segment")
            unsafe_top_pts = ax_top.scatter([], [], s=48, color=str(args.unsafe_color), marker="o",
                                            edgecolors="black", linewidths=0.35)
        else:
            (coll_top_line,) = ax_top.plot([], [], linewidth=2.2, color=str(args.collision_color))
            coll_top_pts = ax_top.scatter([], [], s=12, color=str(args.collision_color), marker="o", alpha=0.0)
            (unsafe_top_line,) = ax_top.plot([], [], linewidth=2.2, color=str(args.unsafe_color))
            unsafe_top_pts = ax_top.scatter([], [], s=12, color=str(args.unsafe_color), marker="o", alpha=0.0)
        for _ln in (coll_top_line, unsafe_top_line):
            try:
                _ln.set_solid_capstyle("round")
                _ln.set_solid_joinstyle("round")
            except Exception:
                pass
        sx = max(1e-6, float(bmx[0] - bmn[0]))
        sy = max(1e-6, float(bmx[1] - bmn[1]))
        span = max(sx, sy)
        cx = 0.5 * float(bmn[0] + bmx[0])
        cy = 0.5 * float(bmn[1] + bmx[1])
        half = 0.52 * span
        ax_top.set_xlim(cx - half, cx + half)
        ax_top.set_ylim(cy - half, cy + half)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_xlabel("x", fontsize=float(args.axis_label_fontsize))
        ax_top.set_ylabel("y", fontsize=float(args.axis_label_fontsize))
        ax_top.tick_params(axis="both", labelsize=float(args.tick_fontsize), pad=1.0)
        if use_fixed_multiview_layout:
            ax_top.set_title("俯视图 (Top-Down XY)")
        else:
            ax_top.set_title("Top-Down (XY)  [Map View]")

    if use_fixed_multiview_layout:
        # Keep exactly one legend to avoid clutter.
        if ax_top is not None:
            ax_top.legend(loc="lower right", fontsize=8, framealpha=0.72, handlelength=1.8)
        elif ax_main is not None:
            ax_main.legend(loc="upper right", fontsize=8, framealpha=0.72, handlelength=1.8)
    else:
        if n_views == 1:
            axes[0].legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                framealpha=0.72,
                handlelength=1.8,
            )
            if ax_top is not None:
                ax_top.legend(loc="lower right", fontsize=8, framealpha=0.72, handlelength=1.8)
        else:
            # Keep legends away from left-top start/obstacle-heavy regions.
            axes[0].legend(loc="upper right", fontsize=8, framealpha=0.72, handlelength=1.8)
            if ax_top is not None:
                ax_top.legend(loc="lower right", fontsize=8, framealpha=0.72, handlelength=1.8)

    status_text = fig.text(
        0.5,
        0.01,
        "",
        ha="center",
        va="bottom",
        fontsize=8.7,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="#999999", boxstyle="round,pad=0.25"),
    )
    if args.status_level == "none":
        status_text.set_visible(False)

    # IMPORTANT: ensure last frame is H-1. We keep the frame count controlled by
    # --stride for both playback modes, so arclen playback remains comparable.
    frames_raw_base = list(range(1, H, args.stride))
    if not frames_raw_base:
        frames_raw_base = [H - 1]
    if frames_raw_base[-1] != H - 1:
        frames_raw_base.append(H - 1)

    H_up = int(uav_xyz_draw.shape[0])
    play_param = str(args.play_param)
    if play_param == "arclen":
        frames_td = _make_playback_frames_arclen(
            uav_xyz_draw,
            n_frames=int(len(frames_raw_base)),
            td_start=int(min(int(plot_up), max(0, H_up - 1))),
        )
        frames_raw = [int(min(int(td) // int(plot_up), int(H - 1))) for td in frames_td]
    elif play_param == "time":
        if traj_time_raw is None or int(np.asarray(traj_time_raw).reshape(-1).shape[0]) != int(H):
            print("[WARN] play_param=time requested but traj_time not found (or wrong shape); falling back to play_param=index")
            frames_raw = list(frames_raw_base)
            frames_td = [int(min(int(t) * int(plot_up), max(0, H_up - 1))) for t in frames_raw]
        else:
            tt = np.asarray(traj_time_raw, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(tt)):
                print("[WARN] play_param=time but traj_time contains non-finite values; falling back to play_param=index")
                frames_raw = list(frames_raw_base)
                frames_td = [int(min(int(t) * int(plot_up), max(0, H_up - 1))) for t in frames_raw]
            else:
                tt = tt - float(tt[0])
                tt = np.maximum.accumulate(tt)
                if float(tt[-1]) <= 1e-12:
                    print("[WARN] play_param=time but traj_time has zero duration; falling back to play_param=index")
                    frames_raw = list(frames_raw_base)
                    frames_td = [int(min(int(t) * int(plot_up), max(0, H_up - 1))) for t in frames_raw]
                else:
                    raw_idx = np.arange(H, dtype=np.float64)
                    draw_idx = np.linspace(0.0, float(H - 1), H_up, dtype=np.float64)
                    t_draw = np.interp(draw_idx, raw_idx, tt).astype(np.float64)
                    td_start = int(min(int(plot_up), max(0, H_up - 1)))
                    targets = np.linspace(float(t_draw[td_start]), float(t_draw[-1]), int(len(frames_raw_base)), dtype=np.float64)
                    # Monotone time->index mapping (avoid argmin(|t-ttarget|) jitter on plateaus).
                    td = np.searchsorted(t_draw, targets, side="right").astype(np.int64) - 1
                    td = np.clip(td, 0, H_up - 1)
                    td[-1] = H_up - 1
                    for i in range(1, int(td.size)):
                        if td[i] < td[i - 1]:
                            td[i] = td[i - 1]
                    frames_td = td.astype(int).tolist()
                    frames_raw = [int(min(int(x) // int(plot_up), int(H - 1))) for x in frames_td]
    else:
        frames_raw = list(frames_raw_base)
        frames_td = [int(min(int(t) * int(plot_up), max(0, H_up - 1))) for t in frames_raw]

    coll_seg_draw = _upsample_seg_mask(_pointmask_to_segmask(coll_mask), plot_up)
    unsafe_seg_draw = _upsample_seg_mask(_pointmask_to_segmask(unsafe_mask), plot_up)

    if args.plot_debug:
        t_last = int(frames_raw[-1]) if frames_raw else int(H - 1)
        td_last = int(frames_td[-1]) if frames_td else int(uav_xyz_draw.shape[0] - 1)
        print(
            "[PLOT_FRAMES] "
            f"play_param={str(args.play_param)} "
            f"n_frames={int(len(frames_raw))} "
            f"t_last={t_last}/{int(H - 1)} "
            f"td_last={td_last}/{int(uav_xyz_draw.shape[0] - 1)} "
            f"draw_pts_last={td_last + 1}/{int(uav_xyz_draw.shape[0])}"
        )

    _dbg_printed_frame0 = {"done": False}

    def update(fi):
        t = int(frames_raw[fi])
        # 't' is the raw waypoint index used for metrics. 'td' is the draw index used
        # for plotting (smooth curve or arclen-based playback).
        td = int(frames_td[fi])
        if args.plot_debug and fi == 0 and (not _dbg_printed_frame0["done"]):
            print(
                "[PLOT_FRAME0] "
                f"t_raw={int(t)} "
                f"td_up={int(td)} "
                f"draw_pts={int(td) + 1}/{int(uav_xyz_draw.shape[0])} "
                f"stride={int(args.stride)} "
                f"play_param={str(args.play_param)}"
            )
            _dbg_printed_frame0["done"] = True
        cm_pt = coll_mask[: t + 1]
        um_pt = unsafe_mask[: t + 1]
        cm_seg = coll_seg_draw[:td] if coll_seg_draw.size > 0 else np.zeros((0,), dtype=bool)
        um_seg = unsafe_seg_draw[:td] if unsafe_seg_draw.size > 0 else np.zeros((0,), dtype=bool)
        for k, (ax_i, raw_i, line_i, ee_i, cline_i, cpts_i, uline_i, upts_i, (ev, az)) in enumerate(
            zip(axes, raw_lines, lines, ee_lines, coll_lines, coll_points, unsafe_lines, unsafe_points, views_3d)
        ):
            if raw_i is not None:
                raw_i.set_data(uav_xyz_raw[: t + 1, 0], uav_xyz_raw[: t + 1, 1])
                raw_i.set_3d_properties(uav_xyz_raw[: t + 1, 2])
            line_i.set_data(uav_xyz_draw[: td + 1, 0], uav_xyz_draw[: td + 1, 1])
            line_i.set_3d_properties(uav_xyz_draw[: td + 1, 2])
            if ee_i is not None and ee_xyz_draw is not None:
                ee_i.set_data(ee_xyz_draw[: td + 1, 0], ee_xyz_draw[: td + 1, 1])
                ee_i.set_3d_properties(ee_xyz_draw[: td + 1, 2])
            xyzc = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], cm_seg)
            cline_i.set_data(xyzc[:, 0], xyzc[:, 1])
            cline_i.set_3d_properties(xyzc[:, 2])
            if np.any(cm_pt):
                ids = np.where(cm_pt)[0]
                ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                cxyz = uav_xyz_draw[ids_up]
                cpts_i._offsets3d = (cxyz[:, 0], cxyz[:, 1], cxyz[:, 2])
            else:
                cpts_i._offsets3d = ([], [], [])
            xyzu = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], um_seg)
            uline_i.set_data(xyzu[:, 0], xyzu[:, 1])
            uline_i.set_3d_properties(xyzu[:, 2])
            if np.any(um_pt):
                ids = np.where(um_pt)[0]
                ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                uxyz = uav_xyz_draw[ids_up]
                upts_i._offsets3d = (uxyz[:, 0], uxyz[:, 1], uxyz[:, 2])
            else:
                upts_i._offsets3d = ([], [], [])
            ax_i.view_init(elev=ev, azim=az)
            if use_fixed_multiview_layout:
                ax_i.set_title(str(view_labels[k]), fontsize=10)
            else:
                if use_traj_npz:
                    base_title = f"{view_labels[k]} [e={ev:.0f}, a={az:.0f}]  t={t}/{H-1}"
                else:
                    base_title = f"{view_labels[k]} [e={ev:.0f}, a={az:.0f}]  sample={sid} t={t}/{H-1}"
                ax_i.set_title(base_title, fontsize=10)

        artists = (
            [ln for ln in raw_lines if ln is not None]
            + list(lines)
            + [ln for ln in ee_lines if ln is not None]
            + list(coll_lines)
            + list(coll_points)
            + list(unsafe_lines)
            + list(unsafe_points)
        )

        if use_fixed_multiview_layout:
            # Front (X-Z)
            if raw_line_front is not None:
                raw_line_front.set_data(uav_xyz_raw[: t + 1, 0], uav_xyz_raw[: t + 1, 2])
            if line_front is not None:
                line_front.set_data(uav_xyz_draw[: td + 1, 0], uav_xyz_draw[: td + 1, 2])
            if ee_line_front is not None and ee_xyz_draw is not None:
                ee_line_front.set_data(ee_xyz_draw[: td + 1, 0], ee_xyz_draw[: td + 1, 2])
            if coll_front_line is not None:
                xyzc = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], cm_seg)
                coll_front_line.set_data(xyzc[:, 0], xyzc[:, 2])
            if coll_front_pts is not None:
                if np.any(cm_pt):
                    ids = np.where(cm_pt)[0]
                    ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                    cproj = uav_xyz_draw[ids_up][:, [0, 2]]
                    coll_front_pts.set_offsets(cproj)
                else:
                    coll_front_pts.set_offsets(np.empty((0, 2), dtype=float))
            if unsafe_front_line is not None:
                xyzu = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], um_seg)
                unsafe_front_line.set_data(xyzu[:, 0], xyzu[:, 2])
            if unsafe_front_pts is not None:
                if np.any(um_pt):
                    ids = np.where(um_pt)[0]
                    ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                    uproj = uav_xyz_draw[ids_up][:, [0, 2]]
                    unsafe_front_pts.set_offsets(uproj)
                else:
                    unsafe_front_pts.set_offsets(np.empty((0, 2), dtype=float))

            # Left (Y-Z)
            if raw_line_left is not None:
                raw_line_left.set_data(uav_xyz_raw[: t + 1, 1], uav_xyz_raw[: t + 1, 2])
            if line_left is not None:
                line_left.set_data(uav_xyz_draw[: td + 1, 1], uav_xyz_draw[: td + 1, 2])
            if ee_line_left is not None and ee_xyz_draw is not None:
                ee_line_left.set_data(ee_xyz_draw[: td + 1, 1], ee_xyz_draw[: td + 1, 2])
            if coll_left_line is not None:
                xyzc = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], cm_seg)
                coll_left_line.set_data(xyzc[:, 1], xyzc[:, 2])
            if coll_left_pts is not None:
                if np.any(cm_pt):
                    ids = np.where(cm_pt)[0]
                    ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                    cproj = uav_xyz_draw[ids_up][:, [1, 2]]
                    coll_left_pts.set_offsets(cproj)
                else:
                    coll_left_pts.set_offsets(np.empty((0, 2), dtype=float))
            if unsafe_left_line is not None:
                xyzu = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], um_seg)
                unsafe_left_line.set_data(xyzu[:, 1], xyzu[:, 2])
            if unsafe_left_pts is not None:
                if np.any(um_pt):
                    ids = np.where(um_pt)[0]
                    ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                    uproj = uav_xyz_draw[ids_up][:, [1, 2]]
                    unsafe_left_pts.set_offsets(uproj)
                else:
                    unsafe_left_pts.set_offsets(np.empty((0, 2), dtype=float))

            for ln in (
                raw_line_front,
                line_front,
                ee_line_front,
                coll_front_line,
                unsafe_front_line,
                raw_line_left,
                line_left,
                ee_line_left,
                coll_left_line,
                unsafe_left_line,
            ):
                if ln is not None:
                    artists.append(ln)
            for pts in (coll_front_pts, unsafe_front_pts, coll_left_pts, unsafe_left_pts):
                if pts is not None:
                    artists.append(pts)
        if line_top is not None:
            if raw_line_top is not None:
                raw_line_top.set_data(uav_xyz_raw[: t + 1, 0], uav_xyz_raw[: t + 1, 1])
            line_top.set_data(uav_xyz_draw[: td + 1, 0], uav_xyz_draw[: td + 1, 1])
            if ee_line_top is not None and ee_xyz_draw is not None:
                ee_line_top.set_data(ee_xyz_draw[: td + 1, 0], ee_xyz_draw[: td + 1, 1])
            xyzc2 = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], cm_seg)
            coll_top_line.set_data(xyzc2[:, 0], xyzc2[:, 1])
            if np.any(cm_pt):
                ids = np.where(cm_pt)[0]
                ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                cxy = uav_xyz_draw[ids_up][:, :2]
                coll_top_pts.set_offsets(cxy)
            else:
                coll_top_pts.set_offsets(np.empty((0, 2), dtype=float))
            xyzu2 = _segments_to_polyline_xyz(uav_xyz_draw[: td + 1], um_seg)
            unsafe_top_line.set_data(xyzu2[:, 0], xyzu2[:, 1])
            if np.any(um_pt):
                ids = np.where(um_pt)[0]
                ids_up = np.clip(ids.astype(np.int64) * int(plot_up), 0, int(uav_xyz_draw.shape[0] - 1))
                uxy = uav_xyz_draw[ids_up][:, :2]
                unsafe_top_pts.set_offsets(uxy)
            else:
                unsafe_top_pts.set_offsets(np.empty((0, 2), dtype=float))
            if use_fixed_multiview_layout:
                ax_top.set_title("俯视图 (Top-Down XY)", fontsize=10)
            else:
                if use_traj_npz:
                    ax_top.set_title(f"Top-Down (XY)  t={t}/{H-1}", fontsize=10)
                else:
                    ax_top.set_title(f"Top-Down (XY) sample={sid} t={t}/{H-1}", fontsize=10)
            if raw_line_top is not None:
                artists.append(raw_line_top)
            artists.extend([x for x in (line_top, ee_line_top, coll_top_line, coll_top_pts, unsafe_top_line, unsafe_top_pts) if x is not None])

        if args.status_level != "none":
            if t >= 1:
                dmin_run = float(np.min(sd_surface_pt[: t + 1]))
                clrmin_run = float(np.min(clearance[: t + 1]))
            else:
                dmin_run = float(sd_surface_pt[0])
                clrmin_run = float(clearance[0])
            unsafe_preview = unsafe_intervals[:3]
            coll_preview = coll_intervals[:3]
            if use_traj_npz:
                head = f"{meta_txt}  t={t}/{H-1}"
            else:
                head = f"sample={sid} best_idx={best_idx}  t={t}/{H-1}"
            if args.status_level == "brief":
                lines_txt = [
                    head,
                    f"d_min={dmin_run:.4f}m clr_min={clrmin_run:.4f}m d_safe={d_safe:.4f}m | coll={has_collision} unsafe={has_unsafe} argmin_t={argmin_t}",
                    f"unsafe_len={unsafe_len} coll_len={coll_len}",
                ]
            else:
                lines_txt = [
                    head,
                    f"d_min={dmin_run:.4f}m  clr_min={clrmin_run:.4f}m  d_safe={d_safe:.4f}m",
                    f"has_collision={has_collision}  has_unsafe={has_unsafe}  argmin_t={argmin_t}",
                    f"unsafe_len={unsafe_len}  coll_len={coll_len}",
                    f"unsafe_intervals={unsafe_preview}  coll_intervals={coll_preview}",
                ]
            status_text.set_text("\n".join(lines_txt))
        artists.append(status_text)

        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=len(frames_raw), interval=50, blit=False)
    ani.save(args.out, writer="pillow", fps=args.fps)
    print("[OK] wrote", args.out)
    plt.close(fig)
    if args.clearance_png:
        _save_clearance_plot(args.clearance_png, clearance, safe_margin)
    if args.dyn_png:
        _save_dynamics_plot(args.dyn_png, dyn, abrupt_idx)
    if args.state_png or args.dump_state:
        out_png = str(args.state_png).strip()
        if not out_png:
            base = os.path.basename(str(args.out)).rsplit(".", 1)[0]
            out_png = f"/tmp/{base}_state.png"
        try:
            # Lazy import to keep make_traj_gif usable even when matplotlib backends differ.
            import plot_state_debug as _psd  # type: ignore

            _psd.save_state_debug_png(
                out_png=out_png,
                npz=d,
                traj=traj,
                sid=int(sid),
                traj_key=str(args.traj_key) if use_traj_npz else "traj_post",
                time_key="traj_time",
                dt_fallback=float(args.dt),
                plot_debug=bool(args.plot_debug),
                ee_xyz=ee_xyz_raw,
                ee_urdf=str(getattr(args, "ee_urdf", "")),
                ee_frame=str(getattr(args, "ee_frame", "gripper_link")),
            )
            print("[OK] wrote", out_png)
        except Exception as e:
            print(f"[WARN] state_png failed: {_short_exc(e)}")


if __name__ == "__main__":
    main()
