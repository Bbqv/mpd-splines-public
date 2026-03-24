import numpy as np
import torch
import torch.nn.functional as F


def _choose_grid_shape(n_items, xyz_min, xyz_max):
    n = int(max(1, n_items))
    lo = np.asarray(xyz_min, dtype=np.float32).reshape(3)
    hi = np.asarray(xyz_max, dtype=np.float32).reshape(3)
    span = np.maximum(hi - lo, 1e-6)
    g = float(np.cbrt(float(span[0] * span[1] * span[2])))
    scaled = np.maximum(span / g, 1e-3)
    base = np.maximum(1, np.round(scaled * np.cbrt(float(n))).astype(np.int64))
    while int(base[0] * base[1] * base[2]) < n:
        axis = int(np.argmax(span / np.maximum(base, 1)))
        base[axis] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _cell_bounds(cell_id, grid_shape, xyz_min, xyz_max):
    nx, ny, nz = [int(v) for v in grid_shape]
    i = int(cell_id)
    ix = i % nx
    iy = (i // nx) % ny
    iz = i // (nx * ny)
    lo = np.asarray(xyz_min, dtype=np.float32).reshape(3)
    hi = np.asarray(xyz_max, dtype=np.float32).reshape(3)
    step = (hi - lo) / np.asarray([nx, ny, nz], dtype=np.float32)
    c_lo = lo + step * np.asarray([ix, iy, iz], dtype=np.float32)
    c_hi = c_lo + step
    return c_lo.astype(np.float32), c_hi.astype(np.float32)


def _size_bin_indices(n_items):
    n = int(max(1, n_items))
    if n == 1:
        return [1]
    if n == 2:
        return [0, 2]
    return [0, 1, 2]


def _sample_in_size_bin(rng, lo, hi, bin_id):
    l = float(lo)
    h = float(hi)
    if h <= l:
        return l
    bins = {
        0: (0.0, 1.0 / 3.0),
        1: (1.0 / 3.0, 2.0 / 3.0),
        2: (2.0 / 3.0, 1.0),
    }
    b_lo, b_hi = bins[int(bin_id)]
    seg_lo = l + (h - l) * b_lo
    seg_hi = l + (h - l) * b_hi
    if seg_hi <= seg_lo:
        seg_lo, seg_hi = l, h
    return float(rng.uniform(seg_lo, seg_hi))


def spheres_from_point_cloud(points, point_radius=0.02, max_points=256, seed=0):
    """
    Convert point cloud points (N,3) to tiny obstacle spheres.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return []
    k = int(max(1, max_points))
    if pts.shape[0] > k:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(pts.shape[0], size=k, replace=False)
        pts = pts[idx]
    r = float(point_radius)
    if r <= 0.0:
        raise ValueError(f"[ERR] point_radius must be > 0, got {r}")
    return [(pts[i].astype(np.float32), r) for i in range(pts.shape[0])]


def sample_random_spheres_uniform(
    n_spheres,
    xyz_min,
    xyz_max,
    r_min,
    r_max,
    seed=0,
    anchor_points=None,
    anchor_clearance=0.0,
    avoid_overlap=False,
    overlap_margin=0.0,
    existing_spheres=None,
    max_tries=5000,
    sampling_mode="stratified",
    radius_mode="mixed",
    center_min_dist=0.0,
):
    """
    Sample random obstacle spheres with optional anchor-feasibility and overlap checks.
    """
    n = int(max(0, n_spheres))
    if n == 0:
        return []
    xyz_min = np.asarray(xyz_min, dtype=np.float32).reshape(3)
    xyz_max = np.asarray(xyz_max, dtype=np.float32).reshape(3)
    if np.any(xyz_max <= xyz_min):
        raise ValueError("[ERR] xyz_max must be > xyz_min for all dimensions")
    r_lo = float(min(r_min, r_max))
    r_hi = float(max(r_min, r_max))
    if r_lo <= 0.0:
        raise ValueError("[ERR] random sphere radius must be > 0")

    anchors = None
    if anchor_points is not None:
        ap = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 3)
        if ap.shape[0] > 0:
            anchors = ap
    anc_margin = float(max(0.0, anchor_clearance))
    ov_margin = float(max(0.0, overlap_margin))
    existing = []
    if existing_spheres is not None:
        for c, r in existing_spheres:
            existing.append((np.asarray(c, dtype=np.float32).reshape(3), float(r)))

    rng = np.random.default_rng(int(seed))
    mode = str(sampling_mode).strip().lower()
    if mode not in ("iid", "stratified", "grid3d"):
        raise ValueError(f"[ERR] unknown sampling_mode='{sampling_mode}', expected iid|stratified|grid3d")
    r_mode = str(radius_mode).strip().lower()
    if r_mode not in ("uniform", "mixed"):
        raise ValueError(f"[ERR] unknown radius_mode='{radius_mode}', expected uniform|mixed")
    c_min_dist = float(max(0.0, center_min_dist))
    bin_ids = _size_bin_indices(n)

    grid_shape = _choose_grid_shape(n, xyz_min, xyz_max)
    n_cells = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
    cell_order = np.arange(n_cells, dtype=np.int64)
    rng.shuffle(cell_order)
    cell_ptr = 0
    used_cells = set()

    out = []
    tries = 0
    while len(out) < n and tries < int(max_tries):
        tries += 1
        if r_mode == "uniform":
            r = float(rng.uniform(r_lo, r_hi))
        else:
            bid = int(bin_ids[len(out) % len(bin_ids)])
            r = _sample_in_size_bin(rng, r_lo, r_hi, bid)
        g_lo = xyz_min + r
        g_hi = xyz_max - r
        if np.any(g_hi <= g_lo):
            continue

        if mode in ("stratified", "grid3d"):
            if mode == "grid3d":
                if len(used_cells) < n_cells:
                    loops = 0
                    cid = None
                    while loops < max(1, n_cells):
                        c_try = int(cell_order[cell_ptr % n_cells])
                        cell_ptr += 1
                        loops += 1
                        if c_try not in used_cells:
                            cid = c_try
                            break
                    if cid is None:
                        cid = int(cell_order[(tries - 1) % n_cells])
                else:
                    cid = int(cell_order[(tries - 1) % n_cells])
            else:
                cid = int(cell_order[(tries - 1) % n_cells])
                if ((tries - 1) % n_cells) == 0 and tries > 1:
                    rng.shuffle(cell_order)
            c_lo, c_hi = _cell_bounds(cid, grid_shape, xyz_min, xyz_max)
            c_lo = np.maximum(c_lo, g_lo)
            c_hi = np.minimum(c_hi, g_hi)
            if np.any(c_hi <= c_lo):
                c_lo = g_lo
                c_hi = g_hi
        else:
            c_lo = g_lo
            c_hi = g_hi
        c = rng.uniform(c_lo, c_hi).astype(np.float32)

        ok = True
        if anchors is not None:
            d = np.linalg.norm(anchors - c[None, :], axis=1) - r
            if float(np.min(d)) < anc_margin:
                ok = False
        if ok and avoid_overlap and len(out) > 0:
            for c2, r2 in out:
                if float(np.linalg.norm(c - c2)) < (r + r2 + ov_margin):
                    ok = False
                    break
        if ok and avoid_overlap and len(existing) > 0:
            for c2, r2 in existing:
                if float(np.linalg.norm(c - c2)) < (r + r2 + ov_margin):
                    ok = False
                    break
        if ok and c_min_dist > 0.0 and len(out) > 0:
            for c2, _ in out:
                if float(np.linalg.norm(c - c2)) < c_min_dist:
                    ok = False
                    break
        if ok and c_min_dist > 0.0 and len(existing) > 0:
            for c2, _ in existing:
                if float(np.linalg.norm(c - c2)) < c_min_dist:
                    ok = False
                    break
        if ok:
            out.append((c, r))
            if mode == "grid3d":
                used_cells.add(int(cid))

    if len(out) < n:
        raise RuntimeError(f"[ERR] random sphere sampling failed: sampled {len(out)}/{n} after {tries} tries")
    return out


def _point_to_aabb_clearance(point_xyz, center_xyz, half_ext_xyz):
    p = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    c = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    h = np.asarray(half_ext_xyz, dtype=np.float32).reshape(3)
    d = np.abs(p - c) - h
    d_out = np.maximum(d, 0.0)
    return float(np.linalg.norm(d_out))


def _sphere_to_aabb_signed_clearance(sphere_center_xyz, sphere_r, box_center_xyz, box_half_ext_xyz):
    d = _point_to_aabb_clearance(sphere_center_xyz, box_center_xyz, box_half_ext_xyz)
    return float(d - float(sphere_r))


def sample_random_boxes_uniform(
    n_boxes,
    xyz_min,
    xyz_max,
    half_min,
    half_max,
    seed=0,
    anchor_points=None,
    anchor_clearance=0.0,
    avoid_overlap=False,
    overlap_margin=0.0,
    existing_spheres=None,
    max_tries=5000,
    sampling_mode="stratified",
    size_mode="mixed",
    center_min_dist=0.0,
):
    """
    Sample random axis-aligned boxes with optional:
      - anchor-point clearance
      - box-box non-overlap
      - sphere-box non-overlap (against existing spheres)
    Returns list[(center_xyz(np.float32[3]), half_ext_xyz(np.float32[3]))].
    """
    n = int(max(0, n_boxes))
    if n == 0:
        return []

    xyz_min = np.asarray(xyz_min, dtype=np.float32).reshape(3)
    xyz_max = np.asarray(xyz_max, dtype=np.float32).reshape(3)
    half_min = np.asarray(half_min, dtype=np.float32).reshape(3)
    half_max = np.asarray(half_max, dtype=np.float32).reshape(3)
    half_lo = np.minimum(half_min, half_max)
    half_hi = np.maximum(half_min, half_max)

    if np.any(xyz_max <= xyz_min):
        raise ValueError("[ERR] xyz_max must be > xyz_min for all dimensions")
    if np.any(half_lo <= 0.0):
        raise ValueError("[ERR] box half extents must be > 0")

    anchors = None
    if anchor_points is not None:
        ap = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 3)
        if ap.shape[0] > 0:
            anchors = ap
    anc_margin = float(max(0.0, anchor_clearance))
    ov_margin = float(max(0.0, overlap_margin))

    sph = []
    if existing_spheres is not None:
        for c, r in existing_spheres:
            sph.append((np.asarray(c, dtype=np.float32).reshape(3), float(r)))

    rng = np.random.default_rng(int(seed))
    mode = str(sampling_mode).strip().lower()
    if mode not in ("iid", "stratified", "grid3d"):
        raise ValueError(f"[ERR] unknown sampling_mode='{sampling_mode}', expected iid|stratified|grid3d")
    sz_mode = str(size_mode).strip().lower()
    if sz_mode not in ("uniform", "mixed"):
        raise ValueError(f"[ERR] unknown size_mode='{size_mode}', expected uniform|mixed")
    c_min_dist = float(max(0.0, center_min_dist))
    bin_ids = _size_bin_indices(n)

    grid_shape = _choose_grid_shape(n, xyz_min, xyz_max)
    n_cells = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
    cell_order = np.arange(n_cells, dtype=np.int64)
    rng.shuffle(cell_order)
    cell_ptr = 0
    used_cells = set()

    out = []
    tries = 0
    while len(out) < n and tries < int(max_tries):
        tries += 1
        if sz_mode == "uniform":
            h = rng.uniform(half_lo, half_hi).astype(np.float32)
        else:
            bid = int(bin_ids[len(out) % len(bin_ids)])
            h = np.asarray([
                _sample_in_size_bin(rng, float(half_lo[0]), float(half_hi[0]), bid),
                _sample_in_size_bin(rng, float(half_lo[1]), float(half_hi[1]), bid),
                _sample_in_size_bin(rng, float(half_lo[2]), float(half_hi[2]), bid),
            ], dtype=np.float32)
        g_lo = xyz_min + h
        g_hi = xyz_max - h
        if np.any(g_hi <= g_lo):
            continue
        if mode in ("stratified", "grid3d"):
            if mode == "grid3d":
                if len(used_cells) < n_cells:
                    loops = 0
                    cid = None
                    while loops < max(1, n_cells):
                        c_try = int(cell_order[cell_ptr % n_cells])
                        cell_ptr += 1
                        loops += 1
                        if c_try not in used_cells:
                            cid = c_try
                            break
                    if cid is None:
                        cid = int(cell_order[(tries - 1) % n_cells])
                else:
                    cid = int(cell_order[(tries - 1) % n_cells])
            else:
                cid = int(cell_order[(tries - 1) % n_cells])
                if ((tries - 1) % n_cells) == 0 and tries > 1:
                    rng.shuffle(cell_order)
            c_lo, c_hi = _cell_bounds(cid, grid_shape, xyz_min, xyz_max)
            c_lo = np.maximum(c_lo, g_lo)
            c_hi = np.minimum(c_hi, g_hi)
            if np.any(c_hi <= c_lo):
                c_lo = g_lo
                c_hi = g_hi
        else:
            c_lo = g_lo
            c_hi = g_hi
        c = rng.uniform(c_lo, c_hi).astype(np.float32)

        ok = True
        if anchors is not None:
            for p in anchors:
                if _point_to_aabb_clearance(p, c, h) < anc_margin:
                    ok = False
                    break
        if ok and avoid_overlap and len(out) > 0:
            for c2, h2 in out:
                if np.all(np.abs(c - c2) <= (h + h2 + ov_margin)):
                    ok = False
                    break
        if ok and c_min_dist > 0.0 and len(out) > 0:
            for c2, _ in out:
                if float(np.linalg.norm(c - c2)) < c_min_dist:
                    ok = False
                    break
        if ok and avoid_overlap and len(sph) > 0:
            for cs, rs in sph:
                if _sphere_to_aabb_signed_clearance(cs, rs, c, h) < ov_margin:
                    ok = False
                    break
        if ok:
            out.append((c, h))
            if mode == "grid3d":
                used_cells.add(int(cid))

    if len(out) < n:
        raise RuntimeError(f"[ERR] random box sampling failed: sampled {len(out)}/{n} after {tries} tries")
    return out


def boxes_to_proxy_spheres(
    boxes,
    radius_scale=0.65,
    max_per_box=8,
    min_radius=0.03,
):
    """
    Convert AABB boxes to a small set of proxy spheres (few, larger).
    Returns list[(center_xyz(np.float32[3]), r(float))].
    """
    out = []
    r_scale = float(max(1e-3, radius_scale))
    max_k = int(max(1, max_per_box))
    r_floor = float(max(1e-4, min_radius))

    for c, h in boxes:
        cc = np.asarray(c, dtype=np.float32).reshape(3)
        hh = np.asarray(h, dtype=np.float32).reshape(3)

        # choose a single radius per box (bigger => fewer effective spheres)
        h_min = float(np.min(hh))
        r = float(max(r_floor, min(h_min, h_min * r_scale)))
        if r <= 0.0:
            continue

        # decide grid resolution from max_k (NOT from box size)
        n_axis = int(max(1, round(max_k ** (1.0 / 3.0))))
        if max_k >= 8:
            n_axis = max(2, n_axis)

        # build a small grid inside the box
        xs = np.linspace(cc[0] - hh[0], cc[0] + hh[0], n_axis, dtype=np.float32)
        ys = np.linspace(cc[1] - hh[1], cc[1] + hh[1], n_axis, dtype=np.float32)
        zs = np.linspace(cc[2] - hh[2], cc[2] + hh[2], n_axis, dtype=np.float32)

        pts = np.stack(np.meshgrid(xs, ys, zs, indexing="xy"), axis=-1).reshape(-1, 3)

        # subsample uniformly if still too many
        if pts.shape[0] > max_k:
            idx = np.linspace(0, pts.shape[0] - 1, max_k, dtype=int)
            pts = pts[idx]

        for p in pts:
            out.append((p.astype(np.float32), r))

    return out


def parse_obstacle_spheres(spec_list):
    """
    Parse repeated sphere specs into list[(center_xyz(np.float32[3]), radius(float))].
    Accepts entries like:
      - "x,y,z,r"
      - (x, y, z, r)
      - [x, y, z, r]
    """
    out = []
    for raw in (spec_list or []):
        if isinstance(raw, (tuple, list, np.ndarray)) and len(raw) == 4:
            x, y, z, r = [float(v) for v in raw]
        else:
            s = str(raw).replace(" ", "")
            parts = s.split(",")
            if len(parts) != 4:
                raise ValueError(f"[ERR] obstacle sphere expects x,y,z,r, got: {raw}")
            x, y, z, r = [float(v) for v in parts]
        if r <= 0.0:
            raise ValueError(f"[ERR] sphere radius must be > 0, got {r} in: {raw}")
        out.append((np.array([x, y, z], dtype=np.float32), float(r)))
    return out


def min_clearance_to_spheres_traj9(traj9: np.ndarray, spheres) -> float:
    """
    Minimum signed clearance between UAV xyz trajectory and obstacle spheres.
    clearance = ||p-c|| - r ; collision if clearance < 0.
    """
    if traj9 is None or len(spheres) == 0:
        return float("inf")
    xyz = np.asarray(traj9[:, :3], dtype=np.float64)
    best = float("inf")
    for c, r in spheres:
        cc = np.asarray(c, dtype=np.float64).reshape(1, 3)
        d = np.linalg.norm(xyz - cc, axis=1) - float(r)
        m = float(np.min(d))
        if m < best:
            best = m
    return best


def _second_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]


def _third_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 3:, :] - 3.0 * x[:, 2:-1, :] + 3.0 * x[:, 1:-2, :] - x[:, :-3, :]


def _piecewise_ref_xyz(q_start_xyz, q_grasp_xyz, q_goal_xyz, H, t_mid):
    """
    Piecewise linear reference:
      [0..t_mid] start->grasp, [t_mid..H-1] grasp->goal.
    """
    t_mid_i = int(np.clip(int(t_mid), 1, int(max(1, H - 2))))
    t0 = torch.linspace(
        0.0, 1.0, t_mid_i + 1, device=q_start_xyz.device, dtype=q_start_xyz.dtype
    ).unsqueeze(-1)
    t1 = torch.linspace(
        0.0, 1.0, int(H - t_mid_i), device=q_start_xyz.device, dtype=q_start_xyz.dtype
    ).unsqueeze(-1)
    seg0 = (1.0 - t0) * q_start_xyz + t0 * q_grasp_xyz
    seg1 = (1.0 - t1) * q_grasp_xyz + t1 * q_goal_xyz
    return torch.cat([seg0, seg1[1:]], dim=0)


def _dist_point_to_segment(p, a, b, eps=1e-9):
    """
    p: (...,3), a:(3,), b:(3,) -> distance (...,)
    """
    ab = b - a
    ap = p - a
    denom = torch.sum(ab * ab, dim=-1, keepdim=True).clamp_min(float(eps))
    t = torch.sum(ap * ab, dim=-1, keepdim=True) / denom
    t = torch.clamp(t, 0.0, 1.0)
    proj = a + t * ab
    return torch.linalg.norm(p - proj, dim=-1)


def _piecewise_line_and_backtrack_losses(xyz, q_start_xyz, q_grasp_xyz, q_goal_xyz, t_mid, prog_eps=0.0):
    """
    xyz: (H,3), split at t_mid for start->grasp and grasp->goal segments.
    """
    H = int(xyz.shape[0])
    t_mid_i = int(np.clip(int(t_mid), 1, int(max(1, H - 2))))

    xyz0 = xyz[: t_mid_i + 1]
    a0, b0 = q_start_xyz, q_grasp_xyz
    d0 = _dist_point_to_segment(xyz0, a0, b0)
    dir0 = b0 - a0
    dir0 = dir0 / (torch.linalg.norm(dir0) + 1e-9)
    s0 = torch.sum((xyz0 - a0) * dir0, dim=-1)
    ds0 = s0[1:] - s0[:-1]
    back0 = torch.relu(float(prog_eps) - ds0)

    xyz1 = xyz[t_mid_i:]
    a1, b1 = q_grasp_xyz, q_goal_xyz
    d1 = _dist_point_to_segment(xyz1, a1, b1)
    dir1 = b1 - a1
    dir1 = dir1 / (torch.linalg.norm(dir1) + 1e-9)
    s1 = torch.sum((xyz1 - a1) * dir1, dim=-1)
    ds1 = s1[1:] - s1[:-1]
    back1 = torch.relu(float(prog_eps) - ds1)

    loss_line = d0.pow(2).mean() + d1.pow(2).mean()
    loss_back = back0.pow(2).mean() + back1.pow(2).mean()
    return loss_line, loss_back


def _cgd_collision_shift_xyz(xyz, spheres, margin, step, eps=1e-6):
    """
    CGD-style shift before projection optimization:
      xyz += step / ||d|| * d  (only when inside inflated sphere by margin).
    xyz: (H,3) or (B,H,3)
    spheres: (Ns,4) [cx,cy,cz,r]
    """
    if (spheres is None) or (spheres.numel() == 0):
        return xyz
    if float(step) <= 0.0:
        return xyz

    xyz_in = xyz
    flat = xyz_in.reshape(-1, 3) if xyz_in.ndim == 3 else xyz_in
    c = spheres[:, :3]
    r = spheres[:, 3] + float(margin)

    diff = flat[:, None, :] - c[None, :, :]
    dist = torch.linalg.norm(diff, dim=-1).clamp_min(float(eps))
    clr = dist - r[None, :]

    j = torch.argmin(clr, dim=1)
    cmin = c[j]
    rmin = r[j]
    d = flat - cmin
    dn = torch.linalg.norm(d, dim=-1).clamp_min(float(eps))
    clr_min = dn - rmin
    mask = (clr_min < 0.0).to(flat.dtype).unsqueeze(-1)

    flat_new = flat + mask * (float(step) / dn).unsqueeze(-1) * d
    if xyz_in.ndim == 3:
        return flat_new.reshape_as(xyz_in)
    return flat_new


@torch.enable_grad()
def obstacle_project_batch(
    traj0: torch.Tensor,
    hard_mask: torch.Tensor,
    spheres,
    n_iters: int = 30,
    lr: float = 0.01,
    w_data: float = 10.0,
    w_v: float = 8.0,
    w_a: float = 50.0,
    w_j: float = 10.0,
    w_len: float = 0.0,
    w_ref: float = 0.0,
    w_len_ratio: float = 0.0,
    max_len_ratio: float = 0.0,
    ref_xyz: torch.Tensor = None,
    straight_len: torch.Tensor = None,
    w_obst: float = 30.0,
    obst_margin: float = 0.10,
    w_uav_v: float = 0.0,
    w_uav_a: float = 0.0,
    w_uav_j: float = 0.0,
    w_uav_line: float = 0.0,
    w_uav_back: float = 0.0,
    w_uav_turn: float = 0.0,
    turn_theta_max_deg: float = 35.0,
    w_uav_curv: float = 0.0,
    prog_eps: float = 0.0,
    q_start_xyz: torch.Tensor = None,
    q_grasp_xyz: torch.Tensor = None,
    q_goal_xyz: torch.Tensor = None,
    t_mid_idx: torch.Tensor = None,
    cgd_shift_step: float = 0.0,
    max_grad_value: float = 0.1,
    max_delta: float = 0.05,
) -> torch.Tensor:
    """
    Lightweight post-projection after diffusion guidance:
      min w_data||x-x0||^2 + w_v||D1 x||^2 + w_a||D2 x||^2 + w_j||D3 x||^2
          + w_len*path_len + w_ref*||xyz-ref_xyz||^2
          + w_len_ratio*hinge(path_len/straight_len - max_len_ratio)^2
          + w_uav_v||D1 xyz||^2 + w_uav_a||D2 xyz||^2 + w_uav_j||D3 xyz||^2
          + w_uav_curv||D2 xyz||^2 + w_uav_turn*turn_hinge_xy
          + w_uav_line*dist_to_start_goal_line^2 + w_uav_back*backtracking_penalty
          + w_obst*hinge(clearance)^2
      s.t. hard anchor states stay fixed (start/goal/grasp indices).
    """
    if n_iters <= 0:
        return traj0

    x0 = traj0.detach()
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=float(lr))

    sph = [(torch.as_tensor(c, dtype=x0.dtype, device=x0.device).view(1, 1, 3), float(r)) for c, r in spheres]
    if len(spheres) > 0:
        sph4 = torch.as_tensor(
            [[float(c[0]), float(c[1]), float(c[2]), float(r)] for c, r in spheres],
            dtype=x0.dtype,
            device=x0.device,
        )
    else:
        sph4 = torch.zeros((0, 4), dtype=x0.dtype, device=x0.device)
    margin = float(max(0.0, obst_margin))
    ref_xyz_t = None
    if ref_xyz is not None:
        ref_xyz_t = torch.as_tensor(ref_xyz, dtype=x0.dtype, device=x0.device)
        if ref_xyz_t.ndim == 2:
            ref_xyz_t = ref_xyz_t.unsqueeze(0)
        if ref_xyz_t.shape[0] == 1 and x0.shape[0] > 1:
            ref_xyz_t = ref_xyz_t.expand(x0.shape[0], -1, -1)
        if ref_xyz_t.shape[0] != x0.shape[0] or ref_xyz_t.shape[1] != x0.shape[1] or ref_xyz_t.shape[2] != 3:
            raise ValueError(
                f"[ERR] ref_xyz shape mismatch: got {tuple(ref_xyz_t.shape)}, "
                f"expected ({x0.shape[0]}, {x0.shape[1]}, 3)"
            )
    straight_len_t = None
    if straight_len is not None:
        straight_len_t = torch.as_tensor(straight_len, dtype=x0.dtype, device=x0.device).reshape(-1)
        if straight_len_t.numel() == 1 and x0.shape[0] > 1:
            straight_len_t = straight_len_t.expand(x0.shape[0])
        if straight_len_t.numel() != x0.shape[0]:
            raise ValueError(
                f"[ERR] straight_len size mismatch: got {straight_len_t.numel()}, expected {x0.shape[0]}"
            )
        straight_len_t = torch.clamp(straight_len_t, min=1e-6)

    B, H, _ = x0.shape

    def _as_bx3(v):
        if v is None:
            return None
        t = torch.as_tensor(v, dtype=x0.dtype, device=x0.device)
        if t.ndim == 1:
            if t.numel() != 3:
                raise ValueError(f"[ERR] expected xyz shape (*,3), got {tuple(t.shape)}")
            t = t.view(1, 3)
        elif t.ndim == 2 and t.shape[1] == 3:
            pass
        else:
            raise ValueError(f"[ERR] expected xyz shape (*,3), got {tuple(t.shape)}")
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1)
        if t.shape[0] != B:
            raise ValueError(f"[ERR] xyz batch mismatch: got {t.shape[0]}, expected {B}")
        return t

    q_start_xyz_t = _as_bx3(q_start_xyz)
    q_grasp_xyz_t = _as_bx3(q_grasp_xyz)
    q_goal_xyz_t = _as_bx3(q_goal_xyz)
    t_mid_idx_t = None
    if t_mid_idx is not None:
        t_mid_idx_t = torch.as_tensor(t_mid_idx, dtype=torch.long, device=x0.device).reshape(-1)
        if t_mid_idx_t.numel() == 1 and B > 1:
            t_mid_idx_t = t_mid_idx_t.expand(B)
        if t_mid_idx_t.numel() != B:
            raise ValueError(f"[ERR] t_mid_idx batch mismatch: got {t_mid_idx_t.numel()}, expected {B}")
        t_mid_idx_t = torch.clamp(t_mid_idx_t, min=1, max=max(1, H - 2))

    for _ in range(int(n_iters)):
        if (float(cgd_shift_step) > 0.0) and (sph4.shape[0] > 0):
            with torch.no_grad():
                x[..., :3] = _cgd_collision_shift_xyz(
                    x[..., :3], spheres=sph4, margin=margin, step=float(cgd_shift_step)
                )
                x[hard_mask] = x0[hard_mask]
        opt.zero_grad(set_to_none=True)

        loss = 0.0
        xyz = x[..., :3]
        if w_data > 0.0:
            loss = loss + float(w_data) * torch.mean((x - x0) ** 2)
        if w_v > 0.0 and x.shape[1] >= 2:
            v = x[:, 1:, :] - x[:, :-1, :]
            loss = loss + float(w_v) * torch.mean(v * v)
        if w_a > 0.0 and x.shape[1] >= 3:
            loss = loss + float(w_a) * torch.mean(_second_diff(x) ** 2)
        if w_j > 0.0 and x.shape[1] >= 4:
            loss = loss + float(w_j) * torch.mean(_third_diff(x) ** 2)
        if w_len > 0.0 and x.shape[1] >= 2:
            step_len = torch.linalg.norm(xyz[:, 1:, :] - xyz[:, :-1, :], dim=-1)
            path_len = torch.sum(step_len, dim=1)
            loss = loss + float(w_len) * torch.mean(path_len)
        if w_ref > 0.0 and ref_xyz_t is not None:
            loss = loss + float(w_ref) * torch.mean((xyz - ref_xyz_t) ** 2)
        if w_len_ratio > 0.0 and max_len_ratio > 0.0 and x.shape[1] >= 2 and straight_len_t is not None:
            step_len = torch.linalg.norm(xyz[:, 1:, :] - xyz[:, :-1, :], dim=-1)
            path_len = torch.sum(step_len, dim=1)
            ratio = path_len / straight_len_t
            pen_ratio = torch.clamp(ratio - float(max_len_ratio), min=0.0)
            loss = loss + float(w_len_ratio) * torch.mean(pen_ratio * pen_ratio)

        if w_obst > 0.0 and len(sph) > 0:
            loss_obst = 0.0
            for c, r in sph:
                dist = torch.linalg.norm(xyz - c, dim=-1)
                clearance = dist - r
                pen = torch.clamp(margin - clearance, min=0.0)
                loss_obst = loss_obst + torch.mean(pen * pen)
            loss = loss + float(w_obst) * loss_obst

        # --- UAV-shape regularization (operate on UAV xyz only) ---
        if xyz.shape[1] >= 2:
            d1_xyz = xyz[:, 1:, :] - xyz[:, :-1, :]
            if w_uav_v > 0.0:
                loss_uav_v = torch.mean(d1_xyz * d1_xyz)
                loss = loss + float(w_uav_v) * loss_uav_v
        if xyz.shape[1] >= 3:
            d2_xyz = xyz[:, 2:, :] - 2.0 * xyz[:, 1:-1, :] + xyz[:, :-2, :]
            if w_uav_a > 0.0:
                loss_uav_a = torch.mean(d2_xyz * d2_xyz)
                loss = loss + float(w_uav_a) * loss_uav_a
            if w_uav_curv > 0.0:
                loss_uav_curv = torch.mean(d2_xyz * d2_xyz)
                loss = loss + float(w_uav_curv) * loss_uav_curv
            if w_uav_j > 0.0 and xyz.shape[1] >= 4:
                d3_xyz = d2_xyz[:, 1:, :] - d2_xyz[:, :-1, :]
                loss_uav_j = torch.mean(d3_xyz * d3_xyz)
                loss = loss + float(w_uav_j) * loss_uav_j

        if w_uav_turn > 0.0 and xyz.shape[1] >= 3:
            v_prev = xyz[:, 1:-1, :] - xyz[:, :-2, :]
            v_next = xyz[:, 2:, :] - xyz[:, 1:-1, :]
            n_prev = torch.linalg.norm(v_prev, dim=-1).clamp_min(1e-9)
            n_next = torch.linalg.norm(v_next, dim=-1).clamp_min(1e-9)
            cosang = torch.sum(v_prev * v_next, dim=-1) / (n_prev * n_next)
            cosang = torch.clamp(cosang, min=-1.0, max=1.0)
            cos_thr = float(np.cos(np.deg2rad(float(turn_theta_max_deg))))
            pen_turn = torch.relu(cos_thr - cosang)
            loss_uav_turn = torch.mean(pen_turn * pen_turn)
            loss = loss + float(w_uav_turn) * loss_uav_turn

        if (w_uav_line > 0.0 or w_uav_back > 0.0) and xyz.shape[1] >= 2:
            use_piecewise = (
                (q_start_xyz_t is not None)
                and (q_grasp_xyz_t is not None)
                and (q_goal_xyz_t is not None)
                and (t_mid_idx_t is not None)
            )
            if use_piecewise:
                loss_line_acc = torch.zeros((), dtype=x.dtype, device=x.device)
                loss_back_acc = torch.zeros((), dtype=x.dtype, device=x.device)
                for bi in range(B):
                    l_line, l_back = _piecewise_line_and_backtrack_losses(
                        xyz=xyz[bi],
                        q_start_xyz=q_start_xyz_t[bi],
                        q_grasp_xyz=q_grasp_xyz_t[bi],
                        q_goal_xyz=q_goal_xyz_t[bi],
                        t_mid=int(t_mid_idx_t[bi].item()),
                        prog_eps=float(prog_eps),
                    )
                    loss_line_acc = loss_line_acc + l_line
                    loss_back_acc = loss_back_acc + l_back
                loss_line_acc = loss_line_acc / float(B)
                loss_back_acc = loss_back_acc / float(B)
                if w_uav_line > 0.0:
                    loss = loss + float(w_uav_line) * loss_line_acc
                if w_uav_back > 0.0:
                    loss = loss + float(w_uav_back) * loss_back_acc
            else:
                a = xyz[:, 0:1, :]
                b = xyz[:, -1:, :]
                u = b - a
                u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-9)
                ap = xyz - a
                s = torch.sum(ap * u, dim=-1)  # [B,H]

                if w_uav_line > 0.0:
                    t_line = s.unsqueeze(-1)
                    proj = a + t_line * u
                    perp = xyz - proj
                    loss_uav_line = torch.mean(perp * perp)
                    loss = loss + float(w_uav_line) * loss_uav_line

                if w_uav_back > 0.0:
                    ds = s[:, 1:] - s[:, :-1]
                    eps_prog = float(prog_eps)
                    loss_uav_back = torch.mean(torch.relu(eps_prog - ds) ** 2)
                    loss = loss + float(w_uav_back) * loss_uav_back
        # --- end UAV-shape regularization ---

        loss.backward()
        if max_grad_value > 0.0:
            torch.nn.utils.clip_grad_value_([x], float(max_grad_value))
        opt.step()

        with torch.no_grad():
            x[hard_mask] = x0[hard_mask]
            if max_delta > 0.0:
                dx = torch.clamp(x - x0, min=-float(max_delta), max=float(max_delta))
                x.copy_(x0 + dx)

    return x.detach()


def apply_obstacle_projection_with_adaptive(
    traj: torch.Tensor,
    hard_mask: torch.Tensor,
    spheres,
    proj_cfg: dict,
    verbose_fn=None,
):
    """
    Apply base projection then optional adaptive rounds on still-in-collision samples.
    Returns:
      - projected trajectory tensor
      - diagnostics dict
    """
    if len(spheres) == 0:
        return traj, {"bad_final": 0}

    n_iters = int(proj_cfg.get("iters", 30))
    lr = float(proj_cfg.get("lr", 0.01))
    w_data = float(proj_cfg.get("w_data", 10.0))
    w_v = float(proj_cfg.get("w_v", 8.0))
    w_a = float(proj_cfg.get("w_a", 50.0))
    w_j = float(proj_cfg.get("w_j", 10.0))
    w_len = float(proj_cfg.get("w_len", 0.0))
    w_ref = float(proj_cfg.get("w_ref", 0.0))
    w_len_ratio = float(proj_cfg.get("w_len_ratio", 0.0))
    max_len_ratio = float(proj_cfg.get("max_len_ratio", 0.0))
    ref_xyz = proj_cfg.get("ref_xyz", None)
    straight_len = proj_cfg.get("straight_len", None)
    w_obst = float(proj_cfg.get("w_obst", 30.0))
    margin = float(proj_cfg.get("margin", 0.10))
    w_uav_v = float(proj_cfg.get("w_uav_v", 0.0))
    w_uav_a = float(proj_cfg.get("w_uav_a", 0.0))
    w_uav_j = float(proj_cfg.get("w_uav_j", 0.0))
    w_uav_line = float(proj_cfg.get("w_uav_line", 0.0))
    w_uav_back = float(proj_cfg.get("w_uav_back", 0.0))
    w_uav_turn = float(proj_cfg.get("w_uav_turn", 40.0))
    turn_theta_max_deg = float(proj_cfg.get("turn_theta_max_deg", 35.0))
    w_uav_curv = float(proj_cfg.get("w_uav_curv", 20.0))
    prog_eps = float(proj_cfg.get("prog_eps", 0.0))
    cgd_shift_step = float(proj_cfg.get("cgd_shift_step", 0.0))
    q_start_xyz = proj_cfg.get("q_start_xyz", None)
    q_grasp_xyz = proj_cfg.get("q_grasp_xyz", None)
    q_goal_xyz = proj_cfg.get("q_goal_xyz", None)
    t_mid_idx = proj_cfg.get("t_mid_idx", None)
    max_grad = float(proj_cfg.get("max_grad_value", 0.1))
    max_delta = float(proj_cfg.get("max_delta", 0.05))
    if verbose_fn is not None and (not bool(proj_cfg.get("_proj_cfg_printed", False))):
        verbose_fn(
            f"[OBST_PROJ_CFG] iters={n_iters} lr={lr} "
            f"w_data={w_data} w_v={w_v} w_a={w_a} w_j={w_j} "
            f"w_len={w_len} w_ref={w_ref} w_len_ratio={w_len_ratio}@{max_len_ratio} "
            f"w_obst={w_obst} margin={margin} "
            f"w_uav_v={w_uav_v} w_uav_a={w_uav_a} w_uav_j={w_uav_j} "
            f"w_uav_line={w_uav_line} w_uav_back={w_uav_back} "
            f"w_uav_turn={w_uav_turn} turn_theta_max_deg={turn_theta_max_deg} w_uav_curv={w_uav_curv} "
            f"prog_eps={prog_eps} "
            f"cgd_shift_step={cgd_shift_step} "
            f"max_grad={max_grad} max_delta={max_delta}"
        )
        proj_cfg["_proj_cfg_printed"] = True

    out = obstacle_project_batch(
        traj0=traj,
        hard_mask=hard_mask,
        spheres=spheres,
        n_iters=n_iters,
        lr=lr,
        w_data=w_data,
        w_v=w_v,
        w_a=w_a,
        w_j=w_j,
        w_len=w_len,
        w_ref=w_ref,
        w_len_ratio=w_len_ratio,
        max_len_ratio=max_len_ratio,
        ref_xyz=ref_xyz,
        straight_len=straight_len,
        w_obst=w_obst,
        obst_margin=margin,
        w_uav_v=w_uav_v,
        w_uav_a=w_uav_a,
        w_uav_j=w_uav_j,
        w_uav_line=w_uav_line,
        w_uav_back=w_uav_back,
        w_uav_turn=w_uav_turn,
        turn_theta_max_deg=turn_theta_max_deg,
        w_uav_curv=w_uav_curv,
        prog_eps=prog_eps,
        q_start_xyz=q_start_xyz,
        q_grasp_xyz=q_grasp_xyz,
        q_goal_xyz=q_goal_xyz,
        t_mid_idx=t_mid_idx,
        cgd_shift_step=cgd_shift_step,
        max_grad_value=max_grad,
        max_delta=max_delta,
    )

    adaptive_enable = bool(proj_cfg.get("adaptive_enable", False))
    if not adaptive_enable:
        return out, {"bad_final": 0}

    rounds = int(max(0, proj_cfg.get("adaptive_rounds", 2)))
    clr_thr = float(proj_cfg.get("adaptive_clearance", 0.0))
    it_mult = float(proj_cfg.get("adaptive_iters_mult", 2.0))
    w_mult = float(proj_cfg.get("adaptive_w_obst_mult", 2.0))
    dx_mult = float(proj_cfg.get("adaptive_max_delta_mult", 2.0))
    g_mult = float(proj_cfg.get("adaptive_max_grad_mult", 1.5))

    s_count = int(out.shape[0])
    out_np = out.detach().cpu().numpy()
    clr = np.asarray([min_clearance_to_spheres_traj9(out_np[s], spheres) for s in range(s_count)], dtype=np.float64)
    bad = np.where(clr < clr_thr)[0]

    for rr in range(rounds):
        if bad.size == 0:
            break
        clr_min_before = float(np.min(clr[bad]))
        bad_prev = bad.copy()
        out_bad_prev = out[bad_prev].detach().clone()
        fac_it = it_mult ** float(rr + 1)
        fac_w = w_mult ** float(rr + 1)
        fac_dx = dx_mult ** float(rr + 1)
        fac_g = g_mult ** float(rr + 1)
        ref_xyz_bad = ref_xyz
        if torch.is_tensor(ref_xyz) and ref_xyz.ndim >= 3 and ref_xyz.shape[0] == s_count:
            ref_xyz_bad = ref_xyz[bad]
        straight_len_bad = straight_len
        if torch.is_tensor(straight_len):
            st = straight_len.reshape(-1)
            if st.numel() == s_count:
                straight_len_bad = st[bad]

        out_bad = obstacle_project_batch(
            traj0=out[bad],
            hard_mask=hard_mask[bad],
            spheres=spheres,
            n_iters=int(max(1, round(n_iters * fac_it))),
            lr=lr,
            w_data=w_data,
            w_v=w_v,
            w_a=w_a,
            w_j=w_j,
            w_len=w_len,
            w_ref=w_ref,
            w_len_ratio=w_len_ratio,
            max_len_ratio=max_len_ratio,
            ref_xyz=ref_xyz_bad,
            straight_len=straight_len_bad,
            w_obst=w_obst * fac_w,
            obst_margin=margin,
            w_uav_v=w_uav_v,
            w_uav_a=w_uav_a,
            w_uav_j=w_uav_j,
            w_uav_line=w_uav_line,
            w_uav_back=w_uav_back,
            w_uav_turn=w_uav_turn,
            turn_theta_max_deg=turn_theta_max_deg,
            w_uav_curv=w_uav_curv,
            prog_eps=prog_eps,
            q_start_xyz=q_start_xyz[bad] if torch.is_tensor(q_start_xyz) and q_start_xyz.ndim >= 2 and q_start_xyz.shape[0] == s_count else q_start_xyz,
            q_grasp_xyz=q_grasp_xyz[bad] if torch.is_tensor(q_grasp_xyz) and q_grasp_xyz.ndim >= 2 and q_grasp_xyz.shape[0] == s_count else q_grasp_xyz,
            q_goal_xyz=q_goal_xyz[bad] if torch.is_tensor(q_goal_xyz) and q_goal_xyz.ndim >= 2 and q_goal_xyz.shape[0] == s_count else q_goal_xyz,
            t_mid_idx=t_mid_idx[bad] if torch.is_tensor(t_mid_idx) and t_mid_idx.ndim >= 1 and t_mid_idx.shape[0] == s_count else t_mid_idx,
            cgd_shift_step=cgd_shift_step,
            max_grad_value=max_grad * fac_g,
            max_delta=max_delta * fac_dx,
        )
        out[bad] = out_bad

        out_np = out.detach().cpu().numpy()
        clr = np.asarray([min_clearance_to_spheres_traj9(out_np[s], spheres) for s in range(s_count)], dtype=np.float64)
        bad_new = np.where(clr < clr_thr)[0]
        clr_min_after = float(np.min(clr[bad_new])) if bad_new.size > 0 else float("inf")
        if verbose_fn is not None:
            verbose_fn(
                f"[OBST_PROJ_ADAPT] round={rr + 1} bad_before={bad.size} bad_after={bad_new.size} "
                f"clr_min_before={clr_min_before:.6f} clr_min_after={clr_min_after:.6f}"
            )
        # Optional safety: rollback this round if it significantly worsens worst bad clearance.
        if np.isfinite(clr_min_before) and np.isfinite(clr_min_after) and (clr_min_after < clr_min_before - 1e-3):
            out[bad_prev] = out_bad_prev
            out_np = out.detach().cpu().numpy()
            clr = np.asarray([min_clearance_to_spheres_traj9(out_np[s], spheres) for s in range(s_count)], dtype=np.float64)
            bad = np.where(clr < clr_thr)[0]
            if verbose_fn is not None:
                verbose_fn(
                    f"[OBST_PROJ_ADAPT] round={rr + 1} rollback=1 "
                    f"(worse clearance: {clr_min_before:.6f}->{clr_min_after:.6f}), stop_adaptive=1"
                )
            break
        bad = bad_new

    return out, {"bad_final": int(bad.size)}


class CGDSphereGuide:
    """
    CGD-style cheap obstacle guide in diffusion loop.
    Push xyz away from obstacle spheres with ~ alpha/dist * dir near obstacles.
    Returned tensor is treated as a guidance gradient in DDIM.
    """

    def __init__(
        self,
        spheres,
        horizon,
        hard_idx,
        alpha=1.0,
        margin=0.15,
        eps=1e-4,
        max_push=0.2,
        time_smooth_k=5,
        alpha_ramp_power=2.0,
        alpha_min_scale=0.0,
        w_obst=1.0,
        w_refline=0.0,
        w_curv=0.0,
        obst_sigma_power=0.0,
        refline_sigma_power=0.0,
        curv_sigma_power=0.0,
        device="cpu",
    ):
        self.spheres = [(torch.as_tensor(c, dtype=torch.float32, device=device), float(r)) for c, r in spheres]
        self.horizon = int(horizon)
        self.hard_idx = sorted(set(int(i) for i in hard_idx if 0 <= int(i) < self.horizon))
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.eps = float(eps)
        self.max_push = float(max_push)
        self.time_smooth_k = int(max(0, time_smooth_k))
        self.alpha_ramp_power = float(max(0.0, alpha_ramp_power))
        self.alpha_min_scale = float(np.clip(alpha_min_scale, 0.0, 1.0))
        self.w_obst = float(w_obst)
        self.w_refline = float(w_refline)
        self.w_curv = float(w_curv)
        self.obst_sigma_power = float(obst_sigma_power)
        self.refline_sigma_power = float(refline_sigma_power)
        self.curv_sigma_power = float(curv_sigma_power)

    def __call__(self, x, context_d=None, guide_progress=None, **kwargs):
        grad = torch.zeros_like(x)
        xyz = x[..., :3]  # [B,H,3]
        B, H, _ = xyz.shape
        if len(self.spheres) == 0:
            return grad

        if guide_progress is None:
            alpha_scale = 1.0
        else:
            gp = float(np.clip(guide_progress, 0.0, 1.0))
            alpha_scale = self.alpha_min_scale + (1.0 - self.alpha_min_scale) * (gp ** self.alpha_ramp_power)
        alpha_eff = self.alpha * alpha_scale
        sigma_ratio = float(max(self.eps, kwargs.get("sigma_ratio", 1.0)))
        w_obst_t = self.w_obst * (sigma_ratio ** self.obst_sigma_power)
        w_refline_t = self.w_refline * (sigma_ratio ** self.refline_sigma_power)
        w_curv_t = self.w_curv * (sigma_ratio ** self.curv_sigma_power)

        centers = torch.stack([c for c, _r in self.spheres], dim=0).to(device=xyz.device, dtype=xyz.dtype)  # [M,3]
        radii = torch.tensor([float(_r) for _c, _r in self.spheres], device=xyz.device, dtype=xyz.dtype)  # [M]

        # diff/dist/clearance: [B,H,M]
        diff = xyz[:, :, None, :] - centers[None, None, :, :]  # [B,H,M,3]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(self.eps)  # [B,H,M]
        clearance = dist - radii[None, None, :]  # [B,H,M]

        # pick the most dangerous sphere per (B,H): minimal clearance
        idx = torch.argmin(clearance, dim=-1)  # [B,H]
        idx3 = idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, 1, 3)  # [B,H,1,3]
        idx1 = idx.unsqueeze(-1).expand(B, H, 1)  # [B,H,1]

        diff_min = diff.gather(2, idx3).squeeze(2)  # [B,H,3]
        dist_min = dist.gather(2, idx1).squeeze(2).unsqueeze(-1)  # [B,H,1]
        clr_min = clearance.gather(2, idx1).squeeze(2).unsqueeze(-1)  # [B,H,1]
        direction = diff_min / dist_min  # [B,H,3]

        # hinge near obstacles only
        pen = torch.clamp(self.margin - clr_min, min=0.0)  # [B,H,1]
        near_w = torch.clamp(pen / (self.margin + self.eps), min=0.0, max=1.0)  # [B,H,1]

        # push magnitude: alpha / clearance (clipped)
        push_mag = alpha_eff / torch.clamp(clr_min, min=self.eps)  # [B,H,1]
        if self.max_push > 0.0:
            push_mag = torch.clamp(push_mag, max=self.max_push)

        grad_xyz = w_obst_t * (near_w * push_mag * direction)

        # Straight-line reference: encourage direct path from current start to current goal.
        if w_refline_t > 0.0 and H >= 2:
            tau = torch.linspace(0.0, 1.0, H, device=xyz.device, dtype=xyz.dtype).view(1, H, 1)
            start = xyz[:, :1, :]
            goal = xyz[:, -1:, :]
            ref = start + tau * (goal - start)
            # d/dx mean(||x-ref||^2)
            grad_ref = (2.0 / float(max(1, H))) * (xyz - ref)
            grad_xyz = grad_xyz + w_refline_t * grad_ref

        # Curvature penalty: discourage zig-zag turns.
        if w_curv_t > 0.0 and H >= 3:
            d2 = xyz[:, 2:, :] - 2.0 * xyz[:, 1:-1, :] + xyz[:, :-2, :]
            grad_curv = torch.zeros_like(xyz)
            grad_curv[:, :-2, :] += d2
            grad_curv[:, 1:-1, :] += -2.0 * d2
            grad_curv[:, 2:, :] += d2
            grad_curv = (2.0 / float(max(1, H - 2))) * grad_curv
            grad_xyz = grad_xyz + w_curv_t * grad_curv

        grad[..., :3] = grad_xyz

        if self.time_smooth_k >= 3 and (self.time_smooth_k % 2 == 1):
            k = self.time_smooth_k
            pad = k // 2
            g = grad[..., :3].transpose(1, 2)
            g = F.pad(g, (pad, pad), mode="replicate")
            kernel = torch.ones((3, 1, k), device=g.device, dtype=g.dtype) / float(k)
            g = F.conv1d(g, kernel, groups=3)
            grad[..., :3] = g.transpose(1, 2)

        if len(self.hard_idx) > 0:
            grad[:, self.hard_idx, :] = 0.0
        return grad


def build_ddim_obstacle_kwargs_from_cfg(obstacle_cfg, horizon, hard_conds, device):
    """
    Build DDIM sampling kwargs from a unified obstacle_cfg dictionary.
    """
    if obstacle_cfg is None or not bool(obstacle_cfg.get("enable", False)):
        return {}
    spheres = parse_obstacle_spheres(obstacle_cfg.get("spheres", []))
    pc_cfg = obstacle_cfg.get("point_cloud", None)
    if pc_cfg is not None:
        pc_points = np.asarray(pc_cfg.get("points", []), dtype=np.float32).reshape(-1, 3)
        if pc_points.shape[0] > 0:
            spheres += spheres_from_point_cloud(
                points=pc_points,
                point_radius=float(pc_cfg.get("point_radius", 0.02)),
                max_points=int(pc_cfg.get("max_points", 256)),
                seed=int(pc_cfg.get("seed", 0)),
            )
    if len(spheres) == 0:
        return {}

    guide_cfg = obstacle_cfg.get("guide", {})
    hard_idx = sorted(set(int(k) for k in (hard_conds or {}).keys() if 0 <= int(k) < int(horizon)))
    guide_fn = CGDSphereGuide(
        spheres=spheres,
        horizon=int(horizon),
        hard_idx=hard_idx,
        alpha=float(guide_cfg.get("alpha", 1.0)),
        margin=float(guide_cfg.get("margin", 0.15)),
        eps=float(guide_cfg.get("eps", 1e-4)),
        max_push=float(guide_cfg.get("max_push", 0.2)),
        time_smooth_k=int(guide_cfg.get("time_smooth_k", 5)),
        alpha_ramp_power=float(guide_cfg.get("alpha_ramp_power", 2.0)),
        alpha_min_scale=float(guide_cfg.get("alpha_min_scale", 0.0)),
        w_obst=float(guide_cfg.get("w_obst", 1.0)),
        w_refline=float(guide_cfg.get("w_refline", 0.0)),
        w_curv=float(guide_cfg.get("w_curv", 0.0)),
        obst_sigma_power=float(guide_cfg.get("obst_sigma_power", 0.0)),
        refline_sigma_power=float(guide_cfg.get("refline_sigma_power", 0.0)),
        curv_sigma_power=float(guide_cfg.get("curv_sigma_power", 0.0)),
        device=device,
    )

    out = {
        "method": "ddim",
        "ddim_sampling_timesteps": int(guide_cfg.get("ddim_steps", 100)),
        "t_start_guide": int(guide_cfg.get("t_start_guide", 40)),
        "guide": guide_fn,
        "guide_lr": float(guide_cfg.get("guide_lr", 0.08)),
        "guide_lr_sigma_power": float(guide_cfg.get("guide_lr_sigma_power", 1.0)),
        "n_guide_steps": int(guide_cfg.get("n_guide_steps", 1)),
        "scale_grad_by_one_minus_alpha": bool(guide_cfg.get("scale_grad_by_one_minus_alpha", False)),
        "clip_grad": bool(guide_cfg.get("clip_grad", True)),
        "clip_grad_rule": str(guide_cfg.get("clip_grad_rule", "value")),
        "max_grad_value": float(guide_cfg.get("max_grad_value", 1.0)),
        "max_perturb_x": float(guide_cfg.get("max_perturb_x", 0.1)),
        "compute_costs_with_xrecon": bool(guide_cfg.get("use_xrecon", True)),
    }
    return out


def make_proj_only_adaptive_preset(spheres):
    """
    Default preset for obstacle-aware DDIM guide + post projection.
    """
    return {
        "enable": True,
        "spheres": list(spheres),
        "guide": {
            "alpha": 0.12,
            "margin": 0.10,
            "max_push": 0.06,
            "ddim_steps": 100,
            "t_start_guide": 60,
            "guide_lr": 0.01,
            "guide_lr_sigma_power": 1.0,
            "n_guide_steps": 4,
            "time_smooth_k": 11,
            "alpha_ramp_power": 3.0,
            "alpha_min_scale": 0.0,
            "w_obst": 1.0,
            "w_refline": 30.0,
            "w_curv": 10.0,
            "obst_sigma_power": -1.0,
            "refline_sigma_power": -0.5,
            "curv_sigma_power": -0.5,
            "scale_grad_by_one_minus_alpha": True,
            "clip_grad": True,
            "clip_grad_rule": "value",
            "max_grad_value": 0.3,
            "max_perturb_x": 0.06,
            "use_xrecon": True,
        },
        "project": {
            "enable": True,
            "iters": 200,
            "lr": 0.01,
            "w_data": 0.5,
            "w_v": 0.0,
            "w_a": 0.0,
            "w_j": 0.0,
            "w_len": 0.0,
            "w_ref": 0.0,
            "w_len_ratio": 0.0,
            "max_len_ratio": 0.0,
            "w_uav_v": 40.0,
            "w_uav_a": 120.0,
            "w_uav_j": 80.0,
            "w_uav_line": 30.0,
            "w_uav_back": 80.0,
            "w_uav_turn": 40.0,
            "turn_theta_max_deg": 35.0,
            "w_uav_curv": 20.0,
            "prog_eps": 0.0,
            "cgd_shift_step": 0.02,
            "w_obst": 120.0,
            "margin": 0.05,
            "max_grad_value": 0.06,
            "max_delta": 0.06,
            "adaptive_enable": True,
            "adaptive_rounds": 2,
            "adaptive_clearance": 0.0,
            "adaptive_iters_mult": 2.0,
            "adaptive_w_obst_mult": 2.0,
            "adaptive_max_delta_mult": 2.0,
            "adaptive_max_grad_mult": 1.5,
            "post_proj_cfg": {
                "enable": True,
                "iters": 80,
                "lr": 0.01,
                "w_data": 1.0,
                "w_v": 20.0,
                "w_a": 80.0,
                "w_j": 40.0,
                "w_len": 5.0,
                "w_ref": 0.0,
                "w_len_ratio": 0.0,
                "max_len_ratio": 0.0,
                "w_uav_turn": 40.0,
                "turn_theta_max_deg": 35.0,
                "w_uav_curv": 20.0,
                "w_obst": 120.0,
                "margin": 0.05,
                "max_grad_value": 0.03,
                "max_delta": 0.02,
                "adaptive_enable": False,
            },
        },
    }


def make_shortcut_post_proj_cfg(project_cfg=None):
    """
    Build post-shortcut projection config.
    Uses stable defaults and allows optional override from project_cfg["post_proj_cfg"].
    """
    cfg = {
        "enable": True,
        "iters": 80,
        "lr": 0.01,
        "w_data": 1.0,
        "w_v": 20.0,
        "w_a": 80.0,
        "w_j": 40.0,
        "w_len": 5.0,
        "w_ref": 0.0,
        "w_len_ratio": 0.0,
        "max_len_ratio": 0.0,
        "w_uav_turn": 40.0,
        "turn_theta_max_deg": 35.0,
        "w_uav_curv": 20.0,
        "w_obst": 120.0,
        "margin": 0.05,
        "max_grad_value": 0.03,
        "max_delta": 0.02,
        "adaptive_enable": False,
    }
    if isinstance(project_cfg, dict):
        post = project_cfg.get("post_proj_cfg", None)
        if isinstance(post, dict):
            cfg.update(post)
    # Backward-compatible alias
    if "max_grad" in cfg and "max_grad_value" not in cfg:
        cfg["max_grad_value"] = cfg["max_grad"]
    return cfg
