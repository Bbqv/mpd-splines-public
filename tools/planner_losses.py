import torch
import torch.nn.functional as F


def smooth_l1(a, b, beta=0.01):
    return F.smooth_l1_loss(a, b, beta=beta)


def huber_norm(v, beta=0.01):
    # v: [...,3] -> SmoothL1( ||v||, 0 )
    return F.smooth_l1_loss(v.norm(dim=-1), torch.zeros_like(v[..., 0]), beta=beta)


def second_diff(x):
    # x: [B,H,3] -> [B,H-2,3]
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def temporal_avg_pool(x, k=9):
    """
    Moving-average low-pass filter along time axis.
    x: [B,H,C]
    returns: [B,H,C] (same length)
    """
    if k <= 1:
        return x
    B, H, C = x.shape
    if H <= 1:
        return x

    # Keep kernel odd and valid for reflect padding.
    k = int(k)
    if k % 2 == 0:
        k += 1
    if k > H:
        k = H if (H % 2 == 1) else (H - 1)
    if k <= 1:
        return x

    pad = k // 2
    # [B,C,H]
    xt = x.transpose(1, 2)
    # reflect padding keeps edges stable
    xt = F.pad(xt, (pad, pad), mode="reflect")
    xt = F.avg_pool1d(xt, kernel_size=k, stride=1)
    return xt.transpose(1, 2).contiguous()


def piecewise_line_ref_and_dir(xyz, t_g):
    """
    xyz: [B,H,3]
    line ref uses anchors from xyz itself: start/tg/goal (hard-written already)
    return:
      x_ref: [B,H,3]
      dir_unit: [B,H,3]  unit direction of corresponding segment
    """
    B, H, _ = xyz.shape
    device = xyz.device
    start = xyz[:, 0]
    tg = xyz[:, t_g]
    goal = xyz[:, -1]

    t = torch.arange(H, device=device).float()  # [H]
    # alpha for two segments
    a1 = (t / max(t_g, 1)).clamp(0, 1)[None, :, None]
    a2 = ((t - t_g) / max((H - 1 - t_g), 1)).clamp(0, 1)[None, :, None]

    p0 = torch.where((t[None, :, None] <= t_g), start[:, None, :], tg[:, None, :])
    p1 = torch.where((t[None, :, None] <= t_g), tg[:, None, :], goal[:, None, :])
    alpha = torch.where((t[None, :, None] <= t_g), a1, a2)

    x_ref = p0 + alpha * (p1 - p0)             # [B,H,3]
    d = (p1 - p0)
    dir_unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x_ref, dir_unit


def orthogonal_error(xyz, x_ref, dir_unit):
    # e = r - (r·d)d
    r = xyz - x_ref
    proj = (r * dir_unit).sum(dim=-1, keepdim=True) * dir_unit
    return r - proj


def speed_profile_loss(xyz, down_stride=4, beta=0.01):
    """
    enforce cumulative arc-length progress ~ linear in time
    xyz: [B,H,3]
    """
    d = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)          # [B,H-1]
    s = torch.cumsum(d, dim=-1)                          # [B,H-1]
    s_end = s[:, -1].clamp_min(1e-6)                     # [B]
    s_norm = s / s_end[:, None]                          # [B,H-1]

    t = torch.arange(xyz.shape[1] - 1, device=xyz.device).float()
    t_norm = t / max((xyz.shape[1] - 2), 1)              # [H-1]

    if down_stride > 1:
        idx = torch.arange(0, xyz.shape[1] - 1, down_stride, device=xyz.device)
        s_norm = s_norm[:, idx]
        t_norm = t_norm[idx]

    return smooth_l1(s_norm, t_norm[None, :].expand_as(s_norm), beta=beta)


def planner_like_losses_xyz(
    xyz_pred, t_g,
    line_stride=8,
    speed_stride=4,
    tg_mask_K=8,
    beta=0.01,
    do_len_over=True,
    len_margin=0.10,
    lp_kernel=9,          # low-pass kernel size (larger suppresses HF zigzag)
):
    """
    xyz_pred: [B,H,3]  (x0_pred xyz)
    returns dict of robust losses

    Key change to kill zigzag:
      - low-pass xyz for line/ortho_dd/speed/len (prevents high-freq "cheating")
      - keep anchors exact after low-pass
      - jerk still computed on raw xyz to punish high-frequency motion directly
    """
    B, H, _ = xyz_pred.shape
    device = xyz_pred.device

    xyz_raw = xyz_pred

    # --- Low-pass for geometry/time losses ---
    xyz_lp = temporal_avg_pool(xyz_raw, k=lp_kernel)

    # Keep hard anchors EXACT (otherwise low-pass smears them)
    xyz_lp[:, 0, :] = xyz_raw[:, 0, :]
    xyz_lp[:, t_g, :] = xyz_raw[:, t_g, :]
    xyz_lp[:, -1, :] = xyz_raw[:, -1, :]

    # --- (1) line_lowfreq + mask tg±K (on low-pass) ---
    x_ref, dir_unit = piecewise_line_ref_and_dir(xyz_lp, t_g)
    e_ortho = orthogonal_error(xyz_lp, x_ref, dir_unit)          # [B,H,3]

    idx = torch.arange(0, H, line_stride, device=device)
    mask = (idx < (t_g - tg_mask_K)) | (idx > (t_g + tg_mask_K))
    idx = idx[mask]
    loss_line = huber_norm(e_ortho[:, idx], beta=beta)

    # --- (2) kill zigzag: second diff of orthogonal error (on low-pass) ---
    e2 = second_diff(e_ortho)  # [B,H-2,3] corresponds to t=1..H-2
    t2 = torch.arange(1, H - 1, device=device)
    mask2 = (t2 < (t_g - tg_mask_K)) | (t2 > (t_g + tg_mask_K))
    loss_ortho_dd = huber_norm(e2[:, mask2], beta=beta)

    # --- (3) speed profile loss (on low-pass) ---
    loss_speed = speed_profile_loss(xyz_lp, down_stride=speed_stride, beta=beta)

    # --- (4) jerk (robust) on RAW xyz (directly punishes high-frequency) ---
    a = second_diff(xyz_raw)          # [B,H-2,3]
    j = second_diff(a)                # [B,H-4,3]
    loss_jerk = huber_norm(j, beta=beta)

    # --- (5) length overshoot (on low-pass; avoid high-freq length cheating) ---
    loss_len = torch.zeros((), device=device)
    if do_len_over:
        d = (xyz_lp[:, 1:] - xyz_lp[:, :-1]).norm(dim=-1)
        length = d.sum(dim=-1)  # [B]
        start = xyz_lp[:, 0]
        tg = xyz_lp[:, t_g]
        goal = xyz_lp[:, -1]
        ref = (start - tg).norm(dim=-1) + (tg - goal).norm(dim=-1)
        ref = ref * (1.0 + len_margin)
        over = F.relu(length - ref)
        loss_len = smooth_l1(over, torch.zeros_like(over), beta=0.05)

    return {
        "pl_line": loss_line,
        "pl_ortho_dd": loss_ortho_dd,
        "pl_speed": loss_speed,
        "pl_jerk": loss_jerk,
        "pl_len": loss_len,
    }


def random_sphere_obstacle_loss_xyz(
    xyz_pred,
    t_g,
    n_spheres=2,
    xyz_min=(-0.40, -0.40, 0.80),
    xyz_max=(0.40, 0.40, 1.80),
    r_min=0.12,
    r_max=0.24,
    margin=0.05,
    tg_mask_k=8,
):
    """
    Random-sphere obstacle penalty on predicted xyz trajectory.
    The penalty ignores hard-constrained times (start, goal, and a tg neighborhood)
    to avoid conflicting with hard conditioning.

    Args:
      xyz_pred: [B,H,3]
      t_g: int
    Returns:
      loss_obst: scalar tensor
      aux dict with sampled sphere stats
    """
    B, H, _ = xyz_pred.shape
    device = xyz_pred.device
    dtype = xyz_pred.dtype

    n = int(max(0, n_spheres))
    if n <= 0:
        z = xyz_pred.new_tensor(0.0)
        return z, {"clr_min": z, "n_spheres": 0}

    xyz_min_t = torch.as_tensor(xyz_min, dtype=dtype, device=device).view(1, 1, 3)
    xyz_max_t = torch.as_tensor(xyz_max, dtype=dtype, device=device).view(1, 1, 3)
    centers = torch.rand((1, n, 3), device=device, dtype=dtype) * (xyz_max_t - xyz_min_t) + xyz_min_t
    radii = torch.rand((1, n), device=device, dtype=dtype) * (float(r_max) - float(r_min)) + float(r_min)

    # [B,H,n]
    diff = xyz_pred[:, :, None, :] - centers[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    clearance = dist - radii[:, None, :]

    # mask out hard-conditioning times
    tmask = torch.ones((H,), device=device, dtype=torch.bool)
    tmask[0] = False
    tmask[H - 1] = False
    k = int(max(0, tg_mask_k))
    t0 = max(0, int(t_g) - k)
    t1 = min(H - 1, int(t_g) + k)
    tmask[t0 : t1 + 1] = False

    if bool(tmask.any()):
        c_use = clearance[:, tmask, :]
    else:
        c_use = clearance

    pen = torch.clamp(float(margin) - c_use, min=0.0)
    loss = torch.mean(pen * pen)

    aux = {
        "clr_min": torch.min(c_use.detach()),
        "n_spheres": n,
    }
    return loss, aux
