import numpy as np
import torch
import torch.nn.functional as F


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
    max_tries=5000,
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

    rng = np.random.default_rng(int(seed))
    out = []
    tries = 0
    while len(out) < n and tries < int(max_tries):
        tries += 1
        r = float(rng.uniform(r_lo, r_hi))
        c = rng.uniform(xyz_min, xyz_max).astype(np.float32)

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
        if ok:
            out.append((c, r))

    if len(out) < n:
        raise RuntimeError(f"[ERR] random sphere sampling failed: sampled {len(out)}/{n} after {tries} tries")
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


@torch.enable_grad()
def obstacle_project_batch(
    traj0: torch.Tensor,
    hard_mask: torch.Tensor,
    spheres,
    n_iters: int = 30,
    lr: float = 0.01,
    w_data: float = 10.0,
    w_a: float = 50.0,
    w_j: float = 10.0,
    w_obst: float = 30.0,
    obst_margin: float = 0.10,
    max_grad_value: float = 0.1,
    max_delta: float = 0.05,
) -> torch.Tensor:
    """
    Lightweight post-projection after diffusion guidance:
      min w_data||x-x0||^2 + w_a||D2 x||^2 + w_j||D3 x||^2 + w_obst*hinge(clearance)^2
      s.t. hard anchor states stay fixed (start/goal/grasp indices).
    """
    if n_iters <= 0:
        return traj0

    x0 = traj0.detach()
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=float(lr))

    sph = [(torch.as_tensor(c, dtype=x0.dtype, device=x0.device).view(1, 1, 3), float(r)) for c, r in spheres]
    margin = float(max(0.0, obst_margin))

    for _ in range(int(n_iters)):
        opt.zero_grad(set_to_none=True)

        loss = 0.0
        if w_data > 0.0:
            loss = loss + float(w_data) * torch.mean((x - x0) ** 2)
        if w_a > 0.0 and x.shape[1] >= 3:
            loss = loss + float(w_a) * torch.mean(_second_diff(x) ** 2)
        if w_j > 0.0 and x.shape[1] >= 4:
            loss = loss + float(w_j) * torch.mean(_third_diff(x) ** 2)

        if w_obst > 0.0 and len(sph) > 0:
            xyz = x[..., :3]
            loss_obst = 0.0
            for c, r in sph:
                dist = torch.linalg.norm(xyz - c, dim=-1)
                clearance = dist - r
                pen = torch.clamp(margin - clearance, min=0.0)
                loss_obst = loss_obst + torch.mean(pen * pen)
            loss = loss + float(w_obst) * loss_obst

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
    w_a = float(proj_cfg.get("w_a", 50.0))
    w_j = float(proj_cfg.get("w_j", 10.0))
    w_obst = float(proj_cfg.get("w_obst", 30.0))
    margin = float(proj_cfg.get("margin", 0.10))
    max_grad = float(proj_cfg.get("max_grad_value", 0.1))
    max_delta = float(proj_cfg.get("max_delta", 0.05))

    out = obstacle_project_batch(
        traj0=traj,
        hard_mask=hard_mask,
        spheres=spheres,
        n_iters=n_iters,
        lr=lr,
        w_data=w_data,
        w_a=w_a,
        w_j=w_j,
        w_obst=w_obst,
        obst_margin=margin,
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
        fac_it = it_mult ** float(rr + 1)
        fac_w = w_mult ** float(rr + 1)
        fac_dx = dx_mult ** float(rr + 1)
        fac_g = g_mult ** float(rr + 1)

        out_bad = obstacle_project_batch(
            traj0=out[bad],
            hard_mask=hard_mask[bad],
            spheres=spheres,
            n_iters=int(max(1, round(n_iters * fac_it))),
            lr=lr,
            w_data=w_data,
            w_a=w_a,
            w_j=w_j,
            w_obst=w_obst * fac_w,
            obst_margin=margin,
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

    def __call__(self, x, context_d=None, guide_progress=None, **kwargs):
        grad = torch.zeros_like(x)
        xyz = x[..., :3]
        grad_xyz = grad[..., :3]
        if guide_progress is None:
            alpha_scale = 1.0
        else:
            gp = float(np.clip(guide_progress, 0.0, 1.0))
            alpha_scale = self.alpha_min_scale + (1.0 - self.alpha_min_scale) * (gp**self.alpha_ramp_power)
        alpha_eff = self.alpha * alpha_scale

        for c, r in self.spheres:
            diff = xyz - c.view(1, 1, 3)
            dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(self.eps)
            direction = diff / dist
            clearance = dist - float(r)
            pen = torch.clamp(self.margin - clearance, min=0.0)
            near_w = torch.clamp(pen / (self.margin + self.eps), min=0.0, max=1.0)
            push_mag = alpha_eff / torch.clamp(clearance, min=self.eps)
            if self.max_push > 0.0:
                push_mag = torch.clamp(push_mag, max=self.max_push)
            grad_xyz += -(near_w * push_mag * direction)

        if self.time_smooth_k >= 3 and (self.time_smooth_k % 2 == 1):
            k = self.time_smooth_k
            pad = k // 2
            g = grad_xyz.transpose(1, 2)
            g = F.pad(g, (pad, pad), mode="replicate")
            kernel = torch.ones((3, 1, k), device=g.device, dtype=g.dtype) / float(k)
            g = F.conv1d(g, kernel, groups=3)
            grad_xyz.copy_(g.transpose(1, 2))

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
        device=device,
    )

    out = {
        "method": "ddim",
        "ddim_sampling_timesteps": int(guide_cfg.get("ddim_steps", 100)),
        "t_start_guide": int(guide_cfg.get("t_start_guide", 40)),
        "guide": guide_fn,
        "guide_lr": float(guide_cfg.get("guide_lr", 0.08)),
        "n_guide_steps": int(guide_cfg.get("n_guide_steps", 1)),
        "scale_grad_by_one_minus_alpha": bool(guide_cfg.get("scale_grad_by_one_minus_alpha", False)),
        "clip_grad": bool(guide_cfg.get("clip_grad", True)),
        "clip_grad_rule": str(guide_cfg.get("clip_grad_rule", "value")),
        "max_grad_value": float(guide_cfg.get("max_grad_value", 1.0)),
        "max_perturb_x": float(guide_cfg.get("max_perturb_x", 0.1)),
        "compute_costs_with_xrecon": bool(guide_cfg.get("use_xrecon", False)),
    }
    return out


def make_proj_only_adaptive_preset(spheres):
    """
    Validated preset from current experiments:
      - guide disabled
      - post projection + adaptive rounds enabled
    """
    return {
        "enable": True,
        "spheres": list(spheres),
        "guide": {
            "alpha": 0.0,
            "margin": 0.10,
            "max_push": 0.0,
            "ddim_steps": 100,
            "t_start_guide": 0,
            "guide_lr": 0.08,
            "n_guide_steps": 1,
            "time_smooth_k": 5,
            "alpha_ramp_power": 2.0,
            "alpha_min_scale": 0.0,
            "scale_grad_by_one_minus_alpha": False,
            "clip_grad": True,
            "clip_grad_rule": "value",
            "max_grad_value": 1.0,
            "max_perturb_x": 0.1,
            "use_xrecon": False,
        },
        "project": {
            "enable": True,
            "iters": 80,
            "lr": 0.005,
            "w_data": 40.0,
            "w_a": 30.0,
            "w_j": 8.0,
            "w_obst": 80.0,
            "margin": 0.05,
            "max_grad_value": 0.05,
            "max_delta": 0.03,
            "adaptive_enable": True,
            "adaptive_rounds": 2,
            "adaptive_clearance": 0.0,
            "adaptive_iters_mult": 2.0,
            "adaptive_w_obst_mult": 2.0,
            "adaptive_max_delta_mult": 2.0,
            "adaptive_max_grad_mult": 1.5,
        },
    }
