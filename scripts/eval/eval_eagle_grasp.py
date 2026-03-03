import os
import argparse
import numpy as np
import math

import torch
import matplotlib.pyplot as plt
import pinocchio as pin

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
import torch.nn as nn

# ===================== cond_encoder adapters =====================

class CondEncoderInputAdapter(nn.Module):
    """
    Wrap an existing cond_encoder (usually nn.Sequential) and force input feature dim to expected_dim.
    """
    def __init__(self, inner: nn.Module, expected_dim: int = 160):
        super().__init__()
        self.inner = inner
        self.expected_dim = expected_dim
        self._printed = False

    def forward(self, c):
        if torch.is_tensor(c) and c.dim() == 2:
            B, K = c.shape
            if K != self.expected_dim:
                if not self._printed:
                    print(f"[COND_ADAPT] cond_encoder input dim {K} -> {self.expected_dim}")
                    self._printed = True
                if K > self.expected_dim:
                    c = c[:, : self.expected_dim].contiguous()
                else:
                    pad = torch.zeros(B, self.expected_dim - K, device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=1).contiguous()
        return self.inner(c)


def patch_all_cond_encoders(denoiser: nn.Module, expected_dim: int = 160) -> int:
    """
    Find all submodules that have attribute 'cond_encoder' and wrap it.
    Return number of patched modules.
    """
    cnt = 0
    for m in denoiser.modules():
        if hasattr(m, "cond_encoder"):
            ce = getattr(m, "cond_encoder")
            if isinstance(ce, nn.Module):
                setattr(m, "cond_encoder", CondEncoderInputAdapter(ce, expected_dim=expected_dim))
                cnt += 1
    return cnt


# ===================== wrappers / pickle placeholders =====================

class ContextSqueezeWrapper(nn.Module):
    """
    Make context embedding compatible with denoiser cond_encoder:
      - ensure context is 2D (B,K)
      - if context is 3D (B,L,K): mean-pool over L -> (B,K)
      - then pad/trim K to expected_dim (default 160, inferred if possible)
    """
    def __init__(self, inner: nn.Module, expected_dim: int = None):
        super().__init__()
        self.inner = inner
        self.expected_dim = expected_dim  # if None, infer on first forward
        self._printed = False

    def _infer_expected_dim(self):
        for name in ["cond_encoder", "context_encoder", "cond_mlp", "c_mlp"]:
            if hasattr(self.inner, name):
                mod = getattr(self.inner, name)
                for m in mod.modules():
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)
        return None

    def forward(self, x, t, context):
        if torch.is_tensor(context):
            if context.dim() == 3:
                context = context.mean(dim=1)
            elif context.dim() > 3:
                context = context.reshape(context.shape[0], -1)

            if self.expected_dim is None:
                self.expected_dim = self._infer_expected_dim()
                if not self._printed:
                    print("[WRAP] inferred expected context dim =", self.expected_dim)
                    self._printed = True

            if self.expected_dim is not None and context.dim() == 2:
                K = int(context.shape[1])
                if K > self.expected_dim:
                    context = context[:, : self.expected_dim].contiguous()
                elif K < self.expected_dim:
                    pad = torch.zeros(context.shape[0], self.expected_dim - K,
                                      device=context.device, dtype=context.dtype)
                    context = torch.cat([context, pad], dim=1).contiguous()

        return self.inner(x, t, context)


class LinearNoneHook:
    """Placeholder to satisfy torch.load pickle references from training-time code."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


# ===================== FK / pinocchio =====================

URDF_PATH = "/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf"
EE_FRAME = "gripper_link"

_model_fk = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
_data_fk = _model_fk.createData()
_fid = _model_fk.getFrameId(EE_FRAME)
assert _model_fk.nq == 9, f"Expected nq=9 but got {_model_fk.nq}"
print("[PIN] nq=", _model_fk.nq, " nv=", _model_fk.nv)


def _quat_norm_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def yaw_to_quat_xyzw(yaw: float):
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def traj5_to_traj9(traj5: np.ndarray, q_start9: np.ndarray, q_goal9: np.ndarray):
    traj5 = np.asarray(traj5, dtype=np.float64)
    H = traj5.shape[0]
    out = np.zeros((H, 9), dtype=np.float64)

    z0 = float(q_start9[2])
    z1 = float(q_goal9[2])
    zs = np.linspace(z0, z1, H, dtype=np.float64)

    out[:, 0] = traj5[:, 0]
    out[:, 1] = traj5[:, 1]
    out[:, 2] = zs
    for t in range(H):
        out[t, 3:7] = yaw_to_quat_xyzw(traj5[t, 2])
    out[:, 7] = traj5[:, 3]
    out[:, 8] = traj5[:, 4]
    return out


def fk_min_grasp_err_from_traj9(traj9, q_grasp_xyz):
    traj9 = np.asarray(traj9, dtype=np.float64)
    q_grasp_xyz = np.asarray(q_grasp_xyz, dtype=np.float64)

    H = traj9.shape[0]
    ee = np.zeros((H, 3), dtype=np.float64)

    for t in range(H):
        q = traj9[t].copy()
        q[3:7] = _quat_norm_xyzw(q[3:7])
        pin.forwardKinematics(_model_fk, _data_fk, q)
        pin.updateFramePlacements(_model_fk, _data_fk)
        ee[t] = _data_fk.oMf[_fid].translation

    d = np.linalg.norm(ee - q_grasp_xyz[None, :], axis=1)
    return float(d.min())


def fk_xyz_and_jac_pos(q9):
    q9 = np.asarray(q9, dtype=np.float64).copy()
    q9[3:7] = _quat_norm_xyzw(q9[3:7])
    pin.forwardKinematics(_model_fk, _data_fk, q9)
    pin.updateFramePlacements(_model_fk, _data_fk)
    ee = _data_fk.oMf[_fid].translation.copy()
    J6 = pin.computeFrameJacobian(_model_fk, _data_fk, q9, _fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Jpos = J6[:3, :].copy()
    return ee, Jpos


def ik_solve_pos_dls(
    model,
    data,
    frame_id,
    q_init,
    p_target,
    iters=200,
    tol=2e-2,
    step=0.5,
    lam=1e-2,
    q_min=None,
    q_max=None,
    verbose=False,
):
    q = np.asarray(q_init, dtype=np.float64).copy()
    arm_idx0 = model.nq - 2
    arm_idx1 = model.nq - 1
    I3 = np.eye(3)

    for it in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        p_now = data.oMf[frame_id].translation
        err = np.asarray(p_target, dtype=np.float64) - p_now
        err_norm = float(np.linalg.norm(err))

        if verbose and (it % 20 == 0 or err_norm < tol):
            print(f"[IK] it={it:03d} |err|={err_norm:.6f}")

        if err_norm < tol:
            break

        J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jpos = J6[:3, :]

        A = Jpos @ Jpos.T + (lam * lam) * I3
        dq = Jpos.T @ np.linalg.solve(A, err)

        q = pin.integrate(model, q, step * dq)

        if (q_min is not None) and (q_max is not None):
            q[arm_idx0] = float(np.clip(q[arm_idx0], q_min[0], q_max[0]))
            q[arm_idx1] = float(np.clip(q[arm_idx1], q_min[1], q_max[1]))

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    final_err = float(np.linalg.norm(np.asarray(p_target) - data.oMf[frame_id].translation))
    return q, final_err


# ===================== helpers =====================

def grasp_index_from_times(H: int, to_grasp: float, to_target: float) -> int:
    frac = float(to_grasp) / float(to_grasp + to_target + 1e-9)
    idx = int(round(frac * (H - 1)))
    return max(0, min(H - 1, idx))


def l2(x: torch.Tensor, y: torch.Tensor, dim=-1) -> torch.Tensor:
    return torch.linalg.norm(x - y, dim=dim)


def _context_to_batch(x: torch.Tensor, B: int) -> torch.Tensor:
    if x is None:
        return None

    if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    if x.dim() == 0:
        x = x.view(1)

    if x.dim() == 1:
        if x.numel() == 1:
            return x.view(1).repeat(B).contiguous()
        x = x.unsqueeze(0)

    if x.dim() == 2 and x.shape[1] == 1:
        x1 = x.squeeze(1)
        N = x1.shape[0]
        if N == B:
            return x1.contiguous()
        if N > B:
            return x1[:B].contiguous()
        reps = (B + N - 1) // N
        return x1.repeat(reps)[:B].contiguous()

    assert x.dim() == 2, f"context must be 1D or 2D after fix, got {tuple(x.shape)}"
    N = x.shape[0]
    if N == B:
        return x.contiguous()
    if N > B:
        return x[:B].contiguous()
    reps = (B + N - 1) // N
    return x.repeat(reps, 1)[:B].contiguous()


def quat_xyzw_to_yaw(q_xyzw):
    x, y, z, w = [float(v) for v in q_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def q9_to_q5(q9):
    q9 = np.asarray(q9, dtype=np.float64)
    x, y = float(q9[0]), float(q9[1])
    yaw = quat_xyzw_to_yaw(q9[3:7])
    j1, j2 = float(q9[7]), float(q9[8])
    return np.array([x, y, yaw, j1, j2], dtype=np.float64)


def try_normalize_cp(dataset, cp_unnorm_tensor):
    if hasattr(dataset, "normalize_control_points"):
        return dataset.normalize_control_points(cp_unnorm_tensor)
    for attr in ["control_point_normalizer", "cp_normalizer", "normalizer"]:
        if hasattr(dataset, attr):
            norm = getattr(dataset, attr)
            if hasattr(norm, "normalize_control_points"):
                return norm.normalize_control_points(cp_unnorm_tensor)
            if hasattr(norm, "normalize"):
                return norm.normalize(cp_unnorm_tensor)
    return cp_unnorm_tensor


def normalize_state9_to_1d(dataset, q_1x9: torch.Tensor, name="q") -> torch.Tensor:
    """
    Normalize a single state (1,9) and return 1D (9,) tensor.
    IMPORTANT: hard_conds values will be 1D to avoid internal (B,B,9) expansion bugs.
    """
    if q_1x9.dim() == 1:
        q_1x9 = q_1x9.unsqueeze(0)
    if tuple(q_1x9.shape) != (1, 9):
        raise RuntimeError(f"{name} expected (1,9), got {tuple(q_1x9.shape)}")

    tmp = q_1x9.unsqueeze(1)          # (1,1,9)
    tmp = try_normalize_cp(dataset, tmp)

    if tmp.dim() == 3:
        tmp = tmp[:, 0, :]            # (1,9)
    elif tmp.dim() == 2:
        pass
    else:
        raise RuntimeError(f"{name} normalize returned weird shape {tuple(tmp.shape)}")

    if tuple(tmp.shape) != (1, 9):
        raise RuntimeError(f"{name} normalize expected (1,9), got {tuple(tmp.shape)}")

    return tmp.squeeze(0).contiguous()  # (9,)


def get_expected_qs_dim(model) -> int:
    try:
        cm = model.context_model
        cm_qs = cm.context_model_qs
        for m in cm_qs.modules():
            if isinstance(m, torch.nn.Linear):
                return int(m.in_features)
    except Exception:
        pass
    return None


def adapt_qs_normalized(context_d: dict, expected_dim: int, idx: int):
    if expected_dim is None:
        return
    if "qs_normalized" not in context_d:
        return
    v = context_d["qs_normalized"]
    if not torch.is_tensor(v):
        return
    if v.dim() == 0:
        v = v.view(1)
    cur = int(v.shape[-1])
    if cur > expected_dim:
        v = v[..., :expected_dim].contiguous()
    elif cur < expected_dim:
        pad = torch.zeros(*v.shape[:-1], expected_dim - cur, device=v.device, dtype=v.dtype)
        v = torch.cat([v, pad], dim=-1).contiguous()
    context_d["qs_normalized"] = v
    if idx == 0:
        print(f"[CTX_FIX] qs_normalized: {cur} -> {expected_dim}")


class ContextModelPadTrimWrapper(nn.Module):
    def __init__(self, inner: nn.Module, expected_dim: int = 160):
        super().__init__()
        self.inner = inner
        self.expected_dim = expected_dim
        self._printed = False

    def forward(self, **kwargs):
        emb = self.inner(**kwargs)

        if torch.is_tensor(emb):
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            elif emb.dim() > 3:
                emb = emb.reshape(emb.shape[0], -1)

            if emb.dim() == 2:
                K = int(emb.shape[1])
                if not self._printed:
                    print("[CTX_MODEL_WRAP] context_emb shape before fix:", (int(emb.shape[0]), K),
                          "-> expected", self.expected_dim)
                    self._printed = True

                if K > self.expected_dim:
                    emb = emb[:, : self.expected_dim].contiguous()
                elif K < self.expected_dim:
                    pad = torch.zeros(emb.shape[0], self.expected_dim - K,
                                      device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=1).contiguous()

        return emb


# ===================== main =====================

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_file_merged", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--save_dir", type=str, default="eval_out")
    ap.add_argument("--H", type=int, default=144)
    ap.add_argument("--to_grasp_default", type=float, default=3.0)
    ap.add_argument("--to_target_default", type=float, default=4.0)
    ap.add_argument("--plot_max_trajs", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = EagleGraspNPZDataset(
        root_dir=args.dataset_file_merged,
        H=args.H,
        only_accepted=True,
    )
    H = args.H
    print("[INFO] dataset len =", len(dataset))
    print("[INFO] H =", H)

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")

    if hasattr(ckpt_obj, "run_inference"):
        model = ckpt_obj
    elif isinstance(ckpt_obj, dict) and "model" in ckpt_obj and hasattr(ckpt_obj["model"], "run_inference"):
        model = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict) and "ema_model" in ckpt_obj and hasattr(ckpt_obj["ema_model"], "run_inference"):
        model = ckpt_obj["ema_model"]
    else:
        keys = list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else None
        raise RuntimeError(f"[ERR] Unknown ckpt format. type={type(ckpt_obj)}, keys={keys}")

    model = model.to(device).eval()

    QS_DIM_EXPECTED = get_expected_qs_dim(model)
    print("[INFO] QS_DIM_EXPECTED =", QS_DIM_EXPECTED)

    if hasattr(model, "context_model"):
        model.context_model = ContextModelPadTrimWrapper(model.context_model, expected_dim=160).to(device)
        print("[INFO] Wrapped context_model to output (B,160)")

    print("[INFO] Loaded model type:", type(model))

    # squeeze internal context if needed
    if hasattr(model, "model"):
        denoiser = model.model
        if isinstance(denoiser, torch.nn.DataParallel):
            inner = denoiser.module
            wrapped = ContextSqueezeWrapper(inner, expected_dim=160).to(device)
            model.model = torch.nn.DataParallel(wrapped)
        else:
            model.model = ContextSqueezeWrapper(denoiser, expected_dim=160).to(device)
    print("[INFO] Wrapped denoiser with ContextSqueezeWrapper (squeeze Bx1xK -> BxK)")

    if hasattr(model, "model"):
        den = model.model
        if isinstance(den, torch.nn.DataParallel):
            patched = patch_all_cond_encoders(den.module, expected_dim=160)
        else:
            patched = patch_all_cond_encoders(den, expected_dim=160)
        print(f"[INFO] patched cond_encoder count = {patched}")

    n_cases = min(args.n_cases, len(dataset))
    for idx in range(n_cases):
        data_sample = dataset[idx]

        data_cpu = {}
        for k, v in data_sample.items():
            data_cpu[k] = v.detach().cpu() if torch.is_tensor(v) else v

        for k, v in list(data_sample.items()):
            if torch.is_tensor(v):
                data_sample[k] = v.to(device)

        q_start = data_sample["q_start"]     # (9,)
        q_goal = data_sample["q_goal"]       # (9,)
        q_grasp_xyz = data_sample["q_grasp"] # (3,)
        traj_gt = data_sample["traj"]        # (H,9)

        if idx == 0:
            print("[DEBUG] traj shape:", tuple(traj_gt.shape))
            print("[DEBUG] traj[0]  :", traj_gt[0].detach().cpu().numpy())
            print("[DEBUG] traj[-1] :", traj_gt[-1].detach().cpu().numpy())
            print("[DEBUG] q_start :", q_start.detach().cpu().numpy())
            print("[DEBUG] q_grasp :", q_grasp_xyz.detach().cpu().numpy())
            print("[DEBUG] q_goal  :", q_goal.detach().cpu().numpy())

        ee_pos_gt = data_sample.get("ee_positions", None)
        if ee_pos_gt is None:
            raise KeyError("Missing 'ee_positions' in data_sample. Dataset must load it from npz.")

        ee_pos_gt_np = ee_pos_gt.detach().cpu().numpy() if torch.is_tensor(ee_pos_gt) else np.asarray(ee_pos_gt)
        q_grasp_np = q_grasp_xyz.detach().cpu().numpy() if torch.is_tensor(q_grasp_xyz) else np.asarray(q_grasp_xyz)

        dists = np.linalg.norm(ee_pos_gt_np - q_grasp_np[None, :], axis=1)
        gt_ee_min_err = float(dists.min())
        gt_best_t = int(dists.argmin())
        print(f"[GT FK] ee_min_err={gt_ee_min_err:.4f} at t*=58 / T={len(dists)}")

        # =================== context ===================
        context_d = dataset.build_context(data_sample=data_sample)
        adapt_qs_normalized(context_d, QS_DIM_EXPECTED, idx)

        if idx == 0:
            print("[CTX_KEYS]", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v)) for k, v in context_d.items()})
            print("[DEBUG context shapes BEFORE]", {k: tuple(v.shape) for k, v in context_d.items() if torch.is_tensor(v)})

        for k, v in list(context_d.items()):
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] == 1:
                context_d[k] = v.squeeze(1)

        if idx == 0:
            print("[DEBUG context shapes AFTER ]", {k: tuple(v.shape) for k, v in context_d.items() if torch.is_tensor(v)})

        # =================== tg-scan + mid hard_conds ===================
        t_g_list = [24, 40, 56, 72, 88, 104, 120]
        t_g_list = [t for t in t_g_list if 1 <= t <= H - 2]
        per_tg = max(5, int(np.ceil(args.n_samples / max(1, len(t_g_list)))))

        # =================== hard_conds: PASS 1D (9,) ONLY ===================
        q_start_hc_1d = normalize_state9_to_1d(
            dataset,
            torch.as_tensor(q_start.detach().cpu().numpy(), device=device, dtype=torch.float32).unsqueeze(0),
            name="q_start_hc"
        )
        q_goal_hc_1d = normalize_state9_to_1d(
            dataset,
            torch.as_tensor(q_goal.detach().cpu().numpy(), device=device, dtype=torch.float32).unsqueeze(0),
            name="q_goal_hc"
        )

        # IK solve q_g in 9D (pinocchio)
        q_init_np = q_start.detach().cpu().numpy()
        q_g_np, _ = ik_solve_pos_dls(
            _model_fk, _data_fk, _fid,
            q_init=q_init_np,
            p_target=q_grasp_np,
            iters=400,
            tol=2e-2,
            step=0.5,
            lam=1e-2,
            verbose=False,
        )
        ee_qg, _ = fk_xyz_and_jac_pos(q_g_np)
        print(f"[IK_CHECK] ||FK(q_g)-grasp|| = {float(np.linalg.norm(ee_qg - q_grasp_np)):.6f} m | nq={_model_fk.nq} nv={_model_fk.nv}")

        q_g_hc_1d = normalize_state9_to_1d(
            dataset,
            torch.as_tensor(q_g_np, device=device, dtype=torch.float32).unsqueeze(0),
            name="q_g_hc"
        )

        all_cp_norm = []
        all_tag_tg = []

        for t_g in t_g_list:
            # IMPORTANT: hard_conds values are 1D (9,), NOT (B,9)
            hard_conds = {0: q_start_hc_1d, H - 1: q_goal_hc_1d, t_g: q_g_hc_1d}

            # context is batched to per_tg, and n_samples=per_tg
            context_d_tg = {}
            for k, v in context_d.items():
                if torch.is_tensor(v):
                    context_d_tg[k] = _context_to_batch(v, per_tg)
                else:
                    context_d_tg[k] = v

            if idx == 0 and t_g == t_g_list[0]:
                print("[TG_DEBUG_FINAL] shapes:",
                      {k: tuple(v.shape) for k, v in context_d_tg.items() if torch.is_tensor(v)})
                print("[HC_DEBUG] hard_conds shapes:",
                      {kk: tuple(vv.shape) for kk, vv in hard_conds.items()})

            cp_norm_tg = model.run_inference(
                context_d=context_d_tg,
                hard_conds=hard_conds,
                n_samples=per_tg,
                horizon=H,
            )

            if idx == 0 and t_g == t_g_list[0]:
                print("[DEBUG] cp_norm_tg shape =", tuple(cp_norm_tg.shape))

            all_cp_norm.append(cp_norm_tg)
            all_tag_tg.append(torch.full((cp_norm_tg.shape[0],), t_g, device="cpu", dtype=torch.int64))

        cp_norm = torch.cat(all_cp_norm, dim=0)
        tag_tg = torch.cat(all_tag_tg, dim=0)

        cp = dataset.unnormalize_control_points(cp_norm)
        print("[DEBUG] cp shape after unnormalize =", tuple(cp.shape))

        # 5D -> 9D conversion for FK eval if needed
        if cp.shape[-1] == 5:
            cp5_np = cp.detach().cpu().numpy()
            q_start9_np = q_start.detach().cpu().numpy()
            q_goal9_np  = q_goal.detach().cpu().numpy()
            cp9_np = np.stack(
                [traj5_to_traj9(cp5_np[i], q_start9_np, q_goal9_np) for i in range(cp5_np.shape[0])],
                axis=0
            )
            cp = torch.as_tensor(cp9_np, device=cp.device, dtype=torch.float32)
            print("[DEBUG] cp converted 5D->9D:", tuple(cp.shape))
        elif cp.shape[-1] != 9:
            raise ValueError(f"[ERR] Unexpected cp last-dim={cp.shape[-1]} with shape={tuple(cp.shape)}")

        # clamp endpoints
        cp[:, 0, :] = q_start[None, :]
        cp[:, -1, :] = q_goal[None, :]

        cp_np = cp.detach().cpu().numpy()
        pred_fk_errs = np.asarray([fk_min_grasp_err_from_traj9(cp_np[s], q_grasp_np) for s in range(cp_np.shape[0])],
                                  dtype=np.float64)

        pred_best = float(pred_fk_errs.min())
        pred_mean = float(pred_fk_errs.mean())
        pred_succ2 = float((pred_fk_errs < 0.02).mean())

        best_idx = int(pred_fk_errs.argmin())
        print(f"[TG_SCAN] best from t_g={int(tag_tg[best_idx])} | best_err={pred_best:.4f} m")
        print(f"[PRED FK] best_ee_min_err={pred_best:.4f} m | mean={pred_mean:.4f} m | succ@2cm={pred_succ2:.3f}")

        # save one case and exit (since n_cases=1 in your cmd)
        out_npz = os.path.join(args.save_dir, f"case_{idx:04d}_samples.npz")
        np.savez(
            out_npz,
            cp=cp_np,
            traj_gt=data_cpu["traj"].numpy() if torch.is_tensor(data_cpu["traj"]) else np.array(data_cpu["traj"]),
            q_start=data_cpu["q_start"].numpy(),
            q_goal=data_cpu["q_goal"].numpy(),
            q_grasp=data_cpu["q_grasp"].numpy(),
            pred_fk_best=np.array([pred_best], dtype=np.float32),
            pred_fk_mean=np.array([pred_mean], dtype=np.float32),
            pred_fk_succ2cm=np.array([pred_succ2], dtype=np.float32),
        )

        print(f"[SAVED] {out_npz}")


if __name__ == "__main__":
    main()