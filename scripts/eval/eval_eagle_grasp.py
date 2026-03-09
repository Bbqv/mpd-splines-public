import os
import argparse
import numpy as np
import math

import torch
import pinocchio as pin
import torch.nn as nn

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset

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
    Wrap denoiser forward(x,t,context):
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


def fk_ee_xyz_from_q9(q9):
    q9 = np.asarray(q9, dtype=np.float64).copy()
    q9[3:7] = _quat_norm_xyzw(q9[3:7])
    pin.forwardKinematics(_model_fk, _data_fk, q9)
    pin.updateFramePlacements(_model_fk, _data_fk)
    return _data_fk.oMf[_fid].translation.copy()


def fk_min_grasp_err_from_traj9(traj9, q_grasp_xyz):
    traj9 = np.asarray(traj9, dtype=np.float64)
    q_grasp_xyz = np.asarray(q_grasp_xyz, dtype=np.float64)
    H = traj9.shape[0]
    ee = np.zeros((H, 3), dtype=np.float64)

    for t in range(H):
        ee[t] = fk_ee_xyz_from_q9(traj9[t])

    d = np.linalg.norm(ee - q_grasp_xyz[None, :], axis=1)
    return float(d.min())


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


def apply_ctx_mode_to_context(context_d: dict, ctx_mode: str, idx: int = 0):
    """
    Apply sensitivity perturbation to context_d["qs_normalized"].
    - orig: no change
    - zero_ctx: set qs_normalized to zero
    - swap_start_goal: swap first 9 dims and next 9 dims (assume start(9)+goal(9)+extra = 22)
    """
    if ctx_mode == "orig":
        return
    if "qs_normalized" not in context_d:
        if idx == 0:
            print("[SENS] qs_normalized not found in context_d; ctx_mode skipped")
        return
    qs = context_d["qs_normalized"]
    if not torch.is_tensor(qs):
        if idx == 0:
            print("[SENS] qs_normalized is not tensor; ctx_mode skipped")
        return

    # ensure 2D (B,K)
    if qs.dim() == 1:
        qs = qs.unsqueeze(0)
    if qs.dim() == 3 and qs.shape[1] == 1:
        qs = qs.squeeze(1)
    if qs.dim() != 2:
        qs = qs.reshape(qs.shape[0], -1)

    if ctx_mode == "zero_ctx":
        qs = torch.zeros_like(qs)
    elif ctx_mode == "swap_start_goal":
        if qs.shape[1] >= 18:
            tmp = qs.clone()
            tmp[:, 0:9] = qs[:, 9:18]
            tmp[:, 9:18] = qs[:, 0:9]
            qs = tmp
        else:
            if idx == 0:
                print("[SENS] swap_start_goal skipped: qs dim < 18")

    context_d["qs_normalized"] = qs
    if idx == 0:
        with torch.no_grad():
            m = float(qs.mean().item())
            s = float(qs.std().item())
            print(f"[SENS] ctx_mode={ctx_mode}, qs stats: mean={m:.4f}, std={s:.4f}, shape={tuple(qs.shape)}")


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


class ContextModelSqueezeOnly(nn.Module):
    """
    Squeeze-only wrapper for context_model:
      - if output is (B,1,K) or (B,L,K), mean-pool over dim=1 -> (B,K)
      - no pad/trim
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._printed = False

    def forward(self, **kwargs):
        emb = self.inner(**kwargs)
        if torch.is_tensor(emb):
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            elif emb.dim() > 3:
                emb = emb.reshape(emb.shape[0], -1)
            if (not self._printed) and emb.dim() == 2:
                print("[CTX_SQUEEZE_ONLY] context_emb shape =", (int(emb.shape[0]), int(emb.shape[1])))
                self._printed = True
        return emb


def _stats_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


# ===================== SMOOTHNESS METRICS =====================

def _wrap_to_pi(a):
    # a: np.ndarray
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_from_xyzw(q_xyzw):
    # q = [x,y,z,w]
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    # yaw from quaternion (assuming z-rotation dominant / free-flyer yaw)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_smooth_metrics_traj9(traj9_np):
    """
    traj9_np: (H,9) in unnormalized metric space.
    Returns dict of smoothness metrics (mean/max of |v|,|a|,|j| for xyz; plus yaw & arm joints).
    """
    traj = np.asarray(traj9_np, dtype=np.float64)
    H = traj.shape[0]

    out = {}

    # ---- UAV xyz ----
    xyz = traj[:, 0:3]  # (H,3)
    if H >= 2:
        v = xyz[1:] - xyz[:-1]            # (H-1,3)
        v_norm = np.linalg.norm(v, axis=1)
        out["uav_v_mean"] = float(v_norm.mean())
        out["uav_v_max"] = float(v_norm.max())
    else:
        out["uav_v_mean"] = 0.0
        out["uav_v_max"] = 0.0

    if H >= 3:
        a = (xyz[2:] - 2.0 * xyz[1:-1] + xyz[:-2])   # (H-2,3)
        a_norm = np.linalg.norm(a, axis=1)
        out["uav_a_mean"] = float(a_norm.mean())
        out["uav_a_max"] = float(a_norm.max())
    else:
        out["uav_a_mean"] = 0.0
        out["uav_a_max"] = 0.0

    if H >= 4:
        # jerk via third difference
        j = (xyz[3:] - 3.0 * xyz[2:-1] + 3.0 * xyz[1:-2] - xyz[:-3])  # (H-3,3)
        j_norm = np.linalg.norm(j, axis=1)
        out["uav_j_mean"] = float(j_norm.mean())
        out["uav_j_max"] = float(j_norm.max())
    else:
        out["uav_j_mean"] = 0.0
        out["uav_j_max"] = 0.0

    # ---- yaw smoothness from quaternion ----
    yaws = np.array([_yaw_from_xyzw(traj[t, 3:7]) for t in range(H)], dtype=np.float64)
    if H >= 2:
        dy = _wrap_to_pi(yaws[1:] - yaws[:-1])
        out["yaw_v_mean"] = float(np.mean(np.abs(dy)))
        out["yaw_v_max"] = float(np.max(np.abs(dy)))
    else:
        out["yaw_v_mean"] = 0.0
        out["yaw_v_max"] = 0.0

    if H >= 3:
        ddy = _wrap_to_pi(yaws[2:] - 2.0 * yaws[1:-1] + yaws[:-2])
        out["yaw_a_mean"] = float(np.mean(np.abs(ddy)))
        out["yaw_a_max"] = float(np.max(np.abs(ddy)))
    else:
        out["yaw_a_mean"] = 0.0
        out["yaw_a_max"] = 0.0

    # ---- arm joints q7,q8 (assume last 2 dims are joints) ----
    arm = traj[:, 7:9]  # (H,2)
    if H >= 2:
        dq = arm[1:] - arm[:-1]
        dq_norm = np.linalg.norm(dq, axis=1)
        out["arm_v_mean"] = float(dq_norm.mean())
        out["arm_v_max"] = float(dq_norm.max())
    else:
        out["arm_v_mean"] = 0.0
        out["arm_v_max"] = 0.0

    if H >= 3:
        ddq = arm[2:] - 2.0 * arm[1:-1] + arm[:-2]
        ddq_norm = np.linalg.norm(ddq, axis=1)
        out["arm_a_mean"] = float(ddq_norm.mean())
        out["arm_a_max"] = float(ddq_norm.max())
    else:
        out["arm_a_mean"] = 0.0
        out["arm_a_max"] = 0.0

    return out


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

    # evaluation mode
    ap.add_argument(
        "--mode",
        type=str,
        default="free",
        choices=["free", "endpoints_hard", "endpoints_and_mid_hard"],
        help="free: no hard cond. endpoints_hard: hard endpoints only. endpoints_and_mid_hard: hard endpoints+mid grasp (legacy)."
    )
    ap.add_argument("--tg_scan", action="store_true", help="If set, scan t_g_list in hard-mid mode.")
    ap.add_argument("--t_g", type=int, default=24, help="Mid-grasp timestep for endpoints_and_mid_hard when tg_scan is false.")
    ap.add_argument("--succ_thresh_m", type=float, default=0.02)

    # sensitivity
    ap.add_argument("--seed", type=int, default=0, help="fixed random seed for sensitivity test")
    ap.add_argument("--ctx_mode", type=str, default="orig", choices=["orig", "swap_start_goal", "zero_ctx"],
                    help="context perturbation mode (sensitivity test)")
    ap.add_argument("--no_wrap_patch", action="store_true",
                    help="Disable pad/trim + denoiser squeeze + cond_encoder adapters; keep context_model squeeze-only to avoid (B,1,K).")

    # endpoint debug print
    ap.add_argument("--dbg_eval", action="store_true",
                    help="Print start/goal and predicted endpoints for case 0 (for coordinate sanity check).")

    # NEW: smooth debug
    ap.add_argument("--dbg_smooth", action="store_true",
                    help="Print smoothness metrics for a couple of samples in case 0.")

    args = ap.parse_args()

    # --- fixed seed for sensitivity ---
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[SENS] seed={args.seed}, ctx_mode={args.ctx_mode}, no_wrap_patch={args.no_wrap_patch}")
    # --- end fixed seed ---

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
    print("[INFO] mode =", args.mode)
    print("[INFO] QS_DIM_EXPECTED =", "TBD (after ckpt load)")

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

    # -------- wrap/patch (optional) --------
    if hasattr(model, "context_model"):
        if args.no_wrap_patch:
            model.context_model = ContextModelSqueezeOnly(model.context_model).to(device)
            print("[INFO] no_wrap_patch=True: use ContextModelSqueezeOnly (no pad/trim)")
        else:
            model.context_model = ContextModelPadTrimWrapper(model.context_model, expected_dim=160).to(device)
            print("[INFO] Wrapped context_model to output (B,160)")

    print("[INFO] Loaded model type:", type(model))

    if not args.no_wrap_patch:
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
    else:
        print("[INFO] no_wrap_patch=True: skip ContextSqueezeWrapper")

    if not args.no_wrap_patch:
        if hasattr(model, "model"):
            den = model.model
            if isinstance(den, torch.nn.DataParallel):
                patched = patch_all_cond_encoders(den.module, expected_dim=160)
            else:
                patched = patch_all_cond_encoders(den, expected_dim=160)
            print(f"[INFO] patched cond_encoder count = {patched}")
    else:
        print("[INFO] no_wrap_patch=True: skip patch_all_cond_encoders")

    n_cases = min(args.n_cases, len(dataset))

    best_start = []
    best_goal = []
    best_grasp = []
    best_succ_all3 = []

    # NEW: smoothness stats (best_by_grasp sample)
    best_uav_a = []
    best_uav_j = []
    best_yaw_a = []
    best_arm_a = []

    for idx in range(n_cases):
        data_sample = dataset[idx]

        data_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in data_sample.items()}
        for k, v in list(data_sample.items()):
            if torch.is_tensor(v):
                data_sample[k] = v.to(device)

        q_start = data_sample["q_start"]       # (9,)
        q_goal = data_sample["q_goal"]         # (9,)
        q_grasp_xyz = data_sample["q_grasp"]   # (3,)

        q_start_np = q_start.detach().cpu().numpy()
        q_goal_np = q_goal.detach().cpu().numpy()
        q_grasp_np = q_grasp_xyz.detach().cpu().numpy()

        # =================== context ===================
        context_d = dataset.build_context(data_sample=data_sample)
        adapt_qs_normalized(context_d, QS_DIM_EXPECTED, idx)
        apply_ctx_mode_to_context(context_d, args.ctx_mode, idx=idx)

        for k, v in list(context_d.items()):
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] == 1:
                context_d[k] = v.squeeze(1)

        # =================== sampling config ===================
        if args.mode == "free":
            hard_conds = {}
            t_g_list = [None]
            per_tg = args.n_samples

        elif args.mode == "endpoints_hard":
            q_start_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_start_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_start_hc"
            )
            q_goal_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_goal_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_goal_hc"
            )
            hard_conds = {0: q_start_hc_1d, H - 1: q_goal_hc_1d}
            t_g_list = [None]
            per_tg = args.n_samples

        else:
            q_start_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_start_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_start_hc"
            )
            q_goal_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_goal_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_goal_hc"
            )

            q_init_np = q_start_np
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
            q_g_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_g_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_g_hc"
            )

            if args.tg_scan:
                t_g_list = [24, 40, 56, 72, 88, 104, 120]
                t_g_list = [t for t in t_g_list if 1 <= t <= H - 2]
            else:
                t_g_list = [int(max(1, min(H - 2, args.t_g)))]
            per_tg = max(1, int(np.ceil(args.n_samples / max(1, len(t_g_list)))))

            hard_conds = {}

        # =================== run inference ===================
        all_cp_norm = []
        all_tag = []

        for t_g in t_g_list:
            if args.mode == "endpoints_and_mid_hard":
                hard_conds_t = {0: q_start_hc_1d, H - 1: q_goal_hc_1d, int(t_g): q_g_hc_1d}
            else:
                hard_conds_t = hard_conds if hard_conds is not None else {}

            if hard_conds_t is None:
                hard_conds_t = {}

            context_d_tg = {}
            for k, v in context_d.items():
                if torch.is_tensor(v):
                    context_d_tg[k] = _context_to_batch(v, per_tg)
                else:
                    context_d_tg[k] = v

            cp_norm_tg = model.run_inference(
                context_d=context_d_tg,
                hard_conds=hard_conds_t,
                n_samples=per_tg,
                horizon=H,
            )
            all_cp_norm.append(cp_norm_tg)
            all_tag.append(torch.full((cp_norm_tg.shape[0],), -1 if t_g is None else int(t_g),
                                      device="cpu", dtype=torch.int64))

        cp_norm = torch.cat(all_cp_norm, dim=0)
        tag_tg = torch.cat(all_tag, dim=0)

        cp = dataset.unnormalize_control_points(cp_norm)

        if cp.shape[-1] == 5:
            cp5_np = cp.detach().cpu().numpy()
            cp9_np = np.stack(
                [traj5_to_traj9(cp5_np[i], q_start_np, q_goal_np) for i in range(cp5_np.shape[0])],
                axis=0
            )
            cp = torch.as_tensor(cp9_np, device=cp.device, dtype=torch.float32)
        elif cp.shape[-1] != 9:
            raise ValueError(f"[ERR] Unexpected cp last-dim={cp.shape[-1]} with shape={tuple(cp.shape)}")

        cp_np = cp.detach().cpu().numpy()

        start_pos_gt = np.asarray(q_start_np[:3], dtype=np.float64)
        goal_pos_gt = np.asarray(q_goal_np[:3], dtype=np.float64)

        # ------------------- sanity debug for endpoints/coords -------------------
        if args.dbg_eval and idx == 0:
            xyz = cp_np[:, :, :3]  # (S,H,3)
            xyz_min = xyz.min(axis=(0, 1))
            xyz_max = xyz.max(axis=(0, 1))
            print("[DBG_EVAL] start_xyz(gt):", start_pos_gt)
            print("[DBG_EVAL] goal_xyz (gt):", goal_pos_gt)
            print("[DBG_EVAL] pred_xyz range: min", xyz_min, "max", xyz_max)
            for s_show in [0, min(1, cp_np.shape[0]-1)]:
                p0 = cp_np[s_show, 0, :3]
                pT = cp_np[s_show, -1, :3]
                print(f"[DBG_EVAL] sample{s_show:02d} traj0_xyz={p0} trajT_xyz={pT} "
                      f"|e0|={np.linalg.norm(p0-start_pos_gt):.4f} |eT|={np.linalg.norm(pT-goal_pos_gt):.4f}")
        # -----------------------------------------------------------------------

        # ---- errors + smoothness per sample ----
        S = cp_np.shape[0]
        start_errs = np.zeros((S,), dtype=np.float64)
        goal_errs  = np.zeros((S,), dtype=np.float64)
        grasp_errs = np.zeros((S,), dtype=np.float64)

        # smoothness arrays (store key ones; you can add more later)
        uav_a_mean = np.zeros((S,), dtype=np.float64)
        uav_j_mean = np.zeros((S,), dtype=np.float64)
        yaw_a_mean = np.zeros((S,), dtype=np.float64)
        arm_a_mean = np.zeros((S,), dtype=np.float64)

        for s in range(S):
            start_pos_pred = np.asarray(cp_np[s, 0, :3], dtype=np.float64)
            goal_pos_pred  = np.asarray(cp_np[s, -1, :3], dtype=np.float64)
            start_errs[s] = float(np.linalg.norm(start_pos_pred - start_pos_gt))
            goal_errs[s]  = float(np.linalg.norm(goal_pos_pred - goal_pos_gt))
            grasp_errs[s] = fk_min_grasp_err_from_traj9(cp_np[s], q_grasp_np)

            sm = compute_smooth_metrics_traj9(cp_np[s])
            uav_a_mean[s] = sm["uav_a_mean"]
            uav_j_mean[s] = sm["uav_j_mean"]
            yaw_a_mean[s] = sm["yaw_a_mean"]
            arm_a_mean[s] = sm["arm_a_mean"]

        # pick best by grasp (keep your current selection rule)
        best_idx = int(np.argmin(grasp_errs))
        e_start_best = float(start_errs[best_idx])
        e_goal_best  = float(goal_errs[best_idx])
        e_grasp_best = float(grasp_errs[best_idx])

        # smoothness of that best sample
        uav_a_best = float(uav_a_mean[best_idx])
        uav_j_best = float(uav_j_mean[best_idx])
        yaw_a_best = float(yaw_a_mean[best_idx])
        arm_a_best = float(arm_a_mean[best_idx])

        # success@2cm(all3)
        all3 = (start_errs < args.succ_thresh_m) & (goal_errs < args.succ_thresh_m) & (grasp_errs < args.succ_thresh_m)
        succ_all3 = float(all3.mean())

        # optional debug for smoothness
        if args.dbg_smooth and idx == 0:
            show_ids = [0, min(1, S - 1), best_idx]
            show_ids = list(dict.fromkeys(show_ids))  # unique keep order
            for sid in show_ids:
                sm = compute_smooth_metrics_traj9(cp_np[sid])
                print(f"[DBG_SMOOTH] sample{sid:02d} "
                      f"uav_a_mean={sm['uav_a_mean']:.6f} uav_j_mean={sm['uav_j_mean']:.6f} "
                      f"yaw_a_mean={sm['yaw_a_mean']:.6f} arm_a_mean={sm['arm_a_mean']:.6f}")

        print(
            f"[CASE {idx:04d}] "
            f"e_start_uav={e_start_best:.4f} m | "
            f"e_goal_uav={e_goal_best:.4f} m | "
            f"e_grasp_ee_min={e_grasp_best:.4f} m | "
            f"smooth(uav_a={uav_a_best:.6f}, uav_j={uav_j_best:.6f}, yaw_a={yaw_a_best:.6f}, arm_a={arm_a_best:.6f}) | "
            f"succ@2cm(all3)={succ_all3:.3f}"
        )

        best_start.append(e_start_best)
        best_goal.append(e_goal_best)
        best_grasp.append(e_grasp_best)
        best_succ_all3.append(succ_all3)

        best_uav_a.append(uav_a_best)
        best_uav_j.append(uav_j_best)
        best_yaw_a.append(yaw_a_best)
        best_arm_a.append(arm_a_best)

        out_npz = os.path.join(args.save_dir, f"case_{idx:04d}_samples.npz")
        np.savez(
            out_npz,
            cp=cp_np,
            q_start=data_cpu["q_start"].numpy(),
            q_goal=data_cpu["q_goal"].numpy(),
            q_grasp=data_cpu["q_grasp"].numpy(),
            best_idx=np.array([best_idx], dtype=np.int64),
            start_errs=start_errs.astype(np.float32),
            goal_errs=goal_errs.astype(np.float32),
            grasp_errs=grasp_errs.astype(np.float32),
            succ_all3=np.array([succ_all3], dtype=np.float32),
            # NEW: smoothness arrays per sample
            uav_a_mean=uav_a_mean.astype(np.float32),
            uav_j_mean=uav_j_mean.astype(np.float32),
            yaw_a_mean=yaw_a_mean.astype(np.float32),
            arm_a_mean=arm_a_mean.astype(np.float32),
            mode=np.array([args.mode]),
            ctx_mode=np.array([args.ctx_mode]),
            seed=np.array([args.seed], dtype=np.int64),
            no_wrap_patch=np.array([int(args.no_wrap_patch)], dtype=np.int64),
            t_g_tag=tag_tg.numpy(),
        )

    if len(best_start) > 0:
        s0 = _stats_np(best_start)
        sg = _stats_np(best_goal)
        sk = _stats_np(best_grasp)
        succ = float(np.mean(np.asarray(best_succ_all3, dtype=np.float64)))

        sm_uav_a = _stats_np(best_uav_a)
        sm_uav_j = _stats_np(best_uav_j)
        sm_yaw_a = _stats_np(best_yaw_a)
        sm_arm_a = _stats_np(best_arm_a)

        print("\n[SUMMARY] (best_by_grasp over samples; meters)")
        print(f"[METRIC] e_start_uav(m):     mean={s0['mean']:.6f}, p50={s0['p50']:.6f}, p90={s0['p90']:.6f}")
        print(f"[METRIC] e_goal_uav(m):      mean={sg['mean']:.6f}, p50={sg['p50']:.6f}, p90={sg['p90']:.6f}")
        print(f"[METRIC] e_grasp_ee_min(m):  mean={sk['mean']:.6f}, p50={sk['p50']:.6f}, p90={sk['p90']:.6f}")
        print(f"[METRIC] succ@2cm(all3): {succ:.4f}")

        print("\n[SUMMARY] (smoothness of best_by_grasp sample; per-step discrete diffs)")
        print(f"[SMOOTH] uav_a_mean: mean={sm_uav_a['mean']:.6f}, p50={sm_uav_a['p50']:.6f}, p90={sm_uav_a['p90']:.6f}")
        print(f"[SMOOTH] uav_j_mean: mean={sm_uav_j['mean']:.6f}, p50={sm_uav_j['p50']:.6f}, p90={sm_uav_j['p90']:.6f}")
        print(f"[SMOOTH] yaw_a_mean: mean={sm_yaw_a['mean']:.6f}, p50={sm_yaw_a['p50']:.6f}, p90={sm_yaw_a['p90']:.6f}")
        print(f"[SMOOTH] arm_a_mean: mean={sm_arm_a['mean']:.6f}, p50={sm_arm_a['p50']:.6f}, p90={sm_arm_a['p90']:.6f}")


if __name__ == "__main__":
    main()