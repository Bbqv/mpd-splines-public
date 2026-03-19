import os
import argparse
import numpy as np
import math

import torch
import pinocchio as pin
import torch.nn as nn
import collections

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
from mpd.models.diffusion_models.models import TemporalUnet
from mpd.models.diffusion_models import GaussianDiffusionModel
from mpd.planning.obstacle_guidance import (
    parse_obstacle_spheres as _parse_obstacle_spheres_impl,
    min_clearance_to_spheres_traj9 as _min_clearance_to_spheres_traj9_impl,
    apply_obstacle_projection_with_adaptive,
    make_proj_only_adaptive_preset,
    sample_random_spheres_uniform,
    spheres_from_point_cloud,
)


# =============================================================================
# small utils: infer ckpt expected conditioning dim
# =============================================================================

def infer_cond_dim_from_denoiser(model) -> int:
    """
    Infer the true conditioning embedding dim expected by this checkpoint.
    We read the first Linear(in_features=K) inside the first cond_encoder we can find.
    Typical values: 160 or 192.
    """
    den = getattr(model, "model", None)
    if den is None:
        return None
    if isinstance(den, torch.nn.DataParallel):
        den = den.module

    for m in den.modules():
        if hasattr(m, "cond_encoder") and isinstance(getattr(m, "cond_encoder"), nn.Module):
            ce = getattr(m, "cond_encoder")
            for sub in ce.modules():
                if isinstance(sub, nn.Linear):
                    return int(sub.in_features)
    return None


# =============================================================================
# cond_encoder adapters
# =============================================================================

class CondEncoderInputAdapter(nn.Module):
    """
    Wrap an existing cond_encoder and force input feature dim to its expected in_features.
    If expected_dim is None, infer from inner's first Linear(in_features).
    """
    def __init__(self, inner: nn.Module, expected_dim: int = None):
        super().__init__()
        self.inner = inner
        self._printed = False

        if expected_dim is None:
            inferred = None
            for m in self.inner.modules():
                if isinstance(m, nn.Linear):
                    inferred = int(m.in_features)
                    break
            self.expected_dim = inferred
        else:
            self.expected_dim = int(expected_dim)

    def forward(self, c):
        if torch.is_tensor(c) and c.dim() == 2 and self.expected_dim is not None:
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


def patch_all_cond_encoders(denoiser: nn.Module) -> int:
    """
    Find all submodules that have attribute 'cond_encoder' and wrap it with auto-dim adapter.
    """
    cnt = 0
    for m in denoiser.modules():
        if hasattr(m, "cond_encoder"):
            ce = getattr(m, "cond_encoder")
            if isinstance(ce, nn.Module):
                setattr(m, "cond_encoder", CondEncoderInputAdapter(ce, expected_dim=None))
                cnt += 1
    return cnt


# =============================================================================
# wrappers / pickle placeholders
# =============================================================================

class ContextSqueezeWrapper(nn.Module):
    """
    Wrap denoiser forward(x,t,context):
      - ensure context is 2D (B,K)
      - if context is 3D (B,L,K): mean-pool over L -> (B,K)
      - then pad/trim K to expected_dim (MUST match ckpt cond_encoder in_features, e.g. 192)
    """
    def __init__(self, inner: nn.Module, expected_dim: int = None):
        super().__init__()
        self.inner = inner
        self.expected_dim = expected_dim  # if None, infer on first forward
        self._printed = False

    def _infer_expected_dim(self):
        # Prefer actual cond_encoder Linear in_features
        if hasattr(self.inner, "cond_encoder"):
            ce = getattr(self.inner, "cond_encoder")
            if isinstance(ce, nn.Module):
                for m in ce.modules():
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)

        # fallback: scan some common names
        for name in ["cond_encoder", "context_encoder", "cond_mlp", "c_mlp"]:
            if hasattr(self.inner, name):
                mod = getattr(self.inner, name)
                if isinstance(mod, nn.Module):
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


# =============================================================================
# CRITICAL FIX: Safe adapter for ckpt ContextModelCombined mismatch
# =============================================================================

class SafeContextModelAdapter(nn.Module):
    """
    Some ckpts have ContextModelCombined like:
      - context_model_qs: produces emb_q = (B,128) from qs_normalized (22)
      - context_model_sg: produces emb_sg = (B,32) from sg_xyz_normalized (6)
      - emb = cat -> (B,160)
      - net expects in_dim=160 and outputs context_emb dim = (B,COND_DIM_EXPECTED) (often 192)

    But sometimes inner.forward fails due to missing key or shape.
    This adapter:
      1) try inner(**kwargs) first
      2) if fails: manually calls available branches, concat, pad/trim to inner.net in_features, then runs inner.net
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._printed = False
        self._printed_pad = False

    def _get_net_expected_in(self):
        if hasattr(self.inner, "net") and isinstance(self.inner.net, nn.Module):
            for m in self.inner.net.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)
        return None

    def forward(self, **kwargs):
        # fast path
        try:
            return self.inner(**kwargs)
        except Exception as e:
            if not self._printed:
                print("[CTX_SAFE] inner forward failed, switching to safe compose. err=", repr(e))
                self._printed = True

        emb_list = []

        # qs branch
        if hasattr(self.inner, "context_model_qs") and self.inner.context_model_qs is not None:
            qs = kwargs.get("qs_normalized", None)
            if qs is not None:
                emb_q = self.inner.context_model_qs(qs)
                emb_list.append(emb_q)

        # ee branch (rare)
        if hasattr(self.inner, "context_model_ee_pose_goal") and self.inner.context_model_ee_pose_goal is not None:
            eo = kwargs.get("ee_goal_orientation_normalized", None)
            ep = kwargs.get("ee_goal_position_normalized", None)
            if eo is not None and ep is not None:
                emb_ee = self.inner.context_model_ee_pose_goal(eo, ep)
                emb_list.append(emb_ee)

        # sg branch
        if hasattr(self.inner, "context_model_sg") and self.inner.context_model_sg is not None:
            sg = kwargs.get("sg_xyz_normalized", None)
            if sg is not None:
                if torch.is_tensor(sg) and sg.dim() == 1:
                    sg = sg.unsqueeze(0)
                emb_sg = self.inner.context_model_sg(sg)
                emb_list.append(emb_sg)

        if len(emb_list) == 0:
            raise RuntimeError("[CTX_SAFE] no emb branches produced; check context_d keys")

        emb = torch.cat(emb_list, dim=-1) if len(emb_list) > 1 else emb_list[0]

        # force emb to (B,K)
        if torch.is_tensor(emb):
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            elif emb.dim() == 3:
                emb = emb.mean(dim=1)
            elif emb.dim() > 3:
                emb = emb.reshape(emb.shape[0], -1)

        exp_in = self._get_net_expected_in()
        if exp_in is not None:
            if emb.dim() != 2:
                raise RuntimeError(f"[CTX_SAFE] emb must be 2D (B,K) before pad/trim, got {tuple(emb.shape)}")
            K = int(emb.shape[1])
            if K != exp_in:
                if not self._printed_pad:
                    print(f"[CTX_SAFE] pad/trim emb {K} -> {exp_in} for inner.net")
                    self._printed_pad = True
                if K > exp_in:
                    emb = emb[:, :exp_in].contiguous()
                else:
                    pad = torch.zeros(emb.shape[0], exp_in - K, device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=1).contiguous()

        if hasattr(self.inner, "net") and isinstance(self.inner.net, nn.Module):
            return self.inner.net(emb)

        return emb


# =============================================================================
# FK / pinocchio
# =============================================================================

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


# =============================================================================
# helpers
# =============================================================================

def _context_to_batch(x: torch.Tensor, B: int) -> torch.Tensor:
    if x is None:
        return None

    if torch.is_tensor(x) and x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    if torch.is_tensor(x) and x.dim() == 0:
        x = x.view(1)

    if torch.is_tensor(x) and x.dim() == 1:
        if x.numel() == 1:
            return x.view(1).repeat(B).contiguous()
        x = x.unsqueeze(0)

    if torch.is_tensor(x) and x.dim() == 2 and x.shape[1] == 1:
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
    if q_1x9.dim() == 1:
        q_1x9 = q_1x9.unsqueeze(0)
    if tuple(q_1x9.shape) != (1, 9):
        raise RuntimeError(f"{name} expected (1,9), got {tuple(q_1x9.shape)}")

    tmp = q_1x9.unsqueeze(1)          # (1,1,9)
    tmp = try_normalize_cp(dataset, tmp)

    if tmp.dim() == 3:
        tmp = tmp[:, 0, :]
    elif tmp.dim() == 2:
        pass
    else:
        raise RuntimeError(f"{name} normalize returned weird shape {tuple(tmp.shape)}")

    if tuple(tmp.shape) != (1, 9):
        raise RuntimeError(f"{name} normalize expected (1,9), got {tuple(tmp.shape)}")

    return tmp.squeeze(0).contiguous()


def get_expected_qs_dim(model) -> int:
    try:
        cm = model.context_model
        if isinstance(cm, SafeContextModelAdapter):
            cm = cm.inner
        if hasattr(cm, "context_model_qs") and cm.context_model_qs is not None:
            for m in cm.context_model_qs.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)
    except Exception:
        pass
    return None


def adapt_qs_normalized(context_d: dict, expected_dim: int, idx: int):
    # qs_normalized must match ckpt (usually 22). Do not pad to cond_dim here.
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


def _stats_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


# =============================================================================
# smoothness metrics
# =============================================================================

def _wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_from_xyzw(q_xyzw):
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_smooth_metrics_traj9(traj9_np):
    traj = np.asarray(traj9_np, dtype=np.float64)
    H = traj.shape[0]
    out = {}

    xyz = traj[:, 0:3]
    if H >= 2:
        v = xyz[1:] - xyz[:-1]
        v_norm = np.linalg.norm(v, axis=1)
        out["uav_v_mean"] = float(v_norm.mean())
        out["uav_v_max"] = float(v_norm.max())
    else:
        out["uav_v_mean"] = 0.0
        out["uav_v_max"] = 0.0

    if H >= 3:
        a = (xyz[2:] - 2.0 * xyz[1:-1] + xyz[:-2])
        a_norm = np.linalg.norm(a, axis=1)
        out["uav_a_mean"] = float(a_norm.mean())
        out["uav_a_max"] = float(a_norm.max())
    else:
        out["uav_a_mean"] = 0.0
        out["uav_a_max"] = 0.0

    if H >= 4:
        j = (xyz[3:] - 3.0 * xyz[2:-1] + 3.0 * xyz[1:-2] - xyz[:-3])
        j_norm = np.linalg.norm(j, axis=1)
        out["uav_j_mean"] = float(j_norm.mean())
        out["uav_j_max"] = float(j_norm.max())
    else:
        out["uav_j_mean"] = 0.0
        out["uav_j_max"] = 0.0

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

    arm = traj[:, 7:9]
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


def _parse_obst_spheres(spec_list):
    return _parse_obstacle_spheres_impl(spec_list)


def _parse_xyz3(spec: str, name: str) -> np.ndarray:
    s = str(spec).replace(" ", "")
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"[ERR] {name} expects x,y,z, got: {spec}")
    out = np.asarray([float(v) for v in parts], dtype=np.float32)
    return out


def min_clearance_to_spheres_traj9(traj9: np.ndarray, spheres) -> float:
    return _min_clearance_to_spheres_traj9_impl(traj9, spheres)


# =============================================================================
# main
# =============================================================================

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

    ap.add_argument(
        "--mode",
        type=str,
        default="free",
        choices=["free", "endpoints_hard", "endpoints_and_mid_hard"],
        help="free: no hard cond. endpoints_hard: hard endpoints only. endpoints_and_mid_hard: hard endpoints+mid grasp."
    )
    ap.add_argument("--tg_scan", action="store_true")
    ap.add_argument("--t_g", type=int, default=24)
    ap.add_argument("--succ_thresh_m", type=float, default=0.02)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx_mode", type=str, default="orig", choices=["orig", "swap_start_goal", "zero_ctx"])
    ap.add_argument("--no_wrap_patch", action="store_true")

    ap.add_argument("--dbg_eval", action="store_true")
    ap.add_argument("--dbg_smooth", action="store_true")

    # Optional: CGD-style obstacle guidance in DDIM sampling
    ap.add_argument("--obst_enable", action="store_true",
                    help="Enable CGD-style obstacle pushing during DDIM sampling.")
    ap.add_argument("--obst_sphere", action="append", default=[],
                    help="Obstacle sphere as x,y,z,r. Repeat for multiple spheres.")
    ap.add_argument("--obst_alpha", type=float, default=0.0,
                    help="CGD pushing scale alpha_obst.")
    ap.add_argument("--obst_margin", type=float, default=0.10,
                    help="Only push when dist < radius + margin.")
    ap.add_argument("--obst_eps", type=float, default=1e-4)
    ap.add_argument("--obst_max_push", type=float, default=0.0,
                    help="Clamp per-step push magnitude.")
    ap.add_argument("--obst_time_smooth_k", type=int, default=5,
                    help="Odd kernel size for time smoothing on guidance gradient (0 to disable).")
    ap.add_argument("--obst_alpha_ramp_power", type=float, default=2.0,
                    help="Guide alpha ramp power across active denoising steps (higher = gentler start).")
    ap.add_argument("--obst_alpha_min_scale", type=float, default=0.0,
                    help="Minimum relative alpha scale at first guided step.")

    # DDIM guidance knobs
    ap.add_argument("--obst_ddim_steps", type=int, default=100)
    ap.add_argument("--obst_t_start_guide", type=int, default=0)
    ap.add_argument("--obst_guide_lr", type=float, default=0.08)
    ap.add_argument("--obst_n_guide_steps", type=int, default=1)
    ap.add_argument("--obst_max_grad_value", type=float, default=1.0)
    ap.add_argument("--obst_max_perturb_x", type=float, default=0.1)
    ap.add_argument("--obst_scale_grad_by_one_minus_alpha", action="store_true",
                    help="Scale guide gradient by sqrt(1-alpha_t) in DDIM (usually smoother).")
    ap.add_argument("--obst_use_xrecon", action="store_true",
                    help="Compute obstacle guide on x_recon instead of noisy x.")
    ap.add_argument("--obst_select_min_clearance", type=float, default=0.0,
                    help="When obstacle is enabled, select best sample among clr_min >= threshold first.")
    # Optional: post-projection (CGD-style simplified) after diffusion sampling
    ap.add_argument("--obst_project_enable", dest="obst_project_enable", action="store_true",
                    help="Enable post-projection refinement after sampling.")
    ap.add_argument("--obst_project_disable", dest="obst_project_enable", action="store_false",
                    help="Disable post-projection refinement after sampling.")
    ap.add_argument("--obst_proj_iters", type=int, default=80)
    ap.add_argument("--obst_proj_lr", type=float, default=0.005)
    ap.add_argument("--obst_proj_w_data", type=float, default=40.0)
    ap.add_argument("--obst_proj_w_a", type=float, default=30.0)
    ap.add_argument("--obst_proj_w_j", type=float, default=8.0)
    ap.add_argument("--obst_proj_w_obst", type=float, default=80.0)
    ap.add_argument("--obst_proj_margin", type=float, default=0.05,
                    help="If <0, fallback to --obst_margin.")
    ap.add_argument("--obst_proj_max_grad_value", type=float, default=0.05)
    ap.add_argument("--obst_proj_max_delta", type=float, default=0.03)
    ap.add_argument("--obst_proj_adaptive_enable", dest="obst_proj_adaptive_enable", action="store_true",
                    help="Enable adaptive re-projection only for still-in-collision samples.")
    ap.add_argument("--obst_proj_adaptive_disable", dest="obst_proj_adaptive_enable", action="store_false",
                    help="Disable adaptive re-projection.")
    ap.add_argument("--obst_proj_adaptive_rounds", type=int, default=2,
                    help="Maximum adaptive re-projection rounds.")
    ap.add_argument("--obst_proj_adaptive_clearance", type=float, default=0.0,
                    help="Target minimum clearance used to define infeasible samples.")
    ap.add_argument("--obst_proj_adaptive_iters_mult", type=float, default=2.0,
                    help="Per-round multiplier on projection iterations.")
    ap.add_argument("--obst_proj_adaptive_w_obst_mult", type=float, default=2.0,
                    help="Per-round multiplier on obstacle loss weight.")
    ap.add_argument("--obst_proj_adaptive_max_delta_mult", type=float, default=2.0,
                    help="Per-round multiplier on max per-state delta clamp.")
    ap.add_argument("--obst_proj_adaptive_max_grad_mult", type=float, default=1.5,
                    help="Per-round multiplier on value gradient clipping.")
    ap.set_defaults(obst_project_enable=True, obst_proj_adaptive_enable=True)
    # MPD-style random obstacle generation
    ap.add_argument("--obst_random_enable", action="store_true",
                    help="Append randomly sampled spheres per case (MPD-style).")
    ap.add_argument("--obst_random_n", type=int, default=0,
                    help="Number of random spheres sampled per case.")
    ap.add_argument("--obst_random_xyz_min", type=str, default="-0.40,-0.40,0.80")
    ap.add_argument("--obst_random_xyz_max", type=str, default="0.40,0.40,1.80")
    ap.add_argument("--obst_random_r_min", type=float, default=0.12)
    ap.add_argument("--obst_random_r_max", type=float, default=0.24)
    ap.add_argument("--obst_random_anchor_clearance", type=float, default=0.0,
                    help="Minimum clearance from hard anchor xyz points when sampling.")
    ap.add_argument("--obst_random_avoid_overlap", action="store_true",
                    help="Avoid sphere-sphere overlaps during random sampling.")
    ap.add_argument("--obst_random_overlap_margin", type=float, default=0.0)
    ap.add_argument("--obst_random_seed_offset", type=int, default=0)
    ap.add_argument("--obst_random_max_tries", type=int, default=5000)
    # GDN-style point cloud obstacles (converted to tiny spheres)
    ap.add_argument("--obst_pc_file", type=str, default="",
                    help="Optional .npy/.npz point cloud file (N,3).")
    ap.add_argument("--obst_pc_key", type=str, default="points",
                    help="Key used when --obst_pc_file is .npz.")
    ap.add_argument("--obst_pc_from_sample_key", type=str, default="",
                    help="Read per-sample point cloud from dataset key if present.")
    ap.add_argument("--obst_pc_radius", type=float, default=0.02)
    ap.add_argument("--obst_pc_max_points", type=int, default=256)
    ap.add_argument("--obst_pc_seed_offset", type=int, default=0)

    args = ap.parse_args()

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[SENS] seed={args.seed}, ctx_mode={args.ctx_mode}, no_wrap_patch={args.no_wrap_patch}")
    fixed_spheres = _parse_obst_spheres(args.obst_sphere)
    rand_xyz_min = _parse_xyz3(args.obst_random_xyz_min, "obst_random_xyz_min")
    rand_xyz_max = _parse_xyz3(args.obst_random_xyz_max, "obst_random_xyz_max")

    static_pc_points = None
    if args.obst_pc_file:
        if not os.path.exists(args.obst_pc_file):
            raise RuntimeError(f"[ERR] obst pc file not found: {args.obst_pc_file}")
        if args.obst_pc_file.endswith(".npy"):
            static_pc_points = np.asarray(np.load(args.obst_pc_file), dtype=np.float32).reshape(-1, 3)
        elif args.obst_pc_file.endswith(".npz"):
            with np.load(args.obst_pc_file) as zf:
                if args.obst_pc_key in zf:
                    arr = zf[args.obst_pc_key]
                elif len(zf.files) > 0:
                    arr = zf[zf.files[0]]
                    print(f"[OBST_PC] key '{args.obst_pc_key}' not found; fallback to '{zf.files[0]}'")
                else:
                    raise RuntimeError(f"[ERR] empty npz file: {args.obst_pc_file}")
            static_pc_points = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        else:
            raise RuntimeError(f"[ERR] obst pc file must be .npy/.npz: {args.obst_pc_file}")

    has_random = bool(args.obst_random_enable and int(args.obst_random_n) > 0)
    has_pc = bool((static_pc_points is not None) or args.obst_pc_from_sample_key)
    has_any_source = (len(fixed_spheres) > 0) or has_random or has_pc

    if args.obst_enable:
        if not has_any_source:
            raise RuntimeError(
                "[ERR] --obst_enable requires obstacle source: --obst_sphere and/or "
                "--obst_random_enable --obst_random_n>0 and/or --obst_pc_file/--obst_pc_from_sample_key"
            )
        print(
            f"[OBST] enabled fixed={len(fixed_spheres)} random={int(has_random)} pc={int(has_pc)} "
            f"alpha={args.obst_alpha} margin={args.obst_margin} "
            f"ddim_steps={args.obst_ddim_steps} t_start={args.obst_t_start_guide} "
            f"guide_lr={args.obst_guide_lr} n_guide_steps={args.obst_n_guide_steps} "
            f"time_smooth_k={args.obst_time_smooth_k} "
            f"alpha_ramp_power={args.obst_alpha_ramp_power} alpha_min_scale={args.obst_alpha_min_scale} "
            f"scale_by_1m_alpha={bool(args.obst_scale_grad_by_one_minus_alpha)}"
        )
        for i, (c, r) in enumerate(fixed_spheres):
            print(f"[OBST] fixed_sphere{i}: center={c.tolist()} r={r:.4f}")
        if has_random:
            print(
                f"[OBST_RANDOM] n={int(args.obst_random_n)} xyz_min={rand_xyz_min.tolist()} "
                f"xyz_max={rand_xyz_max.tolist()} r=[{float(args.obst_random_r_min):.4f},{float(args.obst_random_r_max):.4f}] "
                f"anchor_clearance={float(args.obst_random_anchor_clearance):.4f} "
                f"avoid_overlap={int(bool(args.obst_random_avoid_overlap))}"
            )
        if static_pc_points is not None:
            print(f"[OBST_PC] loaded static points={int(static_pc_points.shape[0])} file={args.obst_pc_file}")
        if args.obst_pc_from_sample_key:
            print(f"[OBST_PC] per-sample key='{args.obst_pc_from_sample_key}'")
    elif has_any_source:
        print("[OBST] obstacle source provided but obst_enable=0; ignoring.")
    if args.obst_enable and args.obst_project_enable:
        proj_margin = float(args.obst_margin) if float(args.obst_proj_margin) < 0.0 else float(args.obst_proj_margin)
        print(
            f"[OBST_PROJ] enabled iters={args.obst_proj_iters} lr={args.obst_proj_lr} "
            f"w_data={args.obst_proj_w_data} w_a={args.obst_proj_w_a} w_j={args.obst_proj_w_j} "
            f"w_obst={args.obst_proj_w_obst} margin={proj_margin} "
            f"max_grad={args.obst_proj_max_grad_value} max_delta={args.obst_proj_max_delta}"
        )
        if args.obst_proj_adaptive_enable:
            print(
                f"[OBST_PROJ] adaptive rounds={args.obst_proj_adaptive_rounds} "
                f"clearance={args.obst_proj_adaptive_clearance} "
                f"iters_mult={args.obst_proj_adaptive_iters_mult} "
                f"w_obst_mult={args.obst_proj_adaptive_w_obst_mult} "
                f"max_delta_mult={args.obst_proj_adaptive_max_delta_mult} "
                f"max_grad_mult={args.obst_proj_adaptive_max_grad_mult}"
            )

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

    def build_model_for_state_dict(state_dict, device, H):
        """
        为你训练出来的 2pt ckpt(state_dict) 重建 GaussianDiffusionModel + TemporalUnet
        关键点：
        - H=144, dx=9
        - n_diffusion_steps=100
        - variance_schedule='cosine'
        - unet_input_dim=32, dim_mults=(1,2,4,8)
        - conditioning_embed_dim 这里对 2pt 我们用 18（start+goal）
        """
        dx = 9
        cond_dim = 18

        denoise_fn = TemporalUnet(
            n_support_points=H,
            state_dim=dx,
            unet_input_dim=32,
            dim_mults=(1, 2, 4, 8),
            self_attention=False,
            conditioning_embed_dim=cond_dim,
        )

        model = GaussianDiffusionModel(
            denoise_fn=denoise_fn,
            variance_schedule="cosine",
            n_diffusion_steps=100,
            clip_denoised=True,
            predict_epsilon=True,
            loss_type="l2",
            horizon=H,
            observation_dim=dx,
            action_dim=0,
            device=device,
        ).to(device)

        # 允许有些 key 对不上（比如你旧 eval 里 wrapper / patch）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[CKPT_LOAD] missing keys:", len(missing))
        print("[CKPT_LOAD] unexpected keys:", len(unexpected))
        if len(unexpected) > 0:
            print("[CKPT_LOAD] unexpected example:", unexpected[:10])
        if len(missing) > 0:
            print("[CKPT_LOAD] missing example:", missing[:10])

        return model


    ckpt_obj = torch.load(args.ckpt, map_location="cpu")

    # ---------- CASE A: ckpt 本身就是一个可直接推断的模型对象 ----------
    if hasattr(ckpt_obj, "run_inference"):
        model = ckpt_obj.to(device).eval()
        print("[CKPT] loaded pickled model object:", type(model))

    # ---------- CASE B: dict 包装（比如 {'step','model','optimizer'}） ----------
    elif isinstance(ckpt_obj, dict):
        keys = list(ckpt_obj.keys())
        print("[CKPT] dict keys:", keys)

        # 1) 你的训练脚本保存：{'step','model','optimizer'}，其中 'model' 是 state_dict
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], (dict, collections.OrderedDict)):
            state_dict = ckpt_obj["model"]
            print("[CKPT] using ckpt_obj['model'] as state_dict")

        # 2) 某些脚本：{'ema_model': state_dict}
        elif "ema_model" in ckpt_obj and isinstance(ckpt_obj["ema_model"], (dict, collections.OrderedDict)):
            state_dict = ckpt_obj["ema_model"]
            print("[CKPT] using ckpt_obj['ema_model'] as state_dict")

        # 3) dict 自己就是 state_dict（heuristic）
        elif ("betas" in ckpt_obj) or any(isinstance(k, str) and k.startswith("model.") for k in ckpt_obj.keys()):
            state_dict = ckpt_obj
            print("[CKPT] dict looks like state_dict itself")

        else:
            raise RuntimeError(f"[ERR] Unknown dict ckpt format. keys={keys}")

        model = build_model_for_state_dict(state_dict, device=device, H=args.H).eval()
        print("[CKPT] built model from state_dict:", type(model))

    # ---------- CASE C: OrderedDict（纯 state_dict） ----------
    elif isinstance(ckpt_obj, collections.OrderedDict):
        print("[CKPT] got OrderedDict state_dict, building model ...")
        model = build_model_for_state_dict(ckpt_obj, device=device, H=args.H).eval()
        print("[CKPT] built model from OrderedDict state_dict:", type(model))

    else:
        raise RuntimeError(f"[ERR] Unknown ckpt format. type={type(ckpt_obj)}")

    # IMPORTANT: infer true cond dim from ckpt (e.g., 192)
    COND_DIM_EXPECTED = infer_cond_dim_from_denoiser(model)
    print("[INFO] COND_DIM_EXPECTED =", COND_DIM_EXPECTED)

    # wrap context_model with Safe adapter
    if hasattr(model, "context_model") and model.context_model is not None:
        model.context_model = SafeContextModelAdapter(model.context_model).to(device)
        print("[INFO] context_model wrapped by SafeContextModelAdapter")

    QS_DIM_EXPECTED = get_expected_qs_dim(model)
    print("[INFO] QS_DIM_EXPECTED =", QS_DIM_EXPECTED)
    print("[INFO] Loaded model type:", type(model))

    # denoiser wrappers: squeeze/pad context to COND_DIM_EXPECTED (NOT hard-coded 160)
    if not args.no_wrap_patch:
        if hasattr(model, "model"):
            denoiser = model.model
            if isinstance(denoiser, torch.nn.DataParallel):
                inner = denoiser.module
                wrapped = ContextSqueezeWrapper(inner, expected_dim=COND_DIM_EXPECTED).to(device)
                model.model = torch.nn.DataParallel(wrapped)
            else:
                model.model = ContextSqueezeWrapper(denoiser, expected_dim=COND_DIM_EXPECTED).to(device)
            print(f"[INFO] Wrapped denoiser with ContextSqueezeWrapper (squeeze/pad context to {COND_DIM_EXPECTED})")
    else:
        print("[INFO] no_wrap_patch=True: skip ContextSqueezeWrapper")

    # patch all cond_encoders: auto infer their own in_features (so 192 stays 192)
    if not args.no_wrap_patch:
        if hasattr(model, "model"):
            den = model.model
            if isinstance(den, torch.nn.DataParallel):
                patched = patch_all_cond_encoders(den.module)
            else:
                patched = patch_all_cond_encoders(den)
            print(f"[INFO] patched cond_encoder count = {patched}")
    else:
        print("[INFO] no_wrap_patch=True: skip patch_all_cond_encoders")

    n_cases = min(args.n_cases, len(dataset))

    best_start = []
    best_goal = []
    best_grasp = []
    best_succ_all3 = []

    best_uav_a = []
    best_uav_j = []
    best_yaw_a = []
    best_arm_a = []
    best_obst_clr = []
    best_anchor_feasible = []

    printed_ctx_once = False

    for idx in range(n_cases):
        data_sample = dataset[idx]

        data_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in data_sample.items()}
        for k, v in list(data_sample.items()):
            if torch.is_tensor(v):
                data_sample[k] = v.to(device)

        q_start = data_sample["q_start"]
        q_goal = data_sample["q_goal"]
        q_grasp_xyz = data_sample["q_grasp"]

        q_start_np = q_start.detach().cpu().numpy()
        q_goal_np = q_goal.detach().cpu().numpy()
        q_grasp_np = q_grasp_xyz.detach().cpu().numpy()

        # =================== context ===================
        context_d = dataset.build_context(data_sample=data_sample)

        # ensure sg key exists (only used if ckpt has sg branch)
        if "sg_xyz_normalized" not in context_d:
            if "context_sg_xyz_normalized" in context_d:
                context_d["sg_xyz_normalized"] = context_d["context_sg_xyz_normalized"]
            elif "context_sg_xyz_normalized" in data_sample:
                context_d["sg_xyz_normalized"] = data_sample["context_sg_xyz_normalized"]
            elif "sg_xyz_normalized" in data_sample:
                context_d["sg_xyz_normalized"] = data_sample["sg_xyz_normalized"]

        if "sg_xyz_normalized" in context_d and torch.is_tensor(context_d["sg_xyz_normalized"]):
            if context_d["sg_xyz_normalized"].dim() == 1:
                context_d["sg_xyz_normalized"] = context_d["sg_xyz_normalized"].unsqueeze(0)

        adapt_qs_normalized(context_d, QS_DIM_EXPECTED, idx)
        apply_ctx_mode_to_context(context_d, args.ctx_mode, idx=idx)

        # squeeze (B,1,K) -> (B,K)
        for k, v in list(context_d.items()):
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] == 1:
                context_d[k] = v.squeeze(1)

        if (not printed_ctx_once) and idx == 0:
            print("[DBG_CTX_KEYS] context_d keys:", list(context_d.keys()))
            for kk, vv in context_d.items():
                if torch.is_tensor(vv):
                    print("[DBG_CTX_SHAPE]", kk, tuple(vv.shape))
            printed_ctx_once = True

        # =================== sampling config ===================
        if args.mode == "free":
            hard_conds = {}
            t_g_list = [None]
            per_tg = args.n_samples

        elif args.mode == "endpoints_hard":
            q_start_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_start_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_start_hc",
            )
            q_goal_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_goal_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_goal_hc",
            )
            hard_conds = {0: q_start_hc_1d, H - 1: q_goal_hc_1d}
            t_g_list = [None]
            per_tg = args.n_samples

        else:
            q_start_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_start_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_start_hc",
            )
            q_goal_hc_1d = normalize_state9_to_1d(
                dataset,
                torch.as_tensor(q_goal_np, device=device, dtype=torch.float32).unsqueeze(0),
                name="q_goal_hc",
            )

            # mid grasp state from dataset preferred
            if "q_grasp_state" in data_sample:
                q_g_np = data_sample["q_grasp_state"].detach().cpu().numpy()
            else:
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
                name="q_g_hc",
            )

            if "t_grasp" in data_sample:
                t_g_from_data = int(data_sample["t_grasp"].item())
            else:
                t_g_from_data = int(args.t_g)
            t_g_from_data = int(max(1, min(H - 2, t_g_from_data)))

            if args.tg_scan:
                t_g_list = [24, 40, 56, 72, 88, 104, 120]
                t_g_list = [t for t in t_g_list if 1 <= t <= H - 2]
            else:
                t_g_list = [t_g_from_data]

            per_tg = max(1, int(np.ceil(args.n_samples / max(1, len(t_g_list)))))
            hard_conds = {}

        # =================== obstacle set (per case) ===================
        case_spheres = []
        if args.obst_enable:
            case_spheres.extend(fixed_spheres)

            if has_random:
                anchor_points = [
                    np.asarray(q_start_np[:3], dtype=np.float32),
                    np.asarray(q_goal_np[:3], dtype=np.float32),
                ]
                if args.mode == "endpoints_and_mid_hard":
                    anchor_points.append(np.asarray(q_g_np[:3], dtype=np.float32))
                rand_seed_case = int(args.seed + args.obst_random_seed_offset + idx)
                sampled = sample_random_spheres_uniform(
                    n_spheres=int(args.obst_random_n),
                    xyz_min=rand_xyz_min,
                    xyz_max=rand_xyz_max,
                    r_min=float(args.obst_random_r_min),
                    r_max=float(args.obst_random_r_max),
                    seed=rand_seed_case,
                    anchor_points=np.asarray(anchor_points, dtype=np.float32),
                    anchor_clearance=float(args.obst_random_anchor_clearance),
                    avoid_overlap=bool(args.obst_random_avoid_overlap),
                    overlap_margin=float(args.obst_random_overlap_margin),
                    max_tries=int(args.obst_random_max_tries),
                )
                case_spheres.extend(sampled)

            pc_points_case = None
            if static_pc_points is not None:
                pc_points_case = static_pc_points
            elif args.obst_pc_from_sample_key and (args.obst_pc_from_sample_key in data_sample):
                pc_raw = data_sample[args.obst_pc_from_sample_key]
                if torch.is_tensor(pc_raw):
                    pc_points_case = pc_raw.detach().cpu().numpy()
                else:
                    pc_points_case = np.asarray(pc_raw)
            if pc_points_case is not None:
                pc_spheres = spheres_from_point_cloud(
                    points=pc_points_case,
                    point_radius=float(args.obst_pc_radius),
                    max_points=int(args.obst_pc_max_points),
                    seed=int(args.seed + args.obst_pc_seed_offset + idx),
                )
                case_spheres.extend(pc_spheres)

            if idx == 0:
                print(f"[OBST_CASE] case={idx:04d} total_spheres={len(case_spheres)}")

        # =================== run inference ===================
        all_cp_norm = []
        all_tag = []

        for t_g in t_g_list:
            if args.mode == "endpoints_and_mid_hard":
                hard_conds_t = {0: q_start_hc_1d, H - 1: q_goal_hc_1d, int(t_g): q_g_hc_1d}
            else:
                hard_conds_t = hard_conds if hard_conds is not None else {}

            context_d_tg = {}
            for k, v in context_d.items():
                if torch.is_tensor(v):
                    context_d_tg[k] = _context_to_batch(v, per_tg)
                else:
                    context_d_tg[k] = v

            diffusion_kwargs = {}
            obstacle_cfg_tg = None
            if args.obst_enable:
                obstacle_cfg_tg = make_proj_only_adaptive_preset(case_spheres)
                obstacle_cfg_tg["guide"].update(
                    dict(
                        alpha=float(args.obst_alpha),
                        margin=float(args.obst_margin),
                        eps=float(args.obst_eps),
                        max_push=float(args.obst_max_push),
                        time_smooth_k=int(args.obst_time_smooth_k),
                        alpha_ramp_power=float(args.obst_alpha_ramp_power),
                        alpha_min_scale=float(args.obst_alpha_min_scale),
                        ddim_steps=int(args.obst_ddim_steps),
                        t_start_guide=int(args.obst_t_start_guide),
                        guide_lr=float(args.obst_guide_lr),
                        n_guide_steps=int(args.obst_n_guide_steps),
                        scale_grad_by_one_minus_alpha=bool(args.obst_scale_grad_by_one_minus_alpha),
                        clip_grad=True,
                        clip_grad_rule="value",
                        max_grad_value=float(args.obst_max_grad_value),
                        max_perturb_x=float(args.obst_max_perturb_x),
                        use_xrecon=bool(args.obst_use_xrecon),
                    )
                )
                proj_margin = float(args.obst_margin) if float(args.obst_proj_margin) < 0.0 else float(args.obst_proj_margin)
                obstacle_cfg_tg["project"].update(
                    dict(
                        enable=bool(args.obst_project_enable),
                        iters=int(args.obst_proj_iters),
                        lr=float(args.obst_proj_lr),
                        w_data=float(args.obst_proj_w_data),
                        w_a=float(args.obst_proj_w_a),
                        w_j=float(args.obst_proj_w_j),
                        w_obst=float(args.obst_proj_w_obst),
                        margin=proj_margin,
                        max_grad_value=float(args.obst_proj_max_grad_value),
                        max_delta=float(args.obst_proj_max_delta),
                        adaptive_enable=bool(args.obst_proj_adaptive_enable),
                        adaptive_rounds=int(args.obst_proj_adaptive_rounds),
                        adaptive_clearance=float(args.obst_proj_adaptive_clearance),
                        adaptive_iters_mult=float(args.obst_proj_adaptive_iters_mult),
                        adaptive_w_obst_mult=float(args.obst_proj_adaptive_w_obst_mult),
                        adaptive_max_delta_mult=float(args.obst_proj_adaptive_max_delta_mult),
                        adaptive_max_grad_mult=float(args.obst_proj_adaptive_max_grad_mult),
                    )
                )

            cp_norm_tg = model.run_inference(
                context_d=context_d_tg,
                hard_conds=hard_conds_t,
                n_samples=per_tg,
                horizon=H,
                obstacle_cfg=obstacle_cfg_tg,
                **diffusion_kwargs,
            )
            all_cp_norm.append(cp_norm_tg)
            all_tag.append(torch.full((cp_norm_tg.shape[0],), -1 if t_g is None else int(t_g),
                                      device="cpu", dtype=torch.int64))

        anchor_feasible = True
        anchor_clr_min = float("inf")
        if args.obst_enable and len(case_spheres) > 0:
            anchor_pts = [np.asarray(q_start_np[:3], dtype=np.float64), np.asarray(q_goal_np[:3], dtype=np.float64)]
            if args.mode == "endpoints_and_mid_hard":
                anchor_pts.append(np.asarray(q_g_np[:3], dtype=np.float64))

            for p in anchor_pts:
                for c, r in case_spheres:
                    d = float(np.linalg.norm(p - np.asarray(c, dtype=np.float64)) - float(r))
                    if d < anchor_clr_min:
                        anchor_clr_min = d
            anchor_feasible = bool(anchor_clr_min >= 0.0)
            if not anchor_feasible:
                print(f"[OBST_INFEASIBLE] case={idx:04d} hard_anchor_clr_min={anchor_clr_min:.6f} (collision at constrained waypoint)")

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

        cp_np_raw = None
        if args.obst_enable and args.obst_project_enable and len(case_spheres) > 0:
            cp_raw = cp.detach().clone()
            S_proj = int(cp.shape[0])
            hard_mask = torch.zeros((S_proj, H), dtype=torch.bool, device=cp.device)
            hard_mask[:, 0] = True
            hard_mask[:, H - 1] = True
            tag_np = tag_tg.detach().cpu().numpy().astype(np.int64)
            for s_i, tgi in enumerate(tag_np.tolist()):
                if 0 <= int(tgi) < H:
                    hard_mask[s_i, int(tgi)] = True

            proj_margin = float(args.obst_margin) if float(args.obst_proj_margin) < 0.0 else float(args.obst_proj_margin)
            proj_cfg = dict(
                iters=int(args.obst_proj_iters),
                lr=float(args.obst_proj_lr),
                w_data=float(args.obst_proj_w_data),
                w_a=float(args.obst_proj_w_a),
                w_j=float(args.obst_proj_w_j),
                w_obst=float(args.obst_proj_w_obst),
                margin=proj_margin,
                max_grad_value=float(args.obst_proj_max_grad_value),
                max_delta=float(args.obst_proj_max_delta),
                adaptive_enable=bool(args.obst_proj_adaptive_enable),
                adaptive_rounds=int(args.obst_proj_adaptive_rounds),
                adaptive_clearance=float(args.obst_proj_adaptive_clearance),
                adaptive_iters_mult=float(args.obst_proj_adaptive_iters_mult),
                adaptive_w_obst_mult=float(args.obst_proj_adaptive_w_obst_mult),
                adaptive_max_delta_mult=float(args.obst_proj_adaptive_max_delta_mult),
                adaptive_max_grad_mult=float(args.obst_proj_adaptive_max_grad_mult),
            )
            cp, _ = apply_obstacle_projection_with_adaptive(
                traj=cp,
                hard_mask=hard_mask,
                spheres=case_spheres if len(case_spheres) > 0 else [],
                proj_cfg=proj_cfg,
                verbose_fn=print,
            )

            cp_np_raw = cp_raw.detach().cpu().numpy()

        cp_np = cp.detach().cpu().numpy()

        start_pos_gt = np.asarray(q_start_np[:3], dtype=np.float64)
        goal_pos_gt = np.asarray(q_goal_np[:3], dtype=np.float64)

        if args.dbg_eval and idx == 0:
            xyz = cp_np[:, :, :3]
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

        S = cp_np.shape[0]
        start_errs = np.zeros((S,), dtype=np.float64)
        goal_errs  = np.zeros((S,), dtype=np.float64)
        grasp_errs = np.zeros((S,), dtype=np.float64)

        uav_a_mean = np.zeros((S,), dtype=np.float64)
        uav_j_mean = np.zeros((S,), dtype=np.float64)
        yaw_a_mean = np.zeros((S,), dtype=np.float64)
        arm_a_mean = np.zeros((S,), dtype=np.float64)
        obst_clr_min = np.full((S,), np.inf, dtype=np.float64)

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
            if args.obst_enable and len(case_spheres) > 0:
                obst_clr_min[s] = min_clearance_to_spheres_traj9(cp_np[s], case_spheres)

        # Selection policy:
        # default: best by grasp error;
        # if obstacle enabled: prioritize feasible samples by clearance threshold.
        if args.obst_enable and len(case_spheres) > 0:
            clr_thr = float(args.obst_select_min_clearance)
            feasible = obst_clr_min >= clr_thr
            if np.any(feasible):
                cand = np.where(feasible)[0]
                best_idx = int(cand[int(np.argmin(grasp_errs[cand]))])
            else:
                best_idx = int(np.argmin(grasp_errs))
        else:
            best_idx = int(np.argmin(grasp_errs))
        e_start_best = float(start_errs[best_idx])
        e_goal_best  = float(goal_errs[best_idx])
        e_grasp_best = float(grasp_errs[best_idx])

        uav_a_best = float(uav_a_mean[best_idx])
        uav_j_best = float(uav_j_mean[best_idx])
        yaw_a_best = float(yaw_a_mean[best_idx])
        arm_a_best = float(arm_a_mean[best_idx])
        obst_clr_best = float(obst_clr_min[best_idx]) if (args.obst_enable and len(case_spheres) > 0) else float("inf")

        all3 = (start_errs < args.succ_thresh_m) & (goal_errs < args.succ_thresh_m) & (grasp_errs < args.succ_thresh_m)
        succ_all3 = float(all3.mean())

        if args.dbg_smooth and idx == 0:
            show_ids = [0, min(1, S - 1), best_idx]
            show_ids = list(dict.fromkeys(show_ids))
            for sid in show_ids:
                sm = compute_smooth_metrics_traj9(cp_np[sid])
                print(f"[DBG_SMOOTH] sample{sid:02d} "
                      f"uav_a_mean={sm['uav_a_mean']:.6f} uav_j_mean={sm['uav_j_mean']:.6f} "
                      f"yaw_a_mean={sm['yaw_a_mean']:.6f} arm_a_mean={sm['arm_a_mean']:.6f}")

        msg = (
            f"[CASE {idx:04d}] "
            f"e_start_uav={e_start_best:.4f} m | "
            f"e_goal_uav={e_goal_best:.4f} m | "
            f"e_grasp_ee_min={e_grasp_best:.4f} m | "
            f"smooth(uav_a={uav_a_best:.6f}, uav_j={uav_j_best:.6f}, yaw_a={yaw_a_best:.6f}, arm_a={arm_a_best:.6f})"
        )
        if args.obst_enable and len(case_spheres) > 0:
            msg += f" | obst_clr_min={obst_clr_best:.4f} m"
        msg += f" | succ@2cm(all3)={succ_all3:.3f}"
        print(msg)

        best_start.append(e_start_best)
        best_goal.append(e_goal_best)
        best_grasp.append(e_grasp_best)
        best_succ_all3.append(succ_all3)

        best_uav_a.append(uav_a_best)
        best_uav_j.append(uav_j_best)
        best_yaw_a.append(yaw_a_best)
        best_arm_a.append(arm_a_best)
        if args.obst_enable and len(case_spheres) > 0:
            best_obst_clr.append(obst_clr_best)
            best_anchor_feasible.append(int(anchor_feasible))

        out_npz = os.path.join(args.save_dir, f"case_{idx:04d}_samples.npz")
        save_d = dict(
            cp=cp_np,
            q_start=data_cpu["q_start"].numpy(),
            q_goal=data_cpu["q_goal"].numpy(),
            q_grasp=data_cpu["q_grasp"].numpy(),
            best_idx=np.array([best_idx], dtype=np.int64),
            start_errs=start_errs.astype(np.float32),
            goal_errs=goal_errs.astype(np.float32),
            grasp_errs=grasp_errs.astype(np.float32),
            succ_all3=np.array([succ_all3], dtype=np.float32),
            uav_a_mean=uav_a_mean.astype(np.float32),
            uav_j_mean=uav_j_mean.astype(np.float32),
            yaw_a_mean=yaw_a_mean.astype(np.float32),
            arm_a_mean=arm_a_mean.astype(np.float32),
            obst_clr_min=obst_clr_min.astype(np.float32),
            mode=np.array([args.mode]),
            ctx_mode=np.array([args.ctx_mode]),
            seed=np.array([args.seed], dtype=np.int64),
            no_wrap_patch=np.array([int(args.no_wrap_patch)], dtype=np.int64),
            t_g_tag=tag_tg.numpy(),
            obst_enable=np.array([int(args.obst_enable)], dtype=np.int64),
            obst_alpha=np.array([float(args.obst_alpha)], dtype=np.float32),
            obst_margin=np.array([float(args.obst_margin)], dtype=np.float32),
            obst_spheres=np.array(
                [[float(c[0]), float(c[1]), float(c[2]), float(r)] for c, r in case_spheres],
                dtype=np.float32,
            ).reshape(-1, 4) if args.obst_enable else np.zeros((0, 4), dtype=np.float32),
            obst_project_enable=np.array([int(args.obst_project_enable)], dtype=np.int64),
            obst_proj_iters=np.array([int(args.obst_proj_iters)], dtype=np.int64),
            obst_proj_lr=np.array([float(args.obst_proj_lr)], dtype=np.float32),
            obst_proj_w_data=np.array([float(args.obst_proj_w_data)], dtype=np.float32),
            obst_proj_w_a=np.array([float(args.obst_proj_w_a)], dtype=np.float32),
            obst_proj_w_j=np.array([float(args.obst_proj_w_j)], dtype=np.float32),
            obst_proj_w_obst=np.array([float(args.obst_proj_w_obst)], dtype=np.float32),
        )
        if cp_np_raw is not None:
            save_d["cp_raw"] = cp_np_raw
        np.savez(out_npz, **save_d)

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
        if args.obst_enable and len(best_obst_clr) > 0:
            so = _stats_np(best_obst_clr)
            print("\n[SUMMARY] (obstacle clearance of best_by_grasp; meters; >0 is collision-free)")
            print(f"[OBST] clr_min(m): mean={so['mean']:.6f}, p50={so['p50']:.6f}, p90={so['p90']:.6f}, min={so['min']:.6f}")
            n_ok = int(np.sum(np.asarray(best_anchor_feasible, dtype=np.int64)))
            n_all = int(len(best_anchor_feasible))
            print(f"[OBST] hard_anchor_feasible_cases: {n_ok}/{n_all}")
            if n_ok > 0:
                bo = np.asarray(best_obst_clr, dtype=np.float64)
                mk = np.asarray(best_anchor_feasible, dtype=np.int64) > 0
                so_ok = _stats_np(bo[mk])
                print(
                    f"[OBST] clr_min_feasible_only(m): mean={so_ok['mean']:.6f}, "
                    f"p50={so_ok['p50']:.6f}, p90={so_ok['p90']:.6f}, min={so_ok['min']:.6f}"
                )


if __name__ == "__main__":
    main()
