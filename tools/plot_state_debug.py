#!/usr/bin/env python3
"""
Full-state debug plot for MPD trajectories saved in NPZ.

Primary use:
  - Read case_XXXX_samples.npz (eval output)
  - Pick one sample (default: -1 -> best_idx_selection / best_idx)
  - Plot position/velocity/acc/attitude/angular-rate vs time

Design goals:
  - Robust to missing keys: "have what plot what", otherwise fall back to proxies.
  - Support non-uniform time (traj_time): numeric differentiation uses np.gradient(..., t).
  - Clearly label "proxy" signals in the figure title so we don't mistake them for true states.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np


def _short_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _ensure_strictly_increasing(t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1).copy()
    if t.size == 0:
        return t
    t = t - float(t[0]) if np.isfinite(float(t[0])) else t
    t = np.maximum.accumulate(t)
    for i in range(1, int(t.size)):
        if not (t[i] > t[i - 1]):
            t[i] = t[i - 1] + float(eps)
    return t


def _quat_xyzw_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to roll/pitch/yaw (rad).
    Assumes quaternion is normalized (we normalize defensively).
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1, 4).copy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    q = q / n
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=1)


def _numeric_derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if y.size <= 1:
        return np.zeros_like(y, dtype=np.float64)
    if t.size != y.size:
        raise ValueError(f"bad derivative shapes: y={y.shape} t={t.shape}")
    t = _ensure_strictly_increasing(t)
    return np.gradient(y, t, edge_order=1)


def _numeric_derivative_vec3(y_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    y_xyz = np.asarray(y_xyz, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if y_xyz.shape[0] <= 1:
        return np.zeros_like(y_xyz, dtype=np.float64)
    if t.shape[0] != y_xyz.shape[0]:
        raise ValueError(f"bad derivative shapes: y={y_xyz.shape} t={t.shape}")
    out = np.zeros_like(y_xyz, dtype=np.float64)
    out[:, 0] = _numeric_derivative(y_xyz[:, 0], t)
    out[:, 1] = _numeric_derivative(y_xyz[:, 1], t)
    out[:, 2] = _numeric_derivative(y_xyz[:, 2], t)
    return out


def _extract_first_existing(npz, keys):
    for k in keys:
        if k in npz:
            return k
    return None


def _extract_sid_from_npz(npz, sample: int) -> int:
    if int(sample) >= 0:
        return int(sample)
    # default: best_idx_selection -> best_idx -> sid_final -> selected_sid
    for k in ["best_idx_selection", "best_idx", "sid_final", "selected_sid"]:
        if k not in npz:
            continue
        try:
            return int(np.asarray(npz[k]).reshape(-1)[0])
        except Exception:
            pass
    return 0


def _extract_traj_from_samples_npz(npz, sid: int, traj_key: str) -> np.ndarray:
    """
    Return (H,D) trajectory array from samples NPZ.
    traj_key follows make_traj_gif's naming: traj_post/traj_raw/traj_arclen.
    For samples NPZ, we map:
      traj_post -> cp[sid]
      traj_raw  -> cp_raw[sid] if present else cp[sid]
      traj_arclen -> traj_arclen[sid] if present else cp[sid]
    """
    if "cp" not in npz:
        raise KeyError("cp not found in samples npz")
    cp = np.asarray(npz["cp"], dtype=np.float64)
    sid = int(max(0, min(int(sid), int(cp.shape[0] - 1))))

    tk = str(traj_key).strip()
    if tk == "traj_raw" and ("cp_raw" in npz):
        try:
            cp_raw = np.asarray(npz["cp_raw"], dtype=np.float64)
            if cp_raw.shape[:2] == cp.shape[:2]:
                return cp_raw[sid]
        except Exception:
            pass
    if tk == "traj_arclen" and ("traj_arclen" in npz):
        try:
            tr = np.asarray(npz["traj_arclen"], dtype=np.float64)
            if tr.ndim == 3 and tr.shape[0] == cp.shape[0]:
                return tr[sid]
        except Exception:
            pass
    # Default / fallback: post-proj cp
    return cp[sid]


def _extract_time(npz, sid: int, H: int, time_key: str, dt_fallback: float) -> Tuple[np.ndarray, str]:
    tk = str(time_key).strip()
    if tk and tk in npz:
        try:
            tt = np.asarray(npz[tk], dtype=np.float64)
            if tt.ndim == 1 and tt.shape[0] == int(H):
                return _ensure_strictly_increasing(tt), f"npz:{tk}"
            if tt.ndim == 2 and tt.shape[1] == int(H):
                sid2 = int(max(0, min(int(sid), int(tt.shape[0] - 1))))
                return _ensure_strictly_increasing(tt[sid2]), f"npz:{tk}[sid]"
        except Exception:
            pass
    dt = float(max(1e-6, dt_fallback))
    tt = dt * np.arange(int(H), dtype=np.float64)
    return _ensure_strictly_increasing(tt), f"fallback:index*dt(dt={dt:.6g})"


def _clearance_to_spheres(xyz: np.ndarray, spheres: np.ndarray, uav_radius: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    sph = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    H = int(xyz.shape[0])
    if H <= 0 or sph.shape[0] == 0:
        return np.full((H,), float("inf"), dtype=np.float64)
    out = np.full((H,), float("inf"), dtype=np.float64)
    for i in range(int(sph.shape[0])):
        c = sph[i, :3].reshape(1, 3)
        r = float(sph[i, 3])
        d = np.linalg.norm(xyz - c, axis=1) - r - float(max(0.0, uav_radius))
        out = np.minimum(out, d)
    return out


def _polyline_curvature_xyz(poly_xyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    xyz = np.asarray(poly_xyz, dtype=np.float64).reshape(-1, 3)
    H = int(xyz.shape[0])
    if H < 3:
        return np.zeros((H,), dtype=np.float64)
    p0 = xyz[:-2]
    p1 = xyz[1:-1]
    p2 = xyz[2:]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cross = np.cross((p1 - p0), (p2 - p1))
    area2 = np.linalg.norm(cross, axis=1)
    denom = a * b * c
    k = np.zeros_like(area2, dtype=np.float64)
    m = denom > float(eps)
    k[m] = 2.0 * area2[m] / denom[m]
    out = np.zeros((H,), dtype=np.float64)
    out[1:-1] = k
    out[0] = out[1]
    out[-1] = out[-2]
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_v_cap_curves(
    xyz: np.ndarray,
    t: np.ndarray,
    clr: Optional[np.ndarray],
    d_goal: Optional[np.ndarray],
    safe_margin: float,
    v_free: float,
    v_max: float,
    v_min_ratio: float,
    v_min_goal: float,
    goal_mode: str,
    goal_sigma: float,
    goal_d_stop: float,
    goal_d_full: float,
    goal_r_stop: float,
    goal_k: float,
    obs_clr0_factor: float,
    obs_clr_k_factor: float,
    a_lat_max: float,
    kappa_ref_p: float,
) -> Dict[str, np.ndarray]:
    H = int(np.asarray(xyz).reshape(-1, 3).shape[0])
    out: Dict[str, np.ndarray] = {}

    v_free = float(v_free if v_free > 0.0 else 1.0)
    v_max = float(v_max if v_max > 0.0 else v_free)
    v_min = float(max(v_free * max(0.0, v_min_ratio), 1e-6))
    v_min_goal = float(max(0.0, v_min_goal))

    # v_cap_obs
    if clr is None:
        v_cap_obs = np.full((H,), float("inf"), dtype=np.float64)
    else:
        sm = float(max(0.0, safe_margin))
        sm_base = float(max(sm, 0.05))
        clr0 = float(max(0.0, sm * float(max(0.0, obs_clr0_factor))))
        k_obs = float(max(1e-6, sm_base * float(max(1e-6, obs_clr_k_factor))))
        sig_obs = _sigmoid((np.asarray(clr, dtype=np.float64).reshape(-1) - clr0) / k_obs)
        sig_obs[~np.isfinite(sig_obs)] = 1.0
        v_cap_obs = v_min + (v_free - v_min) * sig_obs
    out["v_cap_obs"] = np.asarray(v_cap_obs, dtype=np.float64)

    # v_cap_goal
    v_cap_goal = np.full((H,), float("inf"), dtype=np.float64)
    if d_goal is not None:
        dg = np.asarray(d_goal, dtype=np.float64).reshape(-1)
        if dg.shape[0] == H:
            mode = str(goal_mode).strip().lower()
            if mode in ("none", "off", "disable"):
                v_cap_goal = np.full((H,), float("inf"), dtype=np.float64)
            elif mode in ("smoothstep", "smooth"):
                d_stop = float(max(0.0, float(goal_d_stop)))
                d_full = float(max(0.0, float(goal_d_full)))
                if (not np.isfinite(d_full)) or (d_full <= d_stop + 1e-9):
                    d_full = d_stop + float(max(1e-6, float(goal_k)))
                u = (dg - d_stop) / float(max(1e-9, (d_full - d_stop)))
                u = np.clip(u, 0.0, 1.0)
                u[~np.isfinite(u)] = 1.0
                u2 = u * u * (3.0 - 2.0 * u)
                v_cap_goal = float(v_min_goal) + (float(v_free) - float(v_min_goal)) * u2
            elif mode in ("clamp", "linear"):
                # Monotone ramp in distance:
                #   d <= d_stop -> v_min_goal
                #   d >= d_full -> v_free
                # with a linear interpolation between.
                d_stop = float(max(0.0, float(goal_r_stop)))
                d_full = d_stop + float(max(1e-6, float(goal_k)))
                w = (dg - d_stop) / float(max(1e-6, d_full - d_stop))
                w = np.clip(w, 0.0, 1.0)
                w[~np.isfinite(w)] = 1.0
                v_cap_goal = float(v_min_goal) + (float(v_free) - float(v_min_goal)) * w
            elif mode == "sigmoid":
                k_goal = float(max(1e-6, float(goal_k)))
                r_stop = float(max(0.0, float(goal_r_stop)))
                sig_goal = _sigmoid((dg - r_stop) / k_goal)
                sig_goal[~np.isfinite(sig_goal)] = 1.0
                v_cap_goal = v_min_goal + (v_free - v_min_goal) * sig_goal
            else:
                # gaussian valley around closest approach
                xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
                ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1).astype(np.float64)
                s_v = np.zeros((H,), dtype=np.float64)
                if ds.size > 0:
                    s_v[1:] = np.cumsum(ds, axis=0)
                i_star = int(np.nanargmin(dg)) if np.any(np.isfinite(dg)) else int(H // 2)
                i_star = int(max(0, min(H - 1, i_star)))
                sig_m = float(max(1e-6, float(goal_sigma)))
                s0 = float(s_v[i_star])
                w = np.exp(-0.5 * ((s_v - s0) / sig_m) ** 2)
                w = np.clip(w, 0.0, 1.0)
                v_cap_goal = v_free * (1.0 - w) + v_min_goal * w
    out["v_cap_goal"] = np.asarray(v_cap_goal, dtype=np.float64)

    # v_cap_curv
    kappa = _polyline_curvature_xyz(xyz)
    if float(a_lat_max) <= 0.0:
        kk = np.asarray(kappa, dtype=np.float64).reshape(-1)
        kk = kk[np.isfinite(kk)]
        kk = kk[kk > 1e-12]
        if kk.size > 0:
            k_ref = float(np.percentile(kk, float(kappa_ref_p)))
            if np.isfinite(k_ref) and k_ref > 1e-12:
                a_lat_max = float((float(v_free) ** 2) * k_ref)
            else:
                a_lat_max = float("inf")
        else:
            a_lat_max = float("inf")
    k_eps = 1e-9
    if np.isfinite(float(a_lat_max)):
        v_cap_curv = np.sqrt(float(a_lat_max) / np.maximum(kappa, k_eps))
    else:
        v_cap_curv = np.full((H,), float("inf"), dtype=np.float64)
    out["v_cap_curv"] = np.asarray(v_cap_curv, dtype=np.float64)

    v_cap = np.minimum(v_max, np.minimum(np.minimum(v_cap_obs, v_cap_goal), v_cap_curv))
    out["v_cap"] = np.asarray(v_cap, dtype=np.float64)
    out["kappa"] = np.asarray(kappa, dtype=np.float64)
    return out


def save_state_debug_png(
    out_png: str,
    npz=None,
    traj: Optional[np.ndarray] = None,
    sid: int = 0,
    traj_key: str = "traj_post",
    time_key: str = "traj_time",
    dt_fallback: float = 1.0,
    plot_debug: bool = False,
    ee_xyz: Optional[np.ndarray] = None,
    ee_urdf: str = "/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf",
    ee_frame: str = "gripper_link",
    layout: str = "brief",
    plot_caps_all: bool = False,
    v_plot_max: float = 5.0,
) -> str:
    import matplotlib
    import matplotlib.pyplot as plt

    # Prefer a CJK-capable font if available.
    try:
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    if npz is None:
        raise ValueError("npz must be provided")

    keys = list(getattr(npz, "files", list(getattr(npz, "keys", lambda: [])())))
    if plot_debug:
        print("[STATE_KEYS]", keys)

    if traj is None:
        # assume samples npz
        traj = _extract_traj_from_samples_npz(npz, sid=int(sid), traj_key=str(traj_key))

    traj = np.asarray(traj, dtype=np.float64).reshape(int(traj.shape[0]), -1)
    H = int(traj.shape[0])
    if H <= 0:
        raise ValueError("empty trajectory")

    xyz = traj[:, :3].copy()

    # Time
    t, t_src = _extract_time(npz, sid=int(sid), H=H, time_key=str(time_key), dt_fallback=float(dt_fallback))
    t = _ensure_strictly_increasing(t)

    # Exec setpoints (optional)
    exec_xyz = None
    exec_t = None
    exec_src = ""
    if "traj_exec_xyz" in npz and "traj_exec_time" in npz:
        try:
            ex_all = np.asarray(npz["traj_exec_xyz"], dtype=np.float64)
            et_all = np.asarray(npz["traj_exec_time"], dtype=np.float64)
            if ex_all.ndim == 3 and et_all.ndim == 2:
                ex = ex_all[int(sid)]
                et = et_all[int(sid)]
                ex = np.asarray(ex, dtype=np.float64).reshape(-1, 3)
                et = np.asarray(et, dtype=np.float64).reshape(-1)
                if "traj_exec_len" in npz:
                    n_ex = int(np.asarray(npz["traj_exec_len"], dtype=np.int64).reshape(-1)[int(sid)])
                    n_ex = int(max(0, min(int(ex.shape[0]), n_ex)))
                    if n_ex > 0:
                        ex = ex[:n_ex]
                        et = et[:n_ex]
                exec_xyz = ex
                exec_t = _ensure_strictly_increasing(et)
                exec_src = "npz:traj_exec_*"
        except Exception as e:
            if plot_debug:
                print(f"[STATE_EXEC] missing/bad exec arrays: {_short_exc(e)}")
            exec_xyz = None
            exec_t = None

    # Orientation (prefer keys, else infer from traj columns)
    yaw_src = "none"
    rpy = None
    quat = None

    # (1) explicit yaw keys
    yaw_key = _extract_first_existing(npz, ["traj_yaw", "yaw", "psi"])
    if yaw_key is not None:
        try:
            yy = np.asarray(npz[yaw_key], dtype=np.float64)
            if yy.ndim == 1 and yy.shape[0] == H:
                yaw = yy
            elif yy.ndim == 2 and yy.shape[1] == H:
                yaw = yy[int(sid)]
            else:
                yaw = None
            if yaw is not None:
                yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
                yaw_src = f"npz:{yaw_key}"
        except Exception:
            yaw = None
    else:
        yaw = None

    # (2) explicit rpy keys
    if yaw is None:
        rpy_key = _extract_first_existing(npz, ["traj_rpy", "rpy"])
        if rpy_key is not None:
            try:
                rr = np.asarray(npz[rpy_key], dtype=np.float64)
                if rr.ndim == 2 and rr.shape[0] == H and rr.shape[1] >= 3:
                    rpy = rr[:, :3]
                elif rr.ndim == 3 and rr.shape[1] == H and rr.shape[2] >= 3:
                    rpy = rr[int(sid), :, :3]
                if rpy is not None:
                    yaw = np.asarray(rpy[:, 2], dtype=np.float64).reshape(-1)
                    yaw_src = f"npz:{rpy_key}"
            except Exception:
                rpy = None
                yaw = None

    # (3) explicit quat keys
    if yaw is None:
        quat_key = _extract_first_existing(npz, ["traj_quat", "uav_quat", "quat"])
        if quat_key is not None:
            try:
                qq = np.asarray(npz[quat_key], dtype=np.float64)
                if qq.ndim == 2 and qq.shape[0] == H and qq.shape[1] >= 4:
                    quat = qq[:, :4]
                elif qq.ndim == 3 and qq.shape[1] == H and qq.shape[2] >= 4:
                    quat = qq[int(sid), :, :4]
                if quat is not None:
                    rpy = _quat_xyzw_to_rpy(quat)
                    yaw = np.asarray(rpy[:, 2], dtype=np.float64).reshape(-1)
                    yaw_src = f"npz:{quat_key}"
            except Exception:
                quat = None
                rpy = None
                yaw = None

    # (4) traj columns for quat
    if yaw is None and traj.shape[1] >= 7:
        quat = traj[:, 3:7].copy()
        rpy = _quat_xyzw_to_rpy(quat)
        yaw = np.asarray(rpy[:, 2], dtype=np.float64).reshape(-1)
        yaw_src = "traj[:,3:7] quat->rpy"

    # Velocity / acc from xyz(t) (non-uniform safe)
    v_xyz = _numeric_derivative_vec3(xyz, t)
    a_xyz = _numeric_derivative_vec3(v_xyz, t)
    speed = np.linalg.norm(v_xyz, axis=1)
    acc_mag = np.linalg.norm(a_xyz, axis=1)

    # If yaw missing, use proxy from velocity heading.
    yaw_proxy = np.arctan2(v_xyz[:, 1], v_xyz[:, 0])
    if yaw is None:
        yaw = yaw_proxy.copy()
        yaw_src = "proxy:atan2(vy,vx)"
    yaw_unwrap = np.unwrap(np.asarray(yaw, dtype=np.float64).reshape(-1))
    yaw_rate = _numeric_derivative(yaw_unwrap, t)

    # Angular velocity (prefer keys, else yaw_rate proxy)
    omega_src = "proxy:yaw_rate"
    omega = None
    omega_key = _extract_first_existing(npz, ["traj_omega", "omega", "ang_vel"])
    if omega_key is not None:
        try:
            oo = np.asarray(npz[omega_key], dtype=np.float64)
            if oo.ndim == 2 and oo.shape[0] == H and oo.shape[1] >= 3:
                omega = oo[:, :3]
            elif oo.ndim == 3 and oo.shape[1] == H and oo.shape[2] >= 3:
                omega = oo[int(sid), :, :3]
            if omega is not None:
                omega = np.asarray(omega, dtype=np.float64).reshape(H, 3)
                omega_src = f"npz:{omega_key}"
        except Exception:
            omega = None
    if omega is None:
        omega = np.stack([np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate], axis=1)

    # EE/grasp distance (optional): prefer provided ee_xyz, else try pinocchio.
    ee_src = "none"
    ee_xyz_use = None
    if ee_xyz is not None:
        ee_xyz_use = np.asarray(ee_xyz, dtype=np.float64).reshape(H, 3)
        ee_src = "caller"
    else:
        # try to reuse common NPZ EE keys first
        ee_key = _extract_first_existing(npz, ["ee_positions_resampled", "ee_positions", "ee_traj", "ee_xyz", "traj_ee", "ee_path", "ee_pos", "xyz_ee"])
        if ee_key is not None:
            try:
                ee = np.asarray(npz[ee_key], dtype=np.float64)
                if ee.ndim == 2 and ee.shape[0] == H and ee.shape[1] >= 3:
                    ee_xyz_use = ee[:, :3]
                elif ee.ndim == 3 and ee.shape[1] == H and ee.shape[2] >= 3:
                    ee_xyz_use = ee[int(sid), :, :3]
                if ee_xyz_use is not None:
                    ee_xyz_use = np.asarray(ee_xyz_use, dtype=np.float64).reshape(H, 3)
                    ee_src = f"npz:{ee_key}"
            except Exception:
                ee_xyz_use = None
        if ee_xyz_use is None and traj.shape[1] >= 9:
            try:
                import pinocchio as pin  # type: ignore

                if not os.path.exists(str(ee_urdf)):
                    raise FileNotFoundError(f"URDF not found: {ee_urdf}")
                model_fk = pin.buildModelFromUrdf(str(ee_urdf), pin.JointModelFreeFlyer())
                data_fk = model_fk.createData()
                fid = model_fk.getFrameId(str(ee_frame))

                def _quat_norm_xyzw(q):
                    q = np.asarray(q, dtype=np.float64).reshape(4)
                    n = float(np.linalg.norm(q))
                    if (not np.isfinite(n)) or n < 1e-12:
                        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    return q / n

                ee_list = []
                q_all = np.asarray(traj[:, :9], dtype=np.float64).reshape(H, 9)
                for i in range(H):
                    q9 = q_all[i].copy()
                    q9[3:7] = _quat_norm_xyzw(q9[3:7])
                    pin.forwardKinematics(model_fk, data_fk, q9)
                    pin.updateFramePlacements(model_fk, data_fk)
                    ee_list.append(np.asarray(data_fk.oMf[fid].translation, dtype=np.float64).copy())
                ee_xyz_use = np.stack(ee_list, axis=0).astype(np.float64)
                ee_src = f"pinocchio:{ee_frame}"
            except Exception as e:
                if plot_debug:
                    print(f"[STATE_EE] pinocchio FK failed: {_short_exc(e)}")
                ee_xyz_use = None

    d_grasp = None
    grasp_src = "none"
    if ("q_grasp" in npz) and (ee_xyz_use is not None):
        try:
            gg = np.asarray(npz["q_grasp"], dtype=np.float64).reshape(-1)
            if gg.size >= 3 and np.all(np.isfinite(gg[:3])):
                grasp_xyz = gg[:3].copy()
                d_grasp = np.linalg.norm(ee_xyz_use - grasp_xyz[None, :], axis=1)
                grasp_src = "npz:q_grasp + ee_xyz"
        except Exception:
            d_grasp = None

    # clearance (optional)
    clr = None
    clr_src = "none"
    if "obst_spheres" in npz:
        try:
            spheres = np.asarray(npz["obst_spheres"], dtype=np.float64).reshape(-1, 4)
            uav_r = 0.0
            if "obst_uav_radius" in npz:
                try:
                    uav_r = float(np.asarray(npz["obst_uav_radius"], dtype=np.float64).reshape(-1)[0])
                except Exception:
                    uav_r = 0.0
            clr = _clearance_to_spheres(xyz, spheres, uav_radius=uav_r)
            clr_src = f"npz:obst_spheres uav_r={uav_r:.3g}"
        except Exception:
            clr = None

    # v_cap curves (optional) if traj_time meta exists
    vcap = None
    try:
        safe_margin = float(np.asarray(npz["obst_safe_margin"], dtype=np.float64).reshape(-1)[0]) if "obst_safe_margin" in npz else 0.05
        v_free = float(np.asarray(npz["traj_time_v_free"], dtype=np.float64).reshape(-1)[0]) if "traj_time_v_free" in npz else 1.0
        v_max = float(np.asarray(npz["traj_time_v_max"], dtype=np.float64).reshape(-1)[0]) if "traj_time_v_max" in npz else -1.0
        v_min_ratio = float(np.asarray(npz["traj_time_v_min_ratio"], dtype=np.float64).reshape(-1)[0]) if "traj_time_v_min_ratio" in npz else 0.05
        if "traj_time_goal_v_min" in npz:
            v_min_goal = float(np.asarray(npz["traj_time_goal_v_min"], dtype=np.float64).reshape(-1)[0])
        elif "traj_time_v_min_goal" in npz:
            v_min_goal = float(np.asarray(npz["traj_time_v_min_goal"], dtype=np.float64).reshape(-1)[0])
        else:
            v_min_goal = 0.10
        goal_mode = str(np.asarray(npz["traj_time_goal_mode"]).reshape(-1)[0]) if "traj_time_goal_mode" in npz else "smoothstep"
        goal_sigma = float(np.asarray(npz["traj_time_goal_sigma"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_sigma" in npz else 0.25
        # smoothstep params (fall back to legacy clamp params if missing)
        goal_d_stop = float(np.asarray(npz["traj_time_goal_d_stop"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_d_stop" in npz else (
            float(np.asarray(npz["traj_time_goal_r_stop"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_r_stop" in npz else 0.08
        )
        goal_r_stop = float(np.asarray(npz["traj_time_goal_r_stop"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_r_stop" in npz else 0.20
        goal_k = float(np.asarray(npz["traj_time_goal_k"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_k" in npz else 0.05
        goal_d_full = float(np.asarray(npz["traj_time_goal_d_full"], dtype=np.float64).reshape(-1)[0]) if "traj_time_goal_d_full" in npz else (goal_d_stop + goal_k)
        obs_clr0_factor = float(np.asarray(npz["traj_time_obs_clr0_factor"], dtype=np.float64).reshape(-1)[0]) if "traj_time_obs_clr0_factor" in npz else 2.0
        obs_clr_k_factor = float(np.asarray(npz["traj_time_obs_clr_k_factor"], dtype=np.float64).reshape(-1)[0]) if "traj_time_obs_clr_k_factor" in npz else 0.5
        a_lat_max = float(np.asarray(npz["traj_time_a_lat_max"], dtype=np.float64).reshape(-1)[0]) if "traj_time_a_lat_max" in npz else -1.0
        kappa_ref_p = float(np.asarray(npz["traj_time_kappa_ref_p"], dtype=np.float64).reshape(-1)[0]) if "traj_time_kappa_ref_p" in npz else 95.0
        if d_grasp is None:
            # still compute cap curves without goal term
            pass
        vcap = _compute_v_cap_curves(
            xyz=xyz,
            t=t,
            clr=clr,
            d_goal=d_grasp,
            safe_margin=safe_margin,
            v_free=v_free,
            v_max=v_max,
            v_min_ratio=v_min_ratio,
            v_min_goal=v_min_goal,
            goal_mode=goal_mode,
            goal_sigma=goal_sigma,
            goal_d_stop=goal_d_stop,
            goal_d_full=goal_d_full,
            goal_r_stop=goal_r_stop,
            goal_k=goal_k,
            obs_clr0_factor=obs_clr0_factor,
            obs_clr_k_factor=obs_clr_k_factor,
            a_lat_max=a_lat_max,
            kappa_ref_p=kappa_ref_p,
        )
    except Exception:
        vcap = None

    if plot_debug:
        print(
            "[STATE_META] "
            f"sid={int(sid)} H={H} "
            f"time={t_src} yaw={yaw_src} omega={omega_src} "
            f"ee={ee_src} grasp={grasp_src} clearance={clr_src} exec={exec_src or 'none'}"
        )

    # Event markers: closest grasp and minimum clearance.
    t_grasp = None
    if d_grasp is not None:
        try:
            dg = np.asarray(d_grasp, dtype=np.float64).reshape(-1)
            if dg.shape[0] == H and np.any(np.isfinite(dg)):
                i = int(np.nanargmin(dg))
                i = int(max(0, min(H - 1, i)))
                t_grasp = float(t[i])
        except Exception:
            t_grasp = None

    t_clear = None
    if clr is not None:
        try:
            cc = np.asarray(clr, dtype=np.float64).reshape(-1)
            if cc.shape[0] == H and np.any(np.isfinite(cc)):
                i = int(np.nanargmin(cc))
                i = int(max(0, min(H - 1, i)))
                t_clear = float(t[i])
        except Exception:
            t_clear = None

    if plot_debug:
        print(f"[STATE_EVENTS] t_grasp={t_grasp if t_grasp is not None else 'none'} t_clear={t_clear if t_clear is not None else 'none'}")

    # ----------------------------
    # Plot layout (brief / planned16)
    # ----------------------------
    layout_norm = str(layout).strip().lower()
    if layout_norm in ("planned16", "planned_16", "planned"):
        layout_norm = "planned16"
    else:
        layout_norm = "brief"

    LEGEND_FRAMEALPHA = 0.85

    def _legend(ax, fontsize: int = 8, ncol: int = 1, handles=None, labels=None):
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        keep = [(h, l) for (h, l) in zip(handles, labels) if (l is not None and str(l) and str(l) != "_nolegend_")]
        if len(keep) == 0:
            return
        hh, ll = zip(*keep)
        ax.legend(hh, ll, loc="upper right", fontsize=fontsize, framealpha=LEGEND_FRAMEALPHA, ncol=int(max(1, ncol)))

    def _na(ax, title: str, msg: str):
        ax.text(0.5, 0.5, str(msg), ha="center", va="center", transform=ax.transAxes)
        ax.set_title(str(title))
        ax.grid(alpha=0.22)

    def _add_event_lines(ax, label: bool = False):
        if t_grasp is not None:
            ax.axvline(
                t_grasp,
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=("t_grasp (min dist_to_grasp)" if label else "_nolegend_"),
            )
        if t_clear is not None:
            ax.axvline(
                t_clear,
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                label=("t_clear (min clearance)" if label else "_nolegend_"),
            )

    def _extract_npz_series(key: str, H_req: int, sid_req: int):
        if key not in npz:
            return None
        try:
            a = np.asarray(npz[key], dtype=np.float64)
            if H_req > 0:
                if a.ndim == 1 and a.shape[0] == int(H_req):
                    return a
                if a.ndim == 2 and a.shape[0] == int(H_req):
                    return a
                if a.ndim == 2 and a.shape[1] == int(H_req):
                    return a[int(sid_req)]
                if a.ndim == 3 and a.shape[1] == int(H_req):
                    return a[int(sid_req)]
            else:
                # Non-time-series (e.g., per-iter logs)
                if a.ndim == 1:
                    return a
        except Exception:
            return None
        return None

    # Arm joints and jdot (if present)
    arm_q = None
    arm_qdot = None
    if traj.shape[1] >= 9:
        arm_q = np.asarray(traj[:, 7:9], dtype=np.float64)
        arm_qdot = np.zeros_like(arm_q, dtype=np.float64)
        arm_qdot[:, 0] = _numeric_derivative(arm_q[:, 0], t)
        arm_qdot[:, 1] = _numeric_derivative(arm_q[:, 1], t)

    # EE velocity (if EE exists)
    ee_v_xyz = None
    if ee_xyz_use is not None:
        try:
            ee_v_xyz = _numeric_derivative_vec3(ee_xyz_use, t)
        except Exception:
            ee_v_xyz = None

    # EE orientation / omega (prefer NPZ keys; else proxies)
    ee_rpy = None
    ee_omega = None
    ee_att_src = "none"
    ee_omega_src = "none"
    if ee_xyz_use is not None:
        ee_yaw = None
        ee_yaw_key = _extract_first_existing(npz, ["ee_yaw", "yaw_ee", "ee_psi"])
        if ee_yaw_key is not None:
            yy = _extract_npz_series(ee_yaw_key, H_req=H, sid_req=int(sid))
            if yy is not None:
                ee_yaw = np.asarray(yy, dtype=np.float64).reshape(-1)
                ee_att_src = f"npz:{ee_yaw_key}"

        if ee_yaw is None:
            ee_rpy_key = _extract_first_existing(npz, ["ee_rpy", "rpy_ee"])
            if ee_rpy_key is not None:
                rr = _extract_npz_series(ee_rpy_key, H_req=H, sid_req=int(sid))
                if rr is not None:
                    rr = np.asarray(rr, dtype=np.float64)
                    if rr.ndim == 2 and rr.shape[1] >= 3:
                        ee_rpy = rr[:, :3]
                        ee_yaw = ee_rpy[:, 2]
                        ee_att_src = f"npz:{ee_rpy_key}"

        if ee_yaw is None:
            ee_quat_key = _extract_first_existing(npz, ["ee_quat", "quat_ee"])
            if ee_quat_key is not None:
                qq = _extract_npz_series(ee_quat_key, H_req=H, sid_req=int(sid))
                if qq is not None:
                    qq = np.asarray(qq, dtype=np.float64)
                    if qq.ndim == 2 and qq.shape[1] >= 4:
                        ee_rpy = _quat_xyzw_to_rpy(qq[:, :4])
                        ee_yaw = ee_rpy[:, 2]
                        ee_att_src = f"npz:{ee_quat_key}"

        if ee_yaw is None and ee_v_xyz is not None:
            ee_yaw = np.arctan2(ee_v_xyz[:, 1], ee_v_xyz[:, 0])
            ee_att_src = "proxy:atan2(vy,vx)"
        if ee_rpy is None and ee_yaw is not None:
            ee_rpy = np.stack([np.zeros_like(ee_yaw), np.zeros_like(ee_yaw), np.asarray(ee_yaw, dtype=np.float64)], axis=1)

        ee_omega_key = _extract_first_existing(npz, ["ee_omega", "omega_ee", "ee_ang_vel", "ee_angvel"])
        if ee_omega_key is not None:
            oo = _extract_npz_series(ee_omega_key, H_req=H, sid_req=int(sid))
            if oo is not None:
                oo = np.asarray(oo, dtype=np.float64)
                if oo.ndim == 2 and oo.shape[1] >= 3:
                    ee_omega = oo[:, :3]
                    ee_omega_src = f"npz:{ee_omega_key}"
        if ee_omega is None and ee_yaw is not None:
            ee_yaw_unwrap = np.unwrap(np.asarray(ee_yaw, dtype=np.float64).reshape(-1))
            ee_yaw_rate = _numeric_derivative(ee_yaw_unwrap, t)
            ee_omega = np.stack([np.zeros_like(ee_yaw_rate), np.zeros_like(ee_yaw_rate), ee_yaw_rate], axis=1)
            ee_omega_src = "proxy:ee_yaw_rate"

    def _plot_xy_xz(ax_xy, ax_xz):
        # XY
        ax_xy.plot(xyz[:, 0], xyz[:, 1], label="base XY", linewidth=1.8, alpha=0.95)
        if ee_xyz_use is not None:
            ax_xy.plot(ee_xyz_use[:, 0], ee_xyz_use[:, 1], label="ee XY", linewidth=1.6, alpha=0.90)
        ax_xy.scatter([xyz[0, 0]], [xyz[0, 1]], s=22, marker="o", color="k", alpha=0.7, label="_nolegend_")
        ax_xy.scatter([xyz[-1, 0]], [xyz[-1, 1]], s=28, marker="x", color="k", alpha=0.7, label="_nolegend_")
        if "q_grasp" in npz:
            try:
                gg = np.asarray(npz["q_grasp"], dtype=np.float64).reshape(-1)
                if gg.size >= 2 and np.all(np.isfinite(gg[:2])):
                    ax_xy.scatter([gg[0]], [gg[1]], s=60, marker="*", color="tab:red", alpha=0.9, label="grasp")
            except Exception:
                pass
        # Keep subplot box size fixed; preserve aspect ratio by expanding data limits (not shrinking the axes box).
        ax_xy.relim()
        ax_xy.autoscale()
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title("Horizontal Trajectory (XY)")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.grid(alpha=0.22)
        _legend(ax_xy, fontsize=8)

        # XZ
        ax_xz.plot(xyz[:, 0], xyz[:, 2], label="base XZ", linewidth=1.8, alpha=0.95)
        if ee_xyz_use is not None:
            ax_xz.plot(ee_xyz_use[:, 0], ee_xyz_use[:, 2], label="ee XZ", linewidth=1.6, alpha=0.90)
        ax_xz.scatter([xyz[0, 0]], [xyz[0, 2]], s=22, marker="o", color="k", alpha=0.7, label="_nolegend_")
        ax_xz.scatter([xyz[-1, 0]], [xyz[-1, 2]], s=28, marker="x", color="k", alpha=0.7, label="_nolegend_")
        if "q_grasp" in npz:
            try:
                gg = np.asarray(npz["q_grasp"], dtype=np.float64).reshape(-1)
                if gg.size >= 3 and np.all(np.isfinite(gg[:3])):
                    ax_xz.scatter([gg[0]], [gg[2]], s=60, marker="*", color="tab:red", alpha=0.9, label="grasp")
            except Exception:
                pass
        # Keep subplot box size fixed; preserve aspect ratio by expanding data limits (not shrinking the axes box).
        ax_xz.relim()
        ax_xz.autoscale()
        ax_xz.set_aspect("equal", adjustable="datalim")
        ax_xz.set_title("Vertical Profile (XZ)")
        ax_xz.set_xlabel("x")
        ax_xz.set_ylabel("z")
        ax_xz.grid(alpha=0.22)
        _legend(ax_xz, fontsize=8)

    if layout_norm == "planned16":
        fig, axs = plt.subplots(4, 4, figsize=(18.4, 11.0))
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.06, hspace=0.38, wspace=0.28)
        fig.suptitle(
            f"State Debug (planned16)  sid={int(sid)}  H={H} | time={t_src} | yaw={yaw_src} | omega={omega_src}",
            fontsize=11,
        )

        # Row 1: base pos/vel/rpy/omega
        ax = axs[0, 0]
        ax.plot(t, xyz[:, 0], label="x")
        ax.plot(t, xyz[:, 1], label="y")
        ax.plot(t, xyz[:, 2], label="z")
        _add_event_lines(ax, label=True)
        ax.set_title("Base Position xyz(t)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("m")

        ax = axs[0, 1]
        ax.plot(t, v_xyz[:, 0], label="vx")
        ax.plot(t, v_xyz[:, 1], label="vy")
        ax.plot(t, v_xyz[:, 2], label="vz")
        _add_event_lines(ax, label=False)
        ax.set_title("Base Velocity vxyz(t)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("m/s")

        ax = axs[0, 2]
        if rpy is not None:
            ax.plot(t, np.rad2deg(rpy[:, 0]), label="roll")
            ax.plot(t, np.rad2deg(rpy[:, 1]), label="pitch")
            ax.plot(t, np.rad2deg(np.unwrap(rpy[:, 2])), label="yaw")
            ax.set_title("Base Attitude RPY(t)")
            _legend(ax, fontsize=8, ncol=3)
        else:
            ax.plot(t, np.rad2deg(yaw_unwrap), label="yaw")
            ax.plot(t, np.rad2deg(np.unwrap(yaw_proxy)), label="yaw_proxy(v heading)", alpha=0.45)
            ax.set_title("Base Attitude Yaw(t)")
            _legend(ax, fontsize=8)
        _add_event_lines(ax, label=False)
        ax.grid(alpha=0.22)
        ax.set_ylabel("deg")

        ax = axs[0, 3]
        ax.plot(t, omega[:, 0], label="omega_x")
        ax.plot(t, omega[:, 1], label="omega_y")
        ax.plot(t, omega[:, 2], label="omega_z/yaw_rate")
        _add_event_lines(ax, label=False)
        ax.set_title("Base Angular Rate omega(t)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("rad/s")

        # Row 2: EE pos/vel/rpy/omega
        ax = axs[1, 0]
        if ee_xyz_use is not None:
            ax.plot(t, ee_xyz_use[:, 0], label="x")
            ax.plot(t, ee_xyz_use[:, 1], label="y")
            ax.plot(t, ee_xyz_use[:, 2], label="z")
            _add_event_lines(ax, label=False)
            ax.set_title(f"EE Position xyz(t) ({ee_src})")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8, ncol=3)
            ax.set_ylabel("m")
        else:
            _na(ax, "EE Position xyz(t)", "N/A (missing ee_xyz)")

        ax = axs[1, 1]
        if ee_v_xyz is not None:
            ax.plot(t, ee_v_xyz[:, 0], label="vx")
            ax.plot(t, ee_v_xyz[:, 1], label="vy")
            ax.plot(t, ee_v_xyz[:, 2], label="vz")
            _add_event_lines(ax, label=False)
            ax.set_title("EE Linear Velocity vxyz(t) (numeric)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8, ncol=3)
            ax.set_ylabel("m/s")
        else:
            _na(ax, "EE Linear Velocity vxyz(t)", "N/A (missing ee_xyz)")

        ax = axs[1, 2]
        if ee_rpy is not None:
            ax.plot(t, np.rad2deg(ee_rpy[:, 0]), label="roll")
            ax.plot(t, np.rad2deg(ee_rpy[:, 1]), label="pitch")
            ax.plot(t, np.rad2deg(np.unwrap(ee_rpy[:, 2])), label="yaw")
            _add_event_lines(ax, label=False)
            ax.set_title(f"EE Attitude RPY(t) ({ee_att_src})")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8, ncol=3)
            ax.set_ylabel("deg")
        else:
            _na(ax, "EE Attitude RPY(t)", "N/A (missing ee rpy/quat/yaw)")

        ax = axs[1, 3]
        if ee_omega is not None:
            ax.plot(t, ee_omega[:, 0], label="omega_x")
            ax.plot(t, ee_omega[:, 1], label="omega_y")
            ax.plot(t, ee_omega[:, 2], label="omega_z/yaw_rate")
            _add_event_lines(ax, label=False)
            ax.set_title(f"EE Angular Rate omega(t) ({ee_omega_src})")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8, ncol=3)
            ax.set_ylabel("rad/s")
        else:
            _na(ax, "EE Angular Rate omega(t)", "N/A (missing ee omega/yaw)")

        # Row 3: joints / jdot / controls
        ax = axs[2, 0]
        if arm_q is not None:
            ax.plot(t, arm_q[:, 0], label="joint1")
            ax.plot(t, arm_q[:, 1], label="joint2")
            _add_event_lines(ax, label=False)
            ax.set_title("Arm Joints q(t)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
            ax.set_ylabel("rad")
        else:
            _na(ax, "Arm Joints q(t)", "N/A (traj has no arm joints)")

        ax = axs[2, 1]
        if arm_qdot is not None:
            ax.plot(t, arm_qdot[:, 0], label="jdot1")
            ax.plot(t, arm_qdot[:, 1], label="jdot2")
            _add_event_lines(ax, label=False)
            ax.set_title("Arm Joint Velocity qdot(t) (numeric)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
            ax.set_ylabel("rad/s")
        else:
            _na(ax, "Arm Joint Velocity qdot(t)", "N/A (traj has no arm joints)")

        ax = axs[2, 2]
        thr_key = _extract_first_existing(npz, ["thrusters", "uav_thrusters", "base_thrusters", "T"])
        thr = _extract_npz_series(thr_key, H_req=H, sid_req=int(sid)) if thr_key is not None else None
        if thr is not None:
            thr = np.asarray(thr, dtype=np.float64)
            if thr.ndim == 2 and thr.shape[0] == H and thr.shape[1] >= 1:
                for j in range(int(min(8, thr.shape[1]))):
                    ax.plot(t, thr[:, j], label=f"T{j+1}")
                _add_event_lines(ax, label=False)
                ax.set_title(f"Base Control (thrusters) ({thr_key})")
                ax.grid(alpha=0.22)
                _legend(ax, fontsize=8, ncol=2)
        else:
            _na(ax, "Base Control (thrusters)", "N/A (missing key: thrusters/uav_thrusters/...)")

        ax = axs[2, 3]
        tau_key = _extract_first_existing(npz, ["arm_torque", "arm_tau", "tau", "joint_torque", "joint_tau"])
        tau = _extract_npz_series(tau_key, H_req=H, sid_req=int(sid)) if tau_key is not None else None
        if tau is not None:
            tau = np.asarray(tau, dtype=np.float64)
            if tau.ndim == 2 and tau.shape[0] == H and tau.shape[1] >= 1:
                for j in range(int(min(8, tau.shape[1]))):
                    ax.plot(t, tau[:, j], label=f"tau{j+1}")
                _add_event_lines(ax, label=False)
                ax.set_title(f"Arm Control (torque) ({tau_key})")
                ax.grid(alpha=0.22)
                _legend(ax, fontsize=8, ncol=2)
        else:
            _na(ax, "Arm Control (torque)", "N/A (missing key: arm_torque/tau/...)")

        # Row 4: XY / XZ / solver logs
        _plot_xy_xz(axs[3, 0], axs[3, 1])

        ax = axs[3, 2]
        cost_key = _extract_first_existing(npz, ["cost_iter", "crocoddyl_cost", "solver_cost", "costs"])
        cost = _extract_npz_series(cost_key, H_req=-1, sid_req=int(sid)) if cost_key is not None else None
        if cost is not None:
            cost = np.asarray(cost, dtype=np.float64).reshape(-1)
            ax.plot(np.arange(cost.size), cost, label="cost")
            ax.set_title(f"Cost vs Iter ({cost_key})")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
        else:
            _na(ax, "Cost vs Iter", "N/A (missing key: cost_iter/crocoddyl_cost/...)")

        ax = axs[3, 3]
        it_key = _extract_first_existing(npz, ["time_iter", "crocoddyl_time_iter", "solver_time_iter", "iter_time"])
        it = _extract_npz_series(it_key, H_req=-1, sid_req=int(sid)) if it_key is not None else None
        if it is not None:
            it = np.asarray(it, dtype=np.float64).reshape(-1)
            ax.plot(np.arange(it.size), it, label="time/iter")
            ax.set_title(f"Time per Iter ({it_key})")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
        else:
            _na(ax, "Time per Iter", "N/A (missing key: time_iter/...)")

    else:
        # Brief layout: 5x2 (drop the last XY/XZ row for a lighter debug view).
        fig, axs = plt.subplots(5, 2, figsize=(15.0, 13.0))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06, hspace=0.44, wspace=0.22)
        fig.suptitle(
            f"State Debug (brief)  sid={int(sid)}  H={H} | time={t_src} | yaw={yaw_src} | omega={omega_src}",
            fontsize=11,
        )

        # Row 1: position / clearance+grasp
        ax = axs[0, 0]
        ax.plot(t, xyz[:, 0], label="x")
        ax.plot(t, xyz[:, 1], label="y")
        ax.plot(t, xyz[:, 2], label="z")
        _add_event_lines(ax, label=False)
        ax.set_title("Position xyz(t)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("m")

        ax = axs[0, 1]
        if clr is not None:
            ax.plot(t, clr, label="clearance(uav)")
        if d_grasp is not None:
            ax.plot(t, d_grasp, label="dist_to_grasp(ee)")
        if clr is None and d_grasp is None:
            ax.text(0.5, 0.5, "clearance/dist_to_grasp N/A", ha="center", va="center", transform=ax.transAxes)
        _add_event_lines(ax, label=True)
        ax.set_title("Clearance / Grasp Distance")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8)
        ax.set_ylabel("m")

        # Row 2: velocity / speed&caps
        ax = axs[1, 0]
        ax.plot(t, v_xyz[:, 0], label="vx")
        ax.plot(t, v_xyz[:, 1], label="vy")
        ax.plot(t, v_xyz[:, 2], label="vz")
        _add_event_lines(ax, label=False)
        ax.set_title("Velocity vxyz(t) (numeric)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("m/s")

        ax = axs[1, 1]
        ax.plot(t, speed, label="|v| (numeric)", color="k", alpha=0.80, linewidth=1.4)
        ax2 = None
        if vcap is not None:
            ax.plot(t, np.asarray(vcap["v_cap"], dtype=float), label="v_cap", alpha=0.95, linewidth=1.6)
            ax.plot(t, np.asarray(vcap["v_cap_goal"], dtype=float), label="v_cap_goal", alpha=0.90, linewidth=1.4)
            if bool(plot_caps_all):
                ax.plot(t, np.asarray(vcap["v_cap_obs"], dtype=float), label="v_cap_obs", alpha=0.55, linewidth=1.2)
                v_plot_max_f = float(max(0.1, v_plot_max))
                v_curv = np.asarray(vcap["v_cap_curv"], dtype=float).reshape(-1)
                v_curv_plot = np.clip(v_curv, 0.0, v_plot_max_f)
                ax2 = ax.twinx()
                ax2.plot(
                    t,
                    v_curv_plot,
                    label=f"v_cap_curv (clip<{v_plot_max_f:g})",
                    color="tab:purple",
                    alpha=0.35,
                    linewidth=1.0,
                )
                ax2.set_ylabel("m/s (curv cap)")
                ax2.grid(False)
                ax2.set_ylim(0.0, v_plot_max_f)
        _add_event_lines(ax, label=False)
        ax.set_title("Speed & Caps")
        ax.grid(alpha=0.22)
        ax.set_ylabel("m/s")
        if ax2 is not None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            _legend(ax, fontsize=8, handles=(h1 + h2), labels=(l1 + l2))
        else:
            _legend(ax, fontsize=8)

        # Row 3: acceleration (components + |a|) / attitude
        ax = axs[2, 0]
        ax.plot(t, a_xyz[:, 0], label="ax")
        ax.plot(t, a_xyz[:, 1], label="ay")
        ax.plot(t, a_xyz[:, 2], label="az")
        ax.plot(t, acc_mag, label="|a|", color="k", alpha=0.55, linewidth=1.2)
        _add_event_lines(ax, label=False)
        ax.set_title("Acceleration axyz(t) (numeric)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=4)
        ax.set_ylabel("m/s^2")

        ax = axs[2, 1]
        if rpy is not None:
            ax.plot(t, np.rad2deg(rpy[:, 0]), label="roll")
            ax.plot(t, np.rad2deg(rpy[:, 1]), label="pitch")
            ax.plot(t, np.rad2deg(np.unwrap(rpy[:, 2])), label="yaw")
            ax.set_title("Attitude RPY(t)")
            _legend(ax, fontsize=8, ncol=3)
        else:
            ax.plot(t, np.rad2deg(yaw_unwrap), label="yaw")
            ax.plot(t, np.rad2deg(np.unwrap(yaw_proxy)), label="yaw_proxy(v heading)", alpha=0.45)
            ax.set_title("Attitude Yaw(t)")
            _legend(ax, fontsize=8)
        _add_event_lines(ax, label=False)
        ax.grid(alpha=0.22)
        ax.set_ylabel("deg")

        # Row 4: omega / arm joints
        ax = axs[3, 0]
        ax.plot(t, omega[:, 0], label="omega_x")
        ax.plot(t, omega[:, 1], label="omega_y")
        ax.plot(t, omega[:, 2], label="omega_z/yaw_rate")
        _add_event_lines(ax, label=False)
        ax.set_title("Angular Rate omega(t)")
        ax.grid(alpha=0.22)
        _legend(ax, fontsize=8, ncol=3)
        ax.set_ylabel("rad/s")

        ax = axs[3, 1]
        if arm_q is not None:
            ax.plot(t, arm_q[:, 0], label="joint1")
            ax.plot(t, arm_q[:, 1], label="joint2")
            _add_event_lines(ax, label=False)
            ax.set_title("Arm Joints q(t)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
            ax.set_ylabel("rad")
        else:
            _na(ax, "Arm Joints q(t)", "N/A (traj has no arm joints)")

        # Row 5: arm jdot / curvature
        ax = axs[4, 0]
        if arm_qdot is not None:
            ax.plot(t, arm_qdot[:, 0], label="jdot1")
            ax.plot(t, arm_qdot[:, 1], label="jdot2")
            _add_event_lines(ax, label=False)
            ax.set_title("Arm Joint Velocity qdot(t) (numeric)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
            ax.set_ylabel("rad/s")
        else:
            _na(ax, "Arm Joint Velocity qdot(t)", "N/A (traj has no arm joints)")
        ax.set_xlabel("time (s)")

        ax = axs[4, 1]
        if vcap is not None and "kappa" in vcap:
            ax.plot(t, np.asarray(vcap["kappa"], dtype=float), label="kappa")
            _add_event_lines(ax, label=False)
            ax.set_title("Curvature kappa(t)")
            ax.grid(alpha=0.22)
            _legend(ax, fontsize=8)
            ax.set_ylabel("1/m")
        else:
            _na(ax, "Curvature kappa(t)", "N/A (missing curvature)")
        ax.set_xlabel("time (s)")

        # (Dropped) XY / XZ projection row in brief layout.

    out_png = str(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="case_XXXX_samples.npz (contains cp).")
    ap.add_argument("--sample", type=int, default=-1, help="-1 uses best_idx_selection/best_idx, else explicit sid.")
    ap.add_argument("--traj_key", default="traj_post", choices=["traj_post", "traj_raw", "traj_arclen"],
                    help="Which trajectory to use from NPZ (samples mode: maps to cp/cp_raw).")
    ap.add_argument("--out_png", default="", help="Output PNG path (default: /tmp/<case>_sidXX_state.png).")
    ap.add_argument("--time_key", default="traj_time", help="Prefer this time key in NPZ (default traj_time).")
    ap.add_argument("--dt", type=float, default=1.0, help="Fallback dt (seconds) when time_key missing.")
    ap.add_argument("--v_plot_max", type=float, default=5.0,
                    help="Plot clip upper bound for speed caps (m/s). Used for v_cap_curv on secondary axis.")
    ap.add_argument("--plot_caps_all", action="store_true",
                    help="Show v_cap_obs and v_cap_curv in Speed & Caps (default only shows |v|, v_cap, v_cap_goal).")
    ap.add_argument("--layout", type=str, default="brief", choices=["brief", "planned16"],
                    help="Plot layout preset. 'brief' is lightweight; 'planned16' aligns to a 4x4 planned reference grid.")
    ap.add_argument("--plot_debug", action="store_true", help="Print npz keys and chosen sources.")
    ap.add_argument("--ee_urdf", default="/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf",
                    help="URDF for Pinocchio FK when EE keys missing.")
    ap.add_argument("--ee_frame", default="gripper_link", help="EE frame for Pinocchio FK.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    sid = _extract_sid_from_npz(d, int(args.sample))
    cp = np.asarray(d["cp"])
    sid = int(max(0, min(sid, int(cp.shape[0] - 1))))
    best_idx = int(np.asarray(d["best_idx"]).reshape(-1)[0]) if "best_idx" in d else -1
    best_idx_selection = int(np.asarray(d["best_idx_selection"]).reshape(-1)[0]) if "best_idx_selection" in d else best_idx
    best_idx_grasp = int(np.asarray(d["best_idx_grasp"]).reshape(-1)[0]) if "best_idx_grasp" in d else best_idx
    post_proj_applied = int(np.asarray(d["post_proj_applied"]).reshape(-1)[0]) if "post_proj_applied" in d else -1
    obst_hash = str(np.asarray(d["obst_hash"]).reshape(-1)[0]) if "obst_hash" in d else "<none>"
    print(
        "[STATE_META] "
        f"traj_key={str(args.traj_key)} "
        f"sid={int(sid)} best_idx={int(best_idx)} best_idx_selection={int(best_idx_selection)} best_idx_grasp={int(best_idx_grasp)} "
        f"post_proj_applied={int(post_proj_applied)} obst_hash={obst_hash}"
    )

    if not args.out_png:
        base = os.path.basename(str(args.npz)).replace(".npz", "")
        args.out_png = f"/tmp/{base}_sid{sid:03d}_state.png"

    traj = _extract_traj_from_samples_npz(d, sid=sid, traj_key=str(args.traj_key))
    out = save_state_debug_png(
        out_png=str(args.out_png),
        npz=d,
        traj=traj,
        sid=sid,
        traj_key=str(args.traj_key),
        time_key=str(args.time_key),
        dt_fallback=float(args.dt),
        plot_debug=bool(args.plot_debug),
        ee_xyz=None,
        ee_urdf=str(args.ee_urdf),
        ee_frame=str(args.ee_frame),
        layout=str(args.layout),
        plot_caps_all=bool(args.plot_caps_all),
        v_plot_max=float(args.v_plot_max),
    )
    print("[OK] wrote", out)


if __name__ == "__main__":
    main()
