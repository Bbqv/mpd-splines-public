#!/usr/bin/env python3
"""
Recompute traj_time/traj_speed/traj_acc (and optional traj_exec_*) for an existing
case_XXXX_samples.npz WITHOUT changing geometry (cp).

Why: GPU diffusion inference may not be available in some environments, but we still
want to validate/fix time-parameterization logic and state_debug readability.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np


def _short_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _load_eval_module():
    # Import eval_eagle_grasp in a way that works even if scripts/ isn't a package.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    import scripts.eval.eval_eagle_grasp as eeg  # type: ignore

    return eeg


def _clearance_to_spheres(xyz: np.ndarray, spheres: np.ndarray, uav_radius: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    sph = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    H = int(xyz.shape[0])
    if H <= 0 or sph.shape[0] <= 0:
        return np.full((H,), float("inf"), dtype=np.float64)
    centers = sph[:, :3]  # (N,3)
    radii = sph[:, 3]  # (N,)
    diff = xyz[:, None, :] - centers[None, :, :]  # (H,N,3)
    dist = np.linalg.norm(diff, axis=2)  # (H,N)
    clr = dist - radii[None, :] - float(max(0.0, uav_radius))
    out = np.min(clr, axis=1)
    out[~np.isfinite(out)] = float("inf")
    return out


def _build_pinocchio_fk(urdf: str, ee_frame: str):
    import pinocchio as pin  # type: ignore

    if not os.path.exists(str(urdf)):
        raise FileNotFoundError(f"URDF not found: {urdf}")
    model_fk = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data_fk = model_fk.createData()
    fid = model_fk.getFrameId(str(ee_frame))
    return pin, model_fk, data_fk, fid


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if (not np.isfinite(n)) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _compute_d_goal_ee_to_grasp_xyz(
    traj9: np.ndarray,
    grasp_xyz: np.ndarray,
    pin,
    model_fk,
    data_fk,
    fid: int,
) -> np.ndarray:
    traj9 = np.asarray(traj9, dtype=np.float64).reshape(-1, 9)
    H = int(traj9.shape[0])
    out = np.zeros((H,), dtype=np.float64)
    g = np.asarray(grasp_xyz, dtype=np.float64).reshape(3)
    for i in range(H):
        q9 = traj9[i].copy()
        q9[3:7] = _normalize_quat_xyzw(q9[3:7])
        pin.forwardKinematics(model_fk, data_fk, q9)
        pin.updateFramePlacements(model_fk, data_fk)
        ee = np.asarray(data_fk.oMf[fid].translation, dtype=np.float64).reshape(3)
        out[i] = float(np.linalg.norm(ee - g))
    out[~np.isfinite(out)] = float("inf")
    return out


def _resample_exec_uniform_dt(
    eeg,
    t: np.ndarray,
    xyz: np.ndarray,
    dt_ctrl: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reuse eval's helper if present; otherwise do minimal linear interpolation.
    if hasattr(eeg, "_resample_xyz_uniform_dt"):
        return eeg._resample_xyz_uniform_dt(t=t, xyz=xyz, dt_ctrl=dt_ctrl)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if t.size != xyz.shape[0]:
        raise ValueError(f"bad shapes for exec: t={t.shape} xyz={xyz.shape}")
    t = np.maximum.accumulate(t - float(t[0]))
    t_end = float(t[-1]) if t.size else 0.0
    dt = float(max(float(dt_ctrl), 1e-6))
    t_exec = np.arange(0.0, t_end + 1e-12, dt, dtype=np.float64)
    if t_exec.size <= 0:
        t_exec = np.array([0.0], dtype=np.float64)
    if float(t_exec[-1]) < t_end - 1e-9:
        t_exec = np.concatenate([t_exec, np.array([t_end], dtype=np.float64)], axis=0)
    else:
        t_exec[-1] = t_end
    xyz_exec = np.stack([np.interp(t_exec, t, xyz[:, k]) for k in range(3)], axis=1).astype(np.float64)
    vel_exec = np.zeros_like(xyz_exec, dtype=np.float64)
    if t_exec.size >= 2:
        dt_seg = np.maximum(np.diff(t_exec), 1e-6)
        vel_exec[:-1] = np.diff(xyz_exec, axis=0) / dt_seg[:, None]
        vel_exec[-1] = vel_exec[-2]
    return t_exec, xyz_exec, vel_exec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--plot_debug", action="store_true")

    # Match eval_eagle_grasp flags (subset)
    ap.add_argument("--traj_time_v_free", type=float, default=1.0)
    ap.add_argument("--traj_time_v_max", type=float, default=-1.0)
    ap.add_argument("--traj_time_v_min_ratio", type=float, default=0.05)
    ap.add_argument("--traj_time_a_max", type=float, default=3.0)
    ap.add_argument("--traj_time_a_lat_max", type=float, default=-1.0)
    ap.add_argument("--traj_time_kappa_ref_p", type=float, default=95.0)
    ap.add_argument("--traj_time_dt_min", type=float, default=1e-4)
    ap.add_argument("--traj_time_dt_max", type=float, default=10.0)
    ap.add_argument("--traj_time_cap_smooth_window", type=int, default=9,
                    help="Moving-average window applied to final v_cap before accel pass. <=1 disables.")
    ap.add_argument("--traj_time_j_max", type=float, default=0.0,
                    help="Optional jerk limit (m/s^3). <=0 disables.")
    ap.add_argument("--traj_time_j_iters", type=int, default=3,
                    help="Iterations for jerk limiting when --traj_time_j_max>0.")
    ap.add_argument("--traj_time_debug", action="store_true",
                    help="Print v_cap decomposition table around grasp (best_sid only).")

    ap.add_argument(
        "--traj_time_goal_mode",
        type=str,
        default="smoothstep",
        choices=["smoothstep", "clamp", "sigmoid", "valley", "none"],
    )
    ap.add_argument("--traj_time_goal_sigma", type=float, default=0.25)
    ap.add_argument("--traj_time_goal_d_stop", type=float, default=0.04)
    ap.add_argument("--traj_time_goal_d_full", type=float, default=0.60)
    ap.add_argument("--traj_time_goal_v_min", "--traj_time_v_min_goal",
                    dest="traj_time_goal_v_min", type=float, default=0.10)
    ap.add_argument("--traj_time_goal_r_stop", type=float, default=0.20)
    ap.add_argument("--traj_time_goal_k", type=float, default=0.05)

    ap.add_argument("--traj_time_obs_clr0_factor", type=float, default=2.0)
    ap.add_argument("--traj_time_obs_clr_k_factor", type=float, default=0.5)

    ap.add_argument("--traj_exec_enable", action="store_true")
    ap.add_argument("--traj_exec_dt", type=float, default=0.02)

    ap.add_argument("--ee_urdf", default="/home/yongxin/workspace/eagle-mpc-python/models/urdf/s500_uam_arm_effort.urdf")
    ap.add_argument("--ee_frame", default="gripper_link")
    args = ap.parse_args()

    eeg = _load_eval_module()

    d_in = np.load(str(args.in_npz), allow_pickle=True)
    out = {k: d_in[k] for k in d_in.files}
    if args.plot_debug:
        print("[IN_KEYS]", list(d_in.files))

    if "cp" not in out:
        raise KeyError("expected key cp in input npz")
    cp = np.asarray(out["cp"], dtype=np.float64)
    if cp.ndim != 3 or cp.shape[2] < 3:
        raise ValueError(f"bad cp shape: {cp.shape}")
    S, H, D = cp.shape
    if args.plot_debug:
        print(f"[IN] S={S} H={H} D={D}")

    spheres = np.asarray(out.get("obst_spheres", np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
    safe_margin = float(np.asarray(out.get("obst_safe_margin", np.array([0.05])), dtype=np.float64).reshape(-1)[0])
    uav_r = float(np.asarray(out.get("obst_uav_radius", np.array([0.0])), dtype=np.float64).reshape(-1)[0])

    # Optional FK for d_goal: use EE->grasp distance where grasp target is q_grasp[:3] (world xyz).
    grasp_xyz = None
    if "q_grasp" in out:
        gg = np.asarray(out["q_grasp"], dtype=np.float64).reshape(-1)
        if gg.size >= 3 and np.all(np.isfinite(gg[:3])):
            grasp_xyz = gg[:3].copy()

    fk_ctx: Optional[Tuple] = None
    if grasp_xyz is not None and D >= 9:
        try:
            fk_ctx = _build_pinocchio_fk(urdf=str(args.ee_urdf), ee_frame=str(args.ee_frame))
        except Exception as e:
            fk_ctx = None
            print("[WARN] pinocchio FK disabled:", _short_exc(e))

    traj_time = np.zeros((S, H), dtype=np.float32)
    traj_speed = np.zeros((S, H), dtype=np.float32)
    traj_acc = np.zeros((S, H), dtype=np.float32)

    # Exec arrays (ragged -> padded)
    exec_len = None
    exec_time = None
    exec_xyz = None
    exec_vel = None
    exec_list = []

    best_sid = None
    for k in ["best_idx_selection", "best_idx", "sid_final", "selected_sid"]:
        if k in out:
            try:
                best_sid = int(np.asarray(out[k]).reshape(-1)[0])
                break
            except Exception:
                pass
    if best_sid is None:
        best_sid = 0
    best_sid = int(max(0, min(S - 1, best_sid)))

    info_best = None
    clr_best = None
    d_goal_best = None

    for s in range(S):
        xyz = cp[s, :, :3]
        clr = _clearance_to_spheres(xyz, spheres, uav_radius=uav_r)
        d_goal = None
        if fk_ctx is not None and grasp_xyz is not None:
            pin, model_fk, data_fk, fid = fk_ctx
            try:
                d_goal = _compute_d_goal_ee_to_grasp_xyz(cp[s, :, :9], grasp_xyz, pin, model_fk, data_fk, int(fid))
            except Exception as e:
                if args.plot_debug:
                    print(f"[WARN] d_goal FK failed for sid={s}: {_short_exc(e)}")
                d_goal = None

        want_dbg = bool(args.traj_time_debug) and (int(s) == int(best_sid))
        t_arr, v_arr, a_arr, info = eeg._time_parameterize_xyz_by_clearance_and_curvature(
            xyz=xyz,
            clr=clr,
            safe_margin=float(safe_margin),
            d_goal=d_goal,
            goal_mode=str(args.traj_time_goal_mode),
            goal_sigma=float(args.traj_time_goal_sigma),
            goal_d_stop=float(args.traj_time_goal_d_stop),
            goal_d_full=float(args.traj_time_goal_d_full),
            v_free=float(args.traj_time_v_free),
            v_max=float(args.traj_time_v_max),
            v_min_ratio=float(args.traj_time_v_min_ratio),
            v_min_goal=float(args.traj_time_goal_v_min),
            goal_r_stop=float(args.traj_time_goal_r_stop),
            goal_k=float(args.traj_time_goal_k),
            obs_clr0_factor=float(args.traj_time_obs_clr0_factor),
            obs_clr_k_factor=float(args.traj_time_obs_clr_k_factor),
            a_max=float(args.traj_time_a_max),
            a_lat_max=float(args.traj_time_a_lat_max),
            kappa_ref_p=float(args.traj_time_kappa_ref_p),
            dt_min=float(args.traj_time_dt_min),
            dt_max=float(args.traj_time_dt_max),
            cap_smooth_window=int(args.traj_time_cap_smooth_window),
            j_max=float(args.traj_time_j_max),
            j_iters=int(args.traj_time_j_iters),
            return_debug_arrays=want_dbg,
        )
        traj_time[s] = np.asarray(t_arr, dtype=np.float32).reshape(H)
        traj_speed[s] = np.asarray(v_arr, dtype=np.float32).reshape(H)
        traj_acc[s] = np.asarray(a_arr, dtype=np.float32).reshape(H)

        if want_dbg:
            info_best = dict(info) if isinstance(info, dict) else None
            clr_best = np.asarray(clr, dtype=np.float64).reshape(H)
            if d_goal is not None:
                d_goal_best = np.asarray(d_goal, dtype=np.float64).reshape(H)

        if bool(args.traj_exec_enable):
            t_exec, xyz_exec, vel_exec = _resample_exec_uniform_dt(eeg, t=t_arr, xyz=xyz, dt_ctrl=float(args.traj_exec_dt))
            exec_list.append((t_exec, xyz_exec, vel_exec))

    # Build exec padded tensors
    if bool(args.traj_exec_enable) and len(exec_list) == S:
        exec_len = np.zeros((S,), dtype=np.int64)
        exec_max = 1
        for s in range(S):
            n = int(exec_list[s][0].shape[0])
            exec_len[s] = n
            exec_max = max(exec_max, n)
        exec_time = np.zeros((S, exec_max), dtype=np.float32)
        exec_xyz = np.zeros((S, exec_max, 3), dtype=np.float32)
        exec_vel = np.zeros((S, exec_max, 3), dtype=np.float32)
        for s in range(S):
            t_exec, xyz_exec, vel_exec = exec_list[s]
            n = int(exec_len[s])
            exec_time[s, :n] = np.asarray(t_exec, dtype=np.float32).reshape(n)
            exec_xyz[s, :n, :] = np.asarray(xyz_exec, dtype=np.float32).reshape(n, 3)
            exec_vel[s, :n, :] = np.asarray(vel_exec, dtype=np.float32).reshape(n, 3)
            if n < exec_max:
                exec_time[s, n:] = float(t_exec[-1])
                exec_xyz[s, n:, :] = np.asarray(xyz_exec[-1], dtype=np.float32).reshape(1, 3)
                exec_vel[s, n:, :] = 0.0

    # Overwrite / write out keys
    out["traj_time"] = traj_time
    out["traj_speed"] = traj_speed
    out["traj_acc"] = traj_acc
    out["traj_time_enable"] = np.array([1], dtype=np.int64)
    out["traj_time_v_free"] = np.array([float(args.traj_time_v_free)], dtype=np.float32)
    out["traj_time_v_max"] = np.array([float(args.traj_time_v_max)], dtype=np.float32)
    out["traj_time_v_min_ratio"] = np.array([float(args.traj_time_v_min_ratio)], dtype=np.float32)
    out["traj_time_a_max"] = np.array([float(args.traj_time_a_max)], dtype=np.float32)
    out["traj_time_a_lat_max"] = np.array([float(args.traj_time_a_lat_max)], dtype=np.float32)
    out["traj_time_kappa_ref_p"] = np.array([float(args.traj_time_kappa_ref_p)], dtype=np.float32)
    out["traj_time_dt_min"] = np.array([float(args.traj_time_dt_min)], dtype=np.float32)
    out["traj_time_dt_max"] = np.array([float(args.traj_time_dt_max)], dtype=np.float32)
    out["traj_time_cap_smooth_window"] = np.array([int(args.traj_time_cap_smooth_window)], dtype=np.int64)
    out["traj_time_j_max"] = np.array([float(args.traj_time_j_max)], dtype=np.float32)
    out["traj_time_j_iters"] = np.array([int(args.traj_time_j_iters)], dtype=np.int64)
    out["traj_time_goal_mode"] = np.array([str(args.traj_time_goal_mode)])
    out["traj_time_goal_sigma"] = np.array([float(args.traj_time_goal_sigma)], dtype=np.float32)
    out["traj_time_goal_d_stop"] = np.array([float(args.traj_time_goal_d_stop)], dtype=np.float32)
    out["traj_time_goal_d_full"] = np.array([float(args.traj_time_goal_d_full)], dtype=np.float32)
    out["traj_time_goal_v_min"] = np.array([float(args.traj_time_goal_v_min)], dtype=np.float32)
    # Legacy key retained for backward compatibility.
    out["traj_time_v_min_goal"] = np.array([float(args.traj_time_goal_v_min)], dtype=np.float32)
    out["traj_time_goal_r_stop"] = np.array([float(args.traj_time_goal_r_stop)], dtype=np.float32)
    out["traj_time_goal_k"] = np.array([float(args.traj_time_goal_k)], dtype=np.float32)
    out["traj_time_obs_clr0_factor"] = np.array([float(args.traj_time_obs_clr0_factor)], dtype=np.float32)
    out["traj_time_obs_clr_k_factor"] = np.array([float(args.traj_time_obs_clr_k_factor)], dtype=np.float32)

    if exec_len is not None and exec_time is not None:
        out["traj_exec_enable"] = np.array([1], dtype=np.int64)
        out["traj_exec_dt"] = np.array([float(args.traj_exec_dt)], dtype=np.float32)
        out["traj_exec_len"] = np.asarray(exec_len, dtype=np.int64)
        out["traj_exec_time"] = np.asarray(exec_time, dtype=np.float32)
        out["traj_exec_xyz"] = np.asarray(exec_xyz, dtype=np.float32)
        out["traj_exec_vel"] = np.asarray(exec_vel, dtype=np.float32)

    out_npz = str(args.out_npz)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, **out)

    t_best = np.asarray(traj_time[best_sid], dtype=np.float64).reshape(-1)
    print(f"[OK] wrote {out_npz}")
    print(f"[TIME] best_sid={best_sid} t_end={float(t_best[-1]):.4f} (target 5~20s)")

    if bool(args.traj_time_debug):
        if info_best is None:
            print("[TRAJ_TIME_DEBUG] no debug info for best_sid (missing d_goal or debug arrays).")
            return
        if d_goal_best is None or (not np.any(np.isfinite(d_goal_best))):
            print("[TRAJ_TIME_DEBUG] d_goal not available; cannot locate grasp i*.")
            return

        t_best = np.asarray(traj_time[best_sid], dtype=np.float64).reshape(H)
        v_best = np.asarray(traj_speed[best_sid], dtype=np.float64).reshape(H)

        dbg_v_cap = info_best.get("dbg_v_cap", None)
        dbg_v_cap_raw = info_best.get("dbg_v_cap_raw", None)
        dbg_v_goal = info_best.get("dbg_v_cap_goal", None)
        dbg_v_obs = info_best.get("dbg_v_cap_clr", None)
        dbg_v_curv = info_best.get("dbg_v_cap_curv", None)
        dbg_dt = info_best.get("dbg_dt_seg", None)

        i_star = int(np.nanargmin(d_goal_best))
        i_star = int(max(0, min(H - 1, i_star)))
        t_star = float(t_best[i_star])
        dg_star = float(d_goal_best[i_star])
        v_star = float(v_best[i_star])
        vgoal_star = float(dbg_v_goal[i_star]) if isinstance(dbg_v_goal, np.ndarray) else float("nan")
        vobs_star = float(dbg_v_obs[i_star]) if isinstance(dbg_v_obs, np.ndarray) else float("nan")
        vcurv_star = float(dbg_v_curv[i_star]) if isinstance(dbg_v_curv, np.ndarray) else float("nan")
        vcap_star = float(dbg_v_cap[i_star]) if isinstance(dbg_v_cap, np.ndarray) else float("nan")
        print(
            "[TRAJ_TIME_EVENT] "
            f"best_sid={best_sid:03d} i_star={i_star:03d} t_grasp={t_star:.4f} d_goal*={dg_star:.4f} "
            f"v*={v_star:.4f} cap(goal/obs/curv/final)=({vgoal_star:.4f},{vobs_star:.4f},{vcurv_star:.4f},{vcap_star:.4f})"
        )

        i0 = int(max(0, i_star - 10))
        i1 = int(min(H - 1, i_star + 10))
        v_max_cfg = float(info_best.get("v_max", float("inf")))
        print("[TRAJ_TIME_TABLE] i t d_goal clr cap_goal cap_obs cap_curv cap_raw cap_final v_after dom dt")
        for ii in range(i0, i1 + 1):
            t_i = float(t_best[ii])
            dg_i = float(d_goal_best[ii])
            clr_i = float(clr_best[ii]) if clr_best is not None else float("nan")
            vgoal_i = float(dbg_v_goal[ii]) if isinstance(dbg_v_goal, np.ndarray) else float("nan")
            vobs_i = float(dbg_v_obs[ii]) if isinstance(dbg_v_obs, np.ndarray) else float("nan")
            vcurv_i = float(dbg_v_curv[ii]) if isinstance(dbg_v_curv, np.ndarray) else float("nan")
            vcap_raw_i = float(dbg_v_cap_raw[ii]) if isinstance(dbg_v_cap_raw, np.ndarray) else float("nan")
            vcap_i = float(dbg_v_cap[ii]) if isinstance(dbg_v_cap, np.ndarray) else float("nan")
            v_i = float(v_best[ii])
            dt_i = float(dbg_dt[ii]) if isinstance(dbg_dt, np.ndarray) and ii < dbg_dt.shape[0] else float("nan")

            dom = "?"
            vals = [vgoal_i, vobs_i, vcurv_i, v_max_cfg]
            names = ["goal", "obs", "curv", "vmax"]
            try:
                j = int(np.nanargmin(np.asarray(vals, dtype=np.float64)))
                dom = names[j]
            except Exception:
                dom = "?"

            print(
                f"[TRAJ_TIME_ROW] i={ii:03d} t={t_i:.4f} d_goal={dg_i:.4f} clr={clr_i:.4f} "
                f"cap_goal={vgoal_i:.4f} cap_obs={vobs_i:.4f} cap_curv={vcurv_i:.4f} "
                f"cap_raw={vcap_raw_i:.4f} cap_final={vcap_i:.4f} v_after={v_i:.4f} dom={dom} dt={dt_i:.4f}"
            )


if __name__ == "__main__":
    main()
