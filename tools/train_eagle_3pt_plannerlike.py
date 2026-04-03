# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np

try:
    import isaacgym  # noqa: F401
except Exception:
    pass

import torch
from torch.utils.data import DataLoader

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
from mpd.models.diffusion_models.models import TemporalUnet
from mpd.models.diffusion_models import GaussianDiffusionModel
from tools.planner_losses import planner_like_losses_xyz, random_sphere_obstacle_loss_xyz


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def env_int(name, default):
    val = os.environ.get(name, "").strip()
    if val == "":
        return default
    try:
        return int(val)
    except Exception:
        print(f"[env] invalid int for {name}={val!r}, fallback {default}")
        return default


def env_float(name, default):
    val = os.environ.get(name, "").strip()
    if val == "":
        return default
    try:
        return float(val)
    except Exception:
        print(f"[env] invalid float for {name}={val!r}, fallback {default}")
        return default


def env_flag(name, default=False):
    val = os.environ.get(name, "").strip().lower()
    if val == "":
        return default
    return val in ("1", "true", "yes", "y", "on")


def env_vec3(name, default):
    val = os.environ.get(name, "").strip()
    if val == "":
        return tuple(float(x) for x in default)
    parts = [p.strip() for p in val.split(",")]
    if len(parts) != 3:
        print(f"[env] invalid vec3 for {name}={val!r}, fallback {default}")
        return tuple(float(x) for x in default)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        print(f"[env] invalid vec3 for {name}={val!r}, fallback {default}")
        return tuple(float(x) for x in default)


def normalize_q9_batch(dataset, q_b9, device):
    if not torch.is_tensor(q_b9):
        q_b9 = torch.as_tensor(q_b9, dtype=torch.float32, device=device)
    q_b9 = q_b9.float().to(device)
    assert q_b9.dim() == 2 and q_b9.shape[1] == 9, f"q_b9 must be [B,9], got {tuple(q_b9.shape)}"

    tmp = q_b9.unsqueeze(1)  # [B,1,9]
    if hasattr(dataset, "normalize_control_points"):
        tmp = dataset.normalize_control_points(tmp)
    else:
        for attr in ["control_point_normalizer", "cp_normalizer", "normalizer"]:
            if hasattr(dataset, attr):
                norm = getattr(dataset, attr)
                if hasattr(norm, "normalize_control_points"):
                    tmp = norm.normalize_control_points(tmp)
                    break
                if hasattr(norm, "normalize"):
                    tmp = norm.normalize(tmp)
                    break

    if tmp.dim() == 3:
        tmp = tmp[:, 0, :]
    return tmp.contiguous()


def apply_3pt_hard_writeback(traj_bhd, q0_b9, qg_b9, qT_b9, t_grasp_b):
    B, H, D = traj_bhd.shape
    traj_bhd[:, 0, :] = q0_b9
    traj_bhd[:, H - 1, :] = qT_b9
    ar = torch.arange(B, device=traj_bhd.device)
    traj_bhd[ar, t_grasp_b, :] = qg_b9
    return traj_bhd


def make_hard_conds_3pt(t_grasp_b):
    uniq = torch.unique(t_grasp_b).tolist()
    return [int(x) for x in uniq]


def smoothness_loss(traj):
    vel = traj[:, 1:, :] - traj[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    return (vel ** 2).mean(), (acc ** 2).mean()


def jerk_loss_xyz_arm(traj):
    if traj.shape[1] < 4:
        return traj.new_tensor(0.0)
    j = traj[:, 3:, :] - 3 * traj[:, 2:-1, :] + 3 * traj[:, 1:-2, :] - traj[:, :-3, :]
    j_xyz = j[:, :, 0:3]
    j_arm = j[:, :, 7:9]
    return (j_xyz ** 2).mean() + (j_arm ** 2).mean()


def tg_local_smooth_loss_xyz_arm(traj, t_grasp):
    B, H, D = traj.shape
    tg = torch.clamp(t_grasp, 1, H - 2)
    ar = torch.arange(B, device=traj.device)

    q_prev = traj[ar, tg - 1, :]
    q_mid = traj[ar, tg, :]
    q_next = traj[ar, tg + 1, :]

    acc_local = q_next - 2.0 * q_mid + q_prev
    acc_xyz = acc_local[:, 0:3]
    acc_arm = acc_local[:, 7:9]
    return (acc_xyz ** 2).mean() + (acc_arm ** 2).mean()


def endpoint_bc_loss(traj):
    v0 = traj[:, 1, :] - traj[:, 0, :]
    vend = traj[:, -1, :] - traj[:, -2, :]
    a0 = traj[:, 2, :] - 2 * traj[:, 1, :] + traj[:, 0, :]
    aend = traj[:, -1, :] - 2 * traj[:, -2, :] + traj[:, -3, :]
    return (v0 ** 2).mean() + (vend ** 2).mean() + (a0 ** 2).mean() + (aend ** 2).mean()


def ramp(step, start, end):
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return float(step - start) / float(end - start)


def get_pl_weights(step, cfg):
    """
    Conservative curriculum:
      phase-1 (stabilize): speed + jerk
      phase-2 (straighten): line + ortho_dd
      phase-3 (shorten): len
    Planner loss cap remains strict to avoid rare unstable spikes.
    """
    r_stab = ramp(step, cfg["stab_start"], cfg["stab_end"])
    r_line = ramp(step, cfg["line_start"], cfg["line_end"])
    r_len = ramp(step, cfg["len_start"], cfg["len_end"])

    w = {}
    w["speed"] = cfg["w_speed"] * r_stab
    w["jerk"] = cfg["w_jerk"] * r_stab

    w["line"] = cfg["w_line"] * r_line
    w["ortho_dd"] = cfg["w_ortho_dd"] * r_line

    w["len"] = cfg["w_len"] * r_len

    # strict cap: small growth across late phases only
    w["cap"] = cfg["cap_base"] + cfg["cap_line_gain"] * r_line + cfg["cap_len_gain"] * r_len
    return w


def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    data_root = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd"
    out_root = "data_outputs_eagle_3pt_plannerlike"
    ensure_dir(out_root)

    run_dir = os.path.join(out_root, str(int(time.time())))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    H = 144
    ds = EagleGraspNPZDataset(root_dir=data_root, H=H, only_accepted=True)
    print("[dataset] size =", len(ds), "H=", H)

    batch_size = env_int("BATCH_SIZE", 32)
    num_workers = env_int("NUM_WORKERS", 4)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    dx = 9
    cond_embed_dim = 18
    denoise_fn = TemporalUnet(
        n_support_points=H,
        state_dim=dx,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        self_attention=False,
        conditioning_embed_dim=cond_embed_dim,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    pl_cfg = {
        "stab_start": env_int("PL_STAB_START", 0),
        "stab_end": env_int("PL_STAB_END", 3000),
        "line_start": env_int("PL_LINE_START", 5000),
        "line_end": env_int("PL_LINE_END", 12000),
        "len_start": env_int("PL_LEN_START", 9000),
        "len_end": env_int("PL_LEN_END", 18000),
        "w_speed": env_float("PL_W_SPEED", 0.35),
        "w_jerk": env_float("PL_W_JERK", 0.90),
        "w_line": env_float("PL_W_LINE", 0.08),
        "w_ortho_dd": env_float("PL_W_ORTHO_DD", 0.30),
        "w_len": env_float("PL_W_LEN", 0.02),
        "cap_base": env_float("PL_CAP_BASE", 1.8),
        "cap_line_gain": env_float("PL_CAP_LINE_GAIN", 0.8),
        "cap_len_gain": env_float("PL_CAP_LEN_GAIN", 0.4),
    }
    pl_soft_cap = env_flag("PL_SOFT_CAP", default=False)
    pl_use_pred_smooth = env_flag("PL_USE_PRED_SMOOTH", default=False)

    # Optional obstacle-aware fine-tuning loss on x0_pred (random spheres, MPD-style)
    obst_train_enable = env_flag("OBST_TRAIN_ENABLE", default=False)
    obst_w = env_float("OBST_W", 0.20)
    obst_n = env_int("OBST_N_SPHERES", 2)
    obst_margin = env_float("OBST_MARGIN", 0.05)
    obst_r_min = env_float("OBST_R_MIN", 0.12)
    obst_r_max = env_float("OBST_R_MAX", 0.24)
    obst_xyz_min = env_vec3("OBST_XYZ_MIN", (-0.40, -0.40, 0.80))
    obst_xyz_max = env_vec3("OBST_XYZ_MAX", (0.40, 0.40, 1.80))
    obst_tg_mask_k = env_int("OBST_TG_MASK_K", 8)

    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    resume_opt = env_flag("RESUME_OPT", default=True)
    step = 0
    if resume_ckpt:
        print("[resume] loading:", resume_ckpt)
        print("[resume] RESUME_OPT =", int(resume_opt))
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            if resume_opt and "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[resume] optimizer restored")
                except Exception as e:
                    print("[resume] optimizer restore failed:", repr(e))
            elif not resume_opt:
                print("[resume] optimizer skipped (RESUME_OPT=0)")
            else:
                print("[resume] optimizer missing in ckpt")
            step = int(ckpt.get("step", 0))
        else:
            model.load_state_dict(ckpt, strict=True)
        print("[resume] start step =", step)
    else:
        print("[resume] none")

    num_train_steps = env_int("NUM_TRAIN_STEPS", 80000)
    log_every = env_int("LOG_EVERY", 100)
    save_every = env_int("SAVE_EVERY", 2000)

    # 原有平滑正则（在 traj_n/GT+hard writeback 上）
    w_vel = 0.2
    w_acc = 0.4
    w_j = 0.2
    w_bc = 0.4
    w_tg = 1.0

    # planner-like loss 超参（更稳，针对“锯齿折线”）
    pl_beta = env_float("PL_BETA", 0.02)                 # robust huber beta
    pl_line_stride = env_int("PL_LINE_STRIDE", 12)       # lower freq for line fitting
    pl_speed_stride = env_int("PL_SPEED_STRIDE", 4)      # set to 1 for stronger speed-uniformity
    pl_tg_mask_K = env_int("PL_TG_MASK_K", 12)           # mask around t_g
    pl_len_margin = env_float("PL_LEN_MARGIN", 0.10)
    pl_lp_kernel = env_int("PL_LP_KERNEL", 9)

    print("[RUN]", run_dir)
    print(f"[TRAIN] batch_size={batch_size} num_workers={num_workers} num_train_steps={num_train_steps} log_every={log_every} save_every={save_every}")
    print("[W] vel", w_vel, "acc", w_acc, "jerk(xyz+arm)", w_j, "bc", w_bc, "tg_local(xyz+arm)", w_tg)
    print(
        "[PL] beta", pl_beta,
        "line_stride", pl_line_stride,
        "speed_stride", pl_speed_stride,
        "tg_mask_K", pl_tg_mask_K,
        "len_margin", pl_len_margin,
        "lp_kernel", pl_lp_kernel,
    )
    print(
        "[PL] curriculum:",
        f"stab {pl_cfg['stab_start']}->{pl_cfg['stab_end']}",
        f"line {pl_cfg['line_start']}->{pl_cfg['line_end']}",
        f"len {pl_cfg['len_start']}->{pl_cfg['len_end']}",
        f"weights(spd={pl_cfg['w_speed']}, jerk={pl_cfg['w_jerk']}, line={pl_cfg['w_line']}, od={pl_cfg['w_ortho_dd']}, len={pl_cfg['w_len']})",
        f"cap={pl_cfg['cap_base']}+{pl_cfg['cap_line_gain']}*r_line+{pl_cfg['cap_len_gain']}*r_len",
    )
    print("[PL] soft_cap", int(pl_soft_cap), "use_pred_smooth", int(pl_use_pred_smooth))
    print(
        "[OBST_TRAIN]",
        f"enable={int(obst_train_enable)} w={obst_w} n={obst_n} margin={obst_margin} "
        f"r=[{obst_r_min},{obst_r_max}] xyz_min={obst_xyz_min} xyz_max={obst_xyz_max} "
        f"tg_mask_k={obst_tg_mask_k}"
    )

    model.train()

    try:
        while step < num_train_steps:
            for batch in loader:
                traj = batch["traj"].float().to(device)  # [B,H,9]
                q_start = batch["q_start"].float().to(device)
                q_goal = batch["q_goal"].float().to(device)
                q_grasp_state = batch["q_grasp_state"].float().to(device)
                t_grasp = batch["t_grasp"].long().to(device).view(-1)

                t_grasp = torch.clamp(t_grasp, 1, H - 2)

                q0_n = normalize_q9_batch(ds, q_start, device)
                qT_n = normalize_q9_batch(ds, q_goal, device)
                qg_n = normalize_q9_batch(ds, q_grasp_state, device)

                traj_n = traj.clone()
                traj_n = apply_3pt_hard_writeback(traj_n, q0_n, qg_n, qT_n, t_grasp)

                uniq_tg = make_hard_conds_3pt(t_grasp)

                diff_loss_total = traj.new_tensor(0.0)
                count_total = 0

                # planner-like accumulators (over unique tg groups)
                pl_line = traj.new_tensor(0.0)
                pl_ortho_dd = traj.new_tensor(0.0)
                pl_speed = traj.new_tensor(0.0)
                pl_jerk = traj.new_tensor(0.0)
                pl_len = traj.new_tensor(0.0)
                pl_count = 0
                # obstacle accumulators (optional, on x0_pred)
                obst_loss_sum = traj.new_tensor(0.0)
                obst_clr_min_sum = traj.new_tensor(0.0)
                obst_count = 0
                # predicted-trajectory smooth regularizers (optional)
                pred_vel = traj.new_tensor(0.0)
                pred_acc = traj.new_tensor(0.0)
                pred_bc = traj.new_tensor(0.0)
                pred_j = traj.new_tensor(0.0)
                pred_tg = traj.new_tensor(0.0)
                pred_count = 0

                for tg in uniq_tg:
                    mask = (t_grasp == tg)
                    if mask.sum().item() == 0:
                        continue

                    traj_g = traj_n[mask]
                    q0_g = q0_n[mask]
                    qT_g = qT_n[mask]
                    qg_g = qg_n[mask]

                    ctx_g = {"start": q0_g, "goal": qT_g}
                    hard_conds = {0: q0_g, H - 1: qT_g, int(tg): qg_g}

                    dl, info = model.loss(traj_g, ctx_g, hard_conds)
                    diff_loss_total = diff_loss_total + dl
                    count_total += 1

                    # read x0_pred (hard-conditioned) from info
                    x_recon = None
                    if isinstance(info, dict):
                        for k in ["x_recon", "x0_pred", "pred_x0", "x_start_pred"]:
                            if k in info:
                                x_recon = info[k]
                                break

                    if x_recon is not None:
                        xyz_pred = x_recon[..., :3]  # [B,H,3]

                        pl = planner_like_losses_xyz(
                            xyz_pred,
                            t_g=int(tg),
                            line_stride=pl_line_stride,
                            speed_stride=pl_speed_stride,
                            tg_mask_K=pl_tg_mask_K,
                            beta=pl_beta,
                            do_len_over=True,
                            len_margin=pl_len_margin,
                            lp_kernel=pl_lp_kernel,
                        )

                        # nan/inf guard per term
                        for kk in pl:
                            pl[kk] = torch.nan_to_num(pl[kk], nan=0.0, posinf=0.0, neginf=0.0)

                        pl_line = pl_line + pl["pl_line"]
                        pl_ortho_dd = pl_ortho_dd + pl["pl_ortho_dd"]
                        pl_speed = pl_speed + pl["pl_speed"]
                        pl_jerk = pl_jerk + pl["pl_jerk"]
                        pl_len = pl_len + pl["pl_len"]
                        pl_count += 1

                        if obst_train_enable:
                            lo_i, aux_i = random_sphere_obstacle_loss_xyz(
                                xyz_pred,
                                t_g=int(tg),
                                n_spheres=obst_n,
                                xyz_min=obst_xyz_min,
                                xyz_max=obst_xyz_max,
                                r_min=obst_r_min,
                                r_max=obst_r_max,
                                margin=obst_margin,
                                tg_mask_k=obst_tg_mask_k,
                            )
                            lo_i = torch.nan_to_num(lo_i, nan=0.0, posinf=0.0, neginf=0.0)
                            cmin_i = torch.nan_to_num(aux_i["clr_min"], nan=0.0, posinf=0.0, neginf=0.0)
                            obst_loss_sum = obst_loss_sum + lo_i
                            obst_clr_min_sum = obst_clr_min_sum + cmin_i
                            obst_count += 1

                        if pl_use_pred_smooth:
                            # smoothness regularizers on predicted trajectory (not GT traj_n)
                            v_i, a_i = smoothness_loss(x_recon)
                            j_i = jerk_loss_xyz_arm(x_recon)
                            bc_i = endpoint_bc_loss(x_recon)
                            tg_i = tg_local_smooth_loss_xyz_arm(
                                x_recon, torch.full((x_recon.shape[0],), int(tg), device=x_recon.device, dtype=torch.long)
                            )
                            pred_vel = pred_vel + v_i
                            pred_acc = pred_acc + a_i
                            pred_j = pred_j + j_i
                            pred_bc = pred_bc + bc_i
                            pred_tg = pred_tg + tg_i
                            pred_count += 1

                diff_loss = diff_loss_total / max(1, count_total)

                # curriculum weights based on global step
                wpl = get_pl_weights(step, pl_cfg)

                if pl_count > 0:
                    pl_line_m = pl_line / pl_count
                    pl_ortho_dd_m = pl_ortho_dd / pl_count
                    pl_speed_m = pl_speed / pl_count
                    pl_jerk_m = pl_jerk / pl_count
                    pl_len_m = pl_len / pl_count

                    loss_planner = (
                        wpl["line"] * pl_line_m
                        + wpl["ortho_dd"] * pl_ortho_dd_m
                        + wpl["speed"] * pl_speed_m
                        + wpl["jerk"] * pl_jerk_m
                        + wpl["len"] * pl_len_m
                    )

                    # cap mode: hard clamp (old behavior) or soft tanh cap
                    loss_planner = torch.nan_to_num(loss_planner, nan=0.0, posinf=0.0, neginf=0.0)
                    pre = loss_planner.detach()
                    if pl_soft_cap:
                        cap = max(float(wpl["cap"]), 1e-6)
                        loss_planner = cap * torch.tanh(loss_planner / cap)
                    else:
                        loss_planner = loss_planner.clamp(max=wpl["cap"])
                    hit_cap = float(pre.item() >= wpl["cap"] - 1e-6)
                else:
                    pl_line_m = diff_loss.new_tensor(0.0)
                    pl_ortho_dd_m = diff_loss.new_tensor(0.0)
                    pl_speed_m = diff_loss.new_tensor(0.0)
                    pl_jerk_m = diff_loss.new_tensor(0.0)
                    pl_len_m = diff_loss.new_tensor(0.0)
                    loss_planner = diff_loss.new_tensor(0.0)
                    hit_cap = 0.0

                if obst_count > 0:
                    loss_obst = obst_loss_sum / obst_count
                    obst_clr_min_m = obst_clr_min_sum / obst_count
                else:
                    loss_obst = diff_loss.new_tensor(0.0)
                    obst_clr_min_m = diff_loss.new_tensor(0.0)

                # smoothness regularizers mode
                if pl_use_pred_smooth and pred_count > 0:
                    loss_vel = pred_vel / pred_count
                    loss_acc = pred_acc / pred_count
                    loss_bc = pred_bc / pred_count
                    loss_j = pred_j / pred_count
                    loss_tg = pred_tg / pred_count
                else:
                    # old behavior: smooth regularizers on traj_n (= GT + hard writeback)
                    loss_vel, loss_acc = smoothness_loss(traj_n)
                    loss_bc = endpoint_bc_loss(traj_n)
                    loss_j = jerk_loss_xyz_arm(traj_n)
                    loss_tg = tg_local_smooth_loss_xyz_arm(traj_n, t_grasp)

                loss = (
                    diff_loss
                    + loss_planner
                    + (obst_w * loss_obst if obst_train_enable else 0.0)
                    + w_vel * loss_vel
                    + w_acc * loss_acc
                    + w_j * loss_j
                    + w_bc * loss_bc
                    + w_tg * loss_tg
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % log_every == 0:
                    print(
                        f"[step {step}] "
                        f"loss={loss.item():.6f} diff={diff_loss.item():.6f} pl={loss_planner.item():.6f} "
                        f"(cap={wpl['cap']:.2f} hit_cap={hit_cap:.0f} "
                        f"w_line={wpl['line']:.3f} w_od={wpl['ortho_dd']:.3f} w_spd={wpl['speed']:.3f} w_j={wpl['jerk']:.3f} w_len={wpl['len']:.3f}) "
                        f"pl_terms: line={pl_line_m.item():.6f} ortho_dd={pl_ortho_dd_m.item():.6f} "
                        f"speed={pl_speed_m.item():.6f} jerk={pl_jerk_m.item():.6f} len={pl_len_m.item():.6f} "
                        f"obst={loss_obst.item():.6f} obst_clr_min={obst_clr_min_m.item():.6f} "
                        f"pred_vel={loss_vel.item():.6f} pred_acc={loss_acc.item():.6f} "
                        f"jerk(xyz+arm)={loss_j.item():.6f} tg(xyz+arm)={loss_tg.item():.6f} "
                        f"bc={loss_bc.item():.6f} uniq_tg={len(uniq_tg)} pl_count={pl_count} pred_count={pred_count}"
                    )

                if step % save_every == 0 and step > 0:
                    ckpt = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
                    torch.save(ckpt, os.path.join(ckpt_dir, f"step_{step:07d}.pth"))
                    torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
                    print(f"[save] step {step} -> {ckpt_dir}")

                step += 1
                if step >= num_train_steps:
                    break

    except KeyboardInterrupt:
        ckpt = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(ckpt_dir, f"step_{step:07d}_interrupt.pth"))
        torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
        print(f"[interrupt-save] saved at step {step} -> {ckpt_dir}")
        raise

    ckpt = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
    print(f"[done] saved to {os.path.join(ckpt_dir, 'ema_model_current.pth')}")


if __name__ == "__main__":
    main()
