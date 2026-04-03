# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np

# isaacgym 必须先于 torch（如果你的环境有这个要求）
try:
    import isaacgym  # noqa: F401
except Exception:
    pass

import torch
from torch.utils.data import DataLoader

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
from mpd.models.diffusion_models.models import TemporalUnet
from mpd.models.diffusion_models import GaussianDiffusionModel


# ============================================================
# Planner-like losses (operate on model predicted x_recon / x0)
# ============================================================

def piecewise_line_ref_batch(xyz, t_grasp):
    """
    xyz: [B,H,3]  (we will use xyz's own anchors: xyz[:,0], xyz[:,tg], xyz[:,-1])
    t_grasp: [B] int64, each in [1, H-2]
    return ref: [B,H,3] piecewise straight (start->tg->goal)
    """
    B, H, _ = xyz.shape
    tg = torch.clamp(t_grasp, 1, H - 2).long()

    x0 = xyz[:, 0, :]      # [B,3]
    xT = xyz[:, -1, :]     # [B,3]
    xtg = xyz[torch.arange(B, device=xyz.device), tg, :]  # [B,3]

    # alpha over time
    t = torch.arange(H, device=xyz.device).view(1, H, 1).float()  # [1,H,1]

    # seg1: 0..tg
    tg_f = tg.view(B, 1, 1).float()  # [B,1,1]
    a1 = torch.clamp(t / (tg_f + 1e-6), 0.0, 1.0)  # [B,H,1] after broadcast
    ref1 = (1.0 - a1) * x0.view(B, 1, 3) + a1 * xtg.view(B, 1, 3)

    # seg2: tg..H-1
    denom2 = ((H - 1) - tg).view(B, 1, 1).float() + 1e-6
    a2 = torch.clamp((t - tg_f) / denom2, 0.0, 1.0)
    ref2 = (1.0 - a2) * xtg.view(B, 1, 3) + a2 * xT.view(B, 1, 3)

    # choose segment by mask
    mask1 = (t <= tg_f).float()  # [B,H,1]
    ref = mask1 * ref1 + (1.0 - mask1) * ref2

    # enforce exact anchors
    ref[:, 0, :] = x0
    ref[torch.arange(B, device=xyz.device), tg, :] = xtg
    ref[:, -1, :] = xT
    return ref


def loss_line_piecewise(xyz_pred, t_grasp):
    ref = piecewise_line_ref_batch(xyz_pred, t_grasp)
    return ((xyz_pred - ref) ** 2).mean()


def loss_len2(xyz_pred):
    dx = xyz_pred[:, 1:] - xyz_pred[:, :-1]  # [B,H-1,3]
    return (dx ** 2).mean()


def loss_step_uniform(xyz_pred):
    d = torch.norm(xyz_pred[:, 1:] - xyz_pred[:, :-1], dim=-1)  # [B,H-1]
    d_mean = d.mean(dim=1, keepdim=True)
    return ((d - d_mean) ** 2).mean()


def loss_jerk_xyz(xyz_pred):
    # third finite difference on xyz
    if xyz_pred.shape[1] < 4:
        return xyz_pred.new_tensor(0.0)
    x0 = xyz_pred[:, :-3]
    x1 = xyz_pred[:, 1:-2]
    x2 = xyz_pred[:, 2:-1]
    x3 = xyz_pred[:, 3:]
    j = (-x0 + 3 * x1 - 3 * x2 + x3)
    return (j ** 2).mean()


# ============================================================
# Utils / existing losses
# ============================================================

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_q9_batch(dataset, q_b9, device):
    """
    把 [B,9] 的状态归一化到和 traj 同一空间（跟 eval 的 normalize_state9_to_1d 逻辑一致）
    这里优先调用 dataset.normalize_control_points / normalizer
    """
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
        tmp = tmp[:, 0, :]  # [B,9]
    assert tmp.shape == q_b9.shape, f"normalized shape mismatch: {tuple(tmp.shape)} vs {tuple(q_b9.shape)}"
    return tmp.contiguous()


def apply_3pt_hard_writeback(traj_bhd, q0_b9, qg_b9, qT_b9, t_grasp_b):
    """
    traj_bhd: [B,H,9]
    q0_b9/qg_b9/qT_b9: [B,9] (已归一化)
    t_grasp_b: [B] int64，范围 [1, H-2]
    """
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
    """
    全维度 vel/acc（温和正则）
    """
    vel = traj[:, 1:, :] - traj[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    return (vel ** 2).mean(), (acc ** 2).mean()


def jerk_loss_xyz_arm(traj):
    """
    只对 xyz(0:3) + arm(7:9) 做 jerk（三阶差分）
    """
    if traj.shape[1] < 4:
        return traj.new_tensor(0.0)
    j = traj[:, 3:, :] - 3 * traj[:, 2:-1, :] + 3 * traj[:, 1:-2, :] - traj[:, :-3, :]
    j_xyz = j[:, :, 0:3]
    j_arm = j[:, :, 7:9]
    return (j_xyz ** 2).mean() + (j_arm ** 2).mean()


def tg_local_smooth_loss_xyz_arm(traj, t_grasp):
    """
    只惩罚抓取点附近的“折点”（tg-1,tg,tg+1）的二阶差分
    """
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


# ============================================================
# Train
# ============================================================

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

    # -----------------------------
    # dataset
    # -----------------------------
    H = 144
    ds = EagleGraspNPZDataset(root_dir=data_root, H=H, only_accepted=True)
    print("[dataset] size =", len(ds), "H=", H)

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # -----------------------------
    # model
    # -----------------------------
    dx = 9
    cond_embed_dim = 18  # 目前主要靠 hard_conds
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

    # -----------------------------
    # resume
    # -----------------------------
    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    step = 0
    if resume_ckpt:
        print("[resume] loading:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[resume] optimizer restored")
                except Exception as e:
                    print("[resume] optimizer restore failed:", repr(e))
            step = int(ckpt.get("step", 0))
        else:
            model.load_state_dict(ckpt, strict=True)
        print("[resume] start step =", step)
    else:
        print("[resume] none")

    # -----------------------------
    # train config
    # -----------------------------
    num_train_steps = 80000
    log_every = 100
    save_every = 2000

    # 原有平滑正则（可以后续逐步降低）
    w_vel = 0.2
    w_acc = 0.4
    w_j = 0.2
    w_bc = 0.4
    w_tg = 1.0

    # 新增：planner-like（作用在模型预测 x_recon 上）
    w_line = 1.0   # 直线偏好（压绕路）
    w_uni  = 1.0   # 步长均匀（压忽快忽慢）
    w_len  = 0.2   # 路径长度 L2（辅助压绕远）
    w_jxyz = 0.2   # xyz jerk（辅助平滑/避免局部振荡）

    print("[RUN]", run_dir)
    print("[W] vel", w_vel, "acc", w_acc, "jerk(xyz+arm)", w_j, "bc", w_bc, "tg_local(xyz+arm)", w_tg)
    print("[W] planner-like: line", w_line, "uni", w_uni, "len2", w_len, "jerk_xyz", w_jxyz)

    model.train()
    while step < num_train_steps:
        for batch in loader:
            traj = batch["traj"].float().to(device)  # [B,H,9] (GT)
            q_start = batch["q_start"].float().to(device)  # [B,9]
            q_goal = batch["q_goal"].float().to(device)  # [B,9]
            q_grasp_state = batch["q_grasp_state"].float().to(device)  # [B,9]
            t_grasp = batch["t_grasp"].long().to(device).view(-1)  # [B]

            t_grasp = torch.clamp(t_grasp, 1, H - 2)

            # normalize hard states to traj space
            q0_n = normalize_q9_batch(ds, q_start, device)
            qT_n = normalize_q9_batch(ds, q_goal, device)
            qg_n = normalize_q9_batch(ds, q_grasp_state, device)

            # GT trajectory with hard writeback (for diffusion training target & some regularizers)
            traj_n = traj.clone()
            traj_n = apply_3pt_hard_writeback(traj_n, q0_n, qg_n, qT_n, t_grasp)

            uniq_tg = make_hard_conds_3pt(t_grasp)

            # -----------------------------------------
            # Diffusion loss (training core)
            # IMPORTANT: to compute planner-like losses on x_recon,
            # we also need x_recon from the diffusion loss call.
            # We’ll compute per-group and keep x_recon for those groups.
            # -----------------------------------------
            diff_loss_total = 0.0
            count_total = 0

            # planner-like losses computed on predicted recon
            pl_line = 0.0
            pl_uni = 0.0
            pl_len = 0.0
            pl_jxyz = 0.0
            pl_count = 0

            for tg in uniq_tg:
                mask = (t_grasp == tg)
                if mask.sum().item() == 0:
                    continue

                traj_g = traj_n[mask]
                q0_g = q0_n[mask]
                qT_g = qT_n[mask]
                qg_g = qg_n[mask]
                tg_g = t_grasp[mask]

                ctx_g = {"start": q0_g, "goal": qT_g}
                hard_conds = {0: q0_g, H - 1: qT_g, int(tg): qg_g}

                # model.loss returns (loss, info)
                dl, info = model.loss(traj_g, ctx_g, hard_conds)
                diff_loss_total = diff_loss_total + dl
                count_total += 1

                # ---- planner-like losses on x_recon (predicted x0) ----
                # We try to fetch x_recon from info. Different codebases name it differently.
                # Common keys: "x_recon", "x0_pred", "pred_x0".
                x_recon = None
                if isinstance(info, dict):
                    for k in ["x_recon", "x0_pred", "pred_x0", "x_start_pred"]:
                        if k in info:
                            x_recon = info[k]
                            break

                # If not provided, skip (but ideally we want it; see note below)
                if x_recon is not None:
                    xyz_pred = x_recon[..., :3]
                    pl_line = pl_line + loss_line_piecewise(xyz_pred, tg_g)
                    pl_uni = pl_uni + loss_step_uniform(xyz_pred)
                    pl_len = pl_len + loss_len2(xyz_pred)
                    pl_jxyz = pl_jxyz + loss_jerk_xyz(xyz_pred)
                    pl_count += 1

            diff_loss = diff_loss_total / max(1, count_total)

            # fallback if x_recon not available in info:
            # we can still train diffusion + your smoothness regularizers.
            # but for planner-like, we strongly recommend exposing x_recon in model.loss.
            if pl_count > 0:
                loss_planner = (
                    w_line * (pl_line / pl_count)
                    + w_uni * (pl_uni / pl_count)
                    + w_len * (pl_len / pl_count)
                    + w_jxyz * (pl_jxyz / pl_count)
                )
            else:
                loss_planner = diff_loss.new_tensor(0.0)

            # regularizers on traj_n (GT with hard writeback)
            loss_vel, loss_acc = smoothness_loss(traj_n)
            loss_bc = endpoint_bc_loss(traj_n)
            loss_j = jerk_loss_xyz_arm(traj_n)
            loss_tg = tg_local_smooth_loss_xyz_arm(traj_n, t_grasp)

            loss = (
                diff_loss
                + loss_planner
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
                    f"vel={loss_vel.item():.6f} acc={loss_acc.item():.6f} "
                    f"jerk(xyz+arm)={loss_j.item():.6f} tg(xyz+arm)={loss_tg.item():.6f} "
                    f"bc={loss_bc.item():.6f} uniq_tg={len(uniq_tg)} pl_count={pl_count}"
                )

            if step % save_every == 0 and step > 0:
                ckpt = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(ckpt, os.path.join(ckpt_dir, f"step_{step:07d}.pth"))
                torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
                print(f"[save] step {step} -> {ckpt_dir}")

            step += 1
            if step >= num_train_steps:
                break

    ckpt = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
    print(f"[done] saved to {os.path.join(ckpt_dir, 'ema_model_current.pth')}")


if __name__ == "__main__":
    main()