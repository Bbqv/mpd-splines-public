# -*- coding: utf-8 -*-

import os
import time
import random
import inspect
import numpy as np

# 如果环境要求 isaacgym 必须先于 torch import，这里先尝试导入
try:
    import isaacgym  # noqa: F401
except Exception:
    pass

import torch
from torch.utils.data import DataLoader

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
from mpd.models.diffusion_models.models import TemporalUnet
from mpd.models.diffusion_models import GaussianDiffusionModel


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def apply_2pt_hard_conditioning(x, q_start, q_goal):
    """
    x: [B, H, D]
    q_start/q_goal: [B, D]
    """
    x[:, 0, :] = q_start
    x[:, -1, :] = q_goal
    return x


def make_2pt_hard_conds(q_start, q_goal, horizon, n_samples=None):
    """
    q_start, q_goal:
      [D] or [B, D]
    returns:
      {0: [B, D], H-1: [B, D]}
    """
    if q_start.dim() == 1:
        assert n_samples is not None
        q_start = q_start.unsqueeze(0).repeat(n_samples, 1)
    elif q_start.dim() != 2:
        raise ValueError(f"bad q_start shape: {q_start.shape}")

    if q_goal.dim() == 1:
        assert n_samples is not None
        q_goal = q_goal.unsqueeze(0).repeat(n_samples, 1)
    elif q_goal.dim() != 2:
        raise ValueError(f"bad q_goal shape: {q_goal.shape}")

    return {
        0: q_start,
        horizon - 1: q_goal,
    }


def make_context_dict(q_start, q_goal):
    """
    当前仓库的 context_d 需要是 dict，而不是 tensor
    """
    return {
        "start": q_start,
        "goal": q_goal,
    }


def smoothness_loss(traj):
    """
    traj: [B, H, D]
    一阶/二阶差分平滑正则
    """
    vel = traj[:, 1:, :] - traj[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    loss_vel = (vel ** 2).mean()
    loss_acc = (acc ** 2).mean()
    return loss_vel, loss_acc


def endpoint_bc_loss(traj):
    """
    软约束边界平滑:
      v0 ~ 0, a0 ~ 0, vend ~ 0, aend ~ 0
    traj: [B, H, D]
    """
    v0 = traj[:, 1, :] - traj[:, 0, :]
    vend = traj[:, -1, :] - traj[:, -2, :]

    a0 = traj[:, 2, :] - 2 * traj[:, 1, :] + traj[:, 0, :]
    aend = traj[:, -1, :] - 2 * traj[:, -2, :] + traj[:, -3, :]

    return (v0 ** 2).mean() + (vend ** 2).mean() + (a0 ** 2).mean() + (aend ** 2).mean()


@torch.no_grad()
def quick_sanity(model, ds, device):
    model.eval()

    sample = ds[0]
    traj = torch.as_tensor(sample["traj"], dtype=torch.float32, device=device)        # [H, D]
    q_start = torch.as_tensor(sample["q_start"], dtype=torch.float32, device=device)  # [D]
    q_goal = torch.as_tensor(sample["q_goal"], dtype=torch.float32, device=device)    # [D]

    H, D = traj.shape
    n_samples = 4

    q_start_b = q_start.unsqueeze(0).repeat(n_samples, 1)   # [4, 9]
    q_goal_b = q_goal.unsqueeze(0).repeat(n_samples, 1)     # [4, 9]

    context_d = make_context_dict(q_start_b, q_goal_b)
    hard_conds = make_2pt_hard_conds(q_start, q_goal, H, n_samples=n_samples)

    out = model.run_inference(
        context_d=context_d,
        hard_conds=hard_conds,
        n_samples=n_samples,
        horizon=H,
    )

    print("[quick_sanity] out.shape =", tuple(out.shape))
    print("[quick_sanity] start err =", torch.norm(out[:, 0, :] - hard_conds[0], dim=-1).mean().item())
    print("[quick_sanity] goal  err =", torch.norm(out[:, -1, :] - hard_conds[H - 1], dim=-1).mean().item())


def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    data_root = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd"
    out_root = "data_outputs_eagle_2pt"
    ensure_dir(out_root)

    run_dir = os.path.join(out_root, str(int(time.time())))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    ds = EagleGraspNPZDataset(
        root_dir=data_root,
        H=144,
        only_accepted=True,
    )
    print("[dataset] size =", len(ds))

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    horizon = 144
    dx = 9
    context_dim = 18  # 这里只是给 conditioning embed dim 一个容量

    print("[TemporalUnet.__init__ signature]", inspect.signature(TemporalUnet.__init__))
    print("[GaussianDiffusionModel.__init__ signature]", inspect.signature(GaussianDiffusionModel.__init__))
    print("[debug] horizon =", horizon, "dx =", dx, "context_dim =", context_dim)

    denoise_fn = TemporalUnet(
        n_support_points=horizon,
        state_dim=dx,
        conditioning_embed_dim=context_dim,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        self_attention=False,
    )

    model = GaussianDiffusionModel(
        denoise_fn=denoise_fn,
        variance_schedule="cosine",
        n_diffusion_steps=100,
        clip_denoised=True,
        predict_epsilon=True,
        loss_type="l2",
        horizon=horizon,
        observation_dim=dx,
        action_dim=0,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    num_train_steps = 100000
    log_every = 100
    save_every = 2000
    step = 0

    print("[RUN]", run_dir)

    #quick_sanity(model, ds, device)

    model.train()
    while step < num_train_steps:
        for batch in loader:
            traj = batch["traj"].float().to(device)          # [B,H,9]
            q_start = batch["q_start"].float().to(device)    # [B,9]
            q_goal = batch["q_goal"].float().to(device)      # [B,9]

            traj = apply_2pt_hard_conditioning(traj.clone(), q_start, q_goal)

            context_d = make_context_dict(q_start, q_goal)

            hard_conds = {
                0: q_start,
                horizon - 1: q_goal,
            }

            diff_loss, infos = model.loss(traj, context_d, hard_conds)

            loss_vel, loss_acc = smoothness_loss(traj)
            loss_bc = endpoint_bc_loss(traj)

            loss = diff_loss + 0.1 * loss_vel + 0.1 * loss_acc + 0.2 * loss_bc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0:
                print(
                    f"[step {step}] "
                    f"loss={loss.item():.6f} "
                    f"diff={diff_loss.item():.6f} "
                    f"vel={loss_vel.item():.6f} "
                    f"acc={loss_acc.item():.6f} "
                    f"bc={loss_bc.item():.6f}"
                )

            if step % save_every == 0 and step > 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_path = os.path.join(ckpt_dir, f"step_{step:07d}.pth")
                torch.save(ckpt, save_path)
                torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
                print(f"[save] {save_path}")

            step += 1
            if step >= num_train_steps:
                break

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "ema_model_current.pth"))
    print(f"[done] saved to {os.path.join(ckpt_dir, 'ema_model_current.pth')}")


if __name__ == "__main__":
    main()