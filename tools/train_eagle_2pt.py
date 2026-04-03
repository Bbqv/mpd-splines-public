import os, time
import torch
from torch.utils.data import DataLoader, random_split

from mpd.datasets.eagle_grasp_npz_dataset import EagleGraspNPZDataset
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.context_models import ContextModelQs, ContextModelCombined
from mpd.utils.loaders import get_model, get_loss
from mpd.trainer.trainer import Trainer

# ------------------ config ------------------
ROOT = "/home/yongxin/wpj/dataset_room_4x4x2_relaxed_mpd"
H = 144
BATCH = 128
LR = 3e-4
STEPS = 200000          # 先跑 2e5，能很快看稳定性；稳定后再加大
CKPT_EVERY = 5000
DEVICE = "cuda:0"
SEED = 1726484688
OUTDIR = "data_outputs_eagle_2pt"
# -------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)

    ds = EagleGraspNPZDataset(root_dir=ROOT, H=H, only_accepted=True)
    n = len(ds)
    n_val = max(200, int(0.02 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # ---- model (keep SIMPLE + CONSISTENT) ----
    # context: only qs (22-dim). 不要 Combined 里再拼其它，先稳定闭环
    context_model_qs = ContextModelQs(in_dim=ds.context_q_dim, out_dim=128, n_layers=2, act="relu")
    context_model = ContextModelCombined(context_model_qs=context_model_qs, context_model_ee_pose_goal=None, out_dim=128)

    unet_configs = dict(
        state_dim=ds.state_dim,                  # 9
        n_support_points=ds.n_learnable_control_points,  # H
        unet_input_dim=32,
        dim_mults=UNET_DIM_MULTS[1],
        conditioning_type="default",
        conditioning_embed_dim=128,
    )

    model = get_model(
        model_class="GaussianDiffusionModel",
        denoise_fn=TemporalUnet(**unet_configs),
        context_model=context_model,
        tensor_args={"device": device, "dtype": torch.float32},
        variance_schedule="cosine",
        n_diffusion_steps=100,
        predict_epsilon=True,
        **unet_configs,
    ).to(device)

    loss_fn = get_loss(
        loss_class="GaussianDiffusionLoss",
        model=model,
        tensor_args={"device": device, "dtype": torch.float32},
    )

    os.makedirs(OUTDIR, exist_ok=True)
    run_name = str(int(time.time()))
    run_dir = os.path.join(OUTDIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Trainer: 2pt hard conditioning in TRAINING ----
    # 关键：让 Trainer/ loss 在每个 batch 都把 hard_conds 写回 x_noisy / x_recon（你之前的逻辑）
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        results_dir=run_dir,
        train_num_steps=STEPS,
        lr=LR,
        use_ema=True,
        amp=False,
        clip_grad=False,
        steps_til_ckpt=CKPT_EVERY,
        steps_til_summary=CKPT_EVERY,
        # 下面这个开关名字如果你 trainer 里不同，稍后我告诉你怎么 grep 改
        hard_endpoints_only=True,
    )

    trainer.train()

if __name__ == "__main__":
    main()