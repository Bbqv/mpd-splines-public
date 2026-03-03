# mpd/datasets/eagle_grasp_npz_dataset.py
import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class _DummyParametricTrajectory:
    def get_q_trajectory(self, control_points, *args, **kwargs):
        return control_points


def resample_linear(x: np.ndarray, H: int) -> np.ndarray:
    """
    Linear resample (T,D) -> (H,D)
    """
    T, D = x.shape
    if T == H:
        return x.astype(np.float32)
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, H)
    y = np.zeros((H, D), dtype=np.float32)
    for d in range(D):
        y[:, d] = np.interp(t_new, t_old, x[:, d])
    return y


class EagleGraspNPZDataset(Dataset):
    """
    Reads:
      - NPZ files from <root>/accepted/traj_XXXXXX.npz (states, controls, ee_positions, ...)
      - Summary CSV from <root>/logs/summary.csv for grasp/goal and timing params.

    Returns (TRAINING):
      traj/control_points: (H, 9)
        9 = base_pos(3) + base_quat(4) + arm_j(2)
        i.e. [x, y, z, qx, qy, qz, qw, j1, j2]
      cond/context_q: (22,)
        22 = start(9) + grasp_pos(3) + goal(9) + k_grasp_norm(1)

    Additionally returns (for eval/debug):
      states: (T,17) raw (no resample)
      ee_positions: (T,3) raw (no resample)  (planner FK result)
      controls: (T-1,6) raw (if present)
      ee_positions_resampled: (H,3) (optional convenience)
    """

    def __init__(self, root_dir: str, H: int = 141, only_accepted: bool = True):
        # dummy planning_task for compatibility with train.py
        class _Env:
            dim = 3

        class _Task:
            env = _Env()

        self.planning_task = _Task()
        self.planning_task.parametric_trajectory = _DummyParametricTrajectory()

        # ---------- dims ----------
        self.traj_state_dim = 9          # [pos3 + quat4 + joints2]
        self.cond_dim = 22               # start9 + grasp3 + goal9 + k1

        # --- compatibility attributes expected by mpd training code ---
        self.context_q_dim = self.cond_dim
        self.context_ee_goal_pose_dim = 0
        self.context_combined_dim = self.cond_dim

        # --- field keys expected by mpd losses/normalization ---
        self.field_key_control_points = "control_points"
        self.field_key_context_q = "context_q"
        self.field_key_context_ee_goal_pose = "context_ee_goal_pose"
        self.field_key_context_combined = "context_combined"
        self.field_key_task_id = "case_id"

        # trajectory dims expected by diffusion model
        self.control_points_dim = (H, self.traj_state_dim)
        self.state_dim = self.traj_state_dim
        self.dof = self.traj_state_dim
        self.traj_dim = self.traj_state_dim

        self.n_control_points = H
        self.n_learnable_control_points = H
        self.n_removed_control_points = 0

        self.root_dir = root_dir
        self.H = H

        sub = "accepted" if only_accepted else "rejected"
        self.files = sorted(glob.glob(os.path.join(root_dir, sub, "traj_*.npz")))
        assert len(self.files) > 0, f"No npz files found under {root_dir}/{sub}"

        csv_path = os.path.join(root_dir, "logs", "summary.csv")
        assert os.path.exists(csv_path), f"Missing summary.csv at {csv_path}"
        df = pd.read_csv(csv_path)

        # Keep accepted rows, and prefer the successful try per case_id
        if "accepted" in df.columns:
            df = df[df["accepted"] == True].copy()

        # Map case_id -> grasp/goal and timing. Use the row with try_id minimal (first success)
        if "try_id" in df.columns:
            df = df.sort_values(["case_id", "try_id"])
            df = df.groupby("case_id").head(1)

        # index by case_id for fast lookup
        self.by_case = df.set_index("case_id", drop=False)

    def __len__(self):
        return len(self.files)

    def unnormalize_control_points(self, control_points_normalized):
        return control_points_normalized

    def build_context(self, data_sample: dict):
        """
        Minimal context builder for MPD summary / sampling.
        It must return the same keys that ContextModelCombined.forward expects.
        """
        context_d = {
            "qs_normalized": data_sample["qs_normalized"],
        }
        if "ee_goal_pose_normalized" in data_sample:
            context_d["ee_goal_pose_normalized"] = data_sample["ee_goal_pose_normalized"]
        return context_d

    def render(self, *args, **kwargs):
        return

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        base = os.path.basename(npz_path)  # traj_000123.npz
        case_id = int(base.split("_")[1].split(".")[0])

        # read csv row
        if case_id not in self.by_case.index:
            raise KeyError(f"case_id {case_id} not found in summary.csv (file={base})")
        row = self.by_case.loc[case_id]

        grasp = np.array([row["grasp_x"], row["grasp_y"], row["grasp_z"]], dtype=np.float32)
        goal = np.array([row["goal_x"], row["goal_y"], row["goal_z"]], dtype=np.float32)

        dt = float(row["dt"])
        t_grasp = float(row["to_grasp"])  # summary.csv column name in your generator

        # ---- load npz ----
        data = np.load(npz_path, allow_pickle=True)

        # raw trajectories
        if "states" not in data:
            raise KeyError(f"'states' missing in {npz_path}. Keys={list(data.keys())}")
        states = data["states"].astype(np.float32)  # (T,17)
        T_npz = int(states.shape[0])

        if "ee_positions" not in data:
            raise KeyError(f"'ee_positions' missing in {npz_path}. Keys={list(data.keys())}")
        ee_positions = data["ee_positions"].astype(np.float32)  # (T,3)

        controls = None
        if "controls" in data:
            controls = data["controls"].astype(np.float32)  # (T-1,6) typically

        # ---- compute k_grasp ----
        # k_grasp is a time index; clamp in NPZ timeline first (T_npz),
        # then normalize to dataset H timeline (since traj is resampled to H).
        k_grasp_npz = int(round(t_grasp / max(dt, 1e-9)))
        k_grasp_npz = int(np.clip(k_grasp_npz, 0, T_npz - 1))

        # map NPZ index to resampled index approximately
        if T_npz <= 1:
            k_grasp = 0
        else:
            k_grasp = int(round(k_grasp_npz * (self.H - 1) / (T_npz - 1)))
        k_grasp = int(np.clip(k_grasp, 0, self.H - 1))
        k_grasp_norm = np.array([k_grasp / (self.H - 1)], dtype=np.float32)

        # ---- build training traj (H,9) ----
        # traj from states: base_pos(0:3) + base_quat(3:7) + arm_j(7:9) => (T,9)
        traj_raw = np.concatenate([states[:, 0:3], states[:, 3:7], states[:, 7:9]], axis=1).astype(np.float32)  # (T,9)
        traj = resample_linear(traj_raw, self.H)  # (H,9)

        # also resample ee for convenience (eval can use raw ee_positions)
        ee_positions_resampled = resample_linear(ee_positions.astype(np.float32), self.H)  # (H,3)

        # ---- build cond/context (22,) ----
        start = traj[0].copy()       # (9,)
        goal_state = traj[-1].copy() # (9,)
        # overwrite goal position with CSV goal (optional but consistent)
        goal_state[0:3] = goal

        cond = np.concatenate([start, grasp, goal_state, k_grasp_norm], axis=0).astype(np.float32)  # (22,)

        # -------- pad/crop trajectory length to dataset H (needed by TemporalUnet) --------
        # (Note: after resample_linear, traj already has length H; keep this block harmless)
        T = traj.shape[0]
        H = self.H
        if T < H:
            pad = np.repeat(traj[-1:, :], H - T, axis=0)
            traj = np.concatenate([traj, pad], axis=0)
        elif T > H:
            traj = traj[:H]

        out = {
            # raw (your names)
            "traj": torch.from_numpy(traj),
            "cond": torch.from_numpy(cond),

            # raw (framework-expected names)
            "control_points": torch.from_numpy(traj),  # (H,9)
            "context_q": torch.from_numpy(cond),       # (22,)
            "context_ee_goal_pose": torch.zeros(1, dtype=torch.float32),  # keep dim 1
            "context_combined": torch.from_numpy(cond),                   # (22,)
            "case_id": torch.tensor(case_id, dtype=torch.long),

            # normalized (identity for now, to satisfy gaussian_diffusion_loss)
            "control_points_normalized": torch.from_numpy(traj),
            "context_q_normalized": torch.from_numpy(cond),
            "context_ee_goal_pose_normalized": torch.zeros(1, dtype=torch.float32),
            "context_combined_normalized": torch.from_numpy(cond),

            # ---- extra GT for eval/debug ----
            "states": torch.from_numpy(states),                    # (T,17) raw
            "ee_positions": torch.from_numpy(ee_positions),        # (T,3) raw
            "ee_positions_resampled": torch.from_numpy(ee_positions_resampled),  # (H,3)
        }

        if controls is not None:
            out["controls"] = torch.from_numpy(controls)  # (T-1,6)

        # aliases expected elsewhere
        out["qs"] = out["context_q"]
        out["qs_normalized"] = out["context_q_normalized"]

        out["ee_goal_pose"] = out["context_ee_goal_pose"]
        out["ee_goal_pose_normalized"] = out["context_ee_goal_pose_normalized"]

        # convenience slices (world frame)
        # cond = [start(9), grasp(3), goal(9), k(1)]
        out["q_start"] = out["context_q"][0:9].clone()
        out["q_grasp"] = out["context_q"][9:12].clone()
        out["q_goal"]  = out["context_q"][12:21].clone()

        out["hard_conds"] = {}

        for k, v in list(out.items()):
            if isinstance(v, (int, float, bool)):
                out[k] = torch.tensor(v)

        return out