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
        22 = start(9) + goal(9) + grasp_pos(3) + k_grasp_norm(1)

    NEW (scheme-3 support):
      sg_xyz_normalized: (6,) = [start_xyz(3), goal_xyz(3)]
      context_sg_xyz_normalized: same as above (explicit key for context models)
    """

    def __init__(self, root_dir: str, H: int = 141, only_accepted: bool = True):
        class _Env:
            dim = 3

        class _Task:
            env = _Env()

        self.planning_task = _Task()
        self.planning_task.parametric_trajectory = _DummyParametricTrajectory()

        self.traj_state_dim = 9
        self.cond_dim = 22

        self.context_q_dim = self.cond_dim
        self.context_ee_goal_pose_dim = 0
        self.context_combined_dim = self.cond_dim

        self.field_key_control_points = "control_points"
        self.field_key_context_q = "context_q"
        self.field_key_context_ee_goal_pose = "context_ee_goal_pose"
        self.field_key_context_combined = "context_combined"
        self.field_key_task_id = "case_id"

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

        if "accepted" in df.columns:
            df = df[df["accepted"] == True].copy()

        if "try_id" in df.columns:
            df = df.sort_values(["case_id", "try_id"])
            df = df.groupby("case_id").head(1)

        self.by_case = df.set_index("case_id", drop=False)

    def __len__(self):
        return len(self.files)

    def unnormalize_control_points(self, control_points_normalized):
        # In this dataset, normalized == raw (kept for API compatibility)
        return control_points_normalized

    def build_context(self, data_sample: dict):
        """
        Keys for context_model:
          - qs_normalized
          - context_sg_xyz_normalized (preferred explicit key)
          - sg_xyz_normalized (backward compatible)
          - ee_goal_pose_normalized (optional)
        """
        context_d = {
            "qs_normalized": data_sample["qs_normalized"],
        }

        # Prefer explicit key if present
        if "context_sg_xyz_normalized" in data_sample:
            context_d["context_sg_xyz_normalized"] = data_sample["context_sg_xyz_normalized"]
            # also provide sg_xyz_normalized for models expecting it
            context_d["sg_xyz_normalized"] = data_sample["context_sg_xyz_normalized"]
        elif "sg_xyz_normalized" in data_sample:
            context_d["sg_xyz_normalized"] = data_sample["sg_xyz_normalized"]

        if "ee_goal_pose_normalized" in data_sample:
            context_d["ee_goal_pose_normalized"] = data_sample["ee_goal_pose_normalized"]
        return context_d

    def render(self, *args, **kwargs):
        return

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        base = os.path.basename(npz_path)
        case_id = int(base.split("_")[1].split(".")[0])

        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.keys())

        row = self.by_case.loc[case_id] if case_id in self.by_case.index else None

        def _row_get(k, default):
            if row is None:
                return default
            if k not in row.index:
                return default
            v = row[k]
            if pd.isna(v):
                return default
            return v

        if row is not None and all(k in row.index for k in ["grasp_x", "grasp_y", "grasp_z"]):
            grasp = np.array([_row_get("grasp_x", 0.0), _row_get("grasp_y", 0.0), _row_get("grasp_z", 0.0)], dtype=np.float32)
        elif "q_grasp_state" in keys:
            grasp = np.asarray(data["q_grasp_state"], dtype=np.float32).reshape(-1)[:3]
        elif "q_grasp" in keys:
            grasp = np.asarray(data["q_grasp"], dtype=np.float32).reshape(-1)[:3]
        else:
            grasp = np.zeros(3, dtype=np.float32)

        if row is not None and all(k in row.index for k in ["goal_x", "goal_y", "goal_z"]):
            goal_xyz = np.array([_row_get("goal_x", 0.0), _row_get("goal_y", 0.0), _row_get("goal_z", 0.0)], dtype=np.float32)
        elif "q_goal" in keys:
            goal_xyz = np.asarray(data["q_goal"], dtype=np.float32).reshape(-1)[:3]
        else:
            goal_xyz = np.zeros(3, dtype=np.float32)

        dt = float(_row_get("dt", 1.0))
        t_grasp = float(_row_get("to_grasp", float(np.asarray(data["t_grasp"]).reshape(-1)[0]) if "t_grasp" in keys else 0.0))

        if "states" in keys:
            states = data["states"].astype(np.float32)
        elif "traj" in keys:
            states = np.asarray(data["traj"], dtype=np.float32)
            if states.ndim != 2 or states.shape[1] < 9:
                raise KeyError(f"'traj' has invalid shape in {npz_path}: {states.shape}. Keys={list(data.keys())}")
            states = states[:, :9]
        else:
            raise KeyError(f"'states'/'traj' missing in {npz_path}. Keys={list(data.keys())}")
        T_npz = int(states.shape[0])

        if "ee_positions" in keys:
            ee_positions = np.asarray(data["ee_positions"], dtype=np.float32)
        elif "ee_traj" in keys:
            ee_positions = np.asarray(data["ee_traj"], dtype=np.float32)
        elif "traj" in keys:
            ee_positions = np.asarray(data["traj"], dtype=np.float32)[:, :3]
        else:
            raise KeyError(f"'ee_positions'/'ee_traj' missing in {npz_path}. Keys={list(data.keys())}")
        if ee_positions.ndim != 2 or ee_positions.shape[1] < 3:
            raise KeyError(f"invalid ee trajectory shape in {npz_path}: {ee_positions.shape}")
        ee_positions = ee_positions[:, :3].astype(np.float32)

        controls = None
        if "controls" in data:
            controls = data["controls"].astype(np.float32)

        # grasp index (in resampled H grid)
        if ("t_grasp" in keys) and (not np.isfinite(dt) or dt <= 0.0):
            k_grasp_npz = int(round(float(np.asarray(data["t_grasp"]).reshape(-1)[0])))
        else:
            k_grasp_npz = int(round(t_grasp / max(dt, 1e-9)))
        k_grasp_npz = int(np.clip(k_grasp_npz, 0, T_npz - 1))

        if T_npz <= 1:
            k_grasp = 0
        else:
            k_grasp = int(round(k_grasp_npz * (self.H - 1) / (T_npz - 1)))
        k_grasp = int(np.clip(k_grasp, 0, self.H - 1))
        k_grasp_norm = np.array([k_grasp / (self.H - 1)], dtype=np.float32)

        # build raw trajectory [x,y,z,qx,qy,qz,qw,j1,j2]
        traj_raw = np.concatenate([states[:, 0:3], states[:, 3:7], states[:, 7:9]], axis=1).astype(np.float32)
        traj = resample_linear(traj_raw, self.H)

        ee_positions_resampled = resample_linear(ee_positions.astype(np.float32), self.H)

        # ensure exact length H
        T = traj.shape[0]
        H = self.H
        if T < H:
            pad = np.repeat(traj[-1:, :], H - T, axis=0)
            traj = np.concatenate([traj, pad], axis=0)
        elif T > H:
            traj = traj[:H]

        # start/goal states for conditioning
        start_state = traj[0].copy().astype(np.float32)
        goal_state = traj[-1].copy().astype(np.float32)

        # IMPORTANT: make GT trajectory end consistent with goal_xyz conditioning
        # (otherwise diffusion loss pulls toward traj[-1] while aux pulls toward goal_xyz -> conflict)
        traj[-1, 0:3] = goal_xyz
        goal_state[0:3] = goal_xyz

        cond = np.concatenate([start_state, goal_state, grasp, k_grasp_norm], axis=0).astype(np.float32)
        if cond.shape[0] != self.cond_dim:
            raise RuntimeError(f"cond dim mismatch: got {cond.shape[0]} expected {self.cond_dim}")

        # Explicit start/goal xyz conditioning
        # start xyz from start_state (traj[0])
        start_xyz = start_state[0:3].copy()
        sg_xyz = np.concatenate([start_xyz, goal_xyz], axis=0).astype(np.float32)

        out = {
            "traj": torch.from_numpy(traj),
            "cond": torch.from_numpy(cond),

            "control_points": torch.from_numpy(traj),
            "context_q": torch.from_numpy(cond),
            "context_ee_goal_pose": torch.zeros(1, dtype=torch.float32),
            "context_combined": torch.from_numpy(cond),
            "case_id": torch.tensor(case_id, dtype=torch.long),

            "control_points_normalized": torch.from_numpy(traj),
            "context_q_normalized": torch.from_numpy(cond),
            "context_ee_goal_pose_normalized": torch.zeros(1, dtype=torch.float32),
            "context_combined_normalized": torch.from_numpy(cond),

            # NEW: explicit sg keys (preferred by models)
            "context_sg_xyz": torch.from_numpy(sg_xyz),
            "context_sg_xyz_normalized": torch.from_numpy(sg_xyz),

            # Backward compatible key (some code expects this exact name)
            "sg_xyz_normalized": torch.from_numpy(sg_xyz),

            "states": torch.from_numpy(states),
            "ee_positions": torch.from_numpy(ee_positions),
            "ee_positions_resampled": torch.from_numpy(ee_positions_resampled),
        }

        if controls is not None:
            out["controls"] = torch.from_numpy(controls)

        # alias expected by other parts of MPD code
        out["qs"] = out["context_q"]
        out["qs_normalized"] = out["context_q_normalized"]

        out["ee_goal_pose"] = out["context_ee_goal_pose"]
        out["ee_goal_pose_normalized"] = out["context_ee_goal_pose_normalized"]

        # convenience slices (documented layout)
        out["q_start"] = out["context_q"][0:9].clone()
        out["q_goal"] = out["context_q"][9:18].clone()
        out["q_grasp"] = out["context_q"][18:21].clone()
        out["k_grasp_norm"] = out["context_q"][21:22].clone()

        out["hard_conds"] = {}

        for k, v in list(out.items()):
            if isinstance(v, (int, float, bool)):
                out[k] = torch.tensor(v)

        return out
