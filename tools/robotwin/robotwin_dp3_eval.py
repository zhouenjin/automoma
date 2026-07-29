from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = REPO_ROOT / "third_party" / "RoboTwin" / "policy" / "DP3"
DP3_DIFFUSION_ROOT = DP3_ROOT / "3D-Diffusion-Policy"
ISAAC_ENV_ROOT = REPO_ROOT / "third_party" / "IsaacLab-Arena" / "isaaclab-arena-envs"
ISAACLAB_ARENA_ROOT = REPO_ROOT / "third_party" / "IsaacLab-Arena"
LEROBOT_SRC_ROOT = REPO_ROOT / "third_party" / "lerobot" / "src"
TOOLS_EVAL_ROOT = REPO_ROOT / "tools" / "eval"

for path in (
    str(TOOLS_EVAL_ROOT),
    str(DP3_ROOT),
    str(DP3_ROOT / "scripts"),
    str(DP3_DIFFUSION_ROOT),
    str(ISAACLAB_ARENA_ROOT),
    str(ISAAC_ENV_ROOT),
    str(LEROBOT_SRC_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from automoma_dp3_utils import PointCloudConfig, rgbd_to_pointcloud
from ee_pose_trace import EE_TRACE_COLUMNS, ee_trace_values, empty_ee_trace_values, make_ee_fk
from isaaclab_arena.scripts.automoma_replay_common import make_execution_disturbance
from isaaclab_arena.utils.action_interpolation import interpolate_actions
from train_dp3 import TrainDP3Workspace


PER_EPISODE_CSV_COLUMNS = [
    "episode_ix",
    "seed",
    "success",
    "final_door_open",
    "final_door_openness",
    "final_engaged",
    "final_handle_distance",
    "steps",
    "video_path",
]

ACTION_TRACE_CSV_COLUMNS = [
    "episode_ix",
    "policy_step_ix",
    "terminated",
    "truncated",
    "joint_ix",
    "joint_name",
    "raw_policy_action",
    "prepared_action_abs",
    "sim_joint_pos_before",
    "sim_joint_pos_after",
    "raw_policy_step_delta_abs",
    "prepared_action_step_delta_abs",
    "sim_joint_step_delta_abs",
    *EE_TRACE_COLUMNS,
]

EPISODE_TRACE_CSV_COLUMNS = [
    "episode_ix",
    "step",
    "terminated",
    "truncated",
    "openness",
    "door_open",
    "handle_distance",
    "engaged",
    "sim_base_x",
    "sim_base_y",
    "sim_base_yaw",
    "action_base_x",
    "action_base_y",
    "action_base_yaw",
    "raw_policy_base_x",
    "raw_policy_base_y",
    "raw_policy_base_yaw",
]

SUMMIT_FRANKA_ACTION_JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)

CUAKR_SET_OBJECT_OPEN_TARGET = 1.57
CUAKR_SET_OBJECT_DRIVE_TYPE = "acceleration"
CUAKR_SET_OBJECT_DRIVE_DAMPING = 0.1
CUAKR_SET_OBJECT_DRIVE_STIFFNESS = 0.5
CUAKR_SET_OBJECT_DRIVE_MAX_FORCE = 0.05
CUAKR_SET_SIM_STEPS_PER_ACTION = 10
DEFAULT_SET_SIM_STEPS_PER_ACTION = 1
DEFAULT_SET_OBJECT_OPEN_VELOCITY = 20.0
DEFAULT_SET_OBJECT_DRIVE_STIFFNESS = 12.0
DEFAULT_SET_OBJECT_DRIVE_DAMPING = 0.05
DEFAULT_SET_OBJECT_DRIVE_MAX_FORCE = 8.0
DEFAULT_SET_OBJECT_OPEN_EFFORT = 0.0
DEFAULT_SET_OBJECT_WRITE_VELOCITY_STATE = True
DEFAULT_SET_ROBOT_ACTION_ALPHA = 1.0
SET_ROBOT_FILTERED_JOINT_COUNT = 10


def load_module(module_name: str, file_path: Path):
    spec = __import__("importlib.util").util.spec_from_file_location(module_name, file_path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


env_module = load_module("automoma_isaac_env", ISAAC_ENV_ROOT / "env.py")

CAMERA_CHOICES = ("ego_topdown", "ego_wrist", "fix_local")


def append_per_episode_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_EPISODE_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PER_EPISODE_CSV_COLUMNS})


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class ActionTraceLogger:
    def __init__(self, csv_path: Path, joint_names: list[str]) -> None:
        self.csv_path = csv_path
        self.joint_names = joint_names
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=ACTION_TRACE_CSV_COLUMNS)
        self._writer.writeheader()
        self.ee_fk = make_ee_fk(joint_names)
        self._last_raw_policy_action: dict[int, np.ndarray] = {}
        self._last_prepared_action: dict[int, np.ndarray] = {}
        self._last_sim_joint_pos_after: dict[int, np.ndarray] = {}
        self.max_raw_policy_step_delta_abs = 0.0
        self.max_prepared_action_step_delta_abs = 0.0
        self.max_sim_joint_step_delta_abs = 0.0
        self.max_raw_policy_base_step_delta_abs = 0.0
        self.max_prepared_action_base_step_delta_abs = 0.0
        self.max_sim_base_step_delta_abs = 0.0

    def close(self) -> None:
        self._file.close()

    def write_step(
        self,
        *,
        episode_ix: int,
        policy_step_ix: int,
        terminated: bool,
        truncated: bool,
        raw_policy_action: np.ndarray,
        prepared_action_abs: np.ndarray,
        sim_joint_pos_before: np.ndarray,
        sim_joint_pos_after: np.ndarray,
    ) -> None:
        width = min(
            len(self.joint_names),
            int(raw_policy_action.shape[0]),
            int(prepared_action_abs.shape[0]),
            int(sim_joint_pos_before.shape[0]),
            int(sim_joint_pos_after.shape[0]),
        )
        prev_raw = self._last_raw_policy_action.get(episode_ix)
        prev_prepared = self._last_prepared_action.get(episode_ix)
        prev_sim = self._last_sim_joint_pos_after.get(episode_ix)
        raw_delta = (
            np.abs(raw_policy_action[:width] - prev_raw[:width])
            if prev_raw is not None and prev_raw.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        prepared_delta = (
            np.abs(prepared_action_abs[:width] - prev_prepared[:width])
            if prev_prepared is not None and prev_prepared.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        sim_delta = (
            np.abs(sim_joint_pos_after[:width] - prev_sim[:width])
            if prev_sim is not None and prev_sim.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        if not (terminated or truncated):
            self._update_delta_summary(raw_delta, prepared_delta, sim_delta)
        ee_values = (
            empty_ee_trace_values()
            if terminated or truncated
            else ee_trace_values(self.ee_fk, prepared_action_abs[:width], sim_joint_pos_after[:width])
        )
        for joint_ix in range(width):
            self._writer.writerow(
                {
                    "episode_ix": episode_ix,
                    "policy_step_ix": policy_step_ix,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "joint_ix": joint_ix,
                    "joint_name": self.joint_names[joint_ix],
                    "raw_policy_action": float(raw_policy_action[joint_ix]),
                    "prepared_action_abs": float(prepared_action_abs[joint_ix]),
                    "sim_joint_pos_before": float(sim_joint_pos_before[joint_ix]),
                    "sim_joint_pos_after": float(sim_joint_pos_after[joint_ix]),
                    "raw_policy_step_delta_abs": "" if np.isnan(raw_delta[joint_ix]) else float(raw_delta[joint_ix]),
                    "prepared_action_step_delta_abs": ""
                    if np.isnan(prepared_delta[joint_ix])
                    else float(prepared_delta[joint_ix]),
                    "sim_joint_step_delta_abs": "" if np.isnan(sim_delta[joint_ix]) else float(sim_delta[joint_ix]),
                    **ee_values,
                }
            )
        self._last_raw_policy_action[episode_ix] = raw_policy_action.copy()
        self._last_prepared_action[episode_ix] = prepared_action_abs.copy()
        self._last_sim_joint_pos_after[episode_ix] = sim_joint_pos_after.copy()

    def _update_delta_summary(
        self,
        raw_delta: np.ndarray,
        prepared_delta: np.ndarray,
        sim_delta: np.ndarray,
    ) -> None:
        for attr, values in (
            ("max_raw_policy_step_delta_abs", raw_delta),
            ("max_prepared_action_step_delta_abs", prepared_delta),
            ("max_sim_joint_step_delta_abs", sim_delta),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                setattr(self, attr, max(float(getattr(self, attr)), float(np.max(finite))))
        base_width = min(3, raw_delta.shape[0], prepared_delta.shape[0], sim_delta.shape[0])
        if base_width <= 0:
            return
        for attr, values in (
            ("max_raw_policy_base_step_delta_abs", raw_delta[:base_width]),
            ("max_prepared_action_base_step_delta_abs", prepared_delta[:base_width]),
            ("max_sim_base_step_delta_abs", sim_delta[:base_width]),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                setattr(self, attr, max(float(getattr(self, attr)), float(np.max(finite))))

    def summary(self) -> dict[str, float | str]:
        return {
            "csv_path": str(self.csv_path),
            "max_raw_policy_step_delta_abs": self.max_raw_policy_step_delta_abs,
            "max_prepared_action_step_delta_abs": self.max_prepared_action_step_delta_abs,
            "max_sim_joint_step_delta_abs": self.max_sim_joint_step_delta_abs,
            "max_raw_policy_base_step_delta_abs": self.max_raw_policy_base_step_delta_abs,
            "max_prepared_action_base_step_delta_abs": self.max_prepared_action_base_step_delta_abs,
            "max_sim_base_step_delta_abs": self.max_sim_base_step_delta_abs,
        }


class EpisodeTraceLogger:
    def __init__(self, csv_path: Path, openable_object) -> None:
        self.csv_path = csv_path
        self.openable_object = openable_object
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_TRACE_CSV_COLUMNS)
        self._writer.writeheader()

    def close(self) -> None:
        self._file.close()

    def write_step(
        self,
        *,
        env,
        episode_ix: int,
        step: int,
        terminated: bool,
        truncated: bool,
        raw_policy_action: np.ndarray,
        executed_action: np.ndarray,
        sim_joint_pos_after: np.ndarray,
        handle_distance_threshold: float,
    ) -> None:
        diagnostics = get_handle_diagnostics(env, self.openable_object)
        openness = diagnostics["openness"]
        handle_distance = diagnostics["handle_distance"]
        self._writer.writerow(
            {
                "episode_ix": episode_ix,
                "step": step,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "openness": maybe_scalar(openness),
                "door_open": "" if np.isnan(openness) else int(openness >= 0.3),
                "handle_distance": maybe_scalar(handle_distance),
                "engaged": "" if np.isnan(handle_distance) else int(handle_distance <= handle_distance_threshold),
                "sim_base_x": sim_joint_pos_after[0] if sim_joint_pos_after.shape[0] > 0 else "",
                "sim_base_y": sim_joint_pos_after[1] if sim_joint_pos_after.shape[0] > 1 else "",
                "sim_base_yaw": sim_joint_pos_after[2] if sim_joint_pos_after.shape[0] > 2 else "",
                "action_base_x": executed_action[0] if executed_action.shape[0] > 0 else "",
                "action_base_y": executed_action[1] if executed_action.shape[0] > 1 else "",
                "action_base_yaw": executed_action[2] if executed_action.shape[0] > 2 else "",
                "raw_policy_base_x": raw_policy_action[0] if raw_policy_action.shape[0] > 0 else "",
                "raw_policy_base_y": raw_policy_action[1] if raw_policy_action.shape[0] > 1 else "",
                "raw_policy_base_yaw": raw_policy_action[2] if raw_policy_action.shape[0] > 2 else "",
            }
        )


class SimpleEnvConfig:
    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")
        self.embodiment = kwargs.pop("embodiment", "gr1_pink")
        self.object = kwargs.pop("object", "power_drill")
        self.mimic = kwargs.pop("mimic", False)
        self.teleop_device = kwargs.pop("teleop_device", None)
        self.seed = kwargs.pop("seed", 42)
        self.device = kwargs.pop("device", "cuda:0")
        self.disable_fabric = kwargs.pop("disable_fabric", False)
        self.enable_cameras = kwargs.pop("enable_cameras", True)
        self.headless = kwargs.pop("headless", True)
        self.enable_pinocchio = kwargs.pop("enable_pinocchio", True)
        self.episode_length = kwargs.pop("episode_length", 300)
        self.state_dim = kwargs.pop("state_dim", 12)
        self.action_dim = kwargs.pop("action_dim", 12)
        self.camera_height = kwargs.pop("camera_height", 240)
        self.camera_width = kwargs.pop("camera_width", 320)
        self.video = kwargs.pop("video", False)
        self.video_length = kwargs.pop("video_length", 100)
        self.video_interval = kwargs.pop("video_interval", 200)
        self.state_keys = kwargs.pop("state_keys", "joint_pos")
        self.camera_keys = kwargs.pop("camera_keys", "ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb")
        self.task = kwargs.pop("task", "Reach out to the microwave and open it.")
        self.disable_env_checker = kwargs.pop("disable_env_checker", True)
        self.fps = kwargs.pop("fps", 30)
        self.features = kwargs.pop("features", None)
        self.features_map = kwargs.pop("features_map", None)
        self.max_parallel_tasks = kwargs.pop("max_parallel_tasks", 1)
        self.kwargs = kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin DP3 on IsaacLab-Arena AutoMoMa env")
    parser.add_argument("--policy_backend", choices=("dp3", "pi0"), default="dp3")
    parser.add_argument("--pi0_host", default="127.0.0.1")
    parser.add_argument("--pi0_port", type=int, default=8000)
    parser.add_argument("--pi0_prompt", default="open the microwave door")
    parser.add_argument("--pi0_image_key", default="ego_topdown_rgb")
    parser.add_argument("--pi0_wrist_image_key", default="ego_wrist_rgb")
    parser.add_argument("--pi0_right_image_key", default="fix_local_rgb")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--ckpt_setting", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--checkpoint_num", type=int, default=3000)
    parser.add_argument("--camera_view", choices=CAMERA_CHOICES, default="ego_topdown")
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--random_drop_points", type=int, default=5000)
    parser.add_argument("--use_fps", type=str2bool, default=True)
    parser.add_argument("--use_rgb", type=str2bool, default=False)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--eval.n_episodes", dest="n_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--traj_file", type=str, default=None)
    parser.add_argument("--traj_seed", type=int, default=42)
    parser.add_argument("--traj_selection_mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--episode_indices", type=str, default=None)
    parser.add_argument("--env.episode_length", "--max_steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--interpolated", type=int, default=1)
    parser.add_argument("--interpolation_type", default="linear")
    parser.add_argument("--actions_per_inference", type=int, default=None)
    parser.add_argument("--policy_action_dim", type=int, default=12)
    parser.add_argument("--policy_horizon", type=int, default=None)
    parser.add_argument("--policy_n_obs_steps", type=int, default=None)
    parser.add_argument("--policy_n_action_steps", type=int, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--sim_dt", type=float, default=None)
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--env.headless", dest="headless", type=str2bool)
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument("--mobile_base_relative", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--action_execution", choices=("drive", "set"), default="drive")
    parser.add_argument("--set", dest="action_execution", action="store_const", const="set")
    parser.add_argument("--drive", dest="action_execution", action="store_const", const="drive")
    parser.add_argument("--set_object_open_target", type=float, default=None)
    parser.add_argument("--set_object_open_velocity", type=float, default=DEFAULT_SET_OBJECT_OPEN_VELOCITY)
    parser.add_argument("--set_object_drive_type", choices=("acceleration", "force"), default=CUAKR_SET_OBJECT_DRIVE_TYPE)
    parser.add_argument("--set_object_drive_stiffness", type=float, default=DEFAULT_SET_OBJECT_DRIVE_STIFFNESS)
    parser.add_argument("--set_object_drive_damping", type=float, default=DEFAULT_SET_OBJECT_DRIVE_DAMPING)
    parser.add_argument("--set_object_drive_max_force", type=float, default=DEFAULT_SET_OBJECT_DRIVE_MAX_FORCE)
    parser.add_argument("--set_object_open_effort", type=float, default=DEFAULT_SET_OBJECT_OPEN_EFFORT)
    parser.add_argument("--set_object_write_velocity_state", type=str2bool, default=DEFAULT_SET_OBJECT_WRITE_VELOCITY_STATE)
    parser.add_argument("--set_sim_steps_per_action", type=int, default=DEFAULT_SET_SIM_STEPS_PER_ACTION)
    parser.add_argument("--set_robot_action_alpha", type=float, default=DEFAULT_SET_ROBOT_ACTION_ALPHA)
    parser.add_argument("--debug_visualize_handle", type=str2bool, default=False)
    parser.add_argument("--debug_record_handle_diagnostics", type=str2bool, default=False)
    parser.add_argument("--debug_action_trace", type=str2bool, default=False)
    parser.add_argument("--action_trace_csv", type=str, default=None)
    parser.add_argument("--debug_episode_trace_csv", type=str, default=None)
    parser.add_argument("--handle_distance_threshold", type=float, default=0.1)
    parser.add_argument("--robot_object_static_friction", type=float, default=None)
    parser.add_argument("--robot_object_dynamic_friction", type=float, default=None)
    parser.add_argument("--slam_bias_x", type=float, default=0.0)
    parser.add_argument("--slam_bias_y", type=float, default=0.0)
    parser.add_argument("--slam_bias_yaw", type=float, default=0.0)
    parser.add_argument("--slam_bias_ramp_seconds", type=float, default=0.0)
    parser.add_argument("--slam_drift_std_xy", type=float, default=0.0)
    parser.add_argument("--slam_drift_std_yaw", type=float, default=0.0)
    parser.add_argument("--slam_drift_alpha", type=float, default=0.995)
    parser.add_argument("--base_wobble_y_amp", type=float, default=0.0)
    parser.add_argument("--base_wobble_yaw_amp", type=float, default=0.0)
    parser.add_argument("--base_wobble_freq_hz", type=float, default=5.0)
    parser.add_argument("--disturbance_seed", type=int, default=0)
    parser.add_argument("--disturbance_reference_hz", type=float, default=50.0)
    return parser.parse_args()


def make_cfg(args: argparse.Namespace):
    config_path = DP3_DIFFUSION_ROOT / "diffusion_policy_3d" / "config"
    with __import__("hydra").initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        cfg = __import__("hydra").compose(config_name="robot_dp3.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = args.task_name
    cfg.expert_data_num = args.expert_data_num
    cfg.setting = args.ckpt_setting
    cfg.raw_task_name = args.task_name
    cfg.policy.use_pc_color = args.use_rgb
    if args.policy_horizon is not None:
        cfg.horizon = args.policy_horizon
        cfg.policy.horizon = args.policy_horizon
    if args.policy_n_obs_steps is not None:
        cfg.n_obs_steps = args.policy_n_obs_steps
        cfg.dataset_obs_steps = args.policy_n_obs_steps
        cfg.policy.n_obs_steps = args.policy_n_obs_steps
        cfg.task.env_runner.n_obs_steps = args.policy_n_obs_steps
        cfg.task.dataset.pad_before = args.policy_n_obs_steps - 1
    if args.policy_n_action_steps is not None:
        cfg.n_action_steps = args.policy_n_action_steps
        cfg.policy.n_action_steps = args.policy_n_action_steps
        cfg.task.env_runner.n_action_steps = args.policy_n_action_steps
        cfg.task.dataset.pad_after = args.policy_n_action_steps - 1
    cfg.task.shape_meta.obs.agent_pos.shape = [12]
    cfg.task.shape_meta.action.shape = [args.policy_action_dim]
    OmegaConf.set_struct(cfg, True)
    return cfg


def load_policy(cfg, args: argparse.Namespace):
    workspace = TrainDP3Workspace(cfg)
    usr_args = {
        "task_name": args.task_name,
        "ckpt_setting": args.ckpt_setting,
        "expert_data_num": args.expert_data_num,
        "seed": args.seed,
        "checkpoint_num": args.checkpoint_num,
        "output_dir": str(Path(args.checkpoint_root).resolve()) if args.checkpoint_root else None,
    }
    policy, env_runner = workspace.get_policy_and_runner(cfg, usr_args)
    return policy, env_runner


def load_pi0_policy(args: argparse.Namespace):
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exc:
        raise RuntimeError(
            "The Pi0 backend requires openpi-client in the active environment."
        ) from exc

    policy = WebsocketClientPolicy(args.pi0_host, args.pi0_port)
    metadata = policy.get_server_metadata()
    print(f"[pi0] connected to {args.pi0_host}:{args.pi0_port} metadata={metadata}", flush=True)
    return policy


def parse_env_identifiers(task_name: str, task_config: str) -> tuple[str, str]:
    parts = task_config.split("-")
    if len(parts) >= 3:
        object_name = parts[0]
        scene_name = parts[1]
        return object_name, scene_name
    if len(parts) == 2:
        object_name, scene_name = parts
        return object_name, scene_name
    return task_name, task_config


def resolve_set_object_open_target(args: argparse.Namespace) -> tuple[float, str]:
    if args.set_object_open_target is not None:
        return float(args.set_object_open_target), "cli"
    return CUAKR_SET_OBJECT_OPEN_TARGET, "cuakr_default"


def build_env(args: argparse.Namespace):
    object_name, scene_name = parse_env_identifiers(args.task_name, args.task_config)
    traj_file = str(Path(args.traj_file)) if args.traj_file else None
    cfg = SimpleEnvConfig(
        environment="summit_franka_open_door_eval",
        headless=args.headless,
        enable_cameras=True,
        state_keys="joint_pos",
        camera_keys="ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb",
        state_dim=12,
        action_dim=12,
        camera_height=240,
        camera_width=320,
        episode_length=args.max_steps,
        object_name=object_name,
        scene_name=scene_name,
        object_center=True,
        mobile_base_relative=args.mobile_base_relative,
        traj_file=traj_file,
        traj_seed=args.traj_seed,
        traj_selection_mode=args.traj_selection_mode,
        interpolated=args.interpolated,
        interpolation_type=args.interpolation_type,
        decimation=args.decimation,
        sim_dt=args.sim_dt,
        openness_threshold=0.3,
        handle_distance_threshold=args.handle_distance_threshold,
        proximity_threshold=0.12,
        proximity_window_steps=8,
        proximity_required_steps=5,
        disable_fingertip_proximity=False,
        debug_visualize_handle=args.debug_visualize_handle,
        debug_record_handle_diagnostics=args.debug_record_handle_diagnostics,
        debug_marker_scale=1.0,
        robot_object_static_friction=args.robot_object_static_friction,
        robot_object_dynamic_friction=args.robot_object_dynamic_friction,
    )
    env_map = env_module.make_env(n_envs=1, cfg=cfg)
    return env_map["summit_franka_open_door_eval"][0]


def extract_point_cloud(obs: dict, camera_view: str, pc_cfg: PointCloudConfig, rng: np.random.Generator) -> np.ndarray:
    rgb = obs["camera_obs"][f"{camera_view}_rgb"]
    depth = obs["camera_obs"][f"{camera_view}_depth"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    rgb = rgb[0] if rgb.ndim == 4 else rgb
    depth = depth[0] if depth.ndim == 4 else depth
    return rgbd_to_pointcloud(rgb, depth, pc_cfg, rng)


def extract_joint_pos(obs: dict) -> np.ndarray:
    joint_pos = obs["policy"]["joint_pos"]
    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.detach().cpu().numpy()
    return joint_pos[0].astype(np.float32) if joint_pos.ndim == 2 else joint_pos.astype(np.float32)


def extract_rgb_image(obs: dict, key: str) -> np.ndarray:
    image = obs["camera_obs"][key]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {3, 4} and image.shape[-1] not in {3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def make_pi0_observation(obs: dict, args: argparse.Namespace) -> dict:
    return {
        "observation/image": extract_rgb_image(obs, args.pi0_image_key),
        "observation/wrist_image": extract_rgb_image(obs, args.pi0_wrist_image_key),
        "observation/right_image": extract_rgb_image(obs, args.pi0_right_image_key),
        "observation/state": extract_joint_pos(obs).astype(np.float32),
        "prompt": args.pi0_prompt,
    }


def tensor_action_to_numpy(action: torch.Tensor) -> np.ndarray:
    action_np = action.detach().cpu().numpy()
    return action_np[0].astype(np.float32) if action_np.ndim == 2 else action_np.astype(np.float32)


def expand_policy_action_for_env(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 11:
        return np.concatenate([action[:10], action[10:11], action[10:11]]).astype(np.float32)
    return action


def prepare_env_action(env, action: np.ndarray) -> tuple[torch.Tensor | np.ndarray, np.ndarray]:
    action = expand_policy_action_for_env(action)
    batched_action = action[None, :]
    if hasattr(env, "prepare_action"):
        prepared_action = env.prepare_action(batched_action)
    else:
        prepared_action = batched_action
    if isinstance(prepared_action, torch.Tensor):
        prepared_action_np = tensor_action_to_numpy(prepared_action)
    else:
        prepared_action_np = prepared_action[0].astype(np.float32) if prepared_action.ndim == 2 else prepared_action.astype(np.float32)
    return prepared_action, prepared_action_np


def prepared_action_like(reference_action, action: np.ndarray) -> torch.Tensor | np.ndarray:
    batched_action = action[None, :].astype(np.float32)
    if isinstance(reference_action, torch.Tensor):
        return torch.as_tensor(batched_action, dtype=reference_action.dtype, device=reference_action.device)
    return batched_action


def apply_execution_disturbance(prepared_action, disturbance, step: int):
    if disturbance is None:
        return prepared_action, tensor_action_to_numpy(prepared_action) if isinstance(prepared_action, torch.Tensor) else prepared_action[0]
    if isinstance(prepared_action, torch.Tensor):
        disturbed = disturbance.apply(prepared_action, step)
        return disturbed, tensor_action_to_numpy(disturbed)
    disturbed = disturbance.apply(torch.as_tensor(prepared_action, dtype=torch.float32), step)
    return disturbed.numpy(), tensor_action_to_numpy(disturbed)


def get_env_step_dt(env, fallback: float = 0.02) -> float:
    raw_env = getattr(env, "_env", None)
    return float(getattr(raw_env, "step_dt", fallback) or fallback)


def resolve_openable_object(env):
    raw_env = getattr(env, "_env", None)
    arena_env = getattr(getattr(raw_env, "cfg", None), "isaaclab_arena_env", None)
    return getattr(getattr(arena_env, "task", None), "openable_object", None)


def _scalar_from_tensor(value, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0]) if value.size else default
    return float(value)


def get_handle_diagnostics(env, openable_object) -> dict[str, float]:
    if openable_object is None:
        return {"openness": np.nan, "handle_distance": np.nan}
    raw_env = getattr(env, "_env", None)
    try:
        from isaaclab_arena.metrics.handle_proximity_rate import get_cached_handle_proximity_diagnostics

        diagnostics = get_cached_handle_proximity_diagnostics(raw_env, openable_object)
    except Exception:
        diagnostics = None
    if diagnostics is None:
        return {"openness": np.nan, "handle_distance": np.nan}
    return {
        "openness": _scalar_from_tensor(diagnostics.get("openness")),
        "handle_distance": _scalar_from_tensor(diagnostics.get("handle_distance")),
    }


def slow_set_robot_action(current_joint_pos: np.ndarray, target_action: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return target_action.copy()

    slowed_action = target_action.copy()
    width = min(SET_ROBOT_FILTERED_JOINT_COUNT, int(current_joint_pos.shape[0]), int(target_action.shape[0]))
    slowed_action[:width] = current_joint_pos[:width] + alpha * (target_action[:width] - current_joint_pos[:width])
    return slowed_action


def interpolate_prepared_actions_from_current(
    current_joint_pos: np.ndarray,
    target_action: np.ndarray,
    interpolation_factor: int,
    interpolation_type: str,
) -> list[np.ndarray]:
    if interpolation_factor <= 1 or interpolation_type == "none":
        return [target_action.astype(np.float32, copy=True)]

    start = target_action.astype(np.float32, copy=True)
    width = min(start.shape[0], current_joint_pos.shape[0])
    start[:width] = current_joint_pos[:width]
    expanded = interpolate_actions(
        torch.as_tensor(start, dtype=torch.float32),
        torch.as_tensor(target_action, dtype=torch.float32),
        interpolation_factor,
        interpolation_type,
        include_start=False,
    )
    return [expanded[i].cpu().numpy().astype(np.float32) for i in range(expanded.shape[0])]


def set_robot_joint_state(env, action: np.ndarray) -> None:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is None or scene is None or "robot" not in scene.keys():
        raise RuntimeError("Could not access IsaacLab robot for set action execution.")

    robot = scene["robot"]
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    action_tensor = torch.as_tensor(action, dtype=joint_pos.dtype, device=joint_pos.device).reshape(1, -1)

    assigned = False
    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is not None:
        name_to_ix = {name: ix for ix, name in enumerate(joint_names)}
        if all(name in name_to_ix for name in SUMMIT_FRANKA_ACTION_JOINT_NAMES):
            joint_ids = [name_to_ix[name] for name in SUMMIT_FRANKA_ACTION_JOINT_NAMES]
            width = min(len(joint_ids), int(action_tensor.shape[1]))
            joint_pos[:, joint_ids[:width]] = action_tensor[:, :width]
            assigned = True

    if not assigned:
        width = min(int(action_tensor.shape[1]), int(joint_pos.shape[1]))
        joint_pos[:, :width] = action_tensor[:, :width]

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(joint_pos)
    if hasattr(robot, "set_joint_velocity_target"):
        robot.set_joint_velocity_target(joint_vel)
    scene.write_data_to_sim()


def scene_articulation_for_object(env):
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if scene is None:
        return None
    for key in scene.keys():
        if key == "robot":
            continue
        entity = scene[key]
        if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
            return entity
    return None


def configure_cuakr_object_drive(
    env,
    *,
    drive_type: str,
    stiffness: float,
    damping: float,
    max_force: float,
) -> dict[str, object]:
    object_entity = scene_articulation_for_object(env)
    if object_entity is None:
        raise RuntimeError("Could not access IsaacLab object articulation for set action execution.")
    if int(getattr(object_entity, "num_joints", 0)) < 1:
        raise RuntimeError("IsaacLab object articulation has no joints to drive during set action execution.")

    if hasattr(object_entity, "write_joint_stiffness_to_sim"):
        object_entity.write_joint_stiffness_to_sim(stiffness, joint_ids=[0])
    if hasattr(object_entity, "write_joint_damping_to_sim"):
        object_entity.write_joint_damping_to_sim(damping, joint_ids=[0])
    if hasattr(object_entity, "write_joint_effort_limit_to_sim"):
        object_entity.write_joint_effort_limit_to_sim(max_force, joint_ids=[0])

    patched_joint_path = ""
    try:
        from pxr import UsdPhysics

        root_physx_view = getattr(object_entity, "root_physx_view", None)
        stage = getattr(object_entity, "stage", None)
        dof_paths = getattr(root_physx_view, "dof_paths", None)
        if stage is not None and dof_paths:
            patched_joint_path = str(dof_paths[0][0])
            joint_prim = stage.GetPrimAtPath(patched_joint_path)
            drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
            if not drive_api.GetPrim().IsValid():
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive_api.GetTypeAttr().Set(drive_type)
            drive_api.GetDampingAttr().Set(damping)
            drive_api.GetStiffnessAttr().Set(stiffness)
            drive_api.GetMaxForceAttr().Set(max_force)
    except Exception as exc:
        print(f"Warning: failed to patch object USD DriveAPI for cuakr set mode: {exc}", file=sys.stderr)

    return {
        "type": drive_type,
        "damping": damping,
        "stiffness": stiffness,
        "max_force": max_force,
        "joint_path": patched_joint_path,
    }


def apply_object_open_target(
    env,
    target: float,
    velocity: float,
    effort: float,
) -> None:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    object_entity = scene_articulation_for_object(env)
    if raw_env is None or scene is None or object_entity is None:
        raise RuntimeError("Could not access IsaacLab object articulation for set action execution.")
    if int(getattr(object_entity, "num_joints", 0)) < 1:
        raise RuntimeError("IsaacLab object articulation has no joints to open during set action execution.")

    joint_pos_target = object_entity.data.joint_pos.clone()
    joint_vel_target = torch.zeros_like(joint_pos_target)
    joint_effort_target = torch.zeros_like(joint_pos_target)
    joint_pos_target[:, 0] = target
    joint_vel_target[:, 0] = velocity
    joint_effort_target[:, 0] = effort

    if hasattr(object_entity, "set_joint_position_target"):
        object_entity.set_joint_position_target(joint_pos_target)
    if hasattr(object_entity, "set_joint_velocity_target"):
        object_entity.set_joint_velocity_target(joint_vel_target)
    if effort and hasattr(object_entity, "set_joint_effort_target"):
        object_entity.set_joint_effort_target(joint_effort_target)

    root_physx_view = getattr(object_entity, "root_physx_view", None)
    if root_physx_view is not None:
        env_indices = getattr(object_entity, "_ALL_INDICES", None)
        root_physx_view.set_dof_position_targets(joint_pos_target, env_indices)
        root_physx_view.set_dof_velocity_targets(joint_vel_target, env_indices)
        if effort:
            root_physx_view.set_dof_actuation_forces(joint_effort_target, env_indices)
    else:
        scene.write_data_to_sim()


def command_object_open_motor(
    env,
    target: float,
    velocity: float,
    effort: float,
    write_velocity_state: bool,
) -> None:
    object_entity = scene_articulation_for_object(env)
    if object_entity is None or int(getattr(object_entity, "num_joints", 0)) < 1:
        return

    joint_pos_target = object_entity.data.joint_pos.clone()
    joint_vel_target = torch.zeros_like(joint_pos_target)
    joint_effort_target = torch.zeros_like(joint_pos_target)
    joint_pos_target[:, 0] = target
    joint_vel_target[:, 0] = velocity
    joint_effort_target[:, 0] = effort

    root_physx_view = getattr(object_entity, "root_physx_view", None)
    if root_physx_view is not None:
        env_indices = getattr(object_entity, "_ALL_INDICES", None)
        root_physx_view.set_dof_position_targets(joint_pos_target, env_indices)
        root_physx_view.set_dof_velocity_targets(joint_vel_target, env_indices)
        if effort:
            root_physx_view.set_dof_actuation_forces(joint_effort_target, env_indices)

    if write_velocity_state and velocity:
        joint_vel_state = object_entity.data.joint_vel.clone()
        current = object_entity.data.joint_pos[:, 0]
        target_tensor = torch.full_like(current, target)
        remaining = target_tensor - current
        drive_velocity = torch.where(
            torch.abs(remaining) > 1e-4,
            torch.sign(remaining) * abs(float(velocity)),
            torch.zeros_like(remaining),
        )
        joint_vel_state[:, 0] = drive_velocity
        object_entity.write_joint_velocity_to_sim(joint_vel_state)


def install_object_open_velocity_hook(
    env,
    target: float,
    velocity: float,
    effort: float,
    write_velocity_state: bool,
    enabled: bool,
) -> dict[str, object]:
    raw_env = getattr(env, "_env", None)
    sim = getattr(raw_env, "sim", None)
    if raw_env is None or sim is None or not enabled or velocity <= 0.0:
        return {"enabled": False, "timing": "disabled"}

    object_entity = scene_articulation_for_object(env)
    if object_entity is None:
        raise RuntimeError("Could not access IsaacLab object articulation for set object velocity hook.")

    hook_config = {
        "enabled": True,
        "target": float(target),
        "velocity": float(velocity),
        "effort": float(effort),
        "write_velocity_state": bool(write_velocity_state),
    }
    setattr(sim, "_automoma_object_open_velocity_hook_config", hook_config)
    if not getattr(sim, "_automoma_object_open_velocity_hook_installed", False):
        original_step = sim.step

        def hooked_step(*args, **kwargs):
            config = getattr(sim, "_automoma_object_open_velocity_hook_config", None)
            if config and config.get("enabled", False):
                command_object_open_motor(
                    env,
                    config["target"],
                    config["velocity"],
                    config["effort"],
                    config["write_velocity_state"],
                )
            return original_step(*args, **kwargs)

        setattr(sim, "_automoma_object_open_velocity_original_step", original_step)
        sim.step = hooked_step
        setattr(sim, "_automoma_object_open_velocity_hook_installed", True)

    return {
        "enabled": True,
        "timing": "before_each_physx_step_after_scene_write",
        "target": float(target),
        "velocity": float(velocity),
        "effort": float(effort),
        "write_velocity_state": bool(write_velocity_state),
    }


def step_prepared_env_action(env, prepared_action):
    if hasattr(env, "step_prepared_action"):
        return env.step_prepared_action(prepared_action)
    return env.step(prepared_action)


def get_success_metrics(final_info: dict) -> dict[str, float | bool]:
    def pick(key, default=np.nan):
        value = final_info.get(key)
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            return value[0].item() if value.size else default
        return value

    return {
        "success": bool(pick("is_success", False)),
        "final_door_openness": float(pick("final_openness")),
        "final_door_open": bool(pick("final_door_open", False)),
        "final_engaged": bool(pick("final_engaged", False)),
        "final_handle_distance": float(pick("final_handle_distance")),
    }


def snapshot_success_metrics(
    env,
    openness_threshold: float,
    handle_distance_threshold: float,
) -> dict[str, float | bool] | None:
    summaries = getattr(env, "_episode_summaries", None)
    if not summaries:
        return None
    summary = summaries[0]
    final_openness = summary.get("final_door_openness")
    final_handle_distance = summary.get("final_handle_distance")
    if final_openness is None or final_handle_distance is None:
        return None
    final_openness = float(final_openness)
    final_handle_distance = float(final_handle_distance)
    final_door_open = final_openness >= openness_threshold
    final_engaged = final_handle_distance <= handle_distance_threshold
    return {
        "success": bool(final_door_open and final_engaged),
        "final_door_openness": final_openness,
        "final_door_open": final_door_open,
        "final_engaged": final_engaged,
        "final_handle_distance": final_handle_distance,
    }


def maybe_scalar(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def render_frame(env) -> np.ndarray | None:
    frame = env.render()
    if isinstance(frame, list):
        if not frame:
            return None
        frame = frame[0]
    if frame is None:
        return None
    return np.asarray(frame)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--env.episode_length/--max_steps must be >= 1.")
    if args.interpolated < 1:
        raise ValueError("--interpolated must be >= 1.")
    if args.decimation is not None and args.decimation < 1:
        raise ValueError("--decimation must be >= 1.")
    if args.sim_dt is not None and args.sim_dt <= 0.0:
        raise ValueError("--sim_dt must be > 0.")
    if args.disturbance_reference_hz <= 0.0:
        raise ValueError("--disturbance_reference_hz must be > 0.")
    if args.actions_per_inference is not None and args.actions_per_inference < 1:
        raise ValueError("--actions_per_inference must be >= 1.")
    if args.set_sim_steps_per_action < 1:
        raise ValueError("--set_sim_steps_per_action must be >= 1.")
    if not 0.0 < args.set_robot_action_alpha <= 1.0:
        raise ValueError("--set_robot_action_alpha must be in (0, 1].")
    if args.set_object_open_velocity < 0.0:
        raise ValueError("--set_object_open_velocity must be >= 0.")
    if args.set_object_drive_stiffness < 0.0:
        raise ValueError("--set_object_drive_stiffness must be >= 0.")
    if args.set_object_drive_damping < 0.0:
        raise ValueError("--set_object_drive_damping must be >= 0.")
    if args.set_object_drive_max_force < 0.0:
        raise ValueError("--set_object_drive_max_force must be >= 0.")
    if args.set_object_open_effort < 0.0:
        raise ValueError("--set_object_open_effort must be >= 0.")
    torch.cuda.set_device(int(args.gpu_id))
    openness_threshold = 0.3
    handle_distance_threshold = float(args.handle_distance_threshold)

    if args.policy_backend == "dp3":
        cfg = make_cfg(args)
        policy, env_runner = load_policy(cfg, args)
    else:
        policy = load_pi0_policy(args)
        env_runner = None
    env = build_env(args)
    step_dt = get_env_step_dt(env)
    disturbance = make_execution_disturbance(args, step_dt)
    if disturbance is not None:
        print(
            "[disturbance] enabled "
            f"dt={step_dt} slam_bias=({args.slam_bias_x}, {args.slam_bias_y}, {args.slam_bias_yaw}) "
            f"drift_std=({args.slam_drift_std_xy}, {args.slam_drift_std_yaw}) "
            f"wobble=({args.base_wobble_y_amp}, {args.base_wobble_yaw_amp}) seed={args.disturbance_seed}",
            flush=True,
        )
    set_object_open_target, set_object_open_target_source = resolve_set_object_open_target(args)
    set_object_drive: dict[str, object] = {}
    set_object_velocity_hook: dict[str, object] = {}
    if args.action_execution == "set":
        set_object_drive = configure_cuakr_object_drive(
            env,
            drive_type=args.set_object_drive_type,
            stiffness=args.set_object_drive_stiffness,
            damping=args.set_object_drive_damping,
            max_force=args.set_object_drive_max_force,
        )
        set_object_velocity_hook = install_object_open_velocity_hook(
            env,
            target=set_object_open_target,
            velocity=args.set_object_open_velocity,
            effort=args.set_object_open_effort,
            write_velocity_state=args.set_object_write_velocity_state,
            enabled=True,
        )

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "eval" / "robotwin" / f"dp3_{args.task_name}-{args.task_config}-{args.expert_data_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_episode_results.csv"
    videos_dir = output_dir / "videos"
    start_time = time.time()

    if csv_path.exists():
        csv_path.unlink()
    action_trace_path = Path(args.action_trace_csv) if args.action_trace_csv else output_dir / "action_trace_joint_states.csv"
    if args.debug_action_trace and action_trace_path.exists():
        action_trace_path.unlink()
    action_trace_logger = (
        ActionTraceLogger(action_trace_path, list(SUMMIT_FRANKA_ACTION_JOINT_NAMES))
        if args.debug_action_trace
        else None
    )
    episode_trace_logger = (
        EpisodeTraceLogger(Path(args.debug_episode_trace_csv), resolve_openable_object(env))
        if args.debug_episode_trace_csv
        else None
    )

    pc_cfg = PointCloudConfig(
        n_points=args.n_points,
        random_drop_points=args.random_drop_points,
        use_fps=args.use_fps,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        fov_deg=args.fov_deg,
        use_rgb=args.use_rgb,
    )
    rng = np.random.default_rng(args.seed)

    all_episode_metrics: list[dict[str, object]] = []
    all_successes: list[bool] = []
    all_seeds: list[int] = []
    all_episode_indices: list[int] = []
    video_paths: list[str] = []
    episode_indices = (
        [int(item) for item in args.episode_indices.split(",") if item.strip()]
        if args.episode_indices
        else list(range(args.n_episodes))
    )

    try:
        for render_ix, episode_ix in enumerate(episode_indices):
            episode_seed = args.seed + episode_ix
            if hasattr(env, "_traj_next_episode") and args.traj_selection_mode == "sequential":
                env._traj_next_episode = episode_ix
            obs, _info = env.reset(seed=episode_seed)
            if env_runner is not None:
                env_runner.reset_obs()
            if disturbance is not None:
                disturbance.begin_episode(episode_ix)

            frames: list[np.ndarray] = []
            if render_ix < args.max_episodes_rendered:
                frame = render_frame(env)
                if frame is not None:
                    frames.append(frame)

            done = False
            final_info = {"is_success": np.array([False])}
            steps = 0
            while not done and steps < args.max_steps:
                if args.policy_backend == "dp3":
                    point_cloud = extract_point_cloud(obs, args.camera_view, pc_cfg, rng)
                    joint_pos = extract_joint_pos(obs)
                    dp3_obs = {
                        "point_cloud": point_cloud.astype(np.float32),
                        "agent_pos": joint_pos.astype(np.float32),
                    }

                    if len(env_runner.obs) == 0:
                        env_runner.update_obs(dp3_obs)
                        actions = env_runner.get_action(policy)
                    else:
                        actions = env_runner.get_action(policy, dp3_obs)
                else:
                    actions = policy.infer(make_pi0_observation(obs, args))["actions"]
                    actions = np.asarray(actions, dtype=np.float32)
                    if actions.ndim == 3:
                        actions = actions[0]
                selected_actions = actions[:args.actions_per_inference] if args.actions_per_inference else actions
                for action in selected_actions:
                    raw_policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
                    target_prepared_action, target_prepared_action_np = prepare_env_action(env, raw_policy_action)
                    execution_targets_np = interpolate_prepared_actions_from_current(
                        extract_joint_pos(obs),
                        target_prepared_action_np,
                        args.interpolated,
                        args.interpolation_type,
                    )
                    repeat_count = (
                        args.set_sim_steps_per_action
                        if args.action_execution == "set" and args.interpolated <= 1
                        else 1
                    )
                    for target_action_np in execution_targets_np:
                        for _ in range(repeat_count):
                            sim_joint_pos_before = extract_joint_pos(obs)
                            if args.action_execution == "set":
                                executed_action_np = slow_set_robot_action(
                                    sim_joint_pos_before,
                                    target_action_np,
                                    args.set_robot_action_alpha,
                                )
                                executed_prepared_action = prepared_action_like(target_prepared_action, executed_action_np)
                                set_robot_joint_state(env, executed_action_np)
                                apply_object_open_target(
                                    env,
                                    target=set_object_open_target,
                                    velocity=args.set_object_open_velocity,
                                    effort=args.set_object_open_effort,
                                )
                            else:
                                executed_action_np = target_action_np
                                executed_prepared_action = prepared_action_like(target_prepared_action, executed_action_np)
                                executed_prepared_action, executed_action_np = apply_execution_disturbance(
                                    executed_prepared_action,
                                    disturbance,
                                    steps,
                                )
                            obs, _reward, terminated, truncated, info = step_prepared_env_action(env, executed_prepared_action)
                            steps += 1
                            done = bool(terminated[0] or truncated[0])
                            final_info = info.get("final_info", final_info)
                            sim_joint_pos_after = extract_joint_pos(obs)
                            if action_trace_logger is not None:
                                action_trace_logger.write_step(
                                    episode_ix=episode_ix,
                                    policy_step_ix=steps - 1,
                                    terminated=bool(terminated[0]),
                                    truncated=bool(truncated[0]),
                                    raw_policy_action=raw_policy_action,
                                    prepared_action_abs=executed_action_np,
                                    sim_joint_pos_before=sim_joint_pos_before,
                                    sim_joint_pos_after=sim_joint_pos_after,
                                )
                            if episode_trace_logger is not None:
                                episode_trace_logger.write_step(
                                    env=env,
                                    episode_ix=episode_ix,
                                    step=steps - 1,
                                    terminated=bool(terminated[0]),
                                    truncated=bool(truncated[0]),
                                    raw_policy_action=raw_policy_action,
                                    executed_action=executed_action_np,
                                    sim_joint_pos_after=sim_joint_pos_after,
                                    handle_distance_threshold=handle_distance_threshold,
                                )
                            if render_ix < args.max_episodes_rendered:
                                frame = render_frame(env)
                                if frame is not None:
                                    frames.append(frame)
                            if done or steps >= args.max_steps:
                                break
                        if done or steps >= args.max_steps:
                            break
                    if done or steps >= args.max_steps:
                        break
                    if args.policy_backend == "dp3":
                        point_cloud = extract_point_cloud(obs, args.camera_view, pc_cfg, rng)
                        joint_pos = extract_joint_pos(obs)
                        env_runner.update_obs(
                            {
                                "point_cloud": point_cloud.astype(np.float32),
                                "agent_pos": joint_pos.astype(np.float32),
                            }
                        )

            metrics = (
                snapshot_success_metrics(env, openness_threshold, handle_distance_threshold)
                or get_success_metrics(final_info)
            )
            if np.isnan(float(metrics["final_door_openness"])):
                metrics = get_success_metrics(final_info)
            video_path = ""
            if render_ix < args.max_episodes_rendered and frames:
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_file = videos_dir / f"eval_episode_{episode_ix}.mp4"
                fps = int(getattr(env, "metadata", {}).get("render_fps", 30))
                write_video(video_file, frames, fps)
                video_path = str(video_file)
                video_paths.append(video_path)

            row = {
                "episode_ix": episode_ix,
                "seed": episode_seed,
                "success": bool(metrics["success"]),
                "final_door_open": maybe_scalar(metrics["final_door_open"]),
                "final_door_openness": maybe_scalar(metrics["final_door_openness"]),
                "final_engaged": maybe_scalar(metrics["final_engaged"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
                "steps": f"{steps}/{args.max_steps}",
                "video_path": video_path,
            }
            append_per_episode_csv_row(csv_path, row)
            print(row, flush=True)
            if disturbance is not None:
                disturbance.end_episode()

            all_episode_metrics.append({
                "final_door_openness": maybe_scalar(metrics["final_door_openness"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
            })
            all_successes.append(bool(metrics["success"]))
            all_seeds.append(episode_seed)
            all_episode_indices.append(episode_ix)

        elapsed = time.time() - start_time
        eval_info = {
            "per_episode": [
                {
                    "episode_ix": episode_ix,
                    **episode_metrics,
                    "success": success,
                    "seed": seed,
                }
                for episode_ix, episode_metrics, success, seed in zip(
                    all_episode_indices, all_episode_metrics, all_successes, all_seeds, strict=True
                )
            ],
            "aggregated": {
                "pc_success": float(np.nanmean(all_successes) * 100) if all_successes else 0.0,
            },
        }
        if video_paths:
            eval_info["video_paths"] = video_paths
        eval_info["alignment"] = {
            "action_execution": args.action_execution,
            "set_object_open_target": set_object_open_target,
            "set_object_open_target_source": set_object_open_target_source,
            "set_object_open_target_requested": maybe_scalar(args.set_object_open_target),
            "set_object_open_velocity": args.set_object_open_velocity,
            "set_object_open_effort": args.set_object_open_effort,
            "set_object_write_velocity_state": args.set_object_write_velocity_state,
            "set_object_velocity_hook": set_object_velocity_hook,
            "set_object_drive": set_object_drive,
            "cuakr_set_object_drive_reference": {
                "type": CUAKR_SET_OBJECT_DRIVE_TYPE,
                "damping": CUAKR_SET_OBJECT_DRIVE_DAMPING,
                "stiffness": CUAKR_SET_OBJECT_DRIVE_STIFFNESS,
                "max_force": CUAKR_SET_OBJECT_DRIVE_MAX_FORCE,
                "open_velocity": 0.3,
                "open_effort": 0.0,
            },
            "set_sim_steps_per_action": args.set_sim_steps_per_action,
            "cuakr_set_sim_steps_reference": CUAKR_SET_SIM_STEPS_PER_ACTION,
            "set_robot_action_alpha": args.set_robot_action_alpha,
            "set_robot_filtered_joint_count": SET_ROBOT_FILTERED_JOINT_COUNT,
            "interpolated": args.interpolated,
            "interpolation_type": args.interpolation_type,
            "actions_per_inference": args.actions_per_inference,
            "decimation": args.decimation,
            "sim_dt": args.sim_dt,
            "env_step_dt": (
                float(args.sim_dt) * int(args.decimation)
                if args.sim_dt is not None and args.decimation is not None
                else None
            ),
            "max_steps": args.max_steps,
            "mobile_base_relative": bool(args.mobile_base_relative),
            "traj_file": str(Path(args.traj_file).resolve()) if args.traj_file else "",
            "traj_seed": args.traj_seed,
            "traj_selection_mode": args.traj_selection_mode,
            "checkpoint_root": str(Path(args.checkpoint_root).resolve()) if args.checkpoint_root else "",
            "checkpoint_num": args.checkpoint_num,
            "task_name": args.task_name,
            "task_config": args.task_config,
            "expert_data_num": args.expert_data_num,
            "ckpt_setting": args.ckpt_setting,
            "debug_action_trace": bool(args.debug_action_trace),
            "action_trace_csv": str(action_trace_path) if args.debug_action_trace else "",
            "debug_episode_trace_csv": str(Path(args.debug_episode_trace_csv)) if args.debug_episode_trace_csv else "",
            "disturbance": {
                "enabled": disturbance is not None,
                "slam_bias_x": args.slam_bias_x,
                "slam_bias_y": args.slam_bias_y,
                "slam_bias_yaw": args.slam_bias_yaw,
                "slam_bias_ramp_seconds": args.slam_bias_ramp_seconds,
                "slam_drift_std_xy": args.slam_drift_std_xy,
                "slam_drift_std_yaw": args.slam_drift_std_yaw,
                "slam_drift_alpha": args.slam_drift_alpha,
                "base_wobble_y_amp": args.base_wobble_y_amp,
                "base_wobble_yaw_amp": args.base_wobble_yaw_amp,
                "base_wobble_freq_hz": args.base_wobble_freq_hz,
                "disturbance_seed": args.disturbance_seed,
                "disturbance_reference_hz": args.disturbance_reference_hz,
                "step_dt": step_dt,
            },
        }
        if action_trace_logger is not None:
            eval_info["action_trace_summary"] = action_trace_logger.summary()

        with (output_dir / "eval_info.json").open("w") as f:
            json.dump(eval_info, f, indent=2)

        print(f"saved eval results to {csv_path}")
    finally:
        if action_trace_logger is not None:
            action_trace_logger.close()
        if episode_trace_logger is not None:
            episode_trace_logger.close()
        env.close()


if __name__ == "__main__":
    main()
