"""Build 62-column AMP corpus for omnidirectional locomotion (forward, lateral,
backward, pivot).

Per frame layout (matches ``AMPLoader.OBS_DIM_WITH_ROOT``):

    [0:22]   joint_pos        (IsaacLab BFS order)
    [22:44]  joint_vel        (IsaacLab BFS order)
    [44:56]  EE_pos in body frame  (left_hand, right_hand, left_foot, right_foot)
    [56:59]  root_lin_vel in body frame
    [59:62]  root_ang_vel in body frame

The two trailing root-velocity channels exist precisely so the discriminator can
penalize gait modes that move the feet without translating / rotating the COM
(see lateral "kick-without-weight-shift" failure mode).

Sources supported per spec entry:

    - pkl      :  GMR-retargeted ``{root_pos, root_rot (xyzw), dof_pos}`` dict
    - legacy   :  motion_visualization-style 56-col json (root_pos[3] +
                  euler_XYZ[3] + dof_pos[22] + root_lin_vel[3] +
                  root_ang_vel[3] + dof_vel[22]; csv joint order)

A spec is a YAML/JSON file or, more practically, declared inline below in
``_DEFAULT_SPECS`` so the build is reproducible.

Usage::

    python scripts/build_amp_corpus.py \
      --output_root=booster_assets/motions/K1/motion_amp_expert/omni \
      --skip kind=lateral

CLI flags let you regenerate a subset by category, by name, or by source kind.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build 62-col AMP corpus from heterogeneous sources.")
parser.add_argument(
    "--output_root",
    type=str,
    default="booster_assets/motions/K1/motion_amp_expert/omni",
    help="Where the per-category AMP txt files are written (under <output_root>/<category>/).",
)
parser.add_argument(
    "--only",
    nargs="*",
    default=None,
    help="Restrict build to clips whose name matches one of these (substring match).",
)
parser.add_argument(
    "--categories",
    nargs="*",
    default=None,
    choices=["forward", "lateral", "backward", "pivot", "kick"],
    help="Restrict build to these categories.",
)
parser.add_argument("--robot", type=str, default="booster_k1", choices=["booster_k1"])
parser.add_argument(
    "--walk_rollout_dir",
    type=str,
    default=None,
    help="If set, ignore the default specs and build ONLY 'walk_policy' clips from every "
    "*.pkl in this dir. These are leg-only walk rollouts (from the K1-Locomotion recorder); "
    "the head/arm DOFs are filled with the fixed standing pose. Output → <output_root>/walk_policy/.",
)
parser.add_argument("--walk_rollout_weight", type=float, default=0.5, help="MotionWeight for walk_policy clips.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG


# pkl dof order (CSV / K1_JOINT_NAMES) → IsaacLab BFS articulation order.
# Mirrors ``replay_amp_txt.reorder`` for booster_k1.
_PKL_TO_LAB_K1 = [
    0,   # AAHead_yaw
    2,   # ALeft_Shoulder_Pitch
    6,   # ARight_Shoulder_Pitch
    10,  # Left_Hip_Pitch
    16,  # Right_Hip_Pitch
    1,   # Head_pitch
    3,   # Left_Shoulder_Roll
    7,   # Right_Shoulder_Roll
    11,  # Left_Hip_Roll
    17,  # Right_Hip_Roll
    4,   # Left_Elbow_Pitch
    8,   # Right_Elbow_Pitch
    12,  # Left_Hip_Yaw
    18,  # Right_Hip_Yaw
    5,   # Left_Elbow_Yaw
    9,   # Right_Elbow_Yaw
    13,  # Left_Knee_Pitch
    19,  # Right_Knee_Pitch
    14,  # Left_Ankle_Pitch
    20,  # Right_Ankle_Pitch
    15,  # Left_Ankle_Roll
    21,  # Right_Ankle_Roll
]


# CSV / K1_JOINT_NAMES order = the *source* joint order that ``_PKL_TO_LAB_K1``
# maps FROM (index i -> the joint named below). Used to place a leg-only walk
# rollout into a full 22-DOF vector before reordering to IsaacLab BFS order.
_CSV_JOINT_ORDER = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# Fixed head+arm pose (rad) held during the gait: the leg-only walk policy keeps
# the upper body at the deploy DEFAULT_ANGLES standing pose. Mirror of
# ``soccer_commands._DEPLOY_DEFAULT_UPPER_BODY`` (kept in sync by hand).
_FIXED_UPPER_BODY = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.3, "Left_Shoulder_Roll": -1.374,
    "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": -1.2,
    "ARight_Shoulder_Pitch": 0.3, "Right_Shoulder_Roll": 1.374,
    "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 1.2,
}


@dataclass(frozen=True)
class MotionSpec:
    name: str           # output filename stem
    category: str       # forward | lateral | backward | pivot | kick | walk_policy
    kind: str           # 'pkl' | 'legacy' | 'rollout'
    source_path: str
    motion_weight: float
    fps: float | None = None  # only used as fallback for legacy txt
    # When False, output is 56-col (no root_lin_vel / root_ang_vel) — matches
    # the env-side ``AMPObsCfg`` used by soccer_kick_amp et al., which only
    # observes joint+EE (no root velocity).
    include_root_vel: bool = True


@dataclass
class _MotionData:
    name: str
    fps: float
    root_pos: np.ndarray         # (N, 3) world
    root_rot_wxyz: np.ndarray    # (N, 4) wxyz
    dof_pos_pkl: np.ndarray      # (N, 22) csv order


# Per-category source specs. Output path = output_root/category/name.txt.
# MotionWeight is conservative and balanced so the discriminator does not get
# dominated by any single category. Tweak after first training run.
_DEFAULT_SPECS: list[MotionSpec] = [
    # --- forward --------------------------------------------------------
    MotionSpec("walk",     "forward", "pkl",
               "booster_assets/motions/K1/motion_visualization/walk.pkl", 0.50),
    MotionSpec("run",      "forward", "pkl",
               "booster_assets/motions/K1/motion_visualization/run.pkl", 0.50),
    MotionSpec("run2walk", "forward", "pkl",
               "booster_assets/motions/K1/motion_visualization/run2walk.pkl", 0.50),
    MotionSpec("walk2run", "forward", "legacy",
               "booster_assets/motions/K1/motion_visualization/walk2run.txt", 0.50,
               fps=30.0),
    # --- lateral --------------------------------------------------------
    MotionSpec("strafe_walk_left",      "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_walk_left.pkl", 0.30),
    MotionSpec("strafe_walk_right",     "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_walk_right.pkl", 0.30),
    MotionSpec("strafe_walk2run_left",  "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_walk2run_left.pkl", 0.30),
    MotionSpec("strafe_walk2run_right", "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_walk2run_right.pkl", 0.30),
    MotionSpec("strafe_run_left",       "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_run_left.pkl", 0.30),
    MotionSpec("strafe_run_right",      "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_run_right.pkl", 0.30),
    MotionSpec("strafe_run2walk_left",  "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_run2walk_left.pkl", 0.30),
    MotionSpec("strafe_run2walk_right", "lateral", "pkl",
               "/workspace/motions_k1_lateral/strafe_run2walk_right.pkl", 0.30),
    # --- backward -------------------------------------------------------
    MotionSpec("backward_walk",     "backward", "pkl",
               "/workspace/motions_k1_backward/backward_walk.pkl", 0.50),
    MotionSpec("backward_walk2run", "backward", "pkl",
               "/workspace/motions_k1_backward/backward_walk2run.pkl", 0.50),
    MotionSpec("backward_run",      "backward", "pkl",
               "/workspace/motions_k1_backward/backward_run.pkl", 0.50),
    MotionSpec("backward_run2walk", "backward", "pkl",
               "/workspace/motions_k1_backward/backward_run2walk.pkl", 0.50),
    # --- pivot ----------------------------------------------------------
    MotionSpec("pivot_left_slow",  "pivot", "pkl",
               "/workspace/motions_k1_pivot/pivot_left_slow.pkl", 0.40),
    MotionSpec("pivot_left_fast",  "pivot", "pkl",
               "/workspace/motions_k1_pivot/pivot_left_fast.pkl", 0.40),
    MotionSpec("pivot_right_slow", "pivot", "pkl",
               "/workspace/motions_k1_pivot/pivot_right_slow.pkl", 0.40),
    MotionSpec("pivot_right_fast", "pivot", "pkl",
               "/workspace/motions_k1_pivot/pivot_right_fast.pkl", 0.40),
    # --- kick (walk + kick clips, K1 csv joint order). 56-col output to
    #     match the env-side AMPObsCfg used by soccer_kick_amp (no root vel).
    MotionSpec("walk_kick",   "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick.pkl",   0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_1", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-1.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_2", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-2.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_3", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-3.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_4", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-4.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_5", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-5.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_6", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-6.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_7", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-7.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_8", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-8.pkl", 0.40,
               include_root_vel=False),
    MotionSpec("walk_kick_9", "kick", "pkl",
               "booster_assets/motions/K1/pkl_walk_kick/walk-kick-9.pkl", 0.40,
               include_root_vel=False),
]


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0)),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class _Numpy2CompatUnpickler(pickle.Unpickler):
    """Unpickler that reads numpy>=2.0 arrays under numpy<2.0.

    numpy 2.0 renamed the private ``numpy.core`` package to ``numpy._core``, so
    a pickle written by numpy>=2.0 (e.g. GMR-retargeter output) references
    ``numpy._core.*`` and fails to load under Isaac Sim's numpy<2.0 with
    ``ModuleNotFoundError: No module named 'numpy._core'``. Rewrite those module
    paths back to the legacy ``numpy.core`` on the fly.
    """

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _pickle_load(path: str):
    """``pickle.load`` that transparently handles numpy>=2.0 pickles."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
    with open(path, "rb") as f:
        return _Numpy2CompatUnpickler(f).load()


def _load_pkl(spec: MotionSpec) -> _MotionData:
    d = _pickle_load(spec.source_path)
    fps = float(d.get("fps", spec.fps if spec.fps else 30.0))
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(d["root_rot"], dtype=np.float64)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)
    n = root_pos.shape[0]
    if dof_pos.shape[0] != n or root_rot_wxyz.shape[0] != n:
        raise RuntimeError(f"{spec.source_path}: row count mismatch among root_pos/root_rot/dof_pos")
    return _MotionData(spec.name, fps, root_pos, root_rot_wxyz, dof_pos)


def _load_legacy(spec: MotionSpec) -> _MotionData:
    """Parse motion_visualization-style 56-col json into pkl-equivalent dict.

    Layout observed in ``walk2run.txt`` etc.:
      [0:3]   root_pos (world)
      [3:6]   euler XYZ (world frame, intrinsic XYZ used by Rotation.from_euler)
      [6:28]  dof_pos (csv joint order)
      [28:31] root_lin_vel (world)   -- ignored, recomputed
      [31:34] root_ang_vel (world)   -- ignored, recomputed
      [34:56] dof_vel (csv joint order) -- ignored, recomputed
    """
    with open(spec.source_path) as f:
        d = json.load(f)
    arr = np.asarray(d["Frames"], dtype=np.float64)
    if arr.shape[1] != 56:
        raise RuntimeError(f"{spec.source_path}: expected 56-col legacy txt, got {arr.shape[1]}")
    root_pos = arr[:, 0:3]
    euler_xyz = arr[:, 3:6]
    dof_pos = arr[:, 6:28]
    quat_xyzw = Rotation.from_euler("XYZ", euler_xyz, degrees=False).as_quat()
    root_rot_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    fps = 1.0 / float(d["FrameDuration"])
    return _MotionData(spec.name, fps, root_pos, root_rot_wxyz, dof_pos)


def _load_rollout(spec: MotionSpec) -> _MotionData:
    """Load a leg-only walk rollout pkl and expand it to 22-DOF (CSV order).

    The recorder (``K1-Locomotion/scripts/rsl_rl/record_amp_rollout.py``) stores
    only the 12 leg joints (``dof_pos`` + ``joint_names``) plus root pose. The
    head/arm DOFs — which the walk policy holds fixed — are filled here from
    ``_FIXED_UPPER_BODY`` so downstream reordering/EE-FK sees a full 22-DOF pose.
    """
    d = _pickle_load(spec.source_path)
    fps = float(d.get("fps", spec.fps if spec.fps else 50.0))
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(d["root_rot"], dtype=np.float64)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    leg_pos = np.asarray(d["dof_pos"], dtype=np.float64)
    leg_names = list(d["joint_names"])
    n = root_pos.shape[0]
    if leg_pos.shape[0] != n or root_rot_wxyz.shape[0] != n:
        raise RuntimeError(f"{spec.source_path}: row count mismatch among root_pos/root_rot/dof_pos")
    dof_pos = np.zeros((n, 22), dtype=np.float64)
    for j, name in enumerate(_CSV_JOINT_ORDER):
        if name in leg_names:
            dof_pos[:, j] = leg_pos[:, leg_names.index(name)]
        elif name in _FIXED_UPPER_BODY:
            dof_pos[:, j] = _FIXED_UPPER_BODY[name]
        else:
            raise RuntimeError(f"{spec.source_path}: joint '{name}' missing from rollout and _FIXED_UPPER_BODY")
    return _MotionData(spec.name, fps, root_pos, root_rot_wxyz, dof_pos)


def _reorder_pkl_to_lab(arr: np.ndarray) -> np.ndarray:
    return arr[:, _PKL_TO_LAB_K1]


def _write_amp_txt(out_path: str, frames: np.ndarray, fps: float, motion_weight: float) -> None:
    n = frames.shape[0]
    with open(out_path, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.4f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write(f'"MotionWeight": {motion_weight:.3f},\n\n')
        f.write('"Frames":\n[\n')
        for i, row in enumerate(frames):
            row_str = ", ".join(f"{v:.6f}" for v in row)
            term = "" if i == n - 1 else ","
            f.write(f"  [{row_str}]{term}\n")
        f.write("]\n}\n")


def _build_one(motion: _MotionData, out_path: str, scene: InteractiveScene,
               sim: SimulationContext, motion_weight: float,
               include_root_vel: bool = True) -> None:
    dt = 1.0 / motion.fps
    root_pos = motion.root_pos
    root_rot_wxyz = motion.root_rot_wxyz
    dof_pos_pkl = motion.dof_pos_pkl
    n = root_pos.shape[0]

    # Finite-diff velocities -> (N-1) frames, drop last position frame to align.
    lin_vel_w_np = np.diff(root_pos, axis=0) / dt                         # (N-1, 3)
    root_rot_t = torch.tensor(root_rot_wxyz, dtype=torch.float32, device=scene.device)
    dq = quat_mul(quat_conjugate(root_rot_t[:-1]), root_rot_t[1:])
    ang_vel_w_t = axis_angle_from_quat(dq) / dt                           # (N-1, 3)
    dof_vel_pkl = np.diff(dof_pos_pkl, axis=0) / dt                       # (N-1, 22)

    n_use = n - 1
    root_pos_use = root_pos[:n_use]
    root_rot_wxyz_use = root_rot_wxyz[:n_use]
    dof_pos_pkl_use = dof_pos_pkl[:n_use]

    # Body-frame velocities. quat_apply with the conjugate is the world->body
    # rotation, equivalent to IsaacLab's ``root_*_vel_b`` accessor.
    root_rot_use_t = torch.tensor(root_rot_wxyz_use, dtype=torch.float32, device=scene.device)
    lin_vel_w_t = torch.tensor(lin_vel_w_np, dtype=torch.float32, device=scene.device)
    lin_vel_b_t = quat_apply(quat_conjugate(root_rot_use_t), lin_vel_w_t)
    ang_vel_b_t = quat_apply(quat_conjugate(root_rot_use_t), ang_vel_w_t)
    lin_vel_b = lin_vel_b_t.cpu().numpy()
    ang_vel_b = ang_vel_b_t.cpu().numpy()

    # Reorder dof to IsaacLab BFS order.
    dof_pos_lab = _reorder_pkl_to_lab(dof_pos_pkl_use)
    dof_vel_lab = _reorder_pkl_to_lab(dof_vel_pkl)

    # Replay each frame in sim and read EE positions in body frame.
    robot: Articulation = scene["robot"]
    hand_ids, _ = robot.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    foot_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    left_hand_local = torch.tensor([0.0, 0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))
    right_hand_local = torch.tensor([0.0, -0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))

    ee_frames = np.zeros((n_use, 12), dtype=np.float32)

    dof_pos_lab_t = torch.tensor(dof_pos_lab, dtype=torch.float32, device=scene.device)
    dof_vel_lab_t = torch.tensor(dof_vel_lab, dtype=torch.float32, device=scene.device)
    root_state = torch.zeros((scene.num_envs, 13), dtype=torch.float32, device=scene.device)

    for i in range(n_use):
        robot.write_joint_position_to_sim(dof_pos_lab_t[i:i + 1].repeat(scene.num_envs, 1))
        robot.write_joint_velocity_to_sim(dof_vel_lab_t[i:i + 1].repeat(scene.num_envs, 1))

        root_state[:, 0:3] = torch.tensor(root_pos_use[i], dtype=torch.float32, device=scene.device)
        # nudge upward by 5cm to match replay convention (avoid ground penetration in sim)
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = torch.tensor(root_rot_wxyz_use[i], dtype=torch.float32, device=scene.device)
        root_state[:, 7:10] = torch.tensor(lin_vel_w_np[i], dtype=torch.float32, device=scene.device)
        root_state[:, 10:13] = ang_vel_w_t[i].to(scene.device)
        robot.write_root_state_to_sim(root_state)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        left_hand_w = robot.data.body_state_w[:, hand_ids[0], :3] - robot.data.root_state_w[:, 0:3] + \
            quat_apply(robot.data.body_state_w[:, hand_ids[0], 3:7], left_hand_local)
        right_hand_w = robot.data.body_state_w[:, hand_ids[1], :3] - robot.data.root_state_w[:, 0:3] + \
            quat_apply(robot.data.body_state_w[:, hand_ids[1], 3:7], right_hand_local)
        left_hand_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_hand_w)
        right_hand_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_hand_w)
        left_foot_w = robot.data.body_state_w[:, foot_ids[0], :3] - robot.data.root_state_w[:, 0:3]
        right_foot_w = robot.data.body_state_w[:, foot_ids[1], :3] - robot.data.root_state_w[:, 0:3]
        left_foot_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_foot_w)
        right_foot_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_foot_w)

        ee = torch.cat([left_hand_b[0], right_hand_b[0], left_foot_b[0], right_foot_b[0]]).detach().cpu().numpy()
        ee_frames[i] = ee

    frame_cols = [
        dof_pos_lab.astype(np.float32),
        dof_vel_lab.astype(np.float32),
        ee_frames,
    ]
    if include_root_vel:
        frame_cols += [
            lin_vel_b.astype(np.float32),
            ang_vel_b.astype(np.float32),
        ]
    frames = np.concatenate(frame_cols, axis=1)
    expected_width = 62 if include_root_vel else 56
    if frames.shape[1] != expected_width:
        raise AssertionError(
            f"unexpected width {frames.shape[1]} for {motion.name} "
            f"(expected {expected_width})"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_amp_txt(out_path, frames, fps=motion.fps, motion_weight=motion_weight)
    print(f"  ✓ {os.path.basename(out_path)}: {frames.shape[0]} frames @ {motion.fps:.2f}fps "
          f"|lat_v|≈{np.abs(lin_vel_b[:, 1]).mean():.2f}m/s |ω_z|≈{np.abs(ang_vel_b[:, 2]).mean():.2f}r/s")


def _filter_specs(specs: list[MotionSpec]) -> list[MotionSpec]:
    out = list(specs)
    if args_cli.categories:
        out = [s for s in out if s.category in args_cli.categories]
    if args_cli.only:
        out = [s for s in out if any(tok in s.name for tok in args_cli.only)]
    return out


def _rollout_specs(rollout_dir: str, weight: float) -> list[MotionSpec]:
    """One 'walk_policy' spec per *.pkl in a walk-rollout dir (56-col output)."""
    paths = sorted(glob.glob(os.path.join(rollout_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no *.pkl found in {rollout_dir}")
    return [
        MotionSpec(os.path.splitext(os.path.basename(p))[0], "walk_policy", "rollout", p,
                   weight, include_root_vel=False)
        for p in paths
    ]


def main() -> None:
    if args_cli.walk_rollout_dir:
        specs = _rollout_specs(args_cli.walk_rollout_dir, args_cli.walk_rollout_weight)
    else:
        specs = _filter_specs(_DEFAULT_SPECS)
    if not specs:
        print("[build_amp_corpus] no specs match filters; nothing to do.")
        return

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 30.0, device=args_cli.device or "cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.5))

    scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[build_amp_corpus] sim ready; building {len(specs)} clips → {args_cli.output_root}/")

    by_cat: dict[str, list[MotionSpec]] = {}
    for s in specs:
        by_cat.setdefault(s.category, []).append(s)

    for cat, items in by_cat.items():
        print(f"\n[{cat}] {len(items)} clips")
        for spec in items:
            if not os.path.isfile(spec.source_path):
                print(f"  ✗ {spec.name}: source missing ({spec.source_path})")
                continue
            try:
                if spec.kind == "pkl":
                    motion = _load_pkl(spec)
                elif spec.kind == "legacy":
                    motion = _load_legacy(spec)
                elif spec.kind == "rollout":
                    motion = _load_rollout(spec)
                else:
                    raise ValueError(f"unknown kind {spec.kind}")
                out = os.path.join(args_cli.output_root, cat, f"{spec.name}.txt")
                _build_one(
                    motion,
                    out,
                    scene,
                    sim,
                    spec.motion_weight,
                    include_root_vel=spec.include_root_vel,
                )
            except Exception as e:
                print(f"  ✗ {spec.name}: {e}")
                raise

    print("\n[build_amp_corpus] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
