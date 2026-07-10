"""Replay *no-root* AMP clips (56-col joint+EE format) in place.

The AMP training corpus under ``motion_amp_expert/omni/**`` (e.g. the
``omni/kick/walk_kick*.txt`` kick priors) stores only discriminator features:

    [0:22]  joint positions   (robot / sim joint order)
    [22:44] joint velocities   (robot / sim joint order)
    [44:56] end-effector positions (L-hand, R-hand, L-foot, R-foot; root frame)

Root translation/orientation was intentionally dropped, so these clips CANNOT
be replayed with the usual ``replay_amp_txt.py`` (which expects root pose in the
first 6 columns). This script instead pins the root at a fixed pose and streams
the joint angles, so you can inspect the kick / gait *pose and style* on the
spot (the robot will not translate — the walk/approach is not stored).

.. code-block:: bash

    # Single clip
    python scripts/replay_amp_noroot.py \
        --motion booster_assets/motions/K1/motion_amp_expert/omni/kick/walk_kick.txt

    # All kick clips in sequence (default pattern)
    python scripts/replay_amp_noroot.py --loop

    # Slow-motion, custom root height
    python scripts/replay_amp_noroot.py --fps 15 --root_height 0.6
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay no-root (56-col) AMP clips in place.")
parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Path to a single 56-col AMP txt clip. Overrides --pattern.",
)
parser.add_argument(
    "--pattern",
    type=str,
    default="booster_assets/motions/K1/motion_amp_expert/omni/kick/walk_kick*.txt",
    help="Glob of clips to play in sequence when --motion is not given.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Playback frames per second.")
parser.add_argument(
    "--root_height", type=float, default=0.57, help="Fixed pelvis height (m) for the pinned root."
)
parser.add_argument("--loop", action="store_true", help="Loop the playlist forever.")
parser.add_argument(
    "--robot",
    choices=["booster_t1", "booster_k1"],
    default="booster_k1",
    help="Which robot to visualize.",
)
parser.add_argument(
    "--reorder",
    action="store_true",
    help="Apply motion->sim joint reordering. Off by default: these files are "
    "already stored in sim joint order. Enable only if the pose looks scrambled.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG, BOOSTER_T1_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Ground + light + a single robot to pose."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    if args_cli.robot == "booster_k1":
        robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "booster_t1":
        robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError("--robot must be booster_t1 or booster_k1.")


def _load_frames(path: str) -> np.ndarray:
    """Load the 'Frames' array from a JSON-style AMP txt clip -> (T, C)."""
    with open(path) as f:
        data = json.load(f)
    frames = np.asarray(data["Frames"], dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"{path}: expected 2-D Frames, got shape {frames.shape}")
    return frames


def _reorder_motion_to_sim(dof: torch.Tensor) -> torch.Tensor:
    """K1 motion-file joint order -> sim joint order (mirror of replay_amp_txt)."""
    (
        AAHead_yaw, Head_pitch, ALeft_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch,
        Left_Elbow_Yaw, ARight_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch,
        Right_Elbow_Yaw, Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch,
        Left_Ankle_Pitch, Left_Ankle_Roll, Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw,
        Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll,
    ) = torch.split(dof, 1, dim=0)
    return torch.cat([
        AAHead_yaw, ALeft_Shoulder_Pitch, ARight_Shoulder_Pitch, Left_Hip_Pitch, Right_Hip_Pitch,
        Head_pitch, Left_Shoulder_Roll, Right_Shoulder_Roll, Left_Hip_Roll, Right_Hip_Roll,
        Left_Elbow_Pitch, Right_Elbow_Pitch, Left_Hip_Yaw, Right_Hip_Yaw, Left_Elbow_Yaw,
        Right_Elbow_Yaw, Left_Knee_Pitch, Right_Knee_Pitch, Left_Ankle_Pitch, Right_Ankle_Pitch,
        Left_Ankle_Roll, Right_Ankle_Roll,
    ])


def run_simulator(sim: SimulationContext, scene: InteractiveScene, clips: list[str]):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    n_joints = robot.num_joints
    device = scene.device

    # Fixed root: origin, upright, at the requested pelvis height.
    root_state = torch.zeros((scene.num_envs, 13), device=device)
    root_state[:, 2] = float(args_cli.root_height)
    root_state[:, 3] = 1.0  # qw = 1 (identity)

    dof_pos = torch.zeros((scene.num_envs, n_joints), device=device)
    dof_vel = torch.zeros((scene.num_envs, n_joints), device=device)
    frame_dt = 1.0 / float(args_cli.fps)

    # Pre-load every clip once.
    loaded = []
    for path in clips:
        frames = _load_frames(path)
        c = frames.shape[1]
        if c < 2 * n_joints:
            print(f"[SKIP] {path}: {c} cols < {2 * n_joints} needed for {n_joints} joints.")
            continue
        if c != 56:
            print(f"[WARN] {path}: {c} cols (expected 56). Reading first {2 * n_joints} as joint pos/vel.")
        loaded.append((os.path.basename(path), torch.from_numpy(frames).to(device)))
    if not loaded:
        raise RuntimeError("No playable clips found.")

    while simulation_app.is_running():
        for name, frames in loaded:
            print(f"[PLAY] {name}: {frames.shape[0]} frames @ {args_cli.fps} fps")
            for t in range(frames.shape[0]):
                if not simulation_app.is_running():
                    break
                wall = time.time()
                jp = frames[t, 0:n_joints]
                jv = frames[t, n_joints:2 * n_joints]
                if args_cli.reorder:
                    jp = _reorder_motion_to_sim(jp)
                    jv = _reorder_motion_to_sim(jv)
                dof_pos[:] = jp
                dof_vel[:] = jv

                robot.write_joint_position_to_sim(dof_pos)
                robot.write_joint_velocity_to_sim(dof_vel)
                robot.write_root_state_to_sim(root_state)
                scene.write_data_to_sim()
                sim.render()  # render only — no physics step
                scene.update(sim_dt)

                # Keep the camera on the robot.
                look = root_state[0, :3].cpu().numpy()
                sim.set_camera_view(look + np.array([2.0, 2.0, 0.6]), look)

                # Real-time pacing.
                sleep = frame_dt - (time.time() - wall)
                if sleep > 0:
                    time.sleep(sleep)
        if not args_cli.loop:
            break


def main():
    if args_cli.motion:
        clips = [args_cli.motion]
        if not os.path.isfile(clips[0]):
            raise FileNotFoundError(clips[0])
    else:
        clips = sorted(glob.glob(args_cli.pattern))
        if not clips:
            raise FileNotFoundError(f"No files match pattern: {args_cli.pattern}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.fps)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene, clips)


if __name__ == "__main__":
    main()
    simulation_app.close()
