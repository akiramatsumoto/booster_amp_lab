"""Replay *no-root* AMP clips in place.

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

``--leg_only`` handles the 30-col leg-only corpus produced by
``scripts/make_leg_amp_corpus.py`` for the welded-arm K1:

    [0:12]  leg joint positions   (LEG_JOINT_NAMES order)
    [12:24] leg joint velocities  (LEG_JOINT_NAMES order)
    [24:30] foot positions (L-foot, R-foot; root frame) -- shown only via pose

In that mode the 12 leg columns are written by *name* to the robot's leg joints,
so it pairs with ``--robot booster_k1_fixed_arms`` (the 14-DOF welded-arm robot):
the arms hold their welded deploy pose, the head sits at 0, and the legs kick.

.. code-block:: bash

    # Single clip
    python scripts/replay_amp_noroot.py \
        --motion booster_assets/motions/K1/motion_amp_expert/omni/kick/walk_kick.txt

    # All kick clips in sequence (default pattern)
    python scripts/replay_amp_noroot.py --loop

    # Welded-arm robot + 30-col leg-only kick corpus (this change)
    python scripts/replay_amp_noroot.py --loop --fps 30 \
        --robot booster_k1_fixed_arms --leg_only

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
    default=None,
    help="Glob of clips to play in sequence when --motion is not given. Defaults "
    "to the 56-col kick corpus, or the 30-col leg-only corpus under --leg_only.",
)
parser.add_argument(
    "--leg_only",
    action="store_true",
    help="Read 30-col leg-only clips (12 leg pos + 12 leg vel + 6 foot pos) and "
    "write the 12 leg columns by name. Pair with --robot booster_k1_fixed_arms.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Playback frames per second.")
parser.add_argument(
    "--root_height", type=float, default=0.57, help="Fixed pelvis height (m) for the pinned root."
)
parser.add_argument("--loop", action="store_true", help="Loop the playlist forever.")
parser.add_argument(
    "--robot",
    choices=["booster_t1", "booster_k1", "booster_k1_fixed_arms"],
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

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG,
    BOOSTER_K1_FIXED_ARMS_CFG,
    BOOSTER_T1_CFG,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_stand_kick_amp.env_cfg import (
    LEG_JOINT_NAMES,
)


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
    elif args_cli.robot == "booster_k1_fixed_arms":
        robot: ArticulationCfg = BOOSTER_K1_FIXED_ARMS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "booster_t1":
        robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError("--robot must be booster_t1, booster_k1 or booster_k1_fixed_arms.")


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

    # Leg-only mode: resolve the 12 leg columns to robot joint indices by name so
    # the mapping is correct regardless of the articulation's own joint sort. The
    # non-leg joints (welded arms have none; the head) keep their default pose.
    leg_ids = None
    if args_cli.leg_only:
        leg_ids, resolved = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        if len(resolved) != len(LEG_JOINT_NAMES):
            raise RuntimeError(
                f"--leg_only expected {len(LEG_JOINT_NAMES)} leg joints on the robot, "
                f"resolved {len(resolved)}: {resolved}"
            )
        leg_ids = torch.tensor(leg_ids, device=device)

    # Fixed root: origin, upright, at the requested pelvis height.
    root_state = torch.zeros((scene.num_envs, 13), device=device)
    root_state[:, 2] = float(args_cli.root_height)
    root_state[:, 3] = 1.0  # qw = 1 (identity)

    # Seed with the robot's default pose so unwritten joints (head/arms) hold it.
    dof_pos = robot.data.default_joint_pos.clone()
    dof_vel = torch.zeros((scene.num_envs, n_joints), device=device)
    frame_dt = 1.0 / float(args_cli.fps)

    n_leg = len(LEG_JOINT_NAMES)  # 12
    # Columns needed per clip: 30 for leg-only, 2*n_joints for the full-body clips.
    min_cols = 2 * n_leg if args_cli.leg_only else 2 * n_joints

    # Pre-load every clip once.
    loaded = []
    for path in clips:
        frames = _load_frames(path)
        c = frames.shape[1]
        if c < min_cols:
            print(f"[SKIP] {path}: {c} cols < {min_cols} needed.")
            continue
        if args_cli.leg_only and c != 30:
            print(f"[WARN] {path}: {c} cols (expected 30). Reading first {2 * n_leg} as leg pos/vel.")
        elif not args_cli.leg_only and c != 56:
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
                if args_cli.leg_only:
                    # 12 leg pos + 12 leg vel; feet[24:30] ignored (EE, not joints).
                    dof_pos[:, leg_ids] = frames[t, 0:n_leg]
                    dof_vel[:, leg_ids] = frames[t, n_leg:2 * n_leg]
                else:
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
        pattern = args_cli.pattern
        if pattern is None:
            pattern = (
                "booster_assets/motions/K1/motion_amp_expert/omni_legs/kick/stand_kick*.txt"
                if args_cli.leg_only
                else "booster_assets/motions/K1/motion_amp_expert/omni/kick/walk_kick*.txt"
            )
        clips = sorted(glob.glob(pattern))
        if not clips:
            raise FileNotFoundError(f"No files match pattern: {pattern}")

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
