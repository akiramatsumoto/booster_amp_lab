"""Observation functions for the soccer kick task.

Three groups:
  * policy (actor): noisy/proprio + ball pos/mask in body-yaw frame (V1 uses GT
    until virtual perception is wired in V1.1).
  * privileged (critic): everything actor sees + ball lin/ang vel, goal pos,
    target dir, target strength, kick state.
  * AMP discriminator: identical to existing AMPObsCfg (joint pos/vel + hand/foot).

Functions are called with ``(env, command_name=...)`` and read the
:class:`SoccerKickCommand` instance from the command manager.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from booster_rl_tasks.assets.objects.soccer import FIELD_HALF_LENGTH
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerKickCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# --- ACTOR observations ----------------------------------------------------
def ball_pos_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Ball position (xy) in robot's body-yaw frame.

    Returns GT when virtual perception is disabled (V1.0), otherwise the
    noisy/delayed perceived xy from the head-camera detection pipeline.

    Shape: (num_envs, 2)
    """
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return cmd.ball_pos_b[:, :2]
    return cmd.ball_pos_b_perceived[:, :2]


def ball_mask(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Binary "ball detected" flag.

    Returns ones when perception is disabled, otherwise the per-env mask
    from the virtual perception ring buffer.

    Shape: (num_envs, 1)
    """
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.ones(cmd.num_envs, 1, device=cmd.device)
    return cmd.ball_mask_perceived.unsqueeze(-1)


def last_seen_dt(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Time since the most recent ball detection (s).

    Returns zeros when perception is disabled, otherwise the per-env
    counter maintained by ``VirtualPerception``.

    Shape: (num_envs, 1)
    """
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.zeros(cmd.num_envs, 1, device=cmd.device)
    return cmd.last_seen_dt.unsqueeze(-1)


def target_dir_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Kick target direction in body-yaw frame (cos, sin).

    Shape: (num_envs, 2)
    """
    return _cmd(env, command_name).target_dir_b


def target_strength_norm(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Normalized kick target strength scalar in [0, 1].

    Shape: (num_envs, 1)
    """
    return _cmd(env, command_name).target_strength_normalized.unsqueeze(-1)


def is_shoot_flag(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Mode flag: 1 if "shoot", 0 if "pass". V1: always 1.

    Shape: (num_envs, 1)
    """
    return _cmd(env, command_name).is_shoot.float().unsqueeze(-1)


def stop_flag(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Mode flag: 1 if the env is in post-kick "stop" mode, else 0 (kick).

    During training this latches to 1 once a kick succeeds (see
    ``SoccerKickCommand``). At deploy time the same input is driven externally
    to command the policy to stand still or to kick.

    Shape: (num_envs, 1)
    """
    return _cmd(env, command_name).stop_mode.float().unsqueeze(-1)


# --- CRITIC (privileged) observations -------------------------------------
def ball_pos_b_gt(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Ground-truth ball xyz in body-yaw frame.

    Shape: (num_envs, 3)
    """
    return _cmd(env, command_name).ball_pos_b


def ball_vel_b_gt(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Ground-truth ball linear velocity in body-yaw frame.

    Shape: (num_envs, 3)
    """
    return _cmd(env, command_name).ball_vel_b


def goal_pos_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Ground-truth goal-center position (xy) in body-yaw frame.

    Shape: (num_envs, 2)
    """
    return _cmd(env, command_name).goal_pos_b


def goal_dir_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Ground-truth unit direction to goal center in body-yaw frame (cos, sin).

    Shape: (num_envs, 2)
    """
    return _cmd(env, command_name).goal_dir_b


def kick_state_flags(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Three latch flags: kick_contact_awarded, kick_success_awarded, goal_awarded.

    Shape: (num_envs, 3)
    """
    cmd = _cmd(env, command_name)
    return torch.stack(
        [
            cmd.kick_contact_awarded.float(),
            cmd.kick_success_awarded.float(),
            cmd.goal_awarded.float(),
        ],
        dim=-1,
    )


def pass_target_dir_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Privileged: unit direction (cos, sin) to pass target in body-yaw frame.

    Zeros when the episode is in shoot mode.
    Shape: (num_envs, 2)
    """
    return _cmd(env, command_name).pass_target_dir_b


def pass_target_dist(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """Privileged: distance to pass target, normalized by ``FIELD_HALF_LENGTH``.

    Zeros when the episode is in shoot mode.
    Shape: (num_envs, 1)
    """
    cmd = _cmd(env, command_name)
    return (cmd.pass_target_dist_b / float(FIELD_HALF_LENGTH)).unsqueeze(-1)


def receiver_pos_b(env: "ManagerBasedRLEnv", command_name: str = "soccer_kick") -> torch.Tensor:
    """V3 privileged: receiver world position in kicker's body-yaw frame.

    Returns the receiver's xy position rotated into the kicker's yaw frame.
    When no receiver is configured on the command term, returns zeros. The
    z component is dropped because the receiver is constrained to ground
    height (root z ~ 0.57) and provides little signal to the critic.

    Shape: (num_envs, 2)
    """
    cmd = _cmd(env, command_name)
    recv = cmd.receiver
    if recv is None:
        return torch.zeros(cmd.num_envs, 2, device=cmd.device)
    rel_w = torch.zeros(cmd.num_envs, 3, device=cmd.device)
    rel_w[:, :2] = recv.data.root_pos_w[:, :2] - cmd.robot.data.root_pos_w[:, :2]
    rel_b = quat_apply_inverse(cmd.robot_yaw_quat, rel_w)
    return rel_b[:, :2]


# =========================================================================
# V3.2 — receiver-side observations (GT only; perception is kicker-only)
# =========================================================================


def _receiver_yaw_quat(cmd: SoccerKickCommand) -> torch.Tensor:
    """Yaw-only quaternion of the receiver root (or identity if absent)."""
    recv = cmd.receiver
    if recv is None:
        q = torch.zeros(cmd.num_envs, 4, device=cmd.device)
        q[:, 0] = 1.0
        return q
    return yaw_quat(recv.data.root_quat_w)


def receiver_ball_pos_b(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """V3.2 actor obs: GT ball xy in the receiver's body-yaw frame.

    Shape: (num_envs, 2).
    """
    cmd = _cmd(env, command_name)
    recv = cmd.receiver
    if recv is None:
        return torch.zeros(cmd.num_envs, 2, device=cmd.device)
    rel_w = cmd.ball_pos_w - recv.data.root_pos_w  # (N, 3)
    rel_b = quat_apply_inverse(_receiver_yaw_quat(cmd), rel_w)
    return rel_b[:, :2]


def receiver_ball_mask(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """V3.2 actor obs: ball-detection mask for the receiver.

    The receiver does NOT have its own VirtualPerception in V3.2, so the mask
    is constant ones. Kept for symmetry with the kicker's policy input.

    Shape: (num_envs, 1).
    """
    cmd = _cmd(env, command_name)
    return torch.ones(cmd.num_envs, 1, device=cmd.device)


def receiver_kicker_dir_b(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """V3.2 actor obs: unit direction from receiver to kicker in receiver's body-yaw frame.

    Shape: (num_envs, 2).
    """
    cmd = _cmd(env, command_name)
    recv = cmd.receiver
    if recv is None:
        return torch.zeros(cmd.num_envs, 2, device=cmd.device)
    rel_w = torch.zeros(cmd.num_envs, 3, device=cmd.device)
    rel_w[:, :2] = cmd.robot.data.root_pos_w[:, :2] - recv.data.root_pos_w[:, :2]
    rel_b = quat_apply_inverse(_receiver_yaw_quat(cmd), rel_w)[:, :2]
    norm = torch.linalg.norm(rel_b, dim=-1, keepdim=True).clamp_min(1e-6)
    return rel_b / norm
