"""Reward terms for the soccer kick task (V1, kicker only).

Reward groups (planned multi-critic split, see ``soccer_amp_roadmap.md``):
  * "goal" group:    target_progress, ball_approach, kick_success, goal_scored,
                     kick_angle_error, kick_strength_error
  * "aux" group:     pre_kick_body_yaw_alignment, head_*, foot_proximity,
                     pelvis_orientation, AMP style (handled by AMP PPO)

For V1.0 these are all combined into a single critic; group tagging is left
in the docstring so we can introduce the multi-critic split later without
touching the env config.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerKickCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# =========================================================================
# Goal-related rewards
# =========================================================================


def target_progress(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Reward ball motion projected onto the target direction (mode-agnostic).

    Equivalent to ``Δ(ball_xy · target_dir_w) / dt``. Continuous shaping.
    Used by legacy soccer_kick_amp (V1.x) variants. New Stage 1 V4 variants
    should prefer :func:`target_progress_shoot` (is_shoot-gated) so the
    pass mode doesn't get a redundant continuous shaping that competes with
    its single-shot bonus.
    """
    cmd = _cmd(env, command_name)
    proj = (cmd.ball_vel_w[:, :2] * cmd.target_dir_w).sum(-1)
    # m/s; positive when ball moves along target. Clamp to avoid extreme post-bounce spikes.
    return torch.clamp(proj, min=-5.0, max=15.0)


def target_progress_shoot(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """V4 Stage 1 — ``target_progress`` gated to shoot mode only.

    Shoot mode allows multi-attempt scoring, so continuous ball-toward-goal
    shaping is helpful. Pass mode handles its own shaping via
    :func:`target_progress_pass_window` (limited to the first 20 steps after
    kick) and the terminal :func:`ball_at_target_terminal` bonus, so a
    second continuous reward would double-count and distort the pass
    incentive.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    proj = (cmd.ball_vel_w[:, :2] * cmd.target_dir_w).sum(-1)
    is_shoot = cmd.is_shoot.float()
    return is_shoot * torch.clamp(proj, min=-5.0, max=15.0)


def kick_aim_at_goal_shoot(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_ball_speed: float = 1.0,
) -> torch.Tensor:
    """V4 shoot-precision reward: cosine of (ball velocity, ball→goal-center).

    Continuous, fires every step the ball is moving > ``min_ball_speed``.
    Positive when the ball heads toward the goal *center* (not the
    sampled aim-point), so the policy learns to aim at the high-scoring
    region of the goal mouth rather than the goal-line margin.

    Range ``[-1, +1]``. Shoot mode only.
    """
    cmd = _cmd(env, command_name)
    env_origins = env.scene.env_origins
    goal_xy_w_x = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
    goal_xy_w_y = env_origins[:, 1] + 0.0
    rel_x = goal_xy_w_x - cmd.ball_pos_w[:, 0]
    rel_y = goal_xy_w_y - cmd.ball_pos_w[:, 1]
    rel_norm = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(1e-3)
    rel_ux = rel_x / rel_norm
    rel_uy = rel_y / rel_norm
    ball_vx = cmd.ball_vel_w[:, 0]
    ball_vy = cmd.ball_vel_w[:, 1]
    ball_speed = torch.sqrt(ball_vx * ball_vx + ball_vy * ball_vy).clamp_min(1e-3)
    cos_angle = (ball_vx * rel_ux + ball_vy * rel_uy) / ball_speed
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    is_shoot = cmd.is_shoot.float()
    fast_enough = (ball_speed > float(min_ball_speed)).float()
    return is_shoot * fast_enough * cos_angle


def kick_aim_at_pass_target_pass(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_ball_speed: float = 1.0,
) -> torch.Tensor:
    """V4.2 pass-precision reward: cosine of (ball velocity, ball→pass_target).

    Pass-mode counterpart of :func:`kick_aim_at_goal_shoot`. Continuous,
    fires every step the ball is moving > ``min_ball_speed`` in pass mode.
    Encourages the policy to aim each pass at the *current* receiver
    location rather than along a stale target_dir.

    Shape: ``(num_envs,)`` in ``[-1, +1]``.
    """
    cmd = _cmd(env, command_name)
    rel_x = cmd.pass_target_pos_w[:, 0] - cmd.ball_pos_w[:, 0]
    rel_y = cmd.pass_target_pos_w[:, 1] - cmd.ball_pos_w[:, 1]
    rel_norm = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(1e-3)
    rel_ux = rel_x / rel_norm
    rel_uy = rel_y / rel_norm
    ball_vx = cmd.ball_vel_w[:, 0]
    ball_vy = cmd.ball_vel_w[:, 1]
    ball_speed = torch.sqrt(ball_vx * ball_vx + ball_vy * ball_vy).clamp_min(1e-3)
    cos_angle = (ball_vx * rel_ux + ball_vy * rel_uy) / ball_speed
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    is_pass = 1.0 - cmd.is_shoot.float()
    fast_enough = (ball_speed > float(min_ball_speed)).float()
    return is_pass * fast_enough * cos_angle


def kick_power_track_shoot(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    sigma: float = 1.5,
    window_steps: int = 30,
) -> torch.Tensor:
    """V4.2 strength-tracking reward: peak ball speed approaches commanded strength.

    Fires every step in the ``window_steps`` window AFTER the first kick
    contact (shoot mode only). The reward is ``exp(-(peak - cmd)² / σ²)``
    on the running per-env peak kick speed; the policy is rewarded for
    making the peak match the commanded strength.

    Continuous so the policy gets a dense gradient (unlike
    ``kick_strength_error_shoot`` which only fires once per contact).
    This is the lever that pushes the policy to honor the commanded
    strength — without it the policy converged to a "default" ~5 m/s
    kick regardless of command in V4.1.

    Shape: ``(num_envs,)`` in ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    peak = cmd._peak_kick_speed  # type: ignore[attr-defined]
    cmd_strength = cmd.target_strength
    err = peak - cmd_strength
    is_shoot = cmd.is_shoot.float()
    fast_enough = (peak > 1.0).float()
    return is_shoot * in_window.float() * fast_enough * torch.exp(
        -(err * err) / (sigma * sigma)
    )


def kick_power_track_pass(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    sigma: float = 1.0,
    window_steps: int = 15,
) -> torch.Tensor:
    """V4.2 strength-tracking reward (pass mode).

    Pass mode is one-shot, so a tighter sigma + shorter window encourages
    a single, precise-strength kick. Same shape as
    :func:`kick_power_track_shoot` but pass-gated.

    Shape: ``(num_envs,)`` in ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    peak = cmd._peak_kick_speed  # type: ignore[attr-defined]
    cmd_strength = cmd.target_strength
    err = peak - cmd_strength
    is_pass = 1.0 - cmd.is_shoot.float()
    fast_enough = (peak > 1.0).float()
    return is_pass * in_window.float() * fast_enough * torch.exp(
        -(err * err) / (sigma * sigma)
    )


def kick_angle_error_shoot(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_ball_speed: float = 1.0,
    window_steps: int = 15,
) -> torch.Tensor:
    """V4.3 angle-precision penalty for shoot mode (speed-independent, live target).

    Negative counterpart of :func:`kick_aim_at_goal_shoot`. Penalizes
    ``angle²`` between the ball velocity and the live ball→goal-center
    direction, active in the ``window_steps`` window after kick contact.

    Differs from the legacy :func:`kick_angle_error`:
      * No ``ball_xy_speed`` factor — the legacy formula
        ``angle² × ball_speed`` made faster kicks *strictly more penalized*
        for the same angle, creating a perverse incentive to kick weakly
        (the V4.2 plateau).
      * Reference direction is the live ball→goal-center (matches
        ``kick_aim_at_goal_shoot``), not the spawn-time ``target_dir_w``
        which is sampled at a margin within the goal mouth.

    Caller supplies a negative weight (sign lives in the cfg).

    Shape: ``(num_envs,)`` ≥ 0.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))

    env_origins = env.scene.env_origins
    goal_xy_w_x = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
    goal_xy_w_y = env_origins[:, 1] + 0.0
    rel_x = goal_xy_w_x - cmd.ball_pos_w[:, 0]
    rel_y = goal_xy_w_y - cmd.ball_pos_w[:, 1]
    rel_norm = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(1e-3)
    rel_ux = rel_x / rel_norm
    rel_uy = rel_y / rel_norm

    ball_vx = cmd.ball_vel_w[:, 0]
    ball_vy = cmd.ball_vel_w[:, 1]
    ball_speed = torch.sqrt(ball_vx * ball_vx + ball_vy * ball_vy).clamp_min(1e-3)
    cos_angle = (ball_vx * rel_ux + ball_vy * rel_uy) / ball_speed
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    is_shoot = cmd.is_shoot.float()
    fast_enough = (ball_speed > float(min_ball_speed)).float()
    return is_shoot * in_window.float() * fast_enough * angle * angle


def kick_angle_error_pass(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_ball_speed: float = 1.0,
    window_steps: int = 15,
) -> torch.Tensor:
    """V4.3 angle-precision penalty for pass mode (speed-independent, live target).

    Pass-mode counterpart of :func:`kick_angle_error_shoot`. Reference
    direction is the live ball→pass-target (matches
    :func:`kick_aim_at_pass_target_pass`).

    Shape: ``(num_envs,)`` ≥ 0.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))

    rel_x = cmd.pass_target_pos_w[:, 0] - cmd.ball_pos_w[:, 0]
    rel_y = cmd.pass_target_pos_w[:, 1] - cmd.ball_pos_w[:, 1]
    rel_norm = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(1e-3)
    rel_ux = rel_x / rel_norm
    rel_uy = rel_y / rel_norm

    ball_vx = cmd.ball_vel_w[:, 0]
    ball_vy = cmd.ball_vel_w[:, 1]
    ball_speed = torch.sqrt(ball_vx * ball_vx + ball_vy * ball_vy).clamp_min(1e-3)
    cos_angle = (ball_vx * rel_ux + ball_vy * rel_uy) / ball_speed
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    is_pass = 1.0 - cmd.is_shoot.float()
    fast_enough = (ball_speed > float(min_ball_speed)).float()
    return is_pass * in_window.float() * fast_enough * angle * angle


def foot_ball_proximity(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    sigma: float = 0.35,
    recent_visible_dt: float = 0.0,
) -> torch.Tensor:
    """Reward foot getting close to the ball (pre-kick only, perception-gated).

    This is the early-stage incentive for first-touch — without it the policy
    plateaus at ~1 m from the ball without ever attempting contact.

    Shape: (num_envs,), Gaussian on the *minimum* foot-ball distance.

    V3.3: gated by ``cmd.ball_mask_perceived`` so the GT-distance shaping
    is suppressed whenever the head camera does not currently see the ball.
    Forces the policy to actively look around before being credited for
    approaching.
    """
    cmd = _cmd(env, command_name)
    foot_pos_w = cmd.robot.data.body_pos_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    ball_p = cmd.ball_pos_w.unsqueeze(1)
    foot_ball_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1)
    min_d = foot_ball_d.amin(dim=-1)
    pre_kick = (~cmd.kick_contact_awarded).float()
    visible = _ball_visible_or_recent_gate(cmd, recent_visible_dt)
    return pre_kick * visible * torch.exp(-min_d * min_d / (sigma * sigma))


def ball_approach(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing distance between robot trunk and ball (pre-kick only, perception-gated).

    Uses a smooth potential ``exp(-dist / σ)`` to avoid double-counting
    once the ball is already close.

    V3.3: multiplied by ``cmd.ball_mask_perceived`` so the policy gets the
    approach gradient only when the ball is actually visible. This stops the
    "freeze when ball-out-of-FOV" failure mode by making the GT shaping
    invisible to the policy when perception drops out.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    rel = cmd.ball_pos_w - asset.data.root_pos_w
    dist = torch.linalg.norm(rel[:, :2], dim=-1)
    # Gate off after kick: don't reward chasing after the ball has been kicked away.
    pre_kick = (~cmd.kick_contact_awarded).float()
    return pre_kick * cmd.ball_mask_perceived * torch.exp(-dist / 1.5)


def ball_approach_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_delta: float = 0.05,
    recent_visible_dt: float = 0.0,
) -> torch.Tensor:
    """Reward visible pre-kick reduction in robot-ball distance.

    Unlike the absolute ``ball_approach`` potential, this goes to zero when the
    robot hovers near the ball without continuing toward contact.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    rel = cmd.ball_pos_w - asset.data.root_pos_w
    dist = torch.linalg.norm(rel[:, :2], dim=-1)
    prev = getattr(cmd, "_pre_kick_ball_approach_prev_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = dist.clone()
        setattr(cmd, "_pre_kick_ball_approach_prev_dist", prev)
    reset = env.episode_length_buf <= 1
    prev_for_delta = torch.where(reset, dist, prev)
    progress = (prev_for_delta - dist).clamp(min=0.0, max=float(max_delta))
    prev.copy_(dist)
    progress = progress / max(float(max_delta), 1e-6)
    pre_kick = (~cmd.kick_contact_awarded).float()
    visible = _ball_visible_or_recent_gate(cmd, recent_visible_dt)
    return pre_kick * visible * progress


def goal_progress_shoot(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    max_delta: float = 0.10,
    max_backslide: float = 0.05,
    min_ball_speed: float = 0.35,
) -> torch.Tensor:
    """LVDRS-style shoot reward: potential progress from ball to goal center.

    This replaces stacks of velocity-projection and angle shaping with one direct
    objective: reduce ball-goal distance after the shot starts. Positive progress
    is capped and normalized; small backslides stay visible to the critic.
    """
    cmd = _cmd(env, command_name)
    env_origins = env.scene.env_origins
    goal_xy = torch.zeros_like(cmd.ball_pos_w[:, :2])
    goal_xy[:, 0] = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
    goal_xy[:, 1] = env_origins[:, 1]
    dist = torch.linalg.norm(goal_xy - cmd.ball_pos_w[:, :2], dim=-1)
    prev = getattr(cmd, "_shoot_goal_progress_prev_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = dist.clone()
        setattr(cmd, "_shoot_goal_progress_prev_dist", prev)
    reset = env.episode_length_buf <= 1
    prev_for_delta = torch.where(reset, dist, prev)
    delta = (prev_for_delta - dist).clamp(
        min=-float(max_backslide),
        max=float(max_delta),
    )
    prev.copy_(dist)
    progress = delta / max(float(max_delta), 1e-6)
    active = cmd.kick_contact_awarded | (_ball_xy_speed(cmd) > float(min_ball_speed))
    return cmd.is_shoot.float() * active.float() * progress


def kick_contact_bonus(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """One-shot reward on the step the foot first contacts the ball.

    Encourages getting *any* kick out before refining direction/strength.
    """
    cmd = _cmd(env, command_name)
    return cmd.kick_contact_new.float()


def clean_kick_contact_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    target_fraction: float = 0.45,
    min_projected_speed: float = 2.0,
    min_alignment: float = 0.55,
    min_body_target_cos: float = 0.35,
    recent_visible_dt: float = 0.0,
    min_height: float = 0.40,
    max_tilt: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward only deliberate, target-directed first contact.

    The raw contact latch counts any foot touch that moves the ball above the
    speed threshold, including approach bumps. This term pays contact only when
    the ball was visible, the body is roughly target-facing, and the resulting
    ball velocity has a useful projection along the commanded target direction.
    """
    cmd = _cmd(env, command_name)
    new_contact = cmd.kick_contact_new.float()
    visible = _ball_visible_or_recent_gate(cmd, recent_visible_dt)
    ball_speed = _ball_xy_speed(cmd).clamp_min(1e-6)
    proj = (cmd.ball_vel_w[:, :2] * cmd.target_dir_w).sum(-1)
    target = cmd.target_strength.clamp_min(float(min_projected_speed))
    required_proj = torch.maximum(
        torch.full_like(proj, float(min_projected_speed)),
        target * float(target_fraction),
    )
    speed_quality = _bounded_ramp(proj, required_proj * 0.65, required_proj)
    alignment = (proj / ball_speed).clamp(-1.0, 1.0)
    align_quality = _bounded_ramp(alignment, float(min_alignment), 1.0)
    body_quality = _bounded_ramp(
        cmd.target_dir_b[:, 0].clamp(-1.0, 1.0),
        float(min_body_target_cos),
        1.0,
    )
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    return new_contact * visible * speed_quality * align_quality * body_quality * stable


def unclean_kick_contact_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    target_fraction: float = 0.45,
    min_projected_speed: float = 2.0,
    min_alignment: float = 0.55,
    min_body_target_cos: float = 0.35,
    recent_visible_dt: float = 0.0,
    min_height: float = 0.40,
    max_tilt: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for first contacts that look like bumps, glances, or blind kicks."""
    cmd = _cmd(env, command_name)
    new_contact = cmd.kick_contact_new
    visible = _ball_visible_or_recent_bool(cmd, recent_visible_dt)
    ball_speed = _ball_xy_speed(cmd).clamp_min(1e-6)
    proj = (cmd.ball_vel_w[:, :2] * cmd.target_dir_w).sum(-1)
    target = cmd.target_strength.clamp_min(float(min_projected_speed))
    required_proj = torch.maximum(
        torch.full_like(proj, float(min_projected_speed)),
        target * float(target_fraction),
    )
    alignment = (proj / ball_speed).clamp(-1.0, 1.0)
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    clean = (
        visible
        & (proj >= required_proj)
        & (alignment >= float(min_alignment))
        & (cmd.target_dir_b[:, 0] >= float(min_body_target_cos))
        & (stable >= 0.5)
    )
    return (new_contact & ~clean).float()


def arch_contact_lateral_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_lateral_speed: float = 0.35,
    saturation_speed: float = 2.5,
    recent_visible_dt: float = 0.25,
) -> torch.Tensor:
    """LVDRS-style contact reward for side-foot/lateral swing at impact."""
    cmd = _cmd(env, command_name)
    foot_vel_b = _nearest_foot_velocity_b(cmd)
    lateral_speed = foot_vel_b[:, 1].abs()
    shaped = _bounded_ramp(
        lateral_speed,
        float(min_lateral_speed),
        float(saturation_speed),
    )
    visible = _ball_visible_or_recent_gate(cmd, recent_visible_dt)
    return cmd.kick_contact_new.float() * visible * shaped


def toe_poke_contact_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_forward_speed: float = 0.30,
    saturation_speed: float = 1.8,
    recent_visible_dt: float = 0.25,
) -> torch.Tensor:
    """LVDRS-style penalty for forward toe-poke contact instead of an instep kick."""
    cmd = _cmd(env, command_name)
    foot_vel_b = _nearest_foot_velocity_b(cmd)
    forward_speed = foot_vel_b[:, 0].abs()
    shaped = _bounded_ramp(
        forward_speed,
        float(min_forward_speed),
        float(saturation_speed),
    )
    visible = _ball_visible_or_recent_gate(cmd, recent_visible_dt)
    return cmd.kick_contact_new.float() * visible * shaped


def pre_kick_swing_miss_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    near_radius: float = 0.45,
    foot_speed_thresh: float = 1.2,
    max_ball_speed: float = 0.8,
) -> torch.Tensor:
    """Penalize fast foot swings near the visible ball that do not move it.

    This catches the "air kick / skim past the ball" failure mode that raw
    contact-rate metrics do not see.
    """
    cmd = _cmd(env, command_name)
    foot_pos_w = cmd.robot.data.body_pos_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    foot_vel_w = cmd.robot.data.body_lin_vel_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    ball_p = cmd.ball_pos_w.unsqueeze(1)
    min_foot_ball_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1).amin(dim=-1)
    max_foot_speed = torch.linalg.norm(foot_vel_w, dim=-1).amax(dim=-1)
    ball_speed = _ball_xy_speed(cmd)
    pre_kick = ~cmd.kick_contact_awarded
    miss = (
        pre_kick
        & _ball_visible_bool(cmd)
        & (min_foot_ball_d < float(near_radius))
        & (max_foot_speed > float(foot_speed_thresh))
        & (ball_speed < float(max_ball_speed))
    )
    return miss.float()


def visible_pre_kick_time(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    require_visible: bool = True,
) -> torch.Tensor:
    """Per-step pre-contact timer penalty, optionally gated by perception.

    Caller should use a negative weight. Gating by visible ball keeps the policy
    from being punished for short perception dropouts; bounded lost-ball search
    terms handle that case.
    """
    cmd = _cmd(env, command_name)
    pre_kick = (~cmd.kick_contact_awarded).float()
    if bool(require_visible):
        pre_kick = pre_kick * cmd.ball_mask_perceived
    return pre_kick


def kick_success(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """One-shot reward when ball speed along target direction crosses threshold.

    Detects the rising edge of ``kick_success_awarded`` via ``_emit_once``.
    """
    cmd = _cmd(env, command_name)
    flag = _emit_once(cmd, "_kick_success_emitted", cmd.kick_success_awarded)
    return flag.float()


def near_foot_kick(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Encourage kicking with the foot on the ball's spawn side.

    Fires once at the kick-contact event: +1 when the contacting (closest) foot
    matches the latched near side, -1 when the far foot is used, 0 otherwise.
    The near side is the side the ball spawned on relative to the robot.
    """
    cmd = _cmd(env, command_name)
    fresh = cmd.kick_contact_new.float()
    match = cmd.contact_foot_is_left == cmd.near_foot_is_left
    return fresh * (match.float() * 2.0 - 1.0)


def goal_scored_reward(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """One-shot reward when the ball enters the goal mouth (shoot mode only).

    The latch ``cmd.goal_awarded`` is already gated by ``is_shoot`` inside
    :meth:`SoccerKickCommand._update_command`, but we also AND with the
    current ``is_shoot`` mask defensively so a stale latch can't fire in a
    pass-mode episode.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_goal_emitted", cmd.goal_awarded)
    return (fresh & cmd.is_shoot).float()


def pass_landing_reward(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """One-shot reward when a pass lands inside the target radius (pass mode only).

    Detects the rising edge of ``cmd.pass_landing_awarded`` via ``_emit_once``.
    The latch is gated to pass-mode in the command term, but we AND with
    ``~is_shoot`` again here for safety.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_pass_landing_emitted", cmd.pass_landing_awarded)
    return (fresh & (~cmd.is_shoot)).float()


def kick_angle_error(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Penalize the angular error between ball-velocity and target direction.

    Active only in a short window after first contact, weighted by ball speed.
    Linear in angle² (mjlab V1.30 lesson: exp form fails to learn here).
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    active = (ssk >= 0) & (ssk < 15)
    ball_xy_speed = torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)
    fast_enough = ball_xy_speed > 1.0
    # angle between ball velocity and target_dir_w
    ball_dir_x = cmd.ball_vel_w[:, 0]
    ball_dir_y = cmd.ball_vel_w[:, 1]
    target_x = cmd.target_dir_w[:, 0]
    target_y = cmd.target_dir_w[:, 1]
    dot = (ball_dir_x * target_x + ball_dir_y * target_y) / ball_xy_speed.clamp_min(1e-6)
    dot = dot.clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    err = angle * angle * ball_xy_speed
    mask = (active & fast_enough).float()
    return mask * err  # caller applies negative weight


def kick_strength_error(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Penalize abs error between achieved ball speed at contact and commanded strength.

    Emitted once per episode on the first contact step.
    """
    cmd = _cmd(env, command_name)
    ball_xy_speed = torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)
    err = torch.abs(ball_xy_speed - cmd.target_strength)
    return cmd.kick_contact_new.float() * err  # weight negative


# =========================================================================
# Auxiliary / posture rewards
# =========================================================================


def pre_kick_body_yaw_alignment(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize misalignment between robot yaw and target dir (pre-kick only).

    Forces the policy to physically turn before kicking.
    """
    cmd = _cmd(env, command_name)
    pre_kick = (~cmd.kick_contact_awarded).float()
    # cos(theta) between body x-axis and target direction (in body-yaw frame).
    cos_err = cmd.target_dir_b[:, 0].clamp(-1.0, 1.0)
    err2 = (1.0 - cos_err) ** 2
    return pre_kick * err2  # weight negative


def head_yaw_alignment_to_ball(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    yaw_joint_name: str = "AAHead_yaw",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared diff between head yaw and angle to ball in body-yaw frame."""
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    j_idx = asset.joint_names.index(yaw_joint_name)
    head_yaw = asset.data.joint_pos[:, j_idx]
    desired_yaw = torch.atan2(cmd.ball_pos_b[:, 1], cmd.ball_pos_b[:, 0].clamp_min(1e-3))
    err = head_yaw - desired_yaw
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-pi, pi]
    return cmd.ball_mask_perceived * err * err


def head_pitch_alignment_to_ball(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    pitch_joint_name: str = "Head_pitch",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared diff between head pitch and elevation-to-ball.

    K1 head pitch is positive when looking down. Larger distance ↦ smaller pitch.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    j_idx = asset.joint_names.index(pitch_joint_name)
    head_pitch = asset.data.joint_pos[:, j_idx]
    # Approximate the pitch needed to look at ball: atan2(-z_rel, range_xy).
    # In body-yaw frame, the ball is at (x, y, z_rel).
    # Camera is ~0.4 m above ball when robot stands; pitch_target = atan2(0.4 - z_rel, range)
    range_xy = torch.linalg.norm(cmd.ball_pos_b[:, :2], dim=-1).clamp_min(1e-3)
    desired_pitch = torch.atan2(torch.full_like(range_xy, 0.35), range_xy)
    err = head_pitch - desired_pitch
    return cmd.ball_mask_perceived * err * err


# =========================================================================
# V3.3 — Search / active perception rewards
# =========================================================================


def search_yaw_velocity(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_rate: float = 0.5,
    max_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base-yaw rotation when the ball is NOT currently perceived.

    V3.3 / "search" shaping. Without this term the V3.x policy was observed
    to "freeze" whenever the ball left its head-camera FOV — the GT-driven
    approach reward (now perception-gated) provides no gradient, and AMP
    rewards a standing-still motion prior.

    Returns ``clamp((|w_yaw| - min_rate) / max_rate, 0, 1) * (1 - ball_mask)``.
    Positive only when:
        * the camera does not currently see the ball, AND
        * the robot is rotating at > ``min_rate`` rad/s about its body z-axis.

    The shape is monotonic and saturates at ``max_rate`` rad/s so the policy
    doesn't push yaw-rate to infinity.
    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate = asset.data.root_ang_vel_b[:, 2].abs()
    shaped = torch.clamp((yaw_rate - min_rate) / max_rate, min=0.0, max=1.0)
    not_seen = 1.0 - cmd.ball_mask_perceived
    # V5: searching/turning must not be rewarded once the env is in stop mode.
    active = 1.0 - cmd.stop_mode.float()
    return shaped * not_seen * active


def last_seen_dt_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    max_dt: float = 5.0,
) -> torch.Tensor:
    """Monotonic ramp penalty growing with ``cmd.last_seen_dt``.

    V3.3. Returns ``clamp(last_seen_dt, 0, max_dt) / max_dt``. The caller
    should apply a *negative* weight so a longer "ball lost" interval is
    increasingly costly. Combined with ``search_yaw_velocity`` and
    ``head_yaw_search`` this gives the policy a clear gradient to *find*
    the ball with its head camera rather than standing still.

    Shape: (num_envs,) in ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    penalty = torch.clamp(cmd.last_seen_dt, min=0.0, max=max_dt) / max_dt
    # V5: don't penalize losing sight of the ball once the env is standing still.
    return penalty * (1.0 - cmd.stop_mode.float())


def pre_kick_last_seen_dt_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    max_dt: float = 2.0,
) -> torch.Tensor:
    """Lost-ball timer penalty before contact only.

    A post-kick ball may legitimately leave the kicker camera after a strong
    shot; penalizing that couples high ball speed to perception loss. Keep this
    term focused on the OOD pre-kick freeze/search problem.
    """
    cmd = _cmd(env, command_name)
    pre_kick = (~cmd.kick_contact_awarded).float()
    penalty = torch.clamp(cmd.last_seen_dt, min=0.0, max=max_dt) / max_dt
    return pre_kick * penalty


def head_yaw_search(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    yaw_joint_name: str = "AAHead_yaw",
    min_abs_rate: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward active head-yaw sweeping when the ball is not perceived.

    V3.3. Returns ``clamp(|head_yaw_vel| - min_abs_rate, 0, ...) * (1 - ball_mask)``.
    Encourages the policy to slew its head joint when the ball is out of
    FOV, instead of waiting for the base-yaw rotation to bring the ball
    back into view. Vectorized; no Python loops.

    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    j_idx = asset.joint_names.index(yaw_joint_name)
    head_yaw_vel = asset.data.joint_vel[:, j_idx].abs()
    shaped = torch.clamp(head_yaw_vel - min_abs_rate, min=0.0)
    not_seen = 1.0 - cmd.ball_mask_perceived
    # V5: no head-search reward once the env is in stop mode.
    active = 1.0 - cmd.stop_mode.float()
    return shaped * not_seen * active


def lost_ball_freeze_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_base_yaw_rate: float = 0.35,
    min_head_yaw_rate: float = 0.35,
    yaw_joint_name: str = "AAHead_yaw",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still while the ball is not currently perceived.

    This addresses the OOD-looking "zero ball observation -> freeze" failure.
    Active only before a kick has been awarded; after contact the policy should
    recover rather than keep scanning.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    j_idx = asset.joint_names.index(yaw_joint_name)
    base_yaw_rate = asset.data.root_ang_vel_b[:, 2].abs()
    head_yaw_rate = asset.data.joint_vel[:, j_idx].abs()
    frozen = (base_yaw_rate < float(min_base_yaw_rate)) & (
        head_yaw_rate < float(min_head_yaw_rate)
    )
    pre_kick = (~cmd.kick_contact_awarded).float()
    not_seen = 1.0 - cmd.ball_mask_perceived
    return pre_kick * not_seen * frozen.float()


def lost_ball_head_sweep_pose(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    yaw_joint_name: str = "AAHead_yaw",
    max_abs_yaw: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a non-centered head pose during pre-kick lost-ball search."""
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    j_idx = asset.joint_names.index(yaw_joint_name)
    head_yaw = asset.data.joint_pos[:, j_idx].abs()
    sweep = torch.clamp(head_yaw / max(float(max_abs_yaw), 1e-6), min=0.0, max=1.0)
    pre_kick = (~cmd.kick_contact_awarded).float()
    not_seen = 1.0 - cmd.ball_mask_perceived
    return pre_kick * not_seen * sweep


def support_foot_proximity(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 0.20,
    sigma: float = 0.10,
) -> torch.Tensor:
    """Reward the *non-kicking* (closer-to-ball-on-ground) foot to plant near ball.

    Active *just before* kick (pre-kick window). Encourages a proper plant-step.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids, _ = asset.find_bodies(list(foot_body_names), preserve_order=True)
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)
    ball = cmd.ball_pos_w.unsqueeze(1)
    foot_ball_d = torch.linalg.norm(foot_pos - ball, dim=-1)  # (N, 2)
    # support foot = farther-from-ball foot (the planted leg)
    support_d = foot_ball_d.amax(dim=-1)
    err = support_d - target_distance
    pre_kick = (~cmd.kick_contact_awarded).float()
    return pre_kick * torch.exp(-(err * err) / (sigma * sigma))


def feet_proximity_penalty(
    env: "ManagerBasedRLEnv",
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 0.18,
) -> torch.Tensor:
    """Penalize feet being closer than ``min_dist`` (anti-tangle)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids, _ = asset.find_bodies(list(foot_body_names), preserve_order=True)
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]
    diff = foot_pos[:, 0, :] - foot_pos[:, 1, :]
    d = torch.linalg.norm(diff, dim=-1)
    return torch.clamp(min_dist - d, min=0.0)


def pelvis_orientation_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize trunk tilt — squared norm of horizontal projected-gravity component."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    return torch.sum(proj_grav[:, :2] ** 2, dim=-1)


def stand_still(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    lin_sigma: float = 0.5,
    ang_sigma: float = 0.5,
    grace_steps: int = 25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward standing still while the env is in post-kick stop mode.

    V5. Active only when ``cmd.stop_mode`` is set (latched on the first kick
    contact). Rewards low base horizontal linear speed and low base yaw rate
    via a Gaussian kernel, so the policy settles in place after the kick
    instead of chasing the ball. Returns 0 in kick mode, so this term never
    competes with the approach/kick shaping before the kick.

    ``grace_steps`` delays the stillness demand by that many control steps
    after contact (``steps_since_kick``). Since the latch fires on the contact
    step — the most dynamically unstable moment (one foot planted, kicking leg
    mid-swing) — demanding an instant base stop made the robot brake mid-kick
    and fall. The grace window lets the kick follow-through and balance
    recovery finish before the stand-still pressure kicks in.

    Shape: (num_envs,) in ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    yaw_rate = asset.data.root_ang_vel_b[:, 2].abs()
    settle = torch.exp(-(lin_speed**2) / (lin_sigma**2)) * torch.exp(
        -(yaw_rate**2) / (ang_sigma**2)
    )
    after_grace = (cmd.steps_since_kick >= int(grace_steps)).float()
    return cmd.stop_mode.float() * after_grace * settle


def joint_deviation_in_stop(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    grace_steps: int = 25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 deviation of all joints from their default pose, gated on stop mode.

    V5. Complements ``stand_still`` (which only constrains the base): a frozen
    base can still leave the limbs in an arbitrary post-kick pose. This term
    drives the whole body back to the default standing posture once the env has
    latched into stop mode. Zero before the kick, so it never interferes with
    the kicking motion. Use a *negative* weight (it returns a non-negative
    deviation magnitude).

    Like ``stand_still`` it honors ``grace_steps``: the return-to-default
    demand is suppressed for ``grace_steps`` control steps after contact so the
    kicking leg can finish its swing without being penalized for leaving the
    default pose mid-kick.

    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    deviation = torch.sum(torch.abs(angle), dim=1)
    after_grace = (cmd.steps_since_kick >= int(grace_steps)).float()
    return cmd.stop_mode.float() * after_grace * deviation


def post_kick_alive(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
) -> torch.Tensor:
    """Explicit per-step survival bonus once the env is in post-kick stop mode.

    Returns 1.0 every step ``cmd.stop_mode`` is set (latched on the first kick
    contact), 0 otherwise. Unlike the whole-episode ``alive_reward`` this pays
    only *after* the kick, so it directly rewards staying upright through the
    follow-through and the rest of the episode. A fall ends the episode and
    forfeits this stream, making "kick then stay on your feet" strictly more
    valuable than "kick then fall". No grace gate — survival is rewarded from
    the contact step onward (we want it upright during the unstable phase too).

    Use a positive weight. Shape: (num_envs,) in ``{0, 1}``.
    """
    cmd = _cmd(env, command_name)
    return cmd.stop_mode.float()


def alive_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def terminated_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Returns 1 on the step a non-timeout termination fires; 0 otherwise.

    This relies on the termination manager's ``dones`` and ``time_outs`` buffers.
    """
    if not hasattr(env, "termination_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    dones = env.termination_manager.dones.float()
    timeouts = env.termination_manager.time_outs.float()
    return (dones * (1.0 - timeouts)).clamp_max(1.0)


# =========================================================================
# V4 — Mode-conditional kick rewards (shoot vs pass)
# =========================================================================


def kick_power_progress_shoot(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    floor_speed: float = 3.0,
    window_steps: int = 45,
) -> torch.Tensor:
    """Monotonic shoot power reward: peak speed progresses toward command.

    The V4 Gaussian term is too small when the policy is far below a high
    command (e.g. 6 m/s vs 14 m/s). This linear progress reward keeps a useful
    gradient all the way from weak contact to the requested shoot speed.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    peak = cmd._peak_kick_speed  # type: ignore[attr-defined]
    target = cmd.target_strength.clamp_min(float(floor_speed) + 1.0)
    progress = ((peak - float(floor_speed)) / (target - float(floor_speed))).clamp(
        min=0.0, max=1.0
    )
    return cmd.is_shoot.float() * in_window.float() * progress


def shoot_goal_line_speed(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    floor_speed: float = 3.0,
    window_steps: int = 45,
) -> torch.Tensor:
    """Reward useful shoot speed projected toward the live goal center."""
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    env_origins = env.scene.env_origins
    goal_x = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
    goal_y = env_origins[:, 1]
    rel_x = goal_x - cmd.ball_pos_w[:, 0]
    rel_y = goal_y - cmd.ball_pos_w[:, 1]
    rel_norm = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(1e-3)
    ux = rel_x / rel_norm
    uy = rel_y / rel_norm
    proj = cmd.ball_vel_w[:, 0] * ux + cmd.ball_vel_w[:, 1] * uy
    target = cmd.target_strength.clamp_min(float(floor_speed) + 1.0)
    useful = ((proj - float(floor_speed)) / (target - float(floor_speed))).clamp(
        min=0.0, max=1.0
    )
    return cmd.is_shoot.float() * in_window.float() * useful


def shoot_underpowered_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    target_fraction: float = 0.75,
    window_steps: int = 45,
) -> torch.Tensor:
    """Mild hinge penalty when a shoot attempt remains far below command."""
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    peak = cmd._peak_kick_speed  # type: ignore[attr-defined]
    target = cmd.target_strength.clamp_min(1e-3)
    shortfall = (float(target_fraction) - peak / target).clamp(min=0.0, max=1.0)
    return cmd.is_shoot.float() * in_window.float() * shortfall


def post_kick_upright_recovery(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    start_step: int = 8,
    end_step: int = 80,
    min_height: float = 0.45,
    sigma: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward returning to an upright stance after impact."""
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ssk = cmd.steps_since_kick
    in_window = (ssk >= int(start_step)) & (ssk < int(end_step))
    tilt2 = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=-1)
    upright = torch.exp(-tilt2 / (float(sigma) * float(sigma)))
    height_ok = (asset.data.root_pos_w[:, 2] > float(min_height)).float()
    return in_window.float() * height_ok * upright


def post_kick_fall_risk_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    start_step: int = 0,
    end_step: int = 90,
    min_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilt, low base height, and violent roll/pitch after contact."""
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ssk = cmd.steps_since_kick
    in_window = (ssk >= int(start_step)) & (ssk < int(end_step))
    tilt2 = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=-1)
    low_height = (float(min_height) - asset.data.root_pos_w[:, 2]).clamp(min=0.0)
    roll_pitch_rate = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=-1)
    risk = tilt2 + 4.0 * low_height * low_height + 0.05 * roll_pitch_rate
    return in_window.float() * risk


def goal_scored_quick(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    base_bonus: float = 30.0,
    time_bonus: float = 50.0,
) -> torch.Tensor:
    """Time-weighted goal-scored bonus (shoot mode only).

    Fires once when the ball enters the goal in a shoot-mode episode. The
    returned value is ``base_bonus + time_bonus * (1 - elapsed / max_steps)``
    so earlier goals are worth strictly more, encouraging the policy to score
    fast.

    Replaces the legacy ``goal_scored_reward`` for V4 Stage 1. The caller
    should set ``weight=1.0`` (the magnitude lives in this function).

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_goal_quick_emitted", cmd.goal_awarded)
    mask = (fresh & cmd.is_shoot).float()
    max_steps = max(int(env.max_episode_length), 1)
    elapsed = env.episode_length_buf.float()
    frac_remaining = (1.0 - elapsed / max_steps).clamp(min=0.0, max=1.0)
    return mask * (base_bonus + time_bonus * frac_remaining)


def time_to_goal_penalty(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Constant per-step pressure to score quickly (shoot mode only, pre-goal).

    Returns 1.0 every step a shoot-mode episode has not yet scored, 0 otherwise.
    The caller should apply a small negative weight (e.g. ``-0.02``) so the
    cumulative penalty grows linearly with time.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    not_scored = ~cmd.goal_awarded
    mask = (cmd.is_shoot & not_scored).float()
    return mask


def kick_success_first_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    first_bonus: float = 10.0,
    repeat_bonus: float = 3.0,
) -> torch.Tensor:
    """First kick_success worth more than later ones (shoot mode only).

    The first time the ``kick_success`` latch flips on inside a shoot-mode
    episode awards ``first_bonus``; subsequent kick_success events (after the
    policy decides to attempt another shot — currently impossible in this env
    since the latch never clears intra-episode, but the structure is here for
    when a multi-attempt latch is introduced) award ``repeat_bonus``.

    Today this acts as a strong first-touch bonus that decays the
    successive-kick reward to ``repeat_bonus``. We track the per-env
    "first-kick-already-emitted" flag on the command term.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_kick_success_first_emitted", cmd.kick_success_awarded)
    # Track whether the first success has been awarded yet for this episode.
    first_done = getattr(cmd, "_kick_success_first_done", None)
    if first_done is None or first_done.shape != fresh.shape:
        first_done = torch.zeros_like(fresh, dtype=torch.bool)
        setattr(cmd, "_kick_success_first_done", first_done)
    reset = env.episode_length_buf <= 1
    first_done.copy_(first_done & ~reset)
    is_first = fresh & ~first_done
    is_repeat = fresh & first_done
    first_done.copy_(first_done | fresh)
    shoot_mask = cmd.is_shoot.float()
    return shoot_mask * (is_first.float() * first_bonus + is_repeat.float() * repeat_bonus)


def kick_strength_error_shoot(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Penalty on |achieved - cmd| strength, shoot-mode only.

    Weak version (caller supplies a small negative weight) — shoot allows
    fail-and-retry, so the strength fidelity demand is light.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    ball_xy_speed = torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)
    err = torch.abs(ball_xy_speed - cmd.target_strength)
    return cmd.kick_contact_new.float() * cmd.is_shoot.float() * err


def kick_strength_error_pass(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Penalty on |achieved - cmd| strength, pass-mode only.

    Strong version (caller supplies a large negative weight) — passes are
    one-shot, so strength must be right on first touch.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    ball_xy_speed = torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)
    err = torch.abs(ball_xy_speed - cmd.target_strength)
    return cmd.kick_contact_new.float() * (~cmd.is_shoot).float() * err


def multi_kick_penalty(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Penalize a 2nd-or-later foot↔ball contact (pass mode only).

    A new contact event fires the penalty if the kick_contact latch was
    already set when the new contact arrived. We compute this by reading the
    cached "is this a new edge" flag combined with the cumulative-awarded
    latch and a side-channel "already saw one" tracker.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    # ``kick_contact_new`` is True only on the rising edge of contact_awarded,
    # so by construction it fires only on the first touch within an episode.
    # We re-detect 2nd+ touches via foot-ball distance + speed without the
    # rising-edge latching: any "fresh contact-like instant" after the first.
    foot_pos_w = cmd.robot.data.body_pos_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    ball_p = cmd.ball_pos_w.unsqueeze(1)
    foot_ball_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1).amin(dim=-1)
    ball_xy_speed = torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)
    contact_now = (foot_ball_d < cmd.cfg.kick_foot_proximity) & (
        ball_xy_speed > cmd.cfg.kick_ball_speed_thresh
    )
    # 2nd-or-later touch = contact_now AND we already had a previous kick.
    repeat = contact_now & cmd.kick_contact_awarded & ~cmd.kick_contact_new
    # Track edges so we count each repeat-touch only once even if it lingers
    # across multiple frames.
    prev_repeat = getattr(cmd, "_multi_kick_prev_repeat", None)
    if prev_repeat is None or prev_repeat.shape != repeat.shape:
        prev_repeat = torch.zeros_like(repeat, dtype=torch.bool)
        setattr(cmd, "_multi_kick_prev_repeat", prev_repeat)
    repeat_edge = repeat & ~prev_repeat
    prev_repeat.copy_(repeat)
    pass_mask = (~cmd.is_shoot).float()
    return pass_mask * repeat_edge.float()


def approach_after_kick_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the kicker chasing the ball after release (pass mode only).

    Once the first kick has happened (``kick_contact_awarded``) in a
    pass-mode episode, we want the kicker to stay put — chasing the ball is
    a sign of bad pass strength control. Returns the positive amount by
    which the kicker→ball distance is *shrinking* (Δ_dist > 0) post-kick;
    the caller applies a negative weight.

    We track ``prev_dist`` on the command term as a side-channel attribute.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    rel = cmd.ball_pos_w - asset.data.root_pos_w
    dist = torch.linalg.norm(rel[:, :2], dim=-1)
    prev = getattr(cmd, "_approach_after_kick_prev_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = dist.clone()
        setattr(cmd, "_approach_after_kick_prev_dist", prev)
    # Positive when the kicker is getting closer to the ball.
    shrinking = (prev - dist).clamp(min=0.0)
    prev.copy_(dist)
    post_kick = cmd.kick_contact_awarded.float()
    pass_mask = (~cmd.is_shoot).float()
    return pass_mask * post_kick * shrinking


def target_progress_pass_window(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    window_steps: int = 20,
) -> torch.Tensor:
    """Pass-mode replacement for unbounded ``target_progress`` shaping.

    Returns the ball-velocity projection on the target dir for pass-mode
    episodes, but only inside the first ``window_steps`` steps after kick
    contact. Outside that window the reward goes to zero so the kicker is
    not paid for the ball's free flight forever.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    proj = (cmd.ball_vel_w[:, :2] * cmd.target_dir_w).sum(-1)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    pass_mask = (~cmd.is_shoot)
    mask = (pass_mask & in_window).float()
    return mask * torch.clamp(proj, min=-5.0, max=15.0)


def ball_at_target_terminal(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    radius: float = 1.0,
) -> torch.Tensor:
    """Sparse terminal bonus: ball within ``radius`` of pass target at done (pass mode).

    Fires only on the step the env terminates (any done) for a pass-mode
    episode. Returns 1.0 when the condition holds, 0 otherwise. Caller
    supplies the magnitude via weight.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    if not hasattr(env, "termination_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    dones = env.termination_manager.dones
    # Distance from ball to pass target xy.
    d = torch.linalg.norm(cmd.ball_pos_w[:, :2] - cmd.pass_target_pos_w, dim=-1)
    in_radius = d < float(radius)
    return (dones & in_radius & ~cmd.is_shoot).float()


# =========================================================================
# V5 — Bounded hard-kick, recovery, and lost-ball rewards
# =========================================================================


def shoot_power_hard_bounded(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_margin: float = 0.5,
    saturation_margin: float = 4.0,
    window_steps: int = 35,
    min_ball_speed: float = 1.0,
    min_goal_alignment: float = 0.25,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded shoot-mode reward for exceeding commanded peak kick speed.

    Rewards high shoot power without paying for unstable launches:
      * shoot-mode only,
      * active only shortly after kick contact,
      * ball must still be moving and at least roughly goal-directed,
      * reward is multiplied by a recoverable upright/height gate.

    Returns a bounded tensor in ``[0, 1]``. The speed ramp starts at
    ``target_strength + min_margin`` and saturates after
    ``saturation_margin`` additional m/s. Uses the lifetime peak when the
    command term exposes it so multi-attempt shoot episodes keep credit for
    the hardest valid attempt.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    peak = getattr(cmd, "_lifetime_peak_kick_speed", cmd._peak_kick_speed)  # type: ignore[attr-defined]
    lo = cmd.target_strength + float(min_margin)
    hi = lo + float(saturation_margin)
    power = _bounded_ramp(peak, lo, hi)

    ball_speed = _ball_xy_speed(cmd)
    moving = (ball_speed > float(min_ball_speed)).float()
    alignment = _ball_to_goal_velocity_alignment(env, cmd)
    aligned = _bounded_ramp(alignment, float(min_goal_alignment), 1.0)
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    return cmd.is_shoot.float() * in_window.float() * moving * aligned * stable * power


def shoot_goal_speed_bounded(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    min_projected_speed: float = 4.0,
    saturation_speed: float = 12.0,
    window_steps: int = 30,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded dense shoot reward for fast ball velocity toward goal center.

    Unlike angle-only rewards, this pays for the projected speed itself, so
    stronger shots can be preferred. The projected-speed ramp is bounded and
    multiplied by the same recoverable-body gate used by
    :func:`shoot_power_hard_bounded` to avoid rewarding kicks delivered while
    falling.

    Returns ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    ssk = cmd.steps_since_kick
    in_window = (ssk >= 0) & (ssk < int(window_steps))
    goal_dir = _ball_to_goal_dir_w(env, cmd)
    projected_speed = (cmd.ball_vel_w[:, :2] * goal_dir).sum(-1)
    speed_rew = _bounded_ramp(
        projected_speed,
        float(min_projected_speed),
        float(saturation_speed),
    )
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    return cmd.is_shoot.float() * in_window.float() * stable * speed_rew


def post_kick_recovery_stability(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    start_step: int = 2,
    window_steps: int = 90,
    min_ball_speed: float = 0.5,
    require_ball_visible: bool = False,
    target_height: float = 0.57,
    height_std: float = 0.20,
    upright_std: float = 0.35,
    ang_vel_std: float = 2.5,
    lin_vel_std: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded post-kick recovery reward for staying upright and damped.

    ``mode`` may be ``"any"``, ``"shoot"``, or ``"pass"``. The term is
    active after a real kick while the ball is moving faster than
    ``min_ball_speed``. ``require_ball_visible`` can be enabled by cfg variants
    that want this term tied to perception.

    Returns ``[0, 1]`` as a product of upright, height, roll/pitch angular
    velocity, and horizontal base velocity kernels.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ssk = cmd.steps_since_kick
    in_window = (ssk >= int(start_step)) & (ssk < int(window_steps))
    speed_gate = (_ball_xy_speed(cmd) > float(min_ball_speed)).float()
    perception_gate = _ball_visible_gate(cmd, require_ball_visible=require_ball_visible)
    mode_gate = _mode_gate(cmd, mode)

    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - float(target_height))
    ang_error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)
    lin_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=-1)
    stability = (
        torch.exp(-tilt_error / (float(upright_std) ** 2))
        * torch.exp(-height_error / (float(height_std) ** 2))
        * torch.exp(-ang_error / (float(ang_vel_std) ** 2))
        * torch.exp(-lin_error / (float(lin_vel_std) ** 2))
    )
    return mode_gate * in_window.float() * speed_gate * perception_gate * stability


def post_kick_no_fall_bounded(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    start_step: int = 0,
    window_steps: int = 90,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    min_ball_speed: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary bounded reward for remaining recoverable after kick contact.

    This is a lightweight companion to :func:`post_kick_recovery_stability`.
    It is useful when the cfg needs a clear no-fall survival signal without
    adding contact sensors. Returns ``0`` or ``1``.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ssk = cmd.steps_since_kick
    in_window = (ssk >= int(start_step)) & (ssk < int(window_steps))
    speed_gate = (_ball_xy_speed(cmd) > float(min_ball_speed)).float()
    height = asset.data.root_pos_w[:, 2]
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    alive = ((height > float(min_height)) & (tilt < float(max_tilt))).float()
    return _mode_gate(cmd, mode) * in_window.float() * speed_gate * alive


def lost_ball_search_bounded(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    min_lost_dt: float = 0.15,
    max_ball_speed: float = 1.0,
    allow_post_kick: bool = False,
    min_base_yaw_rate: float = 0.25,
    base_yaw_rate_saturation: float = 2.0,
    head_yaw_joint_name: str = "AAHead_yaw",
    min_head_yaw_rate: float = 0.25,
    head_yaw_rate_saturation: float = 1.5,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded active-search reward when perception has lost a slow ball.

    This addresses the lost-ball OOD freeze mode by paying for either base
    yaw sweep or head-yaw sweep while the ball is not perceived. The term is
    zero when virtual perception is disabled, when the ball is still moving
    fast, or after kick contact unless ``allow_post_kick`` is set.

    Returns ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    lost = _ball_lost_mask(cmd, min_lost_dt=min_lost_dt)
    slow_ball = (_ball_xy_speed(cmd) < float(max_ball_speed)).float()
    pre_or_allowed = torch.ones_like(slow_ball)
    if not bool(allow_post_kick):
        pre_or_allowed = (~cmd.kick_contact_awarded).float()

    base_rate = asset.data.root_ang_vel_b[:, 2].abs()
    base_search = _bounded_ramp(
        base_rate,
        float(min_base_yaw_rate),
        float(base_yaw_rate_saturation),
    )
    j_idx = asset.joint_names.index(head_yaw_joint_name)
    head_rate = asset.data.joint_vel[:, j_idx].abs()
    head_search = _bounded_ramp(
        head_rate,
        float(min_head_yaw_rate),
        float(head_yaw_rate_saturation),
    )
    search = torch.maximum(base_search, head_search)
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    return _mode_gate(cmd, mode) * lost.float() * slow_ball * pre_or_allowed * stable * search


def lost_ball_last_seen_search(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    min_lost_dt: float = 0.10,
    max_last_seen_dt: float = 0.90,
    max_ball_speed: float = 1.5,
    allow_post_kick: bool = False,
    head_yaw_joint_name: str = "AAHead_yaw",
    head_yaw_limit: float = 0.95,
    head_yaw_sigma: float = 0.35,
    min_base_yaw_rate: float = 0.20,
    base_yaw_rate_saturation: float = 1.2,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward short-horizon search toward the last perceived ball bearing.

    When ``hold_last_on_miss`` is enabled, the actor still receives the last
    perceived ball xy plus ``last_seen_dt``. This term rewards using that memory:
    keep the head pointed at the last bearing if it is within the neck range, or
    rotate the base in the direction needed to bring a side/back bearing into
    the camera frustum. Longer losses are left to the generic sweep reward.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    lost = _ball_lost_mask(cmd, min_lost_dt=min_lost_dt)
    slow_ball = (_ball_xy_speed(cmd) < float(max_ball_speed)).float()
    pre_or_allowed = torch.ones_like(slow_ball)
    if not bool(allow_post_kick):
        pre_or_allowed = (~cmd.kick_contact_awarded).float()

    # Fade this memory-directed term out as the remembered bearing becomes
    # stale. The monotonic time penalty and reacquire bonus still apply.
    fresh_memory = (
        1.0
        - _bounded_ramp(
            cmd.last_seen_dt,
            float(max_last_seen_dt) * 0.5,
            float(max_last_seen_dt),
        )
    ).clamp(0.0, 1.0)

    last_xy = cmd.ball_pos_b_perceived[:, :2]
    desired_yaw = torch.atan2(last_xy[:, 1], last_xy[:, 0])
    target_head_yaw = desired_yaw.clamp(-float(head_yaw_limit), float(head_yaw_limit))
    j_idx = asset.joint_names.index(head_yaw_joint_name)
    head_yaw = asset.data.joint_pos[:, j_idx]
    head_err = head_yaw - target_head_yaw
    head_err = torch.atan2(torch.sin(head_err), torch.cos(head_err))
    head_quality = torch.exp(-(head_err * head_err) / max(float(head_yaw_sigma) ** 2, 1.0e-6))

    base_needed = (desired_yaw.abs() - float(head_yaw_limit)).clamp_min(0.0)
    base_needed_gate = _bounded_ramp(base_needed, 0.05, 0.50)
    base_yaw_rate = asset.data.root_ang_vel_b[:, 2]
    turn_toward_memory = base_yaw_rate * torch.sign(desired_yaw)
    base_turn_quality = base_needed_gate * _bounded_ramp(
        turn_toward_memory,
        float(min_base_yaw_rate),
        float(base_yaw_rate_saturation),
    )

    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    # If the remembered bearing is outside the neck range, holding the head at
    # the clamp should not satisfy the search objective by itself; the base must
    # rotate toward the remembered ball direction.
    search_quality = torch.maximum((1.0 - base_needed_gate) * head_quality, base_turn_quality)
    return (
        _mode_gate(cmd, mode)
        * lost.float()
        * slow_ball
        * pre_or_allowed
        * fresh_memory
        * stable
        * search_quality
    )


def lost_ball_freeze_penalty_bounded(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    min_lost_dt: float = 0.20,
    max_ball_speed: float = 1.0,
    allow_post_kick: bool = False,
    base_yaw_rate_saturation: float = 1.0,
    head_yaw_joint_name: str = "AAHead_yaw",
    head_yaw_rate_saturation: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded penalty signal for standing still while the ball is lost.

    Caller should use a negative weight. The returned value is largest when
    both base yaw and head yaw are nearly stationary during a lost-ball,
    slow-ball search window.

    Returns ``[0, 1]``.
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    lost = _ball_lost_mask(cmd, min_lost_dt=min_lost_dt)
    slow_ball = (_ball_xy_speed(cmd) < float(max_ball_speed)).float()
    pre_or_allowed = torch.ones_like(slow_ball)
    if not bool(allow_post_kick):
        pre_or_allowed = (~cmd.kick_contact_awarded).float()

    base_rate = asset.data.root_ang_vel_b[:, 2].abs()
    base_motion = _bounded_ramp(base_rate, 0.0, float(base_yaw_rate_saturation))
    j_idx = asset.joint_names.index(head_yaw_joint_name)
    head_rate = asset.data.joint_vel[:, j_idx].abs()
    head_motion = _bounded_ramp(head_rate, 0.0, float(head_yaw_rate_saturation))
    searching = torch.maximum(base_motion, head_motion)
    freeze = (1.0 - searching).clamp(0.0, 1.0)
    return _mode_gate(cmd, mode) * lost.float() * slow_ball * pre_or_allowed * freeze


def lost_ball_reacquire_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    mode: str = "any",
    min_lost_dt: float = 0.20,
    max_ball_speed: float = 1.5,
    allow_post_kick: bool = False,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-shot bounded bonus when the ball is seen again after being lost.

    A small side-channel latch is stored on the command term. It is reset on
    episode start and after a successful reacquisition. Returns ``0`` or ``1``.
    """
    cmd = _cmd(env, command_name)
    lost_now = _ball_lost_mask(cmd, min_lost_dt=min_lost_dt)
    visible = _ball_visible_bool(cmd)
    was_lost = getattr(cmd, "_v5_lost_ball_was_lost", None)
    if was_lost is None or was_lost.shape != lost_now.shape:
        was_lost = torch.zeros_like(lost_now, dtype=torch.bool)
        setattr(cmd, "_v5_lost_ball_was_lost", was_lost)
    reset = env.episode_length_buf <= 1
    was_lost = was_lost & ~reset
    reacquired = visible & was_lost

    next_was_lost = torch.where(visible, torch.zeros_like(was_lost), was_lost | lost_now)
    getattr(cmd, "_v5_lost_ball_was_lost").copy_(next_was_lost)

    slow_ball = (_ball_xy_speed(cmd) < float(max_ball_speed)).float()
    pre_or_allowed = torch.ones_like(slow_ball)
    if not bool(allow_post_kick):
        pre_or_allowed = (~cmd.kick_contact_awarded).float()
    stable = _recoverable_body_gate(env, asset_cfg, min_height=min_height, max_tilt=max_tilt)
    return _mode_gate(cmd, mode) * reacquired.float() * slow_ball * pre_or_allowed * stable


# =========================================================================
# Helpers
# =========================================================================


def _emit_once(cmd: SoccerKickCommand, attr_name: str, latch: torch.Tensor) -> torch.Tensor:
    """Detect rising edge of a latched per-env bool tensor.

    Stores prev state on the command term under ``attr_name`` (created lazily).
    Returns a bool tensor that is True only on the step the latch flips.
    """
    prev = getattr(cmd, attr_name, None)
    if prev is None or prev.shape != latch.shape:
        prev = torch.zeros_like(latch, dtype=torch.bool)
        setattr(cmd, attr_name, prev)
    fresh = latch & ~prev
    prev.copy_(latch)
    return fresh


def _ball_xy_speed(cmd: SoccerKickCommand) -> torch.Tensor:
    return torch.linalg.norm(cmd.ball_vel_w[:, :2], dim=-1)


def _bounded_ramp(
    value: torch.Tensor,
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(lower):
        lower = torch.full_like(value, float(lower))
    if not torch.is_tensor(upper):
        upper = torch.full_like(value, float(upper))
    denom = (upper - lower).clamp_min(1e-6)
    return ((value - lower) / denom).clamp(0.0, 1.0)


def _mode_gate(cmd: SoccerKickCommand, mode: str) -> torch.Tensor:
    if mode == "any":
        return torch.ones_like(cmd.is_shoot, dtype=torch.float)
    if mode == "shoot":
        return cmd.is_shoot.float()
    if mode == "pass":
        return (~cmd.is_shoot).float()
    raise ValueError(f"Unsupported soccer reward mode: {mode!r}")


def _ball_visible_bool(cmd: SoccerKickCommand) -> torch.Tensor:
    if cmd.perception is None:
        return torch.ones_like(cmd.is_shoot, dtype=torch.bool)
    return cmd.ball_mask_perceived >= 0.5


def _ball_visible_or_recent_bool(cmd: SoccerKickCommand, recent_visible_dt: float) -> torch.Tensor:
    if cmd.perception is None:
        return torch.ones_like(cmd.is_shoot, dtype=torch.bool)
    visible = cmd.ball_mask_perceived >= 0.5
    if float(recent_visible_dt) <= 0.0:
        return visible
    recent = cmd.last_seen_dt <= float(recent_visible_dt)
    return visible | recent


def _ball_visible_or_recent_gate(cmd: SoccerKickCommand, recent_visible_dt: float) -> torch.Tensor:
    return _ball_visible_or_recent_bool(cmd, recent_visible_dt).float()


def _ball_visible_gate(
    cmd: SoccerKickCommand,
    require_ball_visible: bool,
) -> torch.Tensor:
    if not bool(require_ball_visible):
        return torch.ones_like(cmd.ball_mask_perceived)
    return _ball_visible_bool(cmd).float()


def _ball_lost_mask(cmd: SoccerKickCommand, min_lost_dt: float) -> torch.Tensor:
    if cmd.perception is None:
        return torch.zeros_like(cmd.is_shoot, dtype=torch.bool)
    return (cmd.ball_mask_perceived < 0.5) & (cmd.last_seen_dt >= float(min_lost_dt))


def _recoverable_body_gate(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    min_height: float,
    max_tilt: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    height_gate = _bounded_ramp(height, float(min_height), float(min_height) + 0.15)
    tilt_gate = (1.0 - _bounded_ramp(tilt, float(max_tilt) * 0.5, float(max_tilt))).clamp(
        0.0, 1.0
    )
    return height_gate * tilt_gate


def _ball_to_goal_dir_w(env: "ManagerBasedRLEnv", cmd: SoccerKickCommand) -> torch.Tensor:
    env_origins = env.scene.env_origins
    goal_xy = torch.zeros_like(cmd.ball_pos_w[:, :2])
    goal_xy[:, 0] = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
    goal_xy[:, 1] = env_origins[:, 1]
    rel = goal_xy - cmd.ball_pos_w[:, :2]
    return rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)


def _ball_to_goal_velocity_alignment(
    env: "ManagerBasedRLEnv", cmd: SoccerKickCommand
) -> torch.Tensor:
    goal_dir = _ball_to_goal_dir_w(env, cmd)
    ball_speed = _ball_xy_speed(cmd).clamp_min(1e-6)
    return ((cmd.ball_vel_w[:, :2] * goal_dir).sum(-1) / ball_speed).clamp(-1.0, 1.0)


def _nearest_foot_velocity_b(cmd: SoccerKickCommand) -> torch.Tensor:
    foot_pos_w = cmd.robot.data.body_pos_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    foot_vel_w = cmd.robot.data.body_lin_vel_w[:, cmd._foot_ids, :]  # type: ignore[attr-defined]
    ball_p = cmd.ball_pos_w.unsqueeze(1)
    nearest = torch.linalg.norm(foot_pos_w - ball_p, dim=-1).argmin(dim=-1)
    env_ids = torch.arange(cmd.num_envs, device=cmd.device)
    vel_w = foot_vel_w[env_ids, nearest]
    return quat_apply_inverse(cmd.robot_yaw_quat, vel_w)


def torque_near_limit(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques that operate close to their effort limit.

    Each joint's applied torque is normalized by its effort limit; the amount by
    which ``|tau| / effort_limit`` exceeds ``threshold`` is squared and summed
    over joints. The penalty is therefore zero in the normal operating band and
    rises sharply as torques approach saturation, discouraging the policy from
    relying on near-peak torque (which is fragile to model/hardware mismatch).

    Note: uses ``applied_torque`` (post-clamp), available for explicit actuators
    such as the K1's ``DelayedPDActuator`` legs/arms.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    tau = asset.data.applied_torque[:, ids]
    lim = asset.data.joint_effort_limits[:, ids].clamp(min=1.0e-6)
    excess = (tau.abs() / lim - threshold).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)
