"""Command term for the single-agent soccer kick task (V1).

Owns:
  * randomized ball spawn pose at episode reset (1-4 m in front of robot)
  * randomized kick target direction in the world (cos/sin) and target strength (m/s)
  * latching kick-contact + kick-success + goal-scored detection
  * privileged ball state in body-yaw frame (read by critic obs / rewards)
  * mode flag (V1 always ``shoot``; V2 will randomize shoot vs pass)

Pure-tensor implementation; no Python loops over envs.
"""
from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import (
        BLUE_ARROW_X_MARKER_CFG,
        GREEN_ARROW_X_MARKER_CFG,
        RED_ARROW_X_MARKER_CFG,
    )

    _MARKERS_AVAILABLE = True
except Exception:  # pragma: no cover — headless / minimal envs
    sim_utils = None  # type: ignore[assignment]
    VisualizationMarkers = None  # type: ignore[assignment]
    VisualizationMarkersCfg = None  # type: ignore[assignment]
    BLUE_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    GREEN_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    RED_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    _MARKERS_AVAILABLE = False

from booster_rl_tasks.assets.objects.soccer import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_HALF_WIDTH,
    GOAL_LINE_X,
    SOCCER_BALL_RADIUS,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerception,
    VirtualPerceptionCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Upper-body (head + arms) joint angles (rad) from the deploy DEFAULT_ANGLES
# standing pose. The walk dataset only carries the 12 leg joints (the gait
# robot's arms/head are fixed), so when a walk state is injected as a kick init
# state these upper-body DOFs are held here (velocity 0). Values are the
# upper-body slice of the firmware DEFAULT_ANGLES array, keyed by K1 joint name.
_DEPLOY_DEFAULT_UPPER_BODY: dict[str, float] = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.3,
    "Left_Shoulder_Roll": -1.374,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -1.2,
    "ARight_Shoulder_Pitch": 0.3,
    "Right_Shoulder_Roll": 1.374,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 1.2,
}


class SoccerKickCommand(CommandTerm):
    """Holds per-env soccer kick state.

    All randomization (ball/robot spawn poses, target direction, target
    strength) happens in :meth:`_resample_command`. Per-step bookkeeping
    (ball state in body frame, kick-contact detection, goal-scored latch) is
    in :meth:`_update_command`.
    """

    cfg: "SoccerKickCommandCfg"

    def __init__(self, cfg: "SoccerKickCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._ball: RigidObject = env.scene[cfg.ball_name]

        # V3: optional receiver — when set, pass-mode envs track this asset's
        # world xy as their pass target rather than sampling a random one.
        self._receiver: Articulation | None = None
        if cfg.receiver_name is not None:
            try:
                self._receiver = env.scene[cfg.receiver_name]
            except KeyError:
                self._receiver = None

        # Resolve foot body indices once.
        foot_ids, _ = self._robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        self._foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.device)

        # V3.2: receiver foot ids (lazily resolved when receiver is attached).
        self._receiver_foot_ids: torch.Tensor | None = None
        if self._receiver is not None:
            recv_foot_ids, _ = self._receiver.find_bodies(
                list(cfg.foot_body_names), preserve_order=True
            )
            self._receiver_foot_ids = torch.as_tensor(
                recv_foot_ids, dtype=torch.long, device=self.device
            )

        N = self.num_envs
        d = self.device

        # Per-env scalars / vectors -----------------------------------------
        self._target_dir_w = torch.zeros(N, 2, device=d)
        self._target_pos_w = torch.zeros(N, 2, device=d)
        self._target_strength = torch.zeros(N, device=d)
        self._target_strength_normalized = torch.zeros(N, device=d)
        self._is_shoot = torch.ones(N, dtype=torch.bool, device=d)

        # V2: pass target xy (world-frame, env-origin-relative for x; world y).
        # Stored as absolute world xy so we can compare against the ball xy
        # directly each step.
        self._pass_target_pos_w = torch.zeros(N, 2, device=d)
        self._pass_landing_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        # Privileged pass-target body-frame buffers (filled in _update_command).
        self._pass_target_dir_b = torch.zeros(N, 2, device=d)
        self._pass_target_dist_b = torch.zeros(N, device=d)

        # Cached per-step world quantities (updated each compute()) ----------
        self._ball_pos_w = torch.zeros(N, 3, device=d)
        self._ball_vel_w = torch.zeros(N, 3, device=d)
        self._robot_pos_w = torch.zeros(N, 3, device=d)
        self._robot_yaw_quat = torch.zeros(N, 4, device=d)
        self._robot_yaw_quat[:, 0] = 1.0  # identity wxyz

        # Privileged / yaw-frame quantities ----------------------------------
        self._ball_pos_b = torch.zeros(N, 3, device=d)
        self._ball_vel_b = torch.zeros(N, 3, device=d)
        self._goal_dir_b = torch.zeros(N, 2, device=d)
        self._target_dir_b = torch.zeros(N, 2, device=d)
        self._goal_pos_b = torch.zeros(N, 2, device=d)

        # Kick-detection latches --------------------------------------------
        self._kick_contact_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._kick_contact_new = torch.zeros(N, dtype=torch.bool, device=d)
        # Near-foot bookkeeping: which foot is on the ball's spawn side (latched
        # at resample) and which foot is currently closest to the ball.
        self._near_foot_is_left = torch.zeros(N, dtype=torch.bool, device=d)
        self._contact_foot_is_left = torch.zeros(N, dtype=torch.bool, device=d)
        self._kick_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        # Episode-lifetime latches for metrics. Multi-attempt shoot clears the
        # reward latches above, but contact/success rates should still mean
        # "ever happened in this episode."
        self._episode_kick_contact_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._episode_kick_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._goal_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._ball_out_of_field = torch.zeros(N, dtype=torch.bool, device=d)
        # V5: post-kick "stop" mode. Latches True once a kick succeeds (when
        # ``enable_stop_after_kick``) and stays set for the rest of the episode.
        # Exposed to the policy via the ``stop_flag`` observation so it can be
        # driven externally at deploy time to command kick vs. stand-still.
        self._stop_mode = torch.zeros(N, dtype=torch.bool, device=d)
        self._steps_since_kick = torch.full((N,), -1, dtype=torch.long, device=d)
        self._peak_kick_speed = torch.zeros(N, device=d)
        # V4.3: lifetime peak (per-episode max, never reset by multi-attempt).
        # ``_peak_kick_speed`` is reset every multi-attempt re-trigger so the
        # display reflects the *current attempt's* peak; the lifetime variant
        # tracks the BEST kick within the entire episode for monitoring.
        self._lifetime_peak_kick_speed = torch.zeros(N, device=d)
        self._kick_contact_pos_w = torch.zeros(N, 3, device=d)

        # V3.2: trap-success latch (one-shot, ball settled at a receiver foot).
        self._trap_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)

        # Metrics for logging ------------------------------------------------
        # Per-episode EMA buffers (updated in ``_resample_command`` before
        # the latches are cleared). The EMA gives a smoothed *per-episode*
        # success rate instead of the per-step-fraction-of-latch-on that the
        # raw latches would average to — necessary because terminal latches
        # like ``goal_awarded`` are True for only ~1 step before reset.
        self._goal_scored_rate_ema = torch.zeros(N, device=d)
        self._kick_contact_rate_ema = torch.zeros(N, device=d)
        self._kick_success_rate_ema = torch.zeros(N, device=d)
        self._pass_landing_rate_ema = torch.zeros(N, device=d)
        self._trap_success_rate_ema = torch.zeros(N, device=d)
        self._kick_contact_rate_ema_shoot = torch.zeros(N, device=d)
        self._kick_contact_rate_ema_pass = torch.zeros(N, device=d)
        self._kick_success_rate_ema_shoot = torch.zeros(N, device=d)
        self._kick_success_rate_ema_pass = torch.zeros(N, device=d)
        self._goal_scored_rate_ema_shoot = torch.zeros(N, device=d)
        self._pass_landing_rate_ema_pass = torch.zeros(N, device=d)
        self._rate_ema_initialized = torch.zeros(N, dtype=torch.bool, device=d)
        self._rate_ema_initialized_shoot = torch.zeros(N, dtype=torch.bool, device=d)
        self._rate_ema_initialized_pass = torch.zeros(N, dtype=torch.bool, device=d)
        # V4.4: lifetime-peak EMA tracked separately per mode. EMA fires only
        # when an episode ends in the matching mode — pass episodes don't
        # pollute the shoot peak history and vice versa.
        self._lifetime_peak_kick_speed_ema_shoot = torch.zeros(N, device=d)
        self._lifetime_peak_kick_speed_ema_pass = torch.zeros(N, device=d)
        self._target_strength_ema_shoot = torch.zeros(N, device=d)
        self._target_strength_ema_pass = torch.zeros(N, device=d)
        self._peak_to_target_ratio_ema_shoot = torch.zeros(N, device=d)
        self._peak_to_target_ratio_ema_pass = torch.zeros(N, device=d)
        self._target_strength_ema_initialized_shoot = torch.zeros(N, dtype=torch.bool, device=d)
        self._target_strength_ema_initialized_pass = torch.zeros(N, dtype=torch.bool, device=d)
        self._peak_metric_ema_initialized_shoot = torch.zeros(N, dtype=torch.bool, device=d)
        self._peak_metric_ema_initialized_pass = torch.zeros(N, dtype=torch.bool, device=d)
        # EMA alpha — 0.03 → effective window of ~30 episodes per env.
        self._rate_ema_alpha: float = 0.03

        self.metrics["kick_contact_rate"] = torch.zeros(N, device=d)
        self.metrics["kick_success_rate"] = torch.zeros(N, device=d)
        # V5: fraction of envs currently latched into post-kick stop mode.
        # Lets us verify in tensorboard that the stop flag actually fires once
        # kicks start succeeding.
        self.metrics["stop_mode_active"] = torch.zeros(N, device=d)
        self.metrics["goal_scored_rate"] = torch.zeros(N, device=d)
        self.metrics["pass_landing_rate"] = torch.zeros(N, device=d)
        self.metrics["ball_visible"] = torch.zeros(N, device=d)
        self.metrics["ball_in_fov"] = torch.zeros(N, device=d)
        self.metrics["ball_occluded"] = torch.zeros(N, device=d)
        self.metrics["ball_in_deadzone"] = torch.zeros(N, device=d)
        self.metrics["ball_detect_prob"] = torch.zeros(N, device=d)
        self.metrics["ball_raw_detected"] = torch.zeros(N, device=d)
        self.metrics["last_seen_dt"] = torch.zeros(N, device=d)
        self.metrics["pre_kick_active"] = torch.zeros(N, device=d)
        self.metrics["pre_kick_ball_visible"] = torch.zeros(N, device=d)
        self.metrics["pre_kick_ball_in_fov"] = torch.zeros(N, device=d)
        self.metrics["pre_kick_last_seen_dt"] = torch.zeros(N, device=d)
        self.metrics["kick_contact_rate_shoot"] = torch.zeros(N, device=d)
        self.metrics["kick_contact_rate_pass"] = torch.zeros(N, device=d)
        self.metrics["kick_success_rate_shoot"] = torch.zeros(N, device=d)
        self.metrics["kick_success_rate_pass"] = torch.zeros(N, device=d)
        self.metrics["goal_scored_rate_shoot"] = torch.zeros(N, device=d)
        self.metrics["pass_landing_rate_pass"] = torch.zeros(N, device=d)
        self.metrics["target_strength"] = torch.zeros(N, device=d)
        self.metrics["target_strength_shoot"] = torch.zeros(N, device=d)
        self.metrics["target_strength_pass"] = torch.zeros(N, device=d)
        self.metrics["peak_kick_speed"] = torch.zeros(N, device=d)
        self.metrics["lifetime_peak_kick_speed"] = torch.zeros(N, device=d)
        # V4.3: per-mode peak speed so we can disambiguate the global
        # ``peak_kick_speed`` which mixes shoot/pass episodes.
        self.metrics["peak_kick_speed_shoot"] = torch.zeros(N, device=d)
        self.metrics["peak_kick_speed_pass"] = torch.zeros(N, device=d)
        self.metrics["peak_to_target_ratio_shoot"] = torch.zeros(N, device=d)
        self.metrics["peak_to_target_ratio_pass"] = torch.zeros(N, device=d)
        self.metrics["trap_success_rate"] = torch.zeros(N, device=d)

        # Virtual perception (optional). When ``cfg.perception`` is None the
        # observations fall back to ground-truth values (V1.0 behaviour). When
        # set, the policy receives noisy/intermittent ball detections matching
        # a head-mounted camera + detector pipeline.
        self._perception: VirtualPerception | None = None
        if cfg.perception is not None:
            self._perception = VirtualPerception(
                cfg=cfg.perception,
                robot=self._robot,
                num_envs=N,
                dt=float(env.step_dt),
                device=d,
            )

        # Perceived ball state buffers (read by observations / debug viz).
        # These mirror the perception module output when it is enabled, and
        # fall back to GT when it is not.
        self._ball_pos_b_perceived = torch.zeros(N, 3, device=d)
        self._ball_mask_perceived = torch.ones(N, device=d)
        self._last_seen_dt_buf = torch.zeros(N, device=d)

        # V4 Step A — ball observation history buffer. Shape:
        # ``(history_len, num_envs, ball_history_dim)`` where each slot stores
        # ``[ball_pos_b_perceived_x, _y, ball_mask, last_seen_dt, ball_speed_b]``.
        # Updated each step inside :meth:`_update_command` and zeroed on reset.
        # Read by ``soccer_role_obs.ball_history_flat`` as an actor observation.
        self._ball_history_len: int = int(cfg.ball_history_len)
        self._ball_history_dim: int = 5
        self._ball_history_buf = torch.zeros(
            self._ball_history_len, N, self._ball_history_dim, device=d
        )

        # --- Walk-state initialization (walk→kick transition) --------------
        # Optional dataset of mid-walk robot states. When loaded, a fraction
        # of envs are reset into a sampled walk state (joint pos/vel + base
        # height/tilt + base velocity) instead of the default standing pose,
        # so the kick policy learns to strike out of a walking gait. See
        # ``_load_init_states`` / ``_resample_command``.
        self._init_states_loaded: bool = False
        if cfg.init_state_dataset_path is not None:
            self._load_init_states(cfg.init_state_dataset_path)

    # --- Walk-state initialization helpers ---------------------------------
    def _load_init_states(self, path: str) -> None:
        """Load a mid-walk state dataset and reorder joints to this robot.

        The dataset is produced by the locomotion ``play.py --dump_states``
        and stores, for each captured frame, the joint pos/vel, base height
        (above the env origin), base roll/pitch (yaw is dropped so the sample
        is reusable at any spawn heading), and the base linear/angular
        velocity expressed in the body frame.
        """
        import os

        d = self.device
        abspath = os.path.expanduser(path)
        data = torch.load(abspath, map_location=d, weights_only=False)

        # Map dataset joint columns into this robot's joint order. Matched
        # joints (the 12 legs) take the live walk values; joints absent from
        # the dataset (the arms/head, which the gait robot keeps fixed) are
        # held at the deploy DEFAULT_ANGLES upper-body pose with zero velocity.
        ds_names = list(data["joint_names"])
        robot_names = list(self._robot.data.joint_names)
        M = int(data["joint_pos"].shape[0])
        J = len(robot_names)

        ds_jp = data["joint_pos"].to(d)
        ds_jv = data["joint_vel"].to(d)

        # Base = deploy default upper-body pose (by name), vel 0. Joints neither
        # in the dataset nor the upper-body table fall back to 0.0.
        base_jp = torch.tensor(
            [_DEPLOY_DEFAULT_UPPER_BODY.get(n, 0.0) for n in robot_names], device=d
        )
        init_jp = base_jp.unsqueeze(0).expand(M, J).clone()
        init_jv = torch.zeros(M, J, device=d)
        matched = []
        for j, name in enumerate(robot_names):
            if name in ds_names:
                c = ds_names.index(name)
                init_jp[:, j] = ds_jp[:, c]
                init_jv[:, j] = ds_jv[:, c]
                matched.append(name)
        unmatched = [n for n in robot_names if n not in ds_names]
        if unmatched:
            print(
                f"[SoccerKickCommand] init-state dataset {abspath!r} has no data for "
                f"{len(unmatched)} robot joints {unmatched}; held at deploy default "
                f"upper-body pose (vel 0)."
            )

        self._init_joint_pos = init_jp.contiguous()
        self._init_joint_vel = init_jv.contiguous()
        self._init_base_height = data["base_height"].to(d).contiguous()
        self._init_roll = data["base_roll"].to(d).contiguous()
        self._init_pitch = data["base_pitch"].to(d).contiguous()
        self._init_lin_vel_b = data["base_lin_vel_b"].to(d).contiguous()
        self._init_ang_vel_b = data["base_ang_vel_b"].to(d).contiguous()
        self._init_count = M
        self._init_states_loaded = self._init_count > 0
        print(
            f"[SoccerKickCommand] Loaded {self._init_count} walk init-states from "
            f"{abspath!r} ({len(matched)}/{J} joints matched, "
            f"init_state_prob={self.cfg.init_state_prob}, "
            f"seed_action={self.cfg.init_state_seed_action})."
        )

    def _seed_action_to_pose(
        self, env_ids_t: torch.Tensor, joint_pos: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Pre-seed the action buffer so the PD setpoint matches ``joint_pos``.

        With ``JointPositionAction`` the applied target is
        ``raw * scale + offset``; we invert that so step-1's ``last_action``
        observation and ``action_rate`` penalty are measured relative to the
        injected pose instead of zero. Only envs flagged in ``mask`` (walk-init
        envs) are seeded; standing-init envs keep the zeroed action buffer.

        Runs inside ``_resample_command``, i.e. *after* ``action_manager.reset``
        has zeroed the buffers (see ``ManagerBasedRLEnv._reset_idx`` order), so
        the seed survives into the next step.
        """
        if not bool(mask.any()):
            return
        am = self._env.action_manager
        term_name = self.cfg.init_state_action_term
        try:
            term = am.get_term(term_name)
        except Exception:
            return  # action term not present — skip seeding gracefully.

        # Target pose for the joints this term controls, in term order.
        joint_ids = term._joint_ids
        pose_term = joint_pos[:, joint_ids]  # (n_reset, action_dim)
        # ``_scale`` / ``_offset`` may be (num_envs_total, action_dim) tensors;
        # index them to the reset envs so they align row-wise with pose_term
        # (otherwise the per-reset rows broadcast against the full env batch).
        scale = term._scale
        offset = term._offset
        if isinstance(scale, torch.Tensor) and scale.dim() == 2:
            scale = scale[env_ids_t]
        if isinstance(offset, torch.Tensor) and offset.dim() == 2:
            offset = offset[env_ids_t]
        raw = (pose_term - offset) / scale  # invert raw*scale+offset = pose

        # Locate this term's slice within the concatenated action vector.
        names = list(am.active_terms)
        dims = list(am.action_term_dim)
        i = names.index(term_name)
        start = int(sum(dims[:i]))
        sl = slice(start, start + dims[i])

        sel = env_ids_t[mask]
        raw_sel = raw[mask]
        am._action[sel, sl] = raw_sel
        am._prev_action[sel, sl] = raw_sel
        term._raw_actions[sel] = raw_sel

    # --- Public properties -------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """Concatenated command vector: target_dir_b (cos,sin), target_strength_norm, is_shoot. (N, 4)"""
        return torch.cat(
            [
                self._target_dir_b,
                self._target_strength_normalized.unsqueeze(-1),
                self._is_shoot.float().unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def target_dir_w(self) -> torch.Tensor:
        return self._target_dir_w

    @property
    def target_pos_w(self) -> torch.Tensor:
        return self._target_pos_w

    @property
    def target_dir_b(self) -> torch.Tensor:
        return self._target_dir_b

    @property
    def target_strength(self) -> torch.Tensor:
        return self._target_strength

    @property
    def target_strength_normalized(self) -> torch.Tensor:
        return self._target_strength_normalized

    @property
    def is_shoot(self) -> torch.Tensor:
        return self._is_shoot

    @property
    def pass_target_pos_w(self) -> torch.Tensor:
        """World-frame xy position of the pass target. Shape: (num_envs, 2)."""
        return self._pass_target_pos_w

    @property
    def pass_landing_awarded(self) -> torch.Tensor:
        return self._pass_landing_awarded

    @property
    def pass_target_dir_b(self) -> torch.Tensor:
        """Unit direction to pass target in body-yaw frame (cos, sin). Zeros in shoot mode."""
        return self._pass_target_dir_b

    @property
    def pass_target_dist_b(self) -> torch.Tensor:
        """Distance to pass target in body-yaw frame. Zeros in shoot mode."""
        return self._pass_target_dist_b

    @property
    def ball_pos_w(self) -> torch.Tensor:
        return self._ball_pos_w

    @property
    def ball_vel_w(self) -> torch.Tensor:
        return self._ball_vel_w

    @property
    def ball_pos_b(self) -> torch.Tensor:
        return self._ball_pos_b

    @property
    def ball_vel_b(self) -> torch.Tensor:
        return self._ball_vel_b

    @property
    def goal_pos_b(self) -> torch.Tensor:
        return self._goal_pos_b

    @property
    def goal_dir_b(self) -> torch.Tensor:
        return self._goal_dir_b

    @property
    def kick_contact_awarded(self) -> torch.Tensor:
        return self._kick_contact_awarded

    @property
    def kick_contact_new(self) -> torch.Tensor:
        return self._kick_contact_new

    @property
    def near_foot_is_left(self) -> torch.Tensor:
        """True for envs whose ball spawned on the robot's left side."""
        return self._near_foot_is_left

    @property
    def contact_foot_is_left(self) -> torch.Tensor:
        """True when the left foot is currently the closest to the ball."""
        return self._contact_foot_is_left

    @property
    def kick_success_awarded(self) -> torch.Tensor:
        return self._kick_success_awarded

    @property
    def stop_mode(self) -> torch.Tensor:
        """Per-env post-kick stop flag (True = stand still, False = kick)."""
        return self._stop_mode

    @property
    def goal_awarded(self) -> torch.Tensor:
        return self._goal_awarded

    @property
    def ball_out_of_field(self) -> torch.Tensor:
        return self._ball_out_of_field

    @property
    def steps_since_kick(self) -> torch.Tensor:
        return self._steps_since_kick

    @property
    def robot(self) -> Articulation:
        return self._robot

    @property
    def ball(self) -> RigidObject:
        return self._ball

    @property
    def receiver(self) -> "Articulation | None":
        """V3: passive receiver articulation, or ``None`` when not configured."""
        return self._receiver

    @property
    def receiver_foot_ids(self) -> "torch.Tensor | None":
        """V3.2: receiver foot body indices, or ``None`` when no receiver configured."""
        return self._receiver_foot_ids

    @property
    def trap_success_awarded(self) -> torch.Tensor:
        """V3.2: per-env one-shot latch — True once the ball has been trapped."""
        return self._trap_success_awarded

    @property
    def robot_yaw_quat(self) -> torch.Tensor:
        """Cached yaw-only quaternion of the kicker's root. Shape: (num_envs, 4)."""
        return self._robot_yaw_quat

    @property
    def ball_pos_b_perceived(self) -> torch.Tensor:
        """Perceived ball xyz in body-yaw frame (z is a constant fill).

        When perception is disabled this matches the GT ``ball_pos_b``.
        Shape: (num_envs, 3).
        """
        return self._ball_pos_b_perceived

    @property
    def ball_mask_perceived(self) -> torch.Tensor:
        """Float mask in {0,1} indicating a perception detection is current.

        When perception is disabled this is identically 1. Shape: (num_envs,).
        """
        return self._ball_mask_perceived

    @property
    def last_seen_dt(self) -> torch.Tensor:
        """Seconds since the most recent perception detection (per env).

        When perception is disabled this is identically 0. Shape: (num_envs,).
        """
        return self._last_seen_dt_buf

    @property
    def perception(self) -> "VirtualPerception | None":
        return self._perception

    @property
    def ball_history(self) -> torch.Tensor:
        """Ball observation history (read by ``soccer_role_obs.ball_history_flat``).

        Time index ``0`` is the oldest frame; index ``history_len-1`` is the
        most recent. Each slot stores
        ``[perceived_ball_x_b, _y_b, ball_mask, last_seen_dt, ball_speed_b]``.

        Shape: ``(history_len, num_envs, 5)``.
        """
        return self._ball_history_buf

    @property
    def ball_history_len(self) -> int:
        return self._ball_history_len

    @property
    def role(self) -> str:
        """Per-agent role string (set via cfg.role). Constant per skill task."""
        return str(self.cfg.role)

    # --- Required overrides -----------------------------------------------
    def _update_metrics(self):
        # Report per-episode EMAs (updated at episode reset in
        # ``_resample_command``), not the raw latch states. Terminal latches
        # like ``goal_awarded`` only stay True for ~1 step before the env
        # resets, so a per-step average of the raw latch ~= 0 even when the
        # policy scores frequently. The EMA captures the "did this just-ended
        # episode achieve X?" signal correctly.
        self.metrics["kick_contact_rate"][:] = self._kick_contact_rate_ema
        self.metrics["kick_success_rate"][:] = self._kick_success_rate_ema
        self.metrics["stop_mode_active"][:] = self._stop_mode.float()
        self.metrics["goal_scored_rate"][:] = self._goal_scored_rate_ema
        self.metrics["pass_landing_rate"][:] = self._pass_landing_rate_ema
        self.metrics["ball_visible"][:] = self._ball_mask_perceived
        if self._perception is not None:
            self.metrics["ball_in_fov"][:] = self._perception.in_fov
            self.metrics["ball_occluded"][:] = self._perception.occluded
            self.metrics["ball_in_deadzone"][:] = self._perception.in_deadzone
            self.metrics["ball_detect_prob"][:] = self._perception.detect_prob
            self.metrics["ball_raw_detected"][:] = self._perception.raw_detected
        else:
            self.metrics["ball_in_fov"][:] = 1.0
            self.metrics["ball_occluded"][:] = 0.0
            self.metrics["ball_in_deadzone"][:] = 0.0
            self.metrics["ball_detect_prob"][:] = 1.0
            self.metrics["ball_raw_detected"][:] = 1.0
        self.metrics["last_seen_dt"][:] = self._last_seen_dt_buf
        pre_kick = (~self._kick_contact_awarded).float()
        self.metrics["pre_kick_active"][:] = pre_kick
        self.metrics["pre_kick_ball_visible"][:] = self._ball_mask_perceived * pre_kick
        self.metrics["pre_kick_ball_in_fov"][:] = self.metrics["ball_in_fov"] * pre_kick
        self.metrics["pre_kick_last_seen_dt"][:] = self._last_seen_dt_buf * pre_kick
        self.metrics["kick_contact_rate_shoot"][:] = self._kick_contact_rate_ema_shoot
        self.metrics["kick_contact_rate_pass"][:] = self._kick_contact_rate_ema_pass
        self.metrics["kick_success_rate_shoot"][:] = self._kick_success_rate_ema_shoot
        self.metrics["kick_success_rate_pass"][:] = self._kick_success_rate_ema_pass
        self.metrics["goal_scored_rate_shoot"][:] = self._goal_scored_rate_ema_shoot
        self.metrics["pass_landing_rate_pass"][:] = self._pass_landing_rate_ema_pass
        self.metrics["target_strength"][:] = self._target_strength
        self.metrics["target_strength_shoot"][:] = self._target_strength_ema_shoot
        self.metrics["target_strength_pass"][:] = self._target_strength_ema_pass
        self.metrics["peak_kick_speed"][:] = self._peak_kick_speed
        self.metrics["lifetime_peak_kick_speed"][:] = self._lifetime_peak_kick_speed
        # V4.4: per-mode peak as **episode-lifetime EMA** (one update per env
        # per episode end, see ``_resample_command``). Cleaner than the
        # scaled current-step approach — each env's EMA only updates with
        # episodes in its own mode, so the displayed mean is the true
        # per-episode lifetime peak averaged across the recent ~30 episodes.
        self.metrics["peak_kick_speed_shoot"][:] = (
            self._lifetime_peak_kick_speed_ema_shoot
        )
        self.metrics["peak_kick_speed_pass"][:] = (
            self._lifetime_peak_kick_speed_ema_pass
        )
        self.metrics["peak_to_target_ratio_shoot"][:] = (
            self._peak_to_target_ratio_ema_shoot
        )
        self.metrics["peak_to_target_ratio_pass"][:] = (
            self._peak_to_target_ratio_ema_pass
        )
        self.metrics["trap_success_rate"][:] = self._trap_success_rate_ema

    def _resample_command(self, env_ids: Sequence[int]):
        # Normalize env_ids to a 1-D long tensor.
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = int(env_ids_t.numel())
        if n == 0:
            return
        d = self.device
        was_shoot = self._is_shoot[env_ids_t].clone()
        prev_target_strength = self._target_strength[env_ids_t].clone()
        prev_lifetime_peak = self._lifetime_peak_kick_speed[env_ids_t].clone()

        # ----- Robot pose --------------------------------------------------
        spawn_x = _uniform(*self.cfg.robot_spawn_x_range, n=n, device=d)
        spawn_y = _uniform(*self.cfg.robot_spawn_y_range, n=n, device=d)
        spawn_yaw = _uniform(*self.cfg.robot_spawn_yaw_range, n=n, device=d)

        env_origins = self._env.scene.env_origins[env_ids_t]

        # ----- Initial body state: standing default vs sampled walk state --
        # Defaults reproduce the original standing reset. When a walk-state
        # dataset is loaded, a per-env Bernoulli draw replaces them with a
        # sampled mid-walk pose/velocity so the kick is trained out of a gait.
        base_z = torch.full((n,), float(self.cfg.robot_spawn_z), device=d)
        base_roll = torch.zeros(n, device=d)
        base_pitch = torch.zeros(n, device=d)
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids_t].clone()
        lin_vel_b = torch.zeros(n, 3, device=d)
        ang_vel_b = torch.zeros(n, 3, device=d)

        use_walk = torch.zeros(n, dtype=torch.bool, device=d)
        if self._init_states_loaded:
            use_walk = torch.rand(n, device=d) < float(self.cfg.init_state_prob)
            if bool(use_walk.any()):
                idx = torch.randint(0, self._init_count, (n,), device=d)
                m = use_walk
                m1 = m.unsqueeze(1)
                base_z = torch.where(m, self._init_base_height[idx], base_z)
                base_roll = torch.where(m, self._init_roll[idx], base_roll)
                base_pitch = torch.where(m, self._init_pitch[idx], base_pitch)
                joint_pos = torch.where(m1, self._init_joint_pos[idx], joint_pos)
                joint_vel = torch.where(m1, self._init_joint_vel[idx], joint_vel)
                lin_vel_b = torch.where(m1, self._init_lin_vel_b[idx], lin_vel_b)
                ang_vel_b = torch.where(m1, self._init_ang_vel_b[idx], ang_vel_b)

        robot_pose = torch.zeros(n, 7, device=d)
        robot_pose[:, 0] = env_origins[:, 0] + spawn_x
        robot_pose[:, 1] = env_origins[:, 1] + spawn_y
        robot_pose[:, 2] = env_origins[:, 2] + base_z
        # Reconstruct orientation from the sampled roll/pitch but the freshly
        # sampled spawn yaw, so the walk sample is reusable at any heading.
        robot_pose[:, 3:7] = quat_from_euler_xyz(base_roll, base_pitch, spawn_yaw)
        self._robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids_t)

        # Body-frame walk velocity → world frame via the reconstructed quat.
        # Standing-init envs keep zero velocity (lin/ang_vel_b are zero there).
        quat_w = robot_pose[:, 3:7]
        root_vel = torch.zeros(n, 6, device=d)
        root_vel[:, 0:3] = quat_apply(quat_w, lin_vel_b)
        root_vel[:, 3:6] = quat_apply(quat_w, ang_vel_b)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # Pre-seed the PD setpoint to the injected pose (walk-init envs only) so
        # step-1's last_action obs / action_rate penalty are continuous.
        if self._init_states_loaded and self.cfg.init_state_seed_action:
            self._seed_action_to_pose(env_ids_t, joint_pos, use_walk)

        # ----- Ball pose ---------------------------------------------------
        ball_dist = _uniform(*self.cfg.ball_spawn_distance_range, n=n, device=d)
        ball_angle = _uniform(*self.cfg.ball_spawn_angle_range, n=n, device=d)
        local_x = ball_dist * torch.cos(ball_angle)
        local_y = ball_dist * torch.sin(ball_angle)
        # Near foot = the side the ball spawned on (body-frame +y is left, so a
        # positive spawn angle puts the ball on the robot's left).
        self._near_foot_is_left[env_ids_t] = ball_angle > 0.0
        cos_y = torch.cos(spawn_yaw)
        sin_y = torch.sin(spawn_yaw)
        offset_x_w = cos_y * local_x - sin_y * local_y
        offset_y_w = sin_y * local_x + cos_y * local_y

        ball_pose = torch.zeros(n, 7, device=d)
        ball_pose[:, 0] = robot_pose[:, 0] + offset_x_w
        ball_pose[:, 1] = robot_pose[:, 1] + offset_y_w
        ball_pose[:, 2] = env_origins[:, 2] + self.cfg.ball_spawn_height
        ball_pose[:, 3] = 1.0
        self._ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids_t)
        self._ball.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=d), env_ids=env_ids_t
        )

        # ----- Mode: shoot vs pass (V2) -----------------------------------
        is_shoot_new = torch.rand(n, device=d) < float(self.cfg.shoot_prob)
        self._is_shoot[env_ids_t] = is_shoot_new

        # ----- Target strength (V4.4: mode-conditional range) ------------
        # Pass mode always uses ``target_strength_range``. Shoot mode uses
        # ``shoot_target_strength_range`` when set, otherwise falls back to
        # ``target_strength_range`` for backward compatibility.
        pass_lo, pass_hi = self.cfg.target_strength_range
        if self.cfg.shoot_target_strength_range is not None:
            shoot_lo, shoot_hi = self.cfg.shoot_target_strength_range
        else:
            shoot_lo, shoot_hi = pass_lo, pass_hi
        # Bias the strength sampling toward the low (weak) end by raising a
        # uniform sample to ``target_strength_sample_exponent``. Exponent 1.0
        # recovers a plain uniform draw; >1 puts more mass near ``lo``.
        exp = float(self.cfg.target_strength_sample_exponent)
        shoot_u = torch.rand(n, device=d) ** exp
        pass_u = torch.rand(n, device=d) ** exp
        shoot_strength = shoot_lo + (shoot_hi - shoot_lo) * shoot_u
        pass_strength = pass_lo + (pass_hi - pass_lo) * pass_u
        target_strength = torch.where(is_shoot_new, shoot_strength, pass_strength)
        self._target_strength[env_ids_t] = target_strength
        # Normalize within the active mode's configured range. The observation
        # remains one scalar plus ``is_shoot``, but 0/1 now mean min/max command
        # for that mode instead of a compressed position in the union range.
        shoot_denom = max(shoot_hi - shoot_lo, 1e-6)
        pass_denom = max(pass_hi - pass_lo, 1e-6)
        shoot_norm = (shoot_strength - shoot_lo) / shoot_denom
        pass_norm = (pass_strength - pass_lo) / pass_denom
        self._target_strength_normalized[env_ids_t] = (
            torch.where(is_shoot_new, shoot_norm, pass_norm).clamp(0.0, 1.0)
        )

        # Robot world xy (env-origin-relative — robot_pose was just written).
        robot_xy_w = robot_pose[:, :2]

        # ----- Shoot-mode target: aim at a point on the goal line --------
        goal_margin = float(self.cfg.goal_half_width) * 0.8
        goal_y = _uniform(-goal_margin, goal_margin, n=n, device=d)
        # World xy of the goal aim-point (env-origin-relative on x).
        shoot_goal_x_w = env_origins[:, 0] + float(self.cfg.goal_line_x)
        shoot_goal_y_w = env_origins[:, 1] + goal_y
        shoot_vec_x = shoot_goal_x_w - robot_xy_w[:, 0]
        shoot_vec_y = shoot_goal_y_w - robot_xy_w[:, 1]

        # ----- Pass-mode target: random xy in the field (env-frame) ------
        # When a receiver asset is attached (V3), use its previous-frame world
        # xy as the pass target instead — the per-step :meth:`_update_command`
        # path will keep this fresh as the receiver moves. ``_update_command``
        # also re-derives ``target_dir_w`` for pass-mode envs each step.
        pass_x_local = _uniform(*self.cfg.pass_target_x_range, n=n, device=d)
        pass_y_local = _uniform(*self.cfg.pass_target_y_range, n=n, device=d)
        pass_x_w = env_origins[:, 0] + pass_x_local
        pass_y_w = env_origins[:, 1] + pass_y_local
        if self._receiver is not None:
            # Receiver may or may not have been reset yet on this step — its
            # ``data.root_pos_w`` reflects last frame's pose, which is fine
            # for use as a target (refined each step in _update_command).
            recv_xy = self._receiver.data.root_pos_w[env_ids_t, :2]
            pass_x_w = recv_xy[:, 0]
            pass_y_w = recv_xy[:, 1]
        pass_vec_x = pass_x_w - robot_xy_w[:, 0]
        pass_vec_y = pass_y_w - robot_xy_w[:, 1]

        # Vectorized selection between shoot vs pass target. The commanded
        # direction is the desired ball travel direction, so derive it from
        # the spawned ball position rather than the robot root position.
        target_pos_x = torch.where(is_shoot_new, shoot_goal_x_w, pass_x_w)
        target_pos_y = torch.where(is_shoot_new, shoot_goal_y_w, pass_y_w)
        self._target_pos_w[env_ids_t, 0] = target_pos_x
        self._target_pos_w[env_ids_t, 1] = target_pos_y
        target_vec_x = target_pos_x - ball_pose[:, 0]
        target_vec_y = target_pos_y - ball_pose[:, 1]
        target_norm = torch.sqrt(target_vec_x * target_vec_x + target_vec_y * target_vec_y).clamp_min(1e-6)
        self._target_dir_w[env_ids_t, 0] = target_vec_x / target_norm
        self._target_dir_w[env_ids_t, 1] = target_vec_y / target_norm

        # Store pass-target absolute world xy (used by landing detection).
        # In shoot mode the buffer is overwritten next pass episode anyway,
        # but we keep it valid (= robot xy) so distance is zero / no spurious
        # landing trigger when is_shoot==True.
        new_pass_x = torch.where(is_shoot_new, robot_xy_w[:, 0], pass_x_w)
        new_pass_y = torch.where(is_shoot_new, robot_xy_w[:, 1], pass_y_w)
        self._pass_target_pos_w[env_ids_t, 0] = new_pass_x
        self._pass_target_pos_w[env_ids_t, 1] = new_pass_y

        # ----- Capture per-episode success EMAs BEFORE clearing latches --
        # This is the only place we can tell "did the just-ended episode
        # achieve X?" — the latches are about to be reset, and terminal
        # latches like ``goal_awarded`` only stay True for ~1 step before
        # reset so per-step averaging would yield ~0.
        alpha = float(self._rate_ema_alpha)
        one_minus = 1.0 - alpha
        just_contact = self._episode_kick_contact_awarded[env_ids_t].float()
        just_success = self._episode_kick_success_awarded[env_ids_t].float()
        just_goal = self._goal_awarded[env_ids_t].float()
        just_pass = self._pass_landing_awarded[env_ids_t].float()
        just_trap = self._trap_success_awarded[env_ids_t].float()
        valid_prev_episode = prev_target_strength > 0.0
        rate_initialized = self._rate_ema_initialized[env_ids_t]
        contact_rate_old = self._kick_contact_rate_ema[env_ids_t]
        success_rate_old = self._kick_success_rate_ema[env_ids_t]
        goal_rate_old = self._goal_scored_rate_ema[env_ids_t]
        pass_rate_old = self._pass_landing_rate_ema[env_ids_t]
        trap_rate_old = self._trap_success_rate_ema[env_ids_t]
        self._kick_contact_rate_ema[env_ids_t] = torch.where(
            valid_prev_episode,
            torch.where(
                rate_initialized,
                alpha * just_contact + one_minus * contact_rate_old,
                just_contact,
            ),
            contact_rate_old,
        )
        self._kick_success_rate_ema[env_ids_t] = torch.where(
            valid_prev_episode,
            torch.where(
                rate_initialized,
                alpha * just_success + one_minus * success_rate_old,
                just_success,
            ),
            success_rate_old,
        )
        self._goal_scored_rate_ema[env_ids_t] = torch.where(
            valid_prev_episode,
            torch.where(
                rate_initialized,
                alpha * just_goal + one_minus * goal_rate_old,
                just_goal,
            ),
            goal_rate_old,
        )
        self._pass_landing_rate_ema[env_ids_t] = torch.where(
            valid_prev_episode,
            torch.where(
                rate_initialized,
                alpha * just_pass + one_minus * pass_rate_old,
                just_pass,
            ),
            pass_rate_old,
        )
        self._trap_success_rate_ema[env_ids_t] = torch.where(
            valid_prev_episode,
            torch.where(
                rate_initialized,
                alpha * just_trap + one_minus * trap_rate_old,
                just_trap,
            ),
            trap_rate_old,
        )
        self._rate_ema_initialized[env_ids_t] = rate_initialized | valid_prev_episode
        shoot_episode = was_shoot & valid_prev_episode
        pass_episode = (~was_shoot) & valid_prev_episode
        rate_initialized_shoot = self._rate_ema_initialized_shoot[env_ids_t]
        rate_initialized_pass = self._rate_ema_initialized_pass[env_ids_t]
        self._kick_contact_rate_ema_shoot[env_ids_t] = torch.where(
            shoot_episode,
            torch.where(
                rate_initialized_shoot,
                alpha * just_contact + one_minus * self._kick_contact_rate_ema_shoot[env_ids_t],
                just_contact,
            ),
            self._kick_contact_rate_ema_shoot[env_ids_t],
        )
        self._kick_contact_rate_ema_pass[env_ids_t] = torch.where(
            pass_episode,
            torch.where(
                rate_initialized_pass,
                alpha * just_contact + one_minus * self._kick_contact_rate_ema_pass[env_ids_t],
                just_contact,
            ),
            self._kick_contact_rate_ema_pass[env_ids_t],
        )
        self._kick_success_rate_ema_shoot[env_ids_t] = torch.where(
            shoot_episode,
            torch.where(
                rate_initialized_shoot,
                alpha * just_success + one_minus * self._kick_success_rate_ema_shoot[env_ids_t],
                just_success,
            ),
            self._kick_success_rate_ema_shoot[env_ids_t],
        )
        self._kick_success_rate_ema_pass[env_ids_t] = torch.where(
            pass_episode,
            torch.where(
                rate_initialized_pass,
                alpha * just_success + one_minus * self._kick_success_rate_ema_pass[env_ids_t],
                just_success,
            ),
            self._kick_success_rate_ema_pass[env_ids_t],
        )
        self._goal_scored_rate_ema_shoot[env_ids_t] = torch.where(
            shoot_episode,
            torch.where(
                rate_initialized_shoot,
                alpha * just_goal + one_minus * self._goal_scored_rate_ema_shoot[env_ids_t],
                just_goal,
            ),
            self._goal_scored_rate_ema_shoot[env_ids_t],
        )
        self._pass_landing_rate_ema_pass[env_ids_t] = torch.where(
            pass_episode,
            torch.where(
                rate_initialized_pass,
                alpha * just_pass + one_minus * self._pass_landing_rate_ema_pass[env_ids_t],
                just_pass,
            ),
            self._pass_landing_rate_ema_pass[env_ids_t],
        )
        self._rate_ema_initialized_shoot[env_ids_t] = (
            rate_initialized_shoot | shoot_episode
        )
        self._rate_ema_initialized_pass[env_ids_t] = (
            rate_initialized_pass | pass_episode
        )
        target_ema_old_shoot = self._target_strength_ema_shoot[env_ids_t]
        target_ema_old_pass = self._target_strength_ema_pass[env_ids_t]
        target_ema_initialized_shoot = self._target_strength_ema_initialized_shoot[env_ids_t]
        target_ema_initialized_pass = self._target_strength_ema_initialized_pass[env_ids_t]
        self._target_strength_ema_shoot[env_ids_t] = torch.where(
            is_shoot_new,
            torch.where(
                target_ema_initialized_shoot,
                alpha * target_strength + one_minus * target_ema_old_shoot,
                target_strength,
            ),
            target_ema_old_shoot,
        )
        self._target_strength_ema_pass[env_ids_t] = torch.where(
            is_shoot_new,
            target_ema_old_pass,
            torch.where(
                target_ema_initialized_pass,
                alpha * target_strength + one_minus * target_ema_old_pass,
                target_strength,
            ),
        )
        self._target_strength_ema_initialized_shoot[env_ids_t] = (
            target_ema_initialized_shoot | is_shoot_new
        )
        self._target_strength_ema_initialized_pass[env_ids_t] = (
            target_ema_initialized_pass | (~is_shoot_new)
        )
        # V4.4: per-mode lifetime-peak EMA. Only update the EMA for the
        # mode this just-ended episode was in. Pass episodes don't update
        # the shoot EMA, so the shoot value reflects only the shoot history.
        peak_at_end = prev_lifetime_peak
        shoot_ema_old = self._lifetime_peak_kick_speed_ema_shoot[env_ids_t]
        pass_ema_old = self._lifetime_peak_kick_speed_ema_pass[env_ids_t]
        ratio_ema_old_shoot = self._peak_to_target_ratio_ema_shoot[env_ids_t]
        ratio_ema_old_pass = self._peak_to_target_ratio_ema_pass[env_ids_t]
        ratio_at_end = (peak_at_end / prev_target_strength.clamp_min(1e-6)).clamp(0.0, 2.0)
        shoot_update = was_shoot & valid_prev_episode
        pass_update = (~was_shoot) & valid_prev_episode
        peak_metric_initialized_shoot = self._peak_metric_ema_initialized_shoot[env_ids_t]
        peak_metric_initialized_pass = self._peak_metric_ema_initialized_pass[env_ids_t]
        self._lifetime_peak_kick_speed_ema_shoot[env_ids_t] = torch.where(
            shoot_update,
            torch.where(
                peak_metric_initialized_shoot,
                alpha * peak_at_end + one_minus * shoot_ema_old,
                peak_at_end,
            ),
            shoot_ema_old,
        )
        self._lifetime_peak_kick_speed_ema_pass[env_ids_t] = torch.where(
            pass_update,
            torch.where(
                peak_metric_initialized_pass,
                alpha * peak_at_end + one_minus * pass_ema_old,
                peak_at_end,
            ),
            pass_ema_old,
        )
        self._peak_to_target_ratio_ema_shoot[env_ids_t] = torch.where(
            shoot_update,
            torch.where(
                peak_metric_initialized_shoot,
                alpha * ratio_at_end + one_minus * ratio_ema_old_shoot,
                ratio_at_end,
            ),
            ratio_ema_old_shoot,
        )
        self._peak_to_target_ratio_ema_pass[env_ids_t] = torch.where(
            pass_update,
            torch.where(
                peak_metric_initialized_pass,
                alpha * ratio_at_end + one_minus * ratio_ema_old_pass,
                ratio_at_end,
            ),
            ratio_ema_old_pass,
        )
        self._peak_metric_ema_initialized_shoot[env_ids_t] = (
            peak_metric_initialized_shoot | shoot_update
        )
        self._peak_metric_ema_initialized_pass[env_ids_t] = (
            peak_metric_initialized_pass | pass_update
        )

        # ----- Reset latches ----------------------------------------------
        self._kick_contact_awarded[env_ids_t] = False
        self._kick_contact_new[env_ids_t] = False
        self._kick_success_awarded[env_ids_t] = False
        self._episode_kick_contact_awarded[env_ids_t] = False
        self._episode_kick_success_awarded[env_ids_t] = False
        self._goal_awarded[env_ids_t] = False
        self._pass_landing_awarded[env_ids_t] = False
        self._ball_out_of_field[env_ids_t] = False
        self._stop_mode[env_ids_t] = False
        self._steps_since_kick[env_ids_t] = -1
        self._peak_kick_speed[env_ids_t] = 0.0
        self._lifetime_peak_kick_speed[env_ids_t] = 0.0
        self._kick_contact_pos_w[env_ids_t] = 0.0
        # V3.2: clear trap latch.
        self._trap_success_awarded[env_ids_t] = False

        # ----- Perception reset (resamples per-env DR coefficients) -------
        if self._perception is not None:
            self._perception.reset(env_ids_t)
            self._ball_pos_b_perceived[env_ids_t] = 0.0
            self._ball_mask_perceived[env_ids_t] = 0.0
            self._last_seen_dt_buf[env_ids_t] = 0.0

        # ----- V4 Step A — clear ball history buffer for reset envs ------
        self._ball_history_buf[:, env_ids_t, :] = 0.0

    def _update_command(self):
        d = self.device

        # ----- Cache world state -----------------------------------------
        self._ball_pos_w = self._ball.data.root_pos_w.clone()
        self._ball_vel_w = self._ball.data.root_lin_vel_w.clone()
        self._robot_pos_w = self._robot.data.root_pos_w.clone()
        self._robot_yaw_quat = yaw_quat(self._robot.data.root_quat_w)

        # ----- V3: refresh pass target from receiver world xy ------------
        # Pass-mode envs track the receiver's actual position; shoot-mode
        # envs keep the goal-aim target dir computed at resample time.
        if self._receiver is not None:
            recv_xy_w = self._receiver.data.root_pos_w[:, :2]
            is_pass = ~self._is_shoot
            pass_mask_xy = is_pass.unsqueeze(-1).float()
            self._pass_target_pos_w = (
                pass_mask_xy * recv_xy_w + (1.0 - pass_mask_xy) * self._pass_target_pos_w
            )
            self._target_pos_w = (
                pass_mask_xy * recv_xy_w + (1.0 - pass_mask_xy) * self._target_pos_w
            )

        # ----- ball xyz in body-yaw frame --------------------------------
        rel_ball_w = self._ball_pos_w - self._robot_pos_w
        self._ball_pos_b = quat_apply_inverse(self._robot_yaw_quat, rel_ball_w)
        self._ball_vel_b = quat_apply_inverse(self._robot_yaw_quat, self._ball_vel_w)

        # ----- Virtual perception update (after ball_pos_w is fresh) ----
        if self._perception is not None:
            self._perception.update(self._robot, self._ball_pos_w)
            self._ball_pos_b_perceived[:, :2] = self._perception.ball_pos_b
            # Constant z fill: ball spawn height (~ ball radius).
            self._ball_pos_b_perceived[:, 2] = self.cfg.ball_spawn_height
            self._ball_mask_perceived = self._perception.ball_mask
            self._last_seen_dt_buf = self._perception.last_seen_dt
        else:
            # GT fallback so observation functions can read the same buffers.
            self._ball_pos_b_perceived = self._ball_pos_b
            self._ball_mask_perceived = torch.ones(self.num_envs, device=d)
            self._last_seen_dt_buf = torch.zeros(self.num_envs, device=d)

        # The target direction is the desired ball travel direction. Keep the
        # command observation and kick-success projection tied to the same
        # ball-to-target vector as the robot and ball move during the episode.
        target_vec_xy = self._target_pos_w - self._ball_pos_w[:, :2]
        target_vec_norm = torch.linalg.norm(target_vec_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        self._target_dir_w = target_vec_xy / target_vec_norm

        # ----- V4 Step A — push current perceived state into history -----
        # Finite-difference ball speed in body-yaw frame computed against the
        # previous (most recent) slot. Gated on both endpoints being visible so
        # we don't push noisy speed estimates across detection dropouts.
        prev_pos = self._ball_history_buf[-1, :, :2]
        prev_mask = self._ball_history_buf[-1, :, 2]
        cur_pos = self._ball_pos_b_perceived[:, :2]
        cur_mask = self._ball_mask_perceived
        dt = float(self._env.step_dt)
        valid_diff = prev_mask * cur_mask
        ball_speed_b = (
            torch.linalg.norm(cur_pos - prev_pos, dim=-1) / max(dt, 1e-6) * valid_diff
        )
        new_slot = torch.stack(
            [
                cur_pos[:, 0],
                cur_pos[:, 1],
                cur_mask,
                self._last_seen_dt_buf,
                ball_speed_b.clamp_max(20.0),
            ],
            dim=-1,
        )
        # Shift left (drop oldest) and append the new slot at the end.
        if self._ball_history_len > 1:
            self._ball_history_buf[:-1] = self._ball_history_buf[1:].clone()
        self._ball_history_buf[-1] = new_slot

        # ----- target dir in body-yaw frame ------------------------------
        target_vec_w = torch.zeros(self.num_envs, 3, device=d)
        target_vec_w[:, 0:2] = self._target_dir_w
        target_b = quat_apply_inverse(self._robot_yaw_quat, target_vec_w)[:, :2]
        norm = torch.linalg.norm(target_b, dim=-1, keepdim=True).clamp_min(1e-6)
        self._target_dir_b = target_b / norm

        # ----- goal direction in body-yaw frame --------------------------
        env_origins = self._env.scene.env_origins
        goal_world_xy = torch.zeros(self.num_envs, 2, device=d)
        goal_world_xy[:, 0] = env_origins[:, 0] + self.cfg.goal_line_x
        goal_world_xy[:, 1] = env_origins[:, 1] + 0.0
        rel_goal_w = torch.zeros(self.num_envs, 3, device=d)
        rel_goal_w[:, 0:2] = goal_world_xy - self._robot_pos_w[:, :2]
        rel_goal_b = quat_apply_inverse(self._robot_yaw_quat, rel_goal_w)
        self._goal_pos_b = rel_goal_b[:, :2]
        gn = torch.linalg.norm(rel_goal_b[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
        self._goal_dir_b = rel_goal_b[:, :2] / gn

        # ----- kick contact detection -----------------------------------
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_ids, :]
        ball_p = self._ball_pos_w.unsqueeze(1)
        foot_ball_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1)
        min_foot_ball_d = foot_ball_d.amin(dim=-1)
        # Foot index order is (left, right); track which foot is closest so the
        # near-foot reward can compare it against the latched spawn side.
        self._contact_foot_is_left = foot_ball_d[:, 0] < foot_ball_d[:, 1]
        ball_xy_speed = torch.linalg.norm(self._ball_vel_w[:, :2], dim=-1)
        contact_now = (min_foot_ball_d < self.cfg.kick_foot_proximity) & (
            ball_xy_speed > self.cfg.kick_ball_speed_thresh
        )
        new_contact = contact_now & ~self._kick_contact_awarded
        self._kick_contact_new = new_contact
        self._kick_contact_awarded = self._kick_contact_awarded | new_contact
        self._episode_kick_contact_awarded = (
            self._episode_kick_contact_awarded | new_contact
        )
        self._kick_contact_pos_w = torch.where(
            new_contact.unsqueeze(-1), self._ball_pos_w, self._kick_contact_pos_w
        )

        # Post-contact step counter
        started = new_contact & (self._steps_since_kick < 0)
        self._steps_since_kick = torch.where(
            started,
            torch.zeros_like(self._steps_since_kick),
            torch.where(
                self._steps_since_kick >= 0,
                self._steps_since_kick + 1,
                self._steps_since_kick,
            ),
        )

        # Track peak ball XY speed (current-attempt — reset on multi-attempt)
        self._peak_kick_speed = torch.maximum(self._peak_kick_speed, ball_xy_speed)
        # V4.3 Track lifetime peak (per-episode max, immune to multi-attempt reset)
        self._lifetime_peak_kick_speed = torch.maximum(
            self._lifetime_peak_kick_speed, ball_xy_speed
        )

        # ----- Kick success latch ---------------------------------------
        ssk = self._steps_since_kick
        in_window = (ssk >= 0) & (ssk < self.cfg.kick_window_steps)
        proj = (self._ball_vel_w[:, :2] * self._target_dir_w).sum(-1)
        base_success_thresh = torch.full_like(
            proj, float(self.cfg.kick_success_speed_thresh)
        )
        shoot_success_thresh = torch.maximum(
            base_success_thresh,
            self._target_strength * float(self.cfg.shoot_success_target_fraction),
        )
        success_thresh = torch.where(self._is_shoot, shoot_success_thresh, base_success_thresh)
        success_now = (
            in_window
            & (~self._kick_success_awarded)
            & (proj > success_thresh)
        )
        self._kick_success_awarded = self._kick_success_awarded | success_now
        self._episode_kick_success_awarded = (
            self._episode_kick_success_awarded | success_now
        )

        # ----- V5 post-kick stop-mode latch -----------------------------
        # Once the foot first contacts the ball, latch the env into stop mode
        # for the rest of the episode. The policy is rewarded for standing
        # still (``stand_still``) while kick/search shaping is gated off.
        # (Latch on first contact rather than kick success.)
        if bool(self.cfg.enable_stop_after_kick):
            self._stop_mode = self._stop_mode | new_contact

        # ----- V4 multi-attempt support (shoot mode only) ---------------
        # When the ball has come to rest AND the foot is away from it, clear
        # the kick latches for shoot-mode envs so a subsequent approach can
        # re-trigger ``kick_contact`` / ``kick_success`` rewards. Pass-mode
        # envs intentionally keep the latch (so ``multi_kick_penalty`` and
        # ``approach_after_kick_penalty`` can fire on second touches).
        # The goal_scored and pass_landing latches stay terminal regardless.
        if bool(self.cfg.enable_multi_attempt_shoot):
            ball_stopped = ball_xy_speed < float(self.cfg.multi_attempt_ball_speed_thresh)
            foot_away = min_foot_ball_d > float(self.cfg.multi_attempt_foot_clear_dist)
            ready_for_next = (
                self._kick_contact_awarded
                & ball_stopped
                & foot_away
                & self._is_shoot
                & (~self._goal_awarded)
                # V5: a stopped env must not re-arm for another attempt — it
                # stays put until episode reset.
                & (~self._stop_mode)
            )
            self._kick_contact_awarded = self._kick_contact_awarded & ~ready_for_next
            self._kick_success_awarded = self._kick_success_awarded & ~ready_for_next
            # Restart the post-kick step counter for any env we just freed up.
            self._steps_since_kick = torch.where(
                ready_for_next,
                torch.full_like(self._steps_since_kick, -1),
                self._steps_since_kick,
            )
            # Reset the peak speed tracker so the next attempt's peak is
            # measured fresh (the metric reflects the *best* kick within the
            # most recent attempt rather than the lifetime max).
            self._peak_kick_speed = torch.where(
                ready_for_next,
                torch.zeros_like(self._peak_kick_speed),
                self._peak_kick_speed,
            )

        # ----- Goal-scored detection ------------------------------------
        bx_local = self._ball_pos_w[:, 0] - env_origins[:, 0]
        by_local = self._ball_pos_w[:, 1] - env_origins[:, 1]
        in_goal = (bx_local > self.cfg.goal_line_x) & (
            torch.abs(by_local) < self.cfg.goal_half_width
        )
        new_goal = in_goal & ~self._goal_awarded & self._is_shoot
        self._goal_awarded = self._goal_awarded | new_goal

        # ----- Pass-target body-frame quantities (privileged) -----------
        # Vector from robot to pass target in world frame.
        pass_rel_w = torch.zeros(self.num_envs, 3, device=d)
        pass_rel_w[:, 0:2] = self._pass_target_pos_w - self._robot_pos_w[:, :2]
        pass_rel_b = quat_apply_inverse(self._robot_yaw_quat, pass_rel_w)
        pass_norm = torch.linalg.norm(pass_rel_b[:, :2], dim=-1).clamp_min(1e-6)
        # Zero out in shoot mode so the critic doesn't get a misleading signal.
        pass_mask = (~self._is_shoot).float().unsqueeze(-1)
        self._pass_target_dir_b = (pass_rel_b[:, :2] / pass_norm.unsqueeze(-1)) * pass_mask
        self._pass_target_dist_b = pass_norm * (~self._is_shoot).float()

        # ----- Pass-landing detection (V2) ------------------------------
        # Ball within radius of pass target, ball speed inside window, and
        # episode is in pass mode. Latch on rising edge.
        ball_to_target_d = torch.linalg.norm(
            self._ball_pos_w[:, :2] - self._pass_target_pos_w, dim=-1
        )
        ball_xy_speed_pass = torch.linalg.norm(self._ball_vel_w[:, :2], dim=-1)
        sp_lo, sp_hi = self.cfg.pass_landing_speed_window
        in_radius = ball_to_target_d < float(self.cfg.pass_landing_radius)
        in_speed = (ball_xy_speed_pass > float(sp_lo)) & (ball_xy_speed_pass < float(sp_hi))
        is_pass = ~self._is_shoot
        new_landing = in_radius & in_speed & is_pass & ~self._pass_landing_awarded
        self._pass_landing_awarded = self._pass_landing_awarded | new_landing

        # ----- Ball-out-of-field flag (used by termination) -------------
        x_out = torch.abs(bx_local) > (FIELD_HALF_LENGTH + 0.5)
        y_out = torch.abs(by_local) > (FIELD_HALF_WIDTH + 0.5)
        self._ball_out_of_field = x_out | y_out

        # ----- V3.2: trap-success latching ------------------------------
        # Ball reaches a receiver foot at low speed in pass mode. Latched
        # so the reward fires once per episode.
        if self._receiver is not None and self._receiver_foot_ids is not None:
            recv_foot_pos_w = self._receiver.data.body_pos_w[
                :, self._receiver_foot_ids, :
            ]
            recv_foot_ball_d = torch.linalg.norm(
                recv_foot_pos_w - self._ball_pos_w.unsqueeze(1), dim=-1
            )
            min_recv_foot_d = recv_foot_ball_d.amin(dim=-1)
            ball_xy_speed_trap = torch.linalg.norm(self._ball_vel_w[:, :2], dim=-1)
            trap_now = (
                (min_recv_foot_d < float(self.cfg.trap_success_radius))
                & (ball_xy_speed_trap < float(self.cfg.trap_success_ball_speed))
                & (~self._is_shoot)
            )
            new_trap = trap_now & ~self._trap_success_awarded
            self._trap_success_awarded = self._trap_success_awarded | new_trap

    # --- Debug visualization ----------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/toggle viewer markers showing target dir, goal dir, and pass target.

        The visualizers are created lazily on the first ``debug_vis=True`` call.
        In headless mode the marker construction may fail (no SimulationApp /
        stage); we swallow that exception so training continues unaffected.
        """
        if debug_vis:
            if not _MARKERS_AVAILABLE:
                return
            if not hasattr(self, "_target_dir_marker"):
                try:
                    show_arrows = bool(getattr(self.cfg, "debug_vis_direction_arrows", False))
                    if show_arrows:
                        target_cfg = RED_ARROW_X_MARKER_CFG.replace(
                            prim_path="/Visuals/Command/soccer_kick_target_dir"
                        )
                        goal_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
                            prim_path="/Visuals/Command/soccer_kick_goal_dir"
                        )
                        pass_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
                            prim_path="/Visuals/Command/soccer_kick_pass_target"
                        )
                        self._target_dir_marker = VisualizationMarkers(target_cfg)
                        self._goal_dir_marker = VisualizationMarkers(goal_cfg)
                        self._pass_target_marker = VisualizationMarkers(pass_cfg)
                    else:
                        self._target_dir_marker = None
                        self._goal_dir_marker = None
                        self._pass_target_marker = None
                    stop_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/soccer_kick_stop_mode",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.09,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    emissive_color=(1.0, 0.05, 0.05),
                                    diffuse_color=(1.0, 0.05, 0.05),
                                ),
                            ),
                        },
                    )
                    self._stop_mode_marker = VisualizationMarkers(stop_cfg)
                except Exception:  # pragma: no cover — headless fallback
                    self._target_dir_marker = None
                    self._goal_dir_marker = None
                    self._pass_target_marker = None
                    self._stop_mode_marker = None
                    return
            for m in (
                getattr(self, "_target_dir_marker", None),
                getattr(self, "_goal_dir_marker", None),
                getattr(self, "_pass_target_marker", None),
                getattr(self, "_stop_mode_marker", None),
            ):
                if m is not None:
                    try:
                        m.set_visibility(True)
                    except Exception:  # pragma: no cover
                        pass
        else:
            for m in (
                getattr(self, "_target_dir_marker", None),
                getattr(self, "_goal_dir_marker", None),
                getattr(self, "_pass_target_marker", None),
                getattr(self, "_stop_mode_marker", None),
            ):
                if m is not None:
                    try:
                        m.set_visibility(False)
                    except Exception:  # pragma: no cover
                        pass

    def _debug_vis_callback(self, event):
        """Per-step viewer update for the kick-target / goal / pass-target markers."""
        if not getattr(self._robot, "is_initialized", True):
            return
        target_marker = getattr(self, "_target_dir_marker", None)
        goal_marker = getattr(self, "_goal_dir_marker", None)
        pass_marker = getattr(self, "_pass_target_marker", None)
        stop_marker = getattr(self, "_stop_mode_marker", None)
        if (
            target_marker is None
            and goal_marker is None
            and pass_marker is None
            and stop_marker is None
        ):
            return

        d = self.device
        N = self.num_envs
        zeros = torch.zeros(N, device=d)

        # --- Kick-target direction arrow (red) ---------------------------
        try:
            if target_marker is not None:
                target_pos = self._robot_pos_w.clone()
                target_pos[:, 2] = target_pos[:, 2] + 0.5
                tgt_yaw = torch.atan2(self._target_dir_w[:, 1], self._target_dir_w[:, 0])
                tgt_quat = quat_from_euler_xyz(zeros, zeros, tgt_yaw)
                target_marker.visualize(translations=target_pos, orientations=tgt_quat)
        except Exception:  # pragma: no cover
            pass

        # --- Goal-direction arrow (green) --------------------------------
        try:
            if goal_marker is not None:
                env_origins = self._env.scene.env_origins
                goal_x_w = env_origins[:, 0] + float(self.cfg.goal_line_x)
                goal_y_w = env_origins[:, 1]
                goal_vec_x = goal_x_w - self._robot_pos_w[:, 0]
                goal_vec_y = goal_y_w - self._robot_pos_w[:, 1]
                goal_pos = self._robot_pos_w.clone()
                goal_pos[:, 2] = goal_pos[:, 2] + 0.4
                goal_yaw = torch.atan2(goal_vec_y, goal_vec_x)
                goal_quat = quat_from_euler_xyz(zeros, zeros, goal_yaw)
                goal_marker.visualize(translations=goal_pos, orientations=goal_quat)
        except Exception:  # pragma: no cover
            pass

        # --- Pass-target marker (blue) -----------------------------------
        # Drawn for every env (anchored at pass-target xy). Pass-target xy in
        # shoot mode is pinned to the robot xy at resample time, so the marker
        # collapses near the robot for shoot envs (acceptable as a debug aid).
        try:
            if pass_marker is not None:
                pass_pos = torch.zeros(N, 3, device=d)
                pass_pos[:, 0] = self._pass_target_pos_w[:, 0]
                pass_pos[:, 1] = self._pass_target_pos_w[:, 1]
                pass_pos[:, 2] = 0.05
                # Point the arrow from robot toward the pass target so the
                # direction is informative; if collinear the yaw collapses to 0.
                pass_vec_x = self._pass_target_pos_w[:, 0] - self._robot_pos_w[:, 0]
                pass_vec_y = self._pass_target_pos_w[:, 1] - self._robot_pos_w[:, 1]
                pass_yaw = torch.atan2(pass_vec_y, pass_vec_x)
                pass_quat = quat_from_euler_xyz(zeros, zeros, pass_yaw)
                pass_marker.visualize(translations=pass_pos, orientations=pass_quat)
        except Exception:  # pragma: no cover
            pass

        # --- Stop-mode indicator (red sphere above the head) -------------
        # Shown only for envs currently latched into post-kick stop mode;
        # other envs are scaled to zero so the sphere disappears. Lets you
        # see at a glance which robots have latched ``stop_mode`` during play.
        try:
            if stop_marker is not None:
                stop_pos = self._robot_pos_w.clone()
                stop_pos[:, 2] = stop_pos[:, 2] + 0.9
                on = self._stop_mode.float().unsqueeze(-1)
                stop_scales = on.expand(N, 3)
                stop_marker.visualize(translations=stop_pos, scales=stop_scales)
        except Exception:  # pragma: no cover
            pass


@configclass
class SoccerKickCommandCfg(CommandTermCfg):
    """Configuration for :class:`SoccerKickCommand`."""

    class_type: type = SoccerKickCommand
    asset_name: str = "robot"
    ball_name: str = "ball"
    # V3: optional name of a receiver articulation in ``env.scene``. When set,
    # pass-mode envs use the receiver's world xy as the live pass target
    # (overriding ``pass_target_{x,y}_range`` sampling). When ``None`` the
    # original V2 random-target behaviour is preserved.
    receiver_name: str | None = None

    # Ball spawn: polar coordinates relative to robot's body-yaw frame.
    ball_spawn_distance_range: tuple[float, float] = (1.0, 3.5)
    ball_spawn_angle_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)  # ±60° forward cone
    ball_spawn_height: float = SOCCER_BALL_RADIUS

    # Robot spawn pose (world frame, relative to env_origin).
    robot_spawn_x_range: tuple[float, float] = (-FIELD_HALF_LENGTH + 1.0, FIELD_HALF_LENGTH - 4.0)
    robot_spawn_y_range: tuple[float, float] = (-FIELD_HALF_WIDTH + 1.0, FIELD_HALF_WIDTH - 1.0)
    robot_spawn_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    robot_spawn_z: float = 0.57  # matches BOOSTER_K1_CFG.init_state.pos.z

    # Kick target in world frame: cos/sin of direction, scalar strength (m/s).
    target_dir_range: tuple[float, float] = (-math.pi, math.pi)
    # Default strength range used by **pass** mode (also fallback for shoot
    # when ``shoot_target_strength_range`` is None). Pass is one-shot
    # accuracy so doesn't need extreme speeds.
    target_strength_range: tuple[float, float] = (3.0, 8.0)
    # V4.4: separate shoot-mode strength range. ``None`` ⇒ use
    # ``target_strength_range``. When set, shoot-mode envs sample their
    # commanded strength from this range (typically wider/higher than pass
    # since the shoot skill aims to drive the ball into a goal that may be
    # 4–10 m away — a higher commanded peak speed gives the policy room to
    # learn powerful kicks). With friction µ=0.6 the ball decelerates at
    # 5.88 m/s²; a 14 m/s peak travels ~16.7 m before stopping, easily
    # reaching the far end of the field.
    shoot_target_strength_range: tuple[float, float] | None = None
    # Bias for the strength sampling. A uniform sample u~U(0,1) is raised to
    # this exponent before mapping into the range, so values >1 oversample the
    # weak (low) end. 1.0 = plain uniform (default, backward compatible).
    target_strength_sample_exponent: float = 1.0

    # Goal definition (used for shoot-mode scoring).
    goal_line_x: float = GOAL_LINE_X
    goal_half_width: float = GOAL_HALF_WIDTH

    # V2: shoot vs pass mode -----------------------------------------------
    # Per-episode probability the kicker is in "shoot" mode (aim at goal).
    # When False, the kicker is in "pass" mode and aims at ``pass_target_pos_w``.
    shoot_prob: float = 0.5
    # Pass target xy sampled uniformly in these ranges (relative to env origin).
    pass_target_x_range: tuple[float, float] = (-2.0, 6.0)
    pass_target_y_range: tuple[float, float] = (-3.5, 3.5)
    # Landing radius (meters) around pass target xy.
    pass_landing_radius: float = 1.0
    # Only count a landing if ball XY speed is in this band — encourages
    # passes with appropriate strength (not a full-power shoot).
    pass_landing_speed_window: tuple[float, float] = (0.5, 4.0)

    # Foot detection for kick contact event.
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link")
    kick_foot_proximity: float = 0.20
    kick_ball_speed_thresh: float = 1.5
    kick_window_steps: int = 10
    kick_success_speed_thresh: float = 2.0
    # Optional shoot-only success hardening for high-power curricula. When
    # positive, shoot-mode ``kick_success`` requires projection above both the
    # fixed threshold and this fraction of the commanded strength. Pass mode
    # remains governed by ``kick_success_speed_thresh``.
    shoot_success_target_fraction: float = 0.0

    # Episode driver — disable timed resampling so resets come only from env reset.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # Optional virtual perception. When None (V1.0 default), policy obs use
    # ground-truth ball state. When a ``VirtualPerceptionCfg`` is supplied,
    # policy obs see noisy/intermittent detections via this module.
    perception: VirtualPerceptionCfg | None = None

    # V3.2: trap-success detection thresholds (pass-mode only). Ball must be
    # within ``trap_success_radius`` meters of a receiver foot AND have an
    # XY speed below ``trap_success_ball_speed`` (m/s) to latch the trap.
    trap_success_radius: float = 0.30
    trap_success_ball_speed: float = 0.4

    # V4 Step A — ball observation history length (number of past frames the
    # actor sees). 10 frames at 50 Hz ≈ 200 ms. Each frame stores 5 scalars,
    # so the flattened history obs is ``history_len * 5`` dims.
    ball_history_len: int = 10

    # V4 — per-agent role (kicker / receiver / defender / idle). Read by
    # :func:`soccer_role_obs.role_one_hot`. For Stage 1 the kicker variant
    # sets ``role="kicker"``; trap / defend stages override it. Used at
    # train time only; deploy code overrides on a per-episode basis.
    role: str = "kicker"

    # V4 — multi-attempt shoot support. When True (default), the
    # ``kick_contact_awarded`` / ``kick_success_awarded`` latches are
    # cleared once the ball has settled (xy-speed < ``multi_attempt_ball_speed_thresh``)
    # AND the foot is at least ``multi_attempt_foot_clear_dist`` from the
    # ball. This lets a shoot-mode policy line up a second attempt and
    # re-earn ``kick_contact`` / ``kick_success`` rewards, supporting the
    # design intent that the shoot skill be robust to missed shots and
    # ball-lost-then-refound situations. Pass-mode envs are unaffected
    # (we want their multi_kick_penalty / approach_after_kick_penalty to
    # keep firing on second touches).
    enable_multi_attempt_shoot: bool = True
    multi_attempt_ball_speed_thresh: float = 0.5
    multi_attempt_foot_clear_dist: float = 0.5

    # V5 — post-kick stop mode. When True, an env latches into "stop" mode on
    # the first successful kick and holds it until episode reset: the policy
    # observes ``stop_flag`` = 1, is rewarded for standing still, and kick /
    # search shaping is gated off. Set False to recover the pure multi-attempt
    # kicking behaviour. At deploy time the ``stop_flag`` observation input is
    # driven externally to switch between kick and stand-still on demand.
    enable_stop_after_kick: bool = True

    # --- Walk-state initialization (walk→kick transition) -----------------
    # Path to a ``.pt`` dataset of mid-walk robot states collected with the
    # locomotion ``play.py --dump_states``. When set, a fraction of envs are
    # reset into a sampled walk state (joint pos/vel + base height/tilt + base
    # velocity) instead of the default standing pose, so the kick policy learns
    # to strike out of a walking gait. ``None`` = original standing reset.
    init_state_dataset_path: str | None = None
    # Per-env probability of using a sampled walk state (vs the default standing
    # pose) at reset. <1.0 mixes walk-init and standing-init episodes, which
    # keeps the from-standstill kick competent too.
    init_state_prob: float = 1.0
    # Pre-seed the action buffer to the injected joint pose so the PD target /
    # last_action obs / action_rate penalty are continuous on the first step
    # (minimizes the setpoint-discontinuity transient). Name of the joint-
    # position action term to seed.
    init_state_seed_action: bool = True
    init_state_action_term: str = "joint_pos"

    # Debug-vis sub-toggles (only matter when ``debug_vis=True``). The
    # direction arrows (target / goal / pass) render as large stretched arrows
    # that clutter the view; default them off so enabling ``debug_vis`` shows
    # only the post-kick stop-mode sphere above the head.
    debug_vis_direction_arrows: bool = False


def _uniform(lo: float, hi: float, *, n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo
