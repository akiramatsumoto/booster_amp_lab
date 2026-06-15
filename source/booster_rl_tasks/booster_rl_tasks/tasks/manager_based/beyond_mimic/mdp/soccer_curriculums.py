# SPDX-License-Identifier: Apache-2.0

"""Curriculum functions specific to the soccer kick task.

These functions mutate the active :class:`SoccerKickCommandCfg` at runtime to
progressively increase task difficulty. They are stateless w.r.t. the policy
and only read :attr:`ManagerBasedRLEnv.common_step_counter` as progress.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class FallRateDomainRandCurriculum(ManagerTermBase):
    """Fall-rate-gated domain randomization curriculum.

    Tracks a per-batch EMA of fall rate (fall_height | fall_tilt).
    At each check interval (~1 iteration), if the EMA stays below
    ``fall_rate_threshold`` for ``consecutive_required`` consecutive
    checks, the curriculum advances one level and tightens the
    randomization ranges.  A single check above the threshold resets
    the consecutive counter.

    Levels
    ------
    0  push ±0.3 m/s  (startup defaults, no extra reset-mode rand)
    1  push ±0.5 m/s  gains scale ×[0.9,1.1]  leg mass add ±0.1~0.3 kg
    2  push ±0.8 m/s  gains scale ×[0.85,1.15] leg mass add ±0.2~0.6 kg
    3  push ±1.0 m/s  gains scale ×[0.8,1.2]   leg mass add ±0.3~1.0 kg
    """

    LEVELS: list[dict] = [
        # Level 0 — mild baseline (matches EventCfg initial values)
        dict(
            push_vel=0.4,       push_interval=(6.0, 10.0),
            trunk_mass=(-0.15,  0.5),
            leg_mass=(-0.1,     0.2),
            com_xy=0.03,
            friction=(0.8,      1.2),
            gains=(0.9,         1.1),
        ),
        # Level 1
        dict(
            push_vel=0.6,       push_interval=(4.0, 8.0),
            trunk_mass=(-0.2,   0.7),
            leg_mass=(-0.15,    0.4),
            com_xy=0.04,
            friction=(0.65,     1.4),
            gains=(0.8,         1.2),
        ),
        # Level 2
        dict(
            push_vel=1.0,       push_interval=(3.0, 6.0),
            trunk_mass=(-0.3,   0.9),
            leg_mass=(-0.25,    0.6),
            com_xy=0.05,
            friction=(0.5,      1.6),
            gains=(0.7,         1.3),
        ),
        # Level 3 — full randomization
        dict(
            push_vel=1.5,       push_interval=(3.0, 6.0),
            trunk_mass=(-0.4,   1.2),
            leg_mass=(-0.35,    0.9),
            com_xy=0.06,
            friction=(0.4,      1.8),
            gains=(0.6,         1.4),
        ),
    ]

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.threshold = p.get("fall_rate_threshold", 0.15)
        self.consecutive_required = p.get("consecutive_required", 5)
        self.alpha = p.get("ema_alpha", 0.1)
        self.check_interval = p.get("check_interval_steps", 4096 * 24)

        self._level: int = 0
        self._consecutive: int = 0
        self._ema_fall_rate: float = 0.0
        self._last_check_step: int = -1

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        fall_rate_threshold: float = 0.15,
        consecutive_required: int = 5,
        ema_alpha: float = 0.1,
        check_interval_steps: int = 4096 * 24,
    ) -> int:
        # --- 1. Update EMA with this batch's fall rate ---
        fall_h = env.termination_manager.get_term("fall_height")[env_ids]
        fall_t = env.termination_manager.get_term("fall_tilt")[env_ids]
        batch_fall_rate = (fall_h | fall_t).float().mean().item()
        self._ema_fall_rate = (
            self.alpha * batch_fall_rate + (1.0 - self.alpha) * self._ema_fall_rate
        )

        # --- 2. Only evaluate at iteration boundaries ---
        step = int(env.common_step_counter)
        if step - self._last_check_step < self.check_interval:
            return self._level
        self._last_check_step = step

        # --- 3. Update consecutive counter ---
        if self._ema_fall_rate < self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        # --- 4. Level up if sustained ---
        if (
            self._consecutive >= self.consecutive_required
            and self._level < len(self.LEVELS) - 1
        ):
            self._level += 1
            self._consecutive = 0
            self._apply_level(env)
            print(
                f"[FallRateCurriculum] level → {self._level}"
                f"  (ema_fall_rate={self._ema_fall_rate:.3f})"
            )

        return self._level

    def _apply_level(self, env: "ManagerBasedRLEnv") -> None:
        p = self.LEVELS[self._level]

        # push_robot: velocity + interval
        v = p["push_vel"]
        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push_cfg.params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
        push_cfg.interval_range_s = p["push_interval"]

        # trunk mass
        env.event_manager.get_term_cfg("trunk_mass").params[
            "mass_distribution_params"
        ] = p["trunk_mass"]

        # leg mass
        env.event_manager.get_term_cfg("randomize_leg_mass").params[
            "mass_distribution_params"
        ] = p["leg_mass"]

        # base CoM
        c = p["com_xy"]
        env.event_manager.get_term_cfg("base_com").params["com_range"] = {
            "x": (-c, c), "y": (-c, c), "z": (-c * 0.25, c * 0.25)
        }

        # joint friction
        lo, hi = p["friction"]
        env.event_manager.get_term_cfg("joint_friction").params[
            "friction_distribution_params"
        ] = (lo, hi)

        # actuator gains
        lo, hi = p["gains"]
        gains_cfg = env.event_manager.get_term_cfg("randomize_actuator_gains")
        gains_cfg.params["stiffness_distribution_params"] = (lo, hi)
        gains_cfg.params["damping_distribution_params"] = (lo, hi)


def ball_distance_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "soccer_kick",
    lo_start: float = 0.4,
    hi_start: float = 1.2,
    lo_final: float = 1.0,
    hi_final: float = 3.5,
    num_steps_to_final: int = 4000 * 4096,
) -> tuple[float, float]:
    """Linearly grow the ball spawn distance range as training progresses.

    The new ``(lo, hi)`` is written into the command term's cfg in-place; the
    soccer kick command term re-reads ``cfg.ball_spawn_distance_range`` at
    every ``_resample_command`` call, so the new range takes effect on the
    next episode reset (no immediate mid-episode jump).

    Args:
        env: The learning environment.
        env_ids: Not used; the curriculum is global across all envs.
        command_name: Name of the soccer-kick command term to mutate.
        lo_start, hi_start: Initial distance range at step 0.
        lo_final, hi_final: Final (full-difficulty) distance range.
        num_steps_to_final: ``common_step_counter`` value at which the range
            should reach ``(lo_final, hi_final)``. Before this, the range is
            linearly interpolated; after this, it is clamped at the final.

    Returns:
        The current ``(lo, hi)`` tuple (also written into the cfg).
    """
    # Linear interpolation in [0, 1] based on global step counter.
    progress = float(env.common_step_counter) / float(max(num_steps_to_final, 1))
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    lo = lo_start + (lo_final - lo_start) * progress
    hi = hi_start + (hi_final - hi_start) * progress

    # Mutate the live command term cfg. SoccerKickCommand reads this each
    # _resample_command, so the change applies on the next episode reset.
    term = env.command_manager.get_term(command_name)
    term.cfg.ball_spawn_distance_range = (lo, hi)
    return (lo, hi)


def shoot_strength_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "soccer_kick",
    lo_start: float = 8.0,
    hi_start: float = 11.0,
    lo_final: float = 12.0,
    hi_final: float = 15.0,
    num_steps_to_final: int = 3500 * 4096,
) -> tuple[float, float]:
    """Linearly grow the shoot-mode command strength range.

    V5.1 uses this to keep early optimization focused on approach/contact with
    achievable power, then reintroduces the 12-15 m/s hard-shot target once the
    policy has re-stabilized contact.
    """
    progress = float(env.common_step_counter) / float(max(num_steps_to_final, 1))
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    lo = lo_start + (lo_final - lo_start) * progress
    hi = hi_start + (hi_final - hi_start) * progress

    term = env.command_manager.get_term(command_name)
    term.cfg.shoot_target_strength_range = (lo, hi)
    return (lo, hi)
