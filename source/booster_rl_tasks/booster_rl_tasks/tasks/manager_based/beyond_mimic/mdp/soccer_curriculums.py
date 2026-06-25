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

    Tracks a per-batch EMA of fall rate (fall_height | fall_tilt). Checks every
    ``check_interval_steps`` of ``common_step_counter`` (which increments by
    ``num_steps_per_env`` per iteration, so ``24 * N`` ≈ every N iterations).

    The level moves *both ways* with hysteresis:
      * UP   if the EMA stays below ``fall_rate_threshold`` for
             ``consecutive_required`` consecutive checks (tighten randomization).
      * DOWN if the EMA stays above ``down_fall_rate_threshold`` for
             ``down_consecutive_required`` consecutive checks (ease off when
             survival regresses).
      * The band between the two thresholds is a deadband that holds the level
        and prevents flapping.

    Levels
    ------
    0  push ±0.3 m/s  (startup defaults, no extra reset-mode rand)
    1  push ±0.5 m/s  gains scale ×[0.9,1.1]  leg mass add ±0.1~0.3 kg
    2  push ±0.8 m/s  gains scale ×[0.85,1.15] leg mass add ±0.2~0.6 kg
    3  push ±1.0 m/s  gains scale ×[0.8,1.2]   leg mass add ±0.3~1.0 kg
    """

    # NOTE: values halved from the original "aggressive" set as a
    # weakened-DR isolation run (diagnosing low contact rate). push/mass/com
    # are additive → halved directly; friction/gains are scales about 1.0 →
    # deviation from 1.0 halved. Restore the original set once contact rate is
    # confirmed to recover.
    LEVELS: list[dict] = [
        # Level 0 — mild baseline (matches EventCfg initial values)
        dict(
            push_vel=0.2,       push_interval=(6.0, 10.0),
            trunk_mass=(-0.075, 0.25),
            leg_mass=(-0.05,    0.1),
            com_xy=0.015,
            friction=(0.9,      1.1),
            gains=(0.95,        1.05),
        ),
        # Level 1
        dict(
            push_vel=0.3,       push_interval=(4.0, 8.0),
            trunk_mass=(-0.1,   0.35),
            leg_mass=(-0.075,   0.2),
            com_xy=0.02,
            friction=(0.825,    1.2),
            gains=(0.9,         1.1),
        ),
        # Level 2
        dict(
            push_vel=0.5,       push_interval=(3.0, 6.0),
            trunk_mass=(-0.15,  0.45),
            leg_mass=(-0.125,   0.3),
            com_xy=0.025,
            friction=(0.75,     1.3),
            gains=(0.85,        1.15),
        ),
        # Level 3 — full randomization (still ~half of the aggressive set)
        dict(
            push_vel=0.75,      push_interval=(3.0, 6.0),
            trunk_mass=(-0.2,   0.6),
            leg_mass=(-0.175,   0.45),
            com_xy=0.03,
            friction=(0.7,      1.4),
            gains=(0.8,         1.2),
        ),
    ]

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.threshold = p.get("fall_rate_threshold", 0.15)
        self.consecutive_required = p.get("consecutive_required", 5)
        self.down_threshold = p.get("down_fall_rate_threshold", 0.30)
        self.down_consecutive_required = p.get("down_consecutive_required", 2)
        self.alpha = p.get("ema_alpha", 0.1)
        self.check_interval = p.get("check_interval_steps", 24 * 20)

        self._level: int = 0
        self._consecutive: int = 0
        self._consecutive_down: int = 0
        self._ema_fall_rate: float = 0.0
        self._last_check_step: int = -1

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        fall_rate_threshold: float = 0.15,
        consecutive_required: int = 5,
        down_fall_rate_threshold: float = 0.30,
        down_consecutive_required: int = 2,
        ema_alpha: float = 0.1,
        check_interval_steps: int = 24 * 20,
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

        # --- 3. Update consecutive counters (hysteresis band) ---
        if self._ema_fall_rate < self.threshold:
            self._consecutive += 1
            self._consecutive_down = 0
        elif self._ema_fall_rate > self.down_threshold:
            self._consecutive_down += 1
            self._consecutive = 0
        else:
            # deadband between the two thresholds: hold the level.
            self._consecutive = 0
            self._consecutive_down = 0

        # --- 4. Level up if survival sustained, down if falls sustained ---
        if (
            self._consecutive >= self.consecutive_required
            and self._level < len(self.LEVELS) - 1
        ):
            self._level += 1
            self._consecutive = 0
            self._apply_level(env)
            print(
                f"[FallRateCurriculum] level ↑ {self._level}"
                f"  (ema_fall_rate={self._ema_fall_rate:.3f})"
            )
        elif (
            self._consecutive_down >= self.down_consecutive_required
            and self._level > 0
        ):
            self._level -= 1
            self._consecutive_down = 0
            self._apply_level(env)
            print(
                f"[FallRateCurriculum] level ↓ {self._level}"
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


class WalkInitProbCurriculum(ManagerTermBase):
    """Fall-rate-gated ramp of the walk-init probability.

    The soccer-kick command resets a fraction ``init_state_prob`` of envs into a
    sampled mid-walk state (see ``SoccerKickCommand._resample_command``). Starting
    that fraction high makes every episode a hard dynamic-balance problem and
    collapses survival; starting it at 0 trains a from-standstill policy that
    never learns the walk→kick transition.

    This curriculum keeps walk-init *always part of the task* (the dataset is
    loaded) but adapts ``init_state_prob`` to the fall rate, mirroring
    ``FallRateDomainRandCurriculum`` with the same hysteresis band:
      * UP   by ``prob_step`` (toward ``prob_max``) when the fall-rate EMA stays
             below ``fall_rate_threshold`` for ``consecutive_required`` checks.
      * DOWN by ``prob_step`` (toward ``prob_min``) when it stays above
             ``down_fall_rate_threshold`` for ``down_consecutive_required``
             checks — so if raising the walk-init fraction hurts survival, it
             backs off automatically.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.command_name = p.get("command_name", "soccer_kick")
        self.threshold = p.get("fall_rate_threshold", 0.15)
        self.consecutive_required = p.get("consecutive_required", 5)
        self.down_threshold = p.get("down_fall_rate_threshold", 0.30)
        self.down_consecutive_required = p.get("down_consecutive_required", 2)
        self.alpha = p.get("ema_alpha", 0.1)
        self.check_interval = p.get("check_interval_steps", 24 * 20)
        self.prob_start = float(p.get("prob_start", 0.0))
        self.prob_step = float(p.get("prob_step", 0.1))
        self.prob_min = float(p.get("prob_min", 0.0))
        self.prob_max = float(p.get("prob_max", 1.0))

        self._prob: float = self.prob_start
        self._consecutive: int = 0
        self._consecutive_down: int = 0
        self._ema_fall_rate: float = 0.0
        self._last_check_step: int = -1
        self._applied: bool = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        command_name: str = "soccer_kick",
        fall_rate_threshold: float = 0.15,
        consecutive_required: int = 5,
        down_fall_rate_threshold: float = 0.30,
        down_consecutive_required: int = 2,
        ema_alpha: float = 0.1,
        check_interval_steps: int = 24 * 20,
        prob_start: float = 0.0,
        prob_step: float = 0.1,
        prob_min: float = 0.0,
        prob_max: float = 1.0,
    ) -> float:
        term = env.command_manager.get_term(self.command_name)
        # Pin the starting probability on the first call (overrides whatever the
        # cfg shipped, so the ramp always begins at ``prob_start``).
        if not self._applied:
            term.cfg.init_state_prob = self._prob
            self._applied = True

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
            return self._prob
        self._last_check_step = step

        # --- 3. Update consecutive counters (hysteresis band) ---
        if self._ema_fall_rate < self.threshold:
            self._consecutive += 1
            self._consecutive_down = 0
        elif self._ema_fall_rate > self.down_threshold:
            self._consecutive_down += 1
            self._consecutive = 0
        else:
            # deadband between the two thresholds: hold the probability.
            self._consecutive = 0
            self._consecutive_down = 0

        # --- 4. Ramp probability up if survival sustained, down if falls spike ---
        if self._consecutive >= self.consecutive_required and self._prob < self.prob_max:
            self._prob = min(self.prob_max, self._prob + self.prob_step)
            self._consecutive = 0
            term.cfg.init_state_prob = self._prob
            print(
                f"[WalkInitProbCurriculum] init_state_prob ↑ {self._prob:.2f}"
                f"  (ema_fall_rate={self._ema_fall_rate:.3f})"
            )
        elif (
            self._consecutive_down >= self.down_consecutive_required
            and self._prob > self.prob_min
        ):
            self._prob = max(self.prob_min, self._prob - self.prob_step)
            self._consecutive_down = 0
            term.cfg.init_state_prob = self._prob
            print(
                f"[WalkInitProbCurriculum] init_state_prob ↓ {self._prob:.2f}"
                f"  (ema_fall_rate={self._ema_fall_rate:.3f})"
            )

        return self._prob


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


class KickErrorWeightCurriculum(ManagerTermBase):
    """Ramp the kick angle/strength error penalties with the kick-contact return.

    The direction/strength penalties should only matter once the policy can
    actually touch the ball. This scales each penalty by an EMA of
    ``Episode_Reward/kick_contact`` (the same quantity logged by the reward
    manager), so the weights grow from ~0 toward full strictness as contact
    becomes reliable::

        weight = base_weight * ema(kick_contact_return) * gain

    ``base_weight`` is each term's configured weight, captured on the first
    call. With ``gain=1000`` and ``kick_contact_return≈0.0089`` the angle
    weight becomes ``-1.0 * 0.0089 * 1000 = -8.9`` and the strength weight
    ``-0.5 * 0.0089 * 1000 = -4.45``.

    Note: the scaling is unbounded by default — set ``max_abs_weight`` to clamp
    the magnitude so a high contact return cannot blow up the penalties, and
    ``min_abs_weight`` to floor the magnitude so a term always keeps a baseline
    strength even before the contact EMA grows (e.g. a fall penalty that must
    bite from the start).

    Alternatively, set ``scale_cap`` to clamp the ``ema * gain`` multiplier
    itself. With ``scale_cap=1.0`` each weight ramps from 0 up to *exactly* its
    configured base value and stops there (never overshoots), which is the
    right behaviour for large-magnitude terms that should reach — but not
    exceed — a target strength as contact becomes reliable (e.g. ``terminated``,
    ``pelvis_orientation``, ``goal_scored``).
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.gain = float(p.get("gain", 1000.0))
        self.alpha = float(p.get("ema_alpha", 0.1))
        self.contact_term = p.get("contact_term", "kick_contact")
        self.scaled_terms = list(
            p.get("scaled_terms", ["kick_angle_error", "kick_strength_error"])
        )
        self.max_abs_weight = p.get("max_abs_weight", None)
        self.min_abs_weight = p.get("min_abs_weight", None)
        self.scale_cap = p.get("scale_cap", None)
        self._ema: float = 0.0
        self._base: dict[str, float] | None = None  # captured on first call

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        gain: float = 1000.0,
        ema_alpha: float = 0.1,
        contact_term: str = "kick_contact",
        scaled_terms: Sequence[str] = ("kick_angle_error", "kick_strength_error"),
        max_abs_weight: float | None = None,
        min_abs_weight: float | None = None,
        scale_cap: float | None = None,
    ) -> float:
        rm = env.reward_manager
        # Capture each scaled term's configured (base) weight once, before we
        # start overwriting it.
        if self._base is None:
            self._base = {n: float(rm.get_term_cfg(n).weight) for n in self.scaled_terms}

        # Replicate Episode_Reward/<contact_term> for the just-finished envs.
        # Curriculum runs before reward_manager.reset, so the per-env episode
        # sums still hold this episode's accumulated (weighted) contact reward.
        sums = rm._episode_sums.get(self.contact_term)
        if sums is not None:
            # ``env_ids`` may be a slice (e.g. ``slice(None)`` when the curriculum
            # is computed for all envs during a resume restore), which has no
            # ``len``. Treat a slice as "all envs".
            n_ids = sums.shape[0] if isinstance(env_ids, slice) else len(env_ids)
            if n_ids > 0:
                kc_batch = float(sums[env_ids].mean().item()) / float(env.max_episode_length_s)
                self._ema = self.alpha * kc_batch + (1.0 - self.alpha) * self._ema

        # Apply the scaled weights live.
        scale = self._ema * self.gain
        if self.scale_cap is not None:
            scale = min(scale, abs(float(self.scale_cap)))
        for n in self.scaled_terms:
            w = self._base[n] * scale
            if self.max_abs_weight is not None:
                cap = abs(float(self.max_abs_weight))
                w = max(-cap, min(cap, w))
            if self.min_abs_weight is not None:
                # Floor the magnitude so the term never ramps below a baseline
                # strength (keeps the base term's sign). Useful for penalties
                # that must always bite even before the contact EMA grows.
                floor = abs(float(self.min_abs_weight))
                if abs(w) < floor:
                    sign = -1.0 if self._base[n] < 0.0 else 1.0
                    w = sign * floor
            rm.get_term_cfg(n).weight = w
        return self._ema
