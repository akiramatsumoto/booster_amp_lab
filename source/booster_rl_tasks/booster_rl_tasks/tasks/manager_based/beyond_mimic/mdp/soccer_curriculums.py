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
    At each check interval (~1 iteration):

    * if the EMA stays below ``fall_rate_threshold`` for
      ``consecutive_required`` consecutive checks, the curriculum advances
      one level and tightens the randomization ranges;
    * if the EMA stays at/above ``fall_rate_threshold`` for
      ``consecutive_drop_required`` consecutive checks, the curriculum drops
      one level and loosens the ranges.

    The "good" and "bad" streak counters are mutually exclusive: a good check
    resets the bad streak and vice versa, so leveling up requires a sustained
    run of healthy iterations while a sustained run of falls walks it back.
    With the defaults (``consecutive_required=100``,
    ``consecutive_drop_required=20``) it takes ~100 iterations to climb a
    level and ~20 to drop one.

    Levels
    ------
    Only the foot-sole ↔ ground friction band ramps with this curriculum; every
    other DR term is fixed at its Level-0 value in EventCfg. The band widens from
    a narrow safe band toward the expected K1 rubber-sole ↔ artificial-turf band:

    0  foot friction static [0.95,1.05] dynamic [0.90,1.00]  (≈ nominal μ 1.0)
    1  foot friction static [0.85,1.08] dynamic [0.75,0.95]
    2  foot friction static [0.70,1.10] dynamic [0.60,0.92]
    3  foot friction static [0.60,1.10] dynamic [0.50,0.90]  (full target band)
    """

    LEVELS: list[dict] = [
        # Level 0 — narrow/safe band centered on the terrain nominal (μ ≈ 1.0)
        dict(foot_static=(0.95, 1.05), foot_dynamic=(0.90, 1.00)),
        # Level 1
        dict(foot_static=(0.85, 1.08), foot_dynamic=(0.75, 0.95)),
        # Level 2
        dict(foot_static=(0.70, 1.10), foot_dynamic=(0.60, 0.92)),
        # Level 3 — full target friction band (recommended turf range)
        dict(foot_static=(0.60, 1.10), foot_dynamic=(0.50, 0.90)),
    ]

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.threshold = p.get("fall_rate_threshold", 0.15)
        self.consecutive_required = p.get("consecutive_required", 100)
        self.consecutive_drop_required = p.get("consecutive_drop_required", 20)
        self.alpha = p.get("ema_alpha", 0.1)
        # ~1 iteration: common_step_counter increments by 1 per env.step(),
        # so one iteration == num_steps_per_env (=24) steps.
        self.check_interval = p.get("check_interval_steps", 24)

        self._level: int = 0
        self._consecutive: int = 0       # consecutive good (below-threshold) checks
        self._consecutive_bad: int = 0   # consecutive bad (at/above-threshold) checks
        self._ema_fall_rate: float = 0.0
        self._last_check_step: int = -1

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        fall_rate_threshold: float = 0.15,
        consecutive_required: int = 100,
        consecutive_drop_required: int = 20,
        ema_alpha: float = 0.1,
        check_interval_steps: int = 24,
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

        # --- 3. Update consecutive good/bad counters (mutually exclusive) ---
        if self._ema_fall_rate < self.threshold:
            self._consecutive += 1
            self._consecutive_bad = 0
        else:
            self._consecutive_bad += 1
            self._consecutive = 0

        # --- 4. Level up if sustained good, level down if sustained bad ---
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
            self._consecutive_bad >= self.consecutive_drop_required
            and self._level > 0
        ):
            self._level -= 1
            self._consecutive_bad = 0
            self._apply_level(env)
            print(
                f"[FallRateCurriculum] level ↓ {self._level}"
                f"  (ema_fall_rate={self._ema_fall_rate:.3f})"
            )

        return self._level

    @staticmethod
    def _get_term_cfg(env: "ManagerBasedRLEnv", name: str):
        """Return the event term cfg, or ``None`` if it is not registered.

        Domain randomization event terms are optional: EventCfg may be emptied
        to disable DR entirely while keeping this curriculum's level tracking.
        Each ``_apply_level`` mutation is guarded so a missing term is skipped
        instead of raising, and re-adding the term reconnects it automatically.
        """
        try:
            return env.event_manager.get_term_cfg(name)
        except (ValueError, KeyError):
            return None

    def _apply_level(self, env: "ManagerBasedRLEnv") -> None:
        p = self.LEVELS[self._level]

        # foot-ground friction band (re-sampled per reset from these ranges).
        # This is the only DR term the curriculum ramps; all other DR terms are
        # held fixed at their Level-0 values in EventCfg.
        foot_cfg = self._get_term_cfg(env, "foot_ground_friction")
        if foot_cfg is not None:
            foot_cfg.params["static_friction_range"] = p["foot_static"]
            foot_cfg.params["dynamic_friction_range"] = p["foot_dynamic"]


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
    the magnitude so a high contact return cannot blow up the penalties.

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
            rm.get_term_cfg(n).weight = w
        return self._ema
