from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import  CurriculumManager
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def apply_curriculum_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    force_value = env.curriculum_manager.get_active_iterable_terms(env_ids)[0][1][0]

    weight = torch.clip(asset.data.root_pos_w[:, 2] / 0.57, 0, 1 )

    if force_value is None or force_value <= 0.0:
        return


    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=env.device)
    torques = torch.zeros_like(forces)

    # forces[:, asset_cfg.body_ids[0], 2] = force_value * weight
    # forces[:, asset_cfg.body_ids[1], 2] = force_value * (1- weight)
    forces[:, :, 2] = force_value 
    force_value *= (1- weight)
    # 应用
    asset.set_external_force_and_torque(forces = forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids,is_global=True)


def _uniform_range(range_values: tuple[float, float], size: tuple[int, ...], device: str) -> torch.Tensor:
    low = float(range_values[0])
    high = float(range_values[1])
    if high <= low:
        return torch.full(size, low, device=device)
    return torch.empty(size, device=device).uniform_(low, high)


def _force_magnitude_mixture_components(
    force_magnitude_mixture: Sequence[tuple[float, float, float]] | None,
) -> tuple[tuple[float, float, float], ...]:
    if not force_magnitude_mixture:
        return ()
    components: list[tuple[float, float, float]] = []
    for raw_component in force_magnitude_mixture:
        if len(raw_component) != 3:
            raise ValueError(
                "force_magnitude_mixture entries must be (min_n, max_n, weight), "
                f"got {raw_component!r}."
            )
        low, high, weight = (float(raw_component[0]), float(raw_component[1]), float(raw_component[2]))
        if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(weight)):
            raise ValueError(f"force_magnitude_mixture contains non-finite values: {raw_component!r}.")
        if weight <= 0.0 or high <= low:
            continue
        components.append((low, high, weight))
    return tuple(components)


def _force_magnitude_mixture_extent(
    components: Sequence[tuple[float, float, float]],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if not components:
        return fallback
    return (min(component[0] for component in components), max(component[1] for component in components))


def _sample_force_magnitude_mixture(
    components: Sequence[tuple[float, float, float]],
    num_samples: int,
    device: str,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty((0,), device=device)
    if not components:
        raise ValueError("force_magnitude_mixture must contain at least one positive-width, positive-weight range.")
    lows = torch.tensor([component[0] for component in components], device=device)
    highs = torch.tensor([component[1] for component in components], device=device)
    weights = torch.tensor([component[2] for component in components], device=device)
    indices = torch.multinomial(weights / weights.sum(), num_samples, replacement=True)
    selected_low = lows[indices]
    selected_high = highs[indices]
    return selected_low + torch.rand(num_samples, device=device) * (selected_high - selected_low)


def _force_duration_mixture_components(
    force_duration_mixture: Sequence[tuple[float, float, float, float, float, float, float]] | None,
) -> tuple[tuple[float, float, float, float, float, float, float], ...]:
    if not force_duration_mixture:
        return ()
    components: list[tuple[float, float, float, float, float, float, float]] = []
    for raw_component in force_duration_mixture:
        if len(raw_component) != 7:
            raise ValueError(
                "force_duration_mixture entries must be "
                "(min_n, max_n, min_duration_s, max_duration_s, min_ramp_s, max_ramp_s, weight), "
                f"got {raw_component!r}."
            )
        force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight = (
            float(raw_component[0]),
            float(raw_component[1]),
            float(raw_component[2]),
            float(raw_component[3]),
            float(raw_component[4]),
            float(raw_component[5]),
            float(raw_component[6]),
        )
        values = (force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"force_duration_mixture contains non-finite values: {raw_component!r}.")
        if weight <= 0.0 or force_high <= force_low or duration_high <= duration_low:
            continue
        components.append((force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight))
    return tuple(components)


def _force_duration_mixture_extents(
    components: Sequence[tuple[float, float, float, float, float, float, float]],
    force_fallback: tuple[float, float],
    duration_fallback: tuple[float, float],
    ramp_fallback: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not components:
        return force_fallback, duration_fallback, ramp_fallback
    return (
        (min(component[0] for component in components), max(component[1] for component in components)),
        (min(component[2] for component in components), max(component[3] for component in components)),
        (min(component[4] for component in components), max(component[5] for component in components)),
    )


def _sample_force_duration_mixture(
    components: Sequence[tuple[float, float, float, float, float, float, float]],
    num_samples: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        empty = torch.empty((0,), device=device)
        return empty, empty, empty
    if not components:
        raise ValueError("force_duration_mixture must contain at least one valid positive-weight component.")
    values = torch.tensor(components, device=device)
    weights = values[:, 6]
    indices = torch.multinomial(weights / weights.sum(), num_samples, replacement=True)
    selected = values[indices]
    force = selected[:, 0] + torch.rand(num_samples, device=device) * (selected[:, 1] - selected[:, 0])
    duration = selected[:, 2] + torch.rand(num_samples, device=device) * (selected[:, 3] - selected[:, 2])
    ramp = selected[:, 4] + torch.rand(num_samples, device=device) * (selected[:, 5] - selected[:, 4])
    return force.clamp_min(0.0), duration.clamp_min(0.0), ramp.clamp_min(0.0)


def _event_curriculum_alpha(env: ManagerBasedRLEnv, start_step: int = 0, duration_steps: int = 0) -> float:
    if duration_steps <= 0:
        return 1.0
    step = float(getattr(env, "common_step_counter", 0))
    return max(0.0, min(1.0, (step - float(start_step)) / max(float(duration_steps), 1.0)))


def _event_lerp_range(start: tuple[float, float], end: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * alpha,
        float(start[1]) + (float(end[1]) - float(start[1])) * alpha,
    )


def _event_adaptive_curriculum_alpha(
    env: ManagerBasedRLEnv,
    key: tuple,
    fallback_alpha: float,
    enabled: bool = False,
    initial_alpha: float = 0.0,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    step_up: float = 0.02,
    step_down: float = 0.08,
    promote_base_contact: float = 0.05,
    demote_base_contact: float = 0.12,
    promote_episode_length: float = 940.0,
    promote_push_failure: float | None = None,
    demote_push_failure: float | None = None,
    min_push_active: float = 0.0,
) -> float:
    if not enabled:
        return fallback_alpha
    states = getattr(env, "_booster_external_wrench_adaptive_curriculum", None)
    if states is None:
        states = {}
        setattr(env, "_booster_external_wrench_adaptive_curriculum", states)
    state = states.get(key)
    if state is None:
        state = {
            "alpha": max(float(min_alpha), min(float(max_alpha), float(initial_alpha))),
            "last_iteration": None,
        }
        states[key] = state
    stats = getattr(env, "_booster_track_adapter_training_stats", None)
    iteration = None if not isinstance(stats, dict) else stats.get("iteration")
    if stats is not None and iteration != state.get("last_iteration"):
        state["last_iteration"] = iteration
        base_contact = stats.get("base_contact")
        episode_length = stats.get("episode_length")
        push_failure = stats.get("push_failure")
        push_active = stats.get("push_active")
        if base_contact is not None:
            base_contact = float(base_contact)
            demote = base_contact >= float(demote_base_contact)
            if push_failure is not None and demote_push_failure is not None:
                demote = demote or float(push_failure) >= float(demote_push_failure)

            promote = base_contact <= float(promote_base_contact)
            if push_failure is not None and promote_push_failure is not None:
                promote = promote and float(push_failure) <= float(promote_push_failure)
            if push_active is not None and min_push_active > 0.0:
                promote = promote and float(push_active) >= float(min_push_active)

            if demote:
                state["alpha"] -= float(step_down)
            elif promote and episode_length is not None and float(episode_length) >= float(promote_episode_length):
                state["alpha"] += float(step_up)
            state["alpha"] = max(float(min_alpha), min(float(max_alpha), float(state["alpha"])))
    return float(state["alpha"])


def _as_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | None, device: str) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def _external_wrench_body_count(asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list):
        return len(body_ids)
    if isinstance(body_ids, torch.Tensor):
        return int(body_ids.numel())
    if isinstance(body_ids, int):
        return 1
    return asset.num_bodies


def _external_wrench_body_quat_w(
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    body_quat_w = asset.data.body_quat_w[env_ids]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, int):
        return body_quat_w[:, [body_ids]]
    if isinstance(body_ids, list):
        return body_quat_w[:, body_ids]
    if isinstance(body_ids, torch.Tensor):
        return body_quat_w[:, body_ids.to(device=body_quat_w.device, dtype=torch.long)]
    return body_quat_w


def _world_vectors_to_link_frame(
    vectors_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> torch.Tensor:
    flat_vectors = vectors_w.reshape(-1, 3)
    flat_quats = body_quat_w.reshape(-1, 4)
    return math_utils.quat_apply_inverse(flat_quats, flat_vectors).reshape_as(vectors_w)


def _external_wrench_pulse_state(
    env: ManagerBasedRLEnv,
    key: tuple,
    num_envs: int,
    num_bodies: int,
    device: str,
) -> dict[str, torch.Tensor]:
    states = getattr(env, "_booster_external_wrench_pulse_states", None)
    if states is None:
        states = {}
        setattr(env, "_booster_external_wrench_pulse_states", states)
    state = states.get(key)
    shape = (num_envs, num_bodies, 3)
    if state is None or state["forces"].shape != shape:
        state = {
            "remaining_s": torch.zeros(num_envs, device=device),
            "duration_s": torch.zeros(num_envs, device=device),
            "ramp_up_s": torch.zeros(num_envs, device=device),
            "next_s": torch.zeros(num_envs, device=device),
            "pulse_count": torch.zeros(num_envs, dtype=torch.long, device=device),
            "forces": torch.zeros(shape, device=device),
            "torques": torch.zeros(shape, device=device),
            "positions": torch.zeros(shape, device=device),
        }
        states[key] = state
    else:
        if "duration_s" not in state:
            state["duration_s"] = torch.zeros(num_envs, device=device)
        if "ramp_up_s" not in state:
            state["ramp_up_s"] = torch.zeros(num_envs, device=device)
    return state


def _command_term_for_external_push(env: ManagerBasedRLEnv, command_name: str | None):
    if not command_name:
        return None
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return None
    try:
        return command_manager.get_term(command_name)
    except Exception:
        return None


def _set_command_metric(command_term, name: str, value: torch.Tensor | float, env_ids: torch.Tensor):
    metrics = getattr(command_term, "metrics", None)
    if not isinstance(metrics, dict):
        return
    num_envs = command_term.num_envs if hasattr(command_term, "num_envs") else None
    device = command_term.device if hasattr(command_term, "device") else env_ids.device
    if num_envs is None:
        existing = metrics.get(name)
        num_envs = existing.shape[0] if isinstance(existing, torch.Tensor) and existing.ndim > 0 else int(env_ids.max()) + 1
    metric = metrics.get(name)
    if not isinstance(metric, torch.Tensor) or metric.shape[0] != int(num_envs):
        metric = torch.zeros(int(num_envs), device=device)
        metrics[name] = metric
    if isinstance(value, torch.Tensor):
        value = value.to(device=metric.device, dtype=metric.dtype)
        if value.ndim == 0:
            metric[env_ids] = value
        else:
            metric[env_ids] = value.reshape(-1)
    else:
        metric[env_ids] = float(value)


def apply_external_wrench_pulse(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    force_magnitude_range: tuple[float, float],
    duration_range_s: tuple[float, float],
    pulse_interval_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
    position_range: dict[str, tuple[float, float]] | None = None,
    torque_z_range: tuple[float, float] = (0.0, 0.0),
    force_z_range: tuple[float, float] = (0.0, 0.0),
    command_name: str | None = "base_velocity",
    equivalent_velocity_abs: float = 1.0,
    equivalent_yaw_abs: float = 0.30,
    failure_mining: bool = True,
    is_global: bool = False,
    force_magnitude_range_final: tuple[float, float] | None = None,
    duration_range_s_final: tuple[float, float] | None = None,
    pulse_interval_range_s_final: tuple[float, float] | None = None,
    torque_z_range_final: tuple[float, float] | None = None,
    activation_probability: float = 1.0,
    activation_probability_final: float | None = None,
    equivalent_velocity_abs_final: float | None = None,
    equivalent_yaw_abs_final: float | None = None,
    curriculum_start_step: int = 0,
    curriculum_duration_steps: int = 0,
    adaptive_curriculum: bool = False,
    adaptive_alpha_initial: float = 0.0,
    adaptive_alpha_min: float = 0.0,
    adaptive_alpha_max: float = 1.0,
    adaptive_alpha_step_up: float = 0.02,
    adaptive_alpha_step_down: float = 0.08,
    adaptive_promote_base_contact: float = 0.05,
    adaptive_demote_base_contact: float = 0.12,
    adaptive_promote_episode_length: float = 940.0,
    adaptive_promote_push_failure: float | None = None,
    adaptive_demote_push_failure: float | None = None,
    adaptive_min_push_active: float = 0.0,
    max_pulses_per_episode: int = 0,
    force_magnitude_mixture: Sequence[tuple[float, float, float]] | None = None,
    force_duration_mixture: Sequence[tuple[float, float, float, float, float, float, float]] | None = None,
    force_duration_mixture_scale_with_curriculum: bool = False,
    force_duration_mixture_force_floor: float = 0.0,
    fixed_direction_xy: tuple[float, float] | None = None,
    fixed_direction_probability: float = 0.0,
    fixed_direction_jitter_rad: float = 0.0,
    ramp_up_range_s: tuple[float, float] = (0.0, 0.0),
    start_command_norm_max: float = 0.0,
    start_command_yaw_scale: float = 0.30,
):
    """Apply short finite-duration external force pulses.

    This is intended to model a physical kick/shove more faithfully than
    root-velocity teleport pushes. The event should be scheduled every policy
    step, while this function owns its per-environment pulse timers. When
    ``is_global`` is true, force directions are sampled in world frame and then
    converted to each link frame before they are passed to IsaacLab. This keeps
    the disturbance world-directional without repeatedly toggling the
    articulation's external-wrench frame.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env_ids_tensor = _as_env_ids(env, env_ids, asset.device)
    if env_ids_tensor.numel() == 0:
        return

    num_bodies = _external_wrench_body_count(asset, asset_cfg)
    key = (asset_cfg.name, str(asset_cfg.body_ids), command_name)
    state = _external_wrench_pulse_state(env, key, env.scene.num_envs, num_bodies, asset.device)
    step_dt = float(getattr(env, "step_dt", 0.02))
    linear_alpha = _event_curriculum_alpha(env, curriculum_start_step, curriculum_duration_steps)
    alpha = _event_adaptive_curriculum_alpha(
        env,
        key,
        linear_alpha,
        enabled=adaptive_curriculum,
        initial_alpha=adaptive_alpha_initial,
        min_alpha=adaptive_alpha_min,
        max_alpha=adaptive_alpha_max,
        step_up=adaptive_alpha_step_up,
        step_down=adaptive_alpha_step_down,
        promote_base_contact=adaptive_promote_base_contact,
        demote_base_contact=adaptive_demote_base_contact,
        promote_episode_length=adaptive_promote_episode_length,
        promote_push_failure=adaptive_promote_push_failure,
        demote_push_failure=adaptive_demote_push_failure,
        min_push_active=adaptive_min_push_active,
    )
    active_force_magnitude_range = (
        _event_lerp_range(force_magnitude_range, force_magnitude_range_final, alpha)
        if force_magnitude_range_final is not None
        else force_magnitude_range
    )
    active_force_magnitude_mixture = _force_magnitude_mixture_components(force_magnitude_mixture)
    active_force_magnitude_range = _force_magnitude_mixture_extent(
        active_force_magnitude_mixture,
        active_force_magnitude_range,
    )
    active_duration_range_s = (
        _event_lerp_range(duration_range_s, duration_range_s_final, alpha)
        if duration_range_s_final is not None
        else duration_range_s
    )
    active_ramp_up_range_s = ramp_up_range_s
    active_force_duration_mixture = _force_duration_mixture_components(force_duration_mixture)
    active_force_magnitude_range, active_duration_range_s, active_ramp_up_range_s = _force_duration_mixture_extents(
        active_force_duration_mixture,
        active_force_magnitude_range,
        active_duration_range_s,
        active_ramp_up_range_s,
    )
    force_duration_mixture_force_scale = 1.0
    if active_force_duration_mixture and force_duration_mixture_scale_with_curriculum:
        force_floor = max(0.0, min(1.0, float(force_duration_mixture_force_floor)))
        force_duration_mixture_force_scale = force_floor + (1.0 - force_floor) * max(0.0, min(1.0, float(alpha)))
        active_force_magnitude_range = (
            float(active_force_magnitude_range[0]) * force_duration_mixture_force_scale,
            float(active_force_magnitude_range[1]) * force_duration_mixture_force_scale,
        )
    active_pulse_interval_range_s = (
        _event_lerp_range(pulse_interval_range_s, pulse_interval_range_s_final, alpha)
        if pulse_interval_range_s_final is not None
        else pulse_interval_range_s
    )
    active_torque_z_range = (
        _event_lerp_range(torque_z_range, torque_z_range_final, alpha)
        if torque_z_range_final is not None
        else torque_z_range
    )
    active_activation_probability = (
        float(activation_probability)
        if activation_probability_final is None
        else float(activation_probability)
        + (float(activation_probability_final) - float(activation_probability)) * alpha
    )
    active_activation_probability = max(0.0, min(1.0, active_activation_probability))
    active_equivalent_velocity_abs = (
        float(equivalent_velocity_abs)
        if equivalent_velocity_abs_final is None
        else float(equivalent_velocity_abs)
        + (float(equivalent_velocity_abs_final) - float(equivalent_velocity_abs)) * alpha
    )
    active_equivalent_yaw_abs = (
        float(equivalent_yaw_abs)
        if equivalent_yaw_abs_final is None
        else float(equivalent_yaw_abs) + (float(equivalent_yaw_abs_final) - float(equivalent_yaw_abs)) * alpha
    )

    # Clear pulses on freshly reset environments so an episode never inherits a kick.
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor):
        reset_ids = env_ids_tensor[episode_length_buf[env_ids_tensor.to(device=episode_length_buf.device)] <= 1]
        if reset_ids.numel() > 0:
            state["remaining_s"][reset_ids] = 0.0
            state["duration_s"][reset_ids] = 0.0
            state["ramp_up_s"][reset_ids] = 0.0
            state["forces"][reset_ids] = 0.0
            state["torques"][reset_ids] = 0.0
            state["positions"][reset_ids] = 0.0
            state["pulse_count"][reset_ids] = 0
            state["next_s"][reset_ids] = _uniform_range(
                active_pulse_interval_range_s, (reset_ids.numel(),), asset.device
            )

    state["remaining_s"][env_ids_tensor] = torch.clamp(state["remaining_s"][env_ids_tensor] - step_dt, min=0.0)
    state["next_s"][env_ids_tensor] = torch.clamp(state["next_s"][env_ids_tensor] - step_dt, min=0.0)

    inactive = state["remaining_s"][env_ids_tensor] <= 0.0
    ready = state["next_s"][env_ids_tensor] <= 0.0
    candidate_ids = env_ids_tensor[inactive & ready]
    started_ids = env_ids_tensor.new_empty((0,))
    max_pulses = int(max_pulses_per_episode)
    if candidate_ids.numel() > 0 and max_pulses > 0:
        under_limit = state["pulse_count"][candidate_ids] < max_pulses
        candidate_ids = candidate_ids[under_limit]
    command_norm_max = float(start_command_norm_max)
    if candidate_ids.numel() > 0 and command_norm_max > 0.0 and command_name:
        try:
            command = env.command_manager.get_command(command_name).to(asset.device)
            command_sample = command[candidate_ids]
            yaw_scale = max(float(start_command_yaw_scale), 0.0)
            command_norm = torch.norm(
                torch.stack(
                    (
                        command_sample[:, 0],
                        command_sample[:, 1],
                        yaw_scale * command_sample[:, 2],
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            candidate_ids = candidate_ids[command_norm <= command_norm_max]
        except Exception:
            candidate_ids = candidate_ids.new_empty((0,))
    if candidate_ids.numel() > 0 and active_activation_probability < 1.0:
        selected = torch.rand(candidate_ids.numel(), device=asset.device) <= active_activation_probability
        rejected_ids = candidate_ids[~selected]
        if rejected_ids.numel() > 0:
            state["next_s"][rejected_ids] = _uniform_range(
                active_pulse_interval_range_s, (rejected_ids.numel(),), asset.device
            ).clamp_min(step_dt)
        start_ids = candidate_ids[selected]
    else:
        start_ids = candidate_ids
    force_high = float(active_force_magnitude_range[1])
    duration_high = float(active_duration_range_s[1])
    if start_ids.numel() > 0 and force_high > 0.0 and duration_high > 0.0:
        started_ids = start_ids
        if active_force_duration_mixture:
            force_mag, duration_s, ramp_up_s = _sample_force_duration_mixture(
                active_force_duration_mixture,
                start_ids.numel(),
                asset.device,
            )
            if force_duration_mixture_force_scale != 1.0:
                force_mag = force_mag * force_duration_mixture_force_scale
            duration_s = duration_s.clamp_min(step_dt)
            ramp_up_s = torch.minimum(ramp_up_s, duration_s)
        elif active_force_magnitude_mixture:
            force_mag = _sample_force_magnitude_mixture(
                active_force_magnitude_mixture,
                start_ids.numel(),
                asset.device,
            ).clamp_min(0.0)
            duration_s = _uniform_range(active_duration_range_s, (start_ids.numel(),), asset.device).clamp_min(step_dt)
            ramp_up_s = _uniform_range(active_ramp_up_range_s, (start_ids.numel(),), asset.device).clamp_min(0.0)
            ramp_up_s = torch.minimum(ramp_up_s, duration_s)
        else:
            force_mag = _uniform_range(active_force_magnitude_range, (start_ids.numel(),), asset.device).clamp_min(0.0)
            duration_s = _uniform_range(active_duration_range_s, (start_ids.numel(),), asset.device).clamp_min(step_dt)
            ramp_up_s = _uniform_range(active_ramp_up_range_s, (start_ids.numel(),), asset.device).clamp_min(0.0)
            ramp_up_s = torch.minimum(ramp_up_s, duration_s)
        next_s = duration_s + _uniform_range(
            active_pulse_interval_range_s, (start_ids.numel(),), asset.device
        ).clamp_min(step_dt)

        angles = torch.empty(start_ids.numel(), device=asset.device).uniform_(-math.pi, math.pi)
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        fixed_probability = max(0.0, min(1.0, float(fixed_direction_probability)))
        if fixed_direction_xy is not None and fixed_probability > 0.0:
            fixed_xy = torch.tensor(
                [float(fixed_direction_xy[0]), float(fixed_direction_xy[1])],
                device=asset.device,
                dtype=direction.dtype,
            )
            fixed_norm = torch.linalg.norm(fixed_xy).clamp_min(1.0e-6)
            fixed_xy = fixed_xy / fixed_norm
            fixed_mask = torch.rand(start_ids.numel(), device=asset.device) <= fixed_probability
            if torch.any(fixed_mask):
                fixed_angle = torch.atan2(fixed_xy[1], fixed_xy[0])
                jitter_abs = max(float(fixed_direction_jitter_rad), 0.0)
                if jitter_abs > 0.0:
                    fixed_angle = fixed_angle + torch.empty(
                        fixed_mask.sum(), device=asset.device
                    ).uniform_(-jitter_abs, jitter_abs)
                    direction[fixed_mask, 0] = torch.cos(fixed_angle)
                    direction[fixed_mask, 1] = torch.sin(fixed_angle)
                else:
                    direction[fixed_mask] = fixed_xy
        equiv_abs = max(float(active_equivalent_velocity_abs), 1.0e-6)
        max_impulse = max(force_high * duration_high, 1.0e-6)
        equivalent_mag = equiv_abs * torch.clamp(force_mag * duration_s / max_impulse, 0.0, 1.0)
        equivalent_push = torch.zeros((start_ids.numel(), 6), device=asset.device)
        equivalent_push[:, 0:2] = direction * equivalent_mag.unsqueeze(-1)

        sampled_mask = torch.zeros(start_ids.numel(), dtype=torch.bool, device=asset.device)
        command_term = _command_term_for_external_push(env, command_name)
        equivalent_velocity_range = {
            "x": (-equiv_abs, equiv_abs),
            "y": (-equiv_abs, equiv_abs),
            "yaw": (-abs(float(active_equivalent_yaw_abs)), abs(float(active_equivalent_yaw_abs))),
        }
        if failure_mining and command_term is not None and hasattr(command_term, "sample_failure_aware_pushes"):
            equivalent_push, sampled_mask = command_term.sample_failure_aware_pushes(
                start_ids, equivalent_velocity_range, equivalent_push
            )
            mined_norm = torch.linalg.norm(equivalent_push[:, :2], dim=-1)
            mined = mined_norm > 1.0e-6
            if torch.any(mined):
                direction[mined] = equivalent_push[mined, :2] / mined_norm[mined].unsqueeze(-1)

        forces = torch.zeros((start_ids.numel(), num_bodies, 3), device=asset.device)
        forces[:, :, 0] = force_mag.unsqueeze(-1) * direction[:, 0:1]
        forces[:, :, 1] = force_mag.unsqueeze(-1) * direction[:, 1:2]
        forces[:, :, 2] = _uniform_range(force_z_range, (start_ids.numel(), num_bodies), asset.device)

        torques = torch.zeros_like(forces)
        torques[:, :, 2] = _uniform_range(active_torque_z_range, (start_ids.numel(), num_bodies), asset.device)

        if is_global:
            body_quat_w = _external_wrench_body_quat_w(asset, start_ids, asset_cfg)
            forces = _world_vectors_to_link_frame(forces, body_quat_w)
            torques = _world_vectors_to_link_frame(torques, body_quat_w)

        position_range = position_range or {}
        positions = torch.zeros_like(forces)
        for axis_idx, axis in enumerate(("x", "y", "z")):
            positions[:, :, axis_idx] = _uniform_range(
                position_range.get(axis, (0.0, 0.0)),
                (start_ids.numel(), num_bodies),
                asset.device,
            )

        state["remaining_s"][start_ids] = duration_s
        state["duration_s"][start_ids] = duration_s
        state["ramp_up_s"][start_ids] = ramp_up_s
        state["next_s"][start_ids] = next_s
        state["pulse_count"][start_ids] += 1
        state["forces"][start_ids] = forces
        state["torques"][start_ids] = torques
        state["positions"][start_ids] = positions

        if command_term is not None and hasattr(command_term, "record_external_push"):
            command_term.record_external_push(start_ids, equivalent_push, sampled_mask, equivalent_velocity_range)

    active = state["remaining_s"][env_ids_tensor] > 0.0
    forces = state["forces"][env_ids_tensor].clone()
    torques = state["torques"][env_ids_tensor].clone()
    positions = state["positions"][env_ids_tensor].clone()
    if torch.any(active):
        active_duration = state["duration_s"][env_ids_tensor].clamp_min(step_dt)
        active_ramp_up = state["ramp_up_s"][env_ids_tensor]
        elapsed_s = torch.clamp(active_duration - state["remaining_s"][env_ids_tensor], min=0.0)
        ramp_scale = torch.ones_like(active_duration)
        ramp_mask = active & (active_ramp_up > 1.0e-6)
        ramp_scale[ramp_mask] = torch.clamp(elapsed_s[ramp_mask] / active_ramp_up[ramp_mask], 0.0, 1.0)
        forces = forces * ramp_scale.view(-1, 1, 1)
        torques = torques * ramp_scale.view(-1, 1, 1)
    if torch.any(~active):
        forces[~active] = 0.0
        torques[~active] = 0.0
        positions[~active] = 0.0
        state["forces"][env_ids_tensor[~active]] = 0.0
        state["torques"][env_ids_tensor[~active]] = 0.0
        state["positions"][env_ids_tensor[~active]] = 0.0

    command_term = _command_term_for_external_push(env, command_name)
    if command_term is not None:
        started = torch.zeros(env_ids_tensor.numel(), device=asset.device)
        if started_ids.numel() > 0:
            local_started = torch.isin(env_ids_tensor, started_ids)
            started[local_started] = 1.0
        applied_force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=1)
        _set_command_metric(command_term, "force_push_active", active.float(), env_ids_tensor)
        _set_command_metric(command_term, "force_push_started", started, env_ids_tensor)
        _set_command_metric(command_term, "force_push_active_count", float(active.sum().item()), env_ids_tensor)
        _set_command_metric(command_term, "force_push_started_count", float(started.sum().item()), env_ids_tensor)
        _set_command_metric(command_term, "force_push_applied_force_n", applied_force_norm, env_ids_tensor)
        _set_command_metric(
            command_term,
            "force_push_applied_force_max_n",
            float(applied_force_norm.max().item()) if applied_force_norm.numel() > 0 else 0.0,
            env_ids_tensor,
        )
        _set_command_metric(command_term, "force_push_curriculum_alpha", float(alpha), env_ids_tensor)
        _set_command_metric(command_term, "force_push_force_min_n", float(active_force_magnitude_range[0]), env_ids_tensor)
        _set_command_metric(command_term, "force_push_force_max_n", float(active_force_magnitude_range[1]), env_ids_tensor)
        _set_command_metric(command_term, "force_push_duration_min_s", float(active_duration_range_s[0]), env_ids_tensor)
        _set_command_metric(command_term, "force_push_duration_max_s", float(active_duration_range_s[1]), env_ids_tensor)
        _set_command_metric(
            command_term,
            "force_push_activation_probability",
            float(active_activation_probability),
            env_ids_tensor,
        )

    asset.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        positions=positions,
        env_ids=env_ids_tensor,
        body_ids=asset_cfg.body_ids,
        is_global=False,
    )

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies, 10)
    coms = asset.root_physx_view.get_coms().clone()

    # cache the nominal CoM on the first call so that repeated resets randomize
    # around the fixed nominal value instead of accumulating drift (this event
    # runs in reset mode, driven by the domain-rand curriculum).
    if not hasattr(asset, "_nominal_coms"):
        asset._nominal_coms = coms.clone()

    # randomize the com in range, only for the reset envs (env_ids), starting
    # from the cached nominal value to keep the randomization absolute
    coms[env_ids[:, None], body_ids, :3] = (
        asset._nominal_coms[env_ids[:, None], body_ids, :3] + rand_samples
    )

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
