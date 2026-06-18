# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Left-right symmetry augmentation for the Booster K1 soccer-kick task.

This implements the ``data_augmentation_func`` consumed by the (AMP) PPO
algorithm in the vendored ``rsl_rl`` fork. For every sample it appends a
single mirror-image copy (left <-> right) of the observation and action,
doubling the batch. Training the policy on both halves makes it symmetric, so
it can kick equally well with either foot.

The mirror is a reflection across the robot's sagittal plane (body x-z plane,
i.e. ``y -> -y``):

  * Linear vectors (lin vel, gravity, ball position/velocity) flip the lateral
    (y) component.
  * Angular velocity (a pseudovector) flips x and z.
  * 2D body-frame directions (target dir, goal dir, ...) flip the 2nd (y/sin)
    component.
  * Joint quantities (joint pos/vel, last action, and the action vector) swap
    the left and right joints; roll/yaw joints additionally negate while pitch
    joints keep their sign. The roll/yaw negation is verified by the robot's
    own symmetric default pose (``Left_Shoulder_Roll = -Right_Shoulder_Roll``,
    ``Left_Elbow_Yaw = -Right_Elbow_Yaw``).

The transform is built lazily from the live observation-term layout
(``env.observation_manager``) and the articulation joint names, so it stays
correct regardless of the runtime joint ordering or obs concatenation order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


# Per-term, per-dimension sign multipliers for the left-right mirror. Terms not
# listed here (scalars, masks, flags, distances) are left unchanged. Joint
# terms are handled separately via the joint permutation.
_VEC3_FLIP_Y = (1.0, -1.0, 1.0)   # linear vec3: flip lateral (y)
_ANGVEL_FLIP = (-1.0, 1.0, -1.0)  # angular vel pseudovector: flip x, z
_DIR2_FLIP_Y = (1.0, -1.0)        # 2D body-frame pos/dir: flip 2nd component

_TERM_SIGN_FLIP: dict[str, tuple[float, ...]] = {
    "base_lin_vel": _VEC3_FLIP_Y,
    "base_ang_vel": _ANGVEL_FLIP,
    "projected_gravity": _VEC3_FLIP_Y,
    "ball_pos_b": _DIR2_FLIP_Y,
    "ball_pos_b_gt": _VEC3_FLIP_Y,
    "ball_vel_b_gt": _VEC3_FLIP_Y,
    "goal_pos_b": _DIR2_FLIP_Y,
    "goal_dir_b": _DIR2_FLIP_Y,
    "target_dir_b": _DIR2_FLIP_Y,
    "pass_target_dir_b": _DIR2_FLIP_Y,
}

# Observation terms that hold per-joint values in articulation order.
_JOINT_TERMS = {"joint_pos", "joint_vel", "actions", "last_action"}

# Per-env transform cache, keyed by id(base_env).
_CACHE: dict[int, dict] = {}


def _build_joint_mirror(joint_names: list[str]) -> tuple[list[int], list[float]]:
    """Build the left-right joint permutation and sign-flip from joint names.

    Returns ``(perm, sign)`` such that the mirrored joint vector is
    ``out[i] = sign[i] * in[perm[i]]``.
    """
    name_to_idx = {nm: i for i, nm in enumerate(joint_names)}
    perm = list(range(len(joint_names)))
    sign = [1.0] * len(joint_names)
    for i, nm in enumerate(joint_names):
        # Mirror partner: swap Left <-> Right (handles ALeft/ARight prefixes).
        if "Left" in nm:
            partner = nm.replace("Left", "Right")
        elif "Right" in nm:
            partner = nm.replace("Right", "Left")
        else:
            partner = nm  # midline joints (head) map to themselves
        perm[i] = name_to_idx.get(partner, i)
        # Roll / yaw joints are anti-symmetric under a left-right mirror.
        low = nm.lower()
        if "roll" in low or "yaw" in low:
            sign[i] = -1.0
    return perm, sign


def _build_group_transform(
    base_env: "ManagerBasedRLEnv",
    obs_type: str,
    device: torch.device,
    jperm: list[int],
    jsign: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a flat gather-index + multiplier for one observation group.

    Returns ``(idx, mult)`` so that ``mirror = obs[:, idx] * mult``.
    """
    om = base_env.observation_manager
    term_names = om.active_terms[obs_type]
    term_dims = om.group_obs_term_dim[obs_type]
    num_joints = len(jperm)

    total = sum(int(shape[0]) for shape in term_dims)
    idx = torch.arange(total, device=device, dtype=torch.long)
    mult = torch.ones(total, device=device)
    jperm_t = torch.tensor(jperm, device=device, dtype=torch.long)
    jsign_t = torch.tensor(jsign, device=device)

    off = 0
    for name, shape in zip(term_names, term_dims):
        dim = int(shape[0])
        if name in _JOINT_TERMS:
            if dim != num_joints:
                raise ValueError(
                    f"Symmetry: joint term '{name}' has dim {dim}, expected {num_joints}."
                )
            idx[off : off + dim] = off + jperm_t
            mult[off : off + dim] = jsign_t
        elif name in _TERM_SIGN_FLIP:
            flip = _TERM_SIGN_FLIP[name]
            if dim != len(flip):
                raise ValueError(
                    f"Symmetry: term '{name}' has dim {dim}, expected {len(flip)}."
                )
            mult[off : off + dim] = torch.tensor(flip, device=device)
        # else: identity (scalars / masks / flags)
        off += dim
    return idx, mult


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Append a left-right mirror copy to the observations and/or actions.

    Args:
        env: The (wrapped) environment instance.
        obs: Observation tensor for ``obs_type`` group, shape (B, obs_dim).
        actions: Action tensor, shape (B, num_joints).
        obs_type: Which observation group ``obs`` belongs to ("policy"/"critic").

    Returns:
        ``(obs_aug, actions_aug)`` with batch size doubled (original first,
        mirror second). Either may be None if its input was None.
    """
    base_env = getattr(env, "unwrapped", env)
    cache = _CACHE.setdefault(id(base_env), {})

    if "joint" not in cache:
        robot = base_env.scene["robot"]
        cache["joint"] = _build_joint_mirror(list(robot.joint_names))
    jperm, jsign = cache["joint"]

    # observations
    if obs is not None:
        key = ("group", obs_type)
        if key not in cache:
            cache[key] = _build_group_transform(base_env, obs_type, obs.device, jperm, jsign)
        idx, mult = cache[key]
        obs_mirror = obs[:, idx] * mult
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    else:
        obs_aug = None

    # actions (joint-space)
    if actions is not None:
        akey = "action"
        if akey not in cache:
            cache[akey] = (
                torch.tensor(jperm, device=actions.device, dtype=torch.long),
                torch.tensor(jsign, device=actions.device),
            )
        ap, asg = cache[akey]
        if actions.shape[1] != ap.shape[0]:
            raise ValueError(
                f"Symmetry: action dim {actions.shape[1]} != num_joints {ap.shape[0]}."
            )
        act_mirror = actions[:, ap] * asg
        actions_aug = torch.cat([actions, act_mirror], dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
