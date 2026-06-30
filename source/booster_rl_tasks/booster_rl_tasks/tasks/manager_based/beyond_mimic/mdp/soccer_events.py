"""Event-manager helpers specific to the soccer kick task.

Most reset/spawn behavior lives inside :class:`SoccerKickCommand` so that ball
+ robot poses are sampled together. This module only adds the domain-
randomization events that act once at scene startup, a small push event, and a
curriculum-controllable per-reset material-friction randomizer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

# Re-exported here so env configs can address everything from this module.
from isaaclab.envs.mdp import (  # noqa: F401  (re-export for convenience)
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    apply_external_force_torque,
    push_by_setting_velocity,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class randomize_body_material_friction(ManagerTermBase):
    """Per-reset friction randomizer that re-reads its ranges every call.

    Unlike :func:`isaaclab.envs.mdp.randomize_rigid_body_material`, which samples
    a fixed set of material buckets **once** at ``__init__`` (so mutating its cfg
    ranges at runtime has no effect), this term samples fresh static/dynamic
    friction from the *current* cfg ranges on every reset. That lets a curriculum
    widen the ranges over training (e.g. ramp foot-ground friction from a narrow
    safe band toward the full expected band) and have the change take effect on
    the next episode reset.

    Restitution is left untouched (set ``restitution`` to override). Dynamic
    friction is clamped to ``<=`` static friction so the sampled pair is always
    physically consistent.

    Use ``mode="reset"`` so the friction is re-rolled each episode.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                "randomize_body_material_friction not supported for asset "
                f"'{self.asset_cfg.name}' of type '{type(self.asset)}'."
            )

        # Resolve the shape-index range for each targeted body so we only write
        # to the requested bodies' collision shapes. Mirrors the bucket-based
        # built-in's per-body shape parsing.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body: list[int] | None = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_view.max_shapes)
            num_shapes = sum(self.num_shapes_per_body)
            expected = self.asset.root_physx_view.max_shapes
            if num_shapes != expected:
                raise ValueError(
                    "randomize_body_material_friction failed to parse shapes per body. "
                    f"Expected {expected}, got {num_shapes}."
                )
            self.body_ids = list(self.asset_cfg.body_ids)
        else:
            self.num_shapes_per_body = None
            self.body_ids = None

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution: float | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        n = len(env_ids)
        if n == 0:
            return

        materials = self.asset.root_physx_view.get_material_properties()  # (E, S, 3) cpu
        total_shapes = self.asset.root_physx_view.max_shapes

        def _sample(num_shapes: int) -> tuple[torch.Tensor, torch.Tensor]:
            lo_s, hi_s = static_friction_range
            lo_d, hi_d = dynamic_friction_range
            static = lo_s + (hi_s - lo_s) * torch.rand(n, num_shapes)
            dynamic = lo_d + (hi_d - lo_d) * torch.rand(n, num_shapes)
            # physical constraint: dynamic <= static
            dynamic = torch.min(dynamic, static)
            return static, dynamic

        if self.num_shapes_per_body is not None:
            for body_id in self.body_ids:  # type: ignore[union-attr]
                start = sum(self.num_shapes_per_body[:body_id])
                end = start + self.num_shapes_per_body[body_id]
                static, dynamic = _sample(end - start)
                materials[env_ids, start:end, 0] = static
                materials[env_ids, start:end, 1] = dynamic
                if restitution is not None:
                    materials[env_ids, start:end, 2] = restitution
        else:
            static, dynamic = _sample(total_shapes)
            materials[env_ids, :, 0] = static
            materials[env_ids, :, 1] = dynamic
            if restitution is not None:
                materials[env_ids, :, 2] = restitution

        self.asset.root_physx_view.set_material_properties(materials, env_ids)
