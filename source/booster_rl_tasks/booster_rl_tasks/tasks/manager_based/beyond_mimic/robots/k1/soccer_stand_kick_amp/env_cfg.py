"""Env-cfg for the stand-and-kick drill.

Inherits :class:`soccer_kick_amp.env_cfg.FlatSoccerKickEnvCfg` unchanged (robot
faces the goal center, ``shoot_goal_aim_spread=0`` straight-shot command, etc.)
and only narrows the ball spawn to a standing-reach cone so the kick can be made
in place without a walking approach.
"""
from __future__ import annotations

import math

from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.env_cfg import (
    FlatSoccerKickEnvCfg,
)


@configclass
class FlatStandKickEnvCfg(FlatSoccerKickEnvCfg):
    """0.3-0.4 m / ±20° ball cone — in reach of an in-place kick swing."""

    def __post_init__(self):
        super().__post_init__()

        # Only change vs the parent: bring the ball into standing-kick reach so
        # no approach step is needed (matches the kick-only AMP style coverage).
        cmd = self.commands.soccer_kick
        cmd.ball_spawn_distance_range = (0.3, 0.4)
        cmd.ball_spawn_angle_range = (-math.radians(20.0), math.radians(20.0))
