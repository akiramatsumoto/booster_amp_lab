"""Env-cfg for the stand-and-kick drill (fixed-arm K1).

Inherits :class:`soccer_kick_amp.env_cfg.FlatSoccerKickEnvCfg` (robot faces the
goal center, ``shoot_goal_aim_spread=0`` straight-shot command, etc.) and changes:

  * Ball spawn narrowed to a standing-reach cone -> no walking approach needed.
  * Robot swapped for the 14-DOF K1 whose 8 arm joints are welded in the URDF.
    The policy drives 12 legs + 2 head joints; the head stays actuated because
    the ball detector is a head-mounted camera and the ``head_*`` rewards steer
    it (see :mod:`soccer_perception`).
  * AMP observations narrowed to the legs (30-dim). The AMP corpus is the
    matching leg-only 30-col corpus — see ``ppo_cfg``. Two joints the policy
    *does* drive are deliberately absent from the discriminator: the head moves
    to search for the ball, a behavior the reference clips never contain, so
    showing the head to the discriminator would fight ``head_yaw_search``.
"""
from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_FIXED_ARMS_CFG as ROBOT_CFG,
    K1_FIXED_ARMS_ACTION_SCALE,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.env_cfg import (
    FlatSoccerKickEnvCfg,
)


# Leg joints in Isaac Lab BFS order — the column order of the leg-only AMP
# corpus (``scripts/make_leg_amp_corpus.py``). ``preserve_order=True`` pins the
# AMP observation to exactly this order, so it stays correct regardless of how
# the articulation happens to sort its joints.
LEG_JOINT_NAMES = [
    "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Left_Hip_Roll", "Right_Hip_Roll",
    "Left_Hip_Yaw", "Right_Hip_Yaw",
    "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch",
    "Left_Ankle_Roll", "Right_Ankle_Roll",
]


@configclass
class FlatStandKickEnvCfg(FlatSoccerKickEnvCfg):
    """0.3-0.4 m / ±20° ball cone — in reach of an in-place kick swing."""

    def __post_init__(self):
        super().__post_init__()

        # --- Ball spawn: within standing-kick reach, no approach step needed ---
        cmd = self.commands.soccer_kick
        cmd.ball_spawn_distance_range = (0.3, 0.4)
        cmd.ball_spawn_angle_range = (-math.radians(20.0), math.radians(20.0))

        # Spawn in the *opponent* half (goal line at env-local x = +4.5) instead
        # of the parent's own-half rectangle. Cap x at 3.0 (≥1.5 m from the goal
        # line) so the ball placed 0.3-0.4 m ahead stays clear of the goal mouth
        # and there is still a meaningful shooting distance for the standing kick.
        cmd.robot_spawn_x_range = (0.0, 3.0)

        # --- Robot: 14-DOF K1 (arms welded at the deploy upper-body pose) ---
        # Replaces the parent's 22-DOF K1. ``joint_names=[".*"]`` in the parent's
        # ActionsCfg now resolves to the 14 remaining DOF on its own.
        #
        # The scale dict has to be rebuilt rather than inherited: the parent's is
        # keyed by K1_ACTION_SCALE's regexes, whose arm patterns match no joint on
        # this articulation, and Isaac Lab raises on an unmatched pattern.
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_FIXED_ARMS_ACTION_SCALE)

        # Head scale 0.19 (the value the parent's per-joint loop intends but never
        # reaches: K1_ACTION_SCALE is keyed by the actuators' regexes — ".*Head.*"
        # — so the parent's concrete-name lookups never hit, leaving the head at
        # the actuator-derived 0.375). Isaac Lab resolves this dict with
        # ``re.fullmatch`` and raises "Multiple matches" if a joint matches two
        # keys, so the pattern must be overwritten, not supplemented with
        # "AAHead_yaw" / "Head_pitch" entries. Applied here only, so the parent
        # task's tuning is left exactly as it is.
        self.actions.joint_pos.scale[".*Head.*"] = 0.19

        # Crouched walking initial posture, matching the K1 locomotion
        # rough_env_cfg init_state. The arms need no entry — they are welded, and
        # a pattern matching no joint would raise in Isaac Lab's name resolution.
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*_Hip_Pitch": -0.26,
                ".*_Hip_Roll": 0.0,
                ".*_Hip_Yaw": 0.0,
                ".*_Knee_Pitch": 0.52,
                ".*_Ankle_Pitch": -0.26,
                ".*_Ankle_Roll": 0.0,
                ".*Head.*": 0.0,
            },
            joint_vel={".*": 0.0},
        )

        # --- AMP observations: legs only, 12 + 12 + 3 + 3 = 30 dims ---
        # The hand terms would be dead weight on a welded arm (constant in the
        # body frame) and would let the discriminator separate policy from
        # reference on that constant alone.
        amp = self.observations.amp_observations
        leg_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
        amp.joint_pos.params = {"asset_cfg": leg_cfg}
        amp.joint_vel.params = {"asset_cfg": leg_cfg}
        amp.left_hand_pos = None
        amp.right_hand_pos = None

        # --- undesired_contacts: drop the now-absent arm/hand bodies ---
        # Welding the arm joints makes the URDF importer lump the arm and hand
        # links into their parent (Trunk), so ``.*hand.*`` / ``.*Arm.*`` match no
        # body and Isaac Lab raises "Not all regular expressions are matched".
        # Trunk now carries the arm geometry, so keeping Trunk still penalizes a
        # fall onto that region. Everything else in the parent's list survives.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[".*_Shank", ".*Hip.*", "Head_.*", "Trunk"],
        )
