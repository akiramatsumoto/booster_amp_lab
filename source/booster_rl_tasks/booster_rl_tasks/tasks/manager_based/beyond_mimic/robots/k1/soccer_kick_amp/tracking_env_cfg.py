"""Soccer kick AMP env config (V1 single-agent kicker).

Layout:
  * Scene = K1 robot + soccer ball + goal posts + flat terrain.
  * Actor obs: proprio + GT ball pos (in body-yaw frame) + ball-mask placeholder
                + target dir (cos/sin) + target strength + mode flag.
  * Critic obs: same + base lin vel + GT ball vel/goal pos + kick latches.
  * AMP obs: shared with locomotion (joint pos/vel + hand/foot positions).
  * Rewards: target_progress, ball_approach, kick_contact, kick_success,
             kick_angle_error, kick_strength_error, goal_scored_reward, posture
             & alignment penalties, AMP style (via AMP PPO).
  * Term: timeout, fall (z<0.3 or tilt>1.3), ball out of field, goal scored.
"""
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_rl_tasks.assets.objects import (
    SOCCER_BALL_CFG,
    soccer_goal_assets,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommandCfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerceptionCfg,
)


# =========================================================================
# Scene
# =========================================================================


@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
    """Scene for the soccer kick task: robot + ball + goal posts + terrain."""

    # ground terrain (flat)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robot — set in subclass post_init
    robot: ArticulationCfg = MISSING
    # ball — instantiated below
    ball: RigidObjectCfg = SOCCER_BALL_CFG  # type: ignore[assignment]
    # goal posts (set in __post_init__ via setattr loop)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


# =========================================================================
# Commands
# =========================================================================


@configclass
class CommandsCfg:
    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        # Virtual head-camera perception. Mirrors the mjlab K1 V1.49 setup
        # (RealSense D435i on Head_2, hold_last_on_miss=False, 10% blind
        # episodes, detection-prob floor 0.30, distance-dependent noise).
        perception=VirtualPerceptionCfg(),
        # V1.1 — lower kick-detection thresholds so early-policy taps still
        # register as kicks (the V1.0 baseline plateaued because peak
        # ball-XY-speed rarely exceeded 1.5 m/s on first-touch).
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        # Foot-link origin ↔ ball-center distance bottoms out at ~0.21 m at
        # contact (ball radius 0.11 m + foot-origin inset), so the default
        # 0.20 m proximity gate never fires even on a clean kick. Widen it so
        # ``kick_contact`` / ``stop_mode`` latch on real contacts.
        kick_foot_proximity=0.30,
        # Fixed ball spawn: 1 m in front, within a ±100° cone. Distance is no
        # longer driven by the curriculum (see CurriculumCfg).
        ball_spawn_distance_range=(1.0, 1.0),
        ball_spawn_angle_range=(-math.radians(100.0), math.radians(100.0)),
        # Kick strength 1-8 m/s, oversampling the weak end (exponent 2 → mean
        # ~3.3 m/s instead of the uniform 4.5).
        target_strength_range=(1.0, 8.0),
        target_strength_sample_exponent=2.0,
    )


# =========================================================================
# Actions
# =========================================================================


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=False
    )


# =========================================================================
# Observations
# =========================================================================


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # proprio (noisy)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # task-conditioning observations (no noise for V1 — virtual perception
        # will inject noise/dropout in V1.1 via separate functions).
        ball_pos_b = ObsTerm(
            func=mdp.soccer_observations.ball_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        ball_mask = ObsTerm(
            func=mdp.soccer_observations.ball_mask,
            params={"command_name": "soccer_kick"},
        )
        last_seen_dt = ObsTerm(
            func=mdp.soccer_observations.last_seen_dt,
            params={"command_name": "soccer_kick"},
        )
        target_dir_b = ObsTerm(
            func=mdp.soccer_observations.target_dir_b,
            params={"command_name": "soccer_kick"},
        )
        target_strength = ObsTerm(
            func=mdp.soccer_observations.target_strength_norm,
            params={"command_name": "soccer_kick"},
        )
        is_shoot = ObsTerm(
            func=mdp.soccer_observations.is_shoot_flag,
            params={"command_name": "soccer_kick"},
        )
        stop_flag = ObsTerm(
            func=mdp.soccer_observations.stop_flag,
            params={"command_name": "soccer_kick"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # proprio (clean) — duplicates of policy obs without noise
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        # ground-truth task-relevant state
        ball_pos_b_gt = ObsTerm(
            func=mdp.soccer_observations.ball_pos_b_gt,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        ball_vel_b_gt = ObsTerm(
            func=mdp.soccer_observations.ball_vel_b_gt,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        goal_pos_b = ObsTerm(
            func=mdp.soccer_observations.goal_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        goal_dir_b = ObsTerm(
            func=mdp.soccer_observations.goal_dir_b,
            params={"command_name": "soccer_kick"},
        )
        target_dir_b = ObsTerm(
            func=mdp.soccer_observations.target_dir_b,
            params={"command_name": "soccer_kick"},
        )
        target_strength = ObsTerm(
            func=mdp.soccer_observations.target_strength_norm,
            params={"command_name": "soccer_kick"},
        )
        is_shoot = ObsTerm(
            func=mdp.soccer_observations.is_shoot_flag,
            params={"command_name": "soccer_kick"},
        )
        stop_flag = ObsTerm(
            func=mdp.soccer_observations.stop_flag,
            params={"command_name": "soccer_kick"},
        )
        kick_flags = ObsTerm(
            func=mdp.soccer_observations.kick_state_flags,
            params={"command_name": "soccer_kick"},
        )
        pass_target_dir_b = ObsTerm(
            func=mdp.soccer_observations.pass_target_dir_b,
            params={"command_name": "soccer_kick"},
        )
        pass_target_dist = ObsTerm(
            func=mdp.soccer_observations.pass_target_dist,
            params={"command_name": "soccer_kick"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AMPObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        left_hand_pos = ObsTerm(func=mdp.get_lefthand_pos, clip=(-100.0, 100.0), scale=1.0)
        right_hand_pos = ObsTerm(func=mdp.get_righthand_pos, clip=(-100.0, 100.0), scale=1.0)
        left_foot_pos = ObsTerm(func=mdp.get_leftfoot_pos, clip=(-100.0, 100.0), scale=1.0)
        right_foot_pos = ObsTerm(func=mdp.get_rightfoot_pos, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    amp_observations: AMPObsCfg = AMPObsCfg()


# =========================================================================
# Events
# =========================================================================


@configclass
class EventCfg:
    # --- Fixed startup events (CPU-bucket based; not efficient for per-reset) ---
    physics_material_robot = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 1.05),
            "dynamic_friction_range": (0.75, 1.05),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    physics_material_ball = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.2, 0.5),
            "num_buckets": 32,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    # Robot + ball pose reset is handled inside SoccerKickCommand._resample_command.

    # --- Curriculum-controlled interval event ---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
    # --- Curriculum-controlled reset-mode events ---
    # All start at Level 0 (mild). FallRateDomainRandCurriculum widens
    # these as the fall rate drops.
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-0.075, 0.25),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.00375, 0.00375)},
        },
    )
    randomize_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Shank", ".*_Ankle_Cross", ".*_foot_link"],
            ),
            "mass_distribution_params": (-0.05, 0.1),
            "operation": "add",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )



# =========================================================================
# Rewards
# =========================================================================


@configclass
class RewardsCfg:
    # ---- Goal-related ----
    target_progress = RewTerm(
        func=mdp.soccer_rewards.target_progress,
        weight=8.0,
        params={"command_name": "soccer_kick"},
    )
    ball_approach = RewTerm(
        func=mdp.soccer_rewards.ball_approach,
        weight=4.0,
        params={"command_name": "soccer_kick"},
    )
    foot_ball_proximity = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=3.0,
        params={"command_name": "soccer_kick", "sigma": 0.35},
    )
    kick_contact = RewTerm(
        func=mdp.soccer_rewards.kick_contact_bonus,
        weight=4.0,
        params={"command_name": "soccer_kick"},
    )
    kick_success = RewTerm(
        func=mdp.soccer_rewards.kick_success,
        weight=10.0,
        params={"command_name": "soccer_kick"},
    )
    goal_scored = RewTerm(
        func=mdp.soccer_rewards.goal_scored_reward,
        weight=250.0,
        params={"command_name": "soccer_kick"},
    )
    # Experimental: reward kicking with the foot on the ball's spawn side.
    # Disabled by default (weight 0); enabled via train.py --near-foot-kick.
    near_foot_kick = RewTerm(
        func=mdp.soccer_rewards.near_foot_kick,
        weight=0.0,
        params={"command_name": "soccer_kick"},
    )
    kick_angle_error = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error,
        weight=-1.0,
        params={"command_name": "soccer_kick"},
    )
    kick_strength_error = RewTerm(
        func=mdp.soccer_rewards.kick_strength_error,
        weight=-0.5,
        params={"command_name": "soccer_kick"},
    )
    # ---- Auxiliary / posture ----
    pre_kick_yaw_align = RewTerm(
        func=mdp.soccer_rewards.pre_kick_body_yaw_alignment,
        weight=-0.5,
        params={"command_name": "soccer_kick"},
    )
    head_yaw_align = RewTerm(
        func=mdp.soccer_rewards.head_yaw_alignment_to_ball,
        weight=-0.3,
        params={"command_name": "soccer_kick"},
    )
    head_pitch_align = RewTerm(
        func=mdp.soccer_rewards.head_pitch_alignment_to_ball,
        weight=-0.3,
        params={"command_name": "soccer_kick"},
    )
    support_foot = RewTerm(
        func=mdp.soccer_rewards.support_foot_proximity,
        weight=0.5,
        params={"command_name": "soccer_kick"},
    )
    feet_proximity = RewTerm(
        func=mdp.soccer_rewards.feet_proximity_penalty,
        weight=-1.0,
    )
    pelvis_orientation = RewTerm(
        func=mdp.soccer_rewards.pelvis_orientation_penalty,
        weight=-5.0,
    )
    # ---- V5 post-kick stop mode ----
    # Rewards standing still once the env latches into stop mode after a
    # successful kick. Zero before the kick, so it never fights the approach
    # / kick shaping. Paired with the ``stop_flag`` observation so the same
    # input can command stand-still vs. kick at deploy time.
    # ``grace_steps`` delays the stop-mode stillness demand until the kick
    # follow-through / balance recovery has finished (stop_mode now latches on
    # the contact step, which is the most unstable moment). 25 steps ≈ 0.5 s.
    stand_still = RewTerm(
        func=mdp.soccer_rewards.stand_still,
        weight=4.0,
        params={"command_name": "soccer_kick", "grace_steps": 25},
    )
    # Whole-body return to default pose once stopped — base velocity alone is
    # not enough to keep a clean standing posture (negative weight).
    stop_joint_deviation = RewTerm(
        func=mdp.soccer_rewards.joint_deviation_in_stop,
        weight=-0.5,
        params={"command_name": "soccer_kick", "grace_steps": 25},
    )
    # Explicit post-kick survival bonus: rewards staying upright every step
    # after the kick (stop_mode). A fall forfeits this stream, so it directly
    # discourages "kick then fall". Active from the contact step (no grace) so
    # it also helps the policy survive the unstable follow-through.
    post_kick_alive = RewTerm(
        func=mdp.soccer_rewards.post_kick_alive,
        weight=5.0,
        params={"command_name": "soccer_kick"},
    )
    alive = RewTerm(func=mdp.soccer_rewards.alive_reward, weight=0.5)
    terminated = RewTerm(func=mdp.soccer_rewards.terminated_penalty, weight=-200.0)

    # ---- V3.3 search-for-ball shaping ----
    # When the head camera does not see the ball, reward the policy for
    # actively turning (base yaw + head yaw) and penalize the time-since-
    # last-detection. Together with the perception-gated ``ball_approach`` /
    # ``foot_ball_proximity`` terms above, this stops the "freeze when
    # ball-out-of-FOV" failure mode and forces an active-search behavior.
    search_yaw_velocity = RewTerm(
        func=mdp.soccer_rewards.search_yaw_velocity,
        weight=2.0,
        params={"command_name": "soccer_kick", "min_rate": 0.5, "max_rate": 3.0},
    )
    last_seen_dt_penalty = RewTerm(
        func=mdp.soccer_rewards.last_seen_dt_penalty,
        weight=-1.0,
        params={"command_name": "soccer_kick", "max_dt": 5.0},
    )
    head_yaw_search = RewTerm(
        func=mdp.soccer_rewards.head_yaw_search,
        weight=0.5,
        params={"command_name": "soccer_kick", "min_abs_rate": 0.5},
    )

    # ---- Standard regularizers (shared with locomotion baseline) ----
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_Shank", ".*Hip.*", ".*hand.*", ".*Arm.*", "Head_.*", "Trunk"],
            ),
            "threshold": 1.0,
        },
    )


# =========================================================================
# Terminations
# =========================================================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall_height = DoneTerm(
        func=mdp.soccer_terminations.fall_height,
        params={"min_height": 0.30},
    )
    fall_tilt = DoneTerm(
        func=mdp.soccer_terminations.fall_tilt,
        params={"max_tilt": 1.3},
    )
    ball_out = DoneTerm(
        func=mdp.soccer_terminations.ball_out_of_field,
        params={"command_name": "soccer_kick"},
    )
    goal_scored_done = DoneTerm(
        func=mdp.soccer_terminations.goal_scored,
        params={"command_name": "soccer_kick"},
    )


@configclass
class CurriculumCfg:
    # Ball distance is fixed at 1 m (see CommandsCfg.soccer_kick); the distance
    # curriculum is intentionally disabled.
    fall_rate_rand = CurrTerm(
        func=mdp.soccer_curriculums.FallRateDomainRandCurriculum,
        params={
            "fall_rate_threshold": 0.15,
            "consecutive_required": 5,
            "ema_alpha": 0.1,
            "check_interval_steps": 4096 * 24,
        },
    )
    # Ramp the kick angle/strength error penalties with the kick-contact return:
    # weight = base_weight * ema(Episode_Reward/kick_contact) * gain. Penalties
    # start near 0 and tighten only once the policy reliably makes contact.
    kick_error_weight = CurrTerm(
        func=mdp.soccer_curriculums.KickErrorWeightCurriculum,
        params={
            "gain": 1000.0,
            "ema_alpha": 0.1,
            "contact_term": "kick_contact",
            "scaled_terms": ["kick_angle_error", "kick_strength_error"],
            "max_abs_weight": None,
        },
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerKickEnvCfg(ManagerBasedRLEnvCfg):
    """Base env cfg for the soccer kick AMP task."""

    # 16m env_spacing > field length 14m so neighboring fields don't overlap.
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=4096, env_spacing=16.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Attach goal posts to the scene as additional AssetBaseCfg fields.
        for name, cfg in soccer_goal_assets().items():
            setattr(self.scene, name, cfg)

        # Simulation timing — 50Hz control, 200Hz physics.
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Bump PhysX patch budget: ball + goal collision pairs add to load.
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Viewer
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, -6.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
