"""Soccer-specific scene assets: ball, goal posts, field dimensions.

Field: 6 m (Y, width) x 9 m (X, length).
Goal mouth: 2.5 m wide x 1.2 m tall. Two upright posts + one crossbar.

The ball is a dynamic rigid sphere; the goal posts are kinematic (collidable but
do not move). Scoring is detected analytically by the soccer command term
(ball x-coord crosses the goal line within the post y-range).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg


# --- Field & goal geometry --------------------------------------------------
# Field is 9m (X, longitudinal) x 6m (Y, transverse).
FIELD_LENGTH_X = 9.0
FIELD_WIDTH_Y = 6.0
FIELD_HALF_LENGTH = FIELD_LENGTH_X / 2.0   # 4.5 m
FIELD_HALF_WIDTH = FIELD_WIDTH_Y / 2.0     # 3.0 m

# Goal mouth: 2.5m wide x 1.2m tall.
GOAL_WIDTH = 2.5
GOAL_HEIGHT = 1.2
GOAL_HALF_WIDTH = GOAL_WIDTH / 2.0  # 1.25 m

# Goal line is the inside face of the goal posts at x = +FIELD_HALF_LENGTH.
# The goal mouth straddles y = 0. Posts sit on the line.
GOAL_LINE_X = FIELD_HALF_LENGTH        # 4.5 m
GOAL_BACK_OFFSET = 0.6                 # how far behind the line the back goes
GOAL_POST_RADIUS = 0.05
GOAL_CROSSBAR_RADIUS = 0.04

# --- Ball -------------------------------------------------------------------
SOCCER_BALL_RADIUS = 0.11   # FIFA Size 5 ~22cm diameter
SOCCER_BALL_MASS = 0.43     # kg

SOCCER_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=SOCCER_BALL_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.05,
            angular_damping=0.10,
            max_linear_velocity=50.0,
            max_angular_velocity=200.0,
            max_depenetration_velocity=100.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=SOCCER_BALL_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.85,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.95, 0.95),
            roughness=0.5,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.5, 0.0, SOCCER_BALL_RADIUS),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)


def soccer_goal_assets(
    goal_x: float = GOAL_LINE_X,
    half_width: float = GOAL_HALF_WIDTH,
    height: float = GOAL_HEIGHT,
    post_radius: float = GOAL_POST_RADIUS,
    crossbar_radius: float = GOAL_CROSSBAR_RADIUS,
) -> dict[str, AssetBaseCfg]:
    """Return a dict of AssetBaseCfg for goal posts + crossbar (kinematic).

    The dict is meant to be unpacked onto a scene cfg via ``setattr``:
    ``for k, v in soccer_goal_assets().items(): setattr(scene_cfg, k, v)``.
    """
    common_rigid = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    )
    common_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    common_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8, dynamic_friction=0.6, restitution=0.2
    )
    visual_white = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.4)

    posts: dict[str, AssetBaseCfg] = {}

    posts["goal_post_left"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPostLeft",
        spawn=sim_utils.CylinderCfg(
            radius=post_radius,
            height=height,
            axis="Z",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_white,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, +half_width, height / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    posts["goal_post_right"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPostRight",
        spawn=sim_utils.CylinderCfg(
            radius=post_radius,
            height=height,
            axis="Z",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_white,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, -half_width, height / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    posts["goal_crossbar"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalCrossbar",
        spawn=sim_utils.CylinderCfg(
            radius=crossbar_radius,
            height=(2.0 * half_width),
            axis="Y",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_white,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    return posts


def soccer_own_goal_assets(
    goal_x: float = -GOAL_LINE_X,
    half_width: float = GOAL_HALF_WIDTH,
    height: float = GOAL_HEIGHT,
    post_radius: float = GOAL_POST_RADIUS,
    crossbar_radius: float = GOAL_CROSSBAR_RADIUS,
) -> dict[str, AssetBaseCfg]:
    """Return a dict of AssetBaseCfg for the *own*-goal posts + crossbar.

    Mirrors :func:`soccer_goal_assets` but spawns at ``goal_x`` (defaulting to
    the negative-x side of the field) and uses distinct prim paths so the
    own-goal can coexist with the shoot-goal in a future combined scene.
    """
    common_rigid = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    )
    common_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    common_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8, dynamic_friction=0.6, restitution=0.2
    )
    visual_red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.4)

    posts: dict[str, AssetBaseCfg] = {}

    posts["own_goal_post_left"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OwnGoalPostLeft",
        spawn=sim_utils.CylinderCfg(
            radius=post_radius,
            height=height,
            axis="Z",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_red,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, +half_width, height / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    posts["own_goal_post_right"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OwnGoalPostRight",
        spawn=sim_utils.CylinderCfg(
            radius=post_radius,
            height=height,
            axis="Z",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_red,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, -half_width, height / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    posts["own_goal_crossbar"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OwnGoalCrossbar",
        spawn=sim_utils.CylinderCfg(
            radius=crossbar_radius,
            height=(2.0 * half_width),
            axis="Y",
            rigid_props=common_rigid,
            collision_props=common_collision,
            physics_material=common_material,
            visual_material=visual_red,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(goal_x, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    return posts
