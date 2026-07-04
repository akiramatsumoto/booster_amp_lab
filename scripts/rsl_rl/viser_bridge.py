"""Viser web viewer bridge for IsaacLab eval/play scripts.

Mirrors the IsaacLab robot state of one environment into a shadow MuJoCo model
and renders it through ``mjviser`` for a browser-based 3D view. Adds velocity
arrows (commanded vs actual, linear vs yaw) and an optional joystick that can
override the ``base_velocity`` command term while the eval loop runs.

This is a slim, single-file rewrite of the holosoma-side bridge tailored to
booster_amp_lab's eval/play scripts. There is no terrain handling, no
checkpoint hot-swap, and no reward streaming — just the parts that are useful
for visually inspecting tracking behaviour in real time.

Usage::

    from viser_bridge import BoosterViserBridge

    bridge = BoosterViserBridge(env)  # env is the unwrapped ManagerBasedRLEnv
    cmd_term = env.command_manager.get_term("base_velocity")
    while running:
        if bridge.joystick_enabled:
            cmd_term.vel_command_b[:, 0] = bridge.joystick_command[0]
            cmd_term.vel_command_b[:, 1] = bridge.joystick_command[1]
            cmd_term.vel_command_b[:, 2] = bridge.joystick_command[2]
        ...
        bridge.update()
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from xml.etree import ElementTree as ET

import numpy as np
import torch

logger = logging.getLogger("booster_viser_bridge")

_DEFAULT_K1_MJCF = "robots/K1/K1_22dof.xml"
_ARROW_SHAFT_RATIO = 0.8
_ARROW_HEAD_RATIO = 0.2
_ARROW_WIDTH = 0.015
_Z = np.array([0.0, 0.0, 1.0])


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) rotating ``src`` onto ``dst``."""
    c = float(np.dot(src, dst))
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(src, dst)
    w = 1.0 + c
    q = np.array([w, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def _colored(mesh, rgba):
    mesh.visual.face_colors = [rgba] * len(mesh.faces)
    return mesh


def _quat_apply(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q_wxyz``."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication ``q1 * q2`` in wxyz convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class _Sphere3D:
    """Persistent sphere node in the viser scene. Position is updated per frame."""

    def __init__(self, server, name: str, radius: float, rgba: tuple[int, int, int, int]):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        self._node = server.scene.add_mesh_trimesh(name, _colored(mesh, rgba))
        self._visible = True

    def update(self, position: np.ndarray, offset: np.ndarray = None) -> None:
        if not self._visible:
            return
        self._node.position = position + (offset if offset is not None else np.zeros(3))

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._node.visible = visible


class _Cylinder3D:
    """Persistent cylinder node. Used for goal posts and the crossbar."""

    def __init__(
        self,
        server,
        name: str,
        radius: float,
        height: float,
        axis: str,
        rgba: tuple[int, int, int, int],
    ):
        import trimesh

        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
        # trimesh cylinder is z-aligned by default. Compose with axis target.
        if axis == "Z":
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        elif axis == "Y":
            quat = _rotation_between(_Z, np.array([0.0, 1.0, 0.0]))
        else:  # X
            quat = _rotation_between(_Z, np.array([1.0, 0.0, 0.0]))
        self._node = server.scene.add_mesh_trimesh(name, _colored(mesh, rgba))
        self._node.wxyz = quat
        self._visible = True

    def update(self, position: np.ndarray, offset: np.ndarray = None) -> None:
        if not self._visible:
            return
        self._node.position = position + (offset if offset is not None else np.zeros(3))

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._node.visible = visible


class _FOVFrustum:
    """Wireframe + translucent pyramid showing the camera FOV cone.

    Drawn from the optical centre on Head_2 out to ``max_range``. Wireframe
    is 8 thin cylinders (4 apex→corner edges + 4 far-quad edges); the inside
    is filled by a single low-opacity pyramid mesh.
    """

    def __init__(
        self,
        server,
        name: str,
        fov_h: float,  # half-angle in radians
        fov_v: float,
        max_range: float,
        rgba_wire: tuple[int, int, int, int] = (255, 230, 50, 230),
        rgba_fill: tuple[int, int, int, int] = (255, 230, 50, 50),
    ):
        import trimesh

        self._fov_h = fov_h
        self._fov_v = fov_v
        self._max_range = max_range

        # Far-plane corner offsets in camera frame (camera looks along +x).
        r = max_range
        h = r * np.tan(fov_h)
        v = r * np.tan(fov_v)
        self._corners_cam = np.array([
            [r, +h, +v],
            [r, -h, +v],
            [r, -h, -v],
            [r, +h, -v],
        ])

        # Wireframe only — the user asked us to drop the translucent fill
        # (looking at the camera apex from inside it filled the whole view).
        # Eight cylinders: 4 apex→corner + 4 far-quad edges.
        self._edges = []
        for i in range(8):
            shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=8)
            shaft.apply_translation([0, 0, 0.5])
            self._edges.append(
                server.scene.add_mesh_trimesh(f"{name}/edge_{i}", _colored(shaft, rgba_wire))
            )
        self._fill = None
        self._visible = True
        self._edge_width = 0.012

    def _set_edge(self, idx: int, start: np.ndarray, end: np.ndarray) -> None:
        d = end - start
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            self._edges[idx].visible = False
            return
        self._edges[idx].visible = self._visible
        direction = d / length
        q = _rotation_between(_Z, direction)
        w = self._edge_width
        self._edges[idx].position = start
        self._edges[idx].wxyz = q
        self._edges[idx].scale = (w, w, length)

    def update(
        self,
        cam_pos_world: np.ndarray,
        cam_quat_wxyz: np.ndarray,
        scene_offset: np.ndarray,
    ) -> None:
        if not self._visible:
            return
        apex = cam_pos_world + scene_offset
        # Compute corners in world frame.
        corners_world = np.stack(
            [cam_pos_world + _quat_apply(cam_quat_wxyz, c) for c in self._corners_cam]
        )
        corners_view = corners_world + scene_offset

        # Apex → corners (edges 0..3).
        for i in range(4):
            self._set_edge(i, apex, corners_view[i])
        # Far-plane quad (edges 4..7).
        for i in range(4):
            self._set_edge(4 + i, corners_view[i], corners_view[(i + 1) % 4])

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        for e in self._edges:
            e.visible = visible


class _Arrow3D:
    """Cylinder shaft + cone head arrow in the viser scene."""

    def __init__(self, server, name: str, rgba: tuple[int, int, int, int]):
        import trimesh

        shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
        shaft.apply_translation([0, 0, 0.5])
        head = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
        self._shaft = server.scene.add_mesh_trimesh(f"{name}/shaft", _colored(shaft, rgba))
        self._head = server.scene.add_mesh_trimesh(f"{name}/head", _colored(head, rgba))
        self._visible = True

    def update(self, start: np.ndarray, end: np.ndarray, offset: np.ndarray) -> None:
        s, e = start + offset, end + offset
        d = e - s
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            self._shaft.visible = False
            self._head.visible = False
            return
        self._shaft.visible = self._visible
        self._head.visible = self._visible
        if not self._visible:
            return
        direction = d / length
        q = _rotation_between(_Z, direction)
        w = _ARROW_WIDTH
        self._shaft.position = s
        self._shaft.wxyz = q
        self._shaft.scale = (w, w, _ARROW_SHAFT_RATIO * length)
        self._head.position = s + direction * _ARROW_SHAFT_RATIO * length
        self._head.wxyz = q
        self._head.scale = (w, w, _ARROW_HEAD_RATIO * length)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._shaft.visible = visible
        self._head.visible = visible


def _resolve_default_mjcf() -> str:
    """Resolve the K1 MJCF path inside the installed booster_assets package."""
    try:
        from booster_assets import BOOSTER_ASSETS_DIR
    except ImportError as e:
        raise RuntimeError(
            "Could not import booster_assets to resolve default MJCF path. "
            "Install booster_assets or pass mjcf_path= explicitly."
        ) from e
    return os.path.join(BOOSTER_ASSETS_DIR, _DEFAULT_K1_MJCF)


_DUPLICATE_NAME_ATTRS = (
    "name", "body", "geom", "joint", "site", "tendon", "actuator",
    "target", "joint1", "joint2",
)


def _rename_in_subtree(elem, prefix: str) -> None:
    """Prefix every name-like attribute in ``elem`` and its descendants."""
    for attr in _DUPLICATE_NAME_ATTRS:
        v = elem.attrib.get(attr)
        if v is not None:
            elem.set(attr, f"{prefix}{v}")
    for child in list(elem):
        _rename_in_subtree(child, prefix)


def _load_shadow_model(mjcf_path: str, *, with_receiver: bool = False):
    """Load the MJCF and (optionally) add a second K1 instance for the receiver.

    Returns ``(model, data, has_receiver)``.
    """
    import copy
    import mujoco as mj

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    # Invisible floor for mj_forward to succeed; rendering uses the viser plane.
    ET.SubElement(worldbody, "geom", {
        "name": "_viser_floor", "type": "plane", "size": "10 10 0.01",
        "rgba": "0 0 0 0", "contype": "1", "conaffinity": "1",
    })

    has_receiver = False
    if with_receiver:
        # Clone the root body (Trunk) subtree under a renamed prefix so we have
        # two independent K1 articulations in the same model. Actuators also
        # get prefixed; their joint references need to be updated in tandem.
        trunk = None
        for body in worldbody.findall("body"):
            if body.attrib.get("name") == "Trunk":
                trunk = body
                break
        if trunk is not None:
            clone = copy.deepcopy(trunk)
            _rename_in_subtree(clone, "rcv_")
            # Offset the cloned trunk's spawn position so it doesn't overlap.
            existing = clone.attrib.get("pos", "0 0 0").split()
            try:
                x = float(existing[0]) + 3.0
                y = float(existing[1])
                z = float(existing[2])
                clone.set("pos", f"{x} {y} {z}")
            except (IndexError, ValueError):
                clone.set("pos", "3.0 0.0 0.57")
            worldbody.append(clone)

            actuator = root.find("actuator")
            if actuator is not None:
                cloned_act = copy.deepcopy(actuator)
                _rename_in_subtree(cloned_act, "rcv_")
                actuator.extend(list(cloned_act))
            has_receiver = True

    mjcf_dir = os.path.dirname(mjcf_path)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", dir=mjcf_dir, delete=False) as tmp:
        tree.write(tmp, xml_declaration=True, encoding="utf-8")
        tmp_path = tmp.name
    try:
        model = mj.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model, mj.MjData(model), has_receiver


class BoosterViserBridge:
    """Viser bridge that mirrors one IsaacLab env into a shadow MuJoCo scene."""

    def __init__(
        self,
        env,
        *,
        env_id: int = 0,
        mjcf_path: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        update_freq: int = 1,
        fps_limit: int = 60,
    ) -> None:
        # late imports so this module is cheap to import without viser installed
        import mujoco as mj
        import viser as _viser
        from mjviser import ViserMujocoScene

        for noisy in ("websockets", "websockets.server", "trimesh", "trimesh.util"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._env = env
        self._env_id = env_id
        self._update_freq = max(1, update_freq)
        self._min_interval = 1.0 / max(fps_limit, 1)
        self._step_count = 0
        self._total_steps = 0
        self._last_update_time = 0.0

        self._mjcf_path = mjcf_path or _resolve_default_mjcf()
        # Detect whether the env has a receiver articulation to render.
        env_has_receiver = False
        try:
            env_has_receiver = "receiver" in env.scene.keys()
        except Exception:
            env_has_receiver = False
        self._mj_model, self._mj_data, self._has_receiver = _load_shadow_model(
            self._mjcf_path, with_receiver=env_has_receiver,
        )
        self._mj = mj
        self._build_joint_addressing()

        self._server = _viser.ViserServer(host=host, port=port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        self._arrow_cmd_lin = _Arrow3D(self._server, "/arrows/cmd_lin", (50, 70, 230, 220))
        self._arrow_cmd_ang = _Arrow3D(self._server, "/arrows/cmd_ang", (50, 150, 50, 220))
        self._arrow_actual_lin = _Arrow3D(self._server, "/arrows/actual_lin", (0, 200, 255, 200))
        self._arrow_actual_ang = _Arrow3D(self._server, "/arrows/actual_ang", (0, 230, 100, 200))
        self._show_arrows = True
        self._arrow_scale = 0.5
        self._arrow_z = 0.2

        # Soccer scene overlays (ball, goal posts, receiver) — created lazily
        # based on what's actually present in the env scene.
        self._ball_node: _Sphere3D | None = None
        self._pass_target_node: _Sphere3D | None = None
        self._goal_nodes: dict[str, _Cylinder3D] = {}
        self._field_node = None
        self._goal_init_pos_local: dict[str, np.ndarray] = {}
        self._fov_kicker: _FOVFrustum | None = None
        self._fov_receiver: _FOVFrustum | None = None
        # FOV parameters read from VirtualPerceptionCfg on the command term.
        self._fov_cfg: dict | None = None
        self._build_soccer_overlay()
        self._build_fov_overlay()

        self._joystick_enabled = False
        self._joystick_vx = 0.0
        self._joystick_vy = 0.0
        self._joystick_yaw = 0.0
        self._speed_multiplier = 1.0
        self._reset_requested = False
        self._soccer_mode_override = "Auto"
        self._soccer_pass_target_x = 3.0
        self._soccer_pass_target_y = 0.0
        self._soccer_strength_norm = 0.5
        self._show_pass_target_marker = True

        self._build_gui()

        logger.info("BoosterViserBridge: http://%s:%d", host, port)
        print(f"[viser] http://{host}:{port}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_soccer_overlay(self) -> None:
        """Create viser primitives for ball, goal posts, and receiver if present.

        These are looked up by name in ``env.scene``. Anything not present is
        silently skipped so the bridge still works for non-soccer tasks.
        """
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return

        # Ball — white sphere
        if "ball" in scene.keys():
            try:
                ball = scene["ball"]
                radius = float(getattr(ball.cfg.spawn, "radius", 0.11))
            except Exception:
                radius = 0.11
            self._ball_node = _Sphere3D(
                self._server, "/soccer/ball", radius=radius, rgba=(240, 240, 240, 255)
            )

        try:
            self._env.command_manager.get_term("soccer_kick")
            self._pass_target_node = _Sphere3D(
                self._server, "/soccer/pass_target", radius=0.16, rgba=(255, 80, 220, 255)
            )
            self._pass_target_node.set_visible(False)
        except Exception:
            self._pass_target_node = None

        # Goal posts + crossbar — orange cylinders. Cache their env-frame init
        # positions once; ``AssetBase`` has no .data view, so we can't read
        # ``root_pos_w`` per frame. Goal geometry never moves.
        #
        # We read from ``env.cfg.scene`` (always available) instead of
        # ``scene[name]`` because AssetBase prims are not always indexable via
        # ``InteractiveScene.__getitem__`` in older Isaac Lab versions.
        env_scene_cfg = getattr(getattr(self._env, "cfg", None), "scene", None)
        post_color = (255, 130, 40, 255)
        for name, axis, radius_attr, height_attr in (
            ("goal_post_left", "Z", 0.05, 1.2),
            ("goal_post_right", "Z", 0.05, 1.2),
            ("goal_crossbar", "Y", 0.04, 2.5),
        ):
            asset_cfg = getattr(env_scene_cfg, name, None)
            if asset_cfg is None:
                continue
            try:
                spawn = asset_cfg.spawn
                radius = float(getattr(spawn, "radius", radius_attr))
                height = float(getattr(spawn, "height", height_attr))
                init_pos = np.array(asset_cfg.init_state.pos, dtype=np.float64)
            except Exception as exc:
                logger.warning(
                    "BoosterViserBridge: could not read cfg for %s (%s); using defaults",
                    name, exc,
                )
                radius = radius_attr
                height = height_attr
                init_pos = np.array(
                    [7.0, 1.25 if "left" in name else (-1.25 if "right" in name else 0.0),
                     0.6 if "post" in name else 1.2],
                    dtype=np.float64,
                )
            self._goal_nodes[name] = _Cylinder3D(
                self._server, f"/soccer/{name}", radius=radius,
                height=height, axis=axis, rgba=post_color,
            )
            self._goal_init_pos_local[name] = init_pos

        # Receiver — drawn by the shadow-model second K1 instance. The
        # ``_has_receiver`` flag was set in ``_load_shadow_model``; pose +
        # joints are synced inside ``_sync_shadow``. No separate viser node is
        # needed here.

        # Field — semi-transparent green plane at z = 0, sized 14 × 9 m, so the
        # user has a ground reference (the shadow MJCF floor is invisible).
        if self._ball_node is not None:
            import trimesh
            field = trimesh.creation.box(extents=[14.0, 9.0, 0.005])
            self._field_node = self._server.scene.add_mesh_trimesh(
                "/soccer/field", _colored(field, (60, 140, 60, 180))
            )

    def _build_fov_overlay(self) -> None:
        """Create the FOV frustum if the soccer task has a perception config.

        Reads FOV / max-range from ``cmd.cfg.perception`` so the visual matches
        the actual virtual-camera the policy is using.
        """
        cmd = None
        try:
            cmd = self._env.command_manager.get_term("soccer_kick")
        except Exception:
            return
        perception_cfg = getattr(cmd.cfg, "perception", None)
        if perception_cfg is None:
            return
        # FOV half-angles (cfg stores full angles in degrees).
        import math
        fov_h = math.radians(float(getattr(perception_cfg, "fov_h_deg", 105.12))) * 0.5
        fov_v = math.radians(float(getattr(perception_cfg, "fov_v_deg", 94.17))) * 0.5
        max_range = float(getattr(perception_cfg, "max_detection_range", 7.0))
        offset_pos = np.array(getattr(perception_cfg, "camera_offset_pos",
                                       (0.00124, 0.04553, -0.01582)), dtype=np.float64)
        offset_quat = np.array(getattr(perception_cfg, "camera_offset_quat",
                                        (0.0054, 0.9986, -0.0028, 0.0511)), dtype=np.float64)
        camera_body = str(getattr(perception_cfg, "camera_body_name", "Head_2"))

        self._fov_cfg = {
            "fov_h": fov_h, "fov_v": fov_v, "max_range": max_range,
            "offset_pos": offset_pos, "offset_quat": offset_quat,
            "camera_body": camera_body,
        }
        # Resolve the head body index for kicker (and receiver if present).
        try:
            robot_bodies = list(self._env.scene["robot"].body_names)
            self._kicker_head_idx = robot_bodies.index(camera_body)
        except Exception:
            self._kicker_head_idx = -1
        self._receiver_head_idx = -1
        if self._has_receiver:
            try:
                recv_bodies = list(self._env.scene["receiver"].body_names)
                self._receiver_head_idx = recv_bodies.index(camera_body)
            except Exception:
                pass

        # Yellow for kicker FOV, light cyan for receiver (when it gets one).
        self._fov_kicker = _FOVFrustum(
            self._server, "/perception/fov_kicker",
            fov_h=fov_h, fov_v=fov_v, max_range=max_range,
            rgba_wire=(255, 230, 50, 230), rgba_fill=(255, 230, 50, 50),
        )
        # We don't actually drive a receiver FOV (V3.2 uses GT), so leave it
        # disabled by default. Keeping the slot here for V3.3.

    def _read_overlay_state(self):
        """Return world-frame poses for the soccer-scene overlays."""
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return {}

        envs_origin = scene.env_origins[self._env_id].detach().cpu().numpy()
        result = {"env_origin": envs_origin}

        def _safe_pos(name: str):
            try:
                return scene[name].data.root_pos_w[self._env_id].detach().cpu().numpy()
            except Exception:
                return None

        if self._ball_node is not None:
            result["ball"] = _safe_pos("ball")
        # Goal posts live at fixed env-relative offsets — recompute their world
        # pose each frame in case the camera-tracking offset changes.
        for name, local in self._goal_init_pos_local.items():
            result[name] = envs_origin + local
        if self._field_node is not None:
            # Field center sits at the env origin, z = 0.
            field_center = envs_origin.copy()
            field_center[2] = 0.0
            result["field"] = field_center
        if self._pass_target_node is not None:
            try:
                cmd = self._env.command_manager.get_term("soccer_kick")
                target_xy = cmd.pass_target_pos_w[self._env_id].detach().cpu().numpy()
                pass_target = np.array([target_xy[0], target_xy[1], envs_origin[2] + 0.16], dtype=np.float64)
                result["pass_target"] = pass_target
            except Exception:
                pass
        return result

    def _build_joint_addressing(self) -> None:
        """Map IsaacLab joint order → MuJoCo qpos addresses for hinge joints.

        Builds two mappings when a receiver K1 is included in the shadow model:
        one for the kicker (joint names as in the MJCF) and one for the
        receiver (joint names prefixed with ``rcv_``).
        """
        mj = self._mj
        model = self._mj_model

        # Collect every free joint and hinge address indexed by name.
        free_qpos_addrs: list[tuple[str, int]] = []  # body name → qpos addr
        hinge_qpos_addr: dict[str, int] = {}
        for jid in range(model.njnt):
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                # Resolve the parent body name to know which K1 this belongs to.
                body_id = int(model.jnt_bodyid[jid])
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
                free_qpos_addrs.append((body_name, int(model.jnt_qposadr[jid])))
            else:
                hinge_qpos_addr[jname] = int(model.jnt_qposadr[jid])

        # Pick the kicker (Trunk) and (if present) receiver (rcv_Trunk) free joints.
        self._free_qpos_addr: int | None = None
        self._receiver_free_qpos_addr: int | None = None
        for body_name, addr in free_qpos_addrs:
            if body_name == "Trunk":
                self._free_qpos_addr = addr
            elif body_name == "rcv_Trunk":
                self._receiver_free_qpos_addr = addr

        # Map sim joint names → kicker and receiver qpos addresses.
        self._sim_joint_names: list[str] = list(self._env.scene["robot"].data.joint_names)
        self._sim_to_qpos: list[int] = []
        self._sim_to_receiver_qpos: list[int] = []
        missing: list[str] = []
        for sim_name in self._sim_joint_names:
            addr = hinge_qpos_addr.get(sim_name, -1)
            if addr < 0:
                missing.append(sim_name)
            self._sim_to_qpos.append(addr)
            if self._has_receiver:
                self._sim_to_receiver_qpos.append(
                    hinge_qpos_addr.get(f"rcv_{sim_name}", -1)
                )

        if missing:
            logger.warning(
                "BoosterViserBridge: %d joints missing in MJCF (will be left at default): %s",
                len(missing), missing[:5],
            )

    def _build_gui(self) -> None:
        import viser as _viser

        server = self._server
        tabs = server.gui.add_tab_group()

        with tabs.add_tab("Controls"):
            with server.gui.add_folder("Info", expand_by_default=True):
                self._info_md = server.gui.add_markdown(self._info_text())

            with server.gui.add_folder("Simulation", expand_by_default=True):
                btn_pause = server.gui.add_button("Pause", icon=_viser.Icon.PLAYER_PAUSE)
                btn_start = server.gui.add_button("Start", icon=_viser.Icon.PLAYER_PLAY, visible=False)
                btn_reset = server.gui.add_button("Reset", icon=_viser.Icon.RELOAD)

                @btn_pause.on_click
                def _(_evt):
                    self._scene.paused = True
                    btn_pause.visible = False
                    btn_start.visible = True
                    self._info_md.content = self._info_text()

                @btn_start.on_click
                def _(_evt):
                    self._scene.paused = False
                    btn_start.visible = False
                    btn_pause.visible = True
                    self._info_md.content = self._info_text()

                @btn_reset.on_click
                def _(_evt):
                    self._reset_requested = True

                speed = server.gui.add_button_group("Speed", ("Slower", "1x", "Faster"))

                @speed.on_click
                def _(_evt):
                    if speed.value == "Slower":
                        self._speed_multiplier = max(0.125, self._speed_multiplier / 2.0)
                    elif speed.value == "Faster":
                        self._speed_multiplier = min(8.0, self._speed_multiplier * 2.0)
                    else:
                        self._speed_multiplier = 1.0

            with server.gui.add_folder("Velocity arrows", expand_by_default=True):
                cb_show = server.gui.add_checkbox("Show", initial_value=True)
                sl_scale = server.gui.add_slider("Scale", min=0.1, max=3.0, step=0.1, initial_value=0.5)
                sl_z = server.gui.add_slider("Height", min=0.0, max=1.0, step=0.05, initial_value=0.2)

                @cb_show.on_update
                def _(_evt):
                    self._show_arrows = cb_show.value
                    for a in (self._arrow_cmd_lin, self._arrow_cmd_ang,
                              self._arrow_actual_lin, self._arrow_actual_ang):
                        a.set_visible(cb_show.value)

                @sl_scale.on_update
                def _(_evt):
                    self._arrow_scale = sl_scale.value

                @sl_z.on_update
                def _(_evt):
                    self._arrow_z = sl_z.value

                server.gui.add_markdown(
                    "Blue lin / Green yaw — commanded\n\nCyan lin / Light-green yaw — actual"
                )

            with server.gui.add_folder("Commands (joystick)", expand_by_default=True):
                cb_joy = server.gui.add_checkbox("Override env command", initial_value=False)
                sl_vx = server.gui.add_slider("lin_vel_x", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                sl_vy = server.gui.add_slider("lin_vel_y", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                sl_yaw = server.gui.add_slider("ang_vel_z", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                btn_zero = server.gui.add_button("Zero", icon=_viser.Icon.SQUARE_X)

                @cb_joy.on_update
                def _(_evt):
                    self._joystick_enabled = cb_joy.value

                @sl_vx.on_update
                def _(_evt):
                    self._joystick_vx = float(sl_vx.value)

                @sl_vy.on_update
                def _(_evt):
                    self._joystick_vy = float(sl_vy.value)

                @sl_yaw.on_update
                def _(_evt):
                    self._joystick_yaw = float(sl_yaw.value)

                @btn_zero.on_click
                def _(_evt):
                    sl_vx.value = sl_vy.value = sl_yaw.value = 0.0
                    self._joystick_vx = self._joystick_vy = self._joystick_yaw = 0.0

            try:
                soccer_cmd = self._env.command_manager.get_term("soccer_kick")
            except Exception:
                soccer_cmd = None
            if soccer_cmd is not None:
                cfg = getattr(soccer_cmd, "cfg", None)
                pass_x_min, pass_x_max = getattr(cfg, "pass_target_x_range", (-2.0, 6.0))
                pass_y_min, pass_y_max = getattr(cfg, "pass_target_y_range", (-3.5, 3.5))
                self._soccer_pass_target_x = float((pass_x_min + pass_x_max) * 0.5)
                self._soccer_pass_target_y = 0.0
                with server.gui.add_folder("Soccer Kick", expand_by_default=True):
                    mode = server.gui.add_button_group("Mode", ("Auto", "Shoot", "Pass"))
                    strength = server.gui.add_slider(
                        "Strength norm", min=0.0, max=1.0, step=0.05, initial_value=self._soccer_strength_norm
                    )
                    pass_x = server.gui.add_slider(
                        "Pass target x", min=float(pass_x_min), max=float(pass_x_max),
                        step=0.1, initial_value=self._soccer_pass_target_x
                    )
                    pass_y = server.gui.add_slider(
                        "Pass target y", min=float(pass_y_min), max=float(pass_y_max),
                        step=0.1, initial_value=self._soccer_pass_target_y
                    )
                    show_target = server.gui.add_checkbox("Show pass target", initial_value=True)

                    @mode.on_click
                    def _(_evt):
                        self._soccer_mode_override = str(mode.value)

                    @strength.on_update
                    def _(_evt):
                        self._soccer_strength_norm = float(strength.value)

                    @pass_x.on_update
                    def _(_evt):
                        self._soccer_pass_target_x = float(pass_x.value)

                    @pass_y.on_update
                    def _(_evt):
                        self._soccer_pass_target_y = float(pass_y.value)

                    @show_target.on_update
                    def _(_evt):
                        self._show_pass_target_marker = bool(show_target.value)
                        if self._pass_target_node is not None:
                            self._pass_target_node.set_visible(self._show_pass_target_marker)

        with tabs.add_tab("Scene"):
            self._scene.create_scene_gui(
                camera_distance=3.0, camera_azimuth=150.0, camera_elevation=25.0,
            )
        with tabs.add_tab("Visualization"):
            self._scene.create_overlay_gui()
        with tabs.add_tab("Groups"):
            self._scene.create_groups_gui()

    def _info_text(self) -> str:
        status = "Paused" if self._scene.paused else "Running"
        lines = [
            f"**Status:** {status}",
            f"**Step:** {self._total_steps}",
            f"**Speed:** {self._speed_multiplier}x",
            f"**Env id:** {self._env_id}",
            f"**Joints mapped:** {sum(a >= 0 for a in self._sim_to_qpos)}/{len(self._sim_to_qpos)}",
        ]
        # Soccer-task mode + ball detection status, if the soccer_kick command
        # term exists. Helps tell SHOOT vs PASS apart during play.
        try:
            cmd = self._env.command_manager.get_term("soccer_kick")
            is_shoot = bool(cmd.is_shoot[self._env_id].item())
            mode = "🥅 SHOOT" if is_shoot else "🤝 PASS"
            strength = float(cmd.target_strength[self._env_id].item())
            lines.append("")
            lines.append(f"**Mode:** {mode}  (target {strength:.1f} m/s)")
            if self._soccer_mode_override != "Auto":
                lines.append(f"**Control override:** {self._soccer_mode_override}")
            if not is_shoot:
                target_xy = cmd.pass_target_pos_w[self._env_id].detach().cpu().numpy()
                origin_xy = self._env.scene.env_origins[self._env_id, :2].detach().cpu().numpy()
                local_xy = target_xy - origin_xy
                lines.append(f"**Pass target:** x={local_xy[0]:+.1f}, y={local_xy[1]:+.1f}")
            mask_t = getattr(cmd, "ball_mask_perceived", None)
            if mask_t is not None:
                seen = bool(mask_t[self._env_id].item() > 0.5)
                perception = getattr(cmd, "perception", None)
                if perception is not None:
                    in_fov = bool(perception.in_fov[self._env_id].item() > 0.5)
                    p_det = float(perception.detect_prob[self._env_id].item())
                    raw_seen = bool(perception.raw_detected[self._env_id].item() > 0.5)
                    if seen:
                        det_text = "✅"
                    elif in_fov:
                        det_text = f"❌ dropped in FOV (p={p_det:.2f}, raw={int(raw_seen)})"
                    else:
                        ball_cam = perception.ball_in_cam[self._env_id].detach().cpu().numpy()
                        det_text = (
                            f"❌ out of FOV "
                            f"(cam x/y/z={ball_cam[0]:+.2f}/{ball_cam[1]:+.2f}/{ball_cam[2]:+.2f})"
                        )
                    lines.append(f"**Ball detected:** {det_text}")
                else:
                    lines.append(f"**Ball detected:** {'✅' if seen else '❌'}")
            last_dt = getattr(cmd, "last_seen_dt", None)
            if last_dt is not None:
                lines.append(f"**Last-seen Δt:** {float(last_dt[self._env_id].item()):.2f} s")
            kc = bool(cmd.kick_contact_awarded[self._env_id].item())
            ks = bool(cmd.kick_success_awarded[self._env_id].item())
            ga = bool(cmd.goal_awarded[self._env_id].item())
            ts = bool(getattr(cmd, "trap_success_awarded", torch.zeros(1))[self._env_id].item()) \
                if hasattr(cmd, "trap_success_awarded") else False
            lines.append(
                f"**Latches:** contact={'✓' if kc else '·'}  "
                f"success={'✓' if ks else '·'}  "
                f"goal={'✓' if ga else '·'}  trap={'✓' if ts else '·'}"
            )
        except Exception:
            pass
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def _read_robot_state(self):
        robot = self._env.scene["robot"]
        rs = robot.data.root_state_w[self._env_id].detach().cpu().numpy()
        # IsaacLab order: pos[3], quat_wxyz[4], lin_vel_w[3], ang_vel_w[3]
        pos = rs[0:3]
        quat_wxyz = rs[3:7]
        lin_vel_w = rs[7:10]
        ang_vel_w = rs[10:13]
        joint_pos = robot.data.joint_pos[self._env_id].detach().cpu().numpy()
        return pos, quat_wxyz, lin_vel_w, ang_vel_w, joint_pos

    def _read_command(self):
        try:
            term = self._env.command_manager.get_term("base_velocity")
            cmd = term.vel_command_b[self._env_id].detach().cpu().numpy()
            cmd_vx = float(cmd[0]) if cmd.size > 0 else 0.0
            cmd_vy = float(cmd[1]) if cmd.size > 1 else 0.0
            cmd_yaw = float(cmd[2]) if cmd.size > 2 else 0.0
        except Exception:
            cmd_vx = cmd_vy = cmd_yaw = 0.0
        return cmd_vx, cmd_vy, cmd_yaw

    def _sync_shadow(self, pos, quat_wxyz, joint_pos) -> None:
        data = self._mj_data
        if self._free_qpos_addr is not None:
            a = self._free_qpos_addr
            data.qpos[a:a + 3] = pos
            # MuJoCo and IsaacLab both store free joint quat as (w, x, y, z).
            data.qpos[a + 3:a + 7] = quat_wxyz
        for sim_idx, addr in enumerate(self._sim_to_qpos):
            if addr >= 0:
                data.qpos[addr] = float(joint_pos[sim_idx])

        # If we have a receiver K1 in the shadow model, sync its pose too.
        if self._has_receiver:
            try:
                recv = self._env.scene["receiver"]
                rs = recv.data.root_state_w[self._env_id].detach().cpu().numpy()
                rjp = recv.data.joint_pos[self._env_id].detach().cpu().numpy()
                if self._receiver_free_qpos_addr is not None:
                    b = self._receiver_free_qpos_addr
                    data.qpos[b:b + 3] = rs[0:3]
                    data.qpos[b + 3:b + 7] = rs[3:7]
                for sim_idx, addr in enumerate(self._sim_to_receiver_qpos):
                    if addr >= 0:
                        data.qpos[addr] = float(rjp[sim_idx])
            except Exception:
                pass

        self._mj.mj_forward(self._mj_model, data)

    def _quat_to_R(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _scene_offset(self, *, zero_z: bool = True) -> np.ndarray:
        """Offset that compensates for mjviser's camera-tracking displacement.

        ``mjviser.scene.ViserMujocoScene`` sets ``scene_offset = -tracked_pos``
        (full 3D) and applies it to everything it renders. To draw a node in
        the same frame as the MJ-rendered K1 we must apply the same offset.

        :param zero_z: keep the legacy behaviour (offset z = 0) for callers
            that only care about xy alignment — e.g. velocity arrows anchored
            at the kicker's trunk look natural without z compensation.
            Soccer-scene overlays should pass ``zero_z=False`` so that the
            ball sits on the floor instead of floating at trunk height.
        """
        if not getattr(self._scene, "camera_tracking_enabled", False):
            return np.zeros(3)
        tracked_id = getattr(self._scene, "_tracked_body_id", None)
        if tracked_id is None or tracked_id >= self._mj_model.nbody:
            return np.zeros(3)
        offset = -self._mj_data.xpos[tracked_id].copy()
        if zero_z:
            offset[2] = 0.0
        return offset

    def _update_arrows(self, pos, R, lin_vel_w, ang_vel_w, cmd) -> None:
        cmd_vx, cmd_vy, cmd_yaw = cmd
        scale = self._arrow_scale
        z_off = self._arrow_z
        offset = self._scene_offset()

        # body → world helper (scale included so lengths render reasonably)
        def b2w(local_vec: np.ndarray) -> np.ndarray:
            return pos + R @ (local_vec * scale)

        base_offset = np.array([0.0, 0.0, z_off])
        origin = b2w(base_offset)

        lin_vel_b = R.T @ lin_vel_w
        ang_vel_b = R.T @ ang_vel_w

        self._arrow_cmd_lin.update(
            origin, b2w(base_offset + np.array([cmd_vx, cmd_vy, 0.0])), offset)
        self._arrow_cmd_ang.update(
            origin, b2w(base_offset + np.array([0.0, 0.0, cmd_yaw])), offset)
        self._arrow_actual_lin.update(
            origin, b2w(base_offset + np.array([lin_vel_b[0], lin_vel_b[1], 0.0])), offset)
        self._arrow_actual_ang.update(
            origin, b2w(base_offset + np.array([0.0, 0.0, ang_vel_b[2]])), offset)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def joystick_enabled(self) -> bool:
        return self._joystick_enabled

    @property
    def joystick_command(self) -> tuple[float, float, float]:
        return self._joystick_vx, self._joystick_vy, self._joystick_yaw

    @property
    def speed_multiplier(self) -> float:
        return self._speed_multiplier

    @property
    def paused(self) -> bool:
        return bool(self._scene.paused)

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def update(self, *, force: bool = False) -> None:
        """Push current env state to the viser scene."""
        if self._scene.paused and not force:
            self._info_md.content = self._info_text()
            return

        if not force:
            self._step_count += 1
            self._total_steps += 1
            if self._step_count % self._update_freq != 0:
                return
            now = time.monotonic()
            interval = self._min_interval / max(self._speed_multiplier, 0.1)
            if (now - self._last_update_time) < interval:
                return
            self._last_update_time = now

        try:
            pos, quat_wxyz, lin_vel_w, ang_vel_w, joint_pos = self._read_robot_state()
        except Exception:
            logger.exception("BoosterViserBridge: failed to read robot state")
            return
        cmd = self._read_command()

        self._sync_shadow(pos, quat_wxyz, joint_pos)

        with self._server.atomic():
            self._scene.update_from_mjdata(self._mj_data)
            R = self._quat_to_R(quat_wxyz)
            if self._show_arrows:
                self._update_arrows(pos, R, lin_vel_w, ang_vel_w, cmd)
            self._update_soccer_overlay()
            if force or self._total_steps % 30 == 0:
                self._info_md.content = self._info_text()

    def _update_soccer_overlay(self) -> None:
        if (
            self._ball_node is None
            and not self._goal_nodes
            and self._field_node is None
            and self._fov_kicker is None
        ):
            return
        overlay = self._read_overlay_state()
        # Use the full 3D camera-tracking offset so soccer-scene primitives
        # share the floor frame with the mjviser-rendered K1.
        offset = self._scene_offset(zero_z=False)

        if self._ball_node is not None and overlay.get("ball") is not None:
            self._ball_node.update(overlay["ball"], offset)
        for name, node in self._goal_nodes.items():
            p = overlay.get(name)
            if p is not None:
                node.update(p, offset)
        if self._field_node is not None and overlay.get("field") is not None:
            self._field_node.position = overlay["field"] + offset
        if self._pass_target_node is not None:
            show_pass_target = False
            try:
                cmd = self._env.command_manager.get_term("soccer_kick")
                show_pass_target = self._show_pass_target_marker and (not bool(cmd.is_shoot[self._env_id].item()))
            except Exception:
                show_pass_target = False
            self._pass_target_node.set_visible(show_pass_target)
            if show_pass_target and overlay.get("pass_target") is not None:
                self._pass_target_node.update(overlay["pass_target"], offset)

        if self._fov_kicker is not None and self._kicker_head_idx >= 0:
            try:
                robot = self._env.scene["robot"]
                head_pos = robot.data.body_pos_w[
                    self._env_id, self._kicker_head_idx
                ].detach().cpu().numpy()
                head_quat = robot.data.body_quat_w[
                    self._env_id, self._kicker_head_idx
                ].detach().cpu().numpy()
                cam_pos = head_pos + _quat_apply(head_quat, self._fov_cfg["offset_pos"])
                cam_quat = _quat_mul(head_quat, self._fov_cfg["offset_quat"])
                self._fov_kicker.update(cam_pos, cam_quat, offset)
            except Exception:
                logger.debug("FOV frustum update failed", exc_info=True)

    def apply_joystick(self, cmd_term) -> None:
        """If joystick is enabled, write its values into the command term.

        Writes to ``cmd_term.vel_command_b`` for *every* env so the override is
        applied uniformly in vectorised eval.
        """
        if not self._joystick_enabled:
            return
        try:
            cmd_term.vel_command_b[:, 0] = self._joystick_vx
            cmd_term.vel_command_b[:, 1] = self._joystick_vy
            if cmd_term.vel_command_b.shape[1] > 2:
                cmd_term.vel_command_b[:, 2] = self._joystick_yaw
        except (AttributeError, IndexError):
            pass

    def apply_soccer_controls(self) -> bool:
        """Apply Controls-tab soccer mode/target overrides to all envs."""
        if self._soccer_mode_override == "Auto":
            return False
        try:
            cmd = self._env.command_manager.get_term("soccer_kick")
        except Exception:
            return False

        device = cmd.is_shoot.device
        is_shoot = self._soccer_mode_override == "Shoot"
        cmd._is_shoot[:] = is_shoot

        env_origins = self._env.scene.env_origins.to(device=device)
        if is_shoot:
            cmd._target_pos_w[:, 0] = env_origins[:, 0] + float(cmd.cfg.goal_line_x)
            cmd._target_pos_w[:, 1] = env_origins[:, 1]
        else:
            target_x = env_origins[:, 0] + float(self._soccer_pass_target_x)
            target_y = env_origins[:, 1] + float(self._soccer_pass_target_y)
            cmd._pass_target_pos_w[:, 0] = target_x
            cmd._pass_target_pos_w[:, 1] = target_y
            cmd._target_pos_w[:, 0] = target_x
            cmd._target_pos_w[:, 1] = target_y

        norm = max(0.0, min(1.0, float(self._soccer_strength_norm)))
        shoot_strength_range = getattr(cmd.cfg, "shoot_target_strength_range", None)
        if is_shoot and shoot_strength_range is not None:
            lo, hi = shoot_strength_range
        else:
            lo, hi = getattr(cmd.cfg, "target_strength_range", (0.0, 1.0))
        cmd._target_strength[:] = float(lo) + norm * (float(hi) - float(lo))
        cmd._target_strength_normalized[:] = norm

        self._refresh_soccer_target_vectors(cmd)
        return True

    def _refresh_soccer_target_vectors(self, cmd) -> None:
        """Refresh target direction tensors after a play-time mode override."""
        try:
            target_vec_xy = cmd._target_pos_w - cmd._ball_pos_w[:, :2]
            target_vec_norm = torch.linalg.norm(target_vec_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            cmd._target_dir_w[:] = target_vec_xy / target_vec_norm

            target_vec_w = torch.zeros(cmd.num_envs, 3, device=cmd._target_dir_w.device)
            target_vec_w[:, :2] = cmd._target_dir_w
            target_b = torch.zeros(cmd.num_envs, 2, device=cmd._target_dir_w.device)
            # Import here to keep the bridge cheap for non-Isaac contexts.
            from isaaclab.utils.math import quat_apply_inverse

            target_b[:] = quat_apply_inverse(cmd._robot_yaw_quat, target_vec_w)[:, :2]
            target_b_norm = torch.linalg.norm(target_b, dim=-1, keepdim=True).clamp_min(1e-6)
            cmd._target_dir_b[:] = target_b / target_b_norm

            pass_rel_w = torch.zeros(cmd.num_envs, 3, device=cmd._target_dir_w.device)
            pass_rel_w[:, :2] = cmd._pass_target_pos_w - cmd._robot_pos_w[:, :2]
            pass_rel_b = quat_apply_inverse(cmd._robot_yaw_quat, pass_rel_w)[:, :2]
            pass_norm = torch.linalg.norm(pass_rel_b, dim=-1).clamp_min(1e-6)
            pass_mask = (~cmd._is_shoot).float()
            cmd._pass_target_dir_b[:] = (pass_rel_b / pass_norm.unsqueeze(-1)) * pass_mask.unsqueeze(-1)
            cmd._pass_target_dist_b[:] = pass_norm * pass_mask
        except Exception:
            logger.debug("Soccer target vector refresh failed", exc_info=True)

    def close(self) -> None:
        try:
            self._server.stop()
        except Exception:
            try:
                self._server.close()
            except Exception:
                pass
