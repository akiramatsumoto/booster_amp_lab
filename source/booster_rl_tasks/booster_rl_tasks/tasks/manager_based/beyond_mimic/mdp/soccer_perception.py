"""Virtual perception module for the Booster K1 soccer kick task.

Ports the SPIRIT of mjlab's ``VirtualPerception`` (see
``mjlab/src/mjlab/tasks/kick/mdp/perception.py``) to the Isaac Lab managers-
based stack:

  * FOV check on a body-mounted optical-frame camera (default: Head_2 link
    on K1 with the RealSense D435i offset).
  * Distance-dependent Bernoulli detection (in-FOV AND in-range), per-env DR.
  * Distance-dependent Gaussian noise sigma(d) = a*d + b on world xy, with
    per-env DR multipliers on (a, b).
  * Latency ring buffer (per-env latency expressed in control steps).
  * Update-frequency decimation (per-env Hz mean); reset-on-miss fast-lock so
    a fresh detection arrives within one step after a dropout streak.
  * Blind episodes (per-env Bernoulli flag at reset zeroing detection prob).
  * Hold-on-miss flag (default False for K1 — emit zeros when ball_mask=0).
  * ``last_seen_dt`` counter (seconds since the most recent detection).

All operations are fully vectorized over ``num_envs``. Quaternions are wxyz,
matching the Isaac Lab convention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class VirtualPerceptionCfg:
    """Parameters for the simulated head-camera ball detector.

    Defaults track the mjlab K1 RealSense D435i deployment, with the V1.49
    DR knobs (lower detection-prob floor + 10% blind episodes).
    """

    # ---------------------------------------------------------------- camera
    camera_body_name: str = "Head_2"
    """Robot body the camera is rigidly attached to (post-yaw, post-pitch)."""

    # Isaac Lab K1 URDF: Head_2 origin sits at the pitch joint with
    # +X = head forward, +Z = head up (verified by the inertial CoM at
    # (0.011, -0.001, 0.081)). The camera sits on the forehead — about
    # 6 cm forward and 10 cm above the joint.
    # The previous defaults (0.00124, 0.04553, -0.01582) and the rotated
    # quaternion are mjlab MJCF values (+Y forward, +Z down) and do NOT
    # match the Isaac Lab body-axis convention.
    camera_offset_pos: tuple[float, float, float] = (0.06, 0.0, 0.10)
    """Camera optical-frame origin offset in the camera body's local frame (m)."""

    camera_offset_quat: tuple[float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0,
    )
    """Camera optical-frame orientation relative to the camera body (wxyz).

    Identity by default — the camera looks forward along Head_2's +X axis.
    """

    # ------------------------------------------------------------------ FOV
    # D-Robotics RDK Stereo Camera Module (SC230AI, 2.28 mm lens):
    # 178° diagonal / 150° horizontal / 80° vertical.
    fov_h_deg: float = 150.0
    """Horizontal FOV (degrees, full angle). Default: D-Robotics RDK Stereo Camera."""

    fov_v_deg: float = 80.0
    """Vertical FOV (degrees, full angle). Default: D-Robotics RDK Stereo Camera."""

    # --------------------------------------------------------------- detection
    max_detection_range: float = 7.0
    """Distance beyond which detection probability decays toward zero (m)."""

    range_decay: float = 2.0
    """Soft falloff distance for range-based detection probability (m)."""

    detection_prob_in_fov_range: tuple[float, float] = (0.30, 0.95)
    """Per-env absolute range for in-FOV detection probability."""

    blind_prob: float = 0.10
    """Per-episode probability that an env is fully blind for the entire
    episode (detection_prob forced to 0)."""

    # -------------------------------------------------------------- xy noise
    noise_a: float = 0.05
    """Linear coefficient in distance-dependent noise model sigma(d) = a*d + b."""

    noise_b: float = 0.08
    """Offset coefficient (m) in sigma(d) = a*d + b."""

    noise_a_range: tuple[float, float] = (0.7, 1.5)
    """Per-env multiplicative range on ``noise_a``."""

    noise_b_range: tuple[float, float] = (0.7, 1.5)
    """Per-env multiplicative range on ``noise_b``."""

    # --------------------------------------------------------------- latency
    latency_mean_range: tuple[float, float] = (0.080, 0.160)
    """Per-env uniform range for the latency Gaussian mean (s)."""

    latency_std_s: float = 0.018
    """Std of the per-env latency Gaussian (s)."""

    update_hz_mean_range: tuple[float, float] = (20.0, 30.0)
    """Per-env uniform range for the detector update-rate Gaussian mean (Hz)."""

    update_hz_std: float = 1.06
    """Std of the detector update-rate Gaussian (Hz)."""

    buffer_size: int = 16
    """Latency ring buffer length, in control steps. Should cover
    ``ceil((max latency mean + 3*latency_std)/dt)``."""

    # -------------------------------------------------------------- misc
    hold_last_on_miss: bool = False
    """When True, hold the most recent valid value on a miss. When False
    (default for K1), emit zeros. ``ball_mask`` is 0 either way."""


def _pitch_down_quat(deg: float) -> tuple[float, float, float, float]:
    """Quaternion rotating the camera optical +X axis downward about local +Y."""
    half = math.radians(float(deg)) * 0.5
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def soccer_vision_train_cfg(
    *,
    hold_last_on_miss: bool = True,
    detection_prob: float = 0.90,
    blind_prob: float = 0.0,
    max_detection_range: float = 7.0,
) -> VirtualPerceptionCfg:
    """Shared soccer virtual-camera preset for kicker/receiver/defender roles.

    The values track the LVDRS/output-sc perception model: structured ball
    detections, 90% in-FOV detection up to 7 m, distance-dependent xy noise,
    roughly 25 Hz detector updates, and ~116 ms perception latency.
    """
    return VirtualPerceptionCfg(
        camera_offset_quat=_pitch_down_quat(40.0),
        hold_last_on_miss=hold_last_on_miss,
        blind_prob=blind_prob,
        max_detection_range=max_detection_range,
        detection_prob_in_fov_range=(detection_prob, detection_prob),
        noise_a=0.124,
        noise_b=0.149,
        noise_a_range=(0.8, 1.2),
        noise_b_range=(0.8, 1.2),
        latency_mean_range=(0.116, 0.116),
        latency_std_s=0.018,
        update_hz_mean_range=(25.36, 25.36),
        update_hz_std=1.06,
    )


def soccer_vision_repair_cfg() -> VirtualPerceptionCfg:
    """Easy shared-camera preset for short approach-repair curricula.

    This keeps the same API and last-seen semantics as the training preset but
    reduces latency/noise/dropout so the policy can relearn clean approach and
    foot placement before harder perception randomization is reintroduced.
    """
    return VirtualPerceptionCfg(
        camera_offset_quat=_pitch_down_quat(40.0),
        hold_last_on_miss=True,
        blind_prob=0.0,
        max_detection_range=8.0,
        detection_prob_in_fov_range=(0.95, 1.0),
        noise_a=0.02,
        noise_b=0.03,
        noise_a_range=(0.8, 1.2),
        noise_b_range=(0.8, 1.2),
        latency_mean_range=(0.020, 0.060),
        latency_std_s=0.006,
        update_hz_mean_range=(30.0, 45.0),
        update_hz_std=1.0,
    )


class VirtualPerception:
    """Stateful simulator of a head-camera ball detection pipeline.

    Each ``update`` call:
      1. Compose camera world pose from the head body pose + static offset.
      2. Transform the ball world position into the camera optical frame.
      3. Run FOV check (forward AND |yaw|<half AND |pitch|<half).
      4. Apply distance-attenuated Bernoulli detection.
      5. Inject Gaussian xy noise scaled by camera-distance.
      6. Express the noisy xy in the robot body-yaw frame.
      7. Decimate to the per-env detector rate (reset-on-miss fast-lock).
      8. Push (pos, mask) into the latency ring buffer and read out the
         buffer entry corresponding to each env's sampled latency.
      9. Update the per-env ``last_seen_dt`` counter.
    """

    def __init__(
        self,
        cfg: VirtualPerceptionCfg,
        robot: "Articulation",
        num_envs: int,
        dt: float,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)

        # Resolve camera body index once. (Re-resolved on first update if
        # ``robot`` happens to be re-bound — kept here for the fast path.)
        head_ids, _ = robot.find_bodies([cfg.camera_body_name], preserve_order=True)
        if len(head_ids) == 0:
            raise ValueError(
                f"camera_body_name {cfg.camera_body_name!r} not found on robot "
                f"(have {robot.body_names!r})"
            )
        self._head_idx = int(head_ids[0])

        # Static optical-frame offset (camera body -> camera optical).
        self._cam_offset_pos = torch.tensor(
            cfg.camera_offset_pos, dtype=torch.float32, device=self.device
        )
        self._cam_offset_quat = torch.tensor(
            cfg.camera_offset_quat, dtype=torch.float32, device=self.device
        )

        # Pre-converted FOV half-angles (radians).
        self._fov_h_half = math.radians(cfg.fov_h_deg) * 0.5
        self._fov_v_half = math.radians(cfg.fov_v_deg) * 0.5

        # Per-env DR coefficients ------------------------------------------
        N = self.num_envs
        d = self.device
        self._noise_a_per_env = torch.full((N,), cfg.noise_a, dtype=torch.float32, device=d)
        self._noise_b_per_env = torch.full((N,), cfg.noise_b, dtype=torch.float32, device=d)
        self._detection_prob_per_env = torch.full(
            (N,), 0.5 * (cfg.detection_prob_in_fov_range[0] + cfg.detection_prob_in_fov_range[1]),
            dtype=torch.float32, device=d,
        )
        self._blind_per_env = torch.zeros((N,), dtype=torch.bool, device=d)

        # Latency / update-rate per-env counters ---------------------------
        self._latency_steps = torch.zeros(N, dtype=torch.long, device=d)
        self._update_period_steps = torch.ones(N, dtype=torch.long, device=d)
        self._steps_since_update = torch.zeros(N, dtype=torch.long, device=d)

        # Ring buffer (head -> newest entry) -------------------------------
        self._buffer_pos = torch.zeros(cfg.buffer_size, N, 2, dtype=torch.float32, device=d)
        self._buffer_mask = torch.zeros(cfg.buffer_size, N, dtype=torch.float32, device=d)
        self._buffer_head = 0

        # Most recent (pre-buffer) detection state -------------------------
        self._last_pos = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._last_mask = torch.zeros(N, dtype=torch.float32, device=d)

        # Outputs (post-buffer) --------------------------------------------
        self._ball_pos_b = torch.zeros(N, 2, dtype=torch.float32, device=d)
        self._ball_mask = torch.zeros(N, dtype=torch.float32, device=d)
        self._last_seen_dt = torch.zeros(N, dtype=torch.float32, device=d)
        self._in_fov = torch.zeros(N, dtype=torch.float32, device=d)
        self._range_prob = torch.zeros(N, dtype=torch.float32, device=d)
        self._detect_prob = torch.zeros(N, dtype=torch.float32, device=d)
        self._raw_detected = torch.zeros(N, dtype=torch.float32, device=d)
        self._ball_in_cam = torch.zeros(N, 3, dtype=torch.float32, device=d)

        # Cached camera pose (useful for debug viz) ------------------------
        self._cam_pos_w = torch.zeros(N, 3, dtype=torch.float32, device=d)
        self._cam_quat_w = torch.zeros(N, 4, dtype=torch.float32, device=d)
        self._cam_quat_w[:, 0] = 1.0

        # Initial per-env sampling -----------------------------------------
        self._sample_per_env(torch.arange(N, dtype=torch.long, device=d))

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset per-env buffers and resample latency / rate / DR coefficients."""
        if env_ids is None or (hasattr(env_ids, "numel") and env_ids.numel() == 0):
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._buffer_pos[:, env_ids] = 0.0
        self._buffer_mask[:, env_ids] = 0.0
        self._steps_since_update[env_ids] = 0
        self._last_pos[env_ids] = 0.0
        self._last_mask[env_ids] = 0.0
        self._ball_pos_b[env_ids] = 0.0
        self._ball_mask[env_ids] = 0.0
        self._last_seen_dt[env_ids] = 0.0
        self._in_fov[env_ids] = 0.0
        self._range_prob[env_ids] = 0.0
        self._detect_prob[env_ids] = 0.0
        self._raw_detected[env_ids] = 0.0
        self._ball_in_cam[env_ids] = 0.0
        self._sample_per_env(env_ids)

    # ---------------------------------------------------------------- sampling
    def _sample_per_env(self, env_ids: torch.Tensor) -> None:
        cfg = self.cfg
        n = env_ids.numel()
        if n == 0:
            return

        def _uniform(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(n, device=self.device) * (hi - lo) + lo

        # Per-env xy-noise coefficients (multiplicative on cfg base values).
        self._noise_a_per_env[env_ids] = cfg.noise_a * _uniform(*cfg.noise_a_range)
        self._noise_b_per_env[env_ids] = cfg.noise_b * _uniform(*cfg.noise_b_range)

        # Per-env in-FOV detection probability (absolute, not multiplicative).
        det_p = _uniform(*cfg.detection_prob_in_fov_range)

        # Blind episodes: force detection probability to zero.
        if cfg.blind_prob > 0.0:
            blind = torch.rand(n, device=self.device) < cfg.blind_prob
            self._blind_per_env[env_ids] = blind
            det_p = torch.where(blind, torch.zeros_like(det_p), det_p)
        else:
            self._blind_per_env[env_ids] = False
        self._detection_prob_per_env[env_ids] = det_p

        # Per-env latency in steps.
        latency_mean = _uniform(*cfg.latency_mean_range)
        latency = (
            latency_mean + torch.randn(n, device=self.device) * cfg.latency_std_s
        ).clamp_min(0.0)
        self._latency_steps[env_ids] = (
            (latency / self.dt).round().long().clamp_(0, cfg.buffer_size - 1)
        )

        # Per-env update-rate period in steps.
        hz_mean = _uniform(*cfg.update_hz_mean_range)
        hz = (
            hz_mean + torch.randn(n, device=self.device) * cfg.update_hz_std
        ).clamp_min(1.0)
        period_s = 1.0 / hz
        self._update_period_steps[env_ids] = (
            (period_s / self.dt).round().long().clamp_min_(1)
        )

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, robot: "Articulation", ball_pos_w: torch.Tensor) -> None:
        """Advance one control step.

        Args:
            robot: Articulation holding the camera body.
            ball_pos_w: World-frame ball position. Shape (num_envs, 3).
        """
        cfg = self.cfg
        N = self.num_envs
        d = self.device

        # ----- Camera world pose ----------------------------------------
        body_pos = robot.data.body_pos_w[:, self._head_idx]
        body_quat = robot.data.body_quat_w[:, self._head_idx]

        offset_pos_b = self._cam_offset_pos.expand(N, -1)
        offset_quat_b = self._cam_offset_quat.expand(N, -1)

        cam_pos_w = body_pos + quat_apply(body_quat, offset_pos_b)
        cam_quat_w = quat_mul(body_quat, offset_quat_b)
        self._cam_pos_w.copy_(cam_pos_w)
        self._cam_quat_w.copy_(cam_quat_w)

        # ----- Ball in camera optical frame -----------------------------
        ball_in_cam = quat_apply_inverse(cam_quat_w, ball_pos_w - cam_pos_w)
        self._ball_in_cam = ball_in_cam
        bx = ball_in_cam[:, 0]
        by = ball_in_cam[:, 1]
        bz = ball_in_cam[:, 2]
        distance = ball_in_cam.norm(dim=-1)

        # ----- FOV --------------------------------------------------------
        forward = bx > 1e-3
        yaw = torch.atan2(by, bx)
        pitch = torch.atan2(bz, bx)
        in_fov = forward & (yaw.abs() < self._fov_h_half) & (pitch.abs() < self._fov_v_half)
        self._in_fov = in_fov.float()

        # ----- Range probability (linear falloff) -----------------------
        range_prob = torch.where(
            distance < cfg.max_detection_range,
            torch.ones_like(distance),
            torch.clamp(
                1.0 - (distance - cfg.max_detection_range) / max(cfg.range_decay, 1e-6),
                min=0.0,
            ),
        )

        # ----- Bernoulli detection --------------------------------------
        p_detect = self._detection_prob_per_env * range_prob * in_fov.float()
        detected = torch.bernoulli(p_detect.clamp(0.0, 1.0)) > 0.5
        detected_f = detected.float()
        self._range_prob = range_prob.clamp(0.0, 1.0)
        self._detect_prob = p_detect.clamp(0.0, 1.0)
        self._raw_detected = detected_f

        # ----- xy noise (world frame, then yaw-rotate into body frame) --
        sigma = self._noise_a_per_env * distance + self._noise_b_per_env
        noise = torch.randn_like(ball_pos_w[:, :2]) * sigma.unsqueeze(-1)
        ball_xy_world = ball_pos_w[:, :2] + noise

        robot_pos_w = robot.data.root_pos_w
        robot_quat_w = robot.data.root_quat_w
        yq = yaw_quat(robot_quat_w)
        rel_xyz = torch.zeros(N, 3, device=d, dtype=ball_xy_world.dtype)
        rel_xyz[:, :2] = ball_xy_world - robot_pos_w[:, :2]
        rel_xy_b = quat_apply_inverse(yq, rel_xyz)[:, :2]

        # ----- Decimation + reset-on-miss fast lock ---------------------
        new_pos = torch.where(detected.unsqueeze(-1), rel_xy_b, self._last_pos)

        self._steps_since_update += 1
        do_update = (self._steps_since_update >= self._update_period_steps) | (
            self._last_mask < 0.5
        )

        # On an update tick where we actually detected, replace last_pos.
        self._last_pos = torch.where(
            (do_update & detected).unsqueeze(-1), new_pos, self._last_pos
        )
        # On an update tick, mask = current detection result (could be 0).
        self._last_mask = torch.where(do_update, detected_f, self._last_mask)
        # Reset the counter on update ticks.
        self._steps_since_update = torch.where(
            do_update, torch.zeros_like(self._steps_since_update), self._steps_since_update
        )

        # ----- Ring buffer write ----------------------------------------
        self._buffer_head = (self._buffer_head + 1) % cfg.buffer_size
        self._buffer_pos[self._buffer_head] = self._last_pos
        self._buffer_mask[self._buffer_head] = self._last_mask

        # ----- Ring buffer read at per-env latency ----------------------
        env_idx = torch.arange(N, device=d)
        read_head = (self._buffer_head - self._latency_steps) % cfg.buffer_size
        out_pos = self._buffer_pos[read_head, env_idx]
        out_mask = self._buffer_mask[read_head, env_idx]

        if not cfg.hold_last_on_miss:
            out_pos = out_pos * out_mask.unsqueeze(-1)

        self._ball_pos_b = out_pos
        self._ball_mask = out_mask

        # ----- last_seen_dt counter -------------------------------------
        # Reset when a detection lands (out_mask == 1), increment otherwise.
        self._last_seen_dt = torch.where(
            out_mask > 0.5,
            torch.zeros_like(self._last_seen_dt),
            self._last_seen_dt + self.dt,
        )

    # --------------------------------------------------------------- output
    @property
    def ball_pos_b(self) -> torch.Tensor:
        """Noisy ball xy in robot body-yaw frame. Shape (num_envs, 2)."""
        return self._ball_pos_b

    @property
    def ball_mask(self) -> torch.Tensor:
        """Float mask in {0, 1}; 1 when a (delayed) detection is available."""
        return self._ball_mask

    @property
    def last_seen_dt(self) -> torch.Tensor:
        """Seconds since the most recent detection (per env)."""
        return self._last_seen_dt

    @property
    def in_fov(self) -> torch.Tensor:
        """Current, non-latent FOV gate before Bernoulli/dropout."""
        return self._in_fov

    @property
    def range_prob(self) -> torch.Tensor:
        """Range attenuation factor used in the current detection probability."""
        return self._range_prob

    @property
    def detect_prob(self) -> torch.Tensor:
        """Current Bernoulli detection probability before latency buffering."""
        return self._detect_prob

    @property
    def raw_detected(self) -> torch.Tensor:
        """Current unbuffered Bernoulli result before decimation/latency."""
        return self._raw_detected

    @property
    def ball_in_cam(self) -> torch.Tensor:
        """Current ball position in the camera optical frame."""
        return self._ball_in_cam

    @property
    def cam_pos_w(self) -> torch.Tensor:
        return self._cam_pos_w

    @property
    def cam_quat_w(self) -> torch.Tensor:
        return self._cam_quat_w
