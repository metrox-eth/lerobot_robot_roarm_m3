"""Config for the RoArm-M3 follower under LeRobot 0.5.2.

Pattern from the reachy2 template
(`lerobot/robots/reachy2/configuration_reachy2.py`): a `RobotConfig` subclass,
`@RobotConfig.register_subclass(...)`, `kw_only`, carrying `id`, `port`,
`max_relative_target`, and a `cameras` dict.

Difference vs reachy2: we DON'T pre-build cameras in __post_init__ (reachy2 has
3 fixed network cameras). The RoArm has two USB cameras whose indices change at
every reboot, so we let the caller (record/eval loop) inject OpenCVCameraConfig
entries into `cameras`. The base `RobotConfig.__post_init__` validates that any
camera given has width/height/fps set (it raises otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("roarm_m3_follower")
@dataclass(kw_only=True)
class RoarmM3FollowerConfig(RobotConfig):
    """Configuration for a single RoArm-M3 follower arm + its cameras.

    Attributes:
        port: serial device of the follower arm, e.g. "/dev/ttyUSB1".
        baudrate: serial baudrate. 1 Mbps since the 2026-05-31 firmware reflash
            (Serial.begin(1000000) — same rate as the internal servo bus), which
            lifts the leader/follower state read from ~23Hz to ~42Hz (enables 30fps).
            Was 115200 before the reflash.
        max_relative_target: per-step safety clamp on |goal - present| (degrees).
            float => same limit on every joint; dict => per-joint; None => off.
            Matches the contract field on the base Robot.
        cameras: mapping of short logical name -> CameraConfig. The record/eval
            loop injects these via camera_setup.build_camera_configs() from the
            single-source camera_config.json. The 3-cam rig:

                "wrist"     OpenCVCameraConfig    (/dev/video_wrist, NO rotation)
                "side"      OpenCVCameraConfig    (/dev/video_front, fixed scene)
                "top_front" RealSenseCameraConfig (338122302234, use_depth=True)

            width/height/fps are mandatory — RobotConfig.__post_init__ raises
            otherwise. wrist is upright since the 2026-05-30 wrist_roll re-zero (no
            ROTATE_180 anymore). See `.claude/commands/cameras.md`.
    """

    # Serial connection to the follower arm.
    port: str = "/dev/ttyUSB1"
    baudrate: int = 1000000

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety. float => same value for all joints; dict => per-joint;
    # None => disabled.
    max_relative_target: float | dict[str, float] | None = None

    # Cameras keyed by short logical name ("wrist", "front"). Empty by default;
    # the record/eval loop injects OpenCVCameraConfig entries (indices vary per boot).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
