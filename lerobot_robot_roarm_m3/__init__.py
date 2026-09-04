"""lerobot_robot_roarm_m3 — RoArm-M3 follower plugin for LeRobot 0.5.2.

The `lerobot_robot_` prefix is REQUIRED for auto-discovery: LeRobot's
`register_third_party_plugins()` (lerobot/utils/import_utils.py:205-233) scans
installed distributions whose Name starts with `lerobot_robot_` and imports the
module of that same name. Importing THIS module must therefore register the
config subclass — which happens as a side effect of importing
`config_roarm_m3_follower` (the `@RobotConfig.register_subclass("roarm_m3_follower")`
decorator runs on import).
"""

from .config_roarm_m3_follower import RoarmM3FollowerConfig
from .roarm_m3_follower import (
    ANGLES_MAX,
    ANGLES_MIN,
    JOINT_NAMES,
    RoarmM3Follower,
)

__all__ = [
    "ANGLES_MAX",
    "ANGLES_MIN",
    "JOINT_NAMES",
    "RoarmM3Follower",
    "RoarmM3FollowerConfig",
]
