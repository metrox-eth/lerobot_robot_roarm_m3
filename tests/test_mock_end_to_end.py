"""MOCK end-to-end test for the RoArm-M3 follower plugin.

No real robot here (rule: physical action = metrox only). This validates:
  - import in py3.12,
  - auto-discovery registration via register_third_party_plugins(),
  - pinned feature order,
  - connect / get_observation / send_action with the DOUBLE-WRITE gripper,
  - clamps (limits + max_relative_target),
  - flat dicts at the boundary (no torch).

Run: <molmoact2_python> -m pytest tests/test_mock_end_to_end.py -q
"""

import numpy as np

import lerobot_robot_roarm_m3 as plugin
from lerobot_robot_roarm_m3 import (
    JOINT_NAMES,
    RoarmM3Follower,
    RoarmM3FollowerConfig,
)


# --------------------------------------------------------------------- mocks


class MockArm:
    """Stand-in for roarm_sdk.roarm.roarm. Records every call."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.state = [3.0, -98.0, 182.0, -2.0, 4.0, 73.0]  # home_position.json
        self.calls = []

    def joints_angle_get(self):
        return list(self.state)

    def joints_angle_ctrl(self, angles, speed, acc):
        self.calls.append(("joints_angle_ctrl", list(angles), speed, acc))
        self.state = list(angles)

    def joint_angle_ctrl(self, joint, angle, speed, acc):
        self.calls.append(("joint_angle_ctrl", joint, angle, speed, acc))

    def torque_set(self, cmd):
        self.calls.append(("torque_set", cmd))

    def disconnect(self):
        self.calls.append(("disconnect",))


class MockCam:
    """Minimal Camera: introspectable shape + async_read frame."""

    def __init__(self, h=480, w=640):
        self.height, self.width = h, w
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    def connect(self, warmup=True):
        self._connected = True

    def async_read(self, timeout_ms=200):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self):
        self._connected = False


def _make_follower(monkeypatch, max_relative_target=None):
    import lerobot_robot_roarm_m3.roarm_m3_follower as mod

    monkeypatch.setattr(mod, "roarm", MockArm)
    monkeypatch.setattr(mod, "make_cameras_from_configs", lambda cfgs: {})

    cfg = RoarmM3FollowerConfig(
        id="test_follower", port="/dev/null", max_relative_target=max_relative_target
    )
    robot = RoarmM3Follower(cfg)
    robot.cameras = {"wrist": MockCam(), "side": MockCam(), "top_front": MockCam()}
    return robot


# --------------------------------------------------------------------- tests


def test_auto_discovery_registration():
    """The config must register under the choice name used by the CLI."""
    from lerobot.robots.config import RobotConfig

    # register_subclass ran on import -> the choice name resolves to our config.
    name = RoarmM3FollowerConfig().type
    assert name == "roarm_m3_follower"
    assert RobotConfig.get_choice_class("roarm_m3_follower") is RoarmM3FollowerConfig


def test_feature_order_pinned():
    """observation/action features keep the pinned joint order; flat float schema."""
    cfg = RoarmM3FollowerConfig(id="x", port="/dev/null")
    robot = RoarmM3Follower(cfg)
    robot.cameras = {}  # no cams for pure-schema check

    expected_joint_keys = [f"{n}.pos" for n in JOINT_NAMES]
    assert list(robot.action_features.keys()) == expected_joint_keys
    assert list(robot.observation_features.keys())[:6] == expected_joint_keys
    assert all(v is float for v in robot.action_features.values())
    # explicit pinned order
    assert JOINT_NAMES == ["base", "shoulder", "elbow", "wrist_tilt", "wrist_roll", "gripper"]


def test_observation_features_include_cameras(monkeypatch):
    robot = _make_follower(monkeypatch)
    feats = robot.observation_features
    assert feats["wrist"] == (480, 640, 3)
    assert feats["side"] == (480, 640, 3)
    assert feats["top_front"] == (480, 640, 3)


def test_connect_get_send_roundtrip(monkeypatch):
    robot = _make_follower(monkeypatch)
    robot.connect()
    assert robot.is_connected

    # get_observation: flat dict, joint floats + camera arrays, no torch.
    obs = robot.get_observation()
    for n in JOINT_NAMES:
        assert isinstance(obs[f"{n}.pos"], float)
    # KNOWN-VALUE roundtrip: mock home -> ~same back (rule #2 functional test).
    assert abs(obs["base.pos"] - 3.0) < 1e-6
    assert abs(obs["gripper.pos"] - 73.0) < 1e-6
    assert obs["wrist"].shape == (480, 640, 3)
    assert isinstance(obs["wrist"], np.ndarray)

    # send_action: floor quantize -> 73.8 becomes 73; returns flat dict actually sent.
    action = {
        "base.pos": 10.0,
        "shoulder.pos": -90.0,
        "elbow.pos": 150.0,
        "wrist_tilt.pos": 0.0,
        "wrist_roll.pos": 5.0,
        "gripper.pos": 73.8,
    }
    sent = robot.send_action(action)
    assert sent["base.pos"] == 10.0
    assert sent["gripper.pos"] == 73.0  # floor of 73.8

    robot.disconnect()
    assert not robot.is_connected


def test_double_write_gripper(monkeypatch):
    """The worker must issue T:122 bundle AND a separate T:121 gripper write."""
    robot = _make_follower(monkeypatch)
    robot.connect()
    arm = robot._arm

    arm.calls.clear()
    robot.send_action({f"{n}.pos": 50.0 for n in JOINT_NAMES})

    # AsyncArmWorker runs in a thread; give it a moment to drain the queue.
    import time

    deadline = time.time() + 2.0
    names = []
    while time.time() < deadline:
        names = [c[0] for c in arm.calls]
        if "joints_angle_ctrl" in names and "joint_angle_ctrl" in names:
            break
        time.sleep(0.02)

    assert "joints_angle_ctrl" in names, "missing T:122 bundle write"
    assert "joint_angle_ctrl" in names, "missing separate T:121 gripper write (force!)"
    # the separate gripper write targets joint=6 with the gripper goal angle
    grip_call = next(c for c in arm.calls if c[0] == "joint_angle_ctrl")
    assert grip_call[1] == 6
    assert abs(grip_call[2] - 50.0) < 1e-6

    robot.disconnect()


def test_wrist_roll_and_limit_clamp(monkeypatch):
    """wrist_roll clamps to +-90 (cable), gripper clamps to <=300."""
    robot = _make_follower(monkeypatch)
    robot.connect()

    sent = robot.send_action(
        {
            "base.pos": 0.0,
            "shoulder.pos": 0.0,
            "elbow.pos": 0.0,
            "wrist_tilt.pos": 0.0,
            "wrist_roll.pos": 175.0,   # over +90 cable limit
            "gripper.pos": 999.0,      # over 300 max
        }
    )
    assert sent["wrist_roll.pos"] == 90.0
    assert sent["gripper.pos"] == 300.0
    robot.disconnect()


def test_max_relative_target_clamp(monkeypatch):
    """Per-step |goal - present| is capped when max_relative_target is set."""
    robot = _make_follower(monkeypatch, max_relative_target=5.0)
    robot.connect()
    # present base = 3.0 ; request +100 -> capped to +5 -> 8.0
    sent = robot.send_action({"base.pos": 103.0})
    assert sent["base.pos"] == 8.0
    robot.disconnect()


def test_plugin_exports():
    for sym in ("RoarmM3Follower", "RoarmM3FollowerConfig", "JOINT_NAMES",
                "ANGLES_MIN", "ANGLES_MAX"):
        assert hasattr(plugin, sym)
