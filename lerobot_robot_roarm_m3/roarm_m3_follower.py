"""RoArm-M3 follower as a LeRobot 0.5.2 Robot (native v3 dataset).

Modeled on the reachy2 wrap-SDK template
(`lerobot/robots/reachy2/robot_reachy2.py`): a thin `Robot` subclass that wraps a
device SDK directly (no MotorsBus). The control logic is ported from the working
0.1.0 fork `/home/openclaw/lerobot/lerobot/common/robot_devices/robots/roarm_m3.py`,
which gives 90-95% pick-and-place with ACT v3. Citations to that file are inline.

Frontier contract (Robot, robot.py):
  - flat dict observations/actions, NO torch.Tensor at the boundary
    (RobotObservation/RobotAction = dict[str, Any]).
  - get_observation -> {<joint>.pos: float, ...} + {<cam>: np.ndarray}
  - send_action(dict) -> dict actually sent.

Pinned joint order (THIS is the dataset schema — do not reorder):
  [base, shoulder, elbow, wrist_tilt, wrist_roll, gripper]
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation

from .async_worker import AsyncArmWorker
from .config_roarm_m3_follower import RoarmM3FollowerConfig

# roarm_sdk is importable in py3.12 (verified). Guard the import so this module
# still imports in a CI env without the SDK; connect() will raise clearly if used.
try:
    from roarm_sdk.roarm import roarm
except Exception:  # pragma: no cover
    roarm = None  # type: ignore[assignment]


# Pinned joint order = the dataset schema. roarm_m3.py:263.
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_tilt", "wrist_roll", "gripper"]

# Per-joint angle limits [base, shoulder, elbow, wrist_tilt, wrist_roll, gripper].
# Wrist-roll (index 4) is restricted to +-90 deg to protect the gripper cables.
# Ported from roarm_m3.py:232-234.
WRIST_R_LIMIT = 90
ANGLES_MIN = [-190, -110, -70, -110, -WRIST_R_LIMIT, -10]
ANGLES_MAX = [190, 110, 190, 110, WRIST_R_LIMIT, 300]


def _quantize_action(goal_pos) -> np.ndarray:
    """Quantize a joint goal vector before sending to the firmware.

    Ported verbatim from roarm_m3.py:22-40. Mode is selected at eval time via the
    env var ROARM_ACTION_MODE (A/B without touching code/config):

      "floor" (DEFAULT, PROVEN — ACT v3 90-95%): astype(int32) truncates toward
              zero -> 1 deg staircase WITH a systematic downward bias
              (73.8 -> 73, 74.9 -> 74).
      "round": np.rint then int32 -> removes the downward bias, keeps the 1 deg step.
      "float": send float degrees -> removes the staircase (servo res 0.088 deg) IF
               the firmware accepts floats. Validate by roundtrip on the robot first
               (send 73.6 deg -> joints_angle_get() ~73.6). See .claude/commands/eval.md.

    The default preserves the proven path exactly; everything else is opt-in.
    """
    arr = goal_pos.numpy() if hasattr(goal_pos, "numpy") else np.asarray(goal_pos)
    mode = os.environ.get("ROARM_ACTION_MODE", "floor").lower()
    if mode == "float":
        return arr.astype(np.float64)
    if mode == "round":
        return np.rint(arr).astype(np.int32)
    return arr.astype(np.int32)  # "floor" = proven default


def _clamp_limits(goal_pos: list) -> list:
    """Clamp each joint to [ANGLES_MIN, ANGLES_MAX]. roarm_m3.py:518."""
    return [max(ANGLES_MIN[i], min(ANGLES_MAX[i], v)) for i, v in enumerate(goal_pos)]


def disable_info_flow(arm) -> bool:
    """Stop the firmware's continuous feedback flow. Send {"T":605,"cmd":0} once at
    connect so reads return FRESH positions instead of stale buffered ones.

    Our firmware ships with `InfoPrint = 2` ('flow feedback', RoArm-M3_config.h:9):
    it streams T:1051 frames non-stop, the OS RX buffer fills to ~4 KB, and every
    joints_angle_get() then returns a STALE (FIFO) reading — the leader-follower lag
    balloons to several seconds (verified live 2026-05-31). cmd:0 = no stream.
    """
    ser = next((o for o in (getattr(arm, a, None) for a in dir(arm))
                if hasattr(o, "write") and hasattr(o, "in_waiting")), None)
    if ser is None:
        return False
    try:
        ser.write(b'{"T":605,"cmd":0}\n')
        time.sleep(0.1)
        ser.reset_input_buffer()
        return True
    except Exception:  # pragma: no cover
        return False


class RoarmM3Follower(Robot):
    """Single RoArm-M3 follower arm + its USB cameras, as a LeRobot 0.5.2 Robot.

    Wraps roarm_sdk directly (no MotorsBus). All serial I/O runs on a background
    AsyncArmWorker so the synchronous 0.5.2 record/eval loop keeps ~20 fps:
    get_observation reads the cache (~0 ms) and send_action enqueues the write
    (~0 ms).
    """

    config_class = RoarmM3FollowerConfig
    name = "roarm_m3_follower"

    def __init__(self, config: RoarmM3FollowerConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.name

        self._arm = None  # roarm SDK handle, created in connect()
        self._worker: AsyncArmWorker | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

        self.logs: dict[str, float] = {}

    # ------------------------------------------------------------------ features

    @property
    def _motors_features(self) -> dict[str, type]:
        # Pinned order via OrderedDict -> {base.pos: float, ... gripper.pos: float}.
        return OrderedDict((f"{name}.pos", float) for name in JOINT_NAMES)

    @property
    def _cameras_features(self) -> dict[str, tuple[int, int, int]]:
        # Short keys ("wrist", "front") -> (H, W, C). reachy2 camera_features pattern.
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        """Flat dict: 6 joint floats + camera image shapes. Callable when disconnected."""
        return {**self._motors_features, **self._cameras_features}

    @property
    def action_features(self) -> dict[str, type]:
        """Flat dict: the 6 joint goal positions. Same keys as send_action expects."""
        return dict(self._motors_features)

    # ------------------------------------------------------------------ state

    @property
    def is_connected(self) -> bool:
        if self._arm is None:
            return False
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return cams_ok

    @property
    def is_calibrated(self) -> bool:
        # No <id>.json calibration file; home_position.json is the source of truth.
        return True

    def calibrate(self) -> None:
        # No-op: the RoArm has no per-robot motor-offset calibration here.
        pass

    def configure(self) -> None:
        """One-time setup: enable follower torque (active, holds position).

        reachy2 calls turn_on()/reset_default_limits() here; our equivalent is
        torque_set(cmd=1) on the follower. roarm_m3.py:312-313.
        """
        if self._arm is not None:
            self._arm.torque_set(cmd=1)

    # ------------------------------------------------------------------ lifecycle

    def connect(self, calibrate: bool = True) -> None:
        """Open the serial arm + cameras, start the async worker.

        Args:
            calibrate: kept for the Robot contract; calibrate() is a no-op here.
        """
        if roarm is None:
            raise ImportError(
                "roarm_sdk is not importable; cannot connect RoarmM3Follower."
            )
        if self.is_connected:
            raise ConnectionError(
                "RoarmM3Follower is already connected. Do not call connect() twice."
            )

        # baudrate 115200 matches the working 0.1.0 path (roarm_m3.py:223).
        self._arm = roarm(
            roarm_type="roarm_m3", port=self.config.port, baudrate=self.config.baudrate
        )
        # Probe the serial link (raises if the port is wrong / contended).
        self._arm.joints_angle_get()

        # Kill the firmware's continuous feedback flow (InfoPrint=2) — otherwise reads
        # return stale buffered positions and the control loop lags by seconds.
        disable_info_flow(self._arm)

        # Enable follower torque (active).
        self.configure()

        # Connect cameras.
        for cam in self.cameras.values():
            cam.connect()

        # Background worker owns ALL serial I/O (reads + writes) in one thread,
        # so a read and a write never race. roarm_m3.py:327-330.
        self._worker = AsyncArmWorker(self._arm, f"follower_{self.id}", read_interval_s=0.15)
        self._worker.start()
        time.sleep(0.20)  # one read cycle to populate the position cache

        if calibrate and not self.is_calibrated:
            self.calibrate()

    # ------------------------------------------------------------------ I/O

    def _read_joint_positions(self) -> np.ndarray:
        """6 joint angles (deg) from the worker cache (~0 ms), with a blocking fallback."""
        cached = self._worker.get_pos() if self._worker is not None else None
        if cached is None:
            cached = np.array(self._arm.joints_angle_get(), dtype=np.float32)
        return np.asarray(cached, dtype=np.float32)

    def get_observation(self) -> RobotObservation:
        """Flat observation: {<joint>.pos: float} + {<cam>: np.ndarray (HWC uint8 RGB)}.

        No torch at the boundary. Joint keys follow the pinned order; camera keys
        are the short logical names ("wrist", "front").
        """
        if not self.is_connected:
            raise ConnectionError("RoarmM3Follower is not connected. Call connect() first.")

        obs: RobotObservation = {}

        before = time.perf_counter()
        pos = self._read_joint_positions()
        for i, name in enumerate(JOINT_NAMES):
            obs[f"{name}.pos"] = float(pos[i])
        self.logs["read_pos_dt_s"] = time.perf_counter() - before

        for cam_key, cam in self.cameras.items():
            before_cam = time.perf_counter()
            obs[cam_key] = cam.async_read()
            self.logs[f"read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_cam

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Enqueue a joint goal; return the dict actually sent.

        Steps (ported from roarm_m3.py:496-529):
          1. strip ".pos", read joints in the pinned order.
          2. _quantize_action (floor by default via ROARM_ACTION_MODE).
          3. clamp to ANGLES_MIN/MAX (wrist_roll already +-90 via the limits).
          4. enqueue on the worker -> non-blocking (~0 ms). The worker performs the
             DOUBLE-WRITE: joints_angle_ctrl (T:122) THEN joint_angle_ctrl(joint=6)
             (T:121, GOAL_TORQUE=300) so the gripper actually clamps. Skipping the
             second write leaves the jaw at force 0 and the cube slips, invisible to
             shape/NaN checks. See .claude/commands/gripper.md.

        max_relative_target clamping (per-step |goal - present| cap) is applied here
        when configured, mirroring roarm_m3.py:503-508.
        """
        if not self.is_connected:
            raise ConnectionError("RoarmM3Follower is not connected. Call connect() first.")

        # Read goals in the pinned order (tolerate missing keys -> hold current pos).
        present = self._read_joint_positions()
        goal = np.array(
            [float(action.get(f"{name}.pos", present[i])) for i, name in enumerate(JOINT_NAMES)],
            dtype=np.float32,
        )

        # Optional safety clamp on the relative step magnitude.
        mrt = self.config.max_relative_target
        if mrt is not None:
            if isinstance(mrt, dict):
                cap = np.array([float(mrt.get(name, np.inf)) for name in JOINT_NAMES], dtype=np.float32)
            else:
                cap = np.full(6, float(mrt), dtype=np.float32)
            diff = np.clip(goal - present, -cap, cap)
            goal = present + diff

        goal = _quantize_action(goal)
        goal = goal.tolist() if isinstance(goal, np.ndarray) else list(goal)
        goal = _clamp_limits(goal)

        speed, acc = 500, 50  # roarm_m3.py:519-520 defaults
        if self._worker is not None:
            self._worker.write(goal, speed, acc)
        else:  # pragma: no cover - worker is always present after connect()
            self._arm.joints_angle_ctrl(angles=goal, speed=speed, acc=acc)
            if len(goal) >= 6:
                self._arm.joint_angle_ctrl(joint=6, angle=float(goal[5]), speed=speed, acc=acc)

        # Return what was actually sent, in the same flat schema.
        return {f"{name}.pos": float(goal[i]) for i, name in enumerate(JOINT_NAMES)}

    # ------------------------------------------------------------------ teardown

    def disconnect(self) -> None:
        """Stop the worker, release torque, close serial + cameras."""
        if not self.is_connected and self._arm is None:
            raise ConnectionError(
                "RoarmM3Follower is not connected. Call connect() before disconnecting."
            )

        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        # Torque-off so the arm can be moved by hand safely.
        if self._arm is not None:
            try:
                self._arm.torque_set(cmd=0)
            except Exception as exc:  # pragma: no cover
                logging.debug(f"torque_set(0) on disconnect failed: {exc}")
            try:
                self._arm.disconnect()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"arm.disconnect() failed: {exc}")
            self._arm = None

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"camera disconnect failed: {exc}")
