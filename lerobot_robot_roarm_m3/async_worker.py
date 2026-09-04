"""Background serial-I/O threads for the RoArm-M3 follower.

Ported verbatim (logic-preserving) from the working 0.1.0 fork
`/home/openclaw/lerobot/lerobot/common/robot_devices/robots/roarm_m3.py:43-187`.

Why this exists: the roarm_sdk serial calls block ~35-50 ms each. The LeRobot 0.5.2
record/eval loop is synchronous and single-threaded, so a blocking get/set inside
the loop would cap the control rate well below the 20 fps target. These workers move
ALL serial I/O onto background threads:

  - AsyncArmReader : read-only poller (used for the passive leader in teleop).
  - AsyncArmWorker : owns reads AND writes for the follower in ONE thread, so a read
                     and a write never race on the same serial port.

The follower control path (write) reproduces the DOUBLE-WRITE gripper sequence:
  1. joints_angle_ctrl(...)  -> T:122 (SyncWritePosEx) : moves all 6 joints,
       but writes GOAL_TORQUE=0 on the CF35-12 gripper (no clamping force).
  2. joint_angle_ctrl(joint=6, ...) -> T:121 (handJointCtrlRad) : re-issues ONLY
       the gripper with GOAL_TORQUE=300 -> the jaw actually holds the cube.
Skipping step 2 leaves the gripper at force 0 and the cube slips, invisible to
shape/NaN checks. See `.claude/commands/gripper.md` (T:121 vs T:122) and
roarm_m3.py:169-171 for the original.
"""

from __future__ import annotations

import logging
import queue as _queue
import threading
import time

import numpy as np

# roarm_sdk is imported lazily by the caller; we only need the type for hints.
# Keeping the import here is safe (roarm_sdk is importable in py3.12, verified),
# but we guard it so this module imports even if the SDK is absent in CI.
try:  # pragma: no cover - exercised at runtime on the robot host
    from roarm_sdk.roarm import roarm
except Exception:  # pragma: no cover
    roarm = object  # type: ignore[assignment, misc]


class AsyncArmReader:
    """Background thread that continuously polls joints_angle_get() on one arm
    and stores the latest reading in a cache. The main control loop reads from
    the cache (~0 ms) instead of blocking on serial I/O (~35-50 ms).

    Uses the arm SDK's own threading.Lock so reads and writes on the same serial
    port interleave safely.

    Ported from roarm_m3.py:43-96.
    """

    def __init__(self, arm, name: str, sleep_between_reads_s: float = 0.005):
        self._arm = arm
        self._name = name
        self._sleep = sleep_between_reads_s
        self._pos: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set -> reader yields the serial port
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"AsyncArmReader-{name}"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def pause(self):
        """Yield the serial port — call before any blocking write sequence."""
        self._pause_event.set()
        time.sleep(0.02)  # let any in-flight read finish

    def resume(self):
        self._pause_event.clear()

    def get_pos(self) -> np.ndarray | None:
        with self._lock:
            return None if self._pos is None else self._pos.copy()

    def _run(self):
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.01)
                continue
            try:
                pos = self._arm.joints_angle_get()
                if pos is not None:
                    arr = np.array(pos, dtype=np.float32)
                    with self._lock:
                        self._pos = arr
            except Exception as exc:  # pragma: no cover - serial flakiness
                logging.debug(f"AsyncArmReader [{self._name}] error: {exc}")
            time.sleep(self._sleep)


class AsyncArmWorker:
    """Single background thread that owns ALL serial I/O for one follower arm.

    Main thread API (non-blocking, ~0 ms):
        worker.write(goal_pos, speed, acc)  — queue command, returns instantly
        worker.get_pos()                    — return latest cached joint state

    Background loop (sequential, zero lock contention with main thread):
        1. Drain write queue — execute latest command (joints_angle_ctrl + the
           separate gripper joint_angle_ctrl, i.e. the DOUBLE-WRITE).
        2. Periodic joint read every read_interval_s.

    Because reads and writes both happen in this single thread, they never race
    each other. No locking between main thread and worker is needed for serial
    access.

    Ported from roarm_m3.py:99-187.
    """

    def __init__(self, arm, name: str, read_interval_s: float = 0.15):
        self._arm = arm
        self._name = name
        self._read_interval = read_interval_s
        self._write_q: _queue.Queue = _queue.Queue(maxsize=1)
        self._pos: np.ndarray | None = None
        self._pos_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"AsyncArmWorker-{name}"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def pause(self):
        """Pause loop so caller can access arm directly (e.g. homing)."""
        self._pause_event.set()
        time.sleep(0.05)  # let any in-flight serial op finish

    def resume(self):
        self._pause_event.clear()

    def write(self, goal_pos: list, speed: int, acc: int) -> None:
        """Non-blocking write. Drops previous command if queue full."""
        cmd = (goal_pos, speed, acc)
        if self._write_q.full():
            try:
                self._write_q.get_nowait()
            except _queue.Empty:
                pass
        try:
            self._write_q.put_nowait(cmd)
        except _queue.Full:
            pass

    def get_pos(self) -> np.ndarray | None:
        with self._pos_lock:
            return None if self._pos is None else self._pos.copy()

    def _run(self):
        last_read = 0.0
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.01)
                continue

            # Priority 1: execute pending write (the DOUBLE-WRITE gripper sequence).
            try:
                goal_pos, speed, acc = self._write_q.get(timeout=0.005)
                self._arm.joints_angle_ctrl(angles=goal_pos, speed=speed, acc=acc)
                if len(goal_pos) >= 6:
                    # T:121 re-issues the gripper with GOAL_TORQUE=300 (force).
                    self._arm.joint_angle_ctrl(
                        joint=6, angle=float(goal_pos[5]), speed=speed, acc=acc
                    )
            except _queue.Empty:
                pass

            # Priority 2: periodic joint read.
            now = time.perf_counter()
            if now - last_read >= self._read_interval and not self._pause_event.is_set():
                try:
                    pos = self._arm.joints_angle_get()
                    if pos is not None:
                        with self._pos_lock:
                            self._pos = np.array(pos, dtype=np.float32)
                    last_read = now
                except Exception as exc:  # pragma: no cover - serial flakiness
                    logging.debug(f"AsyncArmWorker [{self._name}] read error: {exc}")
