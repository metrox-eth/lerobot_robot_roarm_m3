# lerobot_robot_roarm_m3

[Waveshare RoArm-M3](https://www.waveshare.com/wiki/RoArm-M3) (5-DoF + gripper, ESP32, JSON over serial)
follower plugin for [LeRobot](https://github.com/huggingface/lerobot), as a **third-party device**, no fork needed.

LeRobot auto-discovers any installed package whose name starts with `lerobot_robot_`: this one registers
`roarm_m3_follower`, so the standard CLIs (`lerobot-record`, `lerobot-replay`, eval loops, …) can use
`--robot.type=roarm_m3_follower`.

This is the plugin we run in production at Show Robotics: it recorded our datasets (LeRobot v3 format,
30 fps, three cameras) and it is what our ACT policies drive. It started as upstream PR
[huggingface/lerobot#3716](https://github.com/huggingface/lerobot/pull/3716); the maintainers asked for a
third-party package instead, which is this repository.

## What is different about the RoArm-M3

The arm is not a Feetech/Dynamixel bus. It is driven by an onboard ESP32 that speaks **JSON over
serial** (Waveshare's `roarm_sdk`). Every SDK call blocks for ~35–50 ms, which would starve a 30 Hz
record/eval loop, so this plugin runs all serial I/O on a **background worker thread** and the robot
methods read the latest state / post the latest goal without waiting (`async_worker.py`).

## Verified

- LeRobot **0.5.2** on real hardware (record → v3 dataset → train ACT → eval on the arm). LeRobot 0.6.x
  is not verified yet; the plugin follows the 0.6 third-party conventions, reports welcome.
- `roarm_sdk` **0.1.0/0.1.1** ([waveshareteam/waveshare_roarm_sdk](https://github.com/waveshareteam/waveshare_roarm_sdk)).
- Unit tests against a mocked SDK: `pytest tests`.

## Install

```bash
pip install "lerobot>=0.5.2" roarm_sdk
pip install -e .          # or, once published: pip install lerobot_robot_roarm_m3
```

## Use

```python
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_roarm_m3 import RoarmM3FollowerConfig

cfg = RoarmM3FollowerConfig(port="/dev/ttyUSB1", baudrate=115200)   # stock firmware
robot = make_robot_from_config(cfg)
robot.connect()
obs = robot.get_observation()      # joint positions in degrees + camera frames
robot.send_action({...})           # goal positions in degrees, same keys
robot.disconnect()
```

Or from the CLI, e.g. `lerobot-record --robot.type=roarm_m3_follower --robot.port=/dev/ttyUSB1 ...`.

Cameras are passed as the usual `cameras` dict of `CameraConfig`. `camera_setup.py` builds them from a
single JSON file and freezes auto-exposure / white balance after connect, so record and eval see the
same images; `examples/camera_config.example.json` is our three-camera rig (two USB cameras + a RealSense).

## Hardware notes (things that bit us)

- **Baud rate.** The stock firmware talks at **115200**. The config default is 1 000 000 because our two arms
  run a reflashed firmware with `Serial.begin(1000000)`, which lifts the state read rate from ~23 Hz to
  ~42 Hz (needed for a clean 30 fps). On a stock arm, set `baudrate=115200`.
- **One process per serial port.** A second reader on the same port (a teleop server, a monitor) corrupts
  the JSON stream. Kill it before recording.
- **Speed ceiling.** The servos stream at roughly 30°/s through the SDK path; faster goals are clipped by the
  arm, not by this plugin. Use `max_relative_target` to keep steps small.
- **Gripper.** Our home pose is `[0, -98, 182, -2, 1.5, 115]` with the gripper open at 115°. Open/close values
  are rig-specific, measure yours.
- **Dupont connectors** and a Raspberry-Pi-class host talking straight to the arm cost us weeks. A serial
  adapter that does 1 Mbps cleanly, and short cables, matter more than any code here.

## Simulation: the arm and its gripper in MuJoCo

`sim/` has our MJCF of the RoArm-M3 with the parallel-jaw gripper (gripper B) as an exact parallelogram rig,
pivots extracted from the part STLs, plus the kinematics script and its validation. Waveshare's meshes are
fetched from their repository, not redistributed. See `sim/README.md`.

## Status and scope

Follower only. A passive RoArm-M3 leader teleoperator exists in the original PR branch and will be
published as `lerobot_teleoperator_roarm_m3_leader` if there is demand; we record our demonstrations with a
VR controller instead. Issues and PRs welcome.

License: Apache-2.0.
