# RoArm-M3 + gripper B in MuJoCo

An exact kinematic model of the Waveshare RoArm-M3 with its parallel-jaw gripper
("High-performance Robotic Arm Gripper", gripper B), as we run it in our simulator.

## What is in here

- `roarm_m3_gripper_b.xml` — the MJCF: the six arm joints from Waveshare's URDF, the gripper mounted
  on the wrist roll, and the gripper's seven joints (servo cog, two gear levers, two small levers, two jaws).
- `gripper_b_kinematics.py` — the measured topology and the exact cascade. Each side of the gripper is a
  perfect parallelogram: gear lever A→P1 = 43.2 mm, small lever G→P2 = 43.2 mm, jaw (coupler) P1→P2 =
  29.7 mm = |A→G|, so the jaw translates and never rotates. Law: bot = +θ, jaw = −θ, gears 21 teeth 1:1
  (servo → gear_L → gear_R). Pivots were extracted numerically from the part STLs (bore fitting) and
  cross-checked in MuJoCo; residuals are in `gripper_b_v3_validation.json` (all under 0.002 mm).
  `python gripper_b_kinematics.py` re-runs the checks; `--view` animates the gripper.
- `meshes/gripper_b_parts/jaw_thin_*.stl`, `claw_ext_*.stl` — our own jaw plates and claw extensions
  (the fingers we actually run).

## What is not in here (Waveshare's meshes)

Waveshare's repositories carry no license file, so we do not redistribute their geometry.

- Arm links: `./fetch_meshes.sh` downloads `base_link`, `link1`…`link5`, `gripper_link` from
  [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws) (branch `ros2-humble`) into `meshes/`.
- Gripper body and levers: export them from Waveshare's gripper STEP file (available from their wiki) as
  STL, **in assembly coordinates, millimetres**, one file per part, into `meshes/gripper_b_parts/` with these
  names: `body`, `sevo_cog`, `lever_top_L`, `lever_top_R`, `lever_bottom_L`, `lever_bottom_R`, `gripper_L`,
  `gripper_R`. Loaded at the same pose they reassemble the gripper without any per-part calibration.

Waveshare: if you would like these files hosted here, say so and we add them.

## Conventions

- Units: metres in the MJCF (meshes scaled 0.001), radians. Gripper joints range ±1.6 rad.
- Kinematic use (write `qpos`, `mj_forward`) is what we do for visualisation and IK; the position actuators
  exist but no `<equality>` closes the loops, so under `mj_step` drive the seven gripper joints through
  `set_grip(theta)` from the script rather than the servo actuator alone.
- The wrist camera and the gripper are children of `link5` (wrist roll), as on the real arm.
