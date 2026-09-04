#!/usr/bin/env bash
# Downloads the RoArm-M3 link meshes from Waveshare's own repository (roarm_ws, branch
# ros2-humble). They are Waveshare's files and are not redistributed here.
set -e
cd "$(dirname "$0")/meshes"
BASE="https://raw.githubusercontent.com/waveshareteam/roarm_ws/ros2-humble/src/roarm_main/roarm_description/meshes/roarm_m3"
for f in base_link link1 link2 link3 link4 link5 gripper_link; do
  curl -fsSL -o "$f.stl" "$BASE/$f.stl" && echo "ok $f.stl"
done
