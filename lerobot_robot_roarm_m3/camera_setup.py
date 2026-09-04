"""camera_setup.py — single-source camera wiring for the RoArm-M3 3-camera rig.

Both the record loop and the eval read their cameras through THIS module, from
`camera_config.json` (the source of truth, next to the plugin root). That is what
guarantees record<->eval consistency: same resolution, same frozen auto-settings,
same logical names, every time.

Why a separate module (and not just OpenCVCameraConfig fields):
  lerobot 0.5.2 camera configs expose ONLY width/height/fps/rotation/color_mode.
  They do NOT touch white-balance / exposure. So freezing the automatics (the
  thing that stops the green/blue tint drift and keeps record==eval) has to be
  done by US, AFTER the camera opens. There is no post-connect callback in
  lerobot, so the caller invokes `apply_frozen_settings(robot.cameras, raw)`
  immediately after `robot.connect()`.

Cameras (logical name -> physical):
  wrist     : OpenCV /dev/video_wrist  (eye-in-hand, upright since 2026-05-30 re-zero)
  side      : OpenCV /dev/video_front  (fixed scene camera, side view)
  top_front : RealSense 338122302234   (top-down; RGB into the v3 dataset, depth
              dumped IN PARALLEL as 16-bit PNG — see depth_dir / save_depth_png)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("camera-setup")

# camera_config.json lives at the plugin ROOT (one level above this package dir).
DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent.parent / "camera_config.json"


def load_camera_config(path: str | Path | None = None) -> dict:
    """Parse camera_config.json. Returns the raw dict (kept so apply_frozen_settings
    and the depth dump can read the controls/options later)."""
    p = Path(path) if path is not None else DEFAULT_CAMERA_CONFIG
    with open(p) as f:
        return json.load(f)


def camera_names(raw: dict) -> list[str]:
    """Logical camera names in config order (skips `_doc*` keys)."""
    return [k for k in raw.get("cameras", {}) if not k.startswith("_")]


def depth_dump_cameras(raw: dict) -> list[str]:
    """Names of cameras whose depth stream is dumped in parallel (dump_depth=true)."""
    return [
        k
        for k, c in raw.get("cameras", {}).items()
        if not k.startswith("_") and isinstance(c, dict) and c.get("dump_depth")
    ]


def _to_rotation(val: Any):
    from lerobot.cameras.configs import Cv2Rotation

    v = int(val or 0)
    if v == 270:  # the enum stores 270 as -90
        v = -90
    return Cv2Rotation(v)


def build_camera_configs(raw: dict) -> dict:
    """JSON -> {name: CameraConfig}. OpenCV for wrist/side, RealSense for top_front.

    fps/width/height are ALL set on every config (RobotConfig.__post_init__ rejects
    a partial set). Capture fps comes from `capture_fps` (30) — matches the control
    loop fps (30 since the 2026-05-31 1Mbps reflash; the D455F can't do 20fps@640x480
    anyway). Optional per-camera `fourcc` (None=auto): the webcams negotiate YUYV
    @30fps and hold it via the PCIe USB card — do NOT force MJPG (record+dataset are
    YUYV → MJPG would be a train/eval shift).
    """
    from lerobot.cameras.configs import ColorMode, Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig

    w = int(raw["resolution"]["width"])
    h = int(raw["resolution"]["height"])
    cfgs: dict = {}
    for key, c in raw.get("cameras", {}).items():
        if key.startswith("_") or not isinstance(c, dict):
            continue
        fps = int(c["capture_fps"])
        color = ColorMode(c.get("color_mode", "rgb"))
        rot = _to_rotation(c.get("rotation", 0))
        ctype = c["type"]
        if ctype == "opencv":
            backend = (
                Cv2Backends.V4L2
                if str(c.get("backend", "any")).lower() == "v4l2"
                else Cv2Backends.ANY
            )
            cfgs[key] = OpenCVCameraConfig(
                index_or_path=c["index_or_path"],
                fps=fps,
                width=w,
                height=h,
                color_mode=color,
                rotation=rot,
                backend=backend,
                fourcc=c.get("fourcc"),  # None=auto (current: YUYV @30fps, holds via PCIe USB).
                                         # Do NOT default to MJPG: record+dataset are YUYV → shift.
            )
        elif ctype == "intelrealsense":
            cfgs[key] = RealSenseCameraConfig(
                serial_number_or_name=str(c["serial_number_or_name"]),
                fps=fps,
                width=w,
                height=h,
                color_mode=color,
                use_depth=bool(c.get("use_depth", False)),
                rotation=rot,
            )
        else:
            raise ValueError(f"Unknown camera type {ctype!r} for camera {key!r}")
    return cfgs


def apply_frozen_settings(cameras: dict, raw: dict) -> None:
    """Freeze auto white-balance / exposure on each camera, AFTER connect().

    USB (OpenCVCamera): v4l2-ctl on the kernel fd (independent of the cv2 cap, so
      it's safe while the background read thread is live). Voie A = freeze WB, KEEP
      auto-exposure (these webcams have no manual gain -> fixed exposure goes dark).
    RealSense: settle-then-lock auto-exposure (the connect() warmup already let AE
      settle), then force WB. Order matters: disable auto BEFORE writing the value.
    Mock / unknown cameras: no-op (isinstance falls through).
    """
    import subprocess

    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    except Exception:  # pragma: no cover
        OpenCVCamera = ()  # type: ignore[assignment]
    try:
        import pyrealsense2 as rs

        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    except Exception:  # pragma: no cover
        RealSenseCamera = ()  # type: ignore[assignment]
        rs = None

    for key, cam in cameras.items():
        spec = raw.get("cameras", {}).get(key)
        if not isinstance(spec, dict):
            continue

        if OpenCVCamera and isinstance(cam, OpenCVCamera):
            ctrls = spec.get("v4l2_controls", {})
            if ctrls:
                # dict order is significant: disable-auto keys come before their
                # fixed-value keys in camera_config.json.
                ctrl_str = ",".join(f"{k}={v}" for k, v in ctrls.items())
                subprocess.run(
                    ["v4l2-ctl", f"--device={cam.index_or_path}", f"--set-ctrl={ctrl_str}"],
                    check=True,
                )
                log.info(f"[{key}] froze v4l2 controls: {ctrl_str}")

        elif RealSenseCamera and isinstance(cam, RealSenseCamera) and rs is not None:
            opts = spec.get("rs_options", {})
            sensor = cam.rs_profile.get_device().first_color_sensor()
            if opts.get("settle_then_lock", True) and sensor.supports(rs.option.enable_auto_exposure):
                settled = sensor.get_option(rs.option.exposure)
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, settled)
                log.info(f"[{key}] locked exposure at settled value {settled:.0f}")
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
                wb = float(opts.get("white_balance", 4600))
                sensor.set_option(rs.option.white_balance, wb)
                log.info(f"[{key}] locked white_balance at {wb:.0f}")
        # else: mock / unknown camera -> nothing to freeze.


def read_depth_cached(cam):
    """Latest depth frame (uint16 mm) from the camera's async-thread cache, or None.

    NON-BLOCKING: returns the depth frame the background thread already captured
    (aligned with the latest color frame the loop recorded). We deliberately do
    NOT call `cam.read_depth()` here — it clears the new-frame event and blocks
    until the next frame (~1/fps s), which would drag the 20fps record loop down.
    Returns None for cameras with no depth cache (e.g. mock cams) -> caller skips.
    """
    lock = getattr(cam, "frame_lock", None)
    if lock is None:
        return None
    with lock:
        d = getattr(cam, "latest_depth_frame", None)
        return None if d is None else d.copy()


def save_depth_png(depth, path: str | Path) -> None:
    """Write a uint16 (H,W) depth map (mm) as a 16-bit lossless PNG.

    cv2.imwrite preserves uint16 single-channel as a 16-bit PNG natively — no
    h264, no 8-bit truncation, so the millimetre precision survives intact.
    """
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), depth)
