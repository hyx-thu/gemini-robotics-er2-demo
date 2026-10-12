"""Unified Virtual/Mock Robot Backend for Spot and Tinybot on port 8888."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import time
from typing import Any

import fastapi
from fastapi import Response
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
import uvicorn

app = fastapi.FastAPI(title="Virtual Spot & Tinybot Backend", version="0.1.0")

STATE: dict[str, Any] = {
    "connected": True,
    "has_lease": True,
    "standing": True,
    "arm_state": "stowed",
    "gripper_open_fraction": 1.0,
    "holding_item": None,
    "current_waypoint": "dock_station",
    "waypoints": ["dock_station", "kitchen_counter", "office_desk", "snack_table"],
    "head_uv": (500, 500),
    "last_gesture": "idle",
    "last_action": "ready",
    "overlay_points": [],
}


def render_scene_jpeg(mode: str = "spot") -> bytes:
  """Render a synthetic 640x480 workspace scene with interactive objects."""
  img = Image.new("RGB", (640, 480), color=(28, 33, 43))
  draw = ImageDraw.Draw(img)

  # Floor / background grid
  for x in range(0, 640, 40):
    draw.line([(x, 0), (x, 260)], fill=(38, 44, 56), width=1)
  for y in range(0, 260, 40):
    draw.line([(0, y), (640, y)], fill=(38, 44, 56), width=1)

  # Tabletop surface
  draw.polygon([(40, 250), (600, 250), (640, 460), (0, 460)], fill=(120, 92, 68), outline=(155, 122, 92))

  # Objects on the table:
  # 1. Red drink can (left-center)
  if STATE["holding_item"] != "red drink can":
    draw.rectangle([140, 265, 195, 360], fill=(220, 48, 48), outline=(255, 190, 190), width=2)
    draw.ellipse([140, 255, 195, 275], fill=(235, 235, 235), outline=(180, 180, 180))
    draw.text((146, 305), "COLA", fill=(255, 255, 255))

  # 2. Blue coffee mug (center)
  if STATE["holding_item"] != "blue coffee mug":
    draw.rectangle([295, 285, 360, 365], fill=(48, 122, 230), outline=(190, 220, 255), width=2)
    draw.arc([350, 298, 382, 350], start=270, end=90, fill=(190, 220, 255), width=4)
    draw.text((310, 320), "MUG", fill=(255, 255, 255))

  # 3. Green apple (right-center)
  if STATE["holding_item"] != "green apple":
    draw.ellipse([445, 295, 510, 360], fill=(64, 196, 82), outline=(190, 255, 195), width=2)
    draw.line([(477, 282), (482, 296)], fill=(110, 75, 40), width=3)
    draw.text((458, 322), "APPLE", fill=(20, 50, 20))

  # HUD banner
  draw.rectangle([10, 10, 630, 76], fill=(15, 20, 28), outline=(70, 130, 220), width=1)
  t_str = time.strftime("%H:%M:%S")
  draw.text(
      (20, 18),
      f"VIRTUAL ROBOT SIMULATOR ({mode.upper()}) | {t_str} | Waypoint: {STATE['current_waypoint']}",
      fill=(110, 220, 255),
  )
  draw.text(
      (20, 38),
      f"Stand: {STATE['standing']} | Arm: {STATE['arm_state']} | Gripper: {STATE['gripper_open_fraction']:.1f} | Held: {STATE['holding_item'] or 'None'}",
      fill=(200, 215, 230),
  )
  draw.text(
      (20, 56),
      f"Last Action: {STATE['last_action']} | Head UV: {STATE['head_uv']} | Gesture: {STATE['last_gesture']}",
      fill=(160, 230, 160),
  )

  # Draw any Tinybot point overlays
  for pt in STATE["overlay_points"]:
    coords = pt.get("point") or [240, 320]
    label = str(pt.get("label", "target"))
    if len(coords) >= 2:
      py = int((coords[0] / 1000.0) * 480)
      px = int((coords[1] / 1000.0) * 640)
      draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(255, 60, 60), outline=(255, 255, 255), width=2)
      draw.text((px + 12, py - 6), label, fill=(255, 230, 90))

  # Crosshair at center / head_uv
  cx = int((STATE["head_uv"][0] / 1000.0) * 640)
  cy = int((STATE["head_uv"][1] / 1000.0) * 480)
  draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(255, 215, 0), width=1)
  draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(255, 215, 0), width=1)

  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=82)
  return buf.getvalue()


# ---------------------------------------------------------------------------
# Request schemas (matching Spot OpenAPI tool policies)
# ---------------------------------------------------------------------------
class LocalizeRequest(BaseModel):
  waypoint_id: str | None = None
  waypoint_name: str | None = None
  fiducial_init: str = "nearest"
  use_fiducial_id: int | None = None
  refine_fiducial_result_with_icp: bool = True
  do_ambiguity_check: bool = False
  refine_with_visual_features: bool = False
  verify_visual_features_quality: bool = False
  max_distance: float | None = None
  max_yaw: float | None = None


class NavigateRequest(BaseModel):
  name: str
  command_duration: float = 30.0
  timeout: float = 180.0
  feedback_interval: float = 1.0
  power_on: bool = True
  stand: bool = True
  take_lease: bool = True


class StandRequest(BaseModel):
  power_on: bool = True
  take_lease: bool = True
  timeout: float = 10.0


class SitRequest(BaseModel):
  take_lease: bool = True
  timeout: float = 10.0


class TeleopVelocityRequest(BaseModel):
  v_x: float = 0.0
  v_y: float = 0.0
  v_rot: float = 0.0
  duration: float = 0.6
  take_lease: bool = True
  power_on: bool = True
  stand: bool = True
  body_follow_arm: bool = False


class StopActionsRequest(BaseModel):
  take_lease: bool = True
  freeze_arm: bool = True


class LeaseRequest(BaseModel):
  take_lease: bool = True


class OscillationConfigRequest(BaseModel):
  enabled: bool = True
  sample_interval: float = 0.05
  window_sec: float = 1.0
  min_peak_to_peak_m: float = 0.03
  min_direction_changes: int = 4
  min_speed_mps: float = 0.05
  freeze_cooldown_sec: float = 2.0


class ArmJogRequest(BaseModel):
  dx: float = 0.0
  dy: float = 0.0
  dz: float = 0.0
  droll: float = 0.0
  dpitch: float = 0.0
  dyaw: float = 0.0
  seconds: float = 0.8
  take_lease: bool = True
  timeout: float = 3.0


class CameraRollRequest(BaseModel):
  direction: str = "clockwise"
  angle_rad: float = 0.2
  seconds: float = 0.8
  take_lease: bool = True


class Pose3D(BaseModel):
  frame_name: str = "vision"
  x: float = 0.5
  y: float = 0.0
  z: float = 0.2
  qw: float = 1.0
  qx: float = 0.0
  qy: float = 0.0
  qz: float = 0.0


class ReachDistanceRequest(BaseModel):
  pose: Pose3D


class ApproachPoseRequest(BaseModel):
  pose: Pose3D
  standoff_m: float = 0.05
  max_step_m: float = 0.5
  seconds: float = 2.0
  take_lease: bool = True
  timeout: float = 15.0


class PickupWaitRequest(BaseModel):
  monitor_sec: float = 5.0
  upward_threshold_m: float = 0.02
  sample_interval: float = 0.1
  open_duration_sec: float = 1.0
  take_lease: bool = True
  gripper_timeout: float = 5.0
  stow_timeout: float = 10.0


class GripperRequest(BaseModel):
  open_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
  fraction: float | None = None
  max_vel: float | None = None
  max_acc: float | None = None
  take_lease: bool = True


class DetectTargetRequest(BaseModel):
  instruction: str
  model: str | None = None
  api_key: str | None = None


class PickRequest(BaseModel):
  instruction: str = "red drink can"
  timeout: float = 30.0
  take_lease: bool = True


class GraspPixelRequest(BaseModel):
  x: float = 0.5
  y: float = 0.5
  take_lease: bool = True
  grip_max_torque_nm: float = 2.0


class ForceChangeRequest(BaseModel):
  threshold_newtons: float = 3.0
  sample_window_sec: float = 5.0
  interval_sec: float = 0.1


# ---------------------------------------------------------------------------
# Spot Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def get_health():
  return {
      "status": "ok",
      "connected": STATE["connected"],
      "has_lease": STATE["has_lease"],
      "standing": STATE["standing"],
      "arm_state": STATE["arm_state"],
      "virtual_mode": True,
  }


@app.get("/lease")
def get_lease():
  return {"has_lease": STATE["has_lease"], "owner": "virtual-spot-controller"}


@app.post("/lease/take")
def take_lease():
  STATE["has_lease"] = True
  STATE["last_action"] = "take_lease"
  return {"status": "ok", "has_lease": True}


@app.get("/localization")
def get_localization():
  return {
      "localized": True,
      "localization": {
          "waypoint_id": "wp_0",
          "waypoint_name": STATE["current_waypoint"],
      },
  }


@app.post("/localize")
def localize(req: LocalizeRequest):
  if req.waypoint_name:
    STATE["current_waypoint"] = req.waypoint_name
  STATE["last_action"] = f"localize({STATE['current_waypoint']})"
  return {"status": "ok", "localized": True, "waypoint_name": STATE["current_waypoint"]}


@app.get("/battery")
def get_battery():
  return {
      "charge_percentage": 94.0,
      "estimated_runtime_sec": 4800,
      "motor_power_state": "STATE_ON",
  }


@app.post("/faults/behavior/clear")
def clear_faults():
  return {"cleared_faults": [], "failed_faults": []}


@app.get("/waypoints")
def get_waypoints():
  return [
      {"id": f"wp_{i}", "name": name}
      for i, name in enumerate(STATE["waypoints"])
  ]


@app.post("/navigate")
def navigate(req: NavigateRequest):
  STATE["current_waypoint"] = req.name
  STATE["standing"] = True
  STATE["last_action"] = f"navigate -> {req.name}"
  return {"status": "arrived", "destination": req.name}


@app.post("/stand")
def stand(req: StandRequest):
  STATE["standing"] = True
  STATE["last_action"] = "stand"
  return {"status": "standing"}


@app.post("/sit")
def sit(req: SitRequest):
  STATE["standing"] = False
  STATE["last_action"] = "sit"
  return {"status": "sitting"}


@app.post("/teleop/velocity")
def drive(req: TeleopVelocityRequest):
  STATE["last_action"] = f"drive(vx={req.v_x}, vy={req.v_y}, vrot={req.v_rot})"
  return {"status": "completed", "duration": req.duration}


@app.post("/actions/stop")
def stop_actions(req: StopActionsRequest):
  STATE["last_action"] = "stop"
  return {"status": "stopped"}


@app.get("/images/sources")
def get_image_sources():
  return [
      {"name": "hand_color_image", "width": 640, "height": 480, "type": "RGB"},
      {"name": "hand_depth_in_hand_color_frame", "width": 640, "height": 480, "type": "DEPTH"},
  ]


@app.get("/images/{source}")
def get_image(source: str):
  return Response(content=render_scene_jpeg(mode="spot"), media_type="image/jpeg")


@app.post("/arm/deploy")
def deploy_arm(req: LeaseRequest):
  STATE["arm_state"] = "deployed"
  STATE["last_action"] = "deploy_arm"
  return {"status": "deployed"}


@app.post("/arm/carry")
def carry_arm(req: LeaseRequest):
  STATE["arm_state"] = "carry"
  STATE["last_action"] = "carry_arm"
  return {"status": "carry"}


@app.post("/arm/freeze")
def freeze_arm(req: LeaseRequest):
  STATE["last_action"] = "freeze_arm"
  return {"status": "frozen"}


@app.get("/arm/oscillation-monitor")
def get_oscillation_monitor():
  return {"enabled": True, "oscillating": False, "samples": 42}


@app.post("/arm/oscillation-monitor")
def configure_oscillation_monitor(req: OscillationConfigRequest):
  return {"status": "configured", "enabled": req.enabled}


@app.post("/arm/jog")
def jog_arm(req: ArmJogRequest):
  STATE["arm_state"] = "jogged"
  STATE["last_action"] = f"jog_arm(dpitch={req.dpitch:.2f}, dyaw={req.dyaw:.2f})"
  return {"status": "completed"}


@app.post("/arm/camera-roll")
def roll_gripper_camera(req: CameraRollRequest):
  STATE["last_action"] = f"roll_camera({req.direction})"
  return {"status": "completed"}


@app.post("/arm/reach-distance")
def get_arm_reach_distance(req: ReachDistanceRequest):
  dist = math.sqrt(req.pose.x**2 + req.pose.y**2 + req.pose.z**2)
  return {"distance_m": round(dist, 3), "within_arm_reach": dist <= 0.75}


@app.post("/arm/approach")
def approach_pose(req: ApproachPoseRequest):
  STATE["arm_state"] = "approached"
  STATE["last_action"] = "approach_pose"
  return {"status": "completed"}


@app.post("/arm/approach-whole-body")
def approach_pose_whole_body(req: ApproachPoseRequest):
  STATE["arm_state"] = "approached_whole_body"
  STATE["last_action"] = "approach_pose_whole_body"
  return {"status": "completed"}


@app.post("/arm/stow")
def stow_arm(req: LeaseRequest):
  STATE["arm_state"] = "stowed"
  STATE["last_action"] = "stow_arm"
  return {"status": "stowed"}


@app.post("/arm/stow-smart")
def stow_smart(req: dict[str, Any]):
  STATE["arm_state"] = "carry" if STATE["holding_item"] else "stowed"
  STATE["last_action"] = f"stow_smart -> {STATE['arm_state']}"
  return {"status": STATE["arm_state"]}


@app.post("/pickup/wait")
def wait_for_pick_up(req: PickupWaitRequest):
  released = STATE["holding_item"]
  STATE["holding_item"] = None
  STATE["gripper_open_fraction"] = 0.0
  STATE["arm_state"] = "stowed"
  STATE["last_action"] = f"wait_for_pick_up (released {released})"
  return {"status": "handed_over", "released_item": released}


@app.post("/gripper/open")
def open_gripper(req: GripperRequest):
  frac = req.fraction if req.fraction is not None else req.open_fraction
  STATE["gripper_open_fraction"] = frac
  if frac > 0.4:
    STATE["holding_item"] = None
  STATE["last_action"] = f"open_gripper({frac:.2f})"
  return {"status": "opened", "open_fraction": frac}


@app.post("/gripper/close")
def close_gripper(req: GripperRequest):
  STATE["gripper_open_fraction"] = req.open_fraction
  STATE["last_action"] = "close_gripper"
  return {"status": "closed", "open_fraction": req.open_fraction}


@app.post("/detect/pick-target")
def detect_pick_target(req: DetectTargetRequest):
  inst = req.instruction.lower()
  if "blue" in inst or "mug" in inst:
    label, nx, ny = "blue coffee mug", 0.51, 0.68
  elif "green" in inst or "apple" in inst:
    label, nx, ny = "green apple", 0.74, 0.68
  else:
    label, nx, ny = "red drink can", 0.26, 0.65
  STATE["last_action"] = f"detect({label})"
  return {
      "label": label,
      "confidence": 0.96,
      "target": {
          "label": label,
          "normalized_x": nx,
          "normalized_y": ny,
          "pixel_x": int(nx * 640),
          "pixel_y": int(ny * 480),
      },
  }


@app.post("/pick")
def pick_object(req: PickRequest):
  det = detect_pick_target(DetectTargetRequest(instruction=req.instruction))
  STATE["holding_item"] = det["label"]
  STATE["arm_state"] = "carry"
  STATE["gripper_open_fraction"] = 0.2
  STATE["last_action"] = f"pick({det['label']})"
  return {"status": "picked", "item": det["label"], "target": det["target"]}


@app.post("/manipulation/grasp-pixel")
def grasp_pixel(req: GraspPixelRequest):
  if req.x < 0.4:
    item = "red drink can"
  elif req.x < 0.65:
    item = "blue coffee mug"
  else:
    item = "green apple"
  STATE["holding_item"] = item
  STATE["arm_state"] = "carry"
  STATE["gripper_open_fraction"] = 0.2
  STATE["last_action"] = f"grasp_pixel({item})"
  return {"status": "grasped", "item": item}


@app.post("/manipulation/place-pixel")
def place_pixel(req: GraspPixelRequest):
  placed = STATE["holding_item"]
  STATE["holding_item"] = None
  STATE["gripper_open_fraction"] = 1.0
  STATE["last_action"] = f"place_pixel({placed})"
  return {"status": "placed", "item": placed}


@app.post("/force/change")
def detect_force_change(req: ForceChangeRequest):
  STATE["last_action"] = "detect_force_change"
  return {"force_changed": True, "delta_newtons": req.threshold_newtons + 1.2}


@app.post("/connect")
def connect_spot(req: dict[str, Any]):
  STATE["connected"] = True
  return {"status": "connected", "virtual": True}


# ---------------------------------------------------------------------------
# Tinybot Endpoints
# ---------------------------------------------------------------------------
@app.get("/camera_image/")
def get_tinybot_camera_image(camera_id: str = "camera"):
  b64 = base64.b64encode(render_scene_jpeg(mode="tinybot")).decode("ascii")
  return Response(content=f'<img src="data:image/jpeg;base64,{b64}" />', media_type="text/html")


@app.get("/video-stream")
async def stream_tinybot_video():
  async def frame_generator():
    while True:
      jpg = render_scene_jpeg(mode="tinybot")
      yield (
          b"--frame\r\n"
          b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
      )
      await asyncio.sleep(0.5)

  return StreamingResponse(
      frame_generator(),
      media_type="multipart/x-mixed-replace; boundary=frame",
  )


@app.post("/make-gesture")
def tinybot_make_gesture(gesture: str = "nod", speed: int = 150):
  STATE["last_gesture"] = gesture
  STATE["last_action"] = f"gesture({gesture})"
  return {"status": f"completed gesture {gesture} at speed {speed}"}


@app.post("/move-absolute")
def tinybot_move_absolute(req: dict[str, Any]):
  STATE["last_action"] = f"move_absolute({req.get('angles')})"
  return {"status": "ok", "angles": req.get("angles", [])}


@app.post("/move-relative")
def tinybot_move_relative(req: dict[str, Any]):
  STATE["last_action"] = f"move_relative({req.get('relative_angles')})"
  return {"status": "ok", "relative_angles": req.get("relative_angles", [])}


@app.post("/look-to-uv")
def tinybot_look_to_uv(req: dict[str, Any]):
  u = int(req.get("u", 500))
  v = int(req.get("v", 500))
  STATE["head_uv"] = (u, v)
  STATE["last_action"] = f"look_to_uv({u}, {v})"
  return {"status": f"turned head to ({u}, {v})"}


@app.post("/point")
def tinybot_point(points: list[dict[str, Any]]):
  STATE["overlay_points"] = points
  STATE["last_action"] = f"point({len(points)} targets)"
  return {"status": "ok", "count": len(points)}


@app.post("/clear-overlay")
def tinybot_clear_overlay():
  STATE["overlay_points"] = []
  STATE["last_action"] = "clear_overlay"
  return {"status": "cleared"}


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8888, log_level="warning")
