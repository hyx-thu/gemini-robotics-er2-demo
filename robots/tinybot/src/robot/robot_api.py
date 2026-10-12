import time
import cv2
import fastapi
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from enum import Enum
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tinybot import TinyBot
except ImportError as e:
    logger.error(f"Failed to import TinyBot: {e}")
    raise

from camera_streamer import CameraStreamer

app = fastapi.FastAPI()

camera_streamer = CameraStreamer()

import argparse
import os

def _get_device_name() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device-name", "-d", "--device", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.device_name:
        return args.device_name
    return os.environ.get("TINYBOT_DEVICE", os.environ.get("ROBOT_DEVICE_NAME", "/dev/ttyUSB0"))

device_name = _get_device_name()

# Initialize TinyBot
try:
    tiny_bot = TinyBot(DEVICENAME=device_name, DXL_CHAIN=[1, 2])
    logger.info(f"TinyBot initialized successfully on {device_name}.")
except Exception as e:
    logger.warning(f"Failed to initialize TinyBot hardware on {device_name} (expected if not connected): {e}")
    tiny_bot = None

@app.on_event("startup")
def startup_event():
    camera_streamer.start()

@app.on_event("shutdown")
def shutdown_event():
    camera_streamer.stop()

def gen_frames():
    while True:
        frame = camera_streamer.get_frame()
        if frame is None:
            # If we failed to get a frame, sleep a bit to avoid spinning
            time.sleep(0.1)
            continue
        
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video-stream")
def video_stream():
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/health")
def health():
    return {"status": "ok", "camera_mode": camera_streamer.mode}

class PointOverlay(BaseModel):
    u: int = Field(..., ge=0, le=1000)
    v: int = Field(..., ge=0, le=1000)
    label: str

@app.post("/point")
def point(points: list[PointOverlay]):
    overlays = [(p.u, p.v, p.label) for p in points]
    camera_streamer.set_overlays(overlays)
    return {"status": "ok", "points_count": len(points)}

@app.post("/clear-overlay")
def clear_overlay():
    camera_streamer.clear_overlays()
    return {"status": "ok"}

class GestureName(str, Enum):
    nod = "nod"
    no = "no"
    home = "home"
    home_pose = "home_pose"

@app.post("/make-gesture")
def make_gesture(gesture: GestureName, speed: int = 150):
    if tiny_bot is None:
        raise fastapi.HTTPException(status_code=503, detail="TinyBot not initialized (hardware offline?)")
    try:
        tiny_bot.make_gesture(gesture.value, speed)
        return {"status": "ok", "gesture": gesture}
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))

class MoveAbsoluteRequest(BaseModel):
    angles: list[float]
    speeds: list[int]
    blocking: bool = True

class MoveRelativeRequest(BaseModel):
    relative_angles: list[float]
    speeds: list[int]
    blocking: bool = True

@app.post("/move-absolute")
def move_absolute(req: MoveAbsoluteRequest):
    if tiny_bot is None:
        raise fastapi.HTTPException(status_code=503, detail="TinyBot not initialized (hardware offline?)")
    success = tiny_bot.move_absolute(req.angles, req.speeds, req.blocking)
    if not success:
        raise fastapi.HTTPException(status_code=400, detail="Failed to execute move. Check angles/speeds lengths.")
    return {"status": "ok"}

@app.post("/move-relative")
def move_relative(req: MoveRelativeRequest):
    if tiny_bot is None:
        raise fastapi.HTTPException(status_code=503, detail="TinyBot not initialized (hardware offline?)")
    success = tiny_bot.move_relative(req.relative_angles, req.speeds, req.blocking)
    if not success:
        raise fastapi.HTTPException(status_code=400, detail="Failed to execute move. Check angles/speeds lengths.")
    return {"status": "ok"}

class LookToUVRequest(BaseModel):
    u: int = Field(..., ge=0, le=1000)
    v: int = Field(..., ge=0, le=1000)
    speed: int = 150
    blocking: bool = True

@app.post("/look-to-uv")
def look_to_uv(req: LookToUVRequest):
    if tiny_bot is None:
        raise fastapi.HTTPException(status_code=503, detail="TinyBot not initialized (hardware offline?)")
    
    intrinsics = camera_streamer.get_intrinsics()
    if intrinsics is None:
        raise fastapi.HTTPException(status_code=503, detail="Camera intrinsics not available")
        
    width = intrinsics["width"]
    height = intrinsics["height"]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    ppx = intrinsics["ppx"]
    ppy = intrinsics["ppy"]
    
    x = req.u * width / 1000.0
    y = req.v * height / 1000.0
    
    yaw_rad = math.atan((x - ppx) / fx)
    pitch_rad = math.atan((y - ppy) / fy)
    
    relative_yaw = -math.degrees(yaw_rad)
    relative_pitch = math.degrees(pitch_rad)
    
    logger.info(f"look_to_uv: u,v=({req.u},{req.v}) -> pixel=({x},{y}) -> rel_angles=({relative_yaw:.2f}, {relative_pitch:.2f})")
    
    success = tiny_bot.move_relative([relative_yaw, relative_pitch], [req.speed, req.speed], req.blocking)
    if not success:
        raise fastapi.HTTPException(status_code=500, detail="Failed to execute movement")
        
    return {"status": "ok", "relative_angles": [relative_yaw, relative_pitch]}

