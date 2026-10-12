from __future__ import annotations

import os
import base64
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from apps.api.config import load_dot_config
from apps.api.session import SpotSession
from apps.manipulation.gemini_detector import DEFAULT_MODEL
from apps.manipulation.models import Pose3D
from apps.manipulation.spot_client import DEFAULT_COLOR_SOURCE, DEFAULT_DEPTH_SOURCE
from apps.navigation.visualizer import DEFAULT_MAP_PATH


app = FastAPI(title="Spot Agent API", version="0.1.0")
logger = logging.getLogger(__name__)
session = SpotSession()
TELEOP_HTML = Path(__file__).with_name("static") / "teleop.html"
DETECTION_HTML = Path(__file__).with_name("static") / "detection.html"
CAMERAS_HTML = Path(__file__).with_name("static") / "cameras.html"


class ConnectRequest(BaseModel):
    hostname: str = Field(default="192.168.80.3")
    username: str = Field(default="user")
    password: str
    take_lease: bool = False


class NavigateRequest(BaseModel):
    name: str
    command_duration: float = 30.0
    timeout: float = 180.0
    feedback_interval: float = 1.0
    power_on: bool = True
    stand: bool = True
    take_lease: bool = False


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


class LoadMapRequest(BaseModel):
    path: str
    replace_graph: bool = True
    generate_new_anchoring: bool = False
    take_lease: bool = False


class StandRequest(BaseModel):
    power_on: bool = True
    take_lease: bool = False
    timeout: float = 10.0


class SitRequest(BaseModel):
    take_lease: bool = False
    timeout: float = 10.0


class TeleopVelocityRequest(BaseModel):
    v_x: float = 0.0
    v_y: float = 0.0
    v_rot: float = 0.0
    duration: float = 0.6
    take_lease: bool = False
    power_on: bool = False
    stand: bool = False
    body_follow_arm: bool = False


class TeleopStopRequest(BaseModel):
    take_lease: bool = False


class StopActionsRequest(BaseModel):
    take_lease: bool = False
    freeze_arm: bool = True


class ArmFreezeRequest(BaseModel):
    take_lease: bool = False


class ArmOscillationMonitorRequest(BaseModel):
    enabled: bool
    take_lease: bool = True
    sample_interval: float = 0.1
    window_sec: float = 1.2
    min_peak_to_peak_m: float = 0.012
    min_direction_changes: int = 4
    min_speed_mps: float = 0.025
    freeze_cooldown_sec: float = 2.0


class SyncWaypointsRequest(BaseModel):
    overwrite: bool = False
    prune_stale: bool = False


class VisualizeRequest(BaseModel):
    include_point_clouds: bool = True


class LeaseRequest(BaseModel):
    take: bool = False


class ArmRequest(BaseModel):
    take_lease: bool = False
    timeout: float = 10.0


class DeployArmRequest(ArmRequest):
    power_on: bool = True


class ArmJogRequest(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    seconds: float = 0.8
    take_lease: bool = False
    timeout: float = 3.0


class ArmCameraRollRequest(BaseModel):
    direction: str
    angle_rad: float = 0.105
    seconds: float = 0.7
    take_lease: bool = True
    timeout: float = 3.0


class WaitForPickUpRequest(BaseModel):
    monitor_sec: float = 30.0
    upward_threshold_m: float = 0.02
    sample_interval: float = 0.1
    open_duration_sec: float = 3.0
    take_lease: bool = True
    gripper_timeout: float = 5.0
    stow_timeout: float = 10.0


class PoseRequest(BaseModel):
    frame_name: str = "vision"
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


class ArmApproachRequest(BaseModel):
    pose: PoseRequest
    standoff_m: float = 0.0
    seconds: float = 1.2
    take_lease: bool = False
    timeout: float = 5.0
    max_step_m: float = 0.45


class ArmReachDistanceRequest(BaseModel):
    pose: PoseRequest


class OpenGripperRequest(BaseModel):
    take_lease: bool = False
    timeout: float = 5.0
    open_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    max_vel: float | None = Field(default=None, gt=0.0)
    max_acc: float | None = Field(default=None, gt=0.0)


class DetectRequest(BaseModel):
    instruction: str
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    color_source: str = DEFAULT_COLOR_SOURCE
    depth_source: str = DEFAULT_DEPTH_SOURCE
    include_point_cloud: bool = True
    point_cloud_stride: int = Field(default=4, ge=1, le=32)
    max_point_cloud_points: int = Field(default=6000, ge=0, le=50000)


class DetectPickTargetRequest(BaseModel):
    instruction: str
    model: str = DEFAULT_MODEL
    api_key: str | None = None


class PickRequest(BaseModel):
    instruction: str
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    take_lease: bool = False
    timeout: float = 30.0
    grip_max_torque_nm: float = Field(default=2.0, ge=0.5, le=5.5)


class ForceChangeRequest(BaseModel):
    threshold_newtons: float = 5.0
    sample_window_sec: float = 3.0
    interval_sec: float = 0.1


class GraspPixelRequest(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    take_lease: bool = True
    max_depth_m: float = Field(default=2.0, ge=0.2, le=3.0)
    grip_max_torque_nm: float = Field(default=2.0, ge=0.5, le=5.5)


class Scan360Request(BaseModel):
    duration: float = 8.0
    camera_source: str = "hand_color_image"


class PlacePixelRequest(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    take_lease: bool = True
    standoff_m: float = Field(default=0.05, ge=0.03, le=0.15)
    settle_time_sec: float = Field(default=0.5, ge=0.0, le=3.0)


class StowSmartRequest(BaseModel):
    take_lease: bool = True
    timeout: float = 20.0


@app.on_event("startup")
def startup_connect() -> None:
    load_dot_config()
    hostname = os.getenv("SPOT_HOSTNAME") or os.getenv("BOSDYN_CLIENT_HOSTNAME")
    username = os.getenv("BOSDYN_CLIENT_USERNAME")
    password = os.getenv("BOSDYN_CLIENT_PASSWORD")
    if hostname and username and password:
        try:
            session.connect(
                hostname=hostname,
                username=username,
                password=password,
                take_lease=os.getenv("SPOT_TAKE_LEASE", "").lower() in {"1", "true", "yes"},
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Spot startup connection failed; API will remain available: %s: %s",
                type(exc).__name__,
                exc,
            )


@app.on_event("shutdown")
def shutdown() -> None:
    session.stop_arm_oscillation_monitor()
    session.shutdown_lease()


def _run(action):
    try:
        return action()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, **session.status()}


@app.post("/connect")
def connect(request: ConnectRequest) -> dict[str, Any]:
    return _run(lambda: session.connect(
        hostname=request.hostname,
        username=request.username,
        password=request.password,
        take_lease=request.take_lease,
    ))


@app.get("/lease")
def lease_status() -> dict[str, Any]:
    return session.status()


@app.get("/localization")
def localization_status() -> dict[str, Any]:
    return _run(session.localization_status)


@app.post("/localize")
def localize(request: LocalizeRequest) -> dict[str, Any]:
    return _run(lambda: session.localize(
        waypoint_id=request.waypoint_id,
        waypoint_name=request.waypoint_name,
        fiducial_init=request.fiducial_init,
        use_fiducial_id=request.use_fiducial_id,
        refine_fiducial_result_with_icp=request.refine_fiducial_result_with_icp,
        do_ambiguity_check=request.do_ambiguity_check,
        refine_with_visual_features=request.refine_with_visual_features,
        verify_visual_features_quality=request.verify_visual_features_quality,
        max_distance=request.max_distance,
        max_yaw=request.max_yaw,
    ))


@app.post("/map/load")
def load_map(request: LoadMapRequest) -> dict[str, Any]:
    return _run(lambda: session.load_map(
        request.path,
        replace_graph=request.replace_graph,
        generate_new_anchoring=request.generate_new_anchoring,
        take_lease=request.take_lease,
    ))


@app.get("/lease/details")
def lease_details() -> dict[str, str]:
    return _run(lambda: {"leases": session.list_leases()})


@app.get("/battery")
def battery() -> dict[str, Any]:
    return _run(session.battery_status)


@app.post("/faults/behavior/clear")
def clear_behavior_faults() -> dict[str, Any]:
    return _run(session.clear_behavior_faults)


@app.post("/lease/acquire")
def acquire_lease() -> dict[str, Any]:
    return _run(session.acquire_lease)


@app.post("/lease/take")
def take_lease() -> dict[str, Any]:
    return _run(session.take_lease)


@app.post("/lease/release")
def release_lease() -> dict[str, Any]:
    session.shutdown_lease()
    return session.status()


@app.get("/waypoints")
def waypoints() -> list[dict[str, Any]]:
    return session.list_waypoints()


@app.post("/waypoints/sync")
def sync_waypoints(request: SyncWaypointsRequest) -> dict[str, Any]:
    return _run(lambda: session.sync_waypoints(
        overwrite=request.overwrite,
        prune_stale=request.prune_stale,
    ))


@app.post("/navigate")
def navigate(request: NavigateRequest) -> dict[str, Any]:
    return _run(lambda: session.navigate(
        request.name,
        command_duration=request.command_duration,
        timeout=request.timeout,
        feedback_interval=request.feedback_interval,
        power_on=request.power_on,
        stand=request.stand,
        take_lease=request.take_lease,
    ))


@app.post("/stand")
def stand(request: StandRequest) -> dict[str, Any]:
    return _run(lambda: session.stand(
        power_on=request.power_on,
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))


@app.post("/sit")
def sit(request: SitRequest) -> dict[str, Any]:
    return _run(lambda: session.sit(
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))


@app.get("/teleop")
def teleop_page():
    return FileResponse(TELEOP_HTML)


@app.get("/detection")
def detection_page():
    return FileResponse(DETECTION_HTML)


@app.get("/cameras")
def cameras_page():
    return FileResponse(CAMERAS_HTML)


@app.post("/teleop/velocity")
def teleop_velocity(request: TeleopVelocityRequest) -> dict[str, Any]:
    return _run(lambda: session.velocity(
        v_x=request.v_x,
        v_y=request.v_y,
        v_rot=request.v_rot,
        duration=request.duration,
        take_lease=request.take_lease,
        power_on=request.power_on,
        stand=request.stand,
        body_follow_arm=request.body_follow_arm,
    ))


@app.post("/teleop/stop")
def teleop_stop(request: TeleopStopRequest) -> dict[str, Any]:
    return _run(lambda: session.stop(take_lease=request.take_lease))


@app.post("/actions/stop")
def stop_actions(request: StopActionsRequest) -> dict[str, Any]:
    return _run(lambda: session.stop_all_actions(
        take_lease=request.take_lease,
        freeze_arm=request.freeze_arm,
    ))


@app.post("/visualize")
def visualize(request: VisualizeRequest) -> dict[str, Any]:
    return _run(lambda: session.visualize(include_point_clouds=request.include_point_clouds))


@app.get("/visualize/map")
def visualization_map():
    if not DEFAULT_MAP_PATH.exists():
        raise HTTPException(status_code=404, detail="No visualization has been generated yet.")
    return FileResponse(DEFAULT_MAP_PATH)


@app.get("/images/sources")
def image_sources() -> list[dict[str, Any]]:
    return _run(session.image_sources)


def _camera_image_payload(source: str, quality_percent: int) -> tuple[bytes, str]:
    image_response = _run(lambda: session.capture_image(source, quality_percent=quality_percent))
    image = image_response.shot.image
    media_type = "application/octet-stream"
    if image.format == image.FORMAT_JPEG:
        media_type = "image/jpeg"
    return image.data, media_type


@app.get("/images/{source}")
def camera_image(source: str, quality_percent: int = 85):
    data, media_type = _camera_image_payload(source, quality_percent)
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/camera_image/")
def eden_camera_image(camera_id: str, quality_percent: int = 85):
    """Return the HTML data-URI snapshot expected by Eden RobotClient."""
    data, media_type = _camera_image_payload(camera_id, quality_percent)
    encoded = base64.b64encode(data).decode("ascii")
    html = f'<img src="data:{media_type};base64,{encoded}" alt="camera">'
    return Response(
        content=html,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/arm/deploy")
def deploy_arm(request: DeployArmRequest) -> dict[str, Any]:
    return _run(lambda: session.deploy_arm(
        power_on=request.power_on,
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))


@app.post("/arm/carry")
def carry_arm(request: ArmRequest) -> dict[str, Any]:
    return _run(lambda: session.carry_arm(take_lease=request.take_lease, timeout=request.timeout))


@app.post("/arm/freeze")
def freeze_arm(request: ArmFreezeRequest) -> dict[str, Any]:
    return _run(lambda: session.freeze_arm(take_lease=request.take_lease))


@app.get("/arm/oscillation-monitor")
def arm_oscillation_monitor_status() -> dict[str, Any]:
    return session.arm_oscillation_monitor_status()


@app.post("/arm/oscillation-monitor")
def configure_arm_oscillation_monitor(request: ArmOscillationMonitorRequest) -> dict[str, Any]:
    return _run(lambda: session.configure_arm_oscillation_monitor(
        enabled=request.enabled,
        take_lease=request.take_lease,
        sample_interval=request.sample_interval,
        window_sec=request.window_sec,
        min_peak_to_peak_m=request.min_peak_to_peak_m,
        min_direction_changes=request.min_direction_changes,
        min_speed_mps=request.min_speed_mps,
        freeze_cooldown_sec=request.freeze_cooldown_sec,
    ))


@app.post("/arm/jog")
def jog_arm(request: ArmJogRequest) -> dict[str, Any]:
    return _run(lambda: session.jog_arm(
        dx=request.dx,
        dy=request.dy,
        dz=request.dz,
        droll=request.droll,
        dpitch=request.dpitch,
        dyaw=request.dyaw,
        seconds=request.seconds,
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))


@app.post("/arm/camera-roll")
def camera_roll_arm(request: ArmCameraRollRequest) -> dict[str, Any]:
    return _run(lambda: session.roll_camera_view(
        direction=request.direction,
        angle_rad=request.angle_rad,
        seconds=request.seconds,
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))


@app.post("/arm/reach-distance")
def reach_distance(request: ArmReachDistanceRequest) -> dict[str, Any]:
    return _run(lambda: session.reach_distance_to_pose(Pose3D(
        frame_name=request.pose.frame_name,
        x=request.pose.x,
        y=request.pose.y,
        z=request.pose.z,
        qw=request.pose.qw,
        qx=request.pose.qx,
        qy=request.pose.qy,
        qz=request.pose.qz,
    )))


@app.post("/arm/approach")
def approach_arm(request: ArmApproachRequest) -> dict[str, Any]:
    return _run(lambda: session.approach_pose(
        Pose3D(
            frame_name=request.pose.frame_name,
            x=request.pose.x,
            y=request.pose.y,
            z=request.pose.z,
            qw=request.pose.qw,
            qx=request.pose.qx,
            qy=request.pose.qy,
            qz=request.pose.qz,
        ),
        standoff_m=request.standoff_m,
        seconds=request.seconds,
        take_lease=request.take_lease,
        timeout=request.timeout,
        max_step_m=request.max_step_m,
    ))


@app.post("/arm/approach-whole-body")
def approach_arm_whole_body(request: ArmApproachRequest) -> dict[str, Any]:
    return _run(lambda: session.approach_pose_whole_body(
        Pose3D(
            frame_name=request.pose.frame_name,
            x=request.pose.x,
            y=request.pose.y,
            z=request.pose.z,
            qw=request.pose.qw,
            qx=request.pose.qx,
            qy=request.pose.qy,
            qz=request.pose.qz,
        ),
        standoff_m=request.standoff_m,
        seconds=request.seconds,
        take_lease=request.take_lease,
        timeout=request.timeout,
        max_step_m=request.max_step_m,
    ))


@app.post("/arm/stow")
def stow_arm(request: ArmRequest) -> dict[str, Any]:
    return _run(lambda: session.stow_arm(take_lease=request.take_lease, timeout=request.timeout))


@app.post("/pickup/wait")
def wait_for_pick_up(request: WaitForPickUpRequest) -> dict[str, Any]:
    return _run(lambda: session.wait_for_pick_up(
        monitor_sec=request.monitor_sec,
        upward_threshold_m=request.upward_threshold_m,
        sample_interval=request.sample_interval,
        open_duration_sec=request.open_duration_sec,
        take_lease=request.take_lease,
        gripper_timeout=request.gripper_timeout,
        stow_timeout=request.stow_timeout,
    ))


@app.post("/delivery/wait", include_in_schema=False)
def wait_for_delivery_compat(request: WaitForPickUpRequest) -> dict[str, Any]:
    """Compatibility alias for clients using the former route."""
    return wait_for_pick_up(request)


@app.post("/gripper/open")
def open_gripper(request: OpenGripperRequest) -> dict[str, Any]:
    return _run(lambda: session.open_gripper(
        take_lease=request.take_lease,
        timeout=request.timeout,
        open_fraction=request.open_fraction,
        max_vel=request.max_vel,
        max_acc=request.max_acc,
    ))


@app.post("/gripper/close")
def close_gripper(request: OpenGripperRequest) -> dict[str, Any]:
    return _run(lambda: session.close_gripper(
        take_lease=request.take_lease,
        timeout=request.timeout,
        max_vel=request.max_vel,
        max_acc=request.max_acc,
    ))


@app.post("/detect")
def detect(request: DetectRequest) -> dict[str, Any]:
    def action():
        scene = session.detect_scene(
            request.instruction,
            model=request.model,
            api_key=request.api_key,
            color_source=request.color_source,
            depth_source=request.depth_source,
            point_cloud_stride=request.point_cloud_stride,
            max_point_cloud_points=request.max_point_cloud_points if request.include_point_cloud else 0,
        )
        image = scene["image"]
        image_bytes = image.pop("data")
        image["data_url"] = (
            f"data:{image['mime_type']};base64,"
            f"{base64.b64encode(image_bytes).decode('ascii')}"
        )
        if not request.include_point_cloud:
            scene["point_cloud"]["points"] = []
        return scene

    return _run(action)


@app.post("/detect/pick-target")
def detect_pick_target(request: DetectPickTargetRequest) -> dict[str, Any]:
    return _run(lambda: session.detect_pick_target(
        request.instruction,
        model=request.model,
        api_key=request.api_key,
    ))


@app.post("/pick")
def pick(request: PickRequest) -> dict[str, Any]:
    return _run(lambda: session.pick(
        request.instruction,
        model=request.model,
        api_key=request.api_key,
        take_lease=request.take_lease,
        timeout=request.timeout,
        grip_max_torque_nm=request.grip_max_torque_nm,
    ))


@app.post("/force/change")
def force_change(request: ForceChangeRequest) -> dict[str, Any]:
    result = _run(lambda: session.detect_external_force_change(
        threshold_newtons=request.threshold_newtons,
        sample_window_sec=request.sample_window_sec,
        interval_sec=request.interval_sec,
    ))
    return asdict(result)


@app.post("/manipulation/grasp-pixel")
def grasp_pixel(request: GraspPixelRequest) -> dict[str, Any]:
    return _run(lambda: session.grasp_at_pixel(
        x_pct=request.x / 1000.0,
        y_pct=request.y / 1000.0,
        take_lease=request.take_lease,
        max_depth_m=request.max_depth_m,
        grip_max_torque_nm=request.grip_max_torque_nm,
    ))


@app.post("/teleop/scan-360")
def scan_360(request: Scan360Request) -> dict[str, Any]:
    return _run(lambda: session.execute_360_scan(
        duration=request.duration,
        camera_source=request.camera_source,
    ))


@app.post("/manipulation/place-pixel")
def place_pixel(request: PlacePixelRequest) -> dict[str, Any]:
    return _run(lambda: session.place_at_pixel(
        x_pct=request.x / 1000.0,
        y_pct=request.y / 1000.0,
        take_lease=request.take_lease,
        standoff_m=request.standoff_m,
        settle_time_sec=request.settle_time_sec,
    ))


@app.post("/arm/stow-smart")
def stow_smart(request: StowSmartRequest) -> dict[str, Any]:
    return _run(lambda: session.stow_smart(
        take_lease=request.take_lease,
        timeout=request.timeout,
    ))
