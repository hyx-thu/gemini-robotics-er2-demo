from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bosdyn.client
import numpy as np
from bosdyn.api import geometry_pb2, gripper_command_pb2, image_pb2, manipulation_api_pb2, robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient, build_image_request, pixel_to_camera_space
from bosdyn.client.lease import (
    LeaseClient,
    LeaseKeepAlive,
    LeaseNotOwnedByWallet,
    LeaseResponseError,
)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import Quat
from bosdyn.client.power import PowerClient, power_on_motors
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_sit,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient

from apps.manipulation.gemini_detector import GeminiObjectDetector
from apps.manipulation.models import Detection2D, ForceChange, ImageObservation, Pose3D
from apps.manipulation.spot_client import (
    DEFAULT_COLOR_SOURCE,
    DEFAULT_DEPTH_SOURCE,
    _depth_at_pixel,
    constrain_grasp_to_top_down,
)
from apps.navigation.registry import KeypointRegistry
from apps.navigation.visualizer import DEFAULT_MAP_PATH, write_graph_html


MAX_LINEAR_SPEED_MPS = 0.8
MAX_SIDEWAYS_SPEED_MPS = 0.5
MAX_ANGULAR_SPEED_RADPS = 1.0
SUCCESS_STATUS = graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL
ACTIVE_STATUSES = {
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_UNKNOWN,
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_FOLLOWING_ROUTE,
}
TERMINAL_MANIP_STATES = {
    manipulation_api_pb2.MANIP_STATE_DONE,
    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
    manipulation_api_pb2.MANIP_STATE_PLACE_SUCCEEDED,
    manipulation_api_pb2.MANIP_STATE_PLACE_FAILED,
}


class SpotSession:
    """Persistent Spot connection and lease holder used by the HTTP API."""

    def __init__(self, registry: KeypointRegistry | None = None):
        self.registry = registry or KeypointRegistry()
        self.lock = threading.RLock()
        self.hostname: str | None = None
        self.robot = None
        self.graph_nav = None
        self.image = None
        self.lease = None
        self.lease_keepalive: LeaseKeepAlive | None = None
        self.manipulation = None
        self.power = None
        self.command = None
        self.state = None
        self.cancel_actions_event = threading.Event()
        self.arm_oscillation_monitor_stop = threading.Event()
        self.arm_oscillation_monitor_thread: threading.Thread | None = None
        self.arm_oscillation_monitor_config: dict[str, Any] = {}
        self.arm_oscillation_monitor_state: dict[str, Any] = {
            "enabled": False,
            "samples": 0,
            "last_detection": None,
            "last_freeze": None,
            "last_error": None,
        }

    @property
    def connected(self) -> bool:
        return self.robot is not None

    @property
    def holding_lease(self) -> bool:
        return self.lease_keepalive is not None and self.lease_keepalive.is_alive()

    def connect(self, hostname: str, username: str, password: str, *, take_lease: bool = False) -> dict[str, Any]:
        self.stop_arm_oscillation_monitor()
        with self.lock:
            self.shutdown_lease()
            sdk = bosdyn.client.create_standard_sdk("spot-fastapi-server")
            robot = sdk.create_robot(hostname)
            robot.authenticate(username, password)
            robot.time_sync.wait_for_sync()

            self.hostname = hostname
            self.robot = robot
            self.graph_nav = robot.ensure_client(GraphNavClient.default_service_name)
            self.image = robot.ensure_client(ImageClient.default_service_name)
            self.lease = robot.ensure_client(LeaseClient.default_service_name)
            self.manipulation = robot.ensure_client(ManipulationApiClient.default_service_name)
            self.power = robot.ensure_client(PowerClient.default_service_name)
            self.command = robot.ensure_client(RobotCommandClient.default_service_name)
            self.state = robot.ensure_client(RobotStateClient.default_service_name)

            if take_lease:
                self.take_lease()
            return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "hostname": self.hostname,
            "holding_lease": self.holding_lease,
            "arm_oscillation_monitor": self.arm_oscillation_monitor_status(),
        }

    def battery_status(self) -> dict[str, Any]:
        self.require_connected()
        robot_state = self.state.get_robot_state()
        power_state = robot_state.power_state
        if hasattr(power_state, "locomotion_charge_percentage"):
            runtime = None
            if power_state.HasField("locomotion_estimated_runtime"):
                runtime = (
                    power_state.locomotion_estimated_runtime.seconds
                    + power_state.locomotion_estimated_runtime.nanos / 1e9
                )
            charge = (
                power_state.locomotion_charge_percentage.value
                if power_state.HasField("locomotion_charge_percentage")
                else None
            )
            return {
                "charge_percentage": charge,
                "estimated_runtime_sec": runtime,
                "motor_power_state": robot_state_pb2.PowerState.MotorPowerState.Name(
                    power_state.motor_power_state
                ),
                "shore_power_state": robot_state_pb2.PowerState.ShorePowerState.Name(
                    power_state.shore_power_state
                ),
                "robot_power_state": robot_state_pb2.PowerState.RobotPowerState.Name(
                    power_state.robot_power_state
                ),
            }

        batteries = []
        for battery in power_state.battery_states:
            runtime = None
            if battery.HasField("estimated_runtime"):
                runtime = battery.estimated_runtime.seconds + battery.estimated_runtime.nanos / 1e9
            timestamp = None
            if battery.HasField("timestamp"):
                timestamp = battery.timestamp.seconds + battery.timestamp.nanos / 1e9
            batteries.append({
                "identifier": battery.identifier,
                "charge_percentage": battery.charge_percentage.value,
                "estimated_runtime_sec": runtime,
                "current": battery.current.value if battery.HasField("current") else None,
                "voltage": battery.voltage.value if battery.HasField("voltage") else None,
                "temperatures": list(battery.temperatures),
                "communications_loss_percent": battery.communications_loss_percent.value,
                "status": robot_state_pb2.BatteryState.Status.Name(battery.status),
                "timestamp": timestamp,
            })
        charge_values = [
            item["charge_percentage"]
            for item in batteries
            if item["charge_percentage"] is not None
        ]
        return {
            "batteries": batteries,
            "charge_percentage": min(charge_values) if charge_values else None,
        }

    def clear_behavior_faults(self) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            robot_state = self.state.get_robot_state()
            faults = list(robot_state.behavior_fault_state.faults)
            cleared: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for fault in faults:
                fault_id = fault.behavior_fault_id
                fault_info = {
                    "behavior_fault_id": fault_id,
                    "cause": robot_state_pb2.BehaviorFault.Cause.Name(fault.cause),
                    "status": robot_state_pb2.BehaviorFault.Status.Name(fault.status),
                }
                try:
                    is_cleared = self.command.clear_behavior_fault(fault_id)
                    cleared.append({**fault_info, "cleared": bool(is_cleared)})
                except Exception as exc:
                    errors.append({**fault_info, "error": f"{type(exc).__name__}: {exc}"})

            remaining_state = self.state.get_robot_state()
            remaining = [
                {
                    "behavior_fault_id": fault.behavior_fault_id,
                    "cause": robot_state_pb2.BehaviorFault.Cause.Name(fault.cause),
                    "status": robot_state_pb2.BehaviorFault.Status.Name(fault.status),
                }
                for fault in remaining_state.behavior_fault_state.faults
            ]
            return {
                "attempted": len(faults),
                "cleared": cleared,
                "errors": errors,
                "remaining": remaining,
            }

    def require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Spot is not connected. Call POST /connect first or set startup env vars.")

    def take_lease(self) -> dict[str, Any]:
        self.require_connected()
        self.shutdown_lease()
        self.lease.take()
        self.lease_keepalive = LeaseKeepAlive(
            self.lease,
            must_acquire=False,
            return_at_exit=False,
        )
        return self.status()

    def acquire_lease(self) -> dict[str, Any]:
        self.require_connected()
        self.shutdown_lease()
        self.lease_keepalive = LeaseKeepAlive(
            self.lease,
            must_acquire=True,
            return_at_exit=False,
        )
        return self.status()

    def shutdown_lease(self) -> None:
        if self.lease_keepalive is not None:
            self.lease_keepalive.shutdown()
            self.lease_keepalive = None

    def list_leases(self) -> str:
        self.require_connected()
        return str(self.lease.list_leases(include_full_lease_info=True))

    def ensure_lease(self, *, take_if_needed: bool = False) -> None:
        self.require_connected()
        if self.holding_lease:
            try:
                lease = self.lease.lease_wallet.get_lease()
                self.lease.retain_lease(lease)
                return
            except (LeaseNotOwnedByWallet, LeaseResponseError):
                self.shutdown_lease()
        if take_if_needed:
            self.take_lease()
        else:
            self.acquire_lease()

    def sync_waypoints(
        self,
        *,
        overwrite: bool = False,
        prune_stale: bool = False,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            graph = self.graph_nav.download_graph()
            named_waypoints = {
                waypoint.annotations.name.strip(): waypoint.id
                for waypoint in graph.waypoints
                if waypoint.annotations.name.strip()
            }
            synced = self.registry.sync_from_waypoints(
                named_waypoints, overwrite=overwrite
            )
            removed = (
                self.registry.prune_stale_waypoint_ids(
                    {waypoint.id for waypoint in graph.waypoints}
                )
                if prune_stale
                else []
            )
            return {
                "count": len(synced),
                "removed_count": len(removed),
                "synced": [asdict(keypoint) for keypoint in synced],
                "removed": [asdict(keypoint) for keypoint in removed],
            }

    def list_waypoints(self) -> list[dict[str, Any]]:
        return [asdict(keypoint) for keypoint in self.registry.all()]

    def named_waypoints(self) -> dict[str, str]:
        graph = self.graph_nav.download_graph()
        waypoints: dict[str, str] = {}
        for waypoint in graph.waypoints:
            name = waypoint.annotations.name.strip()
            if name:
                waypoints[name] = waypoint.id
        return waypoints

    def load_map(
        self,
        path: str | Path,
        *,
        replace_graph: bool = True,
        generate_new_anchoring: bool = False,
        take_lease: bool = False,
    ) -> dict[str, Any]:
        self.require_connected()
        map_dir = Path(path).expanduser().resolve()
        graph_path = map_dir / "graph"
        waypoint_dir = map_dir / "waypoint_snapshots"
        edge_dir = map_dir / "edge_snapshots"
        if not graph_path.is_file():
            raise ValueError(f"Map directory must contain a graph file: {graph_path}")

        graph = map_pb2.Graph()
        graph.ParseFromString(graph_path.read_bytes())

        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            response = self.graph_nav.upload_graph(
                graph=graph,
                replace_graph=replace_graph,
                generate_new_anchoring=generate_new_anchoring,
            )

            uploaded_waypoint_snapshots: list[str] = []
            missing_waypoint_snapshots: list[str] = []
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                snapshot_path = waypoint_dir / snapshot_id
                if not snapshot_path.is_file():
                    missing_waypoint_snapshots.append(snapshot_id)
                    continue
                snapshot = map_pb2.WaypointSnapshot()
                snapshot.ParseFromString(snapshot_path.read_bytes())
                self.graph_nav.upload_waypoint_snapshot(snapshot)
                uploaded_waypoint_snapshots.append(snapshot_id)

            uploaded_edge_snapshots: list[str] = []
            missing_edge_snapshots: list[str] = []
            for snapshot_id in response.unknown_edge_snapshot_ids:
                snapshot_path = edge_dir / snapshot_id
                if not snapshot_path.is_file():
                    missing_edge_snapshots.append(snapshot_id)
                    continue
                snapshot = map_pb2.EdgeSnapshot()
                snapshot.ParseFromString(snapshot_path.read_bytes())
                self.graph_nav.upload_edge_snapshot(snapshot)
                uploaded_edge_snapshots.append(snapshot_id)

        if missing_waypoint_snapshots or missing_edge_snapshots:
            raise ValueError(
                "Map graph loaded but some required snapshots were missing: "
                f"waypoint={missing_waypoint_snapshots}, edge={missing_edge_snapshots}"
            )

        synced = self.registry.sync_from_waypoints(self.named_waypoints(), overwrite=False)
        return {
            "path": str(map_dir),
            "status": graph_nav_pb2.UploadGraphResponse.Status.Name(response.status),
            "replaced_graph": response.replaced_graph,
            "waypoints": len(graph.waypoints),
            "edges": len(graph.edges),
            "uploaded_waypoint_snapshots": uploaded_waypoint_snapshots,
            "uploaded_edge_snapshots": uploaded_edge_snapshots,
            "loaded_waypoint_snapshot_ids": list(response.loaded_waypoint_snapshot_ids),
            "loaded_edge_snapshot_ids": list(response.loaded_edge_snapshot_ids),
            "synced_keypoints": [asdict(item) for item in synced],
        }

    def localization_status(self) -> dict[str, Any]:
        self.require_connected()
        response = self.graph_nav.get_localization_state()
        return {
            "localization": _localization_to_dict(response.localization),
            "localized": bool(response.localization.waypoint_id),
        }

    def localize(
        self,
        *,
        waypoint_id: str | None = None,
        waypoint_name: str | None = None,
        fiducial_init: str = "nearest",
        use_fiducial_id: int | None = None,
        refine_fiducial_result_with_icp: bool = True,
        do_ambiguity_check: bool = False,
        refine_with_visual_features: bool = False,
        verify_visual_features_quality: bool = False,
        max_distance: float | None = None,
        max_yaw: float | None = None,
    ) -> dict[str, Any]:
        self.require_connected()
        self._require_graph_loaded()
        if waypoint_name:
            waypoint_id = self.registry.get(waypoint_name).waypoint_id

        initial_guess = nav_pb2.Localization()
        if waypoint_id:
            initial_guess.waypoint_id = waypoint_id
            initial_guess.waypoint_tform_body.rotation.w = 1.0

        fiducial_init_value = _fiducial_init_value(fiducial_init)
        response = self.graph_nav.set_localization_full_response(
            initial_guess,
            max_distance=max_distance,
            max_yaw=max_yaw,
            fiducial_init=fiducial_init_value,
            use_fiducial_id=use_fiducial_id,
            refine_fiducial_result_with_icp=refine_fiducial_result_with_icp,
            do_ambiguity_check=do_ambiguity_check,
            refine_with_visual_features=refine_with_visual_features,
            verify_visual_features_quality=verify_visual_features_quality,
        )
        return {
            "status": graph_nav_pb2.SetLocalizationResponse.Status.Name(response.status),
            "localized": response.status == graph_nav_pb2.SetLocalizationResponse.STATUS_OK,
            "localization": _localization_to_dict(response.localization),
            "waypoint_id": response.localization.waypoint_id,
            "error_report": response.error_report,
            "quality_check_result": graph_nav_pb2.SetLocalizationResponse.QualityCheckResult.Name(
                response.quality_check_result
            ),
        }

    def navigate(
        self,
        name: str,
        *,
        command_duration: float,
        timeout: float,
        feedback_interval: float,
        power_on: bool,
        stand: bool,
        take_lease: bool,
    ) -> dict[str, Any]:
        self.require_connected()
        self.cancel_actions_event.clear()
        keypoint = self.registry.get(name)
        with self.lock:
            self._require_navigation_ready()
            self.ensure_lease(take_if_needed=take_lease)
            if power_on:
                power_on_motors(self.power)
            if stand:
                blocking_stand(self.command, timeout_sec=10)

            command_id = self.graph_nav.navigate_to(keypoint.waypoint_id, command_duration)
            deadline = time.monotonic() + timeout
            next_refresh = time.monotonic() + max(command_duration / 2, feedback_interval)
            last_status = graph_nav_pb2.NavigationFeedbackResponse.STATUS_UNKNOWN

            while time.monotonic() < deadline:
                if self.cancel_actions_event.is_set():
                    command_id = self.command.robot_command(RobotCommandBuilder.stop_command())
                    return {
                        "name": name,
                        "waypoint_id": keypoint.waypoint_id,
                        "command_id": command_id,
                        "status": "CANCELLED_BY_STOP_ENDPOINT",
                        "reached_goal": False,
                    }
                feedback = self.graph_nav.navigation_feedback(command_id)
                last_status = feedback.status
                if last_status == SUCCESS_STATUS or last_status not in ACTIVE_STATUSES:
                    return {
                        "name": name,
                        "waypoint_id": keypoint.waypoint_id,
                        "command_id": command_id,
                        "status": graph_nav_pb2.NavigationFeedbackResponse.Status.Name(last_status),
                        "reached_goal": last_status == SUCCESS_STATUS,
                    }
                if time.monotonic() >= next_refresh:
                    command_id = self.graph_nav.navigate_to(
                        keypoint.waypoint_id,
                        command_duration,
                        command_id=command_id,
                    )
                    next_refresh = time.monotonic() + max(command_duration / 2, feedback_interval)
                time.sleep(feedback_interval)

            return {
                "name": name,
                "waypoint_id": keypoint.waypoint_id,
                "command_id": command_id,
                "status": f"TIMED_OUT_WAITING_FOR_{graph_nav_pb2.NavigationFeedbackResponse.Status.Name(last_status)}",
                "reached_goal": False,
            }

    def _require_graph_loaded(self) -> None:
        graph = self.graph_nav.download_graph()
        if not graph.waypoints:
            raise RuntimeError(
                "No GraphNav map is loaded. Load a map with POST /map/load or"
                " select and load the map from the Spot tablet, then localize."
            )

    def _require_navigation_ready(self) -> None:
        self._require_graph_loaded()
        localization = self.graph_nav.get_localization_state().localization
        if not localization.waypoint_id:
            raise RuntimeError(
                "GraphNav is not localized. Call POST /localize near a visible"
                " fiducial or provide the robot's current waypoint before"
                " navigating."
            )

    def stand(self, *, power_on: bool = True, take_lease: bool = False, timeout: float = 10.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            if power_on:
                power_on_motors(self.power)
            blocking_stand(self.command, timeout_sec=timeout)
            return {"standing": True}

    def sit(self, *, take_lease: bool = False, timeout: float = 10.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            blocking_sit(self.command, timeout_sec=timeout)
            return {"sitting": True}

    def velocity(
        self,
        *,
        v_x: float,
        v_y: float,
        v_rot: float,
        duration: float = 0.6,
        take_lease: bool = False,
        power_on: bool = False,
        stand: bool = False,
        body_follow_arm: bool = False,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            if power_on:
                power_on_motors(self.power)
            if stand:
                blocking_stand(self.command, timeout_sec=10)

            safe_duration = max(0.1, min(float(duration), 2.0))
            safe_v_x = _clamp(float(v_x), -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
            safe_v_y = _clamp(float(v_y), -MAX_SIDEWAYS_SPEED_MPS, MAX_SIDEWAYS_SPEED_MPS)
            safe_v_rot = _clamp(float(v_rot), -MAX_ANGULAR_SPEED_RADPS, MAX_ANGULAR_SPEED_RADPS)
            mobility_command = RobotCommandBuilder.synchro_velocity_command(
                safe_v_x, safe_v_y, safe_v_rot
            )
            command = (
                RobotCommandBuilder.build_synchro_command(
                    mobility_command,
                    RobotCommandBuilder.arm_joint_freeze_command(),
                )
                if body_follow_arm
                else mobility_command
            )
            command_id = self.command.robot_command(
                command,
                end_time_secs=time.time() + safe_duration,
                timesync_endpoint=self.robot.time_sync.endpoint,
            )
            return {
                "command_id": command_id,
                "v_x": safe_v_x,
                "v_y": safe_v_y,
                "v_rot": safe_v_rot,
                "duration": safe_duration,
                "body_follow_arm": bool(body_follow_arm),
            }

    def stop(self, *, take_lease: bool = False) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            command_id = self.command.robot_command(RobotCommandBuilder.stop_command())
            return {"command_id": command_id, "stopped": True}

    def stop_all_actions(self, *, take_lease: bool = False, freeze_arm: bool = True) -> dict[str, Any]:
        self.require_connected()
        self.cancel_actions_event.set()
        command_results: list[dict[str, Any]] = []
        errors: list[str] = []

        if take_lease and not self.holding_lease:
            try:
                self.take_lease()
            except Exception as exc:
                errors.append(f"take_lease: {type(exc).__name__}: {exc}")

        try:
            command_id = self.command.robot_command(RobotCommandBuilder.stop_command())
            command_results.append({"command": "body_stop", "command_id": command_id})
        except Exception as exc:
            errors.append(f"body_stop: {type(exc).__name__}: {exc}")

        if freeze_arm:
            try:
                command_id = self.command.robot_command(RobotCommandBuilder.arm_joint_freeze_command())
                command_results.append({"command": "arm_freeze", "command_id": command_id})
            except Exception as exc:
                errors.append(f"arm_freeze: {type(exc).__name__}: {exc}")

        return {
            "cancel_requested": True,
            "commands": command_results,
            "errors": errors,
            "holding_lease": self.holding_lease,
        }

    def visualize(self, *, output: Path = DEFAULT_MAP_PATH, include_point_clouds: bool = True) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            graph = self.graph_nav.download_graph()
            snapshots = {}
            if include_point_clouds:
                for snapshot_id in sorted({waypoint.snapshot_id for waypoint in graph.waypoints if waypoint.snapshot_id}):
                    snapshots[snapshot_id] = self.graph_nav.download_waypoint_snapshot(
                        snapshot_id,
                        download_images=False,
                        do_not_download_point_cloud=False,
                    )
            path = write_graph_html(graph, self.registry, output, snapshots=snapshots)
            return {"path": str(path), "waypoints": len(graph.waypoints), "edges": len(graph.edges)}

    def deploy_arm(self, *, power_on: bool = True, take_lease: bool = False, timeout: float = 10.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            if power_on:
                power_on_motors(self.power)
            command_id = self.command.robot_command(RobotCommandBuilder.arm_ready_command())
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {"command_id": command_id, "arrived": arrived}

    def carry_arm(self, *, take_lease: bool = False, timeout: float = 10.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            command_id = self.command.robot_command(RobotCommandBuilder.arm_carry_command())
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {"command_id": command_id, "arrived": arrived}

    def freeze_arm(self, *, take_lease: bool = False) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            command_id = self.command.robot_command(RobotCommandBuilder.arm_joint_freeze_command())
            return {"command_id": command_id, "frozen": True}

    def reach_distance_to_pose(self, pose: Pose3D) -> dict[str, Any]:
        self.require_connected()
        if pose.frame_name != frame_helpers.VISION_FRAME_NAME:
            raise ValueError(f"Pose must be in {frame_helpers.VISION_FRAME_NAME!r}; got {pose.frame_name!r}.")

        with self.lock:
            robot_state = self.state.get_robot_state()
            transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
            vision_tform_hand = frame_helpers.get_a_tform_b(
                transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.HAND_FRAME_NAME,
                validate=False,
            )
            vision_tform_body = frame_helpers.get_a_tform_b(
                transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.BODY_FRAME_NAME,
                validate=False,
            )
        if vision_tform_hand is None:
            raise ValueError("Could not get current hand pose in vision frame.")
        if vision_tform_body is None:
            raise ValueError("Could not get current body pose in vision frame.")

        hand = (vision_tform_hand.x, vision_tform_hand.y, vision_tform_hand.z)
        vector = np.array([
            float(pose.x) - hand[0],
            float(pose.y) - hand[1],
            float(pose.z) - hand[2],
        ], dtype=np.float64)
        distance = float(np.linalg.norm(vector))
        body_tform_vision = vision_tform_body.inverse()
        object_body = body_tform_vision.transform_point(float(pose.x), float(pose.y), float(pose.z))
        hand_body = body_tform_vision.transform_point(hand[0], hand[1], hand[2])
        body_vector = np.array([
            float(object_body[0] - hand_body[0]),
            float(object_body[1] - hand_body[1]),
            float(object_body[2] - hand_body[2]),
        ], dtype=np.float64)
        planar_body_distance = float(np.linalg.norm(body_vector[:2]))
        return {
            "frame_name": frame_helpers.VISION_FRAME_NAME,
            "distance_m": distance,
            "hand_position": _xyz_dict(hand),
            "object_position": {"x": pose.x, "y": pose.y, "z": pose.z},
            "delta": {
                "x": float(vector[0]),
                "y": float(vector[1]),
                "z": float(vector[2]),
            },
            "body_frame_name": frame_helpers.BODY_FRAME_NAME,
            "delta_body": {
                "x": float(body_vector[0]),
                "y": float(body_vector[1]),
                "z": float(body_vector[2]),
            },
            "planar_distance_body_m": planar_body_distance,
        }

    def approach_pose(
        self,
        pose: Pose3D,
        *,
        standoff_m: float = 0.0,
        take_lease: bool = False,
        seconds: float = 1.2,
        timeout: float = 5.0,
        max_step_m: float = 0.45,
    ) -> dict[str, Any]:
        self.require_connected()
        if pose.frame_name != frame_helpers.VISION_FRAME_NAME:
            raise ValueError(f"Approach pose must be in {frame_helpers.VISION_FRAME_NAME!r}; got {pose.frame_name!r}.")

        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            robot_state = self.state.get_robot_state()
            vision_tform_hand = frame_helpers.get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.HAND_FRAME_NAME,
                validate=False,
            )
            if vision_tform_hand is None:
                raise ValueError("Could not get current hand pose in vision frame.")

            vector = np.array([
                float(pose.x) - vision_tform_hand.x,
                float(pose.y) - vision_tform_hand.y,
                float(pose.z) - vision_tform_hand.z,
            ], dtype=np.float64)
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-6:
                raise ValueError("Detected object pose is too close to current hand pose to compute an approach vector.")

            safe_standoff = _clamp(float(standoff_m), -0.10, 0.10)
            safe_seconds = _clamp(float(seconds), 0.25, 3.0)
            safe_max_step = _clamp(float(max_step_m), 0.05, 0.8)
            direction = vector / distance
            target_distance = max(0.0, distance - safe_standoff)
            step_distance = min(target_distance, safe_max_step)
            target = np.array([vision_tform_hand.x, vision_tform_hand.y, vision_tform_hand.z]) + direction * step_distance

            command = RobotCommandBuilder.arm_pose_command(
                float(target[0]),
                float(target[1]),
                float(target[2]),
                vision_tform_hand.rot.w,
                vision_tform_hand.rot.x,
                vision_tform_hand.rot.y,
                vision_tform_hand.rot.z,
                frame_helpers.VISION_FRAME_NAME,
                seconds=safe_seconds,
            )
            command_id = self.command.robot_command(command)
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {
                "command_id": command_id,
                "arrived": arrived,
                "target": {
                    "frame_name": frame_helpers.VISION_FRAME_NAME,
                    "x": float(target[0]),
                    "y": float(target[1]),
                    "z": float(target[2]),
                    "qw": vision_tform_hand.rot.w,
                    "qx": vision_tform_hand.rot.x,
                    "qy": vision_tform_hand.rot.y,
                    "qz": vision_tform_hand.rot.z,
                },
                "object_pose": asdict(pose),
                "distance_to_object_m": distance,
                "standoff_m": safe_standoff,
                "step_distance_m": step_distance,
                "seconds": safe_seconds,
            }

    def approach_pose_whole_body(
        self,
        pose: Pose3D,
        *,
        standoff_m: float = 0.0,
        take_lease: bool = False,
        seconds: float = 2.0,
        timeout: float = 8.0,
        max_step_m: float = 0.8,
    ) -> dict[str, Any]:
        self.require_connected()
        if pose.frame_name != frame_helpers.VISION_FRAME_NAME:
            raise ValueError(f"Approach pose must be in {frame_helpers.VISION_FRAME_NAME!r}; got {pose.frame_name!r}.")

        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            robot_state = self.state.get_robot_state()
            vision_tform_hand = frame_helpers.get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.HAND_FRAME_NAME,
                validate=False,
            )
            if vision_tform_hand is None:
                raise ValueError("Could not get current hand pose in vision frame.")

            vector = np.array([
                float(pose.x) - vision_tform_hand.x,
                float(pose.y) - vision_tform_hand.y,
                float(pose.z) - vision_tform_hand.z,
            ], dtype=np.float64)
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-6:
                raise ValueError("Detected object pose is too close to current hand pose to compute an approach vector.")

            safe_standoff = _clamp(float(standoff_m), -0.10, 0.10)
            safe_seconds = _clamp(float(seconds), 0.5, 5.0)
            safe_max_step = _clamp(float(max_step_m), 0.05, 1.2)
            direction = vector / distance
            target_distance = max(0.0, distance - safe_standoff)
            step_distance = min(target_distance, safe_max_step)
            target = np.array([vision_tform_hand.x, vision_tform_hand.y, vision_tform_hand.z]) + direction * step_distance

            arm_command = RobotCommandBuilder.arm_pose_command(
                float(target[0]),
                float(target[1]),
                float(target[2]),
                vision_tform_hand.rot.w,
                vision_tform_hand.rot.x,
                vision_tform_hand.rot.y,
                vision_tform_hand.rot.z,
                frame_helpers.VISION_FRAME_NAME,
                seconds=safe_seconds,
            )
            command = RobotCommandBuilder.build_synchro_command(
                RobotCommandBuilder.follow_arm_command(),
                arm_command,
            )
            command_id = self.command.robot_command(command)
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {
                "command_id": command_id,
                "arrived": arrived,
                "whole_body": True,
                "target": {
                    "frame_name": frame_helpers.VISION_FRAME_NAME,
                    "x": float(target[0]),
                    "y": float(target[1]),
                    "z": float(target[2]),
                    "qw": vision_tform_hand.rot.w,
                    "qx": vision_tform_hand.rot.x,
                    "qy": vision_tform_hand.rot.y,
                    "qz": vision_tform_hand.rot.z,
                },
                "object_pose": asdict(pose),
                "distance_to_object_m": distance,
                "standoff_m": safe_standoff,
                "step_distance_m": step_distance,
                "seconds": safe_seconds,
            }

    def arm_oscillation_monitor_status(self) -> dict[str, Any]:
        thread = self.arm_oscillation_monitor_thread
        status = dict(self.arm_oscillation_monitor_state)
        status["enabled"] = bool(thread and thread.is_alive() and not self.arm_oscillation_monitor_stop.is_set())
        status["config"] = dict(self.arm_oscillation_monitor_config)
        return status

    def configure_arm_oscillation_monitor(
        self,
        *,
        enabled: bool,
        take_lease: bool = True,
        sample_interval: float = 0.1,
        window_sec: float = 1.2,
        min_peak_to_peak_m: float = 0.012,
        min_direction_changes: int = 4,
        min_speed_mps: float = 0.025,
        freeze_cooldown_sec: float = 2.0,
    ) -> dict[str, Any]:
        self.require_connected()
        if not enabled:
            self.stop_arm_oscillation_monitor()
            return self.arm_oscillation_monitor_status()

        self.arm_oscillation_monitor_config = {
            "take_lease": bool(take_lease),
            "sample_interval": _clamp(float(sample_interval), 0.05, 0.5),
            "window_sec": _clamp(float(window_sec), 0.5, 3.0),
            "min_peak_to_peak_m": _clamp(float(min_peak_to_peak_m), 0.003, 0.08),
            "min_direction_changes": max(2, min(12, int(min_direction_changes))),
            "min_speed_mps": _clamp(float(min_speed_mps), 0.005, 0.25),
            "freeze_cooldown_sec": _clamp(float(freeze_cooldown_sec), 0.5, 10.0),
        }
        thread = self.arm_oscillation_monitor_thread
        if thread and thread.is_alive():
            self.arm_oscillation_monitor_state["enabled"] = True
            return self.arm_oscillation_monitor_status()

        self.arm_oscillation_monitor_stop.clear()
        self.arm_oscillation_monitor_state.update({
            "enabled": True,
            "samples": 0,
            "last_detection": None,
            "last_freeze": None,
            "last_error": None,
        })
        self.arm_oscillation_monitor_thread = threading.Thread(
            target=self._arm_oscillation_monitor_loop,
            name="arm-oscillation-monitor",
            daemon=True,
        )
        self.arm_oscillation_monitor_thread.start()
        return self.arm_oscillation_monitor_status()

    def stop_arm_oscillation_monitor(self) -> dict[str, Any]:
        self.arm_oscillation_monitor_stop.set()
        thread = self.arm_oscillation_monitor_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self.arm_oscillation_monitor_thread = None
        self.arm_oscillation_monitor_state["enabled"] = False
        return self.arm_oscillation_monitor_status()

    def _arm_oscillation_monitor_loop(self) -> None:
        samples: deque[tuple[float, tuple[float, float, float]]] = deque()
        last_freeze_time = 0.0
        while not self.arm_oscillation_monitor_stop.is_set():
            config = dict(self.arm_oscillation_monitor_config)
            interval = config.get("sample_interval", 0.1)
            try:
                pose = self._current_hand_position()
                now = time.monotonic()
                samples.append((now, pose))
                while samples and now - samples[0][0] > config.get("window_sec", 1.2):
                    samples.popleft()
                self.arm_oscillation_monitor_state["samples"] = len(samples)
                detection = self._detect_arm_oscillation(samples, config)
                if detection and now - last_freeze_time >= config.get("freeze_cooldown_sec", 2.0):
                    with self.lock:
                        self.ensure_lease(take_if_needed=config.get("take_lease", True))
                        command_id = self.command.robot_command(RobotCommandBuilder.arm_joint_freeze_command())
                    last_freeze_time = now
                    detection["command_id"] = command_id
                    detection["monotonic_time"] = now
                    self.arm_oscillation_monitor_state["last_detection"] = detection
                    self.arm_oscillation_monitor_state["last_freeze"] = {
                        "command_id": command_id,
                        "monotonic_time": now,
                    }
                    samples.clear()
            except Exception as exc:
                self.arm_oscillation_monitor_state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.arm_oscillation_monitor_stop.wait(interval)

    def _current_hand_position(self) -> tuple[float, float, float]:
        with self.lock:
            robot_state = self.state.get_robot_state()
            vision_tform_hand = frame_helpers.get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.HAND_FRAME_NAME,
                validate=False,
            )
        if vision_tform_hand is None:
            raise ValueError("Could not get current hand pose in vision frame.")
        return (vision_tform_hand.x, vision_tform_hand.y, vision_tform_hand.z)

    def _detect_arm_oscillation(
        self,
        samples: deque[tuple[float, tuple[float, float, float]]],
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if len(samples) < 6:
            return None
        times = [item[0] for item in samples]
        positions = [item[1] for item in samples]
        duration = times[-1] - times[0]
        if duration <= 0:
            return None

        axis_names = ("x", "y", "z")
        for axis_index, axis_name in enumerate(axis_names):
            values = [position[axis_index] for position in positions]
            peak_to_peak = max(values) - min(values)
            if peak_to_peak < config.get("min_peak_to_peak_m", 0.012):
                continue

            signs: list[int] = []
            for index in range(1, len(values)):
                dt = times[index] - times[index - 1]
                if dt <= 0:
                    continue
                speed = (values[index] - values[index - 1]) / dt
                if abs(speed) < config.get("min_speed_mps", 0.025):
                    continue
                sign = 1 if speed > 0 else -1
                if not signs or signs[-1] != sign:
                    signs.append(sign)

            direction_changes = max(0, len(signs) - 1)
            if direction_changes >= config.get("min_direction_changes", 4):
                return {
                    "axis": axis_name,
                    "duration_sec": duration,
                    "peak_to_peak_m": peak_to_peak,
                    "direction_changes": direction_changes,
                }
        return None

    def jog_arm(
        self,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
        seconds: float = 0.8,
        take_lease: bool = False,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            robot_state = self.state.get_robot_state()
            vision_tform_hand = frame_helpers.get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.HAND_FRAME_NAME,
                validate=False,
            )
            if vision_tform_hand is None:
                raise ValueError("Could not get current hand pose in vision frame.")

            safe_dx = _clamp(float(dx), -0.08, 0.08)
            safe_dy = _clamp(float(dy), -0.08, 0.08)
            safe_dz = _clamp(float(dz), -0.08, 0.08)
            requested_droll = float(droll)
            requested_dpitch = float(dpitch)
            requested_dyaw = float(dyaw)
            safe_seconds = _clamp(float(seconds), 0.25, 2.0)

            delta_x, delta_y, delta_z = vision_tform_hand.rot.transform_point(safe_dx, safe_dy, safe_dz)
            delta_rot = (
                Quat.from_roll(requested_droll)
                .mult(Quat.from_pitch(requested_dpitch))
                .mult(Quat.from_yaw(requested_dyaw))
            )
            new_rot = vision_tform_hand.rot.mult(delta_rot)

            command = RobotCommandBuilder.arm_pose_command(
                vision_tform_hand.x + delta_x,
                vision_tform_hand.y + delta_y,
                vision_tform_hand.z + delta_z,
                new_rot.w,
                new_rot.x,
                new_rot.y,
                new_rot.z,
                frame_helpers.VISION_FRAME_NAME,
                seconds=safe_seconds,
            )
            command_id = self.command.robot_command(command)
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {
                "command_id": command_id,
                "arrived": arrived,
                "dx": safe_dx,
                "dy": safe_dy,
                "dz": safe_dz,
                "droll": requested_droll,
                "dpitch": requested_dpitch,
                "dyaw": requested_dyaw,
                "seconds": safe_seconds,
            }

    def roll_camera_view(
        self,
        *,
        direction: str,
        angle_rad: float = 0.105,
        seconds: float = 0.7,
        take_lease: bool = True,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        normalized = direction.strip().lower().replace("_", "-")
        safe_angle = _clamp(abs(float(angle_rad)), 0.01, math.pi)
        if normalized in {"cw", "clockwise", "right"}:
            sign = -1.0
            direction_name = "clockwise"
        elif normalized in {"ccw", "counterclockwise", "counter-clockwise", "left"}:
            sign = 1.0
            direction_name = "counterclockwise"
        else:
            raise ValueError("direction must be one of: clockwise, counterclockwise")

        remaining = safe_angle
        step_limit = 0.25
        commands: list[dict[str, Any]] = []
        while remaining > 1e-6:
            step = min(step_limit, remaining)
            commands.append(self.jog_arm(
                droll=sign * step,
                seconds=seconds,
                take_lease=take_lease,
                timeout=timeout,
            ))
            remaining -= step

        return {
            "direction": direction_name,
            "angle_rad": safe_angle,
            "steps": len(commands),
            "commands": commands,
        }

    def stow_arm(self, *, take_lease: bool = False, timeout: float = 10.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            command_id = self.command.robot_command(RobotCommandBuilder.arm_stow_command())
            arrived = block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return {"command_id": command_id, "arrived": arrived}

    def open_gripper(
        self,
        *,
        take_lease: bool = False,
        timeout: float = 5.0,
        open_fraction: float = 1.0,
        max_vel: float | None = None,
        max_acc: float | None = None,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            safe_fraction = _clamp(float(open_fraction), 0.0, 1.0)
            command_id = self.command.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(
                safe_fraction,
                max_vel=max_vel,
                max_acc=max_acc,
            ))
            at_goal = self._block_until_gripper_at_goal(command_id, timeout_sec=timeout)
            return {"command_id": command_id, "at_goal": at_goal, "open_fraction": safe_fraction}

    def close_gripper(
        self,
        *,
        take_lease: bool = False,
        timeout: float = 5.0,
        max_vel: float | None = None,
        max_acc: float | None = None,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            command_id = self.command.robot_command(RobotCommandBuilder.claw_gripper_close_command(
                max_vel=max_vel,
                max_acc=max_acc,
            ))
            at_goal = self._block_until_gripper_at_goal(command_id, timeout_sec=timeout)
            return {"command_id": command_id, "at_goal": at_goal, "max_vel": max_vel, "max_acc": max_acc}

    def wait_for_pick_up(
        self,
        *,
        monitor_sec: float = 30.0,
        upward_threshold_m: float = 0.02,
        sample_interval: float = 0.1,
        open_duration_sec: float = 3.0,
        take_lease: bool = True,
        gripper_timeout: float = 5.0,
        stow_timeout: float = 10.0,
    ) -> dict[str, Any]:
        self.require_connected()
        self.cancel_actions_event.clear()
        self.ensure_lease(take_if_needed=take_lease)

        safe_monitor = _clamp(float(monitor_sec), 1.0, 120.0)
        safe_threshold = _clamp(float(upward_threshold_m), 0.005, 0.20)
        safe_interval = _clamp(float(sample_interval), 0.05, 1.0)
        safe_open_duration = _clamp(float(open_duration_sec), 0.1, 10.0)
        safe_gripper_timeout = _clamp(float(gripper_timeout), 1.0, 15.0)
        safe_stow_timeout = _clamp(float(stow_timeout), 1.0, 30.0)

        start_time = time.monotonic()
        initial_position = self._current_hand_position()
        max_position = initial_position
        max_upward_delta = 0.0
        samples = 1

        while time.monotonic() - start_time < safe_monitor:
            if self.cancel_actions_event.is_set():
                return {
                    "triggered": False,
                    "reason": "cancelled",
                    "samples": samples,
                    "elapsed_sec": time.monotonic() - start_time,
                    "max_upward_delta_m": max_upward_delta,
                    "initial_position": _xyz_dict(initial_position),
                    "max_position": _xyz_dict(max_position),
                    "monitor_sec": safe_monitor,
                    "upward_threshold_m": safe_threshold,
                }
            time.sleep(safe_interval)
            position = self._current_hand_position()
            samples += 1
            upward_delta = position[2] - initial_position[2]
            if upward_delta > max_upward_delta:
                max_upward_delta = upward_delta
                max_position = position
            if upward_delta >= safe_threshold:
                open_result = self.open_gripper(take_lease=take_lease, timeout=safe_gripper_timeout)
                time.sleep(safe_open_duration)
                close_result = self.close_gripper(take_lease=take_lease, timeout=safe_gripper_timeout)
                stow_result = self.stow_arm(take_lease=take_lease, timeout=safe_stow_timeout)
                return {
                    "triggered": True,
                    "reason": "upward_motion",
                    "samples": samples,
                    "elapsed_sec": time.monotonic() - start_time,
                    "upward_delta_m": upward_delta,
                    "max_upward_delta_m": max_upward_delta,
                    "initial_position": _xyz_dict(initial_position),
                    "trigger_position": _xyz_dict(position),
                    "max_position": _xyz_dict(max_position),
                    "monitor_sec": safe_monitor,
                    "upward_threshold_m": safe_threshold,
                    "open_duration_sec": safe_open_duration,
                    "open": open_result,
                    "close": close_result,
                    "stow": stow_result,
                }

        return {
            "triggered": False,
            "reason": "timeout",
            "samples": samples,
            "elapsed_sec": time.monotonic() - start_time,
            "max_upward_delta_m": max_upward_delta,
            "initial_position": _xyz_dict(initial_position),
            "max_position": _xyz_dict(max_position),
            "monitor_sec": safe_monitor,
            "upward_threshold_m": safe_threshold,
        }

    def capture_rgbd(
        self,
        *,
        color_source: str = DEFAULT_COLOR_SOURCE,
        depth_source: str = DEFAULT_DEPTH_SOURCE,
        quality_percent: int = 85,
    ) -> ImageObservation:
        self.require_connected()
        requests = [
            build_image_request(color_source, quality_percent=quality_percent),
            build_image_request(
                depth_source,
                image_format=image_pb2.Image.FORMAT_RAW,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
            ),
        ]
        color_response, depth_response = self.image.get_image(requests)
        return ImageObservation(color_response=color_response, depth_response=depth_response)

    def image_sources(self) -> list[dict[str, Any]]:
        self.require_connected()
        sources = self.image.list_image_sources()
        return [
            {
                "name": source.name,
                "cols": source.cols,
                "rows": source.rows,
                "depth_scale": source.depth_scale,
                "image_type": source.image_type,
            }
            for source in sources
        ]

    def capture_image(self, source: str, *, quality_percent: int = 85):
        self.require_connected()
        request = build_image_request(source, quality_percent=quality_percent)
        return self.image.get_image([request])[0]

    def detect_object_2d(
        self,
        instruction: str,
        *,
        model: str,
        api_key: str | None,
        color_source: str = DEFAULT_COLOR_SOURCE,
        depth_source: str = DEFAULT_DEPTH_SOURCE,
    ) -> tuple[Detection2D, Pose3D, ImageObservation]:
        self.require_connected()
        with self.lock:
            observation = self.capture_rgbd(color_source=color_source, depth_source=depth_source)
            detector = GeminiObjectDetector(model=model, api_key=api_key)
            detection = detector.detect(
                instruction=instruction,
                image_bytes=observation.color_bytes,
                mime_type=observation.color_mime_type,
                width=observation.width,
                height=observation.height,
            )
            pose = self.detection_to_3d_pose(detection, observation)
            return detection, pose, observation

    def detect_pick_target(
        self,
        instruction: str,
        *,
        model: str,
        api_key: str | None,
    ) -> dict[str, Any]:
        """Return only the language-conditioned 2D target needed by pick."""
        self.require_connected()
        with self.lock:
            color_response = self.capture_image(DEFAULT_COLOR_SOURCE)
            observation = ImageObservation(color_response=color_response)
            detector = GeminiObjectDetector(model=model, api_key=api_key)
            detection = detector.detect(
                instruction=instruction,
                image_bytes=observation.color_bytes,
                mime_type=observation.color_mime_type,
                width=observation.width,
                height=observation.height,
            )
            if observation.width <= 1 or observation.height <= 1:
                raise ValueError("Detection image dimensions must be greater than one.")
            pixel_x, pixel_y = detection.grasp_px
            normalized_x = round(1000.0 * pixel_x / (observation.width - 1))
            normalized_y = round(1000.0 * pixel_y / (observation.height - 1))
            return {
                "detected": True,
                "instruction": instruction,
                "label": detection.label,
                "confidence": detection.confidence,
                "target": {
                    "normalized_x": max(0, min(1000, normalized_x)),
                    "normalized_y": max(0, min(1000, normalized_y)),
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "image_width": observation.width,
                    "image_height": observation.height,
                },
            }

    def detect_scene(
        self,
        instruction: str,
        *,
        model: str,
        api_key: str | None,
        color_source: str = DEFAULT_COLOR_SOURCE,
        depth_source: str = DEFAULT_DEPTH_SOURCE,
        point_cloud_stride: int = 4,
        max_point_cloud_points: int = 6000,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            observation = self.capture_rgbd(color_source=color_source, depth_source=depth_source)
            detector = GeminiObjectDetector(model=model, api_key=api_key)
            detection_result = detector.detect_with_result(
                instruction=instruction,
                image_bytes=observation.color_bytes,
                mime_type=observation.color_mime_type,
                width=observation.width,
                height=observation.height,
            )
            detection = detection_result.detection
            errors: dict[str, str] = {}

            pose_vision = None
            try:
                pose_vision = asdict(self.detection_to_3d_pose(detection, observation))
            except ValueError as exc:
                errors["pose"] = str(exc)

            grasp_camera = None
            try:
                grasp_camera = self.detection_to_camera_point(detection, observation)
            except ValueError as exc:
                errors["grasp_camera"] = str(exc)

            point_cloud = {
                "frame_name": observation.depth_response.shot.frame_name_image_sensor,
                "stride": max(1, int(point_cloud_stride)),
                "fields": ["x", "y", "z", "u", "v"],
                "points": [],
            }
            if max_point_cloud_points:
                try:
                    point_cloud = self.depth_point_cloud(
                        observation,
                        stride=point_cloud_stride,
                        max_points=max_point_cloud_points,
                    )
                except ValueError as exc:
                    errors["point_cloud"] = str(exc)

            return {
                "detection": asdict(detection),
                "pose": pose_vision,
                "grasp_camera": grasp_camera,
                "model": {
                    "name": model,
                    "prompt": detection_result.prompt,
                    "raw_response": detection_result.raw_response,
                    "parsed_json": detection_result.parsed_json,
                },
                "errors": errors,
                "image": {
                    "source": color_source,
                    "width": observation.width,
                    "height": observation.height,
                    "mime_type": observation.color_mime_type,
                    "data": observation.color_bytes,
                },
                "depth": {
                    "source": depth_source,
                    "width": observation.depth_response.shot.image.cols,
                    "height": observation.depth_response.shot.image.rows,
                    "frame_name": observation.depth_response.shot.frame_name_image_sensor,
                    "camera": self.camera_intrinsics(observation.depth_response),
                },
                "point_cloud": point_cloud,
            }

    def detection_to_3d_pose(
        self,
        detection: Detection2D,
        observation: ImageObservation,
        *,
        frame_name: str = frame_helpers.VISION_FRAME_NAME,
        depth_window_px: int = 5,
    ) -> Pose3D:
        if observation.depth_response is None:
            raise ValueError("A depth image is required to project a 2D detection into 3D.")

        depth_response = observation.depth_response
        depth_m = _depth_at_pixel(depth_response, detection.grasp_px, window_px=depth_window_px)
        point_sensor = pixel_to_camera_space(
            depth_response.source,
            int(round(detection.grasp_px[0])),
            int(round(detection.grasp_px[1])),
            depth=depth_m,
        )
        frame_tform_sensor = frame_helpers.get_a_tform_b(
            depth_response.shot.transforms_snapshot,
            frame_name,
            depth_response.shot.frame_name_image_sensor,
            validate=False,
        )
        if frame_tform_sensor is None:
            raise ValueError(
                f"Could not transform {depth_response.shot.frame_name_image_sensor} into {frame_name}."
            )
        x, y, z = frame_tform_sensor.transform_point(*point_sensor)
        return Pose3D(frame_name=frame_name, x=x, y=y, z=z)

    def detection_to_camera_point(
        self,
        detection: Detection2D,
        observation: ImageObservation,
        *,
        depth_window_px: int = 5,
    ) -> dict[str, float | str]:
        if observation.depth_response is None:
            raise ValueError("A depth image is required to project a 2D detection into 3D.")

        depth_response = observation.depth_response
        depth_m = _depth_at_pixel(depth_response, detection.grasp_px, window_px=depth_window_px)
        x, y, z = pixel_to_camera_space(
            depth_response.source,
            int(round(detection.grasp_px[0])),
            int(round(detection.grasp_px[1])),
            depth=depth_m,
        )
        return {
            "frame_name": depth_response.shot.frame_name_image_sensor,
            "x": x,
            "y": y,
            "z": z,
        }

    def camera_intrinsics(self, image_response) -> dict[str, float]:
        source = image_response.source
        if not source.HasField("pinhole"):
            raise ValueError(f"Image source {source.name!r} does not have pinhole intrinsics.")
        intrinsics = source.pinhole.intrinsics
        return {
            "fx": intrinsics.focal_length.x,
            "fy": intrinsics.focal_length.y,
            "cx": intrinsics.principal_point.x,
            "cy": intrinsics.principal_point.y,
            "depth_scale": source.depth_scale or 1000.0,
        }

    def depth_point_cloud(
        self,
        observation: ImageObservation,
        *,
        stride: int = 4,
        max_points: int = 6000,
    ) -> dict[str, Any]:
        if observation.depth_response is None:
            raise ValueError("A depth image is required to create a point cloud.")
        depth_response = observation.depth_response
        image = depth_response.shot.image
        depth = np.frombuffer(image.data, dtype=np.uint16).reshape((image.rows, image.cols))
        camera = self.camera_intrinsics(depth_response)
        step = max(1, int(stride))
        ys, xs = np.mgrid[0:image.rows:step, 0:image.cols:step]
        sampled_depth = depth[ys, xs].astype(np.float64)
        valid = (sampled_depth > 0) & (sampled_depth < np.iinfo(np.uint16).max)
        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        zs = sampled_depth[valid] / camera["depth_scale"]

        if max_points > 0 and zs.size > max_points:
            indices = np.linspace(0, zs.size - 1, int(max_points), dtype=np.int64)
            xs = xs[indices]
            ys = ys[indices]
            zs = zs[indices]

        us = xs.astype(np.int64)
        vs = ys.astype(np.int64)
        points_x = zs * (xs - camera["cx"]) / camera["fx"]
        points_y = zs * (ys - camera["cy"]) / camera["fy"]
        points = np.column_stack((points_x.round(4), points_y.round(4), zs.round(4), us, vs))
        return {
            "frame_name": depth_response.shot.frame_name_image_sensor,
            "stride": step,
            "fields": ["x", "y", "z", "u", "v"],
            "points": points.tolist(),
        }

    def pick(
        self,
        instruction: str,
        *,
        model: str,
        api_key: str | None,
        take_lease: bool,
        timeout: float,
        grip_max_torque_nm: float = 2.0,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            observation = self.capture_rgbd()
            detector = GeminiObjectDetector(model=model, api_key=api_key)
            detection = detector.detect(
                instruction=instruction,
                image_bytes=observation.color_bytes,
                mime_type=observation.color_mime_type,
                width=observation.width,
                height=observation.height,
            )
            pose = self.detection_to_3d_pose(detection, observation)
            pick = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(x=detection.grasp_px[0], y=detection.grasp_px[1]),
                transforms_snapshot_for_camera=observation.color_response.shot.transforms_snapshot,
                frame_name_image_sensor=observation.color_response.shot.frame_name_image_sensor,
                camera_model=observation.color_response.source.pinhole,
            )
            constrain_grasp_to_top_down(pick)
            request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
            response = self.manipulation.manipulation_api_command(request)
            feedback = self.wait_for_manipulation(response.manipulation_cmd_id, timeout=timeout)
            light_grip = self._apply_light_grip_after_pick(
                feedback.current_state,
                max_torque_nm=grip_max_torque_nm,
            )
            return {
                "command_id": response.manipulation_cmd_id,
                "state": manipulation_api_pb2.ManipulationFeedbackState.Name(feedback.current_state),
                "detection": asdict(detection),
                "pose": asdict(pose),
                "light_grip": light_grip,
            }

    def grasp_at_pixel(
        self,
        x_pct: float,
        y_pct: float,
        *,
        take_lease: bool = True,
        timeout: float = 120.0,
        max_depth_m: float = 2.0,
        grip_max_torque_nm: float = 2.0,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            observation = self.capture_rgbd()

            pixel_x = _normalized_coordinate_to_pixel(x_pct, observation.width)
            pixel_y = _normalized_coordinate_to_pixel(y_pct, observation.height)
            depth_response = observation.depth_response
            if depth_response is None:
                raise ValueError("Aligned hand depth is required for a safe pick.")
            depth_image = depth_response.shot.image
            if (depth_image.cols, depth_image.rows) != (observation.width, observation.height):
                raise ValueError(
                    "Pick RGB/depth dimensions are not aligned: "
                    f"RGB={observation.width}x{observation.height}, "
                    f"depth={depth_image.cols}x{depth_image.rows}."
                )
            depth_m = _depth_at_pixel(depth_response, (pixel_x, pixel_y), window_px=3)
            if depth_m < 0.08 or depth_m > max_depth_m:
                raise ValueError(
                    f"Unsafe pick target depth {depth_m:.3f}m at ({pixel_x}, {pixel_y}); "
                    f"expected 0.08-{max_depth_m:.2f}m."
                )
            point_x, point_y, point_z = pixel_to_camera_space(
                depth_response.source,
                pixel_x,
                pixel_y,
                depth=depth_m,
            )

            pick = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(x=pixel_x, y=pixel_y),
                transforms_snapshot_for_camera=observation.color_response.shot.transforms_snapshot,
                frame_name_image_sensor=observation.color_response.shot.frame_name_image_sensor,
                camera_model=observation.color_response.source.pinhole,
            )
            constrain_grasp_to_top_down(pick)
            request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
            response = self.manipulation.manipulation_api_command(request)
            feedback = self.wait_for_manipulation(response.manipulation_cmd_id, timeout=timeout)
            state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(feedback.current_state)
            light_grip = self._apply_light_grip_after_pick(
                feedback.current_state,
                max_torque_nm=grip_max_torque_nm,
            )
            time.sleep(0.25)
            manipulator_state = self.state.get_robot_state().manipulator_state
            holding_item = bool(manipulator_state.is_gripper_holding_item)
            return {
                "command_id": response.manipulation_cmd_id,
                "state": state_name,
                "success": (
                    feedback.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
                    and holding_item
                ),
                "holding_item": holding_item,
                "gripper_open_percentage": manipulator_state.gripper_open_percentage,
                "light_grip": light_grip,
                "target": {
                    "normalized_x": round(float(x_pct) * 1000),
                    "normalized_y": round(float(y_pct) * 1000),
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "image_width": observation.width,
                    "image_height": observation.height,
                    "depth_m": depth_m,
                    "camera_point": {
                        "frame_name": depth_response.shot.frame_name_image_sensor,
                        "x": point_x,
                        "y": point_y,
                        "z": point_z,
                    },
                },
            }

    def _apply_light_grip_after_pick(
        self,
        manipulation_state: int,
        *,
        max_torque_nm: float,
    ) -> dict[str, Any] | None:
        """Replace the native post-pick hold with a lower-torque gripper hold."""
        if manipulation_state != manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            return None
        command_id = self.command.robot_command(
            RobotCommandBuilder.claw_gripper_close_command(
                max_torque=max_torque_nm,
            )
        )
        applying_force = self._block_until_gripper_at_goal(command_id, timeout_sec=5.0)
        return {
            "command_id": command_id,
            "max_torque_nm": max_torque_nm,
            "holding": applying_force,
        }

    def execute_360_scan(
        self,
        duration: float = 8.0,
        camera_source: str = "hand_color_image",
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=True)
            blocking_stand(self.command, timeout_sec=10)
            
            v_rot = 0.785398  # pi / 4
            command = RobotCommandBuilder.synchro_velocity_command(0.0, 0.0, v_rot)
            
            command_id = self.command.robot_command(
                command,
                end_time_secs=time.time() + duration,
                timesync_endpoint=self.robot.time_sync.endpoint,
            )
            
            time.sleep(duration + 0.5)
            
            return {
                "status": "SUCCESS",
                "message": f"Completed 360 scan rotation. Hand camera feed '{camera_source}' streamed context.",
                "command_id": command_id,
            }

    def _project_pixel_to_3d_pose(
        self,
        observation: ImageObservation,
        pixel_x: int,
        pixel_y: int,
        *,
        frame_name: str = frame_helpers.VISION_FRAME_NAME,
        depth_window_px: int = 5,
    ) -> Pose3D:
        if observation.depth_response is None:
            raise ValueError("A depth image is required to project a pixel into 3D.")

        depth_response = observation.depth_response
        depth_m = _depth_at_pixel(depth_response, [pixel_x, pixel_y], window_px=depth_window_px)
        point_sensor = pixel_to_camera_space(
            depth_response.source,
            pixel_x,
            pixel_y,
            depth=depth_m,
        )
        frame_tform_sensor = frame_helpers.get_a_tform_b(
            depth_response.shot.transforms_snapshot,
            frame_name,
            depth_response.shot.frame_name_image_sensor,
            validate=False,
        )
        if frame_tform_sensor is None:
            raise ValueError(
                f"Could not transform {depth_response.shot.frame_name_image_sensor} into {frame_name}."
            )
        x, y, z = frame_tform_sensor.transform_point(*point_sensor)
        return Pose3D(frame_name=frame_name, x=x, y=y, z=z)

    def place_at_pixel(
        self,
        x_pct: float,
        y_pct: float,
        *,
        take_lease: bool = True,
        timeout: float = 120.0,
        standoff_m: float = 0.05,
        settle_time_sec: float = 0.5,
    ) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            observation = self.capture_rgbd()

            pixel_x = _normalized_coordinate_to_pixel(x_pct, observation.width)
            pixel_y = _normalized_coordinate_to_pixel(y_pct, observation.height)
            pose_3d = self._project_pixel_to_3d_pose(observation, pixel_x, pixel_y)

            approach_result = self.approach_pose_whole_body(
                pose_3d,
                standoff_m=standoff_m,
                seconds=2.5,
                timeout=timeout,
            )
            if not approach_result.get("arrived", False):
                return {
                    "status": "FAILED",
                    "released": False,
                    "reason": "Arm did not reach the requested placement pose; gripper remains closed.",
                    "placed_at": {"x": pixel_x, "y": pixel_y},
                    "pose_3d": asdict(pose_3d),
                    "approach_result": approach_result,
                }

            time.sleep(_clamp(float(settle_time_sec), 0.0, 3.0))
            release_result = self.open_gripper(open_fraction=1.0, timeout=5.0)
            if not release_result.get("at_goal", False):
                return {
                    "status": "FAILED",
                    "released": False,
                    "reason": "Gripper did not confirm that it opened; arm was not stowed.",
                    "placed_at": {"x": pixel_x, "y": pixel_y},
                    "pose_3d": asdict(pose_3d),
                    "approach_result": approach_result,
                    "release_result": release_result,
                }

            stow_result = self.stow_smart(timeout=timeout)
            return {
                "status": "SUCCESS",
                "released": True,
                "placed_at": {"x": pixel_x, "y": pixel_y},
                "pose_3d": asdict(pose_3d),
                "approach_result": approach_result,
                "release_result": release_result,
                "stow_result": stow_result,
            }

    def stow_smart(self, *, take_lease: bool = True, timeout: float = 20.0) -> dict[str, Any]:
        self.require_connected()
        with self.lock:
            self.ensure_lease(take_if_needed=take_lease)
            robot_state = self.state.get_robot_state()
            manipulator_state = robot_state.manipulator_state
            
            is_holding = manipulator_state.is_gripper_holding_item
            
            if is_holding:
                command_id = self.command.robot_command(RobotCommandBuilder.arm_carry_command())
                block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
                return {"stowed": False, "carrying": True, "message": "Arm moved to carry pose since gripper is holding an item."}
            else:
                command_id = self.command.robot_command(RobotCommandBuilder.arm_stow_command())
                block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
                return {"stowed": True, "carrying": False, "message": "Arm stowed completely."}

    def wait_for_manipulation(self, command_id: int, *, timeout: float = 30.0, interval: float = 0.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=command_id)
            feedback = self.manipulation.manipulation_api_feedback_command(request)
            if feedback.current_state in TERMINAL_MANIP_STATES:
                return feedback
            time.sleep(interval)
        raise TimeoutError(f"Manipulation command {command_id} did not finish within {timeout} seconds.")

    def detect_external_force_change(
        self,
        *,
        threshold_newtons: float,
        sample_window_sec: float,
        interval_sec: float,
    ) -> ForceChange:
        self.require_connected()
        baseline = self._end_effector_force_norm()
        deadline = time.monotonic() + sample_window_sec
        current = baseline
        max_delta = 0.0
        while time.monotonic() < deadline:
            current = self._end_effector_force_norm()
            max_delta = max(max_delta, abs(current - baseline))
            if max_delta >= threshold_newtons:
                return ForceChange(baseline, current, max_delta, True)
            time.sleep(interval_sec)
        return ForceChange(baseline, current, max_delta, False)

    def _end_effector_force_norm(self) -> float:
        force = self.state.get_robot_state().manipulator_state.estimated_end_effector_force_in_hand
        return math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z)

    def _block_until_gripper_at_goal(self, command_id: int, *, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            feedback = self.command.robot_command_feedback(command_id)
            gripper_feedback = feedback.feedback.synchronized_feedback.gripper_command_feedback
            if gripper_feedback.HasField("claw_gripper_feedback"):
                status = gripper_feedback.claw_gripper_feedback.status
                if status in {
                    gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_AT_GOAL,
                    gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_APPLYING_FORCE,
                }:
                    return True
            time.sleep(0.1)
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalized_coordinate_to_pixel(value: float, size: int) -> int:
    if size <= 0:
        raise ValueError(f"Image dimension must be positive, got {size}.")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"Normalized coordinate must be in [0, 1], got {normalized}.")
    return int(round(normalized * (size - 1)))


def _xyz_dict(position: tuple[float, float, float]) -> dict[str, float]:
    return {"x": position[0], "y": position[1], "z": position[2]}


def _fiducial_init_value(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    mapping = {
        "none": graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
        "no_fiducial": graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
        "nearest": graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
        "nearest_at_target": graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST_AT_TARGET,
        "specific": graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported fiducial_init {value!r}. Use nearest, nearest_at_target, specific, or none.")
    return mapping[normalized]


def _localization_to_dict(localization) -> dict[str, Any]:
    result: dict[str, Any] = {"waypoint_id": localization.waypoint_id}
    if localization.HasField("waypoint_tform_body"):
        result["waypoint_tform_body"] = _se3_pose_to_dict(localization.waypoint_tform_body)
    if localization.HasField("seed_tform_body"):
        result["seed_tform_body"] = _se3_pose_to_dict(localization.seed_tform_body)
    if localization.HasField("timestamp"):
        result["timestamp_sec"] = localization.timestamp.seconds + localization.timestamp.nanos / 1e9
    return result


def _se3_pose_to_dict(pose) -> dict[str, float]:
    return {
        "x": pose.position.x,
        "y": pose.position.y,
        "z": pose.position.z,
        "qw": pose.rotation.w,
        "qx": pose.rotation.x,
        "qy": pose.rotation.y,
        "qz": pose.rotation.z,
    }
