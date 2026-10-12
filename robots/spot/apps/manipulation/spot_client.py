from __future__ import annotations

import math
import time
from contextlib import contextmanager

import bosdyn.client
import numpy as np
from bosdyn.api import geometry_pb2, gripper_command_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.image import ImageClient, build_image_request, pixel_to_camera_space
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.power import PowerClient, power_on_motors
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient

from apps.manipulation.gemini_detector import GeminiObjectDetector
from apps.manipulation.models import Detection2D, ForceChange, ImageObservation, Pose3D


DEFAULT_COLOR_SOURCE = "hand_color_image"
DEFAULT_DEPTH_SOURCE = "hand_depth_in_hand_color_frame"
TERMINAL_MANIP_STATES = {
    manipulation_api_pb2.MANIP_STATE_DONE,
    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
    manipulation_api_pb2.MANIP_STATE_PLACE_SUCCEEDED,
    manipulation_api_pb2.MANIP_STATE_PLACE_FAILED,
}


def constrain_grasp_to_top_down(
    pick: manipulation_api_pb2.PickObjectInImage,
    *,
    threshold_radians: float = 0.25,
) -> None:
    """Require the gripper x-axis to point down in the vision frame."""
    constraint = pick.grasp_params.allowable_orientation.add()
    alignment = constraint.vector_alignment_with_tolerance
    alignment.axis_on_gripper_ewrt_gripper.CopyFrom(
        geometry_pb2.Vec3(x=1.0, y=0.0, z=0.0)
    )
    alignment.axis_to_align_with_ewrt_frame.CopyFrom(
        geometry_pb2.Vec3(x=0.0, y=0.0, z=-1.0)
    )
    alignment.threshold_radians = threshold_radians
    pick.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME


class SpotManipulationClient:
    def __init__(self, hostname: str, username: str, password: str, app_name: str = "spot-manipulation"):
        sdk = bosdyn.client.create_standard_sdk(app_name)
        self.robot = sdk.create_robot(hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()

        self.command = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image = self.robot.ensure_client(ImageClient.default_service_name)
        self.lease = self.robot.ensure_client(LeaseClient.default_service_name)
        self.manipulation = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.power = self.robot.ensure_client(PowerClient.default_service_name)
        self.state = self.robot.ensure_client(RobotStateClient.default_service_name)

    @contextmanager
    def lease_context(self, *, take: bool = False, return_at_exit: bool = True):
        if take:
            self.lease.take()
            keepalive = LeaseKeepAlive(self.lease, must_acquire=False, return_at_exit=return_at_exit)
        else:
            keepalive = LeaseKeepAlive(self.lease, must_acquire=True, return_at_exit=return_at_exit)
        with keepalive:
            yield

    def deploy_arm(self, *, take_lease: bool = False, timeout: float = 10.0, power_on: bool = True) -> int:
        with self.lease_context(take=take_lease):
            if power_on:
                power_on_motors(self.power)
            command_id = self.command.robot_command(RobotCommandBuilder.arm_ready_command())
            block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return command_id

    def stow_arm(self, *, take_lease: bool = False, timeout: float = 10.0) -> int:
        with self.lease_context(take=take_lease):
            command_id = self.command.robot_command(RobotCommandBuilder.arm_stow_command())
            block_until_arm_arrives(self.command, command_id, timeout_sec=timeout)
            return command_id

    def open_gripper(self, *, take_lease: bool = False, timeout: float = 5.0) -> int:
        with self.lease_context(take=take_lease):
            command_id = self.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            _block_until_gripper_at_goal(self.command, command_id, timeout_sec=timeout)
            return command_id

    def capture_rgbd(
        self,
        *,
        color_source: str = DEFAULT_COLOR_SOURCE,
        depth_source: str = DEFAULT_DEPTH_SOURCE,
        quality_percent: int = 85,
    ) -> ImageObservation:
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

    def detect_object_2d(
        self,
        instruction: str,
        *,
        detector: GeminiObjectDetector | None = None,
        color_source: str = DEFAULT_COLOR_SOURCE,
    ) -> tuple[Detection2D, ImageObservation]:
        observation = self.capture_rgbd(color_source=color_source)
        detector = detector or GeminiObjectDetector()
        detection = detector.detect(
            instruction=instruction,
            image_bytes=observation.color_bytes,
            mime_type=observation.color_mime_type,
            width=observation.width,
            height=observation.height,
        )
        return detection, observation

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

    def pick_detection_in_image(
        self,
        detection: Detection2D,
        observation: ImageObservation,
        *,
        take_lease: bool = False,
        timeout: float = 30.0,
    ) -> int:
        with self.lease_context(take=take_lease):
            pick = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(x=detection.grasp_px[0], y=detection.grasp_px[1]),
                transforms_snapshot_for_camera=observation.color_response.shot.transforms_snapshot,
                frame_name_image_sensor=observation.color_response.shot.frame_name_image_sensor,
                camera_model=observation.color_response.source.pinhole,
            )
            constrain_grasp_to_top_down(pick)
            request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
            response = self.manipulation.manipulation_api_command(request)
            self.wait_for_manipulation(response.manipulation_cmd_id, timeout=timeout)
            return response.manipulation_cmd_id

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
        threshold_newtons: float = 5.0,
        sample_window_sec: float = 3.0,
        interval_sec: float = 0.1,
    ) -> ForceChange:
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


def _depth_at_pixel(depth_response, pixel_xy: tuple[float, float], *, window_px: int = 5) -> float:
    image = depth_response.shot.image
    depth = np.frombuffer(image.data, dtype=np.uint16).reshape((image.rows, image.cols))
    x = int(round(pixel_xy[0]))
    y = int(round(pixel_xy[1]))
    x0 = max(0, x - window_px)
    x1 = min(image.cols, x + window_px + 1)
    y0 = max(0, y - window_px)
    y1 = min(image.rows, y + window_px + 1)
    samples = depth[y0:y1, x0:x1]
    valid = samples[(samples > 0) & (samples < np.iinfo(np.uint16).max)]
    if valid.size == 0:
        raise ValueError(f"No valid depth values near pixel ({x}, {y}).")
    scale = depth_response.source.depth_scale or 1000.0
    return float(np.median(valid) / scale)


def _block_until_gripper_at_goal(command_client: RobotCommandClient, command_id: int, *, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        feedback = command_client.robot_command_feedback(command_id)
        gripper_feedback = feedback.feedback.synchronized_feedback.gripper_command_feedback
        if gripper_feedback.HasField("claw_gripper_feedback"):
            status = gripper_feedback.claw_gripper_feedback.status
            if status == gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_AT_GOAL:
                return True
            if status == gripper_command_pb2.ClawGripperCommand.Feedback.STATUS_APPLYING_FORCE:
                return True
        time.sleep(0.1)
    return False
