from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import bosdyn.client
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient, power_on_motors
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand


SUCCESS_STATUS = graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL
ACTIVE_STATUSES = {
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_UNKNOWN,
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_FOLLOWING_ROUTE,
}


@dataclass(frozen=True)
class NavigationResult:
    command_id: int
    status: int
    status_name: str

    @property
    def reached_goal(self) -> bool:
        return self.status == SUCCESS_STATUS


class SpotNavigationClient:
    """Small wrapper around Spot GraphNav for named-place CLI workflows."""

    def __init__(self, hostname: str, username: str, password: str, app_name: str = "spot-navigation"):
        sdk = bosdyn.client.create_standard_sdk(app_name)
        self.robot = sdk.create_robot(hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()

        self.graph_nav = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.lease = self.robot.ensure_client(LeaseClient.default_service_name)
        self.power = self.robot.ensure_client(PowerClient.default_service_name)
        self.command = self.robot.ensure_client(RobotCommandClient.default_service_name)

    def named_waypoints(self) -> dict[str, str]:
        graph = self.graph_nav.download_graph()
        waypoints: dict[str, str] = {}
        for waypoint in graph.waypoints:
            name = waypoint.annotations.name.strip()
            if name:
                waypoints[name] = waypoint.id
        return waypoints

    def download_graph(self):
        return self.graph_nav.download_graph()

    def download_waypoint_snapshot(self, snapshot_id: str):
        return self.graph_nav.download_waypoint_snapshot(
            snapshot_id,
            download_images=False,
            do_not_download_point_cloud=False,
        )

    def navigate_to_waypoint(
        self,
        waypoint_id: str,
        *,
        command_duration: float,
        timeout: float,
        feedback_interval: float,
        power_on: bool,
        stand: bool,
        acquire_lease: bool,
        take_lease: bool,
    ) -> NavigationResult:
        if take_lease and not acquire_lease:
            raise ValueError("take_lease requires acquire_lease")
        if take_lease:
            self.lease.take()
        lease_context = LeaseKeepAlive(
            self.lease,
            must_acquire=not take_lease,
            return_at_exit=True,
        ) if acquire_lease else nullcontext()
        with lease_context:
            if power_on:
                power_on_motors(self.power)
            if stand:
                blocking_stand(self.command, timeout_sec=10)

            command_id = self.graph_nav.navigate_to(waypoint_id, command_duration)
            deadline = time.monotonic() + timeout
            next_refresh = time.monotonic() + max(command_duration / 2, feedback_interval)
            last_status = graph_nav_pb2.NavigationFeedbackResponse.STATUS_UNKNOWN

            while time.monotonic() < deadline:
                feedback = self.graph_nav.navigation_feedback(command_id)
                last_status = feedback.status
                if last_status == SUCCESS_STATUS or last_status not in ACTIVE_STATUSES:
                    return NavigationResult(
                        command_id=command_id,
                        status=last_status,
                        status_name=graph_nav_pb2.NavigationFeedbackResponse.Status.Name(last_status),
                    )
                if time.monotonic() >= next_refresh:
                    command_id = self.graph_nav.navigate_to(
                        waypoint_id,
                        command_duration,
                        command_id=command_id,
                    )
                    next_refresh = time.monotonic() + max(command_duration / 2, feedback_interval)
                time.sleep(feedback_interval)

            return NavigationResult(
                command_id=command_id,
                status=last_status,
                status_name=f"TIMED_OUT_WAITING_FOR_{graph_nav_pb2.NavigationFeedbackResponse.Status.Name(last_status)}",
            )
