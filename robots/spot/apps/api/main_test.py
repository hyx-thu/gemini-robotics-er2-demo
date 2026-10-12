"""Tests for Spot API static debug pages."""

import unittest
from unittest.mock import patch

from apps.api import main


class CameraDebugPageTest(unittest.TestCase):

    def test_camera_debug_page_is_served(self):
        response = main.cameras_page()

        self.assertEqual(main.CAMERAS_HTML, response.path)
        self.assertTrue(main.CAMERAS_HTML.is_file())

    def test_camera_debug_page_discovers_and_renders_depth_sources(self):
        html = main.CAMERAS_HTML.read_text(encoding="utf-8")

        self.assertIn("/images/sources", html)
        self.assertIn("renderDepth", html)
        self.assertIn("runLimited(sources, 3", html)


class StartupConnectionTest(unittest.TestCase):

    @patch.object(main, "load_dot_config")
    @patch.object(main.session, "connect", side_effect=RuntimeError("offline"))
    @patch.dict(
        main.os.environ,
        {
            "SPOT_HOSTNAME": "192.0.2.1",
            "BOSDYN_CLIENT_USERNAME": "user",
            "BOSDYN_CLIENT_PASSWORD": "password",
        },
    )
    def test_robot_connection_failure_does_not_abort_api_startup(
        self, connect, _load_dot_config
    ):
        main.startup_connect()

        connect.assert_called_once()


class LightweightDetectionEndpointTest(unittest.TestCase):

    @patch.object(main.session, "detect_pick_target")
    def test_dispatches_compact_pick_target_detection(self, detect_pick_target):
        detect_pick_target.return_value = {
            "detected": True,
            "target": {"normalized_x": 500, "normalized_y": 400},
        }

        result = main.detect_pick_target(
            main.DetectPickTargetRequest(instruction="red cube")
        )

        self.assertTrue(result["detected"])
        detect_pick_target.assert_called_once_with(
            "red cube",
            model=main.DEFAULT_MODEL,
            api_key=None,
        )


class CommandEndpointDispatchTest(unittest.TestCase):

    @patch.object(main.session, "stand")
    def test_stand_dispatches_only_stand_parameters(self, stand):
        stand.return_value = {"standing": True}

        result = main.stand(main.StandRequest(take_lease=True))

        self.assertTrue(result["standing"])
        stand.assert_called_once_with(
            power_on=True,
            take_lease=True,
            timeout=10.0,
        )

    @patch.object(main.session, "pick")
    def test_pick_dispatches_light_grip_torque(self, pick):
        pick.return_value = {"state": "MANIP_STATE_GRASP_SUCCEEDED"}

        main.pick(main.PickRequest(instruction="red cube"))

        pick.assert_called_once_with(
            "red cube",
            model=main.DEFAULT_MODEL,
            api_key=None,
            take_lease=False,
            timeout=30.0,
            grip_max_torque_nm=2.0,
        )

    @patch.object(main.session, "wait_for_pick_up")
    def test_wait_for_pick_up_dispatches_monitor_parameters(self, wait_for_pick_up):
        wait_for_pick_up.return_value = {"triggered": False, "reason": "timeout"}

        main.wait_for_pick_up(main.WaitForPickUpRequest())

        wait_for_pick_up.assert_called_once_with(
            monitor_sec=30.0,
            upward_threshold_m=0.02,
            sample_interval=0.1,
            open_duration_sec=3.0,
            take_lease=True,
            gripper_timeout=5.0,
            stow_timeout=10.0,
        )


if __name__ == "__main__":
    unittest.main()
