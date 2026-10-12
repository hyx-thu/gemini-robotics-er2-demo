"""Tests for Spot API session state recovery."""

import unittest
from unittest.mock import Mock, patch

from bosdyn.api import image_pb2, manipulation_api_pb2
from bosdyn.client.lease import LeaseNotOwnedByWallet

from apps.api.session import SpotSession, _normalized_coordinate_to_pixel
from apps.manipulation.models import Detection2D, Pose3D
from apps.manipulation.spot_client import constrain_grasp_to_top_down


class LeaseRecoveryTest(unittest.TestCase):

    def setUp(self):
        self.session = SpotSession()
        self.session.robot = Mock()
        self.session.lease = Mock()
        self.session.lease_keepalive = Mock()
        self.session.lease_keepalive.is_alive.return_value = True

    def test_ensure_lease_retains_current_lease(self):
        lease = Mock()
        self.session.lease.lease_wallet.get_lease.return_value = lease

        self.session.ensure_lease(take_if_needed=True)

        self.session.lease.retain_lease.assert_called_once_with(lease)

    def test_ensure_lease_takes_over_displaced_lease_when_requested(self):
        self.session.lease.lease_wallet.get_lease.side_effect = LeaseNotOwnedByWallet(
            "body", Mock(lease_status="other owner")
        )
        self.session.shutdown_lease = Mock()
        self.session.take_lease = Mock()

        self.session.ensure_lease(take_if_needed=True)

        self.session.shutdown_lease.assert_called_once_with()
        self.session.take_lease.assert_called_once_with()


class PickCoordinateTest(unittest.TestCase):

    def test_maps_normalized_endpoints_inside_image(self):
        self.assertEqual(0, _normalized_coordinate_to_pixel(0.0, 640))
        self.assertEqual(639, _normalized_coordinate_to_pixel(1.0, 640))

    def test_rejects_out_of_range_coordinate(self):
        with self.assertRaises(ValueError):
            _normalized_coordinate_to_pixel(1.01, 640)

    def test_top_down_constraint_aligns_gripper_x_with_negative_vision_z(self):
        pick = manipulation_api_pb2.PickObjectInImage()

        constrain_grasp_to_top_down(pick)

        self.assertEqual("vision", pick.grasp_params.grasp_params_frame_name)
        self.assertEqual(1, len(pick.grasp_params.allowable_orientation))
        alignment = pick.grasp_params.allowable_orientation[
            0
        ].vector_alignment_with_tolerance
        self.assertEqual((1.0, 0.0, 0.0), (
            alignment.axis_on_gripper_ewrt_gripper.x,
            alignment.axis_on_gripper_ewrt_gripper.y,
            alignment.axis_on_gripper_ewrt_gripper.z,
        ))
        self.assertEqual((0.0, 0.0, -1.0), (
            alignment.axis_to_align_with_ewrt_frame.x,
            alignment.axis_to_align_with_ewrt_frame.y,
            alignment.axis_to_align_with_ewrt_frame.z,
        ))
        self.assertAlmostEqual(0.25, alignment.threshold_radians)

    def test_successful_native_pick_switches_to_light_gripper_hold(self):
        session = SpotSession()
        session.command = Mock()
        session.command.robot_command.return_value = 73
        session._block_until_gripper_at_goal = Mock(return_value=True)

        result = session._apply_light_grip_after_pick(
            manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
            max_torque_nm=2.0,
        )

        command = session.command.robot_command.call_args.args[0]
        gripper = command.synchronized_command.gripper_command.claw_gripper_command
        self.assertEqual(2.0, gripper.maximum_torque.value)
        self.assertEqual(2.0, result["max_torque_nm"])


class PlaceSequenceTest(unittest.TestCase):

    def setUp(self):
        self.session = SpotSession()
        self.session.robot = Mock()
        self.session.ensure_lease = Mock()
        self.observation = Mock(width=640, height=480)
        self.session.capture_rgbd = Mock(return_value=self.observation)
        self.pose = Pose3D(frame_name="vision", x=1.0, y=2.0, z=0.5)
        self.session._project_pixel_to_3d_pose = Mock(return_value=self.pose)
        self.session.open_gripper = Mock(return_value={"at_goal": True})
        self.session.stow_smart = Mock(return_value={"arrived": True})

    def test_failed_approach_does_not_release_object(self):
        self.session.approach_pose_whole_body = Mock(
            return_value={"arrived": False, "command_id": 10}
        )

        result = self.session.place_at_pixel(0.5, 0.5, settle_time_sec=0.0)

        self.assertEqual("FAILED", result["status"])
        self.assertFalse(result["released"])
        self.session.open_gripper.assert_not_called()
        self.session.stow_smart.assert_not_called()

    @patch("apps.api.session.time.sleep")
    def test_releases_only_after_confirmed_arrival(self, sleep):
        events = []
        self.session.approach_pose_whole_body = Mock(
            side_effect=lambda *args, **kwargs: events.append("arrive") or {"arrived": True}
        )
        self.session.open_gripper = Mock(
            side_effect=lambda *args, **kwargs: events.append("release") or {"at_goal": True}
        )
        self.session.stow_smart = Mock(
            side_effect=lambda *args, **kwargs: events.append("stow") or {"arrived": True}
        )

        result = self.session.place_at_pixel(0.5, 0.5)

        self.assertEqual("SUCCESS", result["status"])
        self.assertTrue(result["released"])
        self.assertEqual(["arrive", "release", "stow"], events)
        sleep.assert_called_once_with(0.5)


class VelocityCommandTest(unittest.TestCase):

    def setUp(self):
        self.session = SpotSession()
        self.session.robot = Mock()
        self.session.robot.time_sync.endpoint = Mock()
        self.session.command = Mock()
        self.session.command.robot_command.return_value = 42
        self.session.ensure_lease = Mock()

    def test_body_follow_arm_combines_joint_hold_with_mobility(self):
        result = self.session.velocity(
            v_x=0.2,
            v_y=0.0,
            v_rot=0.1,
            duration=0.5,
            body_follow_arm=True,
        )

        command = self.session.command.robot_command.call_args.args[0]
        synchronized = command.synchronized_command
        self.assertTrue(synchronized.HasField("mobility_command"))
        self.assertTrue(synchronized.HasField("arm_command"))
        self.assertTrue(
            synchronized.arm_command.HasField("arm_joint_move_command")
        )
        self.assertTrue(result["body_follow_arm"])


class LightweightDetectionTest(unittest.TestCase):

    @patch("apps.api.session.GeminiObjectDetector")
    def test_returns_only_pick_target_metadata(self, detector_class):
        session = SpotSession()
        session.robot = Mock()
        image = Mock()
        image.data = b"jpeg"
        image.cols = 640
        image.rows = 480
        image.format = image_pb2.Image.FORMAT_JPEG
        image.FORMAT_JPEG = image_pb2.Image.FORMAT_JPEG
        color_response = Mock()
        color_response.shot.image = image
        session.capture_image = Mock(return_value=color_response)
        detector_class.return_value.detect.return_value = Detection2D(
            label="red cube",
            confidence=0.93,
            bbox_xyxy=(319.5, 191.6, 319.5, 191.6),
            grasp_px=(319.5, 191.6),
        )

        result = session.detect_pick_target(
            "red cube", model="model", api_key=None
        )

        self.assertEqual("red cube", result["label"])
        self.assertEqual(500, result["target"]["normalized_x"])
        self.assertEqual(400, result["target"]["normalized_y"])
        self.assertNotIn("image", result)
        self.assertNotIn("point_cloud", result)
        self.assertNotIn("model", result)
        self.assertNotIn("pose", result)


if __name__ == "__main__":
    unittest.main()
