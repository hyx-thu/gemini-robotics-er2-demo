"""Tests for direct Spot movement tools."""

from __future__ import annotations

import json
import unittest

import httpx

from embodiment.spot import robot_client
from tool import tools


class MovementToolDeclarationsTest(unittest.TestCase):

  def test_spot_tools_include_navigation_and_bounded_motion(self):
    declarations = tools.spot_tools()[0]["functionDeclarations"]
    by_name = {declaration["name"]: declaration for declaration in declarations}

    self.assertIn("drive", by_name)
    self.assertIn("stop", by_name)
    self.assertIn("look", by_name)
    self.assertIn("detect", by_name)
    self.assertIn("pick", by_name)
    self.assertIn("wait_for_pick_up", by_name)
    self.assertIn("get_waypoints", by_name)
    self.assertEqual("BLOCKING", by_name["drive"]["behavior"])
    self.assertEqual("BLOCKING", by_name["stop"]["behavior"])
    self.assertEqual("BLOCKING", by_name["look"]["behavior"])
    self.assertEqual(["instruction"], by_name["detect"]["parameters"]["required"])
    self.assertEqual({}, by_name["pick"]["parameters"]["properties"])
    self.assertEqual({}, by_name["place"]["parameters"]["properties"])
    self.assertIn(
        "do not call at startup", by_name["get_waypoints"]["description"]
    )
    self.assertIn(
        "If null or absent", by_name["get_waypoints"]["description"]
    )
    self.assertIn(
        "without discussing localization",
        by_name["get_waypoints"]["description"],
    )
    self.assertIn(
        "navigation_ready=true", by_name["navigate"]["description"]
    )
    self.assertIn("call stow", by_name["navigate"]["description"])
    self.assertNotIn(
        "kitchen", by_name["navigate"]["parameters"]["properties"]["waypoint"]["description"]
    )
    self.assertIn(
        "Do not call automatically at startup",
        by_name["health_check"]["description"],
    )
    self.assertEqual("BLOCKING", by_name["wait_for_pick_up"]["behavior"])

    properties = by_name["drive"]["parameters"]["properties"]
    self.assertEqual((-0.8, 0.8), (properties["v_x"]["minimum"], properties["v_x"]["maximum"]))
    self.assertEqual((-0.5, 0.5), (properties["v_y"]["minimum"], properties["v_y"]["maximum"]))
    self.assertEqual((-1.0, 1.0), (properties["v_rot"]["minimum"], properties["v_rot"]["maximum"]))
    self.assertEqual(
        (0.1, 2.0),
        (properties["duration"]["minimum"], properties["duration"]["maximum"]),
    )
    look_properties = by_name["look"]["parameters"]["properties"]
    self.assertEqual(
        ["up", "down", "left", "right"],
        look_properties["direction"]["enum"],
    )
    self.assertEqual(
        (0.05, 0.35),
        (
            look_properties["angle_rad"]["minimum"],
            look_properties["angle_rad"]["maximum"],
        ),
    )


class SpotMovementClientTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    self.requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      if request.url.path == "/waypoints":
        return httpx.Response(200, json=[{"name": "home1"}])
      if request.url.path == "/localization":
        return httpx.Response(
            200,
            json={
                "localized": True,
                "localization": {"waypoint_id": "home-id"},
            },
        )
      if request.url.path == "/detect/pick-target":
        return httpx.Response(
            200,
            json={
                "detected": True,
                "instruction": "red cube",
                "label": "red cube",
                "confidence": 0.93,
                "target": {
                    "normalized_x": 500,
                    "normalized_y": 400,
                    "pixel_x": 319.5,
                    "pixel_y": 191.6,
                    "image_width": 640,
                    "image_height": 480,
                },
            },
        )
      return httpx.Response(200, json={"ok": True})

    self.client = robot_client.SpotRobotClient(base_url="http://spot.test")
    await self.client._client.aclose()  # pylint: disable=protected-access
    self.client._client = httpx.AsyncClient(  # pylint: disable=protected-access
        base_url="http://spot.test",
        transport=httpx.MockTransport(handler),
    )

  async def asyncTearDown(self):
    await self.client.close()

  async def test_drive_uses_guarded_velocity_request(self):
    result = await self.client.drive(
        v_x=0.3,
        v_y=-0.1,
        v_rot=0.2,
        duration=0.5,
    )

    self.assertEqual({"ok": True}, result)
    request = self.requests[-1]
    self.assertEqual("/teleop/velocity", request.url.path)
    self.assertEqual(
        {
            "v_x": 0.3,
            "v_y": -0.1,
            "v_rot": 0.2,
            "duration": 0.5,
            "take_lease": True,
            "power_on": True,
            "stand": True,
            "body_follow_arm": True,
        },
        json.loads(request.content),
    )

  async def test_stop_cancels_all_motion_and_freezes_arm(self):
    result = await self.client.stop()

    self.assertEqual({"ok": True}, result)
    request = self.requests[-1]
    self.assertEqual("/actions/stop", request.url.path)
    self.assertEqual(
        {"take_lease": True, "freeze_arm": True},
      json.loads(request.content),
    )

  async def test_look_down_rotates_only_the_arm_camera(self):
    result = await self.client.look(direction="down", angle_rad=0.2)

    self.assertEqual("down", result["direction"])
    request = self.requests[-1]
    self.assertEqual("/arm/jog", request.url.path)
    self.assertEqual(
        {
            "dpitch": 0.2,
            "seconds": 0.8,
            "take_lease": True,
            "timeout": 3.0,
        },
        json.loads(request.content),
    )

  async def test_get_waypoints_lists_backend_destinations(self):
    result = await self.client.get_waypoints()

    self.assertEqual([{"name": "home1"}], result["waypoints"])
    self.assertTrue(result["navigation_ready"])
    self.assertEqual(
        ["/waypoints", "/localization"],
        [request.url.path for request in self.requests[-2:]],
    )

  async def test_get_waypoints_returns_names_when_localization_fails(self):
    async def handler(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      if request.url.path == "/waypoints":
        return httpx.Response(200, json=[{"name": "home1"}])
      if request.url.path == "/localization":
        return httpx.Response(503, text="GraphNav unavailable")
      return httpx.Response(404)

    await self.client.close()
    self.client = robot_client.SpotRobotClient(base_url="http://spot.test")
    await self.client._client.aclose()  # pylint: disable=protected-access
    self.client._client = httpx.AsyncClient(  # pylint: disable=protected-access
        transport=httpx.MockTransport(handler), base_url="http://spot.test"
    )

    result = await self.client.get_waypoints()

    self.assertEqual([{"name": "home1"}], result["waypoints"])
    self.assertIsNone(result["navigation_ready"])
    self.assertIsNone(result["localized"])
    self.assertIn("503", result["localization_unavailable"])
    self.assertNotIn("error", result)

  async def test_detect_target_is_consumed_once_by_pick(self):
    detection = await self.client.detect("red cube")

    self.assertTrue(detection["detected"])
    self.assertEqual("/detect/pick-target", self.requests[-1].url.path)
    self.assertEqual(
        {"instruction": "red cube"}, json.loads(self.requests[-1].content)
    )

    result = await self.client.pick()

    self.assertTrue(result["ok"])
    self.assertEqual("/manipulation/grasp-pixel", self.requests[-1].url.path)
    self.assertEqual(
        {
            "x": 500,
            "y": 400,
            "take_lease": True,
            "grip_max_torque_nm": 2.0,
        },
        json.loads(self.requests[-1].content),
    )
    request_count = len(self.requests)
    second_result = await self.client.pick()
    self.assertFalse(second_result["executed"])
    self.assertEqual(request_count, len(self.requests))

  async def test_detect_target_is_consumed_once_by_place(self):
    await self.client.detect("clear area in the middle of the table")

    result = await self.client.place()

    self.assertTrue(result["ok"])
    self.assertEqual("/manipulation/place-pixel", self.requests[-1].url.path)
    self.assertEqual(
        {"x": 500, "y": 400, "take_lease": True},
        json.loads(self.requests[-1].content),
    )
    request_count = len(self.requests)
    second_result = await self.client.place()
    self.assertFalse(second_result["executed"])
    self.assertEqual(request_count, len(self.requests))

  async def test_wait_for_pick_up_calls_renamed_endpoint(self):
    result = await self.client.wait_for_pick_up()

    self.assertTrue(result["ok"])
    self.assertEqual("/pickup/wait", self.requests[-1].url.path)
    self.assertEqual(
        {
            "monitor_sec": 30.0,
            "upward_threshold_m": 0.02,
            "sample_interval": 0.1,
            "open_duration_sec": 3.0,
            "take_lease": True,
            "gripper_timeout": 5.0,
            "stow_timeout": 10.0,
        },
        json.loads(self.requests[-1].content),
    )


if __name__ == "__main__":
  unittest.main()
