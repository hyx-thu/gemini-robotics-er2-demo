"""Sim embodiment: pick/place follow the latest coordinates."""
import asyncio
import unittest

from embodiment.spot import spot_sim_embodiment


class MovedObjectTest(unittest.IsolatedAsyncioTestCase):

  async def test_pick_uses_updated_grasp_point_and_reanchors_trajectory(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    old_traj = [
        {"x": 200, "y": 450, "label": "approach"},
        {"x": 200, "y": 600, "label": "grasp"},
        {"x": 200, "y": 400, "label": "lift"},
    ]
    # Object moved before the grasp: model passes coordinates from the
    # latest frame but an outdated trajectory.
    r = await e.execute_action(
        "pick", trajectory=old_traj, grasp_x=500, grasp_y=650)
    self.assertEqual([500, 650], r["grasp_point"])
    grasp = [p for p in r["trajectory"] if p["label"] == "grasp"][0]
    self.assertEqual((500, 650), (grasp["x"], grasp["y"]))

  async def test_retry_after_redetect_does_not_reuse_old_trajectory(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick", trajectory=[
        {"x": 200, "y": 450}, {"x": 200, "y": 600, "label": "grasp"}])
    # Object moved; model re-detects and re-issues pick without trajectory.
    await e.execute_action("detect", instruction="pen", x=700, y=300)
    r = await e.execute_action("pick")
    grasp = [p for p in r["trajectory"] if p["label"] == "grasp"][0]
    self.assertEqual((700, 300), (grasp["x"], grasp["y"]))

  async def test_place_uses_current_xy(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick", trajectory=[
        {"x": 200, "y": 450}, {"x": 200, "y": 600, "label": "grasp"}])
    await e.execute_action("detect", instruction="notebook", x=600, y=500)
    r = await e.execute_action("place", x=820, y=420, trajectory=[
        {"x": 200, "y": 600}, {"x": 400, "y": 300}, {"x": 600, "y": 500, "label": "place"}])
    self.assertEqual([820, 420], r["place_point"])
    self.assertEqual((820, 420), (r["trajectory"][-1]["x"], r["trajectory"][-1]["y"]))

  async def test_place_path_starts_at_current_hand_not_pick_spot(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick")
    # Lifted and carried: the hand is now at (450, 250).
    r = await e.execute_action("place", x=820, y=420, from_x=450, from_y=250,
                               trajectory=[{"x": 600, "y": 200, "label": "arc"},
                                           {"x": 820, "y": 420, "label": "place"}])
    self.assertEqual([450, 250], r["start_point"])
    self.assertNotIn("pick origin", [p["label"] for p in r["trajectory"]])
    self.assertNotIn("pick_origin", r)
    # Without from_x/y the model's own first waypoint is the start.
    e2 = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e2.execute_action("detect", instruction="pen", x=200, y=600)
    await e2.execute_action("pick")
    r2 = await e2.execute_action("place", x=820, y=420,
                                 trajectory=[{"x": 500, "y": 260}, {"x": 820, "y": 420}])
    self.assertEqual([500, 260], r2["start_point"])


class TargetContextTest(unittest.IsolatedAsyncioTestCase):

  async def test_empty_without_pending_command_and_tracks_pre_pick_detect(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    self.assertEqual("", e.target_context())
    det = await e.execute_action(
        "detect", instruction="pen", x=200, y=600, box_2d=[550, 150, 650, 250]
    )
    self.assertIn("moved away from (x=200, y=600)", det["next_action"])
    ctx = e.target_context()
    self.assertIn("[TARGET] Target detected", ctx)
    self.assertIn("before pick/place is issued", ctx)
    self.assertIn("'pen' at (x=200, y=600) box_2d=[550, 150, 650, 250]", ctx)
    self.assertIn("IMMEDIATELY call detect", ctx)

  async def test_pick_preserves_box_on_minor_refinement_and_tracks_redetect_in_flight(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action(
        "detect", instruction="pen", x=200, y=600, box_2d=[550, 150, 650, 250]
    )
    # Minor grasp-point refinement inside box_2d keeps box_2d in [TARGET].
    await e.execute_action("pick", grasp_x=205, grasp_y=598)
    ctx = e.target_context()
    self.assertIn("box_2d=[550, 150, 650, 250]", ctx)
    # Re-detect while pick is in flight updates next_action and [TARGET] coords.
    det2 = await e.execute_action(
        "detect", instruction="pen", x=750, y=500, box_2d=[450, 700, 550, 800]
    )
    self.assertIn("redirect the in-flight pick", det2["next_action"])
    ctx2 = e.target_context()
    self.assertIn("(x=750, y=500) box_2d=[450, 700, 550, 800]", ctx2)
    self.assertIn("already re-detected the moved object", ctx2)

  async def test_pick_in_flight_monitors_only_the_object(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick")
    ctx = e.target_context()
    self.assertIn("[TARGET] pick command sent", ctx)
    self.assertIn("the OBJECT to grasp 'pen' at (x=200, y=600)", ctx)
    self.assertIn("human hand - IS the robot arm", ctx)
    self.assertIn("re-issue pick(grasp_x, grasp_y)", ctx)
    self.assertIn("do not re-plan", ctx)
    e.on_action_confirmed("pick")
    self.assertEqual("", e.target_context())

  async def test_place_in_flight_monitors_only_the_destination(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick")
    e.on_action_confirmed("pick")
    await e.execute_action("place", x=820, y=420)
    ctx = e.target_context()
    self.assertIn("[TARGET] place command sent", ctx)
    self.assertIn("the DESTINATION", ctx)
    self.assertIn("(x=820, y=420)", ctx)
    self.assertIn("never detect it again", ctx)
    self.assertIn("re-issue place(x, y)", ctx)
    e.on_action_confirmed("place")
    self.assertEqual("", e.target_context())

  async def test_detect_rejects_object_already_held_in_hand(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="yellow pen", x=540, y=730)
    await e.execute_action("pick")
    # Model sees the hand holding the pen and tries to detect it again.
    r = await e.execute_action(
        "detect", instruction="the yellow pen held in the hand", x=530, y=510
    )
    self.assertFalse(r["detected"])
    self.assertTrue(r["already_in_hand"])
    self.assertEqual([530, 510], r["hand_position"])
    self.assertIn("confirm_action", r["next_action"])
    self.assertIn("from_x=530, from_y=510", r["next_action"])


if __name__ == "__main__":
  unittest.main()
