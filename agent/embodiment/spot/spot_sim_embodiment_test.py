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


class TargetContextTest(unittest.IsolatedAsyncioTestCase):

  async def test_empty_without_target(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    self.assertEqual("", e.target_context())

  async def test_detect_not_yet_picked(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    ctx = e.target_context()
    self.assertIn("[TARGET] detected 'pen' at (x=200, y=600)", ctx)
    self.assertIn("not issued yet", ctx)
    self.assertIn("call detect", ctx)

  async def test_pick_in_flight_then_confirmed(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick")
    ctx = e.target_context()
    self.assertIn("pick in flight toward 'pen' at (x=200, y=600)", ctx)
    self.assertIn("re-issue pick(grasp_x, grasp_y)", ctx)
    self.assertIn("confirm_action instead", ctx)
    e.on_action_confirmed("pick")
    self.assertEqual("", e.target_context())

  async def test_place_in_flight(self):
    e = spot_sim_embodiment.SimulatedSpotEmbodiment()
    await e.execute_action("detect", instruction="pen", x=200, y=600)
    await e.execute_action("pick")
    e.on_action_confirmed("pick")
    await e.execute_action("place", x=820, y=420)
    ctx = e.target_context()
    self.assertIn("place in flight toward destination", ctx)
    self.assertIn("(x=820, y=420)", ctx)
    self.assertIn("re-issue place(x, y)", ctx)
    e.on_action_confirmed("place")
    self.assertEqual("", e.target_context())


if __name__ == "__main__":
  unittest.main()
