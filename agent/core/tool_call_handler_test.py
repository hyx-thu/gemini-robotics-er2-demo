"""Tests for declaration-driven blocking tool handling."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from core import tool_call_handler


class FakeBus:

  def __init__(self):
    self.events = []

  def subscribe(self, _event_types, _handler):
    return None

  async def publish(self, event):
    self.events.append(event)


class FakeSessionManager:

  def __init__(self):
    self.messages = []
    self.frames_sent = 0

  async def send_message(self, message):
    self.messages.append(message)

  async def send_latest_video_frame(self):
    self.frames_sent += 1
    return True


class FakeEmbodiment:

  def __init__(self):
    self.calls = []
    self.started = asyncio.Event()
    self.release = asyncio.Event()
    self.wait_for_release = False

  async def execute_action(self, action_name, **args):
    self.calls.append((action_name, args))
    self.started.set()
    if self.wait_for_release:
      await self.release.wait()
    return {"state": "done", "call_count": len(self.calls)}


class FakeUnstablePoller:

  stable_for_seconds = 1.25

  def is_stable_for(self, seconds):
    return False


class ToolCallHandlerTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.bus = FakeBus()
    self.session = FakeSessionManager()
    self.embodiment = FakeEmbodiment()
    self.handler = tool_call_handler.ToolCallHandler(
        self.bus,
        self.session,
        self.embodiment,
        blocking_tools={"detect", "pick", "navigate"},
    )

  async def test_marks_blocking_tool_active_while_it_executes(self):
    self.embodiment.wait_for_release = True
    execution = asyncio.create_task(self.handler._execute_single_tool({
        "id": "call-1",
        "name": "pick",
        "args": {},
    }))
    await self.embodiment.started.wait()
    self.assertTrue(self.handler.tool_executing.is_set())

    self.embodiment.release.set()
    await execution

    self.assertEqual(1, len(self.embodiment.calls))
    self.assertEqual(1, self.session.frames_sent)
    self.assertFalse(self.handler.tool_executing.is_set())

    event_types = [event.type for event in self.bus.events]
    self.assertNotIn(
        tool_call_handler.event_bus.EventType.HEARTBEAT_TRIGGER,
        event_types,
    )

  async def test_executes_repeated_blocking_calls(self):
    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "navigate",
        "args": {"name": "home"},
    })
    await self.handler._execute_single_tool({
        "id": "call-2",
        "name": "navigate",
        "args": {"name": "home"},
    })

    self.assertEqual(2, len(self.embodiment.calls))
    self.assertEqual(2, self.session.frames_sent)
    for message in self.session.messages:
      response = message["toolResponse"]["functionResponses"][0]["response"]
      self.assertNotIn("deduplicated", response)

  async def test_executes_batched_blocking_calls_in_order(self):
    self.embodiment.wait_for_release = True
    event = tool_call_handler.event_bus.Event(
        type=tool_call_handler.event_bus.EventType.TOOL_CALL,
        source=tool_call_handler.event_bus.EventSource.ASSISTANT,
        data={
            "functionCalls": [
                {"id": "call-1", "name": "pick", "args": {}},
                {"id": "call-2", "name": "pick", "args": {}},
            ]
        },
    )

    execution = asyncio.create_task(self.handler._handle_tool_call(event))
    await self.embodiment.started.wait()
    await asyncio.sleep(0)
    self.assertEqual(1, len(self.embodiment.calls))

    self.embodiment.release.set()
    await execution
    self.assertEqual(
        [("pick", {}), ("pick", {})],
        self.embodiment.calls,
    )

  async def test_executes_repeated_non_blocking_calls(self):
    call = {"name": "get_battery", "args": {}}
    await self.handler._execute_single_tool({"id": "call-1", **call})
    await self.handler._execute_single_tool({"id": "call-2", **call})

    self.assertEqual(2, len(self.embodiment.calls))
    self.assertEqual(0, self.session.frames_sent)

  async def test_detect_publishes_backend_target_overlay(self):
    self.embodiment.execute_action = mock.AsyncMock(return_value={
        "detected": True,
        "target": {"normalized_x": 240, "normalized_y": 610},
    })

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "detect",
        "args": {"instruction": "red cube"},
    })

    self.assertEqual("clear_overlay", self.bus.events[0].data["type"])
    self.assertEqual("target", self.bus.events[0].data["layer"])
    self.assertEqual("draw_points", self.bus.events[1].data["type"])
    self.assertEqual("target", self.bus.events[1].data["layer"])
    self.assertEqual(
        [{"x": 240.0, "y": 610.0, "label": "detected target", "kind": "target"}],
        self.bus.events[1].data["points"],
    )
    self.assertEqual("tool_call", self.bus.events[-1].data["type"])

  async def test_detect_overlay_carries_label_and_box(self):
    self.embodiment.execute_action = mock.AsyncMock(return_value={
        "detected": True,
        "target": {
            "label": "red cube",
            "normalized_x": 312,
            "normalized_y": 540,
            "box_2d": [470, 250, 610, 375],
        },
    })

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "detect",
        "args": {"instruction": "red cube", "x": 312, "y": 540},
    })

    point = self.bus.events[1].data["points"][0]
    self.assertEqual("red cube", point["label"])
    self.assertEqual([470.0, 250.0, 610.0, 375.0], point["box"])

  async def test_ui_overlay_payload_is_forwarded_and_stripped(self):
    self.embodiment.execute_action = mock.AsyncMock(return_value={
        "status": "ok",
        "ui_overlay": {
            "type": "draw_points",
            "layer": "annotation",
            "points": [{"x": 700, "y": 520, "label": "blue ball"}],
        },
    })

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "draw_points",
        "args": {"points": [{"x": 700, "y": 520, "label": "blue ball"}]},
    })

    overlay = self.bus.events[0].data
    self.assertEqual("draw_points", overlay["type"])
    self.assertEqual("annotation", overlay["layer"])
    response = self.session.messages[-1]["toolResponse"]["functionResponses"][0][
        "response"
    ]
    self.assertNotIn("ui_overlay", response)
    self.assertNotIn("ui_overlay", self.bus.events[-1].data["result"])

  async def test_pick_keeps_target_overlay_when_embodiment_requests(self):
    self.embodiment.keep_target_overlay = True
    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "pick",
        "args": {},
    })
    self.assertFalse(any(
        event.data.get("type") == "clear_overlay" for event in self.bus.events
    ))

  async def test_detect_is_blocked_until_camera_is_stable_for_three_seconds(self):
    self.embodiment.poller = FakeUnstablePoller()

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "detect",
        "args": {"instruction": "red cube"},
    })

    self.assertEqual([], self.embodiment.calls)
    self.assertFalse(any(
        event.data.get("type") == "draw_points"
        for event in self.bus.events
    ))
    response = self.session.messages[-1]["toolResponse"]["functionResponses"][0][
        "response"
    ]
    self.assertFalse(response["executed"])
    self.assertEqual(3.0, response["required_stable_seconds"])
    self.assertEqual(1.25, response["camera_stable_for_seconds"])

  async def test_pick_response_requires_visual_verification(self):
    self.embodiment.execute_action = mock.AsyncMock(return_value={
        "state": "MANIP_STATE_GRASP_SUCCEEDED",
        "success": True,
        "holding_item": True,
    })

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "pick",
        "args": {},
    })

    response = self.session.messages[-1]["toolResponse"]["functionResponses"][0][
        "response"
    ]
    self.assertEqual("unverified", response["outcome"])
    self.assertTrue(response["visual_verification_required"])
    self.assertTrue(response["post_action_frame_sent"])
    self.assertNotIn("success", response)
    self.assertNotIn("holding_item", response)

  async def test_pick_response_preserves_explicit_backend_failure(self):
    self.embodiment.execute_action = mock.AsyncMock(return_value={
        "state": "MANIP_STATE_GRASP_FAILED",
        "success": False,
        "holding_item": False,
    })

    await self.handler._execute_single_tool({
        "id": "call-1",
        "name": "pick",
        "args": {},
    })

    response = self.session.messages[-1]["toolResponse"]["functionResponses"][0][
        "response"
    ]
    self.assertEqual("failed", response["outcome"])
    self.assertFalse(response["visual_verification_required"])
    self.assertTrue(response["failure_reasons"])

  async def test_sit_is_rejected_without_explicit_user_request(self):
    result = await self.handler._invoke_tool("sit", {})

    self.assertFalse(result["executed"])
    self.assertEqual([], self.embodiment.calls)

  async def test_explicit_user_request_authorizes_one_sit(self):
    await self.handler._record_user_instruction(
        tool_call_handler.event_bus.Event(
            type=tool_call_handler.event_bus.EventType.TEXT_INPUT,
            source=tool_call_handler.event_bus.EventSource.USER,
            data="Go home and then sit down.",
        )
    )

    first = await self.handler._invoke_tool("sit", {})
    second = await self.handler._invoke_tool("sit", {})

    self.assertEqual("done", first["state"])
    self.assertFalse(second["executed"])
    self.assertEqual([("sit", {})], self.embodiment.calls)

  def test_negated_sit_request_is_not_authorized(self):
    self.assertFalse(tool_call_handler.user_requested_sit("Do not sit."))
    self.assertFalse(tool_call_handler.user_requested_sit("Never ask Spot to sit"))
    self.assertTrue(tool_call_handler.user_requested_sit("Please sit down"))

  def test_multilingual_sit_requests_are_authorized(self):
    for text in ("坐 下 来 。", "请坐下", "趴下吧", "お座りして", "座って", "앉아",
                 "Siéntate por favor", "Assieds-toi", "Setz dich", "Siediti",
                 "Сядь"):
      self.assertTrue(tool_call_handler.user_requested_sit(text), text)
    for text in ("不要坐下", "别 坐", "站起来", "座らないで", "앉지 마",
                 "No te sientes", "Setz dich nicht", "presentation"):
      self.assertFalse(tool_call_handler.user_requested_sit(text), text)

  def test_extracts_all_declared_blocking_tools(self):
    tools = [{
        "functionDeclarations": [
            {"name": "pick", "behavior": "BLOCKING"},
            {"name": "get_battery"},
            {"name": "navigate", "behavior": "blocking"},
            {"name": "stop", "behavior": "NON_BLOCKING"},
        ]
    }]

    names = tool_call_handler.blocking_tool_names(tools)

    self.assertIn("pick", names)
    self.assertIn("navigate", names)
    self.assertNotIn("stop", names)
    self.assertNotIn("get_battery", names)


class SupervisedEmbodiment(FakeEmbodiment):
  """Embodiment that opts into the action watchdog (like the sim Spot)."""

  verify_actions_visually = True
  action_timeout_s = 0.4
  action_max_attempts = 3
  keep_target_overlay = True

  def __init__(self):
    super().__init__()
    self.confirmed = []
    self.aborted = []

  def on_action_confirmed(self, name):
    self.confirmed.append(name)

  def on_task_aborted(self, name):
    self.aborted.append(name)


class WatchdogHandlerTest(unittest.IsolatedAsyncioTestCase):

  PLAN = ["detect() - find the red block", "pick() - grab it",
          "detect() - find the spot", "place() - put it down"]

  def setUp(self):
    self.bus = FakeBus()
    self.session = FakeSessionManager()
    self.embodiment = SupervisedEmbodiment()
    self.handler = tool_call_handler.ToolCallHandler(
        self.bus,
        self.session,
        self.embodiment,
        blocking_tools={"detect", "pick", "place", "confirm_action"},
    )

  def tearDown(self):
    self.handler.shutdown()

  def _events(self, typ):
    return [e.data for e in self.bus.events if e.data.get("type") == typ]

  def _last_response(self):
    return self.session.messages[-1]["toolResponse"]["functionResponses"][0]["response"]

  async def _call(self, name, args=None, call_id="c"):
    await self.handler._execute_single_tool(
        {"id": call_id, "name": name, "args": args or {}})
    return self._last_response()

  async def _plan(self, current, status="in_progress"):
    return await self._call("update_plan", {
        "goal": "move block", "steps": self.PLAN,
        "current_step": current, "status": status})

  async def _user_says(self, text):
    await self.handler._record_user_instruction(
        tool_call_handler.event_bus.Event(
            type=tool_call_handler.event_bus.EventType.USER_TRANSCRIPT,
            source=tool_call_handler.event_bus.EventSource.USER,
            data={"text": text}))

  async def test_plan_is_sticky_without_new_user_request(self):
    await self._user_says("move the block")
    await self._plan(0, "planning")
    await self._plan(2)
    # Spontaneous goal change -> original plan kept.
    other = await self._call("update_plan", {
        "goal": "fetch the red bull", "steps": ["detect('can') - a", "pick() - b"],
        "current_step": 0, "status": "planning"})
    self.assertEqual("kept_original_plan", other["status"])
    self.assertEqual("move block", self.handler._plan["goal"])
    # Regressing current_step without user input -> clamped.
    back = await self._plan(0)
    self.assertEqual(2, back["current_step"])
    # After done, a self-invented task is ignored ...
    await self._plan(3, "done")
    again = await self._call("update_plan", {
        "goal": "fetch the red bull", "steps": ["detect('can') - a", "pick() - b"],
        "current_step": 0, "status": "planning"})
    self.assertEqual("ignored", again["status"])
    # ... but a new user request re-opens planning.
    await self._user_says("now bring me the can")
    fresh = await self._call("update_plan", {
        "goal": "fetch the can", "steps": ["detect('can') - a", "pick() - b"],
        "current_step": 0, "status": "planning"})
    self.assertEqual("ok", fresh["status"])

  async def test_user_stop_cancels_task(self):
    await self._user_says("move the block")
    await self._plan(0, "planning")
    await self._call("pick")
    self.assertTrue(self.handler.task_active())
    await self._user_says("停止")
    self.assertFalse(self.handler.task_active())
    self.assertEqual("failed", self.handler._plan["status"])
    self.assertIsNone(self.handler.watchdog.pending)
    rows = [d for d in self._events("tool_call") if d.get("id") == "user-cancel"]
    self.assertEqual(1, len(rows))
    # The model was told to stop (a user-role text turn).
    texts = [m for m in self.session.messages if "clientContent" in m]
    self.assertTrue(any("cancelled" in str(m) for m in texts))

  async def test_pick_arms_watchdog_and_annotates_result(self):
    await self._plan(0, "planning")
    resp = await self._call("pick")
    self.assertEqual(1, resp["watchdog"]["attempt"])
    self.assertEqual(3, resp["watchdog"]["max_attempts"])
    self.assertIn("confirm_action", resp["next_action"])
    self.assertIn("Never ask the user", resp["next_action"])
    statuses = self._events("action_status")
    self.assertEqual("pending", statuses[-1]["state"])
    self.assertEqual("pick", statuses[-1]["action"])
    self.assertIn("pick attempt 1/3", self.handler.heartbeat_context())
    self.handler.watchdog.confirm()

  async def test_heartbeat_includes_embodiment_target_context(self):
    self.embodiment.target_context = lambda: "[TARGET] pick in flight toward 'pen'"
    self.assertIn("[TARGET] pick in flight", self.handler.heartbeat_context())
    self.embodiment.target_context = lambda: ""
    self.assertNotIn("[TARGET]", self.handler.heartbeat_context())

  async def test_plan_may_mark_pick_step_but_not_pass_it(self):
    await self._plan(0, "planning")
    await self._call("pick")
    # Moving from the detect step to the pick step itself is fine.
    ok = await self._plan(1)
    self.assertEqual("ok", ok["status"])
    # Advancing beyond the pick step is rejected until confirmed.
    rejected = await self._plan(2)
    self.assertIn("not been visually confirmed", rejected["error"])
    done = await self._plan(3, "done")
    self.assertIn("error", done)
    # The stored plan still points at the pick step.
    self.assertEqual(1, self.handler._plan["current_step"])

    # Confirming immediately after the command is not credible (no time for
    # the robot to act); the pending action stays and the model is told to
    # keep watching.
    early = await self._call("confirm_action", {"observation": "block lifted"})
    self.assertEqual("not_confirmed", early["status"])
    self.assertIsNotNone(self.handler.watchdog.pending)
    await asyncio.sleep(0.2 * self.handler.watchdog.timeout_s + 0.05)
    confirmed = await self._call("confirm_action", {"observation": "block lifted"})
    self.assertEqual("confirmed", confirmed["status"])
    self.assertEqual(["pick"], self.embodiment.confirmed)
    self.assertEqual("confirmed", self._events("action_status")[-1]["state"])
    advanced = await self._plan(2)
    self.assertEqual("ok", advanced["status"])

  async def test_place_blocked_while_pick_unconfirmed(self):
    await self._plan(1)
    await self._call("pick")
    before = len(self.embodiment.calls)
    resp = await self._call("place")
    self.assertFalse(resp["executed"])
    self.assertIn("pick", resp["error"])
    self.assertEqual(before, len(self.embodiment.calls))  # place never ran
    self.handler.watchdog.confirm()

  async def test_early_reissue_is_merged(self):
    await self._plan(1)
    await self._call("pick")
    resp = await self._call("pick")
    self.assertTrue(resp["watchdog"]["merged"])
    self.assertEqual(1, resp["watchdog"]["attempt"])
    self.assertIn("already in flight", resp["next_action"])
    self.handler.watchdog.confirm()

  async def test_supervisor_retries_then_aborts_and_fails_plan(self):
    # timeout 0.4 s, grace shortened so the whole policy runs in ~3 s.
    self.handler.watchdog.grace_s = 0.3
    # The model is still "on" the detect step when it issues pick (as seen in
    # real sessions); the failure must nevertheless be pinned to the pick step.
    await self._plan(0)
    await self._call("pick")
    deadline = asyncio.get_running_loop().time() + 6.0
    while self.handler.watchdog.pending is not None:
      if asyncio.get_running_loop().time() > deadline:
        self.fail("watchdog did not terminate")
      await asyncio.sleep(0.1)

    # Three attempts went to the embodiment: 1 from the model, 2 auto re-issues.
    picks = [c for c in self.embodiment.calls if c[0] == "pick"]
    self.assertEqual(3, len(picks))
    retry_rows = [d for d in self._events("tool_call") if d.get("source") == "watchdog"
                  and d["name"] == "pick"]
    self.assertEqual([2, 3], [d["result"]["watchdog"]["attempt"] for d in retry_rows])

    states = [s["state"] for s in self._events("action_status")]
    self.assertIn("timed_out", states)
    self.assertIn("retry", states)
    self.assertEqual("failed", states[-1])

    plan_rows = [d for d in self._events("tool_call") if d["name"] == "update_plan"
                 and d.get("source") == "watchdog"]
    self.assertEqual(1, len(plan_rows))
    self.assertEqual("failed", plan_rows[0]["args"]["status"])
    self.assertEqual(self.PLAN, plan_rows[0]["args"]["steps"])
    self.assertEqual(1, plan_rows[0]["args"]["current_step"])  # the pick step
    self.assertEqual(["pick"], self.embodiment.aborted)
    self.assertEqual("failed", self.handler._plan["status"])

    notice = self.handler.heartbeat_context()
    self.assertIn("TASK ABORTED", notice)
    self.assertIn("stay idle", notice)
    # A new pick after the abort starts a fresh task (attempt 1 again).
    resp = await self._call("pick")
    self.assertEqual(1, resp["watchdog"]["attempt"])
    self.handler.watchdog.confirm()

  async def test_model_marking_plan_failed_resets_watchdog(self):
    await self._plan(1)
    await self._call("pick")
    resp = await self._plan(1, "failed")
    self.assertEqual("ok", resp["status"])
    self.assertIsNone(self.handler.watchdog.pending)
    self.assertEqual(["pick"], self.embodiment.aborted)
    self.assertEqual("cleared", self._events("action_status")[-1]["state"])

  def test_pick_response_keeps_watchdog_fields(self):
    shaped = tool_call_handler.pick_response_for_model(
        {"executed": True, "target": {"label": "cube", "normalized_x": 1, "normalized_y": 2},
         "grasp_point": [1, 2], "watchdog": {"attempt": 2, "max_attempts": 3},
         "next_action": "watch"},
        post_action_frame_sent=True,
    )
    self.assertEqual([1, 2], shaped["grasp_point"])
    self.assertEqual(2, shaped["watchdog"]["attempt"])
    self.assertEqual("watch", shaped["next_action"])

  def test_action_step_index_prefers_step_naming_the_action(self):
    idx = tool_call_handler.ToolCallHandler._action_step_index
    self.assertEqual(1, idx("pick", 0, self.PLAN))
    self.assertEqual(3, idx("place", 1, self.PLAN))
    self.assertEqual(2, idx("navigate", 2, self.PLAN))  # fallback: issued step
    self.assertIsNone(idx("pick", None, ["stand()", "look('down')"]))


if __name__ == "__main__":
  unittest.main()
