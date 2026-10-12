"""Tests for SessionManager stream recovery (reconnect / stall detection).

Run from the agent directory:
  python3 -m unittest session_manager_reconnect_test
"""

import asyncio
import logging
import sys
import threading
import types
import unittest

# The workstation interpreter lacks httpx / websocket-client (only needed by
# real tool modules and the real socket, neither of which these tests touch).
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("websocket", types.ModuleType("websocket"))

from core import decision_making  # pylint: disable=g-import-not-at-top
from core import event_bus  # pylint: disable=g-import-not-at-top
import session_manager  # pylint: disable=g-import-not-at-top


class FakeStream:
  """Stands in for GeminiLiveApiStream; the "server" reacts on a thread."""

  def __init__(self, behaviour: str):
    self.behaviour = behaviour  # "ok" | "die_before_setup" | "silent"
    self.sent: list[dict] = []
    self.closed = False
    self.closed_locally = False
    self.close_code = None
    self.close_reason = ""
    self.generation = 0
    self._on_message = None
    self._on_done = None

  def Start(self, on_message, on_done):  # pylint: disable=invalid-name
    self._on_message = on_message
    self._on_done = on_done

  def Send(self, msg):  # pylint: disable=invalid-name
    if self.closed:
      raise BrokenPipeError("socket closed")
    self.sent.append(msg)
    if "setup" not in msg:
      return
    if self.behaviour == "ok":
      threading.Timer(
          0.05, lambda: self._on_message({"setupComplete": {}})
      ).start()
    elif self.behaviour == "die_before_setup":

      def _die():
        self.close_code = 1011
        self.close_reason = "Internal error encountered"
        self.closed = True
        self._on_done()

      threading.Timer(0.05, _die).start()

  def Shutdown(self):  # pylint: disable=invalid-name
    self.closed_locally = True
    if not self.closed:
      self.closed = True
      if self._on_done:
        self._on_done()

  HalfClose = Shutdown


class FakeClient:

  def __init__(self, behaviours):
    self.behaviours = list(behaviours)
    self.streams: list[FakeStream] = []

  def create_stream(self):
    behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
    stream = FakeStream(behaviour)
    self.streams.append(stream)
    return stream


class FakeEmbodiment:

  def __init__(self):
    self.audio_queue = asyncio.Queue()
    self.video_queue = asyncio.Queue()
    self.text_queue = asyncio.Queue()

  def get_audio_queue(self):
    return self.audio_queue

  def get_video_queue(self):
    return self.video_queue

  def get_text_queue(self):
    return self.text_queue


async def _make_manager(behaviours, case: unittest.TestCase | None = None):
  sm = session_manager.SessionManager(
      model="models/test",
      embodiment_instance=FakeEmbodiment(),
      api_key="test-key",
      response_modality="TEXT",
      use_event_driven_heartbeat=True,
  )
  if case is not None:
    # Observation hooks a handler into the root logger that forwards to this
    # manager's loop; drop it when the test ends so later tests stay quiet.
    case.addCleanup(logging.getLogger().removeHandler, sm.observation.log_handler)
  sm.client = FakeClient(behaviours)
  sm.decision_making = decision_making.DecisionMaking(sm.bus)
  sm._ui_queue = asyncio.Queue()  # pylint: disable=protected-access
  published = []
  sm.bus.publish_nowait = published.append
  return sm, published


def _ui_events(sm):
  out = []
  q = sm._ui_queue  # pylint: disable=protected-access
  while not q.empty():
    out.append(q.get_nowait())
  return out


class StreamRecoveryTest(unittest.IsolatedAsyncioTestCase):

  async def test_open_stream_sends_setup_before_concurrent_traffic(self):
    sm, _ = await _make_manager(["ok", "ok"], self)
    await sm._start_stream()  # pylint: disable=protected-access
    first = sm.stream

    # A sender task hammering send_message while the socket is swapped.
    async def _sender():
      for _ in range(20):
        try:
          await sm.send_message({"realtimeInput": {"audio": "x"}})
        except Exception:  # pylint: disable=broad-except
          pass
        await asyncio.sleep(0.005)

    sender = asyncio.create_task(_sender())
    await asyncio.sleep(0.02)
    ok = await sm._open_stream(None)  # pylint: disable=protected-access
    await sender
    self.assertTrue(ok)
    second = sm.stream
    self.assertIsNot(first, second)
    self.assertTrue(first.closed_locally)
    self.assertIn("setup", second.sent[0], "setup must be the first message")
    self.assertEqual(second.generation, 2)

  async def test_reconnect_resumes_then_falls_back_to_fresh(self):
    sm, published = await _make_manager(["ok", "die_before_setup", "ok"], self)
    await sm._start_stream()  # pylint: disable=protected-access
    sm._resumption_handle = "handle-1"  # pylint: disable=protected-access
    sm.decision_making.user_speaking.set()
    sm._heartbeat_in_flight = True  # pylint: disable=protected-access
    # Simulate the server killing the socket.
    dead = sm.stream
    dead.close_code, dead.close_reason, dead.closed = 1011, "Internal", True

    ok = await sm._reconnect()  # pylint: disable=protected-access
    self.assertTrue(ok)

    streams = sm.client.streams
    self.assertEqual(len(streams), 3)
    resume_setup = streams[1].sent[0]["setup"]
    self.assertEqual(
        resume_setup.get("sessionResumption", {}).get("handle"), "handle-1"
    )
    fresh_setup = streams[2].sent[0]["setup"]
    self.assertNotIn("sessionResumption", fresh_setup)
    self.assertIs(sm.stream, streams[2])
    self.assertEqual(sm._stream_generation, 3)  # pylint: disable=protected-access
    # Rejected handle is forgotten; stale flags are cleared.
    self.assertIsNone(sm._resumption_handle)  # pylint: disable=protected-access
    self.assertFalse(sm.decision_making.user_speaking.is_set())
    self.assertFalse(sm._heartbeat_in_flight)  # pylint: disable=protected-access

    states = [e["state"] for e in _ui_events(sm)]
    self.assertEqual(states[0], "reconnecting")
    self.assertEqual(states[-1], "reconnected")
    # The resume socket died while it was current (#2); its SESSION_DONE
    # must carry that generation so the main loop can discard it.
    dones = [
        e for e in published if e.type == event_bus.EventType.SESSION_DONE
    ]
    self.assertEqual(len(dones), 1)
    self.assertEqual(dones[0].data["generation"], 2)
    self.assertNotEqual(
        dones[0].data["generation"], sm._stream_generation  # pylint: disable=protected-access
    )

  async def test_superseded_stream_callbacks_are_ignored(self):
    sm, published = await _make_manager(["ok", "ok"], self)
    await sm._start_stream()  # pylint: disable=protected-access
    await sm._start_stream()  # pylint: disable=protected-access
    self.assertEqual(sm._stream_generation, 2)  # pylint: disable=protected-access

    sm._on_done(1)  # pylint: disable=protected-access
    sm._on_message({"serverContent": {"turnComplete": True}}, 1)  # pylint: disable=protected-access
    await asyncio.sleep(0.01)
    # A superseded stream may emit an informational LOG event, but nothing
    # functional (no SESSION_DONE, no turn/content events) may leak through.
    functional = [
        e for e in published if e.type != event_bus.EventType.LOG
    ]
    self.assertEqual(functional, [])

    sm._on_done(2)  # pylint: disable=protected-access
    sm._on_message({"serverContent": {"turnComplete": True}}, 2)  # pylint: disable=protected-access
    await asyncio.sleep(0.01)
    types_seen = [e.type for e in published]
    self.assertIn(event_bus.EventType.SESSION_DONE, types_seen)
    self.assertIn(event_bus.EventType.MODEL_RESPONSE, types_seen)

  async def test_health_loop_forces_reconnect_on_speech_stall(self):
    sm, _ = await _make_manager(["ok"], self)
    await sm._start_stream()  # pylint: disable=protected-access
    sm._SPEECH_STALL_S = 0.2  # pylint: disable=protected-access
    import time  # pylint: disable=g-import-not-at-top
    now = time.monotonic()
    sm.decision_making.user_speaking.set()
    sm.decision_making.last_user_speech_start = now - 1.0
    sm._last_server_msg_at = now - 1.0  # pylint: disable=protected-access

    task = asyncio.create_task(sm._stream_health_loop())  # pylint: disable=protected-access
    await asyncio.sleep(1.3)
    task.cancel()
    stream = sm.client.streams[0]
    self.assertTrue(stream.closed_locally, "stalled socket must be dropped")
    self.assertFalse(sm.decision_making.user_speaking.is_set())
    states = [e["state"] for e in _ui_events(sm)]
    self.assertIn("stalled", states)

  async def test_health_loop_leaves_healthy_long_speech_alone(self):
    sm, _ = await _make_manager(["ok"], self)
    await sm._start_stream()  # pylint: disable=protected-access
    sm._SPEECH_STALL_S = 0.2  # pylint: disable=protected-access
    import time  # pylint: disable=g-import-not-at-top
    sm.decision_making.user_speaking.set()
    sm.decision_making.last_user_speech_start = time.monotonic() - 1.0

    # A healthy server keeps talking (sessionResumptionUpdate ~2/s) even
    # during a long utterance: the liveness clock keeps moving.
    async def _chatty_server():
      while True:
        sm._last_server_msg_at = time.monotonic()  # pylint: disable=protected-access
        await asyncio.sleep(0.05)

    chatter = asyncio.create_task(_chatty_server())
    task = asyncio.create_task(sm._stream_health_loop())  # pylint: disable=protected-access
    await asyncio.sleep(1.3)
    task.cancel()
    chatter.cancel()
    self.assertFalse(sm.client.streams[0].closed_locally)
    self.assertTrue(sm.decision_making.user_speaking.is_set())


if __name__ == "__main__":
  unittest.main()
