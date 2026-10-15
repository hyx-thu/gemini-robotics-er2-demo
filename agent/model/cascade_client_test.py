"""Cascade stream: Live-protocol emulation over ASR + generateContent."""

import queue
import threading
import time
import unittest
from unittest import mock

from model import cascade_client


class FakeAsr:
  closed = False

  def __init__(self):
    self.sent = []

  def Send(self, msg):  # pylint: disable=invalid-name
    self.sent.append(msg)

  def Shutdown(self):  # pylint: disable=invalid-name
    self.closed = True


class CascadeStreamTest(unittest.TestCase):

  def setUp(self):
    self.client = cascade_client.CascadeClient("gemini-4-argon", "k")
    self.replies = queue.Queue()
    self.client.agent.generate = self._fake_generate
    self.msgs = []
    self.stream = self.client.create_stream()
    self.stream.Start(self.msgs.append, lambda: None)
    self.stream._asr = FakeAsr()  # pylint: disable=protected-access

  def tearDown(self):
    self.stream.Shutdown()

  def _fake_generate(self):
    parts = self.replies.get(timeout=5)
    self.client.agent.add("model", parts)
    return {"parts": parts, "usage": {"promptTokenCount": 10, "candidatesTokenCount": 2}}

  def _wait(self, pred, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
      if pred():
        return True
      time.sleep(0.02)
    return False

  def test_text_turn_with_tool_round_trip(self):
    self.stream.Send({"realtimeInput": {"video": {"data": "AAAA"}}})
    self.replies.put([{"functionCall": {"name": "detect", "args": {"x": 1}, "id": "c1"}}])
    self.replies.put([{"text": "好的"}])
    self.stream.Send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "拿笔"}]}], "turnComplete": True}})
    self.assertTrue(self._wait(lambda: any("toolCall" in m for m in self.msgs)))
    self.stream.Send({"toolResponse": {"functionResponses": [
        {"id": "c1", "name": "detect", "response": {"ok": True}, "scheduling": "SILENT"}]}})
    self.assertTrue(self._wait(lambda: any(
        (m.get("serverContent") or {}).get("turnComplete") for m in self.msgs)))
    texts = [p["text"] for m in self.msgs for p in
             ((m.get("serverContent") or {}).get("modelTurn") or {}).get("parts", [])]
    self.assertEqual(["好的"], texts)
    hist = self.client.agent.history
    self.assertEqual("user", hist[0]["role"])
    self.assertTrue(any("inlineData" in p for p in hist[0]["parts"]))
    fr = [p for c in hist for p in c["parts"] if "functionResponse" in p][0]
    self.assertNotIn("scheduling", fr["functionResponse"])
    done = [m for m in self.msgs if (m.get("serverContent") or {}).get("turnComplete")][0]
    self.assertEqual(2, done["usageMetadata"]["responseTokenCount"])

  def test_idle_heartbeat_ack_is_dropped_from_history(self):
    self.replies.put([{"functionCall": {"name": "ack", "args": {}, "id": "a1"}}])
    self.stream.Send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "[HEARTBEAT] tick"}]}]}})
    self.assertTrue(self._wait(lambda: any("toolCall" in m for m in self.msgs)))
    self.stream.Send({"toolResponse": {"functionResponses": [{"id": "a1", "name": "ack", "response": {}}]}})
    self.assertTrue(self._wait(lambda: any(
        (m.get("serverContent") or {}).get("turnComplete") for m in self.msgs)))
    self.assertEqual([], self.client.agent.history)

  def test_asr_utterance_becomes_user_turn(self):
    self.replies.put([{"text": "你好"}])
    self.stream._on_asr_message({"serverContent": {"inputTranscription": {"text": "你 好 "}}})  # pylint: disable=protected-access
    self.stream._on_asr_message({"voiceActivity": {"type": "ACTIVITY_END"}})  # pylint: disable=protected-access
    self.assertTrue(self._wait(lambda: any(
        (m.get("serverContent") or {}).get("turnComplete") for m in self.msgs)))
    self.assertEqual("你好", self.client.agent.history[0]["parts"][0]["text"])
    self.assertTrue(any("voiceActivity" in m for m in self.msgs))
    self.assertTrue(any((m.get("serverContent") or {}).get("inputTranscription") for m in self.msgs))

  def test_audio_is_forwarded_to_asr(self):
    msg = {"realtimeInput": {"audio": {"data": "xx"}}}
    self.stream.Send(msg)
    self.assertEqual([msg], self.stream._asr.sent)  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
