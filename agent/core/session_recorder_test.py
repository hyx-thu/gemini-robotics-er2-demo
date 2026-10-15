"""Session recorder: what gets uploaded where (uploader stubbed)."""

import json
import unittest
from unittest import mock

from core import session_recorder


class FakeUploader:

  def __init__(self, bucket):
    self.items = []
    self.uploaded = 0
    self.failed = 0

  def put(self, name, data, ctype):
    self.items.append((name, data, ctype))

  def close(self, timeout=0):
    pass


class SessionRecorderTest(unittest.TestCase):

  def setUp(self):
    patcher = mock.patch.object(session_recorder, "_Uploader", FakeUploader)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.rec = session_recorder.SessionRecorder("bkt", {"model": "m"})
    self.up = self.rec._up  # pylint: disable=protected-access

  def names(self):
    return [n for n, _, _ in self.up.items]

  def test_layout_and_meta(self):
    self.assertTrue(self.rec.prefix.startswith("sessions/"))
    self.assertTrue(self.names()[0].endswith("/meta.json"))
    meta = json.loads(self.up.items[0][1])
    self.assertEqual("m", meta["model"])

  def test_frames_are_rate_limited(self):
    self.rec.frame(b"jpg1")
    self.rec.frame(b"jpg2")  # within 0.5 s -> dropped
    frames = [n for n in self.names() if "/frames/" in n]
    self.assertEqual(1, len(frames))

  def test_silent_audio_is_skipped_loud_audio_kept(self):
    silent = b"\x00\x00" * 16000 * 21
    self.rec.audio(silent)
    self.assertFalse([n for n in self.names() if "/audio/" in n])
    loud = (b"\x00\x10" * 16000) * 21
    self.rec.audio(loud)
    wavs = [n for n in self.names() if "/audio/" in n]
    self.assertEqual(1, len(wavs))

  def test_events_flushed_on_close(self):
    self.rec.event("user_transcript", {"text": "拿笔"})
    status = self.rec.close()
    ev = [d for n, d, _ in self.up.items if n.endswith("events.jsonl")][-1]
    line = json.loads(ev.decode().splitlines()[0])
    self.assertEqual("user_transcript", line["kind"])
    self.assertEqual("拿笔", line["data"]["text"])
    self.assertIn("gs://bkt/sessions/", status["path"])
    self.rec.event("late", {})  # ignored after close
    self.assertEqual(1, self.rec.events)


if __name__ == "__main__":
  unittest.main()
