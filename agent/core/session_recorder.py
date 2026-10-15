"""Opt-in session recorder for building a real-world test dataset.

When the user switches recording on (UI "REC" toggle), everything the model
actually saw / heard / did in the session is written to a PRIVATE GCS bucket
(env RECORDING_BUCKET; bucket has public access prevention + a 30-day
lifecycle rule):

  gs://<bucket>/sessions/<YYYY-MM-DD>/<session_id>/
    meta.json            mode / model / agent / revision / start & stop time
    events.jsonl         transcripts, model text, tool calls + results, plan,
                         watchdog / ER2 status, heartbeat texts (+ t offsets)
    frames/f_<ms>.jpg    camera frames actually sent to the model (<= 2 fps)
    audio/a_<ms>.wav     mic audio in 20 s chunks (silent chunks skipped)

Uploads run on a background thread with ADC (Cloud Run service account) via
the GCS JSON API; nothing is written to the container disk or to git.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
import uuid
import wave
from typing import Any

logger = logging.getLogger(__name__)

_SCOPES = ("https://www.googleapis.com/auth/devstorage.read_write",)
_FRAME_MIN_PERIOD_S = 0.5
_AUDIO_CHUNK_S = 20.0
_SILENCE_PEAK = 350


def recording_bucket() -> str | None:
  return os.environ.get("RECORDING_BUCKET") or None


class _Uploader:
  """Background GCS uploader (one per recorder)."""

  def __init__(self, bucket: str):
    self._bucket = bucket
    self._q: queue.Queue = queue.Queue(maxsize=2000)
    self._creds = None
    self.uploaded = 0
    self.failed = 0
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def put(self, name: str, data: bytes, content_type: str) -> None:
    try:
      self._q.put_nowait((name, data, content_type))
    except queue.Full:
      self.failed += 1

  def close(self, timeout: float = 20.0) -> None:
    self._q.put(None)
    self._thread.join(timeout)

  def _token(self) -> str:
    import google.auth  # pylint: disable=g-import-not-at-top
    import google.auth.transport.requests  # pylint: disable=g-import-not-at-top
    if self._creds is None:
      self._creds, _ = google.auth.default(scopes=_SCOPES)
    if not self._creds.valid:
      self._creds.refresh(google.auth.transport.requests.Request())
    return self._creds.token

  def _run(self) -> None:
    while True:
      item = self._q.get()
      if item is None:
        return
      name, data, ctype = item
      url = (
          f"https://storage.googleapis.com/upload/storage/v1/b/{self._bucket}/o"
          f"?uploadType=media&name={urllib.parse.quote(name, safe='')}"
      )
      for attempt in range(3):
        try:
          req = urllib.request.Request(
              url, data=data, method="POST",
              headers={"Content-Type": ctype,
                       "Authorization": f"Bearer {self._token()}"},
          )
          with urllib.request.urlopen(req, timeout=30):
            pass
          self.uploaded += 1
          break
        except Exception as e:  # pylint: disable=broad-except
          if attempt == 2:
            self.failed += 1
            logger.warning("[recorder] upload %s failed: %s", name, e)
          time.sleep(1.0 + attempt)


class SessionRecorder:
  """Collects frames / audio / events of one session and uploads them."""

  def __init__(self, bucket: str, meta: dict[str, Any], sample_rate: int = 16000):
    now = datetime.datetime.now(datetime.timezone.utc)
    self.session_id = now.strftime("%H%M%S-") + uuid.uuid4().hex[:6]
    self.prefix = f"sessions/{now:%Y-%m-%d}/{self.session_id}"
    self.bucket = bucket
    self._t0 = time.monotonic()
    self._meta = {**meta, "session_id": self.session_id,
                  "started_at": now.isoformat(), "bucket": bucket}
    self._rate = sample_rate
    self._up = _Uploader(bucket)
    self._lock = threading.Lock()
    self._events: list[str] = []
    self._last_frame = 0.0
    self._audio = bytearray()
    self._audio_start = 0.0
    self._audio_loud = False
    self.frames = 0
    self.events = 0
    self.closed = False
    self._put_json("meta.json", self._meta)
    logger.info("[recorder] recording to gs://%s/%s", bucket, self.prefix)

  def _ms(self) -> int:
    return int((time.monotonic() - self._t0) * 1000)

  def _put_json(self, name: str, obj: Any) -> None:
    self._up.put(f"{self.prefix}/{name}",
                 json.dumps(obj, ensure_ascii=False, indent=1).encode(),
                 "application/json")

  # ---- inputs ----

  def frame(self, jpeg: bytes) -> None:
    if self.closed or not jpeg:
      return
    now = time.monotonic()
    if now - self._last_frame < _FRAME_MIN_PERIOD_S:
      return
    self._last_frame = now
    self.frames += 1
    self._up.put(f"{self.prefix}/frames/f_{self._ms():08d}.jpg", jpeg, "image/jpeg")

  def audio(self, pcm: bytes) -> None:
    if self.closed or not pcm:
      return
    with self._lock:
      if not self._audio:
        self._audio_start = self._ms()
      self._audio.extend(pcm)
      if not self._audio_loud and len(pcm) >= 2:
        samples = memoryview(pcm).cast("h")
        if max(abs(s) for s in samples) > _SILENCE_PEAK:
          self._audio_loud = True
      if len(self._audio) >= self._rate * 2 * _AUDIO_CHUNK_S:
        self._flush_audio_locked()

  def _flush_audio_locked(self) -> None:
    if self._audio and self._audio_loud:
      buf = io.BytesIO()
      with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self._rate)
        w.writeframes(bytes(self._audio))
      self._up.put(f"{self.prefix}/audio/a_{int(self._audio_start):08d}.wav",
                   buf.getvalue(), "audio/wav")
    self._audio = bytearray()
    self._audio_loud = False

  def event(self, kind: str, data: Any) -> None:
    if self.closed:
      return
    try:
      line = json.dumps({"t_ms": self._ms(), "kind": kind, "data": data},
                        ensure_ascii=False, default=str)
    except Exception:  # pylint: disable=broad-except
      return
    with self._lock:
      self._events.append(line)
      self.events += 1
      # Flush periodically so a crashed session still leaves a trace.
      if len(self._events) % 50 == 0:
        self._flush_events_locked()

  def _flush_events_locked(self) -> None:
    self._up.put(f"{self.prefix}/events.jsonl",
                 ("\n".join(self._events) + "\n").encode(), "application/x-ndjson")

  # ---- lifecycle ----

  def status(self) -> dict[str, Any]:
    return {"session_id": self.session_id, "frames": self.frames,
            "events": self.events, "uploaded": self._up.uploaded,
            "failed": self._up.failed,
            "path": f"gs://{self.bucket}/{self.prefix}/"}

  def close(self) -> dict[str, Any]:
    if self.closed:
      return self.status()
    with self._lock:
      self._flush_audio_locked()
      self._flush_events_locked()
      self.closed = True
    self._meta["stopped_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    self._meta["frames"] = self.frames
    self._meta["events"] = self.events
    self._put_json("meta.json", self._meta)
    threading.Thread(target=self._up.close, daemon=True).start()
    logger.info("[recorder] stopped: %s", self.status())
    return self.status()
