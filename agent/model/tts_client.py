"""TTS client with Gemini native TTS (multilingual) and Cloud TTS fallback.

Providers (selected via TTS_PROVIDER env var or constructor):
  * "gemini" (default): Gemini TTS models (e.g. gemini-3.8-flash-tts) via the
    generateContent REST API. Auto-detects language (Chinese, English, Japanese,
    ... 24+ languages) and supports code-switching within one sentence.
  * "cloud": Google Cloud Text-to-Speech REST API (single-language voices such
    as en-US-Chirp3-HD-Puck).

Both providers return 24 kHz, 16-bit, mono LINEAR16 PCM so the browser audio
pipeline is unchanged.
"""

import asyncio
import base64
import collections.abc
import json
import logging
import os
import re
import struct
import threading
import urllib.error
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_QUOTA_PROJECT = "robotics-hri"
_DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
_GEMINI_PREBUILT_VOICES = {
    "zephyr", "puck", "charon", "kore", "fenrir", "leda", "orus", "aoede",
    "callirrhoe", "autonoe", "enceladus", "iapetus", "umbriel", "algieba",
    "despina", "erinome", "algenib", "rasalgethi", "laomedeia", "achernar",
    "alnilam", "schedar", "gacrux", "pulcherrima", "achird", "zubenelgenubi",
    "vindemiatrix", "sadachbia", "sadaltager", "sulafat",
}


def apply_audio_gain(audio_data: bytes, gain: float) -> bytes:
  """Scales PCM LINEAR16 audio samples by gain and clips to int16 range."""
  if gain == 1.0 or not audio_data:
    return audio_data
  # Ensure even byte length for int16 view
  if len(audio_data) & 1:
    audio_data = audio_data[:-1]
  samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
  samples *= gain
  np.clip(samples, -32768, 32767, out=samples)
  return samples.astype(np.int16).tobytes()


def _strip_wav_header(data: bytes) -> bytes:
  """Returns raw PCM bytes, stripping a RIFF/WAVE header if present."""
  if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
    return data
  pos = 12
  while pos + 8 <= len(data):
    chunk_id = data[pos : pos + 4]
    chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
    if chunk_id == b"data":
      return data[pos + 8 : pos + 8 + chunk_size]
    pos += 8 + chunk_size + (chunk_size & 1)
  return data[44:]


def _to_gemini_voice(voice_name: str) -> str:
  """Maps e.g. 'en-US-Chirp3-HD-Puck' -> 'Puck'; passes Gemini names through."""
  candidate = voice_name.split("-")[-1] if "-" in voice_name else voice_name
  if candidate.lower() in _GEMINI_PREBUILT_VOICES:
    return candidate[0].upper() + candidate[1:].lower()
  logger.warning("Unknown Gemini TTS voice '%s', falling back to Puck", voice_name)
  return "Puck"


def split_into_spoken_clauses(text: str) -> list[str]:
  """Split text into natural spoken clauses while preserving smooth prosody."""
  cleaned = text.strip()
  if not cleaned:
    return []
  raw_sentences = re.split(r"(?<=[。！？.!?;；\n])\s*", cleaned)
  clauses: list[str] = []
  for sent in raw_sentences:
    sent = sent.strip()
    if not sent:
      continue
    if clauses and len(clauses[-1]) + len(sent) <= 90:
      clauses[-1] = (
          f"{clauses[-1]}{sent}"
          if re.search(r"[\u4e00-\u9fff。！？；]$", clauses[-1])
          else f"{clauses[-1]} {sent}"
      )
    else:
      clauses.append(sent)
  return clauses or [cleaned]


def transcribe_audio_pcm(
    pcm_data: bytes,
    sample_rate: int = 16000,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> str:
  """Transcribe 16-bit mono PCM audio via Gemini Flash Lite REST API."""
  if not pcm_data or len(pcm_data) < sample_rate:  # <0.5s of audio
    return ""
  resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("TTS_API_KEY")
  if not resolved_key:
    return ""
  import io  # pylint: disable=g-import-not-at-top
  import wave  # pylint: disable=g-import-not-at-top

  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)
  wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
  url = (
      "https://generativelanguage.googleapis.com/v1beta/models/"
      f"{model}:generateContent"
  )
  payload = {
      "contents": [{
          "parts": [
              {"inlineData": {"mimeType": "audio/wav", "data": wav_b64}},
              {
                  "text": (
                      "Transcribe the spoken audio verbatim in its original"
                      " language (use Simplified Chinese characters for"
                      " Mandarin Chinese). Output ONLY the transcription text"
                      " with no spaces between Chinese characters, or an empty"
                      " string if there is no speech."
                  )
              },
          ]
      }],
      "generationConfig": {"temperature": 0.0},
  }
  req = urllib.request.Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json", "x-goog-api-key": resolved_key},
      method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=6) as resp:
      if resp.status != 200:
        return ""
      data = json.loads(resp.read().decode("utf-8"))
      parts = (
          data.get("candidates", [{}])[0]
          .get("content", {})
          .get("parts", [])
      )
      text = "".join(p.get("text", "") for p in parts).strip()
      return text
  except Exception as e:  # pylint: disable=broad-except
    logger.debug("transcribe_audio_pcm failed: %s", e)
    return ""



class Tts3pClient:
  """Streaming TTS client (Gemini TTS SSE streaming + clause pipelining, or Cloud TTS REST)."""

  def __init__(
      self,
      voice_name: str = "en-US-Chirp3-HD-Puck",
      language_code: str = "en-US",
      quota_project: str = _DEFAULT_QUOTA_PROJECT,
      api_key: str | None = None,
      audio_gain: float = 1.0,
      provider: str | None = None,
      gemini_model: str | None = None,
  ):
    self._voice_name = voice_name
    self._language_code = language_code
    self._quota_project = quota_project
    self._api_key = (
        api_key
        or os.getenv("TTS_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    self._audio_gain = audio_gain
    self._provider = (provider or os.getenv("TTS_PROVIDER", "gemini")).lower()
    self._gemini_model = (
        gemini_model
        or os.getenv("GEMINI_TTS_MODEL", _DEFAULT_GEMINI_TTS_MODEL)
    )
    logger.info(
        "TTS provider=%s model=%s voice=%s (streaming=True)",
        self._provider,
        self._gemini_model if self._provider == "gemini" else "cloud-tts",
        self._voice_name,
    )

  # ---- Request builders -------------------------------------------------

  def _gemini_body(self, text: str) -> dict:
    return {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": _to_gemini_voice(self._voice_name)
                    }
                }
            },
        },
    }

  def _build_request(self, text: str, stream_sse: bool = False) -> urllib.request.Request:
    if not self._api_key:
      raise ValueError("api_key is required for Tts3pClient")
    # API key in a header, never in the URL (keeps it out of logs / proxies).
    headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}

    if self._provider == "gemini":
      method_suffix = (
          "streamGenerateContent?alt=sse" if stream_sse else "generateContent"
      )
      url = (
          "https://generativelanguage.googleapis.com/v1beta/models/"
          f"{self._gemini_model}:{method_suffix}"
      )
      body = self._gemini_body(text)
    else:
      url = "https://texttospeech.googleapis.com/v1/text:synthesize"
      body = {
          "input": {"text": text},
          "voice": {
              "languageCode": self._language_code,
              "name": self._voice_name,
          },
          "audioConfig": {
              "audioEncoding": "LINEAR16",
              "sampleRateHertz": 24000,
          },
      }

    return urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

  def _extract_audio(self, payload: dict) -> bytes:
    if self._provider == "gemini":
      chunks: list[bytes] = []
      for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
          inline = part.get("inlineData")
          if inline and inline.get("data"):
            chunks.append(_strip_wav_header(base64.b64decode(inline["data"])))
      if chunks:
        return b"".join(chunks)
      raise RuntimeError(f"Gemini TTS returned no audio: {payload}")
    return _strip_wav_header(base64.b64decode(payload["audioContent"]))

  # ---- Public API --------------------------------------------------------

  def synthesize(self, text: str) -> bytes:
    """Synthesize text to 24 kHz LINEAR16 PCM using the REST API."""
    req = self._build_request(text, stream_sse=False)
    try:
      with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 200:
          raise RuntimeError(
              f"TTS REST request failed with status: {response.status}"
          )
        payload = json.loads(response.read().decode("utf-8"))
        audio_data = self._extract_audio(payload)
        logger.info(
            "TTS[%s] synthesized %d bytes for text: %s...",
            self._provider,
            len(audio_data),
            text[:50],
        )
        return apply_audio_gain(audio_data, self._audio_gain)
    except urllib.error.HTTPError as e:
      detail = e.read().decode("utf-8", errors="replace")
      logger.error("TTS REST HTTP error %s: %s", e.code, detail[:500])
      raise
    except Exception as e:
      logger.error("TTS REST error: %s", e)
      raise

  async def _stream_clause(
      self, clause: str, cancel_event: threading.Event
  ) -> collections.abc.AsyncIterator[bytes]:
    """Stream 4096-byte PCM chunks for a single clause via SSE (with REST fallback)."""
    chunk_size = 4096
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    clause_stop = threading.Event()

    def _stopped() -> bool:
      return cancel_event.is_set() or clause_stop.is_set()

    def _sse_worker() -> None:
      got_audio = False
      try:
        if self._provider == "gemini":
          req = self._build_request(clause, stream_sse=True)
          with urllib.request.urlopen(req, timeout=45) as resp:
            if resp.status == 200:
              for raw_line in resp:
                if _stopped():
                  return
                line_s = raw_line.decode("utf-8", errors="ignore").strip()
                if not line_s.startswith("data:"):
                  continue
                data_str = line_s[5:].strip()
                if not data_str or data_str == "[DONE]":
                  continue
                try:
                  payload = json.loads(data_str)
                  pcm = self._extract_audio(payload)
                except Exception:  # pylint: disable=broad-except
                  continue
                if pcm:
                  got_audio = True
                  pcm = apply_audio_gain(pcm, self._audio_gain)
                  for i in range(0, len(pcm), chunk_size):
                    if _stopped():
                      return
                    loop.call_soon_threadsafe(
                        queue.put_nowait, pcm[i : i + chunk_size]
                    )
        if not got_audio and not _stopped():
          pcm = self.synthesize(clause)
          for i in range(0, len(pcm), chunk_size):
            if _stopped():
              return
            loop.call_soon_threadsafe(
                queue.put_nowait, pcm[i : i + chunk_size]
            )
      except Exception as e:  # pylint: disable=broad-except
        if not _stopped():
          logger.error("Streaming TTS clause failed: %s", e)
      finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    worker_fut = loop.run_in_executor(None, _sse_worker)
    try:
      while True:
        chunk = await queue.get()
        if chunk is None:
          break
        yield chunk
        await asyncio.sleep(0.005)
    finally:
      clause_stop.set()
      # Do not block cancellation if the socket is still closing
      worker_fut.cancel()

  async def _prefetch_clause(
      self, clause: str, cancel_event: threading.Event
  ) -> list[bytes]:
    chunks: list[bytes] = []
    async for ch in self._stream_clause(clause, cancel_event):
      if cancel_event.is_set():
        break
      chunks.append(ch)
    return chunks

  async def synthesize_stream(
      self, text: str
  ) -> collections.abc.AsyncIterator[bytes]:
    """Stream TTS audio chunks with clause-level pipelining and SSE streaming."""
    clauses = split_into_spoken_clauses(text)
    if not clauses:
      return

    cancel_event = threading.Event()
    prefetch_tasks: list[asyncio.Task[list[bytes]]] = []
    try:
      # Start prefetching clause #2 immediately while clause #1 streams live.
      if len(clauses) > 1:
        prefetch_tasks.append(
            asyncio.create_task(self._prefetch_clause(clauses[1], cancel_event))
        )

      async for chunk in self._stream_clause(clauses[0], cancel_event):
        yield chunk

      for idx in range(1, len(clauses)):
        if idx + 1 < len(clauses):
          prefetch_tasks.append(
              asyncio.create_task(
                  self._prefetch_clause(clauses[idx + 1], cancel_event)
              )
          )
        current_task = prefetch_tasks.pop(0)
        for chunk in await current_task:
          if cancel_event.is_set():
            return
          yield chunk
          await asyncio.sleep(0.005)
    except asyncio.CancelledError:
      cancel_event.set()
      logger.info("TTS stream cancelled (barge-in)")
      raise
    finally:
      cancel_event.set()
      for t in prefetch_tasks:
        if not t.done():
          t.cancel()

