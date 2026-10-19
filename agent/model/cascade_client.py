"""Cascade "Live" stream: Gemini ASR -> non-Live LLM (generateContent) -> TTS.

``gemini-4-argon`` is not a Live (bidi) model. This module wraps it behind
the same streaming interface as ``GeminiLiveApiStream`` (Start / Send /
HalfClose / Shutdown + Live JSON messages), so SessionManager, DecisionMaking,
ToolCallHandler, the heartbeat scheduler and the UI work unchanged:

* realtimeInput.audio  -> forwarded to the Live transcription model
  (``gemini-3.5-transcribe-live``); its voiceActivity / inputTranscription
  messages are passed through, and at ACTIVITY_END the utterance becomes a
  user turn for the LLM.
* realtimeInput.video  -> latest frame, attached to every LLM turn.
* clientContent        -> a text user turn (heartbeats, typed text, notices).
* toolResponse         -> function responses; the LLM is called again.
* LLM output           -> serverContent.modelTurn text parts (spoken by our
  TTS, as for the text-only ER model), toolCall messages, and a final
  serverContent.turnComplete with usageMetadata.

The conversation lives on the client, so a reconnect (new stream) keeps it.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Any, Callable

from model import generate_agent
from model import live_api_client

logger = logging.getLogger(__name__)

ASR_MODEL = "gemini-3.5-transcribe-live"
# Tools whose response must not wake the model again (Live: SILENT).
_IDLE_TOOLS = frozenset({"ack"})
_MAX_ROUNDS = 8
_TOOL_RESPONSE_TIMEOUT_S = 120.0
# Grace after ACTIVITY_END for trailing transcription chunks.
_UTTERANCE_FLUSH_S = 0.6
_CJK_GAP = re.compile(r"(?<=[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])\s+(?=[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])")


def _clean_transcript(text: str) -> str:
  return _CJK_GAP.sub("", " ".join(text.split())).strip()


class CascadeStream:
  """Live-compatible stream backed by ASR + generateContent."""

  def __init__(self, client: "CascadeClient"):
    self._client = client
    self._on_message: Callable[[dict], None] | None = None
    self._on_done: Callable[[], None] | None = None
    self._events: queue.Queue = queue.Queue()
    self._worker: threading.Thread | None = None
    self._asr = None
    self._asr_lock = threading.Lock()
    self._utterance: list[str] = []
    self._flush_timer: threading.Timer | None = None
    self._latest_frame: str | None = None
    self._done_called = False
    self._asr_failures = 0
    self._ready_emitted = False
    # GeminiLiveApiStream-compatible diagnostics.
    self.close_code: int | None = None
    self.close_reason: str = ""
    self.closed = False
    self.closed_locally = False
    self.generation = 0
    self.model_resource_prefix = None

  # ---- stream interface ----

  def Start(self, on_message, on_done) -> None:  # pylint: disable=invalid-name
    self._on_message = on_message
    self._on_done = on_done
    self._worker = threading.Thread(target=self._run, daemon=True)
    self._worker.start()

  def Send(self, msg: Any) -> None:  # pylint: disable=invalid-name
    if self.closed or not isinstance(msg, dict):
      return
    if "setup" in msg:
      self._configure(msg["setup"])
      return
    rt = msg.get("realtimeInput")
    if isinstance(rt, dict):
      if "video" in rt:
        self._latest_frame = (rt.get("video") or {}).get("data")
      if "audio" in rt or "audioStreamEnd" in rt:
        asr = self._asr
        if asr is not None and not asr.closed:
          try:
            asr.Send(msg)
          except Exception:  # pylint: disable=broad-except
            pass
      return
    cc = msg.get("clientContent")
    if isinstance(cc, dict):
      texts = []
      for turn in cc.get("turns") or []:
        for part in turn.get("parts") or []:
          if part.get("text"):
            texts.append(part["text"])
      if texts:
        self._events.put(("text", "\n".join(texts)))
      return
    tr = msg.get("toolResponse")
    if isinstance(tr, dict):
      self._events.put(("tool_response", tr.get("functionResponses") or []))

  def HalfClose(self) -> None:  # pylint: disable=invalid-name
    self.Shutdown()

  def GetStatus(self) -> Any:  # pylint: disable=invalid-name
    return None

  def Shutdown(self) -> None:  # pylint: disable=invalid-name
    if self.closed:
      return
    self.closed_locally = True
    self._close()

  # ---- internals ----

  def _emit(self, msg: dict) -> None:
    if self._on_message is not None and not self.closed:
      try:
        self._on_message(msg)
      except Exception:  # pylint: disable=broad-except
        logger.exception("cascade on_message failed")

  def _close(self, reason: str = "") -> None:
    if self.closed:
      return
    self.closed = True
    self.close_reason = self.close_reason or reason
    self._events.put(("shutdown", None))
    asr = self._asr
    if asr is not None:
      try:
        asr.Shutdown()
      except Exception:  # pylint: disable=broad-except
        pass
    if not self._done_called and self._on_done is not None:
      self._done_called = True
      try:
        self._on_done()
      except Exception:  # pylint: disable=broad-except
        logger.exception("cascade on_done failed")

  def _configure(self, setup: dict) -> None:
    agent = self._client.agent
    si = ((setup.get("systemInstruction") or {}).get("parts") or [{}])[0].get("text", "")
    agent.system_instruction = si
    agent.tools = generate_agent.strip_declarations(setup.get("tools") or [])
    gen = {}
    thinking = (setup.get("generationConfig") or {}).get("thinkingConfig")
    if thinking:
      if "thinkingBudget" in thinking:
        gen["thinkingConfig"] = {
            "thinkingBudget": thinking["thinkingBudget"],
        }
      else:
        gen["thinkingConfig"] = {
            "thinkingLevel": thinking.get("thinkingLevel", "LOW"),
        }
    agent.generation_config = gen
    threading.Thread(target=self._open_asr, daemon=True).start()

  def _asr_client(self):
    """Vertex AI (global, ADC) first; AI Studio key as fallback."""
    if self._client.asr_backend == "vertex":
      try:
        return live_api_client.VertexLiveApiClient(
            project=os.environ.get("VERTEX_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("VERTEX_ASR_LOCATION", "global"),
        ), "vertex"
      except Exception as e:  # pylint: disable=broad-except
        logger.warning("Cascade ASR: Vertex unavailable (%s); using AI Studio", e)
        self._client.asr_backend = "aistudio"
    return live_api_client.GeminiLiveApiClient(api_key=self._client.api_key), "aistudio"

  def _open_asr(self) -> None:
    try:
      client, backend = self._asr_client()
      asr = client.create_stream()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("Cascade ASR connect failed: %s", e)
      if self._client.asr_backend == "vertex":
        self._client.asr_backend = "aistudio"
        threading.Timer(0.2, self._open_asr).start()
        return
      self._close(f"ASR connect failed: {e}")
      return
    ready = threading.Event()

    def on_msg(m: dict) -> None:
      if "setupComplete" in m:
        ready.set()
        return
      self._on_asr_message(m)

    def on_done() -> None:
      if self.closed:
        return
      with self._asr_lock:
        if self._asr is asr:
          self._asr = None
      # The transcription socket times out on its own (~1 min idle, "The
      # operation was aborted"); reopen it quietly - the LLM conversation and
      # the session are unaffected.
      self._asr_failures += 1
      if self._asr_failures > 5:
        logger.error("Cascade ASR keeps failing: %s", asr.close_reason)
        self.close_code = asr.close_code
        self._close(f"ASR stream ended: {asr.close_reason}")
        return
      logger.info("Cascade ASR ended (%s); reopening", asr.close_reason)
      threading.Timer(0.5 * self._asr_failures, self._open_asr).start()

    asr.Start(on_msg, on_done)
    # Vertex requires inputAudioTranscription for the ASR model (1007 else).
    asr.Send({"setup": {"model": f"models/{self._client.asr_model}",
                        "inputAudioTranscription": {}}})
    if not ready.wait(10.0):
      logger.error("Cascade ASR (%s): no setupComplete (%s)", backend,
                   asr.close_reason)
      asr.Shutdown()
      if backend == "vertex":
        self._client.asr_backend = "aistudio"
        threading.Timer(0.2, self._open_asr).start()
        return
      self._close("ASR setup timeout")
      return
    with self._asr_lock:
      self._asr = asr
    first = not self._ready_emitted
    self._ready_emitted = True
    if first:
      logger.info(
          "Cascade ready: ASR=%s (%s) -> LLM=%s", self._client.asr_model,
          backend, self._client.agent.model,
      )
      self._emit({"setupComplete": {}})
    else:
      threading.Timer(30.0, self._reset_asr_failures, args=(asr,)).start()

  def _reset_asr_failures(self, asr) -> None:
    if self._asr is asr and not asr.closed:
      self._asr_failures = 0

  def _on_asr_message(self, m: dict) -> None:
    va = m.get("voiceActivity")
    if isinstance(va, dict):
      kind = va.get("type") or va.get("voiceActivityType")
      if kind == "ACTIVITY_START":
        if self._flush_timer is not None:
          self._flush_timer.cancel()
          self._flush_timer = None
      elif kind == "ACTIVITY_END":
        self._flush_timer = threading.Timer(_UTTERANCE_FLUSH_S, self._flush_utterance)
        self._flush_timer.daemon = True
        self._flush_timer.start()
      self._emit({"voiceActivity": va, "serverContent": m.get("serverContent") or {}})
      return
    sc = m.get("serverContent") or {}
    it = (sc.get("inputTranscription") or {}).get("text")
    if it:
      self._utterance.append(it)
      self._emit({"serverContent": {"inputTranscription": {"text": it}}})
    # generationComplete / turnComplete of the ASR model are not LLM turns.

  def _flush_utterance(self) -> None:
    text = _clean_transcript(" ".join(self._utterance))
    self._utterance = []
    if text:
      logger.info("Cascade user utterance: %s", text)
      self._events.put(("text", text))

  def _frame_parts(self) -> list[dict]:
    frame = self._latest_frame
    return [generate_agent.image_part(frame)] if frame else []

  def _run(self) -> None:
    deferred: list[tuple[str, Any]] = []
    while True:
      ev = deferred.pop(0) if deferred else self._events.get()
      kind, payload = ev
      if kind == "shutdown":
        return
      if kind == "text":
        try:
          self._run_turn(payload, deferred)
        except Exception:  # pylint: disable=broad-except
          logger.exception("cascade turn failed")
          self._emit({"serverContent": {"turnComplete": True}})
      # Stray tool responses outside a turn are ignored.

  def _run_turn(self, text: str, deferred: list) -> None:
    agent = self._client.agent
    start = len(agent.history)
    is_heartbeat = text.lstrip().startswith("[HEARTBEAT]")
    agent.add("user", [{"text": text}] + self._frame_parts())
    spoke = False
    all_idle = True
    usage: dict = {}
    for _ in range(_MAX_ROUNDS):
      out = agent.generate()
      if self.closed:
        return
      usage = out.get("usage") or usage
      calls = []
      for part in out["parts"]:
        if "functionCall" in part:
          calls.append(dict(part["functionCall"]))
        elif part.get("text"):
          if part.get("thought"):
            self._emit({"serverContent": {"modelTurn": {"parts": [
                {"text": part["text"], "thought": True}]}}})
          else:
            spoke = True
            self._emit({"serverContent": {"modelTurn": {"parts": [
                {"text": part["text"]}]}}})
      if not calls:
        break
      for i, fc in enumerate(calls):
        fc.setdefault("id", f"cascade_{int(time.time() * 1000)}_{i}")
        fc.setdefault("args", {})
      if any(fc.get("name") not in _IDLE_TOOLS for fc in calls):
        all_idle = False
      self._emit({"toolCall": {"functionCalls": calls}})
      responses = self._await_responses(
          {fc["id"]: fc["name"] for fc in calls}, deferred
      )
      if responses is None:
        return
      agent.add_tool_responses(responses, self._frame_parts())
      if all(fc.get("name") in _IDLE_TOOLS for fc in calls):
        break
    self._emit({
        "serverContent": {"turnComplete": True},
        "usageMetadata": generate_agent.live_usage(usage),
    })
    if is_heartbeat and all_idle and not spoke:
      # Idle heartbeat -> ack: keep it out of the context.
      agent.truncate(start)

  def _await_responses(self, pending: dict[str, str], deferred: list):
    got: dict[str, dict] = {}
    deadline = time.monotonic() + _TOOL_RESPONSE_TIMEOUT_S
    while len(got) < len(pending):
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        break
      try:
        kind, payload = self._events.get(timeout=remaining)
      except queue.Empty:
        break
      if kind == "shutdown":
        return None
      if kind != "tool_response":
        deferred.append((kind, payload))
        continue
      for fr in payload:
        fid = str(fr.get("id", ""))
        if fid in pending:
          got[fid] = {
              "functionResponse": {
                  "id": fid,
                  "name": fr.get("name") or pending[fid],
                  "response": fr.get("response") or {},
              }
          }
    for fid, name in pending.items():
      if fid not in got:
        logger.warning("Cascade: no tool response for %s (%s)", name, fid)
        got[fid] = {"functionResponse": {
            "id": fid, "name": name,
            "response": {"error": "no response (timeout)"}}}
    return [got[fid] for fid in pending]


class CascadeClient:
  """Creates CascadeStreams; owns the conversation so reconnects keep it."""

  def __init__(self, model: str, api_key: str | None, asr_model: str = ASR_MODEL):
    if not api_key:
      raise ValueError("api_key is required for the cascade model.")
    self.api_key = api_key
    self.asr_model = asr_model
    vertex_ok = bool(os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    self.asr_backend = (
        "vertex"
        if vertex_ok and os.environ.get("LIVE_BACKEND", "").lower() != "aistudio"
        else "aistudio"
    )
    self.agent = generate_agent.GenerateContentAgent(model=model, api_key=api_key)
    logger.info("CascadeClient: ASR=%s -> LLM=%s (+ server TTS)", asr_model, model)

  def create_stream(self) -> CascadeStream:
    return CascadeStream(self)
