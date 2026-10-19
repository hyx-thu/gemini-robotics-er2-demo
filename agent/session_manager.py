r"""Manages a Gemini Live bidirectional streaming session (Lite version).

This is a simplified version of the session manager that uses only
the public Gemini Live API (WebSocket + JSON) with zero external dependencies.
"""

import asyncio
import base64
import datetime
import json
import logging
import os
import re
import time
from typing import Any, Mapping

from core import audio_handler
from core import decision_making
from core import event_bus
from core import observation
from core import tool_call_handler
from model import live_api_client


logger = logging.getLogger(__name__)

# Lowest allowed spacing between two heartbeat turns, for every model/mode.
MIN_HEARTBEAT_INTERVAL_S = 3.0

_REAL_EMBODIMENT_ACTION_TOOLS = frozenset({
    "detect",
    "pick",
    "place",
    "look",
    "drive",
    "navigate",
    "stow",
    "wait_for_pick_up",
    "stop",
    "confirm_action",
})

_ACTION_INTENT_RE = re.compile(
    r"(把|拿|抓|取|摘|放|夹|移|递|捡|收|扔|打开|合上|笔|本|笔记本|杯|水杯|盒|"
    r"眼镜|手机|纸|球|块|桌面|桌|工具|调用|试一下|演示|pick|place|grab|put|"
    r"move|take|bring|lift|drop|detect|look|clip|pen|notebook|cup|mug|"
    r"glasses|phone|box|block)",
    re.IGNORECASE,
)

_CHITCHAT_ONLY_RE = re.compile(
    r"^(你好|嗨|哈喽|喂|可以听到吗|能听到吗|听得到吗|在吗|你是谁|天气|几号|几点|"
    r"hello|hi|hey|can you hear me|how are you)[？?！!。.,\s]*$",
    re.IGNORECASE,
)

_UNFULFILLED_ACTION_REPLY_RE = re.compile(
    r"(我来看一下|让我看看|我看一下|您可以|你可以|双手|先打开|将笔|把笔|对准|"
    r"穿过线圈|展示的只是一个图像|没有实体|无法替您|let me look|let me see|"
    r"you can place|you can clip|i am an ai)",
    re.IGNORECASE,
)


def _is_action_request_text(text: str) -> bool:
  """Return True if user text requests physical manipulation, inspection, or tool use."""
  cleaned = (text or "").strip()
  if not cleaned or cleaned.startswith("[HEARTBEAT]") or cleaned == "🎤 正在识别语音…":
    return False
  if _CHITCHAT_ONLY_RE.match(cleaned):
    return False
  return bool(_ACTION_INTENT_RE.search(cleaned))


def _is_unfulfilled_action_reply(text: str) -> bool:
  """Return True if assistant reply promised to look or gave human DIY instructions without calling tools."""
  cleaned = (text or "").strip()
  if not cleaned:
    return False
  return bool(_UNFULFILLED_ACTION_REPLY_RE.search(cleaned))


# ---- User language tracking (reply-language lock) ---------------------------

_LANGUAGE_NAMES = {
    "zh": "Mandarin Chinese (简体中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "ru": "Russian (русский)",
    "ar": "Arabic (العربية)",
    "hi": "Hindi (हिन्दी)",
    "th": "Thai (ไทย)",
    "en": "English",
}

# Function words that are near-unique to each major Latin-script language.
_LATIN_LANG_MARKERS = {
    "en": r"\b(the|and|is|are|you|with|this|that|have|will|can|please|it's|i'm)\b",
    "de": r"\b(und|ist|nicht|das|die|der|ich|sie|wir|ein|eine|mit|für|auf|können|bitte)\b|[äöüß]",
    "es": r"\b(el|la|los|las|es|está|para|con|por|una|uno|puedo|gracias|aquí|qué)\b|[ñ¿¡]",
    "fr": r"\b(le|la|les|est|pour|avec|une|vous|nous|je|c'est|merci|voici)\b|[àâçéèêëîïôûùüÿœ]",
    "pt": r"\b(o|a|os|as|é|para|com|uma|você|não|obrigado|aqui|está)\b|[ãõç]",
    "it": r"\b(il|lo|gli|le|è|per|con|una|sono|questo|grazie|ecco)\b",
}


def detect_language(text: str) -> str | None:
  """Best-effort language code from script / function words; None if unknown."""
  s = (text or "").strip()
  if not s:
    return None
  if re.search(r"[\u3040-\u30ff]", s):
    return "ja"
  if re.search(r"[\uac00-\ud7af]", s):
    return "ko"
  if re.search(r"[\u4e00-\u9fff]", s):
    return "zh"
  if re.search(r"[\u0400-\u04ff]", s):
    return "ru"
  if re.search(r"[\u0600-\u06ff]", s):
    return "ar"
  if re.search(r"[\u0900-\u097f]", s):
    return "hi"
  if re.search(r"[\u0e00-\u0e7f]", s):
    return "th"
  latin_letters = len(re.findall(r"[A-Za-zÀ-ÿ]", s))
  if latin_letters < 6:
    return None
  lower = s.lower()
  best, best_hits = None, 0
  for code, pat in _LATIN_LANG_MARKERS.items():
    hits = len(re.findall(pat, lower))
    if hits > best_hits:
      best, best_hits = code, hits
  if best is None:
    return "en" if latin_letters >= 12 else None
  return best


def language_display_name(code: str | None) -> str | None:
  if not code:
    return None
  return _LANGUAGE_NAMES.get(code, code)


def reply_language_mismatch(user_lang: str | None, reply: str) -> str | None:
  """Return the detected reply language if it clearly differs from user_lang."""
  if not user_lang:
    return None
  r = (reply or "").strip()
  # Need enough signal: ignore short acks / numbers / tool echoes.
  if len(re.findall(r"[A-Za-zÀ-ÿ\u0400-\u04ff\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]", r)) < 8:
    return None
  reply_lang = detect_language(r)
  if reply_lang is None or reply_lang == user_lang:
    return None
  # Chinese replies legitimately contain English tool / coordinate tokens.
  if user_lang == "zh" and re.search(r"[\u4e00-\u9fff]", r):
    return None
  # Japanese text contains kanji; a Chinese user reply with kana only is a mismatch.
  if user_lang == "ja" and re.search(r"[\u3040-\u30ff]", r):
    return None
  return reply_lang


def _strip_tool_behavior(tool: dict) -> dict:
  """Return a deep-ish copy of a tool group with 'behavior' removed."""
  if "functionDeclarations" not in tool:
    return {k: v for k, v in tool.items() if k != "behavior"}
  return {
      **tool,
      "functionDeclarations": [
          {k: v for k, v in decl.items() if k != "behavior"}
          for decl in tool["functionDeclarations"]
      ],
  }


def get_generic_heartbeat_text() -> str:
  """Heartbeat text for models NOT trained on the ack protocol.

  Spells out that the tick is machine-generated and that anything the user
  said since the last tick must be handled first, so the model does not fall
  into answering the user with ``ack``.
  """
  return (
      "[HEARTBEAT] Automatic timer tick from the control loop - this is NOT"
      " the user speaking. PRIORITY ORDER: (1) If the user has said or typed"
      " anything since the last tick that you have not acted on yet, handle"
      " it NOW: call the required action tools (detect/pick/drive/look/...)"
      " and answer the user by voice. Do NOT call 'ack' in that case."
      " (2) Otherwise, if a task is active, observe the camera and continue"
      " the current step (tool call) or call 'ack' if it is still in"
      " progress. (3) Otherwise call 'ack' exactly once and end your turn."
  )


def get_default_heartbeat_text() -> str:
  """Returns the default heartbeat text to send to the model."""
  return (
      "[HEARTBEAT] If no task is active, call 'ack' exactly once and end"
      " your turn; you will receive another heartbeat shortly. If a task is"
      " active: observe the scene. If the current step is progressing"
      " correctly, call 'ack' once. If the current step is complete, call"
      " 'run_instruction' with the next step. If the overall goal is"
      " achieved, call 'reset' and inform the user."
  )


class SessionManager:
  """Manages a Gemini Live bidirectional streaming session.

  Uses an EventBus for typed pub/sub event dispatching. Components register
  as self-subscribing handlers; the session manager bridges UI-relevant
  events to the WebSocket via a ui_queue.
  """

  _DEFAULT_SYSTEM_INSTRUCTION = (
      "You are a helpful AI assistant. Keep your responses"
      " concise. You can see the user's camera or screen"
      " which is shared as realtime input images with you."
  )

  def __init__(
      self,
      model: str,
      embodiment_instance,
      input_sample_rate: int = 16000,
      tools=None,
      system_instruction: str | None = None,
      developer_instruction: str | None = None,
      response_modality: str = "AUDIO",
      dump_video_dir: str | None = None,
      api_key: str | None = None,
      heartbeat_interval_seconds: float = 2.0,
      heartbeat_enabled: bool = True,
      heartbeat_min_delay_seconds: float = 0.0,
      agent_peers: dict[str, str] | None = None,
      peer_name: str = "unknown",
      use_event_driven_heartbeat: bool = False,
      media_resolution: str = "low",
      heartbeat_text: str = "",
      enable_send_message_to_user: bool = False,
      endpoint_type: str = "gemini_live_api",
      thinking_level: str = "none",
      supports_blocking_tools: bool = True,
      heartbeat_only_when_task_active: bool = False,
      er_orchestrator_config: dict[str, Any] | None = None,
  ):
    self.model = model
    # Composite "Live + ER2" mode: config for the ER robot-task orchestrator
    # (model, tools, system_instruction, thinking_level); built in
    # start_session once the ToolCallHandler exists.
    self._er_orchestrator_config = er_orchestrator_config
    self.er_orchestrator = None
    # Opt-in dataset recorder (core/session_recorder.py), toggled from the UI.
    self.recorder = None
    self.recording_meta: dict[str, Any] = {}
    self.supports_blocking_tools = supports_blocking_tools
    # For models not trained on the ack protocol: skip heartbeats while no
    # task is active (idle ticks only teach them to answer with ack / to
    # hallucinate a stuck task). Heartbeats resume as soon as a plan starts
    # or an action awaits confirmation, so the observe->act loop still runs.
    self.heartbeat_only_when_task_active = heartbeat_only_when_task_active
    self._idle_skip_logged = False
    self.embodiment = embodiment_instance
    self.input_sample_rate = input_sample_rate
    self.tools = tools or []
    self.system_instruction = (
        system_instruction or self._DEFAULT_SYSTEM_INSTRUCTION
    )
    self.developer_instruction = developer_instruction or ""
    self.thinking_level = thinking_level
    self.response_modality = response_modality
    self.dump_video_dir = dump_video_dir
    self.heartbeat_interval_seconds = heartbeat_interval_seconds
    self.heartbeat_enabled = heartbeat_enabled
    self.heartbeat_min_delay_seconds = heartbeat_min_delay_seconds
    self._agent_peers = agent_peers or {}
    self._peer_name = peer_name
    self.use_event_driven_heartbeat = use_event_driven_heartbeat
    self.heartbeat_text = heartbeat_text
    self._enable_send_message_to_user = enable_send_message_to_user
    self._last_heartbeat_sent_time = 0.0
    self._interrupted = False

    self._heartbeat_sent_count = 0
    self._turn_completed_count = 0
    self._heartbeat_in_flight = False

    self.api_key = api_key
    self._endpoint_type = endpoint_type

    # Create the API client.
    # Vertex AI (OAuth) for gemini-3.8-live; Developer API key for the rest.
    self.client = live_api_client.make_client(model, api_key)
    self.stream = None
    self._stream_lock = asyncio.Lock()
    # --- Stream health (see _stream_health_loop / _reconnect) ---
    # Monotonic time of the last message received from the Live API.
    self._last_server_msg_at = 0.0
    # Latest resumable session handle announced by the server; used to
    # resume the conversation (plan, history) when the socket is replaced.
    self._resumption_handle: str | None = None
    self._setup_complete = asyncio.Event()
    self._ui_queue: asyncio.Queue | None = None
    self._reconnect_count = 0
    # Incremented for every socket we open. Callbacks from a superseded
    # socket (closed during a reconnect) carry the old number and are
    # ignored, so a dying old stream can never tear down its replacement.
    self._stream_generation = 0
    self.loop = asyncio.get_running_loop()

    # Create the event bus.
    self.bus = event_bus.EventBus()

    # Initialize Observation early to expose queues.
    self.observation = observation.Observation(
        self.bus,
        self.loop,
        self,  # SessionManager provides send_message for stream I/O
        0,  # Session start time not available yet
        self.embodiment,
        self.input_sample_rate,
        self.dump_video_dir,
        media_resolution=media_resolution,
    )
    self.decision_making = None
    self.tool_handler = None
    self.audio_handler = None
    # Latest browser geolocation fix ({latitude, longitude, accuracy,
    # timestamp}); consumed by get_current_location / google_maps_search.
    self.client_geolocation: dict[str, Any] | None = None
    # Language the user is currently speaking/typing (code + display name),
    # updated from every transcript; used to pin the model's reply language.
    self.user_language: str | None = None
    self.user_language_name: str | None = None
    self._last_user_input_time = 0.0

  def set_recording(self, enabled: bool) -> dict[str, Any]:
    """Start/stop recording this session to the private dataset bucket."""
    from core import session_recorder  # pylint: disable=g-import-not-at-top
    if enabled:
      if self.recorder is not None and not self.recorder.closed:
        return {"state": "on", **self.recorder.status()}
      bucket = session_recorder.recording_bucket()
      if not bucket:
        return {"state": "unavailable",
                "reason": "RECORDING_BUCKET is not configured on this server"}
      meta = {"model": self.model, "modality": self.response_modality,
              "thinking_level": self.thinking_level,
              "revision": os.environ.get("K_REVISION", "local"),
              **self.recording_meta}
      self.recorder = session_recorder.SessionRecorder(
          bucket, meta, sample_rate=self.input_sample_rate)
      plan = getattr(self.tool_handler, "_plan", None)
      if plan:
        self.recorder.event("plan_snapshot", plan)
      return {"state": "on", **self.recorder.status()}
    if self.recorder is None:
      return {"state": "off"}
    status = self.recorder.close()
    self.recorder = None
    return {"state": "off", **status}

  async def _record_bus_event(self, event: event_bus.Event) -> None:
    rec = self.recorder
    if rec is None:
      return
    data = event.data
    if isinstance(data, dict):
      data = {k: v for k, v in data.items() if k not in ("action_data",)}
    elif isinstance(data, (bytes, bytearray)):
      return
    rec.event(event.type.name.lower(), data)

  def note_user_language(self, text: str) -> None:
    """Update the tracked user language from a transcript / typed text."""
    code = detect_language(text)
    if code and code != self.user_language:
      logger.info("User language switched: %s -> %s", self.user_language, code)
    if code:
      self.user_language = code
      self.user_language_name = language_display_name(code)
    self._last_user_input_time = time.monotonic()

  def _language_lock_line(self) -> str:
    if not self.user_language_name:
      return ""
    return (
        f"\n[LANGUAGE LOCK] The user speaks {self.user_language_name}. Anything"
        f" you say or write MUST be in {self.user_language_name} only (this"
        " tick is English for system reasons, not because the user switched)."
    )

  def _get_heartbeat_text(self) -> str:
    if self.heartbeat_text:
      base = self.heartbeat_text
    else:
      base = (
          getattr(self.embodiment, "heartbeat_text", None)
          or get_default_heartbeat_text()
      )
    # Action watchdog: tell the model what command is awaiting visual
    # confirmation, when to re-issue it, or that the task was aborted.
    handler = self.tool_handler
    context = handler.heartbeat_context() if handler is not None else ""
    if context:
      base = f"{base}\n{context}"
    return base + self._language_lock_line()

  # ---- Realtime video throttle (TPM quota) ---------------------------------

  # Each 1 FPS webcam frame costs ~260 prompt tokens *per turn it stays in
  # context*; while the agent is idle that is pure quota burn. Frames are
  # therefore throttled when nothing is happening and streamed at full rate
  # while the user is talking, a task is running or an action was just sent.
  _IDLE_VIDEO_PERIOD_S = 3.0
  _IDLE_VIDEO_AFTER_S = 8.0

  def video_min_period(self) -> float:
    """Minimum seconds between realtime frames right now (0 = full rate)."""
    now = time.monotonic()
    dm = self.decision_making
    if dm is not None:
      if dm.user_speaking.is_set() or dm.model_generating.is_set():
        return 0.0
      if dm.last_user_speech_end and now - dm.last_user_speech_end < self._IDLE_VIDEO_AFTER_S:
        return 0.0
    if self._last_user_input_time and now - self._last_user_input_time < self._IDLE_VIDEO_AFTER_S:
      return 0.0
    th = self.tool_handler
    if th is not None and (th.task_active() or th.tool_executing.is_set()):
      return 0.0
    last_action = getattr(self, "_last_action_turn_time", 0.0)
    if last_action and self.loop.time() - last_action < self._IDLE_VIDEO_AFTER_S:
      return 0.0
    return self._IDLE_VIDEO_PERIOD_S

  def set_embodiment(self, embodiment):
    """Updates the embodiment dynamically."""
    self.embodiment = embodiment
    self.observation.set_embodiment(embodiment)
    if self.tool_handler:
      self.tool_handler.set_embodiment(embodiment)

  async def send_message(self, msg: Any) -> None:
    """Send a JSON message to the Gemini Live API stream.

    Serializes all writes through an asyncio.Lock to prevent concurrent
    stream.Send calls from different tasks.
    """
    assert self.stream is not None, "Stream is not initialized"
    async with self._stream_lock:
      await asyncio.get_running_loop().run_in_executor(
          None, self.stream.Send, msg
      )

  async def send_text_with_fresh_video(self, text: str) -> None:
    """Atomically send a current Spot frame followed by user text."""
    poller = getattr(self.embodiment, "poller", None)
    chunk = b""
    if poller is not None:
      try:
        chunk = await asyncio.wait_for(
            poller.wait_for_next_frame(), timeout=2.0
        )
      except asyncio.TimeoutError:
        logger.warning("Timed out waiting for a pre-text camera frame")

    messages = []
    if chunk:
      await self.observation.dump_frame(chunk)
      messages.append({
          "realtimeInput": {
              "video": {
                  "mimeType": "image/jpeg",
                  "data": base64.b64encode(chunk).decode("utf-8"),
              }
          }
      })
    messages.append({
        "clientContent": {
            "turns": [{
                "role": "user",
                "parts": [{"text": text}],
            }],
            "turnComplete": True,
        }
    })

    assert self.stream is not None, "Stream is not initialized"
    async with self._stream_lock:
      for message in messages:
        await asyncio.get_running_loop().run_in_executor(
            None, self.stream.Send, message
        )
    if chunk:
      logger.info("Sent synchronized fresh video frame with user text")

  async def send_latest_video_frame(self, timeout: float = 2.0) -> bool:
    """Send the first poller frame captured after this method is called."""
    poller = getattr(self.embodiment, "poller", None)
    if poller is None:
      return False
    try:
      chunk = await asyncio.wait_for(
          poller.wait_for_next_frame(), timeout=timeout
      )
    except asyncio.TimeoutError:
      logger.warning("Timed out waiting for a post-tool camera frame")
      return False
    if not chunk:
      logger.warning("Camera poller stopped before producing a post-tool frame")
      return False

    await self.observation.dump_frame(chunk)
    await self.send_message({
        "realtimeInput": {
            "video": {
                "mimeType": "image/jpeg",
                "data": base64.b64encode(chunk).decode("utf-8"),
            }
        }
    })
    logger.info("Sent synchronized fresh video frame")
    return True

  def get_audio_queue(self) -> asyncio.Queue:
    return self.observation.get_audio_queue()

  def get_video_queue(self) -> asyncio.Queue:
    return self.observation.get_video_queue()

  def get_text_queue(self) -> asyncio.Queue:
    return self.observation.get_text_queue()

  def _on_message(self, msg, generation: int | None = None):
    """Callback for incoming Gemini stream messages (JSON dicts)."""
    if msg is None:
      return
    if generation is not None and generation != self._stream_generation:
      # A socket we already replaced is still draining; nothing it says
      # applies to the current session anymore.
      logger.debug("Dropping message from superseded stream #%d", generation)
      return
    logger.debug("Received message: %s", msg)
    # Runs on the stream's reader thread: plain attribute writes only.
    self._last_server_msg_at = time.monotonic()
    if isinstance(msg, dict):
      update = msg.get("sessionResumptionUpdate")
      if isinstance(update, dict) and update.get("resumable"):
        handle = update.get("newHandle")
        if handle:
          self._resumption_handle = handle
      if "setupComplete" in msg:
        self.loop.call_soon_threadsafe(self._setup_complete.set)
    self.loop.call_soon_threadsafe(
        self.bus.publish_nowait,
        event_bus.Event(
            type=event_bus.EventType.MODEL_RESPONSE,
            source=event_bus.EventSource.ASSISTANT,
            data=msg,
        ),
    )

  def _on_done(self, generation: int | None = None):
    """Callback when the Gemini stream ends."""
    if generation is not None and generation != self._stream_generation:
      logger.info("[Session] superseded stream #%d closed", generation)
      return
    if getattr(self, "_closing", False):
      logger.info("[Session] stream #%s closed at session end", generation)
      return
    logger.error(
        "[Session] on_done fired — stream #%s disconnected", generation
    )
    self.loop.call_soon_threadsafe(
        self.bus.publish_nowait,
        event_bus.Event(
            type=event_bus.EventType.SESSION_DONE,
            source=event_bus.EventSource.ASSISTANT,
            data={"generation": generation},
        ),
    )

  async def _start_stream(self):
    """Open a new Live API socket, make it current and start its reader.

    The TLS/WebSocket handshake blocks, so it runs off the event loop.
    """
    stream = await asyncio.get_running_loop().run_in_executor(
        None, self.client.create_stream
    )
    self._stream_generation += 1
    gen = self._stream_generation
    stream.generation = gen
    self.stream = stream
    stream.Start(
        lambda msg: self._on_message(msg, gen),
        lambda: self._on_done(gen),
    )
    return stream

  async def _send_setup(self, resume_handle: str | None = None) -> None:
    """Sends the JSON setup message to the Live API stream.

    Args:
      resume_handle: when set, asks the server to resume the previous
        session (conversation history, plan state) on this new socket.
    """
    # Vertex AI needs projects/…/models/X; GeminiLiveApiStream.Send rewrites
    # "models/X" to the full resource name, so keep the short form here.
    model_name = live_api_client.wire_model(self.model)
    if not model_name.startswith("models/"):
      model_name = f"models/{model_name}"

    # Combine SI and DI into a single system instruction.
    combined_prompt = self.system_instruction
    if self.developer_instruction:
      combined_prompt += "\n\n" + self.developer_instruction

    modality_str = self.response_modality.upper()
    if modality_str == "AUDIO":
      modality = "AUDIO"
    elif modality_str == "TEXT":
      modality = "TEXT"
    else:
      raise ValueError(f"Unsupported response modality: {self.response_modality}")

    setup_dict: dict[str, Any] = {
        "setup": {
            "model": model_name,
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": combined_prompt}]
            },
            "generationConfig": {
                "responseModalities": [modality]
            },
            "inputAudioTranscription": {},
            "tools": []
        }
    }
    if modality == "AUDIO":
      setup_dict["setup"]["outputAudioTranscription"] = {}

    # Add thinking config if a level is set. includeThoughts streams the
    # model's reasoning summaries back so the UI can render them.
    # For default-thinking models (Robotics-ER, Argon), omitting thinkingConfig
    # when thinking_level == "none" leaves the model's built-in dynamic thinking
    # budget (~400-600 thought tokens) enabled; explicitly pass thinkingBudget=0
    # to disable thinking.
    if self.thinking_level and self.thinking_level != "none":
      setup_dict["setup"]["generationConfig"]["thinkingConfig"] = {
          "thinkingLevel": self.thinking_level.upper(),
          "includeThoughts": True,
      }
    elif self.thinking_level == "none" and (
        "robotics" in model_name or "argon" in model_name
    ):
      setup_dict["setup"]["generationConfig"]["thinkingConfig"] = {
          "thinkingBudget": 0,
      }

    # Add tools. For models that reject "behavior": "BLOCKING" (1007), send
    # the declarations without the field; self.tools is left untouched so the
    # ToolCallHandler keeps its blocking scheduling semantics.
    for tool in self.tools:
      setup_dict["setup"]["tools"].append(
          tool if self.supports_blocking_tools else _strip_tool_behavior(tool)
      )

    # Server-side sliding-window context compression. The server drops the
    # OLDEST turns first and keeps the most recent ones (incl. their video
    # frames) up to targetTokens; the system instruction is never dropped.
    # Every turn re-bills the whole context against the per-minute quota, so
    # the window is a trade-off between memory and 1011 quota errors.
    #   ER (~8.6K SI, ~63 tok/frame):  trigger 40K -> keep 28K
    #     => ~19K of recent history ≈ 1 min of task activity.
    #   Live (~9.5-12.6K SI, ~260 tok/frame): trigger 48K -> keep 32K
    #     => ~20K of recent history ≈ 45-60 s.
    if getattr(self, "heartbeat_only_when_task_active", False):
      trigger, target = 48000, 32000
    else:
      # 40K/28K tripped 1011 (per-minute quota) about once a minute on ER.
      trigger, target = 24000, 16000
    trigger = int(os.environ.get("COMPRESSION_TRIGGER_TOKENS", trigger))
    target = int(os.environ.get("COMPRESSION_TARGET_TOKENS", target))
    setup_dict["setup"]["contextWindowCompression"] = {
        "triggerTokens": trigger,
        "slidingWindow": {"targetTokens": min(target, trigger - 1000)},
    }

    if resume_handle:
      setup_dict["setup"]["sessionResumption"] = {"handle": resume_handle}

    logger.info(
        "[Session] Sending setup: model=%s, modality=%s, tools=%d%s",
        model_name,
        modality_str,
        len(self.tools),
        " (resuming previous session)" if resume_handle else "",
    )

    assert self.stream is not None
    self._setup_complete.clear()
    await asyncio.get_running_loop().run_in_executor(
        None, self.stream.Send, setup_dict
    )

  # ---- Stream health & recovery --------------------------------------------

  # Speech started (server VAD) but the server then went completely silent:
  # no ACTIVITY_END, no transcription, not even resumption updates. Seen when
  # the Live API backend wedges before closing with 1011. After this long we
  # replace the socket ourselves instead of leaving the user unheard.
  _SPEECH_STALL_S = 12.0
  # A heartbeat turn normally completes within ~1 s. If the server has sent
  # nothing at all for this long after a heartbeat, the socket is dead.
  _SERVER_SILENCE_S = 30.0
  _SETUP_COMPLETE_TIMEOUT_S = 6.0
  # Longest plausible single utterance. ``user_speaking`` set longer than
  # this without an ACTIVITY_END is treated as stale VAD state.
  _STALE_SPEECH_S = 30.0

  def _notify_ui(self, payload: dict) -> None:
    """Best-effort status notice for the browser (dropped when no UI)."""
    if self._ui_queue is None:
      return
    try:
      self._ui_queue.put_nowait(payload)
    except Exception:  # pylint: disable=broad-except
      pass

  def _reset_turn_state(self) -> None:
    """Forget per-socket state that would otherwise outlive a reconnect.

    The new session has no VAD / generation state, but our flags would keep
    reporting ``user_speaking`` / ``model_generating`` and hold every
    heartbeat (the 10 s "sending anyway" path is defeated by the final
    pre-send hold check), freezing the agent.
    """
    dm = self.decision_making
    if dm is not None:
      if dm.user_speaking.is_set():
        logger.info("Clearing stale user_speaking flag after reconnect")
      dm.user_speaking.clear()
      dm.model_generating.clear()
      dm.last_user_speech_end = None
    self._heartbeat_in_flight = False
    self._interrupted = False

  async def _close_stream(self, stream) -> None:
    """Close a stream off the event loop (websocket close() blocks)."""
    if stream is None:
      return
    try:
      await asyncio.get_running_loop().run_in_executor(None, stream.Shutdown)
    except Exception:  # pylint: disable=broad-except
      pass

  async def _open_stream(self, resume_handle: str | None) -> bool:
    """Create a socket, send setup and wait for setupComplete."""
    # Hold the send lock while swapping sockets: the audio/video sender
    # tasks follow ``self.stream`` dynamically and must not slip a
    # realtimeInput frame onto the new socket before its setup message.
    async with self._stream_lock:
      old = self.stream
      stream = await self._start_stream()
      if old is not None and old is not stream:
        await self._close_stream(old)
      await self._send_setup(resume_handle)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + self._SETUP_COMPLETE_TIMEOUT_S
    while not self._setup_complete.is_set():
      if getattr(stream, "closed", False):
        logger.warning(
            "Stream closed before setupComplete (code=%s %s)%s",
            getattr(stream, "close_code", None),
            getattr(stream, "close_reason", ""),
            " (resume attempt)" if resume_handle else "",
        )
        return False
      remaining = deadline - loop.time()
      if remaining <= 0:
        logger.warning(
            "No setupComplete within %.0fs%s",
            self._SETUP_COMPLETE_TIMEOUT_S,
            " (resume attempt)" if resume_handle else "",
        )
        return False
      try:
        await asyncio.wait_for(
            self._setup_complete.wait(), timeout=min(0.25, remaining)
        )
      except asyncio.TimeoutError:
        pass
    return True

  async def _reconnect(self, max_retries: int = 3) -> bool:
    """Replace the Gemini stream with exponential backoff.

    Tries to resume the previous session first (so the model keeps its
    conversation and plan context) and falls back to a fresh session.
    Returns False when the stream cannot be re-established.
    """
    now = time.monotonic()
    last_rc = getattr(self, '_last_reconnect_time', 0)
    rapid_count = getattr(self, '_rapid_reconnect_count', 0)

    if now - last_rc < 5.0:
      rapid_count += 1
    else:
      rapid_count = 0

    self._last_reconnect_time = now
    self._rapid_reconnect_count = rapid_count

    if rapid_count >= max_retries:
      logger.error("Stream disconnecting too rapidly (%d times). Aborting.", rapid_count)
      self._notify_ui({
          "type": "stream_status", "state": "failed",
          "reason": "stream disconnecting too rapidly",
      })
      return False

    dead = self.stream
    close_code = getattr(dead, "close_code", None)
    close_reason = getattr(dead, "close_reason", "") or ""
    if getattr(dead, "closed_locally", False) and close_code is None:
      close_reason = close_reason or "closed locally (stall detected)"
    self._reconnect_count += 1
    # The old socket is gone: stale speech/turn flags must not survive it.
    self._reset_turn_state()

    quota_exhausted = "exhausted" in close_reason.lower() or "quota" in close_reason.lower()
    if quota_exhausted:
      # A per-minute token quota was tripped. Reconnecting right away and
      # sending a heartbeat would just trip it again; let the window roll.
      self._quota_cooldown_until = time.monotonic() + 45.0
      logger.warning(
          "Per-minute token quota exhausted; entering 45s low-activity"
          " cooldown (idle heartbeats paused, video throttled)."
      )

    for attempt in range(max_retries):
      wait_time = 2**attempt
      if quota_exhausted:
        wait_time = max(wait_time, 6 * (attempt + 1))
      logger.warning(
          "Stream #%s disconnected (code=%s%s). Reconnecting in %ds"
          " (attempt %d/%d)...",
          getattr(dead, "generation", "?"),
          close_code,
          f" {close_reason}" if close_reason else "",
          wait_time,
          attempt + 1,
          max_retries,
      )
      self._notify_ui({
          "type": "stream_status", "state": "reconnecting",
          "attempt": attempt + 1, "max_attempts": max_retries,
          "code": close_code, "reason": close_reason,
      })
      await asyncio.sleep(wait_time)
      handle = self._resumption_handle
      for resume in ([True, False] if handle else [False]):
        try:
          ok = await self._open_stream(handle if resume else None)
        except Exception as e:  # pylint: disable=broad-except
          logger.error(
              "Reconnection attempt %d (%s) failed: %s",
              attempt + 1, "resume" if resume else "fresh", e,
          )
          ok = False
        if ok:
          self._reset_turn_state()
          self._last_server_msg_at = time.monotonic()
          logger.info(
              "Reconnected successfully on attempt %d (%s session).",
              attempt + 1, "resumed" if resume else "fresh",
          )
          self._notify_ui({
              "type": "stream_status", "state": "reconnected",
              "resumed": resume, "attempt": attempt + 1,
          })
          # Re-prime the model with a heartbeat as soon as the cadence allows.
          self._last_heartbeat_sent_time = 0.0
          return True
        if resume:
          # The server did not accept the handle; it is not going to.
          self._resumption_handle = None
    logger.error("All %d reconnection attempts failed.", max_retries)
    self._notify_ui({
        "type": "stream_status", "state": "failed",
        "reason": f"{max_retries} reconnection attempts failed",
    })
    return False

  async def _force_reconnect(self, why: str) -> None:
    """Drop the current socket; the read loop's on_done drives _reconnect."""
    logger.warning("Forcing stream reconnect: %s", why)
    self._notify_ui({"type": "stream_status", "state": "stalled", "reason": why})
    self._reset_turn_state()
    await self._close_stream(self.stream)

  async def _stream_health_loop(self) -> None:
    """Detect a wedged Live API socket and recover from it."""
    try:
      while True:
        await asyncio.sleep(1.0)
        if self.stream is None:
          continue
        now = time.monotonic()
        since_msg = now - (self._last_server_msg_at or now)
        dm = self.decision_making
        if (
            dm is not None
            and dm.user_speaking.is_set()
            and dm.last_user_speech_start is not None
            and now - dm.last_user_speech_start > self._SPEECH_STALL_S
            and since_msg > self._SPEECH_STALL_S
        ):
          await self._force_reconnect(
              f"speech started {now - dm.last_user_speech_start:.0f}s ago but"
              f" the server has been silent for {since_msg:.0f}s"
          )
          await asyncio.sleep(self._SPEECH_STALL_S)
          continue
        if (
            self._heartbeat_in_flight
            and self._last_heartbeat_sent_time
            and since_msg > self._SERVER_SILENCE_S
        ):
          await self._force_reconnect(
              f"no server message for {since_msg:.0f}s after a heartbeat"
          )
          await asyncio.sleep(self._SERVER_SILENCE_S)
    except asyncio.CancelledError:
      pass
    except Exception as e:  # pylint: disable=broad-except
      logger.error("Stream health loop failed: %s", e)

  async def start_session(
      self,
      audio_output_callback,
      audio_interrupt_callback=None,
      text_output_callback=None,
  ):
    self.session_start_time = self.loop.time()
    self.observation.session_start_time = self.session_start_time

    # --- Create the Gemini stream ---
    await self._start_stream()

    # --- Register handlers on the bus ---

    # DecisionMaking: routes MODEL_RESPONSE → typed events
    try:
      ctx_limit = int(os.environ.get("CONTEXT_WINDOW_LIMIT", "0")) or None
    except ValueError:
      ctx_limit = None
    self.decision_making = decision_making.DecisionMaking(
        self.bus,
        text_output_callback=text_output_callback,
        audio_interrupt_callback=audio_interrupt_callback,
        context_window_limit=ctx_limit,
        utterance_audio_provider=self.observation.consume_utterance_pcm,
        refine_audio_transcripts=(self.response_modality.upper() == "TEXT"),
        api_key=self.api_key,
        input_sample_rate=self.input_sample_rate,
    )

    # AudioResponseHandler: accumulates AUDIO_CHUNK → AUDIO_RESPONSE
    self.audio_handler = audio_handler.AudioResponseHandler(
        self.bus,
        audio_output_callback=audio_output_callback,
    )

    # ToolCallHandler: executes TOOL_CALL → TOOL_RESULT
    self.tool_handler = tool_call_handler.ToolCallHandler(
        self.bus,
        self,  # session_manager for send_message
        self.embodiment,
        agent_peers=self._agent_peers,
        peer_name=self._peer_name,
        text_output_callback=text_output_callback,
        enable_send_message_to_user=self._enable_send_message_to_user,
        blocking_tools=tool_call_handler.blocking_tool_names(self.tools),
    )
    if self._er_orchestrator_config:
      from core import er_orchestrator  # pylint: disable=g-import-not-at-top
      cfg = self._er_orchestrator_config
      self.er_orchestrator = er_orchestrator.ErOrchestrator(
          session=self,
          model=cfg["model"],
          api_key=self.api_key,
          system_instruction=cfg.get("system_instruction", ""),
          tools=cfg.get("tools") or [],
          thinking_level=cfg.get("thinking_level", "low"),
      )
      # The ER tools' blocking semantics must be known to the handler too.
      self.tool_handler._blocking_tools = frozenset(  # pylint: disable=protected-access
          self.tool_handler._blocking_tools  # pylint: disable=protected-access
          | tool_call_handler.blocking_tool_names(cfg.get("tools") or [])
      )

    # --- UI queue bridge: events → async generator yield ---

    ui_queue: asyncio.Queue = asyncio.Queue()
    # Lets the reconnect / stall logic post stream_status notices to the UI.
    self._ui_queue = ui_queue

    _UI_EVENT_TYPES = [
        event_bus.EventType.GEMINI_TEXT,
        event_bus.EventType.GEMINI_THOUGHT,
        event_bus.EventType.USER_TRANSCRIPT,
        event_bus.EventType.AUDIO_RESPONSE,
        event_bus.EventType.TOOL_RESULT,
        event_bus.EventType.TURN_COMPLETE,
        event_bus.EventType.TELEMETRY,
        event_bus.EventType.INTERRUPTED,
        event_bus.EventType.ERROR,
        event_bus.EventType.LOG,
        event_bus.EventType.SESSION_DONE,
        event_bus.EventType.TEXT_INPUT,
        event_bus.EventType.TRANSPARENT_HISTORY,
    ]

    self.bus.subscribe(
        [
            event_bus.EventType.USER_TRANSCRIPT,
            event_bus.EventType.GEMINI_TEXT,
            event_bus.EventType.GEMINI_THOUGHT,
            event_bus.EventType.TOOL_RESULT,
            event_bus.EventType.TURN_COMPLETE,
            event_bus.EventType.TEXT_INPUT,
            event_bus.EventType.INTERRUPTED,
        ],
        self._record_bus_event,
    )

    async def _ui_handler(event: event_bus.Event):
      await ui_queue.put(event)

    self.bus.subscribe(_UI_EVENT_TYPES, _ui_handler)

    # --- Send Setup Message ---

    await self._send_setup()
    self._last_server_msg_at = time.monotonic()

    # --- Start input tasks and heartbeat ---

    has_poller = getattr(self.embodiment, "poller", None) is not None
    self.observation.start_input_tasks(
        skip_video=self.use_event_driven_heartbeat and has_poller
    )

    heartbeat_trigger = None
    if self.use_event_driven_heartbeat:
      heartbeat_signal = asyncio.Event()
      _trigger_source = [""]
      loop = asyncio.get_running_loop()
      # If a heartbeat turn gets no turnComplete for this long, assume the
      # turn is stuck and send a recovery heartbeat.
      safety_timeout = 10.0
      # Quiet time required after the user stops speaking before a heartbeat
      # may be injected, so the model answers the user instead of the
      # heartbeat. Reuses the configurable min-delay knob (floor 1 s).
      post_speech_grace = max(1.0, self.heartbeat_min_delay_seconds)
      # After a turn that executed an action tool, wait briefly so that a
      # function-response-driven continuation (NON_BLOCKING tools) can start
      # before we consider sending a heartbeat.
      post_action_settle = 1.0
      self._last_action_turn_time = 0.0
      self._pending_user_action_text = ""
      self._current_turn_assistant_text = ""
      self._pending_action_nudge = ""
      self._action_nudge_retries = 0
      self._language_nudge_count = 0
      if not hasattr(self, "_quota_cooldown_until"):
        self._quota_cooldown_until = 0.0

      async def _on_user_or_assistant_text(event: event_bus.Event):
        if event.type == event_bus.EventType.USER_TRANSCRIPT:
          txt = (
              event.data.get("text", "")
              if isinstance(event.data, dict)
              else str(event.data or "")
          )
          if txt.strip() and txt.strip() != "🎤 正在识别语音…":
            self.note_user_language(txt)
          is_replace = (
              bool(event.data.get("replace"))
              if isinstance(event.data, dict)
              else False
          )
          if is_replace:
            if _is_action_request_text(txt):
              self._pending_user_action_text = txt.strip()
          else:
            combined = f"{self._pending_user_action_text}{txt}".strip()
            if _is_action_request_text(combined) or _is_action_request_text(txt):
              self._pending_user_action_text = combined[-200:]
              self._action_nudge_retries = 0
        elif event.type == event_bus.EventType.TEXT_INPUT:
          txt = str(event.data or "").strip()
          if txt and not txt.startswith("[HEARTBEAT]"):
            self.note_user_language(txt)
            if _is_action_request_text(txt):
              self._pending_user_action_text = txt[:200]
              self._action_nudge_retries = 0
            else:
              self._pending_user_action_text = ""
        elif event.type == event_bus.EventType.GEMINI_TEXT:
          txt = (
              event.data.get("text", "")
              if isinstance(event.data, dict)
              else str(event.data or "")
          )
          self._current_turn_assistant_text = (
              f"{self._current_turn_assistant_text}{txt}"[-300:]
          )

      async def _on_interrupt(event: event_bus.Event):
        if isinstance(event.data, dict) and event.data.get("tts_preempt"):
          return  # Internal TTS pre-emption, not a user barge-in.
        self._interrupted = True

      async def _on_heartbeat_trigger(event: event_bus.Event):
        source = event.type.value
        if event.type == event_bus.EventType.TURN_COMPLETE:
          self._heartbeat_in_flight = False
          had_tool_call = (
              event.data.get("had_tool_call", False)
              if isinstance(event.data, dict)
              else False
          )
          tool_names = (
              list(event.data.get("tool_names", []))
              if isinstance(event.data, dict)
              else []
          )
          interaction_status = (
              str(event.data.get("interaction_status", "") or "")
              if isinstance(event.data, dict)
              else ""
          )
          assistant_reply = self._current_turn_assistant_text
          self._current_turn_assistant_text = ""
          called_embodiment_action = bool(
              set(tool_names) & _REAL_EMBODIMENT_ACTION_TOOLS
          )
          # Reply-language lock: native-audio Live models occasionally drift
          # into German / Spanish / English after a run of English-formatted
          # tool responses and heartbeats. If the spoken reply is clearly not
          # in the user's language, correct it right away (bounded so a
          # stubborn model cannot loop).
          wrong_lang = reply_language_mismatch(
              self.user_language, assistant_reply
          )
          if (
              wrong_lang
              and self.user_language_name
              and self._language_nudge_count < 3
          ):
            self._language_nudge_count += 1
            logger.warning(
                "Reply language mismatch: user=%s reply=%s (%r); correcting",
                self.user_language, wrong_lang, assistant_reply[:80],
            )
            self._pending_action_nudge = (
                "[HEARTBEAT] LANGUAGE CORRECTION: Your last spoken reply was"
                f" in {language_display_name(wrong_lang)}, but the user is"
                f" speaking {self.user_language_name}. Immediately repeat the"
                " essential content of that reply in"
                f" {self.user_language_name} (one or two short sentences), and"
                f" from now on speak ONLY {self.user_language_name} until the"
                " user switches language. Continue any pending tool calls as"
                " normal."
            )
            # Delivered at the next heartbeat slot (respects the 3 s floor).
          elif not wrong_lang and assistant_reply.strip():
            self._language_nudge_count = 0

          if called_embodiment_action:
            self._pending_user_action_text = ""
            if not self._pending_action_nudge.startswith(
                "[HEARTBEAT] LANGUAGE"
            ):
              self._pending_action_nudge = ""
            self._action_nudge_retries = 0
          elif self._pending_action_nudge:
            pass  # A language correction is already queued for this turn.
          elif interaction_status == "IN_PROGRESS" and not tool_names:
            # Extended-thinking Live models emit an intermediate turnComplete
            # with interactionStatus=IN_PROGRESS right after speaking their
            # opening acknowledgment while toolCalls are still being generated.
            # Give the server 1.5s to deliver those toolCalls before nudging.
            self._last_action_turn_time = loop.time()
          elif (
              ("update_plan" in tool_names or "draw_points" in tool_names)
              and self._pending_user_action_text
              and self.tool_handler is not None
              and self.tool_handler.task_active()
              and self._action_nudge_retries < 2
          ):
            self._action_nudge_retries += 1
            self._pending_action_nudge = (
                "[HEARTBEAT] You called update_plan, but you have NOT called"
                " the embodiment action tool for the current step yet. Execute"
                " the current step right now by calling"
                " detect(instruction=..., x=..., y=...,"
                " box_2d=[ymin,xmin,ymax,xmax], grasp_part=...) or"
                " pick/place/look immediately. Do NOT only speak or call ack."
            )
            # Delivered at the next heartbeat slot (respects the 3 s floor).
          elif (
              not called_embodiment_action
              and (
                  _is_action_request_text(self._pending_user_action_text)
                  or _is_unfulfilled_action_reply(assistant_reply)
              )
              and self._action_nudge_retries < 2
          ):
            self._action_nudge_retries += 1
            req_desc = self._pending_user_action_text or "user manipulation request"
            self._pending_action_nudge = (
                "[HEARTBEAT] SYSTEM MANDATE: The user just requested a physical"
                f" robot task ({req_desc!r}). You are the embodied Spot robot"
                " arm — NEVER give manual human DIY instructions and NEVER"
                " stop after saying you will look. Right now in this turn, you"
                " MUST call update_plan(goal=..., steps=['detect() - ...',"
                " 'pick() - ...', 'detect() - ...', 'place() - ...'],"
                " current_step=0, status='in_progress') AND call"
                " detect(instruction=..., x=..., y=...,"
                " box_2d=[ymin,xmin,ymax,xmax], grasp_part=...) on the target"
                " object in the camera frame!"
            )
            # Delivered at the next heartbeat slot (respects the 3 s floor).

          if had_tool_call:
            self._last_action_turn_time = loop.time()
          if self._interrupted:
            # The user barged in; the model is now serving the user. The
            # hold logic (user_speaking / post_speech_grace /
            # model_generating) keeps the heartbeat away until that is done.
            self._interrupted = False
            source = (
                "turn_complete_action"
                if had_tool_call
                else "turn_complete_after_interrupt"
            )
          elif had_tool_call:
            # An *action* tool ran this turn (ack-only turns are reported as
            # idle by DecisionMaking). Give any function-response-driven
            # continuation a moment to start (post_action_settle hold), then
            # continue the observe→act loop at the configured cadence.
            source = "turn_complete_action"
          else:
            source = "turn_complete"
        elif event.data and isinstance(event.data, dict):
          source = event.data.get("source", source)
        _trigger_source[0] = source
        heartbeat_signal.set()

      if self.heartbeat_enabled:
        self.bus.subscribe(
            [
                event_bus.EventType.USER_TRANSCRIPT,
                event_bus.EventType.TEXT_INPUT,
                event_bus.EventType.GEMINI_TEXT,
            ],
            _on_user_or_assistant_text,
        )
        self.bus.subscribe(
            [
                event_bus.EventType.TURN_COMPLETE,
                event_bus.EventType.HEARTBEAT_TRIGGER,
            ],
            _on_heartbeat_trigger,
        )
        self.bus.subscribe(
            [event_bus.EventType.INTERRUPTED],
            _on_interrupt,
        )

      def _heartbeat_hold_reason() -> str | None:
        """Why a heartbeat must wait right now (None when it may be sent)."""
        if (
            self.tool_handler is not None
            and self.tool_handler.tool_executing.is_set()
        ):
          return "blocking_tool"
        if (
            not self._pending_action_nudge
            and loop.time() - self._last_action_turn_time < post_action_settle
        ):
          return "post_action_settle"
        dm = self.decision_making
        if dm is None:
          return None
        last_loud = getattr(self.observation, "last_loud_audio_time", 0.0)
        if last_loud and time.monotonic() - last_loud < 0.6:
          return "mic_active"
        if dm.user_speaking.is_set():
          started = dm.last_user_speech_start or time.monotonic()
          if time.monotonic() - started < self._STALE_SPEECH_S:
            return "user_speaking"
          # ACTIVITY_START with no ACTIVITY_END for a very long time: the
          # server VAD state is stale (seen when the Live API audio pipeline
          # wedges). Stop treating it as live speech.
          logger.warning(
              "user_speaking set for >%.0fs without ACTIVITY_END; clearing",
              self._STALE_SPEECH_S,
          )
          dm.user_speaking.clear()
        if not self._pending_action_nudge and dm.last_user_speech_end is not None:
          since_speech = time.monotonic() - dm.last_user_speech_end
          if since_speech < post_speech_grace:
            return "post_speech_grace"
        if dm.model_generating.is_set():
          return "model_generating"
        if dm.speaking.is_set():
          return "tts_speaking"
        return None

      def _effective_heartbeat_interval() -> float:
        """Heartbeat cadence; never below MIN_HEARTBEAT_INTERVAL_S (3 s)."""
        return max(MIN_HEARTBEAT_INTERVAL_S, _raw_heartbeat_interval())

      def _raw_heartbeat_interval() -> float:
        if self._pending_action_nudge:
          return 0.0  # next slot, i.e. exactly the 3 s floor
        base = max(0.1, self.heartbeat_interval_seconds)
        now = loop.time()
        task_active = (
            self.tool_handler is not None and self.tool_handler.task_active()
        )
        if task_active:
          # Active task: observe -> act at the floor cadence for every model
          # (non-native Live models have a slower idle base of 6 s).
          return MIN_HEARTBEAT_INTERVAL_S
        recent_action = (now - self._last_action_turn_time) < 8.0
        if recent_action:
          return base
        # Idle: an ack tick is pure quota cost. Back off progressively the
        # longer the user has been silent, and pause during a quota cooldown.
        if time.monotonic() < getattr(self, "_quota_cooldown_until", 0.0):
          return max(base, 30.0)
        since_user = time.monotonic() - (self._last_user_input_time or 0.0)
        if self._last_user_input_time and since_user < 60.0:
          return max(base, 12.0)
        return max(base, 25.0)

      async def _heartbeat_loop():
        try:
          _trigger_source[0] = "session_start"
          heartbeat_signal.set()

          while True:
            interval = _effective_heartbeat_interval()

            # 1) Wait for an idle turn to complete, or fall back to a timer:
            #    the configured interval when nothing is in flight, the
            #    safety timeout when a heartbeat turn appears stuck.
            now = loop.time()
            if self._heartbeat_in_flight:
              deadline = self._last_heartbeat_sent_time + safety_timeout
            else:
              deadline = self._last_heartbeat_sent_time + interval
            try:
              await asyncio.wait_for(
                  heartbeat_signal.wait(), timeout=max(0.0, deadline - now)
              )
              trigger = _trigger_source[0]
            except asyncio.TimeoutError:
              if self._heartbeat_in_flight:
                trigger = "safety_timeout"
                logger.warning(
                    "No turnComplete within %.0fs of the last heartbeat,"
                    " sending recovery heartbeat",
                    safety_timeout,
                )
              else:
                trigger = "idle_interval"
            heartbeat_signal.clear()

            try:
              # 2) Enforce the cadence: never send heartbeats closer together
              #    than the effective interval, re-evaluating every 0.25s so
              #    we immediately switch to fast cadence when a task starts.
              while True:
                interval = _effective_heartbeat_interval()
                wait = self._last_heartbeat_sent_time + interval - loop.time()
                if wait <= 0:
                  break
                await asyncio.sleep(min(0.25, wait))

              # 3) Hold while a heartbeat turn would disrupt something more
              #    important. A blocking robot action or active user speech
              #    holds until complete (user_speaking is bounded by
              #    _STALE_SPEECH_S); other soft holds (TTS / model generating)
              #    are bounded by safety_timeout.
              hold_started = loop.time()
              last_reason = None
              forced_reason = None
              while True:
                reason = _heartbeat_hold_reason()
                if reason is None:
                  break
                if reason != last_reason:
                  logger.info("Holding %s heartbeat: %s", trigger, reason)
                  last_reason = reason
                if reason in ("blocking_tool", "user_speaking", "mic_active"):
                  hold_started = loop.time()
                elif loop.time() - hold_started > safety_timeout:
                  logger.warning(
                      "Heartbeat held for %.0fs (%s); sending anyway",
                      safety_timeout,
                      reason,
                  )
                  forced_reason = reason
                  break
                await asyncio.sleep(0.1)
              if last_reason is not None:
                # Something happened meanwhile; re-evaluate from the top so a
                # fresh trigger / cadence window is honoured.
                heartbeat_signal.clear()
                while True:
                  interval = _effective_heartbeat_interval()
                  wait = self._last_heartbeat_sent_time + interval - loop.time()
                  if wait <= 0:
                    break
                  await asyncio.sleep(min(0.25, wait))

              # 3b) Idle-skip for models not trained on the heartbeat protocol.
              if (
                  self.heartbeat_only_when_task_active
                  and not self._pending_action_nudge
                  and not (
                      self.tool_handler is not None
                      and self.tool_handler.task_active()
                  )
              ):
                if not self._idle_skip_logged:
                  logger.info(
                      "Skipping idle heartbeats (no active task; model is not"
                      " heartbeat-native)"
                  )
                  self._idle_skip_logged = True
                # Advance the cadence clock without sending so the loop does
                # not spin; the next trigger / interval re-evaluates.
                self._last_heartbeat_sent_time = loop.time()
                continue
              self._idle_skip_logged = False

              # 4) Attach the freshest visual context.
              if has_poller:
                chunk = await self.observation.get_video_queue().get()
                video_msg = {
                    "realtimeInput": {
                        "video": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(chunk).decode("utf-8"),
                        }
                    }
                }
                await self.observation.dump_frame(chunk)
                await self.send_message(video_msg)
                logger.info("Sent video frame with heartbeat")
              else:
                # Webcam frames stream continuously (1 FPS); only wait for a
                # new one if the latest frame is already stale.
                frame_age = time.monotonic() - getattr(
                    self.observation, "last_frame_sent_time", 0.0
                )
                if frame_age > 0.75:
                  try:
                    await asyncio.wait_for(
                        self.observation.wait_for_next_frame(), timeout=2.0
                    )
                  except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for webcam frame, proceeding"
                    )

              heartbeat_signal.clear()

              # The frame wait can take up to a second; if the user started
              # talking (or anything else came up) meanwhile, start over
              # instead of injecting a heartbeat into their turn. A heartbeat
              # that was force-released after the hold timeout is not
              # re-aborted for that same (stale) reason, otherwise a stuck
              # flag would silence the agent for good.
              late_reason = _heartbeat_hold_reason()
              if late_reason is not None and late_reason != forced_reason:
                logger.info(
                    "Aborting %s heartbeat before send: %s", trigger, late_reason
                )
                continue

              # 5) Send the heartbeat turn.
              self._heartbeat_sent_count += 1
              self._heartbeat_in_flight = True
              if self._pending_action_nudge:
                hb_text = self._pending_action_nudge
                self._pending_action_nudge = ""
                trigger = f"{trigger}+action_nudge"
              else:
                hb_text = self._get_heartbeat_text()
              heartbeat_msg = {
                  "clientContent": {
                      "turns": [{
                          "role": "user",
                          "parts": [{"text": hb_text}],
                      }],
                      "turnComplete": True,
                  }
              }
              await self.send_message(heartbeat_msg)
              if self.recorder is not None:
                self.recorder.event("heartbeat", {"trigger": trigger, "text": hb_text})
              self._last_heartbeat_sent_time = loop.time()
              logger.info("Sent heartbeat (trigger=%s)", trigger)
              await ui_queue.put({"type": "heartbeat_sent", "trigger": trigger})
            except Exception as e:
              logger.error("Heartbeat send failed: %s", e)
              await asyncio.sleep(1.0)
        except asyncio.CancelledError:
          pass
        except Exception as e:
          logger.error("Heartbeat loop failed: %s", e)

      heartbeat_task = (
          asyncio.create_task(_heartbeat_loop())
          if self.heartbeat_enabled
          else None
      )

    else:
      # Legacy fixed-interval heartbeat with turn_complete chaining.
      heartbeat_trigger = asyncio.Event()

      async def _heartbeat_loop():
        try:
          await asyncio.sleep(2.0)
          while True:
            if (
                self.tool_handler is not None
                and self.tool_handler.tool_executing.is_set()
            ):
              await asyncio.sleep(0.5)
              continue
            while (
                self.decision_making is not None
                and self.decision_making.speaking.is_set()
            ):
              await asyncio.sleep(1.0)

            heartbeat_msg = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": self._get_heartbeat_text()}],
                    }],
                    "turnComplete": True,
                }
            }
            heartbeat_trigger.clear()
            await self.send_message(heartbeat_msg)
            hb_send_time = time.monotonic()
            logger.info("Sent heartbeat")
            self._heartbeat_sent_count += 1
            await ui_queue.put({"type": "heartbeat_sent"})
            try:
              await asyncio.wait_for(
                  heartbeat_trigger.wait(),
                  timeout=self.heartbeat_interval_seconds,
              )
            except asyncio.TimeoutError:
              pass
            elapsed = time.monotonic() - hb_send_time
            remaining = self.heartbeat_min_delay_seconds - elapsed
            if remaining > 0:
              await asyncio.sleep(remaining)
        except asyncio.CancelledError:
          pass
        except Exception as e:
          logger.error("Heartbeat loop failed: %s", e)

      heartbeat_task = (
          asyncio.create_task(_heartbeat_loop())
          if self.heartbeat_enabled
          else None
      )

    # --- Start the event bus ---

    self.bus.start()

    # Watches for a wedged Live API socket (speech started but the server
    # went silent / no reply to a heartbeat) and replaces it.
    health_task = asyncio.create_task(self._stream_health_loop())

    # --- Main loop: drain UI queue and yield to WebSocket ---

    try:
      while True:
        event = await ui_queue.get()

        if isinstance(event, dict):
          yield event
          continue

        if event.type == event_bus.EventType.SESSION_DONE:
          gen = (
              event.data.get("generation")
              if isinstance(event.data, dict)
              else None
          )
          if gen is not None and gen != self._stream_generation:
            # A socket that was already replaced finished closing.
            logger.info("Ignoring SESSION_DONE from superseded stream #%d", gen)
            continue
          if await self._reconnect():
            continue
          # Deliver any pending status notices (e.g. the "failed" one) so
          # the browser can tell the user why the session ended.
          while not ui_queue.empty():
            pending = ui_queue.get_nowait()
            if isinstance(pending, dict):
              yield pending
          break

        if event.type == event_bus.EventType.TURN_COMPLETE:
          self._turn_completed_count += 1

        # Forward INTERRUPTED to the UI only for real user barge-ins (the
        # browser then stops TTS playback). Interruptions caused by our own
        # heartbeat / text turns pre-empting an in-flight model turn are
        # internal noise and are suppressed. The server VAD tells the two
        # apart: a barge-in is preceded by voiceActivity ACTIVITY_START.
        if event.type == event_bus.EventType.INTERRUPTED:
          is_tts_preempt = isinstance(event.data, dict) and event.data.get(
              "tts_preempt"
          )
          user_barge_in = isinstance(event.data, dict) and bool(
              event.data.get("user_barge_in")
          )
          dm = self.decision_making
          if not user_barge_in and dm is not None:
            if dm.user_speaking.is_set():
              user_barge_in = True
            elif dm.last_user_speech_start is not None:
              user_barge_in = (
                  time.monotonic() - dm.last_user_speech_start
              ) < 2.0
          if not is_tts_preempt and not user_barge_in:
            logger.info("Suppressing INTERRUPTED — not a user barge-in")
            continue
          if user_barge_in:
            logger.info("User barge-in: forwarding INTERRUPTED to UI")

        ui_event = self._to_ui_event(event)
        yield ui_event
        if (
            ui_event.get("type") == "turn_complete"
            and not self.use_event_driven_heartbeat
        ):
          if heartbeat_trigger is not None:
            heartbeat_trigger.set()

    finally:
      self._closing = True
      if self.recorder is not None:
        try:
          self.recorder.close()
        except Exception:  # pylint: disable=broad-except
          logger.exception("recorder close failed")
        self.recorder = None
      health_task.cancel()
      if heartbeat_task:
        heartbeat_task.cancel()
      if self.tool_handler is not None:
        self.tool_handler.shutdown()
      await self.observation.stop_input_tasks()
      self.observation.close()
      await self.bus.shutdown()
      self._ui_queue = None
      if self.stream:
        try:
          self.stream.Shutdown()
        except Exception:  # pylint: disable=broad-except
          pass
        self.stream = None

  def _to_ui_event(self, event: event_bus.Event) -> Mapping[str, Any]:
    """Convert a typed Event to the UI dict format for the WebSocket."""

    if event.type == event_bus.EventType.GEMINI_TEXT:
      return {"type": "gemini", "text": event.data.get("text", "")}
    elif event.type == event_bus.EventType.GEMINI_THOUGHT:
      return {"type": "gemini_thought", "text": event.data.get("text", "")}
    elif event.type == event_bus.EventType.TEXT_INPUT:
      return {"type": "text_input", "text": event.data}
    elif event.type == event_bus.EventType.USER_TRANSCRIPT:
      d: dict[str, Any] = {
          "type": "user_transcript",
          "text": event.data.get("text", ""),
      }
      if event.data.get("replace"):
        d["replace"] = True
      if "utterance_id" in event.data:
        d["utterance_id"] = event.data["utterance_id"]
      return d
    elif event.type == event_bus.EventType.AUDIO_RESPONSE:
      return {
          "type": "audio_response",
          "data": event.data.get("audio_data", ""),
      }
    elif event.type == event_bus.EventType.TOOL_RESULT:
      return event.data  # Already a dict
    elif event.type == event_bus.EventType.TURN_COMPLETE:
      d: dict[str, Any] = {"type": "turn_complete"}
      if isinstance(event.data, dict) and "had_tool_call" in event.data:
        d["had_tool_call"] = event.data["had_tool_call"]
      return d
    elif event.type == event_bus.EventType.TELEMETRY:
      d: dict[str, Any] = {"type": "telemetry"}
      if isinstance(event.data, dict):
        for key in ["token_usage", "server_ttft_ms"]:
          if key in event.data:
            d[key] = event.data[key]
      return d
    elif event.type == event_bus.EventType.INTERRUPTED:
      return {"type": "interrupted"}
    elif event.type == event_bus.EventType.ERROR:
      return {"type": "error", "error": event.data}
    elif event.type == event_bus.EventType.TRANSPARENT_HISTORY:
      # In Lite, transparent history data is already a dict (no proto).
      return {"type": "transparent_history", "data": event.data}
    elif event.type == event_bus.EventType.LOG:
      return {"type": "log", "data": event.data}
    else:
      logger.warning("Unknown UI event type: %s", event.type)
      return {"type": "unknown", "data": str(event.data)}
