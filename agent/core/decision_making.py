"""Decides the next step based on model responses.

Subscribes to MODEL_RESPONSE events and routes them into typed events
(TOOL_CALL, GEMINI_TEXT, AUDIO_CHUNK, TURN_COMPLETE, INTERRUPTED) for
downstream handlers.

This Lite version parses JSON dicts from the public BidiGenerateContent
WebSocket API instead of protobuf messages.
"""

import asyncio
import base64
import logging
import re
import time
from typing import Any, Awaitable, Callable

from core import event_bus
from model import tts_client


def _is_internal_heartbeat_text(text: str) -> bool:
  return text.lstrip().startswith("[HEARTBEAT]")


def _is_ignored_output_transcription(text: str) -> bool:
  """Return True for non-speech placeholders or canned transcription errors."""
  stripped = text.strip()
  if not stripped:
    return True
  if _is_internal_heartbeat_text(stripped):
    return True
  if stripped.startswith("<") and stripped.endswith(">"):
    return True
  if stripped.lower().startswith("an error occurred"):
    return True
  return False


def _is_reasoning_leak_in_transcription(text: str) -> bool:
  """Return True if outputTranscription contains internal markdown chain-of-thought."""
  if "`" in text or "```" in text:
    return True
  for line in text.splitlines():
    ls = line.lstrip()
    if ls.startswith("* ") or ls.startswith("- "):
      return True
  lower = text.lower()
  if any(
      marker in lower
      for marker in (
          "current step:",
          "i already called",
          "system instructions",
          "developer instruction",
      )
  ):
    return True
  return False


_PINYIN_TOKENS = frozenset({
    "ni", "nhao", "nihao", "wo", "zhe", "ge", "bi", "jia", "zai", "ben",
    "zi", "zhong", "jian", "bang", "cha", "yi", "xia", "zen", "me", "ba",
    "fang", "dao", "shang", "na", "qu", "kan", "hao", "ma", "qing", "gei",
    "yan", "jing", "shou", "ji", "zhuo", "bei",
})


def normalize_input_transcription(text: str) -> str:
  """Clean spacing and leading VAD noise in server inputTranscription chunks."""
  if not text:
    return ""
  s = text
  # Remove spaces between CJK characters (e.g. ' 你 能 帮 我 ' -> '你能帮我').
  s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
  # Remove spaces between CJK characters and punctuation.
  s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？、；：,.!?;:])", "", s)
  s = re.sub(r"(?<=[，。！？、；：,.!?;:])\s+(?=[\u4e00-\u9fff])", "", s)
  cjk_chars = re.findall(r"[\u4e00-\u9fff]", s)
  if cjk_chars:
    # Strip leading/trailing whitespace adjacent to CJK so multi-chunk CJK
    # transcripts join without inter-chunk spaces.
    s = re.sub(r"^\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"(?<=[\u4e00-\u9fff。！？；，,.!?])\s+$", "", s)
    # Strip short leading non-CJK VAD noise before a Chinese sentence
    # (e.g. 'pauca怎么把笔...' -> '怎么把笔...', 'Hallo. 你可以听到吗?' -> '你可以听到吗?').
    if len(cjk_chars) >= 2:
      s = re.sub(
          r"^[A-Za-zÀ-ÿ\u0100-\u024f\s,.;:!?¿¡'\-]{1,18}(?=[\u4e00-\u9fff])",
          "",
          s,
      )
  return s


def is_garbled_input_transcription(text: str) -> bool:
  """Detect legacy sidecar ASR gibberish (Spanish/Portuguese/diacritics/Pinyin)."""
  if not text or not text.strip():
    return False
  # If it contains actual CJK characters, it is not pure Latin/foreign gibberish.
  if re.search(r"[\u4e00-\u9fff]", text):
    return False
  # Latin diacritics or Spanish/Portuguese inverted punctuation (e.g. 'Sí. Hoy es ¿Qué?', 'pão. Vamos ver...').
  if re.search(r"[À-ÿ¿¡\u0100-\u024f]", text):
    return True
  # Non-ASCII, non-CJK scripts (Cyrillic, Arabic, Devanagari, Thai, etc.).
  if re.search(r"[^\x00-\x7f\u2000-\u206f\u3000-\u303f\uff00-\uffef]", text):
    return True
  # CJK fullwidth punctuation paired with purely ASCII words (e.g. 'ni bao cha yi xia ... 。').
  if re.search(r"[。？！，、]", text) and re.search(r"[A-Za-z]", text):
    return True
  # Romanized Mandarin Pinyin tokens without CJK characters.
  tokens = re.findall(r"[a-zA-Z]+", text.lower())
  if len(tokens) >= 2:
    pinyin_hits = sum(1 for t in tokens if t in _PINYIN_TOKENS)
    if pinyin_hits >= 2 and pinyin_hits / len(tokens) >= 0.35:
      return True
  return False


# Fallback context window size used for the CTX utilisation metric when no
# explicit limit is configured. gemini-robotics-er-2-streaming-preview reports
# inputTokenLimit=131072 via the models API.
_DEFAULT_CONTEXT_WINDOW_LIMIT = 131072


def _modality_breakdown(details: Any) -> dict[str, int]:
  """Convert ``promptTokensDetails`` entries into ``{modality: tokens}``."""
  out: dict[str, int] = {}
  for item in details or []:
    if not isinstance(item, dict):
      continue
    modality = str(item.get("modality", "UNKNOWN"))
    out[modality] = out.get(modality, 0) + int(item.get("tokenCount", 0) or 0)
  return out


class DecisionMaking:
  """A component that routes model responses into typed events on the event bus.

  This component decides the next step based on model responses. Currently
  this is implemented as pure event routing — each model response is parsed
  and re-published as a more specific event type for downstream handlers.

  Future extensions may add planning, state tracking, or other decision
  logic here.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      text_output_callback: Callable[[str], Awaitable[None]] | None = None,
      audio_interrupt_callback: Callable[[], Awaitable[None]] | None = None,
      context_window_limit: int | None = None,
      utterance_audio_provider: Callable[[], bytes] | None = None,
      refine_audio_transcripts: bool = False,
      api_key: str | None = None,
      input_sample_rate: int = 16000,
  ):
    self._bus = bus
    self._text_output_callback = text_output_callback
    self._audio_interrupt_callback = audio_interrupt_callback
    self._utterance_audio_provider = utterance_audio_provider
    self._refine_audio_transcripts = refine_audio_transcripts
    self._api_key = api_key
    self._input_sample_rate = input_sample_rate
    # Simple timing instead of telemetry tracker.
    self._turn_start_time: float | None = None
    self._first_response_time: float | None = None
    # True while TTS synthesis + playback is in-flight. The heartbeat
    # loop checks this to avoid sending prompts that would interrupt
    # speech playback.
    self.speaking = asyncio.Event()
    self._pending_tts_count = 0
    self._tts_queue: list[str] = []
    self._state_lock = asyncio.Lock()
    # Names of the tool calls issued by the model in the current turn. Used
    # to decide whether the turn did real work or was an idle ``ack`` turn.
    self._current_turn_tool_names: list[str] = []
    # Tracks whether the current turn has received any inlineData audio chunk
    # so we can distinguish real speech transcriptions from pre-tool thought leaks.
    self._turn_had_audio: bool = False
    # Utterance tracking for clean UI replacement when inputTranscription is garbled.
    self._utterance_seq: int = 0
    self._utterance_open: bool = False
    self._utterance_garbled: bool = False
    self.last_transcript_garbled: bool = False
    self.last_garbled_utterance_id: int | None = None
    self._refine_task: asyncio.Task | None = None
    # Server-side VAD state. ``user_speaking`` is set between the
    # ``voiceActivity`` ACTIVITY_START and ACTIVITY_END signals; the heartbeat
    # scheduler uses it (plus ``last_user_speech_end``) to avoid injecting a
    # heartbeat turn while the user is talking or the model is about to
    # answer them.
    self.user_speaking = asyncio.Event()
    self.last_user_speech_start: float | None = None
    self.last_user_speech_end: float | None = None
    # Set from the first model output of a turn (tool call / model part)
    # until ``turnComplete`` or ``interrupted``. The heartbeat scheduler holds
    # while this is set so a heartbeat never cuts into a reply the model is
    # already producing (e.g. its answer to the user's speech).
    self.model_generating = asyncio.Event()
    # The currently active non-blocking TTS background task, if any.
    # Used to cancel in-flight TTS when the user interrupts the model.
    self._active_tts_task: asyncio.Task | None = None
    # Token usage of the most recent turn (from usageMetadata messages) plus
    # the running total of output tokens for the session.
    self._usage_metadata: dict[str, Any] = {}
    self._cumulative_output_tokens = 0
    self.context_window_limit = (
        context_window_limit or _DEFAULT_CONTEXT_WINDOW_LIMIT
    )
    bus.subscribe([event_bus.EventType.MODEL_RESPONSE], self._handle)
    bus.subscribe([event_bus.EventType.TEXT_INPUT], self._handle_text_input)

  # ---- TTS lifecycle management -------------------------------------------

  async def _decrement_tts_and_maybe_clear(self) -> None:
    """Decrement TTS counter; clear speaking when it hits zero."""
    async with self._state_lock:
      self._pending_tts_count -= 1
      if self._pending_tts_count <= 0 and not self._tts_queue:
        self._pending_tts_count = 0
        self.speaking.clear()

  async def _cancel_active_tts(self) -> None:
    """Cancel the currently active TTS background task and clear queued speech."""
    async with self._state_lock:
      self._tts_queue.clear()
    task = self._active_tts_task
    if task is not None and not task.done():
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        pass
    self._active_tts_task = None
    async with self._state_lock:
      self._pending_tts_count = 0
      self.speaking.clear()

  async def interrupt_tts_for_user(self, reason: str = "user_barge_in") -> None:
    """Immediately stop active TTS synthesis and buffered browser playback."""
    was_speaking = (
        self.speaking.is_set()
        or bool(self._tts_queue)
        or (
            self._active_tts_task is not None
            and not self._active_tts_task.done()
        )
    )
    if was_speaking:
      logging.info("Interrupting active server TTS due to %s", reason)
      await self._cancel_active_tts()
    if self._audio_interrupt_callback:
      try:
        await self._audio_interrupt_callback()
      except Exception as e:  # pylint: disable=broad-except
        logging.error("audio_interrupt_callback failed: %s", e)
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.INTERRUPTED,
            source=event_bus.EventSource.USER,
            data={"user_barge_in": True, "reason": reason},
        )
    )

  async def _handle_text_input(self, event: event_bus.Event) -> None:
    """Stop active TTS when the user sends a text message (excluding heartbeats)."""
    text = str(event.data or "")
    if text and not _is_internal_heartbeat_text(text):
      await self.interrupt_tts_for_user("user_text_input")

  async def _start_tts_task(self, text: str) -> None:
    """Enqueue assistant speech for sequential non-blocking TTS playback.

    Queuing rather than preempting ensures that multi-step tool chains
    (e.g. update_plan + send_message -> detect -> pick + send_message) play
    the full first explanation to completion before speaking the follow-up
    status update, while user barge-in still cancels immediately.
    """
    cleaned = (text or "").strip()
    if not cleaned:
      return
    async with self._state_lock:
      if self._tts_queue and self._tts_queue[-1] == cleaned:
        return
      self._tts_queue.append(cleaned)
      self.speaking.set()
      self._pending_tts_count = len(self._tts_queue)
      need_worker = (
          self._active_tts_task is None or self._active_tts_task.done()
      )
      if need_worker:
        self._active_tts_task = asyncio.create_task(self._drain_tts_queue())

  async def _drain_tts_queue(self) -> None:
    """Sequentially synthesize all queued assistant speech items."""
    try:
      while True:
        async with self._state_lock:
          if not self._tts_queue:
            break
          next_text = self._tts_queue.pop(0)
        if self._text_output_callback:
          await self._text_output_callback(next_text)
    except asyncio.CancelledError:
      logging.info("TTS queue worker cancelled")
    except Exception as e:  # pylint: disable=broad-except
      logging.error("text_output_callback failed: %s", e)
    finally:
      async with self._state_lock:
        self._tts_queue.clear()
        self._pending_tts_count = 0
        self.speaking.clear()

  def _schedule_transcript_refinement(self, utterance_id: int) -> None:
    """Run non-blocking audio transcription refinement for utterance_id."""
    if self._utterance_audio_provider is None:
      return
    if self._refine_task is not None and not self._refine_task.done():
      self._refine_task.cancel()
    self._refine_task = asyncio.create_task(
        self._run_transcript_refinement(utterance_id)
    )

  async def _run_transcript_refinement(self, utterance_id: int) -> None:
    """Wait briefly for trailing inputTranscription chunks, then refine via ASR."""
    try:
      await asyncio.sleep(0.25)
      if self._utterance_audio_provider is None:
        return
      pcm_bytes = self._utterance_audio_provider()
      if not pcm_bytes:
        return
      refined = await asyncio.to_thread(
          tts_client.transcribe_audio_pcm,
          pcm_bytes,
          self._input_sample_rate,
          self._api_key,
      )
      cleaned = normalize_input_transcription(refined).strip()
      if cleaned and not is_garbled_input_transcription(cleaned):
        self.last_transcript_garbled = False
        logging.info(
            "Refined user audio transcript (utterance=%d): %s",
            utterance_id,
            cleaned,
        )
        await self._bus.publish(
            event_bus.Event(
                type=event_bus.EventType.USER_TRANSCRIPT,
                source=event_bus.EventSource.USER,
                data={
                    "text": cleaned,
                    "replace": True,
                    "utterance_id": utterance_id,
                },
            )
        )
    except asyncio.CancelledError:
      pass
    except Exception as e:  # pylint: disable=broad-except
      logging.debug("Transcript refinement failed: %s", e)

  # ---- Simple timing helpers ----------------------------------------------

  def _mark_turn_start(self) -> None:
    if self._turn_start_time is None:
      self._turn_start_time = time.monotonic()

  def _mark_first_response(self) -> None:
    # Any model output means a turn is in progress until turnComplete or
    # interrupted clears it (both go through _reset_timing).
    self._utterance_open = False
    self.model_generating.set()
    if self._first_response_time is None:
      self._first_response_time = time.monotonic()

  def _reset_timing(self) -> dict[str, Any]:
    """Reset timing and return metrics for the completed turn."""
    metrics = {}
    if self._turn_start_time and self._first_response_time:
      metrics['ttft_ms'] = round(
          (self._first_response_time - self._turn_start_time) * 1000, 1
      )
    if self._usage_metadata:
      metrics['token_usage'] = dict(self._usage_metadata)
    self._turn_start_time = None
    self._first_response_time = None
    self._usage_metadata = {}
    self._turn_had_audio = False
    self._utterance_open = False
    self.model_generating.clear()
    return metrics

  # ---- Main event handler -------------------------------------------------

  async def _handle(self, event: event_bus.Event) -> None:
    """Process a MODEL_RESPONSE event.

    The event.data is a parsed JSON dictionary.
    """
    response = event.data
    self._mark_turn_start()

    # --- Token usage (arrives together with turnComplete; ingest first so
    # the TELEMETRY event emitted on turn completion includes it) ---
    if "usageMetadata" in response:
      self._record_usage(response["usageMetadata"])

    # --- Server-side voice activity detection ---
    if "voiceActivity" in response:
      started = self._record_voice_activity(response["voiceActivity"])
      if started:
        self._utterance_open = False
        await self.interrupt_tts_for_user("server_vad_activity_start")

    # --- Tool call cancellation (model abandoned pending calls) ---
    if "toolCallCancellation" in response:
      ids = (response.get("toolCallCancellation") or {}).get("ids", [])
      logging.info('Tool call cancellation received for ids=%s', ids)

    # --- Tool call ---
    if "toolCall" in response:
      for fc in response["toolCall"].get("functionCalls", []):
        self._current_turn_tool_names.append(str(fc.get("name", "")))
      self._mark_first_response()
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_CALL,
              source=event_bus.EventSource.ASSISTANT,
              data=response["toolCall"],
          )
      )

    # --- Server content ---
    if "serverContent" in response:
      await self._handle_server_content(response["serverContent"])

  def _record_usage(self, usage: dict) -> None:
    """Translate a ``usageMetadata`` payload into the UI's token_usage shape."""
    if not isinstance(usage, dict):
      return
    prompt = int(usage.get("promptTokenCount", 0) or 0)
    response_tokens = int(usage.get("responseTokenCount", 0) or 0)
    thoughts = int(usage.get("thoughtsTokenCount", 0) or 0)
    total = int(usage.get("totalTokenCount", prompt + response_tokens) or 0)
    self._cumulative_output_tokens += response_tokens + thoughts
    limit = self.context_window_limit
    self._usage_metadata = {
        "prompt_token_count": prompt,
        "response_token_count": response_tokens,
        "thoughts_token_count": thoughts,
        "total_token_count": total,
        "cumulative_output_tokens": self._cumulative_output_tokens,
        "context_window_limit": limit,
        "context_window_utilization_pct": (
            round(prompt / limit * 100.0, 1) if limit else None
        ),
        "prompt_tokens_by_modality": _modality_breakdown(
            usage.get("promptTokensDetails")
        ),
    }

  def _record_voice_activity(self, activity: dict) -> bool:
    """Track ACTIVITY_START / ACTIVITY_END signals from the server VAD."""
    if not isinstance(activity, dict):
      return False
    kind = str(activity.get("type") or activity.get("voiceActivityType") or "")
    now = time.monotonic()
    if kind == "ACTIVITY_START":
      self.user_speaking.set()
      self.last_user_speech_start = now
      logging.info('User speech started (server VAD)')
      return True
    elif kind == "ACTIVITY_END":
      self.user_speaking.clear()
      self.last_user_speech_end = now
      logging.info('User speech ended (server VAD)')
    return False

  async def _handle_server_content(self, server_content: dict) -> None:
    """Route server content to appropriate events."""

    # Handle model turn (audio/text parts)
    if "modelTurn" in server_content:
      for part in server_content["modelTurn"].get("parts", []):
        # In Gemini Live API, text parts look like: {"text": "Hello"}
        # Some are thoughts if thought=True, though public API might not have this yet.
        if part.get("thought"):
          self._mark_first_response()
          await self._bus.publish(
              event_bus.Event(
                  type=event_bus.EventType.GEMINI_THOUGHT,
                  source=event_bus.EventSource.ASSISTANT,
                  data={"text": part.get("text", "")},
              )
          )
        elif "inlineData" in part:
          self._turn_had_audio = True
          self._mark_first_response()
          # Publish raw audio chunk — AudioResponseHandler accumulates.
          await self._bus.publish(
              event_bus.Event(
                  type=event_bus.EventType.AUDIO_CHUNK,
                  source=event_bus.EventSource.ASSISTANT,
                  data=base64.b64decode(part["inlineData"].get("data", "")),
              )
          )
        elif "text" in part:
          if _is_internal_heartbeat_text(part["text"]):
            continue
          self._mark_first_response()
          # Publish text event immediately so downstream handlers (UI,
          # logging) receive it without waiting for TTS synthesis.
          await self._bus.publish(
              event_bus.Event(
                  type=event_bus.EventType.GEMINI_TEXT,
                  source=event_bus.EventSource.ASSISTANT,
                  data={"text": part["text"]},
              )
          )
          # Start TTS in a non-blocking background task.
          await self._start_tts_task(part["text"])

    # Handle output transcription (native-audio models' spoken words to text).
    # Filter out internal heartbeat echoes, <no speech> placeholders, and
    # canned transcription-layer error strings. Route pre-tool reasoning leaks
    # (markdown bullets/backticks) to GEMINI_THOUGHT instead of GEMINI_TEXT.
    if "outputTranscription" in server_content:
      text = server_content["outputTranscription"].get("text", "")
      if text and not _is_ignored_output_transcription(text):
        self._mark_first_response()
        event_type = (
            event_bus.EventType.GEMINI_THOUGHT
            if _is_reasoning_leak_in_transcription(text)
            else event_bus.EventType.GEMINI_TEXT
        )
        await self._bus.publish(
            event_bus.Event(
                type=event_type,
                source=event_bus.EventSource.ASSISTANT,
                data={"text": text},
            )
        )

    # Handle input transcription (user's audio translation to text).
    # In Gemini Live API, inputTranscription arrives at ACTIVITY_END, right as
    # the model's native audio response starts. Only interrupt if server-side
    # 3P TTS from a previous turn is still actively speaking.
    if "inputTranscription" in server_content:
      raw_text = server_content["inputTranscription"].get("text", "")
      if raw_text:
        if raw_text.strip() and self.speaking.is_set():
          await self.interrupt_tts_for_user("input_transcription")
        if not self._utterance_open:
          self._utterance_seq += 1
          self._utterance_open = True
          self._utterance_garbled = False
        uid = self._utterance_seq
        if is_garbled_input_transcription(raw_text) or self._utterance_garbled:
          if not self._utterance_garbled:
            self._utterance_garbled = True
            self.last_transcript_garbled = True
            self.last_garbled_utterance_id = uid
            await self._bus.publish(
                event_bus.Event(
                    type=event_bus.EventType.USER_TRANSCRIPT,
                    source=event_bus.EventSource.USER,
                    data={
                        "text": "🎤 正在识别语音…",
                        "replace": True,
                        "utterance_id": uid,
                    },
                )
            )
          self._schedule_transcript_refinement(uid)
        else:
          clean_text = normalize_input_transcription(raw_text)
          if clean_text:
            await self._bus.publish(
                event_bus.Event(
                    type=event_bus.EventType.USER_TRANSCRIPT,
                    source=event_bus.EventSource.USER,
                    data={"text": clean_text, "utterance_id": uid},
                )
            )

    # Handle turn complete.
    if server_content.get("turnComplete"):
      tool_names = list(self._current_turn_tool_names)
      self._current_turn_tool_names = []  # Reset for next turn
      interaction_status = str(server_content.get("interactionStatus", "") or "")
      # Only *action* tools count: a turn that merely acknowledged the
      # heartbeat is an idle turn, and the heartbeat scheduler must keep
      # driving the loop at its configured interval afterwards.
      had_tool_call = any(
          name not in event_bus.IDLE_TOOL_NAMES for name in tool_names
      )
      logging.info(
          'Turn complete (had_tool_call=%s, tools=%s)', had_tool_call, tool_names
      )

      turn_metrics = self._reset_timing()

      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TURN_COMPLETE,
              source=event_bus.EventSource.ASSISTANT,
              data={
                  'had_tool_call': had_tool_call,
                  'tool_names': tool_names,
                  'interaction_status': interaction_status,
              },
          )
      )

      if turn_metrics:
        await self._bus.publish(
            event_bus.Event(
                type=event_bus.EventType.TELEMETRY,
                source=event_bus.EventSource.ASSISTANT,
                data=turn_metrics,
            )
        )

    # Handle interruption.
    if server_content.get("interrupted"):
      self._reset_timing()
      # Cancel any active TTS playback immediately so audio stops as
      # soon as the user interrupts.
      await self._cancel_active_tts()
      if self._audio_interrupt_callback:
        try:
          await self._audio_interrupt_callback()
        except Exception as e:  # pylint: disable=broad-except
          logging.error('audio_interrupt_callback failed: %s', e)
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.INTERRUPTED,
              source=event_bus.EventSource.ASSISTANT,
          )
      )
