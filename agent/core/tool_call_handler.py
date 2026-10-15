"""Handler that executes tool calls from the model.

Subscribes to TOOL_CALL events, executes functions via the embodiment,
sends JSON dict responses back to the Gemini session, and publishes
TOOL_RESULT events.

This Lite version uses JSON dicts for the public BidiGenerateContent API
instead of protobuf messages.
"""

import asyncio
from collections.abc import Callable
import datetime
import json
import logging
import re
import time
from typing import Any

import httpx

from core import action_watchdog
from core import event_bus
from tool import grounding_tools


_PICK_CAMERA_STABLE_SECONDS = 3.0
# Supervisor loop cadence for the action watchdog (seconds).
_WATCHDOG_TICK_S = 0.5
# confirm_action is only accepted once the command has had time to take
# effect and at least one camera frame captured after it has been sent.
_MIN_CONFIRM_DWELL_S = 3.0
# retry_step is two-phase: the failure must still be visible on a later
# heartbeat (between these bounds) before the plan is moved back.
_RETRY_RECHECK_MIN_S = 2.0
_RETRY_RECHECK_MAX_S = 30.0
# Plan-level retries (going back to redo steps) allowed per plan.
_MAX_PLAN_RETRIES = 2
_MIN_CONFIRM_FRAME_LAG_S = 1.0

# Spoken / typed commands that cancel the running task (any language).
_USER_STOP_RE = re.compile(
    r"\b(stop|cancel|abort|halt|enough|never mind|nevermind)\b|停止|停下|停一下|"
    r"别动|不要动|终止|取消|算了|不用了|中止|停止任务|やめて|止めて|그만|멈춰",
    re.IGNORECASE,
)


_NEGATED_SIT_PATTERN = re.compile(
    r"(?:\b(?:do\s+not|don't|never)\b[^.!?]{0,40}\bsit\b|\bnot\s+to\s+sit\b"
    # zh / ja / ko
    r"|(?:不要|别|不准|不许|不用|不必|无需)\s*(?:坐|趴)"
    r"|座らないで|すわらないで|앉지\s*마"
    # es / pt / fr / de / it / ru
    r"|\bno\s+te\s+sientes\b|\bnão\s+(?:se\s+)?sente\b|\bne\s+t['’]assieds\s+pas\b"
    r"|\bsetz\s+dich\s+nicht\b|\bnicht\s+(?:hin)?setzen\b|\bnon\s+sederti\b|\bне\s+садись\b)",
    re.IGNORECASE,
)
# Sit requests in any common language. Latin-script words use \b; CJK phrases
# cannot (no word boundaries), and ASR often inserts spaces between CJK
# characters ("坐 下 来"), which user_requested_sit() strips first.
_SIT_PATTERN = re.compile(
    r"\b(?:sit(?:\s+down)?|take\s+a\s+seat)\b"
    r"|坐下|坐好|坐着|坐吧|请坐|坐一下|趴下|蹲下"
    r"|座って|すわって|お座り|おすわり|座れ|座る"
    r"|앉아|앉으"
    r"|\bsi[eé]ntat[e]\b|\bsi[eé]ntese\b|\bsentad[oa]\b"
    r"|\bsenta\b|\bsente-se\b|\bsentar\b"
    r"|\bassieds-toi\b|\basseyez-vous\b|\bassis\b"
    r"|\bsetz\s+dich\b|\bhinsetzen\b|\bsitz\b"
    r"|\bsiediti\b|\bsedut[oa]\b"
    r"|\bсядь\b|\bсадись\b|\bсидеть\b",
    re.IGNORECASE,
)
_CJK_GAP_RE = re.compile(r"(?<=[^\x00-\x7f])\s+(?=[^\x00-\x7f])")


def user_requested_sit(text: str) -> bool:
  """Return whether a user utterance explicitly asks Spot to sit (any language)."""
  text = _CJK_GAP_RE.sub("", text or "")
  return bool(_SIT_PATTERN.search(text)) and not bool(
      _NEGATED_SIT_PATTERN.search(text)
  )


def pick_response_for_model(
    result: Any,
    post_action_frame_sent: bool,
) -> dict[str, Any]:
  """Describe pick completion without claiming visually unverified success."""
  if not isinstance(result, dict):
    result = {"result": result}

  if (
      result.get("executed") is False
      and "required_stable_seconds" in result
  ):
    return {
        "outcome": "blocked_camera_unstable",
        "executed": False,
        "visual_verification_required": False,
        "post_action_frame_sent": post_action_frame_sent,
        "camera_stable_for_seconds": result.get(
            "camera_stable_for_seconds", 0.0
        ),
        "required_stable_seconds": result["required_stable_seconds"],
        "instruction": result.get("retry_instruction", result.get("error", "")),
    }

  failure_reasons = []
  if result.get("error"):
    failure_reasons.append(str(result["error"]))
  if result.get("success") is False:
    failure_reasons.append("backend success diagnostic is false")
  if result.get("holding_item") is False:
    failure_reasons.append("gripper holding-item sensor is false")
  backend_state = str(result.get("state", ""))
  if "FAILED" in backend_state or "NO_SOLUTION" in backend_state:
    failure_reasons.append(f"backend manipulation state is {backend_state}")

  response: dict[str, Any] = {
      "outcome": "failed" if failure_reasons else "unverified",
      "visual_verification_required": not failure_reasons,
      "post_action_frame_sent": post_action_frame_sent,
      "instruction": (
          "Inspect the fresh post-pick camera image. Claim success only if the"
          " requested object is visibly secured by the gripper and has moved"
          " from its original location. Backend completion and gripper sensor"
          " signals are diagnostics, not proof of a successful pick. If the"
          " image is missing or ambiguous, report that the pick is unverified."
      ),
  }
  if backend_state:
    response["backend_state"] = backend_state
  if failure_reasons:
    response["failure_reasons"] = failure_reasons
  if isinstance(result.get("target"), dict):
    response["target"] = result["target"]
  # Watchdog-supervised embodiments: keep the attempt bookkeeping and the
  # explicit next step so the model follows the confirm/retry protocol.
  for key in ("grasp_point", "grasp_part", "trajectory", "watchdog", "next_action"):
    if key in result:
      response[key] = result[key]
  return response


def blocking_tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str]:
  """Return the names of tools whose declarations mark them BLOCKING."""
  names = set()
  for group in tools or []:
    declarations = group.get("functionDeclarations", [group])
    for declaration in declarations:
      if str(declaration.get("behavior", "")).upper() == "BLOCKING":
        name = declaration.get("name")
        if name:
          names.add(name)
  return frozenset(names)


class ToolCallHandler:
  """A handler that executes tool calls and publishes results back to the event bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      session_manager: Any,
      embodiment: Any,
      agent_peers: dict[str, str] | None = None,
      peer_name: str = "unknown",
      text_output_callback: Any | None = None,
      clock: Callable[[], datetime.datetime] = datetime.datetime.utcnow,
      enable_send_message_to_user: bool = False,
      blocking_tools: frozenset[str] | set[str] | None = None,
  ):
    self._bus = bus
    self._clock = clock
    self._session_manager = session_manager
    self._embodiment = embodiment
    self._agent_peers = agent_peers or {}
    self._peer_name = peer_name
    # Optional TTS callback so send_message can speak the message
    # out loud before delivering it to the peer agent.
    self._text_output_callback = text_output_callback
    self._enable_send_message_to_user = enable_send_message_to_user
    self._blocking_tools = frozenset(blocking_tools or ())
    self._blocking_execution_count = 0
    self._sit_authorized = False
    self._user_transcript = ""
    # Last accepted update_plan (goal/steps/current_step/status); used to gate
    # advancing past an unconfirmed action and to mark the plan failed when
    # the watchdog aborts a task.
    self._plan: dict[str, Any] | None = None
    # Plan stickiness bookkeeping (see _execute_update_plan).
    self._last_user_input_time = 0.0
    self._plan_started_at = 0.0
    self._plan_finished_at = 0.0
    # retry_step bookkeeping (reset with every new plan).
    self._plan_retries = 0
    self._retry_intent: tuple[int, float] | None = None
    # Command -> visual confirmation -> retry -> abort supervisor. Enabled by
    # embodiments that set ``verify_actions_visually = True``.
    self._watchdog = self._build_watchdog(embodiment)
    self._watchdog_task: asyncio.Task | None = None
    # Set while any blocking tool is executing. The heartbeat
    # loop checks this and skips sending prompts to avoid re-triggering
    # the model before the tool result is in context.
    self.tool_executing = asyncio.Event()
    bus.subscribe([event_bus.EventType.TOOL_CALL], self._handle_tool_call)
    bus.subscribe(
        [event_bus.EventType.TEXT_INPUT, event_bus.EventType.USER_TRANSCRIPT],
        self._record_user_instruction,
    )

  @staticmethod
  def _build_watchdog(embodiment: Any) -> action_watchdog.ActionWatchdog | None:
    if not getattr(embodiment, "verify_actions_visually", False):
      return None
    return action_watchdog.ActionWatchdog(
        timeout_s=float(
            getattr(embodiment, "action_timeout_s", action_watchdog.DEFAULT_TIMEOUT_S)
        ),
        max_attempts=int(
            getattr(
                embodiment, "action_max_attempts", action_watchdog.DEFAULT_MAX_ATTEMPTS
            )
        ),
        verified_actions=frozenset(
            getattr(
                embodiment, "verified_actions", action_watchdog.DEFAULT_VERIFIED_ACTIONS
            )
        ),
    )

  @property
  def watchdog(self) -> action_watchdog.ActionWatchdog | None:
    return self._watchdog

  def task_active(self) -> bool:
    """True while a plan is in progress or an action awaits confirmation."""
    if self._plan and self._plan.get("status") in ("planning", "in_progress"):
      return True
    wd = self._watchdog
    return bool(wd is not None and getattr(wd, "pending", None) is not None)

  def heartbeat_context(self) -> str:
    """Extra heartbeat instructions: plan progress + watchdog state."""
    parts: list[str] = []
    plan = self._plan
    if plan and plan.get("status") in ("planning", "in_progress"):
      steps = [str(s) for s in plan.get("steps") or []]
      cur = int(plan.get("current_step", 0) or 0)
      cur = max(0, min(cur, len(steps) - 1)) if steps else 0
      cur_text = steps[cur] if steps else ""
      nxt_text = steps[cur + 1] if cur + 1 < len(steps) else "(none - goal reached)"
      pending = self._watchdog.pending if self._watchdog is not None else None
      goal = str(plan.get("goal") or "").strip()
      if pending is not None:
        rule = (
            f"The {pending.name} command of this step is awaiting VISUAL"
            " confirmation (see WATCHDOG below). If the frame shows its"
            " effect - OR shows that this and later steps are already done"
            " (e.g. the object is already at its destination) - call"
            " confirm_action, then update_plan jumping current_step to the"
            " first step whose effect is NOT yet visible."
        )
      else:
        rule = (
            "FOLLOW THIS PLAN; do not restart or rewrite it. Judge progress"
            " against the GOAL and the latest camera frame. (a) If the scene"
            " shows this step's effect AND possibly later steps already"
            " achieved (the robot may run ahead), call update_plan with the"
            " SAME steps and current_step set to the first step whose effect"
            " is NOT yet visible (or status='done' if the goal is reached),"
            " then immediately call that step's tool - never redo finished"
            " steps. (b) If this step's tool has not been called yet, call it"
            " now; for pick/place use coordinates measured on the LATEST"
            " frame (pick: grasp_x/grasp_y, place: x/y) - if the object (before"
            " it is picked) or the destination moved since your last detect,"
            " call detect again first instead of reusing old coordinates; an"
            " object already picked moves with the hand, which is expected and"
            " never a reason to re-detect it. (c) If it is still in"
            " progress, call ack once and keep"
            " watching. (d) Small deviations (slightly different spot, object"
            " nudged) are NOT failures - carry on with the next step. Only if"
            " the scene makes the goal impossible as planned (object dropped"
            " far away, grasp clearly missed after the watchdog says so,"
            " target blocked) re-issue the tool or update_plan keeping the"
            " same goal and completed steps and revising ONLY the remaining"
            " steps. Never invent a new goal."
        )
      parts.append(
          f"[PLAN] goal = {goal!r}; step {cur + 1}/{len(steps)} ="
          f" {cur_text!r}; next = {nxt_text!r}. {rule}"
      )
    target_fn = getattr(self._embodiment, "target_context", None)
    if callable(target_fn):
      try:
        target_text = target_fn()
      except Exception:  # pylint: disable=broad-except
        logger.exception("target_context failed")
        target_text = ""
      if target_text:
        parts.append(target_text)
    if self._watchdog is not None:
      wd_text = self._watchdog.heartbeat_text()
      if wd_text:
        parts.append(wd_text)
    return "\n".join(parts)

  def shutdown(self) -> None:
    """Cancel the watchdog supervisor loop (session teardown)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is not None:
      orch.cancel()
    if self._watchdog_task is not None and not self._watchdog_task.done():
      self._watchdog_task.cancel()
    self._watchdog_task = None

  def set_session_manager(self, session_manager: Any):
    """Updates the session manager reference."""
    self._session_manager = session_manager

  def set_embodiment(self, embodiment: Any):
    """Updates the embodiment reference."""
    self._embodiment = embodiment
    self.shutdown()
    self._watchdog = self._build_watchdog(embodiment)


  async def _handle_tool_call(self, event: event_bus.Event) -> None:
    """Handle a TOOL_CALL event by executing all function calls."""
    tool_call = event.data
    function_calls = tool_call.get("functionCalls", [])
    for fc in function_calls:
      try:
        await self._execute_single_tool(fc)
      except Exception:  # pylint: disable=broad-except
        logging.exception('Tool call %s failed', fc.get("name", "unknown"))

  async def execute_for_agent(self, fc: dict) -> dict[str, Any]:
    """Run a tool for a non-Live sub-agent (e.g. the ER2 orchestrator).

    Same execution path as a Live tool call (watchdog, overlays, UI events),
    but the FunctionResponse is returned instead of sent to the Live stream.
    """
    box: dict[str, Any] = {}

    async def sink(function_response: dict[str, Any]) -> None:
      box["fr"] = function_response

    await self._execute_single_tool(fc, sink=sink)
    return box.get("fr") or {
        "id": str(fc.get("id", "")), "name": fc.get("name", ""),
        "response": {"error": "tool produced no response"},
    }

  async def _execute_single_tool(self, fc: dict, sink: Any = None) -> None:
    """Execute a single function call and publish the result."""
    func_name = fc.get("name", "")
    args = fc.get("args", {})
    call_id = str(fc.get("id", ""))
    is_blocking = func_name in self._blocking_tools
    if is_blocking:
      self._blocking_execution_count += 1
      self.tool_executing.set()
    try:
      watchdog_block = (
          self._watchdog.blocks(func_name) if self._watchdog is not None else None
      )
      if watchdog_block is not None:
        # Another command is still awaiting visual confirmation.
        result = watchdog_block
      elif func_name in {"detect", "pick", "place"}:
        if func_name == "detect":
          await self._clear_pick_target()
        stability_error = self._camera_stability_error(func_name)
        if stability_error is not None:
          if func_name in {"pick", "place"}:
            robot = getattr(self._embodiment, "robot", None)
            clear_target = getattr(robot, "clear_detected_target", None)
            if clear_target is not None:
              clear_target()
          result = stability_error
        else:
          result = await self._invoke_tool(func_name, args)
          if func_name == "detect":
            await self._publish_detected_target(result)
      else:
        result = await self._invoke_tool(func_name, args)
      if (
          self._watchdog is not None
          and watchdog_block is None
          and self._watchdog.is_verified_action(func_name)
      ):
        result = await self._arm_watchdog(func_name, args, result)
      # Embodiments without a robot-side renderer hand us UI overlay
      # payloads (draw_points / clear_overlay); publish and strip them.
      result = await self._publish_ui_overlay(result)

      # A blocking FunctionResponse resumes model generation. Send the latest
      # visual context first instead of creating a competing heartbeat turn.
      post_action_frame_sent = False
      if is_blocking and sink is None:
        send_frame = getattr(
            self._session_manager, "send_latest_video_frame", None
        )
        if send_frame is not None:
          post_action_frame_sent = bool(await send_frame())

      # Ensure result is a dict for the JSON response.
      if not isinstance(result, dict):
        result_dict = {'result': result}
      else:
        result_dict = dict(result)
      if func_name == "pick":
        result_dict = pick_response_for_model(result, post_action_frame_sent)

      # Idle acknowledgements must not wake the model up again: deliver the
      # response with SILENT scheduling so it is only appended to the
      # context. Without this, each ack response re-triggers generation and
      # the model acks again in a tight loop (~2 Hz), never completing the
      # turn and burning ~6K prompt tokens per round.
      function_response: dict[str, Any] = {
          "id": call_id,
          "name": func_name,
          "response": result_dict,
      }
      if func_name in event_bus.IDLE_TOOL_NAMES:
        # v1beta FunctionResponse.scheduling (top level) is what the Live
        # API honours; omit for models that reject function response scheduling
        # with WebSocket 1007 (e.g. gemini-3.8-live-extended-thinking).
        if getattr(self._session_manager, "supports_blocking_tools", True):
          function_response["scheduling"] = "SILENT"
          result_dict.setdefault("scheduling", "SILENT")
      else:
        user_lang = getattr(
            self._session_manager, "user_language_name", None
        )
        if user_lang:
          rule = (
              f"The user is speaking {user_lang}. Your reply (voice and text)"
              f" MUST be entirely in {user_lang}. Do NOT switch to English,"
              " German, Spanish or any other language because this tool"
              " response is written in English."
          )
        else:
          rule = (
              "Speak to the user in the EXACT language the user spoke or typed"
              " (e.g. reply in Mandarin Chinese if the user spoke Chinese;"
              " reply in English if the user spoke English). Do NOT switch"
              " language because this tool response is in English."
          )
        result_dict.setdefault("reply_language_rule", rule)

      # Send tool response back to the Gemini session as a JSON dict.
      msg = {"toolResponse": {"functionResponses": [function_response]}}

      if sink is not None:
        await sink(function_response)
      else:
        if self._session_manager is None:
          raise RuntimeError('Session manager is not initialized')
        await self._session_manager.send_message(msg)

      # Publish result for UI and logging.
      action_data = json.dumps(fc)

      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.USER,
              data={
                  "type": "tool_call",
                  "id": call_id,
                  "name": func_name,
                  "args": args,
                  "result": result,
                  "action_data": action_data,
              },
          )
      )
      logging.info(
          'Tool call %s id=%s completed with result: %s',
          func_name,
          call_id,
          result,
      )
    finally:
      if func_name in {"pick", "place"} and not getattr(
          self._embodiment, "keep_target_overlay", False
      ):
        await self._clear_pick_target()
      if is_blocking:
        self._blocking_execution_count = max(
            0, self._blocking_execution_count - 1
        )
        if self._blocking_execution_count == 0:
            self.tool_executing.clear()

  async def _record_user_instruction(self, event: event_bus.Event) -> None:
    """Track user requests: sit authorisation, stop/cancel, plan stickiness."""
    if event.type == event_bus.EventType.TEXT_INPUT:
      text = str(event.data or "")
      if text.lstrip().startswith("[HEARTBEAT]"):
        return
      self._user_transcript = text
    else:
      data = event.data or {}
      text = data.get("text", "") if isinstance(data, dict) else str(data)
      if isinstance(data, dict) and data.get("replace"):
        self._user_transcript = text
      else:
        self._user_transcript = f"{self._user_transcript} {text}".strip()
    if text.strip() and text.strip() != "🎤 正在识别语音…":
      # Any genuine user input re-opens the right to (re)plan.
      self._last_user_input_time = time.monotonic()
    self._sit_authorized = user_requested_sit(self._user_transcript)
    if self.task_active() and _USER_STOP_RE.search(text or ""):
      await self.cancel_task("user said stop")

  async def cancel_task(self, reason: str = "cancelled by user") -> dict[str, Any]:
    """Stop the current task on user request (UI button or spoken command)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is not None:
      orch.cancel()
    wd = self._watchdog
    pending_name = wd.pending.name if (wd is not None and wd.pending) else None
    if wd is not None:
      wd.reset()
    hook = getattr(self._embodiment, "on_task_aborted", None)
    if hook is not None:
      try:
        hook(pending_name)
      except Exception:  # pylint: disable=broad-except
        logging.exception("on_task_aborted hook failed")
    stop = getattr(self._embodiment, "execute_action", None)
    if stop is not None:
      try:
        await stop("stop")
      except Exception:  # pylint: disable=broad-except
        logging.debug("stop() not available on this embodiment")
    had_plan = bool(self._plan and self._plan.get("status") in ("planning", "in_progress"))
    if had_plan:
      self._plan["status"] = "failed"
      self._plan_finished_at = time.monotonic()
      plan_args = {**self._plan, "note": f"⏹ {reason}"}
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={
                  "type": "tool_call",
                  "id": "user-cancel",
                  "name": "update_plan",
                  "args": plan_args,
                  "result": {"status": "ok", "plan_status": "failed", "by": "user", "reason": reason},
                  "source": "watchdog",
              },
          )
      )
    await self._publish_action_status({"state": "cleared"})
    await self._clear_pick_target()
    # Tell the model (as a user turn, so it answers and goes idle).
    sm = self._session_manager
    send = getattr(sm, "send_text_with_fresh_video", None) or getattr(sm, "send_message", None)
    notice = (
        "[USER COMMAND] Stop. The user cancelled the current task"
        f" ({reason}). Do NOT call any more action tools for it; the plan has"
        " been marked failed/cancelled. Tell the user in ONE short sentence, in"
        " the user's language, that the task was stopped, then stay idle (only"
        " ack on heartbeats) until the user gives a new instruction."
    )
    try:
      if send is not None and send.__name__ == "send_text_with_fresh_video":
        await send(notice)
      elif send is not None:
        await send({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": notice}]}],
                "turnComplete": True,
            }
        })
    except Exception:  # pylint: disable=broad-except
      logging.exception("Failed to notify the model about task cancellation")
    logging.info("Task cancelled (%s); had_plan=%s pending=%s", reason, had_plan, pending_name)
    return {"status": "cancelled", "had_plan": had_plan, "pending": pending_name}

  def _camera_stability_error(self, action_name: str) -> dict[str, Any] | None:
    """Reject pixel-grounded manipulation until the camera is stable."""
    poller = getattr(self._embodiment, "poller", None)
    is_stable_for = getattr(poller, "is_stable_for", None)
    if is_stable_for is None or is_stable_for(_PICK_CAMERA_STABLE_SECONDS):
      return None
    stable_for = float(getattr(poller, "stable_for_seconds", 0.0))
    if action_name in {"pick", "place"}:
      next_action = action_name
    else:
      next_action = "pick or place"
    return {
        "error": (
            f"Camera view is not yet stable; {action_name} was not executed."
        ),
        "executed": False,
        "camera_stable_for_seconds": round(stable_for, 2),
        "required_stable_seconds": _PICK_CAMERA_STABLE_SECONDS,
        "retry_instruction": (
            "Wait until the hand-camera view has remained stable for three"
            " seconds and inspect the latest frame again. Call detect with an"
            f" exact target description; after detection succeeds, call {next_action}"
            " with no arguments and without moving the camera."
        ),
    }

  async def _publish_detected_target(self, result: Any) -> None:
    """Overlay the detect target (backend- or model-selected) on the UI."""
    if not isinstance(result, dict):
      return
    target = result.get("target")
    if not isinstance(target, dict):
      return
    try:
      x = float(target["normalized_x"])
      y = float(target["normalized_y"])
    except (KeyError, TypeError, ValueError):
      return
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
      return
    label = str(
        target.get("label") or result.get("label") or "detected target"
    )
    grasp_part = str(target.get("grasp_part") or result.get("grasp_part") or "").strip()
    point: dict[str, Any] = {"x": x, "y": y, "label": label, "kind": "target"}
    if grasp_part:
      point["grasp_part"] = grasp_part
    box = target.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
      try:
        point["box"] = [float(v) for v in box]
      except (TypeError, ValueError):
        pass
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "draw_points", "layer": "target", "points": [point]},
        )
    )

  async def _publish_ui_overlay(self, result: Any) -> Any:
    """Forward an embodiment-provided ``ui_overlay`` payload to the UI.

    Embodiments without a robot-side renderer (e.g. the simulated Spot)
    return ``{"ui_overlay": {"type": "draw_points"|"clear_overlay", ...}}``
    from draw_points / clear_overlay. The payload is published to the UI and
    stripped from the result sent to the model.
    """
    if not isinstance(result, dict):
      return result
    overlay = result.get("ui_overlay")
    if not isinstance(overlay, dict) or overlay.get("type") not in (
        "draw_points",
        "clear_overlay",
    ):
      return result
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data=dict(overlay),
        )
    )
    stripped = dict(result)
    stripped.pop("ui_overlay", None)
    return stripped

  async def _clear_pick_target(self) -> None:
    """Remove the pick marker (target layer only; model annotations stay)."""
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "clear_overlay", "layer": "target"},
        )
    )


  async def _invoke_tool(self, func_name: str, args: dict[str, Any]) -> Any:
    try:
      if func_name in grounding_tools.GROUNDING_TOOL_NAMES:
        sm = self._session_manager
        return await asyncio.to_thread(
            grounding_tools.run_grounding_tool,
            func_name,
            args,
            getattr(sm, "api_key", None),
            getattr(sm, "client_geolocation", None),
            getattr(sm, "user_language_name", None),
        )
      if func_name == "sit":
        if not self._sit_authorized:
          return {
              "executed": False,
              "error": "Sit rejected: the user did not explicitly ask Spot to sit.",
          }
        self._sit_authorized = False
      if func_name == "send_message":
        return await self._execute_send_message(args)
      if func_name == "update_plan":
        return await self._execute_update_plan(args)
      if func_name == "confirm_action":
        return await self._execute_confirm_action(args)
      if func_name == "retry_step":
        return await self._execute_retry_step(args)
      if func_name == "robot_task":
        return await self._execute_robot_task(args)
      return await self._embodiment.execute_action(func_name, **args)
    except Exception as exc:  # pylint: disable=broad-except
      logging.error('Tool execution error for %s: %s', func_name, exc)
      return f'Error: {exc}'


  async def _execute_update_plan(self, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and echo the plan; the UI renders it from the tool_call event.

    With an action watchdog active, the plan may not advance past the step
    whose command is still awaiting visual confirmation (the model must call
    ``confirm_action`` first). Marking the plan failed is always allowed.
    """
    steps = args.get("steps") or []
    if not isinstance(steps, list) or not steps:
      return {"error": "steps must be a non-empty list of strings"}
    try:
      current = int(args.get("current_step", 0))
    except (TypeError, ValueError):
      current = 0
    current = max(0, min(current, len(steps) - 1))
    status = str(args.get("status", "in_progress"))

    # ---- Plan stickiness (minimise spontaneous replans) --------------------
    # A plan may only change goal / restart from step 0 when the USER asked
    # for something after the current plan was started (or finished). Without
    # a new user request the original plan is kept and the model is told to
    # continue from where it is; progress never regresses.
    prev = self._plan
    clamped_from: int | None = None
    user_spoke_since_plan = (
        self._last_user_input_time > self._plan_started_at
    )
    new_goal = str(args.get("goal") or "").strip()
    if prev and status in ("planning", "in_progress") and not user_spoke_since_plan:
      prev_status = prev.get("status")
      prev_goal = str(prev.get("goal") or "").strip()
      prev_steps = [str(s) for s in prev.get("steps") or []]
      prev_cur = int(prev.get("current_step", 0) or 0)
      goal_changed = bool(new_goal) and new_goal != prev_goal
      if prev_status in ("done", "failed") and (
          time.monotonic() - self._plan_finished_at < 120.0
      ):
        # Task just ended and the user has not asked for anything new: do not
        # start a self-invented task.
        logging.info("update_plan ignored: new task without a user request")
        return {
            "status": "ignored",
            "plan_status": prev_status,
            "reason": (
                "The previous task is already"
                f" {prev_status} and the user has not given a new instruction."
                " Do not start tasks on your own."
            ),
            "next_action": "Stay idle: call ack on heartbeats until the user asks for something.",
        }
      if prev_status in ("planning", "in_progress"):
        if goal_changed:
          logging.info(
              "update_plan kept original goal %r (model proposed %r)",
              prev_goal, new_goal,
          )
          return {
              "status": "kept_original_plan",
              "plan_status": prev_status,
              "goal": prev_goal,
              "current_step": prev_cur,
              "current_step_text": prev_steps[prev_cur] if prev_steps else "",
              "reason": (
                  "The user did not ask for a different task, so the goal"
                  f" stays {prev_goal!r}. Continue with the current step; only"
                  " revise the REMAINING steps of this goal if the scene truly"
                  " requires it."
              ),
          }
        if current < prev_cur:
          # Never move backwards without a user request; keep progress.
          logging.info(
              "update_plan clamped current_step %d -> %d (no regression)",
              current, prev_cur,
          )
          clamped_from = current
          current = prev_cur
          if current >= len(steps):
            steps = prev_steps
            current = min(prev_cur, len(steps) - 1)
    if not prev or prev.get("status") in ("done", "failed"):
      self._plan_started_at = time.monotonic()
      self._plan_retries = 0
      self._retry_intent = None

    wd = self._watchdog
    if wd is not None and wd.pending is not None and status != "failed":
      p = wd.pending
      # The plan may mark the pending action's own step as current (e.g. move
      # from "detect" to "pick"), but not move beyond it or finish.
      limit = self._action_step_index(p.name, p.plan_step, [str(s) for s in steps])
      advancing = status == "done" or (limit is not None and current > limit)
      if advancing:
        return {
            "error": (
                f"Plan not advanced: {p.name} (attempt {p.attempt}/"
                f"{wd.max_attempts}) has not been visually confirmed."
            ),
            "plan_status": self._plan["status"] if self._plan else "in_progress",
            "next_action": (
                "Look at the latest camera frame. If you can SEE that the"
                f" {p.name} completed, call confirm_action(observation=...)"
                " first and then update_plan again. If nothing changed, call"
                " ack and keep watching; the watchdog will tell you when to"
                f" re-issue {p.name}."
            ),
            "watchdog": wd.status(),
        }

    goal_str = str(args.get("goal", "") or "").strip()
    self._plan = {
        "goal": goal_str,
        "steps": [str(s) for s in steps],
        "current_step": current,
        "status": status,
    }
    dm = getattr(self._session_manager, "decision_making", None)
    if (
        dm is not None
        and getattr(dm, "last_transcript_garbled", False)
        and goal_str
        and current == 0
    ):
      dm.last_transcript_garbled = False
      uid = getattr(dm, "last_garbled_utterance_id", None)
      payload: dict[str, Any] = {"text": goal_str, "replace": True}
      if uid is not None:
        payload["utterance_id"] = uid
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.USER_TRANSCRIPT,
              source=event_bus.EventSource.USER,
              data=payload,
          )
      )
    if status in ("done", "failed"):
      self._plan_finished_at = time.monotonic()
    if status in ("done", "failed") and wd is not None:
      if status == "failed" and wd.pending is not None:
        hook = getattr(self._embodiment, "on_task_aborted", None)
        if hook is not None:
          try:
            hook(wd.pending.name)
          except Exception:  # pylint: disable=broad-except
            logging.exception("on_task_aborted hook failed")
      wd.reset()
      await self._publish_action_status({"state": "cleared"})
    result_payload: dict[str, Any] = {
        "status": "ok",
        "plan_status": status,
        "total_steps": len(steps),
        "current_step": current,
        "current_step_text": steps[current],
        "instruction": (
            "Plan displayed to the user. Now execute the current step with its"
            " action tool, then visually confirm the result before calling"
            " update_plan again to advance."
            if status in ("planning", "in_progress")
            else "Plan status recorded."
        ),
    }
    if clamped_from is not None:
      result_payload["note"] = (
          f"current_step was NOT moved back to {clamped_from}: progress is"
          f" kept at step {current}. Do not tell the user you are redoing"
          " earlier steps; continue with the current step."
      )
    return result_payload

  # ---- Action watchdog (command -> visual confirmation -> retry -> abort) ----

  @staticmethod
  def _action_step_index(
      name: str, issued_step: int | None, steps: list[str]
  ) -> int | None:
    """Index of the plan step that performs ``name`` (search from issued_step).

    Steps are free text such as "pick() - grab the red block"; the first step
    at or after the one current when the command was issued that mentions the
    action name is taken as the action's own step. Falls back to issued_step.
    """
    start = max(0, issued_step or 0)
    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for idx in range(start, len(steps)):
      if pattern.search(steps[idx]):
        return idx
    return issued_step

  async def _arm_watchdog(
      self, func_name: str, args: dict[str, Any], result: Any
  ) -> Any:
    """Register a verified command with the watchdog and annotate the result."""
    wd = self._watchdog
    if wd is None or not isinstance(result, dict) or result.get("error"):
      return result
    plan_step = self._plan["current_step"] if self._plan else None
    info = wd.on_issued(func_name, args, plan_step=plan_step, source="model")
    if info.get("exhausted"):
      reason = (
          f"{func_name} was not executed after {info['max_attempts']} attempts"
          f" (no visible change within {wd.timeout_s:.0f}s each)."
      )
      await self._abort_task(reason)
      return {
          "executed": False,
          "error": reason,
          "task_status": "aborted",
          "next_action": (
              "The task was aborted. Tell the user in one short sentence,"
              " then stay idle and wait for a new instruction."
          ),
      }
    annotated = dict(result)
    annotated["watchdog"] = info
    if info.get("retargeted"):
      annotated["next_action"] = (
          f"{func_name} re-targeted to the new position (attempt"
          f" {info['attempt']}/{info['max_attempts']}, confirmation window"
          f" restarted: {info['confirm_within_s']:.0f}s). Keep watching the"
          f" camera: call confirm_action(observation=...) when you SEE the"
          f" {func_name} completed, otherwise ack."
      )
      await self._publish_action_status({**wd.status(), "state": "pending"})
      return annotated
    if info.get("merged"):
      annotated["next_action"] = (
          f"{func_name} attempt {info['attempt']}/{info['max_attempts']} is"
          f" already in flight ({info['confirm_within_s']}s left in its"
          " confirmation window); this call did not start a new attempt (any"
          " new coordinates it carried were applied). Keep"
          " watching the camera: call confirm_action(observation=...) when you"
          f" SEE the {func_name} completed, otherwise ack."
      )
      return annotated
    annotated["next_action"] = (
        f"Command sent (attempt {info['attempt']}/{info['max_attempts']})."
        " Watch the next camera frames; the arm/hand you see is the robot"
        f" executing it. When the {func_name} has VISIBLY completed call"
        " confirm_action(observation=...), then update_plan to advance. If"
        f" nothing changes within {wd.timeout_s:.0f}s the watchdog will ask"
        " you to re-issue the command. Never ask the user to do it by hand."
    )
    await self._publish_action_status(
        {**wd.status(), "state": "retry" if info["retry"] else "pending"}
    )
    self._ensure_watchdog_loop()
    return annotated

  async def _execute_robot_task(self, args: dict[str, Any]) -> dict[str, Any]:
    """Composite mode: hand a robot task to the ER orchestrator (async)."""
    orch = getattr(self._session_manager, "er_orchestrator", None)
    if orch is None:
      return {"error": "robot_task is not available in this mode."}
    instruction = str(args.get("instruction", "")).strip()
    if not instruction:
      return {"error": "instruction is required."}
    return orch.start(instruction)

  async def _execute_retry_step(self, args: dict[str, Any]) -> dict[str, Any]:
    """Escalation level 2: redo earlier step(s) of the SAME plan.

    Used only when the in-flight pick/place can no longer succeed (object
    dropped, scene changed a lot). Two-phase: the first call only records the
    intent; a second call on a later heartbeat (problem still visible)
    cancels the in-flight command, rolls back the embodiment and moves the
    plan back. At most _MAX_PLAN_RETRIES per plan; after that the model must
    keep the plan or replan the remaining steps.
    """
    plan = self._plan
    if not plan or plan.get("status") not in ("planning", "in_progress"):
      return {"status": "not_applicable", "note": "No task in progress."}
    steps = [str(x) for x in plan.get("steps") or []]
    cur = int(plan.get("current_step", 0) or 0)
    try:
      step = int(args.get("step"))
    except (TypeError, ValueError):
      return {"error": "step must be the 0-based index of the plan step to redo."}
    if step < 0 or step > cur:
      return {
          "error": f"step must be between 0 and the current step ({cur}).",
          "next_action": "Keep the plan; call ack if nothing needs redoing.",
      }
    observation = str(args.get("observation", "")).strip()
    now = time.monotonic()
    intent = self._retry_intent
    confirmed = (
        intent is not None
        and intent[0] == step
        and _RETRY_RECHECK_MIN_S <= now - intent[1] <= _RETRY_RECHECK_MAX_S
    )
    if not confirmed:
      self._retry_intent = (step, now)
      logging.info("retry_step(%d) noted, awaiting recheck: %s", step, observation)
      return {
          "status": "awaiting_recheck",
          "step": step,
          "next_action": (
              "Nothing changed yet - the plan and the command in flight are"
              " kept. Look again on the NEXT heartbeat: only if the problem is"
              " still clearly visible, call retry_step again with the same"
              " step; if the scene is fine after all, just continue (ack /"
              " confirm_action)."
          ),
      }
    self._retry_intent = None
    wd = self._watchdog
    pending = wd.pending.name if wd is not None and wd.pending is not None else None
    if pending is not None:
      wd.reset()
      hook = getattr(self._embodiment, "on_retry", None) or getattr(
          self._embodiment, "on_task_aborted", None
      )
      if hook is not None:
        try:
          hook(pending)
        except Exception:  # pylint: disable=broad-except
          logging.exception("on_retry hook failed")
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={"type": "clear_overlay", "layer": "trajectory"},
          )
      )
      await self._publish_action_status({"state": "cleared"})
    if self._plan_retries >= _MAX_PLAN_RETRIES:
      logging.info("retry_step(%d) refused: budget used; cancelled %s", step, pending)
      return {
          "status": "retry_budget_exhausted",
          "cancelled_command": pending,
          "next_action": (
              f"{_MAX_PLAN_RETRIES} retries were already used for this plan."
              " If the original remaining steps can still reach the goal,"
              " continue with the current step. Only if they truly cannot,"
              " replan: update_plan with the SAME goal, keep the completed"
              " steps and revise ONLY the remaining ones (or status='failed'"
              " if the goal is impossible), and tell the user in one sentence."
          ),
      }
    self._plan_retries += 1
    plan["current_step"] = step
    plan["status"] = "in_progress"
    note = observation or "retry"
    logging.info(
        "retry_step %d -> %d (retry %d/%d, cancelled %s): %s",
        cur, step, self._plan_retries, _MAX_PLAN_RETRIES, pending, observation,
    )
    # Let the UI plan panel follow (it renders update_plan tool events).
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={
                "type": "tool_call",
                "id": f"retry-step-{self._plan_retries}",
                "name": "update_plan",
                "args": {
                    "goal": plan.get("goal", ""),
                    "steps": steps,
                    "current_step": step,
                    "status": "in_progress",
                    "note": note,
                },
                "result": {"status": "ok", "current_step": step,
                           "plan_status": "in_progress"},
                "source": "retry",
            },
        )
    )
    return {
        "status": "retrying",
        "current_step": step,
        "current_step_text": steps[step] if step < len(steps) else "",
        "retries_used": self._plan_retries,
        "max_retries": _MAX_PLAN_RETRIES,
        "cancelled_command": pending,
        "next_action": (
            f"Same plan, same goal. Redo step {step + 1} now on the CURRENT"
            " frame (e.g. detect the object again, then pick it), then"
            " continue with the remaining steps unchanged. Tell the user in"
            " one short sentence that you are retrying."
        ),
    }

  async def _execute_confirm_action(self, args: dict[str, Any]) -> dict[str, Any]:
    wd = self._watchdog
    if wd is None:
      return {
          "status": "not_applicable",
          "note": "This embodiment does not use confirm_action; call update_plan.",
      }
    observation = str(args.get("observation", "")).strip()
    p = wd.pending
    if p is not None:
      # A confirmation is only credible once the robot had time to act AND
      # the model has seen at least one frame captured after the command.
      elapsed = wd.elapsed()
      min_dwell = min(_MIN_CONFIRM_DWELL_S, 0.2 * wd.timeout_s)
      obs = getattr(self._session_manager, "observation", None)
      last_frame = float(getattr(obs, "last_frame_sent_time", 0.0) or 0.0)
      frame_after_cmd = (
          obs is None
          or last_frame >= p.issued_at + min(_MIN_CONFIRM_FRAME_LAG_S, min_dwell)
      )
      if elapsed < min_dwell or not frame_after_cmd:
        logging.info(
            "confirm_action(%s) too early: elapsed=%.1fs frame_after_cmd=%s",
            p.name, elapsed, frame_after_cmd,
        )
        return {
            "status": "not_confirmed",
            "action": p.name,
            "elapsed_s": round(elapsed, 1),
            "reason": (
                f"Only {elapsed:.1f}s since the {p.name} command was sent and"
                " no camera frame captured after it shows the result yet. A"
                " tool result is not evidence - the physical effect must be"
                " visible."
            ),
            "next_action": (
                "Call ack now and keep watching the next camera frames. Call"
                f" confirm_action again ONLY when you can SEE that the {p.name}"
                " happened (pick: object lifted / gone from its spot; place:"
                " object resting at the target and the gripper empty). If the"
                " frames show the grasp missed or the object is not where"
                " expected, re-issue the command or update_plan with revised"
                " steps."
            ),
            "watchdog": wd.status(),
        }
    result = wd.confirm(observation)
    if result.get("status") == "confirmed":
      hook = getattr(self._embodiment, "on_action_confirmed", None)
      if hook is not None:
        try:
          hook(result["action"])
        except Exception:  # pylint: disable=broad-except
          logging.exception("on_action_confirmed hook failed")
      result["next_action"] = (
          "Call update_plan with current_step advanced to the next step"
          " (same steps list, status='in_progress') and continue."
      )
      await self._publish_action_status({
          "state": "confirmed",
          "action": result["action"],
          "attempt": result["attempts_used"],
          "max_attempts": wd.max_attempts,
          "observation": observation,
      })
    return result

  def _ensure_watchdog_loop(self) -> None:
    if self._watchdog_task is None or self._watchdog_task.done():
      self._watchdog_task = asyncio.create_task(self._watchdog_loop())

  async def _watchdog_loop(self) -> None:
    """Supervise the pending command until confirmed, retried out, or aborted."""
    wd = self._watchdog
    if wd is None:
      return
    last_notified: tuple[str, int] | None = None
    ticks = 0
    try:
      while wd.pending is not None:
        await asyncio.sleep(_WATCHDOG_TICK_S)
        ticks += 1
        p = wd.pending
        if p is None:
          break
        phase = wd.phase()
        if phase == action_watchdog.EXHAUSTED:
          await self._abort_task(
              f"{p.name} was not executed after {p.attempt} attempts"
              f" (no visible change within {wd.timeout_s:.0f}s each)."
          )
          break
        if phase == action_watchdog.TIMED_OUT:
          key = (p.name, p.attempt)
          if last_notified != key:
            last_notified = key
            # Start the grace clock here too, so the policy terminates even
            # if heartbeats (which carry the notice to the model) are delayed.
            wd.mark_timeout_notified()
            logging.info(
                "[watchdog] %s attempt %d/%d timed out; waiting for the model to"
                " re-issue", p.name, p.attempt, wd.max_attempts,
            )
            await self._publish_action_status({**wd.status(), "state": "timed_out"})
        elif phase == action_watchdog.OVERDUE:
          # The model did not re-issue within the grace period: re-send the
          # command on its behalf so the policy stays deterministic.
          await self._auto_reissue(p.name, p.args)
        elif ticks % 4 == 0:
          # Periodic sync (every ~2 s) so the UI countdown stays honest.
          await self._publish_action_status(wd.status())
    except asyncio.CancelledError:
      pass
    except Exception:  # pylint: disable=broad-except
      logging.exception("[watchdog] supervisor loop failed")

  async def _auto_reissue(self, name: str, args: dict[str, Any]) -> None:
    wd = self._watchdog
    if wd is None:
      return
    try:
      result = await self._embodiment.execute_action(name, **args)
    except Exception as exc:  # pylint: disable=broad-except
      result = {"error": f"re-issue failed: {exc}"}
    plan_step = self._plan["current_step"] if self._plan else None
    info = wd.on_issued(name, args, plan_step=plan_step, source="watchdog")
    if info.get("exhausted"):
      await self._abort_task(
          f"{name} was not executed after {info['max_attempts']} attempts"
          f" (no visible change within {wd.timeout_s:.0f}s each)."
      )
      return
    logging.info(
        "[watchdog] auto re-issued %s (attempt %d/%d)", name, info["attempt"],
        info["max_attempts"],
    )
    ui_result = dict(result) if isinstance(result, dict) else {"result": result}
    ui_result.pop("ui_overlay", None)
    ui_result["watchdog"] = info
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={
                "type": "tool_call",
                "id": f"watchdog-{name}-{info['attempt']}",
                "name": name,
                "args": {"watchdog_retry": info["attempt"]},
                "result": ui_result,
                "source": "watchdog",
            },
        )
    )
    await self._publish_action_status({**wd.status(), "state": "retry"})

  async def _abort_task(self, reason: str) -> None:
    """Watchdog gave up: mark the plan failed, notify UI, return to idle."""
    wd = self._watchdog
    if wd is None:
      return
    info = wd.fail(reason)
    logging.warning("[watchdog] aborting task: %s", reason)
    hook = getattr(self._embodiment, "on_task_aborted", None)
    if hook is not None:
      try:
        hook(info.get("action"))
      except Exception:  # pylint: disable=broad-except
        logging.exception("on_task_aborted hook failed")
    await self._publish_action_status({
        "state": "failed",
        "action": info.get("action"),
        "attempt": info.get("attempts"),
        "max_attempts": info.get("max_attempts"),
        "reason": reason,
    })
    if self._plan and self._plan.get("status") in ("planning", "in_progress"):
      self._plan["status"] = "failed"
      self._plan_finished_at = time.monotonic()
      # Point the FAILED marker at the step that performs the aborted action
      # (the model may not have advanced the plan past the preceding step).
      steps = [str(s) for s in (self._plan.get("steps") or [])]
      failed_idx = self._action_step_index(
          str(info.get("action") or ""), self._plan.get("current_step"), steps
      )
      if failed_idx is not None and 0 <= failed_idx < len(steps):
        self._plan["current_step"] = failed_idx
      plan_args = {**self._plan, "note": f"watchdog: {reason}"}
      await self._bus.publish(
          event_bus.Event(
              type=event_bus.EventType.TOOL_RESULT,
              source=event_bus.EventSource.ASSISTANT,
              data={
                  "type": "tool_call",
                  "id": "watchdog-abort",
                  "name": "update_plan",
                  "args": plan_args,
                  "result": {
                      "status": "ok",
                      "plan_status": "failed",
                      "by": "watchdog",
                      "reason": reason,
                  },
                  "source": "watchdog",
              },
          )
      )
    await self._clear_pick_target()

  async def _publish_action_status(self, status: dict[str, Any]) -> None:
    await self._bus.publish(
        event_bus.Event(
            type=event_bus.EventType.TOOL_RESULT,
            source=event_bus.EventSource.ASSISTANT,
            data={"type": "action_status", **status},
        )
    )

  async def _execute_send_message(self, args: dict[str, Any]) -> str:
    """Send a message to another robot agent via HTTP POST.

    Waits for any in-flight TTS/audio playback to finish before sending,
    so the receiving agent doesn't start speaking over the sender.
    """
    target = args.get("target", "")
    message = args.get("message", "")

    if not target:
      return "Error: 'target' is required"
    if not message:
      return "Error: 'message' is required"

    if target == "user":
      if not self._enable_send_message_to_user:
        return (
            "Error: sending messages to user is not enabled."
            " Use inline text instead."
        )
      dm = getattr(self._session_manager, "decision_making", None)
      if dm is not None and hasattr(dm, "_start_tts_task"):
        logging.info(
            "send_message to user: speaking message via DecisionMaking TTS task"
        )
        await dm._start_tts_task(message)
      elif self._text_output_callback:

        async def _run_tts():
          try:
            logging.info(
                "send_message to user: speaking message via TTS (non-blocking)"
            )
            await self._text_output_callback(message)
          except Exception as e:  # pylint: disable=broad-except
            logging.exception("TTS for send_message to user failed: %s", e)

        asyncio.create_task(_run_tts())
      return f"Message delivered to user: {message}"

    target_url = self._agent_peers.get(target)
    if not target_url:
      available = list(self._agent_peers.keys())
      return f"Error: unknown target '{target}'. Available peers: {available}"

    # Speak the message out loud via TTS before sending to peer,
    # so the user hears what is being communicated and the message
    # arrives after playback finishes.
    if self._text_output_callback:
      try:
        logging.info(
            "send_message to %s: speaking message via TTS first", target
        )
        await self._text_output_callback(message)
      except Exception as e:  # pylint: disable=broad-except
        logging.error("TTS for send_message failed: %s", e)

    try:
      async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{target_url}/api/send",
            json={"text": message, "source": self._peer_name},
            timeout=10.0,
        )
        if resp.status_code == 200:
          return f"Message delivered to {target}: {message}"
        else:
          return (
              f"Error sending to {target}: HTTP {resp.status_code} {resp.text}"
          )
    except httpx.ConnectError:
      return f"Error: could not connect to {target} at {target_url}"
    except Exception as e:  # pylint: disable=broad-except
      return f"Error sending to {target}: {e}"
