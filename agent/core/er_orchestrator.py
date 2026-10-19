"""ER2 robot-task orchestrator for the composite "Live + ER2" mode.

Gemini 3.8 Live talks to the user and keeps the light tools; when the user
asks for a physical task it calls ``robot_task(instruction)``. This class then
runs Gemini Robotics-ER 2 (non-streaming, generateContent) as the robot's
planner/executor:

* first turn: the task + the latest camera frame -> ER2 plans (update_plan)
  and calls detect / pick / place ...;
* every tool call is executed through the shared ToolCallHandler (same
  watchdog, overlays, plan panel, [TARGET]/escalation rules as the other
  modes) and its result goes back to ER2 with a fresh frame;
* while the task is active and ER2 is idle, a heartbeat turn (embodiment
  heartbeat text + plan/watchdog/target context + frame) is sent every few
  seconds, exactly like the streaming ER model receives them;
* ER2's send_message(user) and the final outcome are relayed to the Live
  model as short [ROBOT] notes, so the user hears them in the Live voice.

The loop ends when the plan is done / failed / cancelled, on a new
robot_task, or after a safety timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from core import event_bus
from model import generate_agent

logger = logging.getLogger(__name__)

_HEARTBEAT_S = 3.0
_MAX_TASK_S = 300.0
_MAX_ROUNDS_PER_TURN = 8
# Turns without any plan before giving up (ER2 decided there is no task).
_MAX_IDLE_TURNS_WITHOUT_PLAN = 3
_IDLE_TOOLS = frozenset({"ack"})

_TASK_PREAMBLE = (
    "[ROBOT TASK] The user asked: {instruction!r}. You are the robot's planner"
    " and executor. Follow the plan protocol: call update_plan first, then"
    " execute one step at a time with your tools on the CURRENT camera frame,"
    " verify each pick/place visually (confirm_action), and finish with"
    " update_plan(status='done'). Talk to the user only via"
    " send_message(target='user') in {language}, briefly."
)


class ErOrchestrator:
  """Runs one ER2 robot task at a time on behalf of the Live front-end."""

  def __init__(
      self,
      session: Any,
      model: str,
      api_key: str,
      system_instruction: str,
      tools: list[dict[str, Any]],
      thinking_level: str = "low",
  ):
    self._session = session
    self._model = model
    self._api_key = api_key
    self._si = system_instruction
    self._tools = tools
    self._thinking = thinking_level
    self._task: asyncio.Task | None = None
    self._instruction = ""

  @property
  def running(self) -> bool:
    return self._task is not None and not self._task.done()

  def start(self, instruction: str) -> dict[str, Any]:
    replaced = self.running
    if replaced:
      self._task.cancel()
    self._instruction = instruction
    self._task = asyncio.create_task(self._run(instruction))
    logger.info("[ER2] task %s: %s", "replaced" if replaced else "started", instruction)
    return {
        "status": "started",
        "replaced_previous_task": replaced,
        "note": (
            "The robotics planner is now executing the task; progress arrives"
            " as [ROBOT] messages. Tell the user in one short sentence that the"
            " robot is on it; do not describe steps yourself."
        ),
    }

  def cancel(self) -> None:
    if self.running:
      self._task.cancel()

  # ---- helpers ----

  def _handler(self):
    return self._session.tool_handler

  def _frame_parts(self) -> list[dict]:
    obs = getattr(self._session, "observation", None)
    chunk = getattr(obs, "latest_frame", None)
    if not chunk:
      return []
    return [generate_agent.image_part(generate_agent.jpeg_to_b64(chunk))]

  def _plan_status(self) -> str | None:
    plan = getattr(self._handler(), "_plan", None)
    return plan.get("status") if plan else None

  async def _tell_live(self, text: str) -> None:
    """Relay a robot note to the Live model (it speaks to the user)."""
    try:
      await self._session.send_message({
          "clientContent": {
              "turns": [{"role": "user", "parts": [{"text": text}]}],
              "turnComplete": True,
          }
      })
    except Exception:  # pylint: disable=broad-except
      logger.exception("[ER2] relay to Live failed")

  async def _status(self, state: str, **extra: Any) -> None:
    """UI badge / chat line: what the ER2 planner is doing right now."""
    bus = getattr(self._session, "bus", None)
    if bus is None:
      return
    await bus.publish(event_bus.Event(
        type=event_bus.EventType.TOOL_RESULT,
        source=event_bus.EventSource.ASSISTANT,
        data={"type": "er2_status", "state": state, "model": self._model,
              "instruction": self._instruction, **extra},
    ))

  async def _publish_text(self, text: str) -> None:
    # ER2 (non-streaming) sometimes returns fragments of its reasoning as
    # plain text parts ("ve", "0", "{thought ..."); show only real sentences.
    text = re.sub(r"^\W*(thought)?\W*", "", text.strip(), flags=re.IGNORECASE)
    if len(text) < 12:
      return
    bus = getattr(self._session, "bus", None)
    if bus is None:
      return
    await bus.publish(event_bus.Event(
        type=event_bus.EventType.GEMINI_THOUGHT,
        source=event_bus.EventSource.ASSISTANT,
        data={"text": f"[ER2] {text}"},
    ))

  # ---- main loop ----

  async def _run(self, instruction: str) -> None:
    loop = asyncio.get_running_loop()
    gen_cfg = {}
    if self._thinking == "none":
      gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    elif self._thinking:
      gen_cfg["thinkingConfig"] = {"thinkingLevel": self._thinking.upper()}
    agent = generate_agent.GenerateContentAgent(
        model=self._model, api_key=self._api_key,
        system_instruction=self._si, tools=self._tools,
        generation_config=gen_cfg, timeout_s=60.0,
    )
    language = getattr(self._session, "user_language_name", None) or "the user's language"
    started = time.monotonic()
    await self._status("started")
    saw_plan = False
    plan_relayed = False
    idle_without_plan = 0
    outcome = "stopped"
    parts = [{"text": _TASK_PREAMBLE.format(instruction=instruction, language=language)}]
    parts += self._frame_parts()
    try:
      while True:
        if time.monotonic() - started > _MAX_TASK_S:
          outcome = "timeout"
          break
        agent.add("user", parts)
        acted = await self._turn(agent, loop)
        status = self._plan_status()
        if status in ("planning", "in_progress"):
          saw_plan = True
          if not plan_relayed:
            plan_relayed = True
            plan = getattr(self._handler(), "_plan", None) or {}
            steps = [str(x) for x in plan.get("steps") or []]
            await self._status("executing", goal=plan.get("goal", ""),
                               steps=len(steps))
            await self._tell_live(
                "[ROBOT] The robotics planner (ER2) made this plan - goal: "
                f"{plan.get('goal', '')!r}; steps: "
                + "; ".join(f"{i + 1}. {t}" for i, t in enumerate(steps))
                + f". Tell the user in ONE short sentence in {language} that"
                " ER2 has planned it and is executing (no need to read every"
                " step). Do not call robot tools."
            )
        elif saw_plan and status in ("done", "failed"):
          outcome = status
          break
        if not saw_plan and not acted:
          idle_without_plan += 1
          if idle_without_plan >= _MAX_IDLE_TURNS_WITHOUT_PLAN:
            outcome = "no_plan"
            break
        # Idle (or acted): wait one heartbeat, then show ER2 the new state.
        await asyncio.sleep(_HEARTBEAT_S)
        hb = self._session._get_heartbeat_text()  # pylint: disable=protected-access
        parts = [{"text": hb}] + self._frame_parts()
    except asyncio.CancelledError:
      logger.info("[ER2] task cancelled: %s", instruction)
      raise
    except Exception:  # pylint: disable=broad-except
      logger.exception("[ER2] task loop failed")
      outcome = "error"
    plan = getattr(self._handler(), "_plan", None) or {}
    steps = [str(x) for x in plan.get("steps") or []]
    cur = int(plan.get("current_step", 0) or 0)
    done_steps = len(steps) if outcome == "done" else min(cur, len(steps))
    result = {
        "task": instruction,
        "status": outcome,
        "goal": plan.get("goal", ""),
        "steps_done": done_steps,
        "steps_total": len(steps),
        "stopped_at": steps[cur] if steps and outcome != "done" and cur < len(steps) else "",
        "last_note": plan.get("note", ""),
        "duration_s": round(time.monotonic() - started),
    }
    logger.info("[ER2] task finished: %s", result)
    await self._status("finished", result=result)
    await self._tell_live(
        "[ROBOT_TASK_RESULT] " + json.dumps(result, ensure_ascii=False)
        + f"\nThis is the result of your robot_task call, produced by ER2."
        f" Tell the user the outcome in one or two short sentences in"
        f" {language} (if it failed, say where it stopped and offer to try"
        " again). Do not call robot tools unless the user asks again."
    )

  async def _turn(self, agent, loop) -> bool:
    """One ER2 turn incl. tool rounds. True if any non-idle tool ran."""
    acted = False
    handler = self._handler()
    for _ in range(_MAX_ROUNDS_PER_TURN):
      out = await loop.run_in_executor(None, agent.generate)
      calls = []
      for part in out["parts"]:
        if "functionCall" in part:
          calls.append(dict(part["functionCall"]))
        elif part.get("text") and not part.get("thought"):
          await self._publish_text(part["text"])
      if not calls:
        return acted
      responses = []
      for i, fc in enumerate(calls):
        fc.setdefault("id", f"er2_{int(time.time() * 1000)}_{i}")
        fc.setdefault("args", {})
        fc["_agent"] = "er2"
        name = fc.get("name")
        if name not in _IDLE_TOOLS:
          acted = True
        fr = await handler.execute_for_agent(fc)
        if name == "send_message" and (fc["args"].get("target") == "user"):
          await self._tell_live(
              "[ROBOT] say to the user, briefly and in their language: "
              f"{fc['args'].get('message', '')}"
          )
        responses.append({"functionResponse": {
            "id": fr.get("id", fc["id"]),
            "name": fr.get("name", name),
            "response": fr.get("response") or {},
        }})
      agent.add_tool_responses(responses, self._frame_parts())
      if all(fc.get("name") in _IDLE_TOOLS for fc in calls):
        return acted
    return acted
