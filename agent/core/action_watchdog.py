"""Action watchdog: command -> visual confirmation -> retry -> abort.

The ER model drives the robot purely through tool calls (``pick``, ``place``
...). Whatever the camera shows afterwards (a robot gripper or a hand-shaped
end effector) is, by definition, the robot executing that command - the model
never asks the user to do anything by hand. If nothing visibly changes after a
command, the low-level SDK did not execute it:

* after ``timeout_s`` the command is re-issued (attempt 2, 3, ...),
* after ``max_attempts`` unanswered attempts the task is aborted and the agent
  returns to idle.

This module is a pure state machine (injectable clock, no I/O) so the policy
is deterministic and unit-testable. ``ToolCallHandler`` wires it to the tool
calls, the heartbeat text and the UI.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable


DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ATTEMPTS = 3
# How long after a timeout notice the model gets to re-issue the command
# itself before the watchdog re-issues it on the model's behalf.
DEFAULT_GRACE_S = 4.0
# Actions whose effect must be seen in the camera before the plan may advance.
DEFAULT_VERIFIED_ACTIONS = frozenset({"pick", "place"})

# Lifecycle states of a pending action (see ``ActionWatchdog.phase``).
WAITING = "waiting"        # inside the confirmation window
TIMED_OUT = "timed_out"    # window elapsed, attempts left, model should re-issue
OVERDUE = "overdue"        # model did not re-issue within the grace period
EXHAUSTED = "exhausted"    # last attempt timed out -> abort the task

# Target-position parameters of pick/place.
_TARGET_KEYS = ("grasp_x", "grasp_y", "x", "y")


def _retargeted(old: dict[str, Any] | None, new: dict[str, Any] | None) -> bool:
  """True if ``new`` carries a target position different from ``old``."""
  old, new = old or {}, new or {}
  changed = False
  for key in _TARGET_KEYS:
    if key in new and new.get(key) is not None:
      try:
        if old.get(key) is None or round(float(new[key])) != round(float(old[key])):
          changed = True
      except (TypeError, ValueError):
        continue
  return changed


@dataclasses.dataclass
class PendingAction:
  """A command that was sent to the robot and awaits visual confirmation."""

  name: str
  args: dict[str, Any]
  attempt: int
  issued_at: float
  plan_step: int | None = None
  # Set once the model has been told (via heartbeat) that this attempt timed
  # out; used to measure the grace period before an automatic re-issue.
  timeout_notified_at: float | None = None
  source: str = "model"  # who issued this attempt: "model" | "watchdog"


class ActionWatchdog:
  """Deterministic supervisor for visually verified robot commands."""

  def __init__(
      self,
      timeout_s: float = DEFAULT_TIMEOUT_S,
      max_attempts: int = DEFAULT_MAX_ATTEMPTS,
      grace_s: float = DEFAULT_GRACE_S,
      verified_actions: frozenset[str] | set[str] = DEFAULT_VERIFIED_ACTIONS,
      clock: Callable[[], float] = time.monotonic,
  ):
    self.timeout_s = float(timeout_s)
    self.max_attempts = max(1, int(max_attempts))
    self.grace_s = float(grace_s)
    self.verified_actions = frozenset(verified_actions)
    self._clock = clock
    self.pending: PendingAction | None = None
    # One-shot abort notice for the model, delivered via the next heartbeats.
    self.failure_notice: str | None = None
    self._failure_notices_left = 0
    self.last_failure: dict[str, Any] | None = None
    self.history: list[dict[str, Any]] = []

  # ---- queries ----

  def is_verified_action(self, name: str) -> bool:
    return name in self.verified_actions

  def elapsed(self) -> float:
    if self.pending is None:
      return 0.0
    return max(0.0, self._clock() - self.pending.issued_at)

  def remaining(self) -> float:
    return max(0.0, self.timeout_s - self.elapsed())

  def phase(self) -> str | None:
    """Classify the pending action by age; None when nothing is pending."""
    p = self.pending
    if p is None:
      return None
    elapsed = self.elapsed()
    if elapsed < self.timeout_s:
      return WAITING
    if p.attempt >= self.max_attempts:
      return EXHAUSTED
    if p.timeout_notified_at is None:
      return TIMED_OUT
    if self._clock() - p.timeout_notified_at < self.grace_s:
      return TIMED_OUT
    return OVERDUE

  def status(self) -> dict[str, Any]:
    """Compact status payload for the UI."""
    p = self.pending
    if p is None:
      return {"state": "idle"}
    phase = self.phase()
    return {
        "state": "pending" if phase == WAITING else phase,
        "action": p.name,
        "attempt": p.attempt,
        "max_attempts": self.max_attempts,
        "elapsed_s": round(self.elapsed(), 1),
        "timeout_s": self.timeout_s,
        "source": p.source,
    }

  # ---- transitions ----

  def blocks(self, name: str) -> dict[str, Any] | None:
    """Error payload if ``name`` may not be issued while another is pending."""
    p = self.pending
    if p is None or not self.is_verified_action(name) or name == p.name:
      return None
    return {
        "executed": False,
        "error": (
            f"{name} rejected: {p.name} (attempt {p.attempt}/"
            f"{self.max_attempts}) has not been visually confirmed yet."
        ),
        "next_action": (
            f"Watch the camera. When you SEE that the {p.name} completed, call"
            " confirm_action(observation=...) and then update_plan to advance."
            f" If nothing changes the watchdog will tell you to re-issue {p.name}."
        ),
        "watchdog": self.status(),
    }

  def on_issued(
      self,
      name: str,
      args: dict[str, Any] | None = None,
      plan_step: int | None = None,
      source: str = "model",
  ) -> dict[str, Any]:
    """Record that a verified command was sent; returns info for the result.

    A command with the same name as the pending one counts as a retry. When the
    retry budget is exhausted the pending action is NOT re-armed and the
    returned dict has ``exhausted=True``; the caller aborts the task.
    """
    now = self._clock()
    p = self.pending
    if p is not None and p.name == name:
      if self.elapsed() < self.timeout_s and _retargeted(p.args, args):
        # Same command re-issued with a NEW target position (the object /
        # destination moved and was re-detected): the arm must travel to the
        # new spot, so restart the confirmation window - same attempt - and
        # adopt the new args so a watchdog re-issue never uses the old ones.
        p.args = dict(args or {})
        p.issued_at = now
        p.timeout_notified_at = None
        p.source = source
        return {
            "action": name,
            "attempt": p.attempt,
            "max_attempts": self.max_attempts,
            "retry": False,
            "retargeted": True,
            "confirm_within_s": self.timeout_s,
            "source": source,
        }
      if self.elapsed() < self.timeout_s:
        # Same command re-sent while the current attempt is still inside its
        # confirmation window (model impatience, or the model echoing a
        # watchdog re-issue): fold it into the in-flight attempt so the retry
        # budget is never double-counted.
        return {
            "action": name,
            "attempt": p.attempt,
            "max_attempts": self.max_attempts,
            "retry": False,
            "merged": True,
            "confirm_within_s": round(self.remaining(), 1),
            "source": p.source,
        }
      attempt = p.attempt + 1
      if attempt > self.max_attempts:
        return {
            "exhausted": True,
            "action": name,
            "attempt": p.attempt,
            "max_attempts": self.max_attempts,
        }
      plan_step = p.plan_step if plan_step is None else plan_step
    else:
      attempt = 1
    self.pending = PendingAction(
        name=name,
        args=dict(args or {}),
        attempt=attempt,
        issued_at=now,
        plan_step=plan_step,
        source=source,
    )
    self.history.append({"action": name, "attempt": attempt, "t": now, "source": source})
    return {
        "action": name,
        "attempt": attempt,
        "max_attempts": self.max_attempts,
        "retry": attempt > 1,
        "confirm_within_s": self.timeout_s,
        "source": source,
    }

  def mark_timeout_notified(self) -> None:
    """The model has been told the current attempt timed out (starts grace)."""
    if self.pending is not None and self.pending.timeout_notified_at is None:
      self.pending.timeout_notified_at = self._clock()

  def confirm(self, observation: str = "") -> dict[str, Any]:
    """The model saw the command executed: clear the pending action."""
    p = self.pending
    if p is None:
      return {
          "status": "nothing_pending",
          "note": (
              "No command is awaiting confirmation. Continue with the plan;"
              " call update_plan to advance when the step is complete."
          ),
      }
    self.pending = None
    return {
        "status": "confirmed",
        "action": p.name,
        "attempts_used": p.attempt,
        "observation": observation,
    }

  def fail(self, reason: str) -> dict[str, Any]:
    """Abort: clear the pending action and queue the notice for the model."""
    p = self.pending
    info = {
        "action": p.name if p else None,
        "attempts": p.attempt if p else 0,
        "max_attempts": self.max_attempts,
        "reason": reason,
    }
    self.pending = None
    self.last_failure = info
    self.failure_notice = (
        f"[WATCHDOG] TASK ABORTED: {reason} The plan has been marked FAILED"
        " and the user can see it. Tell the user in ONE short sentence that"
        f" the {info['action']} command got no response from the robot and the"
        " task was stopped. Then stay idle: only call ack on heartbeats until"
        " the user gives a new instruction. Do NOT retry on your own and do"
        " NOT ask the user to perform the action."
    )
    self._failure_notices_left = 2
    return info

  def reset(self) -> None:
    self.pending = None
    self.failure_notice = None
    self._failure_notices_left = 0

  # ---- model-facing text ----

  def heartbeat_text(self) -> str:
    """Instructions appended to the heartbeat prompt (empty when idle)."""
    if self.pending is None:
      if self.failure_notice and self._failure_notices_left > 0:
        self._failure_notices_left -= 1
        text = self.failure_notice
        if self._failure_notices_left == 0:
          self.failure_notice = None
        return text
      return ""
    p = self.pending
    phase = self.phase()
    head = f"[WATCHDOG] {p.name} attempt {p.attempt}/{self.max_attempts}"
    if phase == WAITING:
      sent_by = "" if p.source == "model" else " (re-sent by the watchdog)"
      return (
          f"{head}{sent_by} was sent {self.elapsed():.0f}s ago; confirmation"
          f" window {self.timeout_s:.0f}s. Look at the latest camera frame:"
          " remember that any hand or gripper visible in the camera IS the"
          f" robot arm executing your command. (1) If the {p.name} has VISIBLY"
          " completed (for pick: a hand is grasping, lifting, or holding the"
          " object off the surface; for place: the object is at its"
          " destination, or the scene is already even further along), call"
          " confirm_action(observation='what you see') and then update_plan to"
          " the first step not yet done (or if a hand/arm has already entered"
          " the pick process and is touching/closing on the object right now,"
          " call ack while it lifts; NEVER re-detect or re-pick an object once"
          " a hand/arm is touching, grasping, or holding it!). (2) If NO hand"
          " or arm has entered the pick/grasp process and the [TARGET] check"
          " shows the untouched tabletop object (for pick) or destination (for"
          " place) has moved on the surface (>~60 units or outside box_2d),"
          " do NOT wait for timeout: IMMEDIATELY call detect on the current"
          f" frame with its new coordinates and re-issue {p.name} with the new"
          " position! (3) Otherwise, if the target is still at the same spot"
          " on the table and nothing has changed yet, call ack once and keep"
          f" watching (do not re-issue {p.name} to the same unchanged"
          " coordinates before timeout, and never ask the user to do it)."
      )
    if phase in (TIMED_OUT, OVERDUE):
      self.mark_timeout_notified()
      return (
          f"{head} TIMED OUT: no visible change within {self.timeout_s:.0f}s,"
          " so the robot's low-level SDK did not execute the command."
          " Look at the LATEST frame first: if a hand is ALREADY grasping,"
          " lifting, or holding the object (for pick) or the object is at its"
          " destination (for place), the command DID complete - call"
          " confirm_action(observation=...) now instead of retrying!"
          " Otherwise, if the untouched object on the surface or the"
          " destination has moved, call detect again on the current frame"
          f" first; then re-issue {p.name}() with the CURRENT coordinates"
          f" (attempt {p.attempt + 1}/{self.max_attempts}) and tell the user"
          " in one short sentence that you are retrying. Do not ask the user"
          " to perform the action."
      )
    # EXHAUSTED: the supervisor loop aborts the task; keep the model quiet.
    return (
        f"{head} TIMED OUT and no attempts remain; the task is being aborted."
        " Call ack once and wait for instructions."
    )
