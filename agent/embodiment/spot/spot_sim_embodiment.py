"""Spot embodiment over a browser webcam (no robot backend attached).

Uses the browser webcam + microphone as the robot's eyes and ears (exactly
like HumanEmbodiment) while exposing the full Spot tool set. Robot commands
are NOT dispatched to any SDK: each call is acknowledged as a *sent command*
and surfaced in the UI. From the model's point of view it is commanding a real
robot: whatever manipulator the camera shows afterwards (a robot gripper or a
hand-shaped end effector) is the robot carrying out that command, and the
model must verify the outcome visually before advancing its plan.

Verified commands (``pick`` / ``place``) are supervised by the action watchdog
(see ``core/action_watchdog.py``): if nothing visibly changes within
``action_timeout_s`` the command is re-issued, and after ``action_max_attempts``
unanswered attempts the task is aborted. Targets are therefore kept until the
command is confirmed so a retry can re-use them.

Because there is no backend detector, ``detect`` takes the model's own
normalized pixel coordinates (0-1000) for the target; they are echoed back as
a ``target`` dict so the UI overlays the point/box on the camera image and
``pick``/``place`` can report where they acted. ``draw_points`` /
``clear_overlay`` let the model annotate the camera view for the user.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from embodiment import base
from tool import tools as tools_lib

logger = logging.getLogger(__name__)

_WAYPOINTS = ["dock_station", "kitchen_counter", "office_desk", "snack_table"]

# A detect older than this is flagged to the model as possibly stale.
_STALE_TARGET_S = 6.0

# Heartbeat target check: a shift beyond this (0-1000 units) counts as moved.
_MOVED_THRESHOLD = 60
# Escalation ladder when the in-flight command can no longer succeed.
_ESCALATION = (
    " ONLY if the command clearly can no longer succeed as sent (e.g. the"
    " carried object fell and lies released on a surface with no hand"
    " holding it, or the scene changed drastically) escalate cautiously:"
    " keep the original plan and first call retry_step(step=<index of the"
    " step to redo, e.g. that object's detect step>, observation=...) - it"
    " must be confirmed by calling it again on the next heartbeat. Replan"
    " (update_plan: same goal, completed steps kept, only remaining steps"
    " revised) only if retrying cannot reach the goal."
)
# Any hand seen doing the work counts as the robot arm (webcam sim).
_ANY_HAND_IS_ARM = (
    " Any hand seen executing it - robot gripper or human hand - IS the robot"
    " arm executing the command (NEVER detect or pick an object already held in"
    " a hand)."
)

# Matches detect instructions that mistakenly target an object already held in
# a hand/gripper instead of an unpicked tabletop object or destination.
_HELD_IN_HAND_RE = re.compile(
    r"\b(?:held\s+(?:in|by)|in\s+(?:the\s+|your\s+|my\s+|a\s+)?(?:hand|gripper|fingers)"
    r"|being\s+held|already\s+(?:held|picked|lifted|in\s+hand))\b"
    r"|手中|手里|手握|拿着的|握着的|抓着的",
    re.IGNORECASE,
)

_VERIFY = (
    "Command sent to the robot. Watch the next camera frames and confirm the"
    " expected physical result before treating the step as complete."
)

_HEARTBEAT_TEXT = (
    "[HEARTBEAT] If no task is active, call 'ack' exactly once and end your"
    " turn; you will receive another heartbeat shortly. If a task is active:"
    " look at the latest camera frame and perform this 3-way check on the"
    " current step's target (see [TARGET] below): (1) HAND/ARM IN PICK PROCESS"
    " OR ACTION COMPLETED: any hand seen doing it - robot gripper or human"
    " hand - counts as the robot arm; if a hand is grasping, lifting, or"
    " holding the target object off the surface (for pick) or the object rests"
    " at the destination (for place), the command HAS ALREADY SUCCEEDED -"
    " immediately call 'confirm_action' (if pick/place is pending) and then"
    " 'update_plan' to advance (or if a hand/arm is actively touching/closing"
    " on the object right now, call 'ack' while it lifts; NEVER call detect or"
    " pick once a hand/arm has entered the pick process or holds the object!)."
    " (2) TARGET MOVED ON TABLE BEFORE GRASP: if NO hand or arm is touching,"
    " grasping, or holding the target object and the untouched target on the"
    " table (before pick or while pick is pending) or placement destination"
    " (for place) has clearly moved from its [TARGET] position (>~60 units or"
    " outside box_2d), IMMEDIATELY call 'detect' on the current frame with its"
    " new coordinates and then call 'pick' (or 'place') with the updated"
    " coordinates/trajectory - do NOT call 'ack' and do NOT wait for the 30s"
    " timeout! (3) TARGET UNCHANGED & STILL WAITING: if the target is still at"
    " its [TARGET] spot on the table and the action is still in progress, call"
    " 'ack' once and keep watching (or if pick/place has not been sent yet"
    " after detect, call pick/place now); if the goal is achieved, call"
    " update_plan(status='done') and tell the user. Never ask the user to"
    " perform a physical action."
)


class SimulatedSpotEmbodiment(base.Embodiment):
  """Spot tool surface over a browser webcam, with command-level actuation."""

  poller = None  # No robot camera poller: video comes from the browser.

  # Tell the ToolCallHandler not to wipe the target marker after pick/place:
  # the user keeps seeing where the arm was commanded to act.
  keep_target_overlay = True

  # Action watchdog policy: pick/place must be confirmed from the camera
  # within 30 s, else the command is re-issued; 3 unanswered attempts abort.
  verify_actions_visually = True
  verified_actions = frozenset({"pick", "place"})
  action_timeout_s = 30.0
  action_max_attempts = 3

  # Spot-specific idle/active heartbeat prompt (replaces the generic one).
  heartbeat_text = _HEARTBEAT_TEXT

  def __init__(self) -> None:
    self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    self.text_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    self._state: dict[str, Any] = {
        "standing": True,
        "arm": "stowed",
        "gripper_open_fraction": 1.0,
        "holding": None,
        "waypoint": _WAYPOINTS[0],
        "last_detect": None,
        # Targets of the in-flight pick/place, kept until confirmed so a
        # watchdog retry can re-use them.
        "pick_target": None,
        "place_target": None,
        "pick_origin": None,
        "last_trajectory": None,
        "issued": [],
    }
    logger.info("SimulatedSpotEmbodiment initialized (webcam-in-the-loop)")

  # ---- Queues ----

  def get_audio_queue(self) -> asyncio.Queue:
    return self.audio_queue

  def get_video_queue(self) -> asyncio.Queue:
    return self.video_queue

  def get_text_queue(self) -> asyncio.Queue:
    return self.text_queue

  # ---- Tools & SI ----

  def get_tools(self) -> list[dict[str, Any]]:
    return tools_lib.spot_sim_tools()

  def get_system_instruction(self) -> str:
    return ""

  async def initialize(self) -> None:
    return None

  async def close(self) -> None:
    return None

  # ---- Watchdog hooks ----

  def on_action_confirmed(self, action_name: str) -> None:
    """The model saw the command executed: the target is consumed."""
    s = self._state
    if action_name == "pick":
      s["pick_target"] = None
    elif action_name == "place":
      s["place_target"] = None
      s["pick_origin"] = None
      s["last_trajectory"] = None
    logger.info("[SimSpot] %s confirmed from camera", action_name)

  def on_task_aborted(self, action_name: str | None) -> None:
    """The watchdog gave up on ``action_name``: roll back the intended state."""
    s = self._state
    if action_name == "pick":
      s["holding"] = None
      s["arm"] = "stowed"
      s["gripper_open_fraction"] = 1.0
    s["pick_target"] = None
    s["place_target"] = None
    s["pick_origin"] = None
    s["last_trajectory"] = None
    s["last_detect"] = None
    logger.warning("[SimSpot] task aborted after unanswered %s", action_name)

  def on_retry(self, action_name: str | None) -> None:
    """retry_step cancelled the in-flight command: redo from an earlier step."""
    s = self._state
    if action_name in ("pick", "place"):
      # The object is not (reliably) in the gripper anymore.
      s["holding"] = None
      s["arm"] = "stowed"
      s["gripper_open_fraction"] = 1.0
    for key in ("pick_target", "place_target", "pick_origin", "last_trajectory",
                "last_detect"):
      s[key] = None
    logger.info("[SimSpot] retry: cancelled in-flight %s", action_name)

  # ---- Heartbeat target tracking ----

  @staticmethod
  def _describe_target(target: dict[str, Any]) -> str:
    desc = (
        f"{target.get('label') or 'target'!r} at (x={target.get('normalized_x')},"
        f" y={target.get('normalized_y')})"
    )
    box = target.get("box_2d")
    if box:
      desc += f" box_2d={list(box)}"
    return desc

  @staticmethod
  def _point_moved(target: dict[str, Any], x: int, y: int) -> bool:
    """True if (x, y) is outside target's box_2d or >_MOVED_THRESHOLD away."""
    tx = target.get("normalized_x")
    ty = target.get("normalized_y")
    if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
      if max(abs(x - int(tx)), abs(y - int(ty))) > _MOVED_THRESHOLD:
        return True
    box = target.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
      ymin, xmin, ymax, xmax = [int(v) for v in box]
      if x < xmin or x > xmax or y < ymin or y > ymax:
        return True
    return False

  def target_context(self) -> str:
    """Heartbeat check before and during a pick/place command.

    Monitors the active target parameter:
    - After detect() and BEFORE pick()/place() is issued (last_detect): checks
      if the detected target moved on the table before pick/place is sent.
    - While pick() or place() is in flight (pick_target / place_target): checks
      if the target object (still untouched on the table, before any hand/arm
      grasps it) or the placement destination moved, instructing the model to
      immediately re-run detect() and re-issue pick()/place().
    """
    s = self._state
    now = time.monotonic()
    last_det = s.get("last_detect")
    if s.get("place_target") is not None:
      cmd, t, at = "place", last_det or s["place_target"], (
          s.get("last_detect_at") if last_det else s.get("place_at")
      )
      what = "the DESTINATION"
      reissue = "place(x, y)"
      redetected_note = (
          " You already re-detected the moved destination at this position;"
          " if it is still there on the current frame, IMMEDIATELY call"
          " place(x, y) with the updated trajectory now!"
          if last_det is not None
          else ""
      )
      extra = (
          " The carried object moving with the hand is part of the execution,"
          " not a change of the target - never detect it again."
          + redetected_note
      )
      move_cond = (
          f"if the destination container/spot has clearly moved on the table"
          f" (outside that box or more than ~{_MOVED_THRESHOLD} units away"
          " from the coordinates above)"
      )
    elif s.get("pick_target") is not None:
      cmd, t, at = "pick", last_det or s["pick_target"], (
          s.get("last_detect_at") if last_det else s.get("pick_at")
      )
      what = "the OBJECT to grasp"
      reissue = "pick(grasp_x, grasp_y)"
      redetected_note = (
          " You already re-detected the moved object at this new position;"
          " if it is still there on the current frame (and not yet in a hand),"
          " IMMEDIATELY call pick(grasp_x, grasp_y) with the updated"
          " trajectory now!"
          if last_det is not None
          else ""
      )
      extra = (
          " (1) If a hand or gripper is grasping, lifting, or holding this"
          " object off the table (or has already entered the pick process and"
          " is touching/closing on it), do NOT re-detect: call"
          " confirm_action(observation=...) IMMEDIATELY once lifted/held (or"
          " ack while the hand finishes lifting) and NEVER call detect or pick"
          " on an object in a hand!"
          + redetected_note
      )
      move_cond = (
          "(2) If NO hand or arm has entered the pick process (the object is"
          " still lying UNTOUCHED on the table surface with no hand holding or"
          " touching it) and its position in the CURRENT camera frame has"
          f" clearly moved on the table (outside that box or more than"
          f" ~{_MOVED_THRESHOLD} units away from the coordinates above)"
      )
    elif last_det is not None:
      age = now - float(s.get("last_detect_at") or now)
      return (
          f"[TARGET] Target detected {age:.0f}s ago (before pick/place is"
          f" issued): {self._describe_target(last_det)}."
          + _ANY_HAND_IS_ARM
          + " Before issuing pick or place on this heartbeat, check the target"
          " in the CURRENT camera frame: (1) if a hand or robot arm has"
          " ALREADY entered the pick process (touching, grasping, lifting, or"
          " holding this object), NEVER call detect or pick on it (if already"
          " lifted in a hand, advance to detect the placement destination);"
          " (2) if the target is still lying UNTOUCHED on the table surface"
          " (with NO hand touching or holding it) — or is the placement"
          " destination — and has clearly moved on the table (outside that"
          f" box or more than ~{_MOVED_THRESHOLD} units away from"
          f" (x={last_det.get('normalized_x')},"
          f" y={last_det.get('normalized_y')})), you MUST IMMEDIATELY call"
          " detect for this target on the current frame with its NEW"
          " coordinates first, and then call pick(grasp_x, grasp_y) or"
          " place(x, y) with the new position - do not re-plan; (3) only if"
          " the target has NOT moved from the coordinates above, issue"
          " pick(grasp_x, grasp_y) or place(x, y) now."
      )
    else:
      return ""
    age = now - float(at or now)
    return (
        f"[TARGET] {cmd} command sent {age:.0f}s ago and being executed by the"
        f" robot arm; its target parameter is {what}"
        f" {self._describe_target(t)}." + _ANY_HAND_IS_ARM + extra
        + f" Monitor ONLY this target on the CURRENT camera frame: {move_cond},"
        " IMMEDIATELY call detect for this target alone on the current frame"
        f" and re-issue {reissue} with its NEW position (do NOT call ack and do"
        " NOT wait for the 30s timeout) - do not re-plan, go back, or detect"
        " anything else. If it has not moved on the table (or is already lifted"
        " by the hand), call confirm_action when the frame shows the command"
        " completed, otherwise ack."
        + _ESCALATION
    )

  # ---- Actuation ----

  def _log(self, name: str, kwargs: dict[str, Any]) -> None:
    self._state["issued"].append({"t": time.time(), "tool": name, "args": kwargs})
    logger.info("[SimSpot] issued %s(%s)", name, kwargs)

  @staticmethod
  def _coord(value: Any) -> int | None:
    """Parse a normalized 0-1000 coordinate; None if missing/invalid."""
    try:
      v = int(round(float(value)))
    except (TypeError, ValueError):
      return None
    return max(0, min(1000, v))

  @classmethod
  def _box(cls, value: Any) -> list[int] | None:
    """Parse an optional [ymin, xmin, ymax, xmax] box in 0-1000 coords."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
      return None
    coords = [cls._coord(v) for v in value]
    if any(c is None for c in coords):
      return None
    ymin, xmin, ymax, xmax = coords  # type: ignore[misc]
    if ymax < ymin:
      ymin, ymax = ymax, ymin
    if xmax < xmin:
      xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]

  @classmethod
  def _parse_trajectory(
      cls,
      raw_traj: Any,
      target: dict[str, Any] | None = None,
      origin: dict[str, Any] | None = None,
      action: str = "pick",
  ) -> list[dict[str, Any]]:
    """Parse model-supplied trajectory or synthesize a default 2D path."""
    pts: list[dict[str, Any]] = []
    if isinstance(raw_traj, (list, tuple)):
      for idx, item in enumerate(raw_traj):
        if isinstance(item, dict):
          x = cls._coord(item.get("x"))
          y = cls._coord(item.get("y"))
          lbl = str(item.get("label") or f"wp{idx + 1}").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
          x = cls._coord(item[0])
          y = cls._coord(item[1])
          lbl = f"wp{idx + 1}"
        else:
          continue
        if x is not None and y is not None:
          pts.append({"x": x, "y": y, "label": lbl})

    tx = int(target["normalized_x"]) if isinstance(target, dict) and "normalized_x" in target else None
    ty = int(target["normalized_y"]) if isinstance(target, dict) and "normalized_y" in target else None
    ox = int(origin["normalized_x"]) if isinstance(origin, dict) and "normalized_x" in origin else None
    oy = int(origin["normalized_y"]) if isinstance(origin, dict) and "normalized_y" in origin else None

    if len(pts) >= 2:
      if action == "place" and tx is not None and ty is not None:
        # The path starts where the carried object is NOW (``origin`` = the
        # current hand position, when known), never at the old pick spot:
        # the object has been lifted and moved since.
        if ox is not None and oy is not None:
          d0 = abs(pts[0]["x"] - ox) + abs(pts[0]["y"] - oy)
          if d0 > 60 and len(pts) < 6:
            pts.insert(0, {"x": ox, "y": oy, "label": "start (in hand)"})
        d_end = abs(pts[-1]["x"] - tx) + abs(pts[-1]["y"] - ty)
        if d_end > 60:
          pts.append({"x": tx, "y": ty, "label": "place"})
      return pts

    # Fallback synthesis when model omits trajectory or passes <2 points.
    if action == "pick" and tx is not None and ty is not None:
      return [
          {"x": tx, "y": max(40, ty - 140), "label": "approach"},
          {"x": tx, "y": ty, "label": "grasp"},
          {"x": tx, "y": max(30, ty - 190), "label": "lift"},
      ]
    if action == "place" and tx is not None and ty is not None:
      if ox is not None and oy is not None:
        apex_y = max(40, min(oy, ty) - 160)
        mid_x = (ox + tx) // 2
        return [
            {"x": ox, "y": oy, "label": "start (in hand)"},
            {"x": ox, "y": apex_y, "label": "lift"},
            {"x": mid_x, "y": max(35, apex_y - 25), "label": "arc"},
            {"x": tx, "y": ty, "label": "place"},
        ]
      return [
          {"x": max(40, tx - 140), "y": max(40, ty - 150), "label": "carry"},
          {"x": tx, "y": max(40, ty - 120), "label": "above"},
          {"x": tx, "y": ty, "label": "place"},
      ]
    return pts

  @staticmethod
  def _anchor_trajectory(
      traj: list[dict[str, Any]], target: dict[str, Any], label: str
  ) -> list[dict[str, Any]]:
    """Ensure the grasp/place waypoint lies on the CURRENT target point.

    The model may send a trajectory planned on an older frame; if the target
    was re-detected (or re-specified) elsewhere, the closing waypoint is moved
    onto the new point so the command acts where the object is now.
    """
    try:
      tx = int(target["normalized_x"])
      ty = int(target["normalized_y"])
    except (KeyError, TypeError, ValueError):
      return traj
    if not traj:
      return traj
    out = [dict(p) for p in traj]
    idx = next(
        (i for i, p in enumerate(out) if str(p.get("label", "")).lower() == label),
        len(out) - 1 if label == "place" else None,
    )
    if idx is None:
      # No labelled grasp waypoint: use the one closest to the target.
      idx = min(range(len(out)), key=lambda i: abs(out[i]["x"] - tx) + abs(out[i]["y"] - ty))
    if abs(out[idx]["x"] - tx) + abs(out[idx]["y"] - ty) > 40:
      dx, dy = tx - out[idx]["x"], ty - out[idx]["y"]
      # Shift the approach/grasp (pick) or the final approach (place) with it.
      lo = 0 if label == "grasp" else max(0, idx - 1)
      for i in range(lo, idx + 1):
        out[i]["x"] = max(0, min(1000, out[i]["x"] + dx))
        out[i]["y"] = max(0, min(1000, out[i]["y"] + dy))
      out[idx]["x"], out[idx]["y"] = tx, ty
    return out

  async def execute_action(self, action_name: str, **kwargs: Any) -> Any:
    s = self._state
    self._log(action_name, kwargs)

    if action_name == "ack":
      return "No action needed."

    if action_name == "health_check":
      return {
          "status": "SUCCESS",
          "health": {"connected": True, "has_lease": True, "standing": s["standing"]},
          "battery": {"charge_percentage": 94.0, "estimated_runtime_sec": 4800},
          "arm": s["arm"],
          "holding": s["holding"],
      }

    if action_name == "get_waypoints":
      return {
          "waypoints": [{"id": f"wp_{i}", "name": n} for i, n in enumerate(_WAYPOINTS)],
          "navigation_ready": True,
          "localized": True,
          "current_waypoint": s["waypoint"],
      }

    if action_name == "navigate":
      dest = kwargs.get("waypoint") or kwargs.get("name") or ""
      if dest not in _WAYPOINTS:
        return {"error": f"Unknown waypoint '{dest}'. Call get_waypoints first."}
      prev = s["waypoint"]
      s["waypoint"] = dest
      s["standing"] = True
      return {"status": "arrived", "from_waypoint": prev, "destination": dest, "note": _VERIFY}

    if action_name == "drive":
      return {"status": "completed", **{k: kwargs.get(k) for k in ("v_x", "v_y", "v_rot", "duration")}, "note": _VERIFY}

    if action_name == "stop":
      return {"status": "stopped", "note": "All motion commands cancelled."}

    if action_name == "look":
      return {
          "status": "completed",
          "direction": kwargs.get("direction"),
          "angle_rad": kwargs.get("angle_rad", 0.15),
          "note": _VERIFY,
      }

    if action_name == "stand":
      s["standing"] = True
      return {"status": "standing", "note": _VERIFY}

    if action_name == "sit":
      s["standing"] = False
      return {"status": "sitting", "note": _VERIFY}

    if action_name == "make_gesture":
      gesture = str(kwargs.get("gesture", "")).strip()
      if gesture not in tools_lib.SPOT_GESTURES:
        return {"error": f"Unknown gesture '{gesture}'. Use one of {list(tools_lib.SPOT_GESTURES)}."}
      return {
          "status": "completed",
          "gesture": gesture,
          "note": "Social gesture performed; no visual confirmation needed. Keep the conversation going.",
      }

    if action_name == "stow":
      s["arm"] = "carry" if s["holding"] else "stowed"
      return {"status": s["arm"], "holding": s["holding"], "note": _VERIFY}

    if action_name == "detect":
      instruction = str(kwargs.get("instruction", "")).strip()
      if not instruction:
        return {"error": "instruction is required", "detected": False}
      # No backend detector: the model grounds the object itself and passes
      # normalized pixel coordinates. We store them as the one-time target
      # (same shape as the real backend) so the UI overlays the point/box and
      # pick/place can report where they acted.
      x = self._coord(kwargs.get("x"))
      y = self._coord(kwargs.get("y"))
      box = self._box(kwargs.get("box_2d"))
      if (x is None or y is None) and box is not None:
        x = (box[1] + box[3]) // 2
        y = (box[0] + box[2]) // 2
      if x is None or y is None:
        return {
            "detected": False,
            "error": (
                "x and y are required: look at the current frame and pass the"
                " target's center in 0-1000 normalized image coordinates"
                " (x: 0=left..1000=right, y: 0=top..1000=bottom)."
            ),
        }
      if _HELD_IN_HAND_RE.search(instruction):
        # The model saw the object already in the hand/gripper and mistakenly
        # tried to detect it as a pick target. Record the hand position as the
        # carried origin for the upcoming place() and redirect the model to
        # confirm_action / detect(destination) + place().
        s["pick_origin"] = {
            "label": instruction,
            "normalized_x": x,
            "normalized_y": y,
        }
        if s.get("pick_target") is not None:
          return {
              "detected": False,
              "already_in_hand": True,
              "hand_position": [x, y],
              "error": (
                  f"'{instruction}' is ALREADY held in the hand (which IS the"
                  " robot arm executing your pick). Never detect or pick an"
                  " object that is already held in a hand!"
              ),
              "next_action": (
                  "The pick is already complete! Call"
                  " confirm_action(observation='object is lifted and held in"
                  " the hand') NOW, then call detect for the placement"
                  f" destination and place(from_x={x}, from_y={y})."
              ),
          }
        return {
            "detected": False,
            "already_in_hand": True,
            "hand_position": [x, y],
            "error": (
                f"'{instruction}' is ALREADY held in the hand (which IS the"
                " robot arm). Do NOT call pick() on an object in the hand!"
            ),
            "next_action": (
                "Skip pick() - the object is already grasped at"
                f" ({x}, {y}). Call detect for the placement destination"
                " (container/surface) and then call"
                f" place(from_x={x}, from_y={y})."
            ),
        }
      target: dict[str, Any] = {
          "label": instruction,
          "normalized_x": x,
          "normalized_y": y,
      }
      grasp_part = str(kwargs.get("grasp_part") or "").strip()
      if grasp_part:
        target["grasp_part"] = grasp_part
      if box is not None:
        target["box_2d"] = box
      s["last_detect"] = dict(target)
      s["last_detect_at"] = time.monotonic()
      if s.get("pick_target") is not None and s.get("place_target") is None:
        next_act = (
            f"Target re-detected at (x={x}, y={y}). Now IMMEDIATELY call"
            f" pick(trajectory=..., grasp_x={x}, grasp_y={y}) to redirect the"
            " in-flight pick command to this new position."
        )
      elif s.get("place_target") is not None:
        next_act = (
            f"Destination re-detected at (x={x}, y={y}). Now IMMEDIATELY call"
            f" place(trajectory=..., x={x}, y={y}) to redirect the in-flight"
            " place command to this new position."
        )
      else:
        next_act = (
            f"Check the latest camera frame: if the target has moved away from"
            f" (x={x}, y={y}) on the table (and no hand/arm has entered the"
            " pick process), call detect again with its new coordinates first;"
            " otherwise call pick or place now with your recommended 2D"
            " trajectory."
        )
      res: dict[str, Any] = {
          "detected": True,
          "instruction": instruction,
          "label": instruction,
          "target": target,
          "grasp_point": [x, y],
          "note": (
              "Target registered at the grasp affordance coordinates you"
              " provided and overlaid on the camera image. If the target"
              " moved on the table (with no hand grasping it) or the marker"
              " is not on the optimal graspable part, call detect again first."
          ),
          "next_action": next_act,
      }
      if grasp_part:
        res["grasp_part"] = grasp_part
      return res

    if action_name == "pick":
      # A fresh detect wins; otherwise re-use the in-flight target (retry).
      fresh_detect = s["last_detect"] is not None
      target = s["last_detect"] or s["pick_target"]
      gx = self._coord(kwargs.get("grasp_x"))
      gy = self._coord(kwargs.get("grasp_y"))
      if not target and gx is not None and gy is not None:
        # The model measured the grasp point on the latest frame directly.
        target = {"label": str(kwargs.get("grasp_part") or "target"),
                  "normalized_x": gx, "normalized_y": gy}
      if not target:
        return {"error": "No detected target is available. Call detect first.", "executed": False}
      retry = not fresh_detect
      target = dict(target)
      # pick() may refine / update the grasp point measured on the CURRENT
      # frame (the object may have moved since detect): it always wins.
      gpart = str(kwargs.get("grasp_part") or target.get("grasp_part") or "").strip()
      coords_updated = False
      if gx is not None and gy is not None:
        coords_updated = (
            gx != target.get("normalized_x") or gy != target.get("normalized_y")
        )
        moved_outside = self._point_moved(target, gx, gy)
        target["normalized_x"] = gx
        target["normalized_y"] = gy
        if moved_outside:
          target.pop("box_2d", None)
      if gpart:
        target["grasp_part"] = gpart
      target_age = time.monotonic() - float(s.get("last_detect_at") or time.monotonic())
      s["pick_target"] = target
      s["pick_origin"] = dict(target)
      s["pick_at"] = time.monotonic()
      s["last_detect"] = None
      s["holding"] = target["label"]
      s["arm"] = "carry"
      s["gripper_open_fraction"] = 0.2
      raw_traj = kwargs.get("trajectory")
      if retry and not raw_traj and not coords_updated and s.get("last_trajectory"):
        traj = list(s["last_trajectory"])
      else:
        traj = self._parse_trajectory(raw_traj, target=target, action="pick")
        # Make sure the grasp waypoint sits on the (possibly updated) target.
        traj = self._anchor_trajectory(traj, target, "grasp")
        s["last_trajectory"] = list(traj)
      pick_res: dict[str, Any] = {
          "status": "grasp command re-sent" if retry else "grasp command sent",
          "target_label": target["label"],
          "target": target,
          "grasp_point": [target["normalized_x"], target["normalized_y"]],
          "trajectory": traj,
          "executed": True,
          "note": _VERIFY,
          "ui_overlay": {
              "type": "draw_points",
              "layer": "trajectory",
              "connect": True,
              "action": "pick",
              "points": traj,
              "target": target,
          },
      }
      if gpart:
        pick_res["grasp_part"] = gpart
      if fresh_detect and not coords_updated and target_age > _STALE_TARGET_S:
        pick_res["warning"] = (
            f"The grasp point comes from a detect made {target_age:.0f}s ago."
            " If the object has moved since, call detect again on the latest"
            " frame before re-issuing pick."
        )
      return pick_res

    if action_name == "place":
      fresh_detect = s["last_detect"] is not None
      target = s["last_detect"] or s["place_target"]
      px = self._coord(kwargs.get("x"))
      py = self._coord(kwargs.get("y"))
      coords_updated = False
      if px is not None and py is not None:
        # Placement point measured on the CURRENT frame always wins.
        base = dict(target) if target else {"label": "placement target"}
        coords_updated = (
            px != base.get("normalized_x") or py != base.get("normalized_y")
        )
        moved_outside = self._point_moved(base, px, py)
        base["normalized_x"] = px
        base["normalized_y"] = py
        if moved_outside:
          base.pop("box_2d", None)
        target = base
      retry = not fresh_detect and not coords_updated and target is not None
      placed = s["holding"]
      origin = s.get("pick_origin")
      # Where the carried object is NOW (measured by the model on the latest
      # frame). Without it the model's own first waypoint is the start; the
      # old pick spot is only a last-resort fallback for a synthesized path.
      fx = self._coord(kwargs.get("from_x"))
      fy = self._coord(kwargs.get("from_y"))
      start: dict[str, Any] | None = None
      if fx is not None and fy is not None:
        start = {"label": f"{placed or 'object'} (in hand)",
                 "normalized_x": fx, "normalized_y": fy}
      s["place_target"] = target
      s["last_detect"] = None
      s["place_at"] = time.monotonic()
      s["holding"] = None
      s["gripper_open_fraction"] = 1.0
      raw_traj = kwargs.get("trajectory")
      if retry and not raw_traj and s.get("last_trajectory"):
        traj = list(s["last_trajectory"])
      else:
        has_path = isinstance(raw_traj, (list, tuple)) and len(raw_traj) >= 2
        traj = self._parse_trajectory(
            raw_traj, target=target,
            origin=start or (None if has_path else origin), action="place",
        )
        if target:
          traj = self._anchor_trajectory(traj, target, "place")
        s["last_trajectory"] = list(traj)
      result: dict[str, Any] = {
          "status": "place command re-sent" if retry else "place command sent",
          "placed": placed,
          "trajectory": traj,
          "executed": True,
          "note": _VERIFY,
      }
      if traj:
        result["start_point"] = [traj[0]["x"], traj[0]["y"]]
      if target:
        result["target"] = target
        result["place_point"] = [target["normalized_x"], target["normalized_y"]]
      else:
        result["warning"] = "No placement target was detected; placing at the current hand position."
      if traj:
        result["ui_overlay"] = {
            "type": "draw_points",
            "layer": "trajectory",
            "connect": True,
            "action": "place",
            "points": traj,
            "origin": start,
            "target": target,
        }
      return result

    if action_name == "wait_for_pick_up":
      released = s["holding"]
      s["holding"] = None
      s["arm"] = "stowed"
      return {
          "status": "monitoring for hand-over",
          "released_item": released,
          "monitor_sec": kwargs.get("monitor_sec", 30.0),
          "upward_threshold_m": kwargs.get("upward_threshold_m", 0.02),
          "open_duration_sec": kwargs.get("open_duration_sec", 3.0),
          "note": _VERIFY,
      }

    if action_name == "draw_points":
      points = []
      for p in kwargs.get("points") or []:
        if not isinstance(p, dict):
          continue
        x = self._coord(p.get("x"))
        y = self._coord(p.get("y"))
        if x is None or y is None:
          continue
        item: dict[str, Any] = {"x": x, "y": y, "label": str(p.get("label", ""))}
        box = self._box(p.get("box_2d") or p.get("box"))
        if box is not None:
          item["box"] = box
        points.append(item)
      if not points:
        return {"error": "points must contain at least one {x, y} entry (0-1000)."}
      connect = bool(kwargs.get("connect", False))
      return {
          "status": "ok",
          "drawn": len(points),
          "connect": connect,
          "ui_overlay": {
              "type": "draw_points",
              "layer": "annotation",
              "connect": connect,
              "points": points,
          },
      }

    if action_name == "clear_overlay":
      return {"status": "ok", "ui_overlay": {"type": "clear_overlay"}}

    return {"error": f"Unknown action: {action_name}"}
