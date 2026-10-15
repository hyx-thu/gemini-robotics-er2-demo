"""Tool declarations for Proactive Agent.

All tools are declared as plain Python dicts matching the Gemini API JSON
format. No protobuf dependencies.

Tool sets return raw dicts matching the Gemini API.

Typical usage:

    from tool import tools

    my_tools = tools.robot_tools()
"""

from typing import Any


# ---------------------------------------------------------------------------
# Individual tool declarations
# ---------------------------------------------------------------------------


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    behavior: str | None = None,
) -> dict[str, Any]:
  """Helper to build a single FunctionDeclaration dict."""
  d: dict[str, Any] = {
      "name": name,
      "description": description,
      "parameters": parameters,
  }
  if behavior is not None:
    d["behavior"] = behavior
  return d


def run_instruction_tool() -> dict[str, Any]:
  """Tool to send a natural language instruction to the robot."""
  return _tool(
      name="run_instruction",
      description=(
          "When you call this tool, the robot will start executing the"
          " instruction. The robot continues executing until you call"
          " `run_instruction` again with a new instruction. Observe the scene"
          " visually to determine if the current step is complete."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "instruction": {
                  "type": "STRING",
                  "description": (
                      "A specific, unambiguous, atomic (involving single"
                      " action) natural language instruction for the robot"
                      " (e.g. 'bring the cup to the table', 'put the dice in"
                      " the green tray'). This instruction should be specific"
                      " enough to describe the exact object, e.g. 'the white"
                      " cup on the center of the table'. Always use 'pick and"
                      " place' instead of just 'pick'. Always give concrete"
                      " from and to locations. It probably should not contain"
                      " the word 'and' except in 'pick and place'."
                  ),
              },
          },
          "required": ["instruction"],
      },
      behavior="BLOCKING",
  )


def draw_points_tool() -> dict[str, Any]:
  """Tool to draw labeled point/box/path overlays on the camera image."""
  return _tool(
      name="draw_points",
      description=(
          "Draw one or more labeled point, 2D bounding-box, or connected path"
          " overlays on the camera image to highlight objects or planned paths"
          " in the scene. Coordinates are integers 0-1000 where (0,0) is"
          " top-left and (1000,1000) is bottom-right. Set connect=true when"
          " the points represent an ordered path or trajectory."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "points": {
                  "type": "ARRAY",
                  "description": "Ordered list of points (and optional boxes) to draw.",
                  "items": {
                      "type": "OBJECT",
                      "properties": {
                          "x": {
                              "type": "NUMBER",
                              "description": (
                                  "Horizontal position, 0-1000"
                                  " (0=left, 1000=right)."
                              ),
                          },
                          "y": {
                              "type": "NUMBER",
                              "description": (
                                  "Vertical position, 0-1000"
                                  " (0=top, 1000=bottom)."
                              ),
                          },
                          "label": {
                              "type": "STRING",
                              "description": (
                                  "Text label displayed next to the point."
                              ),
                          },
                          "box_2d": {
                              "type": "ARRAY",
                              "items": {"type": "INTEGER"},
                              "description": (
                                  "Optional 2D bounding box [ymin, xmin, ymax,"
                                  " xmax] in 0-1000 normalized coordinates."
                              ),
                          },
                      },
                      "required": ["x", "y"],
                  },
              },
              "connect": {
                  "type": "BOOLEAN",
                  "description": (
                      "If true, connect the points in order with directional"
                      " arrows to visualize a path or trajectory."
                  ),
              },
          },
          "required": ["points"],
      },
      behavior="NON_BLOCKING",
  )


def stop_tool() -> dict[str, Any]:
  """Tool to stop all robot motion immediately."""
  return _tool(
      name="stop",
      description="Stop all robot motion immediately.",
      parameters={"type": "OBJECT", "properties": {}},
  )


def ack_tool() -> dict[str, Any]:
  """Tool to acknowledge the heartbeat without taking action.

  Declared NON_BLOCKING on purpose: a BLOCKING ``ack`` makes the Live API
  resume generation as soon as its function response arrives, and the model
  (whose instructions say "call ack when nothing is needed") simply acks
  again — an endless ack→response→ack loop inside a single turn that never
  reaches ``turnComplete``. NON_BLOCKING lets the model close the turn right
  after acking; the ToolCallHandler additionally delivers the ack response
  with ``scheduling: SILENT`` so it never re-triggers generation. The next
  model step is then driven by the heartbeat scheduler at its configured
  interval.
  """
  return _tool(
      name="ack",
      description=(
          "Call this tool to acknowledge the regular heartbeat and indicate"
          " that no new intervention is needed. This is the default action when"
          " no other specific action is required. Call it at most once per"
          " heartbeat and then end your turn."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="NON_BLOCKING",
  )


def make_gesture_tool() -> dict[str, Any]:
  """Tool to make the robot perform a gesture."""
  return _tool(
      name="make_gesture",
      description="Make the robot perform a gesture.",
      parameters={
          "type": "OBJECT",
          "properties": {
              "gesture": {
                  "type": "STRING",
                  "description": (
                      "The name of the gesture to perform. Valid gestures"
                      " are: 'wave', 'thinking_pose', 'dance_move',"
                      " 'alive', 'left_hand_wave', 'right_hand_wave',"
                      " 'idle_nod', 'idle_hands', 'greeting_crowd',"
                      " 'look_right', 'addresses_audience',"
                      " 'addressing_audience_on_the_left',"
                      " 'addressing_the_audience_on_the_right',"
                      " 'agrees_with_someone', 'arms_wide_open',"
                      " 'both_arms_in_one_direction',"
                      " 'bringing_hands_together',"
                      " 'calming_audience_on_the_left',"
                      " 'calming_down_audience',"
                      " 'came_up_with_an_idea', 'celebrates',"
                      " 'comes_up_with_an_idea', 'disappointed',"
                      " 'greets_with_both_hands', 'hands_up',"
                      " 'has_a_question', 'is_afraid_of_something',"
                      " 'is_demanding_sth', 'looks_around',"
                      " 'looks_around_starts_by_turning_its_head_left',"
                      " 'looks_down', 'looks_up_raises_hands',"
                      " 'makes_a_point', 'open_arm_welcoming',"
                      " 'opens_its_arms_looking_at_the_scene_in_front_of_it',"
                      " 'pointing_to_the_left_and_explaining_sth',"
                      " 'points_at_something',"
                      " 'points_somewhere_and_agrees',"
                      " 'raises_hand_and_agrees', 'goodbye',"
                      " 'hello', 'hello_1', 'no',"
                      " 'showing_scary_move', 'thinks_about_sth',"
                      " 'thinks_how_to_solve_a_problem',"
                      " 'thumbs_up',"
                      " 'turns_its_head_right_and_notices_sth',"
                      " 'turns_left_and_then_pays_attention_to_sth_in_center',"
                      " 'victory_sign',"
                      " 'want_audience_to_calm_down',"
                      " 'wants_audience_to_calm_down'."
                  ),
              },
          },
          "required": ["gesture"],
      },
      behavior="NON_BLOCKING",
  )


def send_message_tool(
    enable_send_message_to_user: bool = False,
) -> dict[str, Any]:
  """Tool to send a message to another robot agent (or the user)."""
  if enable_send_message_to_user:
    desc = (
        "Send a text message to the user or another robot agent in the fleet."
        " Use this to delegate tasks or coordinate with other robots or to"
        " talk to the user."
    )
    target_desc = (
        "The target robot agent name (e.g. 'duo', 'apollo', 'user')."
    )
  else:
    desc = (
        "Send a text message to another robot agent in the fleet."
        " Use this to delegate tasks or coordinate with other robots."
    )
    target_desc = "The target robot agent name (e.g. 'duo', 'apollo')."

  return _tool(
      name="send_message",
      description=desc,
      parameters={
          "type": "OBJECT",
          "properties": {
              "target": {
                  "type": "STRING",
                  "description": target_desc,
              },
              "message": {
                  "type": "STRING",
                  "description": (
                      "The message to send. Can be a task instruction, "
                      "status update, or coordination signal. Do not send "
                      "emojis in the message."
                  ),
              },
          },
          "required": ["target", "message"],
      },
      behavior="NON_BLOCKING",
  )


def get_calendar_tool() -> dict[str, Any]:
  """Tool to retrieve upcoming events from the shared calendar."""
  return _tool(
      name="get_calendar",
      description=(
          "Retrieve upcoming events from the shared family calendar. Use this"
          " to check for planned activities."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def turn_head_to_uv_tool() -> dict[str, Any]:
  """Tool to turn the robot head to a normalized pixel coordinate."""
  return _tool(
      name="turn_head_to_uv",
      description=(
          "Turn the robot head to look at a specific point in the current"
          " camera view. Takes two arguments: `u` (horizontal, 0=left, 1=right)"
          " and `v` (vertical, 0=top, 1=bottom). Both are normalized pixel"
          " coordinates between 0 and 1."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "u": {
                  "type": "NUMBER",
                  "description": (
                      "Horizontal pixel coordinate, normalized between 0 and"
                      " 1. 0 is the left edge, 1 is the right edge."
                  ),
              },
              "v": {
                  "type": "NUMBER",
                  "description": (
                      "Vertical pixel coordinate, normalized between 0 and"
                      " 1. 0 is the top edge, 1 is the bottom edge."
                  ),
              },
          },
          "required": ["u", "v"],
      },
      behavior="NON_BLOCKING",
  )


# ---------------------------------------------------------------------------
# Tool sets — return lists of dicts wrapped in {"functionDeclarations": [...]}
# ---------------------------------------------------------------------------

def _wrap(decls: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Wrap a list of function declarations in the Gemini API format.

  Every tool set additionally receives the shared information tools
  (``get_current_location`` / ``google_search`` / ``google_maps_search``) so
  that all agents and all models can answer location / web / maps questions.
  """
  from tool import grounding_tools  # pylint: disable=g-import-not-at-top

  names = {d.get("name") for d in decls}
  merged = list(decls) + [
      d
      for d in grounding_tools.grounding_tool_declarations()
      if d.get("name") not in names
  ]
  return [{"functionDeclarations": merged}]


def robot_tools() -> list[dict[str, Any]]:
  """Motion control tools: run_instruction, stop, ack."""
  return _wrap([run_instruction_tool(), stop_tool(), ack_tool()])


def update_plan_tool() -> dict[str, Any]:
  """Tool for the model to publish / update its full task plan for the UI."""
  return _tool(
      name="update_plan",
      description=(
          "Publish or update your complete task plan so the user can see it."
          " Call this FIRST when you receive a new task (with the full ordered"
          " list of steps and current_step=0), then call it again every time"
          " a step is visually confirmed complete (advance current_step) or"
          " the plan changes (replan with a new steps list). Set status to"
          " 'done' when the overall goal is achieved or 'failed' if you give"
          " up. This tool does not move the robot; it only updates the display."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "goal": {
                  "type": "STRING",
                  "description": "One-sentence summary of the overall goal.",
              },
              "steps": {
                  "type": "ARRAY",
                  "items": {"type": "STRING"},
                  "description": (
                      "Complete ordered list of ATOMIC steps, each beginning"
                      " with exactly one tool call prefix, e.g."
                      " \"detect('red cup') - ...\", \"pick() - ...\","
                      " \"detect('tray') - ...\", \"place() - ...\","
                      " \"stow() - ...\". The gripper holds ONE object at a"
                      " time: every object that must move gets its own"
                      " detect->pick->detect->place sequence (never 'pick A and"
                      " B' or 'pick and place X' in one step; never a second"
                      " pick before the previous place). If the target is"
                      " under / inside / behind / covered by another object,"
                      " first move that occluder away with its own pick->place"
                      " pair, then handle the target. Clear an occupied"
                      " destination before placing there."
                  ),
              },
              "current_step": {
                  "type": "INTEGER",
                  "description": (
                      "0-based index of the step you are about to execute or"
                      " are currently executing."
                  ),
              },
              "status": {
                  "type": "STRING",
                  "description": "One of: planning, in_progress, done, failed.",
              },
              "note": {
                  "type": "STRING",
                  "description": (
                      "Optional short note, e.g. why you replanned or what you"
                      " observed that confirmed the last step."
                  ),
              },
          },
          "required": ["goal", "steps", "current_step", "status"],
      },
      behavior="NON_BLOCKING",
  )


def human_tools() -> list[dict[str, Any]]:
  """Human/local mode tools: run_instruction, stop, ack."""
  return _wrap([
      update_plan_tool(),
      run_instruction_tool(),
      stop_tool(),
      ack_tool(),
      send_message_tool(enable_send_message_to_user=True),
  ])


def all_tools() -> list[dict[str, Any]]:
  """All available tools (robot + UI + gesture + ack)."""
  return _wrap([
      run_instruction_tool(),
      draw_points_tool(),
      stop_tool(),
      ack_tool(),
      make_gesture_tool(),
  ])


def collab_tools() -> list[dict[str, Any]]:
  """Robot + collaboration tools for multi-agent coordination."""
  return _wrap([
      run_instruction_tool(),
      stop_tool(),
      ack_tool(),
      send_message_tool(),
  ])


def navigate_tool() -> dict[str, Any]:
  """Tool to navigate to a named waypoint."""
  return _tool(
      name="navigate",
      description=(
          "Navigate the robot to a named waypoint using GraphNav. The robot"
          " will walk to the specified waypoint and report when it arrives."
          " Call only after get_waypoints returns navigation_ready=true, use"
          " an exact name from that result, and call stow before navigating."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "waypoint": {
                  "type": "STRING",
                  "description": (
                      "Exact waypoint name from the latest successful"
                      " get_waypoints result. Never invent, normalize, or reuse"
                      " a name from an earlier session."
                  ),
              },
          },
          "required": ["waypoint"],
      },
      behavior="BLOCKING",
  )


def drive_tool() -> dict[str, Any]:
  """Tool for short body-relative movement without GraphNav waypoints."""
  return _tool(
      name="drive",
      description=(
          "Drive Spot without a waypoint for a short, bounded interval using"
          " body-frame velocities. Positive v_x moves forward, positive v_y"
          " moves left, and positive v_rot turns counterclockwise. The robot"
          " powers on and stands automatically. The current camera-arm joint"
          " pose is held relative to the body, so the gripper camera moves"
          " with Spot. Use small commands near people, furniture, stairs, or"
          " other obstacles."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "v_x": {
                  "type": "NUMBER",
                  "minimum": -0.8,
                  "maximum": 0.8,
                  "description": (
                      "Forward velocity in meters per second, from -0.8"
                      " backward to 0.8 forward."
                  ),
              },
              "v_y": {
                  "type": "NUMBER",
                  "minimum": -0.5,
                  "maximum": 0.5,
                  "description": (
                      "Sideways velocity in meters per second, from -0.5"
                      " right to 0.5 left."
                  ),
              },
              "v_rot": {
                  "type": "NUMBER",
                  "minimum": -1.0,
                  "maximum": 1.0,
                  "description": (
                      "Yaw velocity in radians per second, from -1.0"
                      " clockwise to 1.0 counterclockwise."
                  ),
              },
              "duration": {
                  "type": "NUMBER",
                  "minimum": 0.1,
                  "maximum": 2.0,
                  "description": "How long to drive, in seconds (0.1 to 2.0).",
              },
          },
          "required": ["v_x", "v_y", "v_rot", "duration"],
      },
      behavior="BLOCKING",
  )


def stop_spot_tool() -> dict[str, Any]:
  """Tool to immediately stop Spot motion and freeze the arm."""
  return _tool(
      name="stop",
      description=(
          "Immediately cancel base motion and navigation and freeze the arm."
          " Call this whenever motion is unsafe or the user asks Spot to stop."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def look_tool() -> dict[str, Any]:
  """Tool to aim the gripper camera by rotating only the arm."""
  return _tool(
      name="look",
      description=(
          "Aim Spot's gripper-mounted camera by rotating only the arm; the"
          " robot body and feet remain stationary. Use small steps and"
          " inspect the fresh camera frame returned after each move."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "direction": {
                  "type": "STRING",
                  "enum": ["up", "down", "left", "right"],
                  "description": "Direction to aim the camera view.",
              },
              "angle_rad": {
                  "type": "NUMBER",
                  "minimum": 0.05,
                  "maximum": 0.35,
                  "description": (
                      "Positive angular step in radians. Use 0.15 for a"
                      " small adjustment and 0.3 for a larger adjustment."
                  ),
              },
          },
          "required": ["direction"],
      },
      behavior="BLOCKING",
  )


def detect_tool() -> dict[str, Any]:
  """Detect an object or placement location and store a one-time target."""
  return _tool(
      name="detect",
      description=(
          "Use the Spot backend's Gemini Robotics detector to locate exactly"
          " one language-specified object or placement location in the stable"
          " hand-camera image."
          " When the user asks to pick up an object, assume it is in the"
          " current view and call this tool instead of claiming it is not"
          " visible without attempting detection."
          " The backend returns and stores the grasp location and the UI"
          " overlays it. Call only after the camera view has remained stable"
          " for at least three seconds. After a correct detection, call pick"
          " or place with no arguments before any camera or robot movement."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "instruction": {
                  "type": "STRING",
                  "description": (
                      "A concise description identifying exactly one visible"
                      ' object, such as "the red cube on the floor".'
                  ),
              },
          },
          "required": ["instruction"],
      },
      behavior="BLOCKING",
  )


def pick_tool() -> dict[str, Any]:
  """Grasp the target stored by the latest detect call."""
  return _tool(
      name="pick",
      description=(
          "Grasp the one-time location stored by the most recent successful"
          " detect call. This tool accepts no pixel coordinates. Call it only"
          " after confirming the detection overlay is on the requested object"
          " and before any robot or camera motion. The target is consumed"
          " after one attempt. A completed call still requires visual success"
          " verification. The backend reduces the gripper hold torque after"
          " the native Spot grasp succeeds."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def deploy_arm_tool() -> dict[str, Any]:
  """Deploy arm."""
  return _tool(
      name="deploy_arm",
      description=(
          "Deploy the robot arm from its stowed position. Must be called"
          " before any arm or gripper operations."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def stow_arm_tool() -> dict[str, Any]:
  """Stow arm."""
  return _tool(
      name="stow_arm",
      description=(
          "Stow the robot arm back into its carry position. Call this before"
          " navigating to avoid collisions."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def open_gripper_tool() -> dict[str, Any]:
  """Open gripper."""
  return _tool(
      name="open_gripper",
      description=(
          "Open the robot gripper. Optionally specify the fraction to open to."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "fraction": {
                  "type": "NUMBER",
                  "description": (
                      "How far to open (0=closed, 1=fully open). Default is"
                      " 1.0."
                  ),
              },
          },
      },
      behavior="NON_BLOCKING",
  )


def close_gripper_tool() -> dict[str, Any]:
  """Close gripper."""
  return _tool(
      name="close_gripper",
      description="Close the robot gripper to grasp an object.",
      parameters={"type": "OBJECT", "properties": {}},
      behavior="NON_BLOCKING",
  )


SPOT_GESTURES = ("high_wave", "shake_hand")


def spot_gesture_tool() -> dict[str, Any]:
  """Social arm gestures for greeting / introductions (Spot)."""
  return _tool(
      name="make_gesture",
      description=(
          "Perform a friendly social greeting gesture (the robot lifts a front"
          " paw; a humanoid uses its hand). Use it naturally"
          " while you talk, not as a task step:"
          " 'high_wave' = a big, high wave to say hello or goodbye - call it"
          " when the user greets you (e.g. 'hello', '你好', 'こんにちは'), when"
          " you introduce yourself or are asked who you are, and when the user"
          " says goodbye;"
          " 'shake_hand' = extend the gripper for a handshake - call it when"
          " the user offers or asks for a handshake ('let's shake hands',"
          " '握个手') or says 'nice to meet you'."
          " Call it at most once per greeting and keep talking in the same"
          " turn; never interrupt an ongoing pick/place task for a gesture"
          " unless the user explicitly asks. No visual confirmation is needed."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "gesture": {
                  "type": "STRING",
                  "enum": list(SPOT_GESTURES),
                  "description": (
                      "'high_wave' (greeting / introduction / goodbye) or"
                      " 'shake_hand' (handshake)."
                  ),
              },
          },
          "required": ["gesture"],
      },
      behavior="BLOCKING",
  )


def stand_tool() -> dict[str, Any]:
  """Stand up."""
  return _tool(
      name="stand",
      description=(
          "Power on the robot and stand up. Call this to wake the robot from"
          " sitting."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def sit_tool() -> dict[str, Any]:
  """Sit down."""
  return _tool(
      name="sit",
      description=(
          "Sit the robot down and power off its motors. Call this only when"
          " the user's current instruction explicitly asks Spot to sit; never"
          " sit automatically when a task ends or Spot becomes idle."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def get_battery_tool() -> dict[str, Any]:
  """Get battery."""
  return _tool(
      name="get_battery",
      description="Check the robot battery level and power state.",
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def get_waypoints_tool() -> dict[str, Any]:
  """Get waypoints."""
  return _tool(
      name="get_waypoints",
      description=(
          "List registered navigation destinations and report GraphNav"
          " readiness. Use only when the user asks for destinations or"
          " requests waypoint navigation; do not call at startup or for"
          " manipulation, camera, status, or local-drive tasks. If the user"
          " only asks for names, return them without discussing localization."
          " For navigation, use a returned name only when navigation_ready is"
          " true. If false, explain that GraphNav must be loaded and localized."
          " If null or absent, say readiness could not be verified; do not"
          " assume localization is missing and do not call navigate."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def health_check_tool() -> dict[str, Any]:
  """Tool to check the robot connection status, lease holder state, and battery levels."""
  return _tool(
      name="health_check",
      description=(
          "Check robot connection, lease holder, and battery state. Use when"
          " the user asks for status, before physical motion when control"
          " state is unknown, or after evidence of a connection, lease, or"
          " battery problem. Do not call automatically at startup or while"
          " idle."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def place_tool() -> dict[str, Any]:
  """Place the held object at the target stored by detect."""
  return _tool(
      name="place",
      description=(
          "Place the currently held object at the one-time location stored by"
          " the most recent successful detect call. This tool accepts no"
          " coordinates. Call it immediately after detecting the intended"
          " placement surface, before any robot or camera movement. The"
          " backend moves the hand to the projected 3D target and releases"
          " only after arm arrival is confirmed; a failed approach leaves the"
          " gripper closed."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def wait_for_pick_up_tool() -> dict[str, Any]:
  """Wait for a recipient to lift and take the held object."""
  return _tool(
      name="wait_for_pick_up",
      description=(
          "Wait for a person to pick up the object currently held by Spot."
          " Call from carry pose after reaching the recipient. The tool"
          " monitors upward hand motion; when the hand rises by the threshold,"
          " it opens the gripper, waits briefly, closes the gripper, and stows"
          " the arm. A timeout leaves the object held."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "monitor_sec": {
                  "type": "NUMBER",
                  "minimum": 1.0,
                  "maximum": 120.0,
                  "description": "Maximum seconds to wait for pickup.",
              },
              "upward_threshold_m": {
                  "type": "NUMBER",
                  "minimum": 0.005,
                  "maximum": 0.2,
                  "description": "Upward hand displacement that triggers release; default 0.02 m.",
              },
              "open_duration_sec": {
                  "type": "NUMBER",
                  "minimum": 0.1,
                  "maximum": 10.0,
                  "description": "Seconds to keep the gripper open; default 3 seconds.",
              },
          },
      },
      behavior="BLOCKING",
  )


def stow_tool() -> dict[str, Any]:
  """Tool to stow the arm safely, choosing carry pose if holding an object."""
  return _tool(
      name="stow",
      description=(
          "Stow the arm safely. If the robot is holding an object, it will"
          " move the arm to a stable carry pose to prevent collision. If it"
          " is not holding anything, it will stow the arm completely."
      ),
      parameters={"type": "OBJECT", "properties": {}},
      behavior="BLOCKING",
  )


def spot_tools() -> list[dict[str, Any]]:
  """Spot navigation + manipulation tools."""
  return _wrap([
      update_plan_tool(),
      health_check_tool(),
      get_waypoints_tool(),
      navigate_tool(),
      drive_tool(),
      stop_spot_tool(),
      look_tool(),
      detect_tool(),
      pick_tool(),
      place_tool(),
      wait_for_pick_up_tool(),
      stand_tool(),
      sit_tool(),
      stow_tool(),
      ack_tool(),
      send_message_tool(enable_send_message_to_user=True),
  ])


def detect_sim_tool() -> dict[str, Any]:
  """Simulation variant of detect: the model localizes the target and its optimal grasp point.

  There is no backend detector in webcam-in-the-loop mode, so the ER model
  must ground the requested object in the current frame, pass its full 2D
  bounding box ``box_2d``, and set ``(x, y)`` to the RECOMMENDED GRASP
  AFFORDANCE POINT (not merely the geometric box center).
  """
  return _tool(
      name="detect",
      description=(
          "Locate one language-specified object (for pick) or placement"
          " location (for place) in the CURRENT camera frame and register it"
          " as the target. There is no backend detector in this session: YOU"
          " must ground the target yourself in normalized image coordinates"
          " 0-1000 ((0,0)=top-left, (1000,1000)=bottom-right). Provide:"
          " (1) box_2d = [ymin, xmin, ymax, xmax] tightly enclosing the ENTIRE"
          " visible object or placement area; and (2) (x, y) = the OPTIMAL"
          " GRASP AFFORDANCE POINT where the parallel-jaw gripper fingers"
          " should close (for pick), or the stable release point (for place)."
          " IMPORTANT: For pick targets, (x, y) MUST be a physically"
          " graspable affordance point rather than a naive bounding-box"
          " center — e.g. the handle of a mug/cup/bag, the narrow waist or"
          " mid-body of a bottle/can (avoiding slippery caps), the solid rim"
          " edge of an open bowl/box/plate (never the hollow interior), the"
          " shaft/handle of a tool/utensil near its center of mass, or an"
          " unobstructed graspable ridge with finger clearance."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "instruction": {
                  "type": "STRING",
                  "description": (
                      "A concise description identifying the visible object"
                      ' or placement spot, such as "the white mug" or "the'
                      ' blue tray".'
                  ),
              },
              "x": {
                  "type": "INTEGER",
                  "description": (
                      "Horizontal coordinate (0=left..1000=right) of the"
                      " RECOMMENDED GRASP AFFORDANCE POINT on the object (e.g."
                      " handle, rim edge, bottle waist, tool shaft) for pick,"
                      " or stable contact point for place."
                  ),
              },
              "y": {
                  "type": "INTEGER",
                  "description": (
                      "Vertical coordinate (0=top..1000=bottom) of the"
                      " RECOMMENDED GRASP AFFORDANCE POINT on the object for"
                      " pick, or stable contact point for place."
                  ),
              },
              "box_2d": {
                  "type": "ARRAY",
                  "items": {"type": "INTEGER"},
                  "description": (
                      "2D bounding box [ymin, xmin, ymax, xmax] in 0-1000"
                      " normalized coordinates tightly enclosing the ENTIRE"
                      " target object or placement region."
                  ),
              },
              "grasp_part": {
                  "type": "STRING",
                  "description": (
                      "Short name of the specific graspable part at (x, y),"
                      " e.g. 'handle', 'rim_edge', 'bottle_waist', 'shaft',"
                      " 'pinch_ridge', 'body_center', or 'placement_surface'."
                  ),
              },
          },
          "required": ["instruction", "x", "y", "box_2d"],
      },
      behavior="BLOCKING",
  )


def confirm_action_tool() -> dict[str, Any]:
  """Report that the pending pick/place has visibly completed."""
  return _tool(
      name="confirm_action",
      description=(
          "Report that the most recent pick or place command has VISIBLY been"
          " executed by the robot, as seen in the camera frames (pick: the"
          " object is lifted / no longer at its detected spot; place: the"
          " object rests at the placement target). Call it ONLY after you see"
          " the result - never from the tool result alone. The plan cannot"
          " advance past the command's step until this is called. If nothing"
          " changes within the confirmation window, do not call this; the"
          " watchdog will ask you to re-issue the command instead."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "observation": {
                  "type": "STRING",
                  "description": (
                      "One short sentence describing the visual evidence,"
                      ' e.g. "the red block is now lifted above the table".'
                  ),
              },
          },
          "required": ["observation"],
      },
      behavior="BLOCKING",
  )


def retry_step_tool() -> dict[str, Any]:
  """Redo earlier step(s) of the SAME plan after a real failure."""
  return _tool(
      name="retry_step",
      description=(
          "Escalation for a REAL failure of the pick/place in flight (e.g. the"
          " carried object fell and lies released on a surface with no hand"
          " holding it, or the scene changed so much that the command cannot"
          " succeed). Keeps the SAME plan and goal, cancels the command in"
          " flight and moves the plan back to `step` so you redo it (usually"
          " the object's detect step, then pick and place again). Be very"
          " cautious: small deviations and an object moving WITH the hand are"
          " NOT failures. Two-phase: the first call only records the problem;"
          " call it again on the next heartbeat with the same step only if the"
          " problem is still clearly visible. Max 2 retries per plan; replan"
          " the remaining steps with update_plan only if retrying cannot reach"
          " the goal."
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "step": {
                  "type": "INTEGER",
                  "description": (
                      "0-based index of the plan step to redo (<= the current"
                      " step), e.g. the detect step of the dropped object."
                  ),
              },
              "observation": {
                  "type": "STRING",
                  "description": (
                      "One short sentence of visual evidence, in the user's"
                      " language, e.g. \"纸团掉在了桌上，夹爪是空的\"."
                  ),
              },
          },
          "required": ["step", "observation"],
      },
      behavior="BLOCKING",
  )


_WATCHDOG_NOTE = (
    " The command is supervised: watch the next camera frames - the arm or"
    " hand you see is the robot executing it. When the result is visible,"
    " call confirm_action(observation=...). If nothing changes within the"
    " confirmation window the watchdog tells you to call this tool again"
    " (max 3 attempts, then the task is aborted). Calling it again while the"
    " current attempt is still in its window does not start a new attempt."
    " Never ask the user to perform the action by hand."
)

_TRAJECTORY_SCHEMA = {
    "type": "ARRAY",
    "description": (
        "Recommended 2D gripper movement path as an ordered list of 3 to 5"
        " waypoints in 0-1000 normalized image coordinates (x: 0=left..1000="
        "right, y: 0=top..1000=bottom). For pick: pre-grasp approach aligned"
        " with the grasp affordance -> grasp at the optimal grasp point (x,y)"
        " -> post-grasp lift above obstacles. For place: current/pick"
        " position -> lift/arc waypoints clearing obstacles -> final"
        " placement target (x,y). Rendered on the camera overlay as a"
        " directed trajectory with arrows."
    ),
    "items": {
        "type": "OBJECT",
        "properties": {
            "x": {
                "type": "INTEGER",
                "description": "Horizontal coordinate 0-1000.",
            },
            "y": {
                "type": "INTEGER",
                "description": "Vertical coordinate 0-1000.",
            },
            "label": {
                "type": "STRING",
                "description": (
                    "Short step label, e.g. 'approach', 'grasp', 'lift',"
                    " 'arc', 'place'."
                ),
            },
        },
        "required": ["x", "y"],
    },
}


def pick_sim_tool() -> dict[str, Any]:
  """pick() for the webcam-in-the-loop Spot (watchdog-supervised)."""
  return _tool(
      name="pick",
      description=(
          "Grasp the target object registered by the most recent detect call"
          " at its optimal grasp affordance point and provide your"
          " recommended 2D gripper trajectory (approach -> grasp -> lift)."
          " You may optionally pass grasp_x, grasp_y, and grasp_part to"
          " refine the exact grasp affordance point (e.g. cup handle, bowl"
          " rim, bottle waist, tool shaft) if the detected point was only an"
          " object center. BEFORE calling: make sure (1) the gripper is empty"
          " (the previous object was already place()d), (2) nothing is lying"
          " on top of / covering the target — if it is, pick and place that"
          " object away first, (3) you are grasping exactly ONE object, (4)"
          " the object is still where you detected it: look at the LATEST"
          " frame and pass grasp_x/grasp_y measured on it (if the object has"
          " moved, call detect again first). The trajectory must end its"
          " 'grasp' waypoint on that current point."
          + _WATCHDOG_NOTE
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "trajectory": _TRAJECTORY_SCHEMA,
              "grasp_x": {
                  "type": "INTEGER",
                  "description": (
                      "Optional refined horizontal grasp affordance coordinate"
                      " (0-1000) where the gripper fingers should close."
                  ),
              },
              "grasp_y": {
                  "type": "INTEGER",
                  "description": (
                      "Optional refined vertical grasp affordance coordinate"
                      " (0-1000) where the gripper fingers should close."
                  ),
              },
              "grasp_part": {
                  "type": "STRING",
                  "description": (
                      "Optional name of the grasped part, e.g. 'handle',"
                      " 'rim_edge', 'bottle_waist', 'shaft'."
                  ),
              },
          },
          "required": ["trajectory"],
      },
      behavior="BLOCKING",
  )


def place_sim_tool() -> dict[str, Any]:
  """place() for the webcam-in-the-loop Spot (watchdog-supervised)."""
  return _tool(
      name="place",
      description=(
          "Place the currently held object at the one-time location stored by"
          " the most recent successful detect call and provide your"
          " recommended 2D transport/placement trajectory (from the pick/carry"
          " position -> intermediate obstacle-clearance arc waypoints ->"
          " placement target). Call it immediately after detecting the"
          " intended placement spot. The destination may have moved since"
          " detect: look at the LATEST frame and pass x/y of the placement"
          " point measured on it (if it moved a lot, call detect again"
          " first); the trajectory must end on that current point."
          + _WATCHDOG_NOTE
      ),
      parameters={
          "type": "OBJECT",
          "properties": {
              "trajectory": _TRAJECTORY_SCHEMA,
              "x": {
                  "type": "INTEGER",
                  "description": (
                      "Placement point x (0-1000) measured on the CURRENT"
                      " camera frame; overrides the earlier detect."
                  ),
              },
              "y": {
                  "type": "INTEGER",
                  "description": (
                      "Placement point y (0-1000) measured on the CURRENT"
                      " camera frame; overrides the earlier detect."
                  ),
              },
          },
          "required": ["trajectory"],
      },
      behavior="BLOCKING",
  )


def spot_sim_tools() -> list[dict[str, Any]]:
  """Spot tool surface for webcam-in-the-loop operation.

  Same as spot_tools() but detect takes model-provided coordinates, pick /
  place are supervised by the action watchdog (confirm_action closes the
  loop), and the model can annotate the camera image with draw_points /
  clear_overlay.
  """
  return _wrap([
      update_plan_tool(),
      health_check_tool(),
      get_waypoints_tool(),
      navigate_tool(),
      drive_tool(),
      stop_spot_tool(),
      look_tool(),
      detect_sim_tool(),
      pick_sim_tool(),
      place_sim_tool(),
      confirm_action_tool(),
      retry_step_tool(),
      wait_for_pick_up_tool(),
      stand_tool(),
      sit_tool(),
      spot_gesture_tool(),
      stow_tool(),
      draw_points_tool(),
      clear_overlay_tool(),
      ack_tool(),
      send_message_tool(enable_send_message_to_user=True),
  ])



def tinybot_make_gesture_tool() -> dict[str, Any]:
  """Tool to make the tinybot perform a gesture."""
  return _tool(
      name="make_gesture",
      description="Make the tinybot perform a gesture.",
      parameters={
          "type": "OBJECT",
          "properties": {
              "gesture": {
                  "type": "STRING",
                  "description": (
                      "The name of the gesture to perform. Valid gestures"
                      " are: 'nod', 'no', 'home', 'home_pose'."
                  ),
              },
              "speed": {
                  "type": "INTEGER",
                  "description": "Speed of gesture (default: 150).",
              },
          },
          "required": ["gesture"],
      },
      behavior="NON_BLOCKING",
  )


def move_absolute_tool() -> dict[str, Any]:
  """Tool to move tinybot joints to absolute angles."""
  return _tool(
      name="move_absolute",
      description="Move tinybot joints to absolute angles in degrees.",
      parameters={
          "type": "OBJECT",
          "properties": {
              "angles": {
                  "type": "ARRAY",
                  "items": {"type": "NUMBER"},
                  "description": "Target joint angles in degrees.",
              },
              "speeds": {
                  "type": "ARRAY",
                  "items": {"type": "INTEGER"},
                  "description": "Movement speeds for each joint.",
              },
              "blocking": {
                  "type": "BOOLEAN",
                  "description": (
                      "Whether to block until movement is complete (default:"
                      " true)."
                  ),
              },
          },
          "required": ["angles", "speeds"],
      },
      behavior="BLOCKING",
  )


def move_relative_tool() -> dict[str, Any]:
  """Tool to move tinybot joints by relative angles."""
  return _tool(
      name="move_relative",
      description="Move tinybot joints by relative angles in degrees.",
      parameters={
          "type": "OBJECT",
          "properties": {
              "relative_angles": {
                  "type": "ARRAY",
                  "items": {"type": "NUMBER"},
                  "description": "Relative angles in degrees.",
              },
              "speeds": {
                  "type": "ARRAY",
                  "items": {"type": "INTEGER"},
                  "description": "Movement speeds for each joint.",
              },
              "blocking": {
                  "type": "BOOLEAN",
                  "description": (
                      "Whether to block until movement is complete (default:"
                      " true)."
                  ),
              },
          },
          "required": ["relative_angles", "speeds"],
      },
      behavior="BLOCKING",
  )


def clear_overlay_tool() -> dict[str, Any]:
  """Tool to clear point overlays on the camera image."""
  return _tool(
      name="clear_overlay",
      description="Clear all drawn point overlays from the camera image.",
      parameters={"type": "OBJECT", "properties": {}},
      behavior="NON_BLOCKING",
  )


def tinybot_tools() -> list[dict[str, Any]]:
  """Tinybot motion + drawing tools."""
  return _wrap([
      update_plan_tool(),
      turn_head_to_uv_tool(),
      tinybot_make_gesture_tool(),
      move_absolute_tool(),
      move_relative_tool(),
      draw_points_tool(),
      clear_overlay_tool(),
      stop_tool(),
      ack_tool(),
      send_message_tool(enable_send_message_to_user=True),
  ])

