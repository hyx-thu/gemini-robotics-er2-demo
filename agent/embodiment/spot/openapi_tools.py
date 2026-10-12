"""Convert the Spot backend OpenAPI contract into Gemini tool declarations."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolPolicy:
  name: str
  description: str
  behavior: str | None = None


@dataclasses.dataclass(frozen=True)
class OpenApiOperation:
  name: str
  method: str
  path: str
  path_parameters: tuple[str, ...]
  query_parameters: tuple[str, ...]
  has_json_body: bool


# Authentication, map-file loading, registry mutation, lease release, HTML,
# visualization, and raw image operations are intentionally not model tools.
TOOL_POLICIES: dict[tuple[str, str], ToolPolicy] = {
    ("get", "/health"): ToolPolicy(
        "get_robot_status",
        "Check whether the Spot backend is connected and holding the lease. Also returns arm oscillation-monitor state. Call this before motion when control state is uncertain.",
    ),
    ("get", "/lease"): ToolPolicy(
        "get_lease_status",
        "Check whether this API server currently controls Spot through a lease. Motion tools require a lease.",
    ),
    ("post", "/lease/take"): ToolPolicy(
        "take_lease",
        "Take Spot's lease for this API server and keep it alive. This may override another controller; call only when robot control is intended.",
    ),
    ("get", "/localization"): ToolPolicy(
        "get_localization",
        "Get the current GraphNav localization, including the localized waypoint and seed transform. Use this to verify localization before navigation.",
    ),
    ("post", "/localize"): ToolPolicy(
        "localize",
        "Initialize or refresh GraphNav localization. Prefer waypoint_name for a known named location; otherwise use fiducials. Verify the result with get_localization before navigating.",
        "BLOCKING",
    ),
    ("get", "/battery"): ToolPolicy(
        "get_battery",
        "Get battery percentage, estimated remaining runtime, and motor power state. Check before long navigation or manipulation tasks.",
    ),
    ("post", "/faults/behavior/clear"): ToolPolicy(
        "clear_behavior_faults",
        "Attempt to clear all clearable behavior faults. Inspect the returned cleared and failed fault IDs before retrying motion.",
    ),
    ("get", "/waypoints"): ToolPolicy(
        "get_waypoints",
        "List the currently loaded GraphNav waypoints and their human-readable names. Always call this before navigate and use a returned name exactly.",
    ),
    ("post", "/navigate"): ToolPolicy(
        "navigate",
        "Navigate to an exact name returned by get_waypoints. The call waits for arrival or failure. Stow the arm first and ensure localization, lease, motor power, and standing state.",
        "BLOCKING",
    ),
    ("post", "/stand"): ToolPolicy(
        "stand",
        "Power motors when requested and command Spot to stand. Use before driving, navigation, or arm manipulation when Spot may be sitting.",
        "BLOCKING",
    ),
    ("post", "/sit"): ToolPolicy(
        "sit",
        "Command Spot to sit while leaving the API connection intact. Call only when the user's current instruction explicitly asks Spot to sit; never sit automatically after a task or while idle.",
        "BLOCKING",
    ),
    ("post", "/teleop/velocity"): ToolPolicy(
        "drive",
        "Drive Spot for a short duration using body-frame velocities: +x forward, +y left, and +rotation counterclockwise. Set body_follow_arm true to hold the current camera-arm joint pose relative to the moving body, preserving the aimed view during search. Prefer navigate for named destinations and use small commands near obstacles.",
    ),
    ("post", "/actions/stop"): ToolPolicy(
        "stop",
        "Immediately cancel ongoing navigation and base motion, stop the robot command stream, and optionally freeze the arm. Use whenever motion is unsafe or an action must be aborted.",
    ),
    ("get", "/images/sources"): ToolPolicy(
        "get_camera_sources",
        "List available Spot image sources with image type, dimensions, and depth scale. Use this to inspect camera capabilities and registered depth sources.",
    ),
    ("post", "/arm/deploy"): ToolPolicy(
        "deploy_arm",
        "Move the arm from stow into the ready pose for viewing or manipulation. Ensure the surrounding arm workspace is clear.",
        "BLOCKING",
    ),
    ("post", "/arm/carry"): ToolPolicy(
        "carry_arm",
        "Move the arm to the carry pose. Use to present a held object or transition before delivery; use stow_arm before navigation.",
        "BLOCKING",
    ),
    ("post", "/arm/freeze"): ToolPolicy(
        "freeze_arm",
        "Stop current arm motion and command the gripper to hold its present Cartesian pose. Use to halt oscillation or an unsafe arm trajectory.",
    ),
    ("get", "/arm/oscillation-monitor"): ToolPolicy(
        "get_oscillation_monitor",
        "Return whether automatic arm oscillation monitoring is enabled, its thresholds, sample count, and latest detection, freeze, or error.",
    ),
    ("post", "/arm/oscillation-monitor"): ToolPolicy(
        "configure_oscillation_monitor",
        "Enable or disable automatic oscillation monitoring. When enabled, repeated Cartesian direction changes above the configured thresholds cause freeze_arm to be called.",
    ),
    ("post", "/arm/jog"): ToolPolicy(
        "jog_arm",
        "Move the gripper relative to its current hand pose. Translation is expressed in hand-frame meters and rotation in radians; rotations are composed roll, then pitch, then yaw. This directly moves the arm, so use deliberate increments.",
    ),
    ("post", "/arm/camera-roll"): ToolPolicy(
        "roll_gripper_camera",
        "Roll the gripper camera around its optical viewing axis by a requested angle. Use this instead of jog_arm when the intent is specifically to rotate the camera image clockwise or counterclockwise.",
    ),
    ("post", "/arm/reach-distance"): ToolPolicy(
        "get_arm_reach_distance",
        "Measure the straight-line distance from the gripper to a target 3D pose and report whether it exceeds the arm-only reach threshold. This does not move the robot.",
    ),
    ("post", "/arm/approach"): ToolPolicy(
        "approach_pose",
        "Move only the arm toward a supplied target pose, stopping at the requested standoff. Call get_arm_reach_distance first when reachability is uncertain.",
        "BLOCKING",
    ),
    ("post", "/arm/approach-whole-body"): ToolPolicy(
        "approach_pose_whole_body",
        "Coordinate the base and arm to approach a target pose when arm-only reach is insufficient. The base moves toward the target vector, then the arm completes the approach; keep the path and workspace clear.",
        "BLOCKING",
    ),
    ("post", "/arm/stow"): ToolPolicy(
        "stow_arm",
        "Move the arm into its compact stowed pose. Always call before navigation unless carrying or presenting an object explicitly requires another pose.",
        "BLOCKING",
    ),
    ("post", "/pickup/wait"): ToolPolicy(
        "wait_for_pick_up",
        "Monitor a held item for upward gripper motion indicating recipient pickup. On detection, open the gripper, wait, close it, and stow the arm. Call from carry pose after reaching the recipient.",
        "BLOCKING",
    ),
    ("post", "/gripper/open"): ToolPolicy(
        "open_gripper",
        "Open the gripper to a fraction from 0 fully closed to 1 fully open. For drink pickup, about 0.6 leaves a narrower opening than the default full-open pose.",
    ),
    ("post", "/gripper/close"): ToolPolicy(
        "close_gripper",
        "Close the gripper toward a requested fraction, where 0 is fully closed and 1 is fully open. Use low max_vel and max_acc for a gentle grasp.",
    ),
    ("post", "/pick"): ToolPolicy(
        "pick",
        "Capture aligned hand-camera RGB and depth, use Gemini Robotics ER to identify the requested object pixel, then submit that pixel and camera calibration to Spot's native PickObjectInImage manipulation service. Spot plans the body, arm, and gripper motion. Backend completion is not proof of grasp success: inspect the fresh post-action visual input and claim success only when the requested object is visibly secured by the gripper and moved from its original location.",
        "BLOCKING",
    ),
    ("post", "/detect/pick-target"): ToolPolicy(
        "detect",
        "Run language-conditioned detection on the hand camera and return only the label, confidence, image dimensions, and normalized/pixel grasp target needed for a subsequent pick. This endpoint does not return image data, point clouds, raw model output, or a 3D scene.",
        "BLOCKING",
    ),
    ("post", "/force/change"): ToolPolicy(
        "detect_force_change",
        "Sample gripper force and report when force magnitude changes from its initial baseline by at least the threshold. Use while holding or presenting an object; this does not release the gripper automatically.",
        "BLOCKING",
    ),
}


COMMON_PARAMETER_DESCRIPTIONS = {
    "take_lease": "If true, take Spot's lease when this server does not already hold it.",
    "timeout": "Maximum seconds to wait for the command to complete.",
    "power_on": "If true, power on Spot's motors when needed.",
    "seconds": "Commanded arm motion duration in seconds.",
    "api_key": "Optional Gemini API-key override. Omit to use the server-configured key.",
    "model": "Gemini Robotics ER model name. Omit to use the server default.",
    "instruction": "Concise visual target description identifying one object, for example 'middle of red drink can'.",
    "frame_name": "Coordinate frame for the pose. Arm approach tools require 'vision'.",
    "x": "Target x coordinate in meters in frame_name.",
    "y": "Target y coordinate in meters in frame_name.",
    "z": "Target z coordinate in meters in frame_name.",
    "qw": "Quaternion scalar component for target gripper orientation.",
    "qx": "Quaternion x component for target gripper orientation.",
    "qy": "Quaternion y component for target gripper orientation.",
    "qz": "Quaternion z component for target gripper orientation.",
}


TOOL_PARAMETER_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "localize": {
        "waypoint_id": "GraphNav waypoint ID to use as the localization seed. Prefer waypoint_name when a human-readable name is known.",
        "waypoint_name": "Exact named waypoint to use as the localization seed, such as 'snack1b'.",
        "fiducial_init": "Fiducial initialization mode: 'nearest', 'nearest_at_target', or 'specific'.",
        "use_fiducial_id": "Specific AprilTag fiducial ID when fiducial_init is 'specific'.",
        "refine_fiducial_result_with_icp": "Refine fiducial localization against map point clouds using ICP.",
        "do_ambiguity_check": "Reject ambiguous localization candidates instead of selecting one automatically.",
        "refine_with_visual_features": "Refine localization using visual features after the initial estimate.",
        "verify_visual_features_quality": "Require sufficient visual-feature quality when visual refinement is enabled.",
        "max_distance": "Optional maximum translation in meters allowed during localization refinement.",
        "max_yaw": "Optional maximum yaw difference in radians allowed during localization refinement.",
    },
    "navigate": {
        "name": "Exact destination name returned by get_waypoints; never invent or normalize a name.",
        "command_duration": "Seconds assigned to each GraphNav navigation command before it is refreshed.",
        "timeout": "Overall maximum navigation time in seconds.",
        "feedback_interval": "Seconds between GraphNav feedback checks.",
        "stand": "If true, command Spot to stand before navigation.",
    },
    "drive": {
        "v_x": "Forward body velocity in meters/second; positive moves forward and negative backward.",
        "v_y": "Lateral body velocity in meters/second; positive moves left and negative right.",
        "v_rot": "Yaw velocity in radians/second; positive rotates counterclockwise.",
        "duration": "How long to apply the velocity command, in seconds.",
        "stand": "If true, stand Spot before applying velocity.",
        "body_follow_arm": "If true, hold the current arm joint pose relative to the body so the gripper camera moves with Spot.",
    },
    "stop": {
        "freeze_arm": "If true, also stop and hold the arm at its current Cartesian pose.",
    },
    "configure_oscillation_monitor": {
        "enabled": "Enable or disable automatic arm oscillation monitoring.",
        "sample_interval": "Seconds between gripper-position samples.",
        "window_sec": "Rolling analysis-window duration in seconds.",
        "min_peak_to_peak_m": "Minimum Cartesian peak-to-peak displacement in meters required to classify oscillation.",
        "min_direction_changes": "Minimum direction reversals within the window required to classify oscillation.",
        "min_speed_mps": "Ignore direction changes slower than this speed in meters/second.",
        "freeze_cooldown_sec": "Minimum seconds between automatic freeze commands.",
    },
    "jog_arm": {
        "dx": "Relative translation in meters along the hand frame's +x axis.",
        "dy": "Relative translation in meters along the hand frame's +y axis.",
        "dz": "Relative translation in meters along the hand frame's +z axis.",
        "droll": "Relative hand-frame roll in radians. Positive follows the right-hand rule.",
        "dpitch": "Relative hand-frame pitch in radians. Positive follows the right-hand rule.",
        "dyaw": "Relative hand-frame yaw in radians. Positive follows the right-hand rule.",
    },
    "roll_gripper_camera": {
        "direction": "Camera-image roll direction: 'clockwise' or 'counterclockwise'.",
        "angle_rad": "Total positive rotation magnitude in radians.",
    },
    "get_arm_reach_distance": {
        "pose": "Target gripper pose to evaluate without moving the robot.",
    },
    "approach_pose": {
        "pose": "Target gripper pose in the vision frame.",
        "standoff_m": "Distance in meters to stop before the target along the approach vector; 0 reaches the target.",
        "max_step_m": "Maximum allowed arm-only Cartesian move in meters; larger requests are rejected.",
    },
    "approach_pose_whole_body": {
        "pose": "Target gripper pose in the vision frame.",
        "standoff_m": "Distance in meters to stop before the target along the approach vector; 0 reaches the target.",
        "max_step_m": "Arm-only reach threshold in meters; when exceeded, move the base toward the target before moving the arm.",
    },
    "wait_for_pick_up": {
        "monitor_sec": "Maximum seconds to wait for the recipient to lift the held item.",
        "upward_threshold_m": "Upward gripper displacement in meters that indicates recipient pickup.",
        "sample_interval": "Seconds between gripper-position samples.",
        "open_duration_sec": "Seconds to leave the gripper open after detecting pickup.",
        "gripper_timeout": "Maximum seconds for each gripper open or close command.",
        "stow_timeout": "Maximum seconds for the final arm-stow command.",
    },
    "open_gripper": {
        "open_fraction": "Target opening fraction: 0 fully closed, 1 fully open. Use about 0.6 before grasping a drink can.",
        "max_vel": "Optional maximum gripper velocity in radians/second; lower values move more gently.",
        "max_acc": "Optional maximum gripper acceleration in radians/second squared.",
    },
    "close_gripper": {
        "open_fraction": "Target opening fraction: 0 fully closed, 1 fully open. Use 0 for a complete close.",
        "max_vel": "Optional maximum gripper velocity in radians/second; use a low value for a slow grasp.",
        "max_acc": "Optional maximum gripper acceleration in radians/second squared; use a low value for a gentle grasp.",
    },
    "pick": {
        "timeout": "Overall maximum seconds for detection, whole-body approach, and grasp.",
    },
    "detect_force_change": {
        "threshold_newtons": "Minimum change from baseline force magnitude, in newtons, required for detection.",
        "sample_window_sec": "Maximum monitoring duration in seconds.",
        "interval_sec": "Seconds between force samples.",
    },
}


def _resolve_ref(document: dict[str, Any], ref: str) -> dict[str, Any]:
  if not ref.startswith("#/"):
    raise ValueError(f"Unsupported external OpenAPI reference: {ref}")
  value: Any = document
  for part in ref[2:].split("/"):
    value = value[part.replace("~1", "/").replace("~0", "~")]
  if not isinstance(value, dict):
    raise ValueError(f"OpenAPI reference does not point to a schema: {ref}")
  return value


def _gemini_schema(
    schema: dict[str, Any] | None,
    document: dict[str, Any],
) -> dict[str, Any]:
  if not schema:
    return {"type": "OBJECT", "properties": {}}
  if "$ref" in schema:
    resolved = dict(_resolve_ref(document, schema["$ref"]))
    resolved.update({key: value for key, value in schema.items() if key != "$ref"})
    return _gemini_schema(resolved, document)

  variants = schema.get("anyOf") or schema.get("oneOf")
  if variants:
    non_null = [item for item in variants if item.get("type") != "null"]
    if len(non_null) == 1:
      merged = dict(non_null[0])
      if schema.get("description"):
        merged["description"] = schema["description"]
      return _gemini_schema(merged, document)

  result: dict[str, Any] = {}
  schema_type = schema.get("type")
  if schema_type:
    result["type"] = str(schema_type).upper()
  elif "properties" in schema:
    result["type"] = "OBJECT"

  description = schema.get("description") or schema.get("title")
  if "default" in schema:
    default_value = json.dumps(schema["default"], ensure_ascii=True)
    description = f"{description or 'Value'}. Default: {default_value}."
  if description:
    result["description"] = str(description)
  if "enum" in schema:
    result["enum"] = schema["enum"]
  for key in ("minimum", "maximum", "minItems", "maxItems"):
    if key in schema:
      result[key] = schema[key]

  if result.get("type") == "OBJECT":
    result["properties"] = {
        name: _gemini_schema(value, document)
        for name, value in schema.get("properties", {}).items()
    }
    if schema.get("required"):
      result["required"] = list(schema["required"])
  elif result.get("type") == "ARRAY":
    result["items"] = _gemini_schema(schema.get("items", {}), document)
  return result


def _apply_parameter_descriptions(
    tool_name: str,
    schema: dict[str, Any],
) -> None:
  """Apply operational docs while retaining defaults extracted from OpenAPI."""
  overrides = TOOL_PARAMETER_DESCRIPTIONS.get(tool_name, {})
  for name, property_schema in schema.get("properties", {}).items():
    description = overrides.get(name) or COMMON_PARAMETER_DESCRIPTIONS.get(name)
    if description:
      existing = property_schema.get("description", "")
      default_marker = ". Default: "
      default_note = ""
      if default_marker in existing:
        default_note = default_marker + existing.split(default_marker, 1)[1]
      base_description = description.rstrip(".")
      property_schema["description"] = (
          base_description + default_note
          if default_note
          else base_description + "."
      )
    if property_schema.get("type") == "OBJECT":
      _apply_parameter_descriptions(tool_name, property_schema)


def build_openapi_tools(
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, OpenApiOperation]]:
  """Build allowlisted Gemini declarations and dispatch metadata."""
  declarations: list[dict[str, Any]] = []
  operations: dict[str, OpenApiOperation] = {}

  for (method, path), policy in TOOL_POLICIES.items():
    operation = document.get("paths", {}).get(path, {}).get(method)
    if operation is None:
      continue

    properties: dict[str, Any] = {}
    required: list[str] = []
    path_parameters: list[str] = []
    query_parameters: list[str] = []
    for parameter in operation.get("parameters", []):
      parameter = (
          _resolve_ref(document, parameter["$ref"])
          if "$ref" in parameter
          else parameter
      )
      name = parameter["name"]
      properties[name] = _gemini_schema(parameter.get("schema"), document)
      if parameter.get("description"):
        properties[name]["description"] = parameter["description"]
      if parameter.get("required"):
        required.append(name)
      if parameter.get("in") == "path":
        path_parameters.append(name)
      elif parameter.get("in") == "query":
        query_parameters.append(name)

    content = operation.get("requestBody", {}).get("content", {})
    body_schema = content.get("application/json", {}).get("schema")
    has_json_body = body_schema is not None
    if body_schema:
      converted = _gemini_schema(body_schema, document)
      properties.update(converted.get("properties", {}))
      required.extend(converted.get("required", []))

    _apply_parameter_descriptions(
        policy.name,
        {"type": "OBJECT", "properties": properties},
    )

    parameters: dict[str, Any] = {
        "type": "OBJECT",
        "properties": properties,
    }
    if required:
      parameters["required"] = list(dict.fromkeys(required))
    declaration: dict[str, Any] = {
        "name": policy.name,
        "description": policy.description,
        "parameters": parameters,
    }
    if policy.behavior:
      declaration["behavior"] = policy.behavior
    declarations.append(declaration)
    operations[policy.name] = OpenApiOperation(
        name=policy.name,
        method=method,
        path=path,
        path_parameters=tuple(path_parameters),
        query_parameters=tuple(query_parameters),
        has_json_body=has_json_body,
    )

  return declarations, operations
