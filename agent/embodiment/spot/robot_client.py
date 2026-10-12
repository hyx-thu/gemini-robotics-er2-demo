"""Spot robot client with Spot-specific camera and action configuration (Lite)."""

from __future__ import annotations

import logging

import httpx

from embodiment import robot_client
from embodiment.spot import openapi_tools

logger = logging.getLogger(__name__)

# Dedicated timeout for long-running operations (navigate, pick).
_LONG_TIMEOUT = httpx.Timeout(200.0)


class SpotRobotClient(robot_client.RobotClient):
  """RobotClient configured for the Boston Dynamics Spot platform.

  Spot's FastAPI server (physical-agents) exposes a different image endpoint
  than Atari: raw image bytes at GET /images/{source} rather than
  HTML-embedded base64.  This class overrides ``get_camera_snapshot`` and
  adds async helpers for every Spot-specific endpoint.
  """

  CAMERA_IDS = [
      "hand_color_image",
  ]

  # Spot does not expose MJPEG streaming endpoints.
  ENDPOINT_MAP = {}

  def __init__(
      self,
      base_url: str = "http://localhost:8888",
      timeout: float = 30.0,
  ):
    super().__init__(base_url=base_url, timeout=timeout)
    self.tool_declarations: list[dict] = []
    self.openapi_operations: dict[str, openapi_tools.OpenApiOperation] = {}
    self._detected_target: dict | None = None

  async def load_openapi_tools(self) -> list[dict]:
    """Load Gemini tools from the backend's live OpenAPI document."""
    response = await self._client.get("/openapi.json")
    response.raise_for_status()
    declarations, operations = openapi_tools.build_openapi_tools(response.json())
    if not declarations:
      raise RuntimeError("Spot OpenAPI document contained no allowlisted operations")
    self.tool_declarations = declarations
    self.openapi_operations = operations
    logger.info("Loaded %d Spot tools from OpenAPI", len(declarations))
    return declarations

  async def execute_openapi_action(self, action_name: str, **kwargs) -> dict:
    """Execute an allowlisted operation using its OpenAPI dispatch metadata."""
    operation = self.openapi_operations.get(action_name)
    if operation is None:
      raise ValueError(f"Unknown Spot OpenAPI action: {action_name}")

    path = operation.path
    remaining = dict(kwargs)
    for name in operation.path_parameters:
      if name not in remaining:
        raise ValueError(f"Missing path parameter: {name}")
      path = path.replace("{" + name + "}", str(remaining.pop(name)))

    query = {
        name: remaining.pop(name)
        for name in operation.query_parameters
        if name in remaining
    }
    request_kwargs: dict = {"params": query}
    if operation.has_json_body:
      request_kwargs["json"] = remaining
    timeout = _LONG_TIMEOUT if operation.path in {
        "/navigate",
        "/pick",
        "/pickup/wait",
        "/arm/approach",
        "/arm/approach-whole-body",
    } else self._client.timeout

    try:
      response = await self._client.request(
          operation.method, path, timeout=timeout, **request_kwargs
      )
      response.raise_for_status()
      if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
      return {"result": response.text}
    except httpx.HTTPStatusError as exc:
      try:
        detail = exc.response.json()
      except ValueError:
        detail = exc.response.text
      return {"error": f"HTTP {exc.response.status_code}", "detail": detail}
    except Exception as exc:  # pylint: disable=broad-except
      logger.error("OpenAPI action %s failed: %s", action_name, exc)
      return {"error": str(exc)}

  # ---- Camera snapshot (override) ----

  async def get_camera_snapshot(self, camera_id: str) -> bytes | None:
    """Fetch a single image from the Spot FastAPI server.

    Unlike the base class (which parses HTML-embedded base64), Spot's
    ``GET /images/{source}`` returns raw image bytes directly.
    """
    try:
      resp = await self._client.get(
          f"/images/{camera_id}",
          timeout=self._camera_timeout,
      )
      resp.raise_for_status()
      return resp.content
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("camera snapshot %s error: %s", camera_id, e)
      return None

  # ---- Navigation ----

  async def navigate(
      self,
      waypoint: str,
      timeout: float = 180,
      power_on: bool = True,
      stand: bool = True,
      take_lease: bool = True,
  ) -> dict:
    """Navigate to a recorded waypoint by name."""
    try:
      resp = await self._client.post(
          "/navigate",
          json={
              "name": waypoint,
              "timeout": timeout,
              "power_on": power_on,
              "stand": stand,
              "take_lease": take_lease,
          },
          timeout=_LONG_TIMEOUT,
      )
      resp.raise_for_status()
      return resp.json()
    except httpx.HTTPStatusError as e:
      try:
        detail = e.response.json().get("detail", e.response.text)
      except ValueError:
        detail = e.response.text
      logger.error("navigate rejected: %s", detail)
      return {"error": f"HTTP {e.response.status_code}", "detail": detail}
    except Exception as e:  # pylint: disable=broad-except
      logger.error("navigate error: %s", e)
      return {"error": str(e)}

  async def drive(
      self,
      v_x: float,
      v_y: float,
      v_rot: float,
      duration: float,
  ) -> dict:
    """Drive with short body-frame velocity commands without GraphNav."""
    try:
      resp = await self._client.post(
          "/teleop/velocity",
          json={
              "v_x": v_x,
              "v_y": v_y,
              "v_rot": v_rot,
              "duration": duration,
              "take_lease": True,
              "power_on": True,
              "stand": True,
              "body_follow_arm": True,
          },
          timeout=max(float(duration), 0.1) + 15.0,
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("drive error: %s", e)
      return {"error": str(e)}

  async def stop(self) -> dict:
    """Cancel base, navigation, and arm actions immediately."""
    try:
      resp = await self._client.post(
          "/actions/stop",
          json={"take_lease": True, "freeze_arm": True},
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stop error: %s", e)
      return {"error": str(e)}

  async def look(
      self,
      direction: str,
      angle_rad: float = 0.15,
  ) -> dict:
    """Aim the gripper camera with arm rotation while keeping the base still."""
    normalized_direction = direction.strip().lower()
    if normalized_direction not in {"up", "down", "left", "right"}:
      raise ValueError("direction must be one of: up, down, left, right")

    angle = min(0.35, max(0.05, float(angle_rad)))
    rotation = {
        "up": {"dpitch": -angle},
        "down": {"dpitch": angle},
        "left": {"dyaw": angle},
        "right": {"dyaw": -angle},
    }[normalized_direction]
    try:
      resp = await self._client.post(
          "/arm/jog",
          json={
              **rotation,
              "seconds": 0.8,
              "take_lease": True,
              "timeout": 3.0,
          },
      )
      resp.raise_for_status()
      result = resp.json()
      result["direction"] = normalized_direction
      return result
    except Exception as e:  # pylint: disable=broad-except
      logger.error("look error: %s", e)
      return {"error": str(e)}

  async def stand(
      self,
      power_on: bool = True,
      take_lease: bool = True,
      timeout: float = 15,
  ) -> dict:
    """Command the robot to stand."""
    try:
      resp = await self._client.post(
          "/stand",
          json={
              "power_on": power_on,
              "take_lease": take_lease,
              "timeout": timeout,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stand error: %s", e)
      return {"error": str(e)}

  async def sit(
      self,
      take_lease: bool = True,
  ) -> dict:
    """Command the robot to sit."""
    try:
      resp = await self._client.post(
          "/sit",
          json={
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("sit error: %s", e)
      return {"error": str(e)}

  # ---- Arm ----

  async def deploy_arm(
      self,
      take_lease: bool = True,
  ) -> dict:
    """Deploy the robot arm."""
    try:
      resp = await self._client.post(
          "/arm/deploy",
          json={
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("deploy_arm error: %s", e)
      return {"error": str(e)}

  async def stow_arm(
      self,
      take_lease: bool = True,
  ) -> dict:
    """Stow the robot arm."""
    try:
      resp = await self._client.post(
          "/arm/stow",
          json={
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stow_arm error: %s", e)
      return {"error": str(e)}

  async def carry_arm(
      self,
      take_lease: bool = True,
  ) -> dict:
    """Move arm to carry pose."""
    try:
      resp = await self._client.post(
          "/arm/carry",
          json={
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("carry_arm error: %s", e)
      return {"error": str(e)}

  # ---- Gripper ----

  async def open_gripper(
      self,
      fraction: float = 1.0,
      take_lease: bool = True,
  ) -> dict:
    """Open the gripper."""
    try:
      resp = await self._client.post(
          "/gripper/open",
          json={
              "fraction": fraction,
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("open_gripper error: %s", e)
      return {"error": str(e)}

  async def close_gripper(
      self,
      take_lease: bool = True,
  ) -> dict:
    """Close the gripper."""
    try:
      resp = await self._client.post(
          "/gripper/close",
          json={
              "take_lease": take_lease,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("close_gripper error: %s", e)
      return {"error": str(e)}

  # ---- Manipulation API ----

  def clear_detected_target(self) -> None:
    """Invalidate a detection after any action that can move the camera."""
    self._detected_target = None

  async def detect(self, instruction: str) -> dict:
    """Detect one language-specified object and cache its grasp target."""
    self.clear_detected_target()
    try:
      resp = await self._client.post(
          "/detect/pick-target",
          json={
              "instruction": instruction,
          },
          timeout=_LONG_TIMEOUT,
      )
      resp.raise_for_status()
      detection = resp.json()
      target = detection.get("target")
      if not isinstance(target, dict):
        raise ValueError("Detection response did not include a target")
      self._detected_target = target
      return {
          "detected": True,
          "instruction": instruction,
          "label": detection.get("label", ""),
          "confidence": detection.get("confidence"),
          "target": target,
          "next_action": (
              "Call pick or place with no arguments before moving the robot"
              " or camera."
          ),
      }
    except Exception as e:  # pylint: disable=broad-except
      self.clear_detected_target()
      logger.error("detect error: %s", e)
      return {"error": str(e), "detected": False}

  async def pick(
      self,
      take_lease: bool = True,
      timeout: float = 120,
  ) -> dict:
    """Grasp the one-time target produced by the latest detect call."""
    target = self._detected_target
    self.clear_detected_target()
    if target is None:
      return {
          "error": "No detected target is available. Call detect first.",
          "executed": False,
      }
    try:
      resp = await self._client.post(
          "/manipulation/grasp-pixel",
          json={
              "x": target["normalized_x"],
              "y": target["normalized_y"],
              "take_lease": take_lease,
              "grip_max_torque_nm": 2.0,
          },
          timeout=_LONG_TIMEOUT,
      )
      resp.raise_for_status()
      result = resp.json()
      result["detected_target"] = target
      return result
    except Exception as e:  # pylint: disable=broad-except
      logger.error("pick error: %s", e)
      return {"error": str(e)}

  async def place(
      self,
      take_lease: bool = True,
      timeout: float = 120,
  ) -> dict:
    """Place an object at the one-time target produced by detect."""
    target = self._detected_target
    self.clear_detected_target()
    if target is None:
      return {
          "error": "No detected target is available. Call detect first.",
          "executed": False,
      }
    try:
      resp = await self._client.post(
          "/manipulation/place-pixel",
          json={
              "x": target["normalized_x"],
              "y": target["normalized_y"],
              "take_lease": take_lease,
          },
          timeout=_LONG_TIMEOUT,
      )
      resp.raise_for_status()
      result = resp.json()
      result["detected_target"] = target
      return result
    except Exception as e:  # pylint: disable=broad-except
      logger.error("place error: %s", e)
      return {"error": str(e)}

  async def wait_for_pick_up(
      self,
      monitor_sec: float = 30.0,
      upward_threshold_m: float = 0.02,
      sample_interval: float = 0.1,
      open_duration_sec: float = 3.0,
      take_lease: bool = True,
      gripper_timeout: float = 5.0,
      stow_timeout: float = 10.0,
  ) -> dict:
    """Wait for a recipient to lift the held item, then release and stow."""
    try:
      resp = await self._client.post(
          "/pickup/wait",
          json={
              "monitor_sec": monitor_sec,
              "upward_threshold_m": upward_threshold_m,
              "sample_interval": sample_interval,
              "open_duration_sec": open_duration_sec,
              "take_lease": take_lease,
              "gripper_timeout": gripper_timeout,
              "stow_timeout": stow_timeout,
          },
          timeout=_LONG_TIMEOUT,
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("wait_for_pick_up error: %s", e)
      return {"error": str(e)}

  # ---- Waypoints ----

  async def get_waypoints(self) -> dict:
    """Get registered waypoints and current GraphNav readiness."""
    try:
      waypoints_response = await self._client.get("/waypoints")
      waypoints_response.raise_for_status()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("get_waypoints error: %s", e)
      return {"error": str(e)}

    result = {
        "waypoints": waypoints_response.json(),
        "navigation_ready": None,
        "localized": None,
    }
    try:
      localization_response = await self._client.get("/localization")
      localization_response.raise_for_status()
      localization = localization_response.json()
      navigation_ready = bool(localization.get("localized"))
      result.update({
          "navigation_ready": navigation_ready,
          "localized": navigation_ready,
          "localization": localization.get("localization", {}),
      })
      if not navigation_ready:
        result["warning"] = (
            "Waypoint names may be stale registry entries. Do not call"
            " navigate until a GraphNav map is loaded and localization is"
            " established."
        )
      return result
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("get_waypoints localization unavailable: %s", e)
      result["localization_unavailable"] = str(e)
      result["warning"] = (
          "Waypoint names are available, but navigation readiness could not"
          " be determined. Do not call navigate until localization is"
          " verified."
      )
      return result

  # ---- Status ----

  async def get_battery(self) -> dict:
    """Get the current battery status."""
    try:
      resp = await self._client.get("/battery")
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("get_battery error: %s", e)
      return {"error": str(e)}

  async def stop_actions(
      self,
      take_lease: bool = True,
      freeze_arm: bool = True,
  ) -> dict:
    """Stop all current robot actions."""
    try:
      resp = await self._client.post(
          "/actions/stop",
          json={
              "take_lease": take_lease,
              "freeze_arm": freeze_arm,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stop_actions error: %s", e)
      return {"error": str(e)}

  async def stow(
      self,
      take_lease: bool = True,
      timeout: float = 20.0,
  ) -> dict:
    """Stow arm safely, choosing carry pose if holding an object."""
    try:
      resp = await self._client.post(
          "/arm/stow-smart",
          json={
              "take_lease": take_lease,
              "timeout": timeout,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stow error: %s", e)
      return {"error": str(e)}

  async def health_check(self) -> dict:
    """Check robot health status and battery."""
    try:
      health_resp = await self._client.get("/health")
      health_resp.raise_for_status()
      health_data = health_resp.json()
      
      battery_data = {}
      if health_data.get("connected"):
        battery_resp = await self._client.get("/battery")
        if battery_resp.status_code == 200:
          battery_data = battery_resp.json()
          
      return {
          "status": "SUCCESS",
          "health": health_data,
          "battery": battery_data,
      }
    except Exception as e:  # pylint: disable=broad-except
      logger.error("health_check error: %s", e)
      return {"error": str(e)}

  async def connect(
      self,
      hostname: str | None = None,
      username: str | None = None,
      password: str | None = None,
  ) -> dict:
    """Connect (or reconnect) to the Spot robot."""
    body: dict = {}
    if hostname is not None:
      body["hostname"] = hostname
    if username is not None:
      body["username"] = username
    if password is not None:
      body["password"] = password
    try:
      resp = await self._client.post("/connect", json=body)
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("connect error: %s", e)
      return {"error": str(e)}
