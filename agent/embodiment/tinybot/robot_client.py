"""Tinybot robot client (Lite version)."""

import logging
from typing import Any

from embodiment import robot_client

logger = logging.getLogger(__name__)


class TinybotRobotClient(robot_client.RobotClient):
  """RobotClient configured for the Tinybot platform."""

  CAMERA_IDS = ["camera"]

  # Map camera ID to the video-stream endpoint
  ENDPOINT_MAP = {
      "camera": "/video-stream",
  }

  def __init__(
      self,
      base_url: str = "http://localhost:8000",
      timeout: float = 30.0,
  ):
    super().__init__(base_url=base_url, timeout=timeout)

  async def make_gesture(self, gesture: str, speed: int = 150) -> str:
    """Execute a gesture on the tinybot."""
    try:
      resp = await self._client.post(
          "/make-gesture",
          params={"gesture": gesture, "speed": speed},
      )
      resp.raise_for_status()
      return resp.json().get("status", "ok")
    except Exception as e:  # pylint: disable=broad-except
      logger.error("make_gesture error: %s", e)
      return f"error: {e}"

  async def move_absolute(
      self, angles: list[float], speeds: list[int], blocking: bool = True
  ) -> dict[str, Any]:
    """Move tinybot joints to absolute angles."""
    try:
      resp = await self._client.post(
          "/move-absolute",
          json={
              "angles": angles,
              "speeds": speeds,
              "blocking": blocking,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("move_absolute error: %s", e)
      return {"error": str(e)}

  async def move_relative(
      self,
      relative_angles: list[float],
      speeds: list[int],
      blocking: bool = True,
  ) -> dict[str, Any]:
    """Move tinybot joints by relative angles."""
    try:
      resp = await self._client.post(
          "/move-relative",
          json={
              "relative_angles": relative_angles,
              "speeds": speeds,
              "blocking": blocking,
          },
      )
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("move_relative error: %s", e)
      return {"error": str(e)}

  async def turn_head_to_uv(self, u: float, v: float) -> str:
    """Turn tinybot head to look at normalized pixel coordinates (0.0-1.0)."""
    # Scale from 0.0-1.0 to 0-1000
    u_int = int(u * 1000)
    v_int = int(v * 1000)
    try:
      resp = await self._client.post(
          "/look-to-uv",
          json={
              "u": u_int,
              "v": v_int,
              "speed": 150,
              "blocking": True,
          },
      )
      resp.raise_for_status()
      return resp.json().get("status", "ok")
    except Exception as e:  # pylint: disable=broad-except
      logger.error("turn_head_to_uv error: %s", e)
      return f"error: {e}"

  async def get_health(self) -> dict[str, Any]:
    """Check tinybot health status."""
    try:
      resp = await self._client.get("/health")
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("health error: %s", e)
      return {"error": str(e)}

  async def point(self, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Draw points on the tinybot camera stream overlay."""
    try:
      resp = await self._client.post("/point", json=points)
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("point error: %s", e)
      return {"error": str(e)}

  async def clear_overlay(self) -> dict[str, Any]:
    """Clear point overlays from tinybot camera stream."""
    try:
      resp = await self._client.post("/clear-overlay")
      resp.raise_for_status()
      return resp.json()
    except Exception as e:  # pylint: disable=broad-except
      logger.error("clear_overlay error: %s", e)
      return {"error": str(e)}
