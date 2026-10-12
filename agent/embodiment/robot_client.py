"""HTTP-based robot client (Lite version).

Async client for communicating with a robot FastAPI backend over HTTP.
Uses httpx for async HTTP.
"""

import asyncio
import base64
import logging
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class RobotClient:
  """Lightweight async HTTP client for a FastAPI robot server.

  Subclass and set CAMERA_IDS / ENDPOINT_MAP for your embodiment.
  """

  # Override in subclasses.
  CAMERA_IDS: list[str] = []
  ENDPOINT_MAP: dict[str, str] = {}

  def __init__(
      self,
      base_url: str = "http://localhost:8888",
      timeout: float = 5.0,
  ) -> None:
    self._base_url = base_url
    self._client = httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
    )
    self._camera_timeout = httpx.Timeout(10.0)

  # ----- Robot control -----

  async def run_instruction(self, instruction: str) -> str:
    """Send a natural-language instruction to the robot (fire-and-forget)."""
    try:
      resp = await self._client.get(
          "/run_instruction/", params={"instruction": instruction}
      )
      resp.raise_for_status()
      msg = resp.json().get("message", "ok")
      logger.info("run_instruction(%s) -> %s", instruction, msg)
      return msg
    except Exception as e:  # pylint: disable=broad-except
      logger.error("run_instruction error: %s", e)
      return f"error: {e}"

  async def run_instruction_for_duration(
      self, instruction: str, duration_seconds: float = 30.0
  ) -> str:
    """Send a run instruction, sleep, then stop."""
    try:
      resp = await self._client.get(
          "/run_instruction/", params={"instruction": instruction}
      )
      resp.raise_for_status()
      msg = resp.json().get("message", "ok")
      logger.info(
          "run_instruction_for_duration(%s) started -> %s", instruction, msg
      )
      await asyncio.sleep(duration_seconds)
      stop_msg = await self.stop()
      logger.info(
          "run_instruction_for_duration(%s) stopped -> %s",
          instruction,
          stop_msg,
      )
      return f"Started: {msg}, Slept: {duration_seconds}s, Stopped: {stop_msg}"
    except Exception as e:  # pylint: disable=broad-except
      logger.error("run_instruction_for_duration error: %s", e)
      return f"error: {e}"

  async def stop(self) -> dict | str:
    """Stop the robot immediately."""
    try:
      resp = await self._client.get("/stop/")
      resp.raise_for_status()
      msg = resp.json().get("message", "ok")
      logger.info("stop() -> %s", msg)
      return msg
    except Exception as e:  # pylint: disable=broad-except
      logger.error("stop error: %s", e)
      return f"error: {e}"

  async def reset(self) -> str:
    """Reset robot to its default pose."""
    logger.info("reset() -> calling return to reset pose")
    return await self.run_instruction_for_duration(
        instruction="return to a reset pose until you can see the table",
        duration_seconds=15.0,
    )

  async def make_gesture(self, gesture: str) -> str:
    """Perform a gesture on the robot."""
    try:
      resp = await self._client.get(
          "/make_gesture/", params={"gesture": gesture}
      )
      resp.raise_for_status()
      msg = resp.json().get("message", "ok")
      logger.info("make_gesture(%s) -> %s", gesture, msg)
      return msg
    except Exception as e:  # pylint: disable=broad-except
      logger.error("make_gesture error: %s", e)
      return f"error: {e}"

  async def turn_head_to_uv(self, u: float, v: float) -> str:
    """Turn the robot head to look at a normalized pixel coordinate."""
    try:
      resp = await self._client.get(
          "/turn_head_to_uv/", params={"u": u, "v": v}
      )
      resp.raise_for_status()
      msg = resp.json().get("message", "ok")
      logger.info("turn_head_to_uv(u=%s, v=%s) -> %s", u, v, msg)
      return msg
    except Exception as e:  # pylint: disable=broad-except
      logger.error("turn_head_to_uv error: %s", e)
      return f"error: {e}"

  # ----- Camera snapshots -----

  async def get_camera_snapshot(self, camera_id: str) -> bytes | None:
    """Fetch a single JPEG via /camera_image/?camera_id=X."""
    try:
      resp = await self._client.get(
          "/camera_image/",
          params={"camera_id": camera_id},
          timeout=self._camera_timeout,
      )
      resp.raise_for_status()
      html = resp.text
      marker = "base64,"
      idx = html.find(marker)
      if idx < 0:
        return None
      start = idx + len(marker)
      end = html.find('"', start)
      if end < 0:
        end = len(html)
      return base64.b64decode(html[start:end])
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("camera snapshot %s error: %s", camera_id, e)
      return None

  async def stream_camera(self, camera_id: str) -> AsyncGenerator[bytes, None]:
    """Stream MJPEG frames from the robot's FastAPI backend."""
    if camera_id not in self.ENDPOINT_MAP:
      raise ValueError(f"Unknown camera ID for streaming: {camera_id}")

    endpoint = self.ENDPOINT_MAP[camera_id]
    backoff = 0.5
    max_backoff = 5.0

    while True:
      client = httpx.AsyncClient(
          base_url=self._base_url, timeout=httpx.Timeout(30.0)
      )
      try:
        async with client.stream("GET", endpoint) as response:
          response.raise_for_status()
          backoff = 0.5
          buffer = b""
          async for chunk in response.aiter_bytes():
            buffer += chunk
            while True:
              start_idx = buffer.find(b"\xff\xd8")
              if start_idx == -1:
                break
              end_idx = buffer.find(b"\xff\xd9", start_idx)
              if end_idx == -1:
                break
              jpeg_bytes = buffer[start_idx : end_idx + 2]
              buffer = buffer[end_idx + 2 :]
              yield jpeg_bytes
      except asyncio.CancelledError:
        await client.aclose()
        return
      except Exception:  # pylint: disable=broad-except
        pass
      finally:
        await client.aclose()

      await asyncio.sleep(backoff)
      backoff = min(backoff * 2, max_backoff)

  async def get_all_snapshots(
      self,
      camera_ids: list[str] | None = None,
  ) -> dict[str, bytes]:
    """Fetch snapshots from all cameras concurrently."""
    ids = camera_ids or self.CAMERA_IDS

    async def _fetch(cid: str):
      img = await self.get_camera_snapshot(cid)
      return (cid, img)

    results = await asyncio.gather(
        *[_fetch(cid) for cid in ids], return_exceptions=True
    )
    images: dict[str, bytes] = {}
    for item in results:
      if isinstance(item, tuple):
        name, data = item
        if isinstance(data, bytes):
          images[name] = data
    return images

  async def close(self) -> None:
    """Close the underlying HTTP client."""
    await self._client.aclose()
