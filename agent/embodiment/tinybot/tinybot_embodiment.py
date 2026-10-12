"""Tinybot embodiment for Eden Lite agent."""

import asyncio
import logging

import camera_poller
from embodiment import base
from embodiment.tinybot import robot_client
from tool import tools as tools_lib

logger = logging.getLogger(__name__)


class TinybotEmbodiment(base.Embodiment):
  """Embodiment for a physical Tinybot robot."""

  def __init__(
      self,
      robot_url: str,
      poll_hz: float = 2.0,
      push_hz: float = 1.0,
  ):
    self.audio_queue = asyncio.Queue()
    self.video_queue = asyncio.Queue()
    self.text_queue = asyncio.Queue()
    self.robot = robot_client.TinybotRobotClient(base_url=robot_url)
    self.poller = camera_poller.CameraPoller(
        robot_client=self.robot,
        video_input_queue=self.video_queue,
        camera_ids=["camera"],
        poll_hz=poll_hz,
        push_hz=push_hz,
    )
    self.poller_task = asyncio.create_task(self.poller.run())
    logger.info("TinybotEmbodiment initialized with URL: %s", robot_url)

  def get_audio_queue(self) -> asyncio.Queue:
    return self.audio_queue

  def get_video_queue(self) -> asyncio.Queue:
    return self.video_queue

  def get_text_queue(self) -> asyncio.Queue:
    return self.text_queue

  def get_tools(self) -> list[dict]:
    return tools_lib.tinybot_tools()

  def get_system_instruction(self) -> str:
    return ""

  async def execute_action(self, action_name: str, **kwargs):
    logger.info(
        "TinybotEmbodiment executing action: %s with args: %s",
        action_name,
        kwargs,
    )
    if action_name == "make_gesture":
      return await self.robot.make_gesture(**kwargs)
    elif action_name == "move_absolute":
      return await self.robot.move_absolute(**kwargs)
    elif action_name == "move_relative":
      return await self.robot.move_relative(**kwargs)
    elif action_name == "turn_head_to_uv":
      return await self.robot.turn_head_to_uv(**kwargs)

    elif action_name == "draw_points":
      # Map 'x', 'y' to 'u', 'v' for Tinybot API
      raw_points = kwargs.get("points", [])
      mapped_points = []
      for p in raw_points:
        mapped_points.append({
            "u": int(p.get("x", 0)),
            "v": int(p.get("y", 0)),
            "label": p.get("label", ""),
        })
      return await self.robot.point(points=mapped_points)
    elif action_name == "clear_overlay":
      return await self.robot.clear_overlay()
    elif action_name == "stop":
      return await self.robot.stop()
    elif action_name == "ack":
      return {"status": "ok"}
    elif action_name == "send_message":
      target = kwargs.get("target", "unknown")
      message = kwargs.get("message", "")
      logger.info(
          "[TinybotEmbodiment] send_message to %s: %s", target, message
      )
      return f"Message sent to {target}"
    else:
      raise ValueError(f"Unknown action: {action_name}")

  async def close(self):
    self.poller.stop()
    await self.poller_task
    await self.robot.close()
