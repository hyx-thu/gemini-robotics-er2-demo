"""Spot embodiment for Boston Dynamics Spot robot control (Lite version)."""

import asyncio
import logging

import camera_poller
from embodiment import base
from embodiment.spot import robot_client
from tool import tools as tools_lib

logger = logging.getLogger(__name__)


class SpotEmbodiment(base.Embodiment):
  """Embodiment for a physical Boston Dynamics Spot robot."""

  def __init__(
      self,
      robot_url: str,
      poll_hz: float = 2.0,
      push_hz: float = 1.0,
  ):
    self.audio_queue = asyncio.Queue()
    self.video_queue = asyncio.Queue()
    self.text_queue = asyncio.Queue()
    self.robot = robot_client.SpotRobotClient(base_url=robot_url)
    self.poller = camera_poller.CameraPoller(
        robot_client=self.robot,
        video_input_queue=self.video_queue,
        camera_ids=["hand_color_image"],
        poll_hz=poll_hz,
        push_hz=push_hz,
    )
    self.poller_task = asyncio.create_task(self.poller.run())
    logger.info("SpotEmbodiment initialized with URL: %s", robot_url)

  def get_audio_queue(self) -> asyncio.Queue:
    return self.audio_queue

  def get_video_queue(self) -> asyncio.Queue:
    return self.video_queue

  def get_text_queue(self) -> asyncio.Queue:
    return self.text_queue

  def get_tools(self) -> list[dict]:
    return tools_lib.spot_tools()

  async def initialize(self) -> None:
    """Initialize SpotEmbodiment."""
    pass

  def get_system_instruction(self) -> str:
    # Instructions are loaded via Agent DI in Lite version.
    return ""

  async def execute_action(self, action_name: str, **kwargs):
    logger.info(
        "SpotEmbodiment executing action: %s with args: %s",
        action_name,
        kwargs,
    )
    if action_name == "ack":
      return "No action needed."

    if action_name not in {
        "detect",
        "pick",
        "place",
        "health_check",
        "get_waypoints",
    }:
      self.robot.clear_detected_target()

    # Legacy fallback when the backend contract is unavailable.
    fallback_actions = {
        "health_check": self.robot.health_check,
        "get_waypoints": self.robot.get_waypoints,
        "navigate": self.robot.navigate,
        "drive": self.robot.drive,
        "stop": self.robot.stop,
        "look": self.robot.look,
        "detect": self.robot.detect,
        "pick": self.robot.pick,
        "place": self.robot.place,
        "wait_for_pick_up": self.robot.wait_for_pick_up,
        "stand": self.robot.stand,
        "sit": self.robot.sit,
        "stow": self.robot.stow,
    }
    action = fallback_actions.get(action_name)
    if action is None:
      raise ValueError(f"Unknown action: {action_name}")
    return await action(**kwargs)

  async def close(self):
    self.poller.stop()
    await self.poller_task
    await self.robot.close()
