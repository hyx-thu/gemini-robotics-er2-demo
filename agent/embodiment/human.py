"""Human embodiment for local/browser mode (Lite version).

Implements the Embodiment interface for a human user interacting via
browser webcam and microphone.
"""

import asyncio
import logging
from typing import Any

from embodiment import base
from tool import tools as tools_lib
from prompt import si_builder

logger = logging.getLogger(__name__)


class HumanEmbodiment(base.Embodiment):
  """Embodiment for a human (local/browser mode)."""

  def __init__(self) -> None:
    self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    self.video_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    self.text_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    logger.info("HumanEmbodiment initialized")

  # ---- Queues ----

  def get_audio_queue(self) -> asyncio.Queue:
    return self.audio_queue

  def get_video_queue(self) -> asyncio.Queue:
    return self.video_queue

  def get_text_queue(self) -> asyncio.Queue:
    return self.text_queue

  # ---- Action execution ----

  async def execute_action(self, action_name: str, **kwargs: Any) -> str:
    if action_name == "ack":
      return "ok"
    if action_name == "send_message":
      target = kwargs.get("target", "unknown")
      message = kwargs.get("message", "")
      logger.info(
          "[HumanEmbodiment] send_message to %s: %s", target, message
      )
      return f"Message sent to {target}"

    logger.info(
        "[HumanEmbodiment] Executing action: %s with args: %s",
        action_name,
        kwargs,
    )
    return f"Human executed {action_name}"

  # ---- Tools & SI ----

  def get_tools(self) -> list[dict[str, Any]]:
    return tools_lib.human_tools()

  def get_system_instruction(self) -> str:
    return ""
