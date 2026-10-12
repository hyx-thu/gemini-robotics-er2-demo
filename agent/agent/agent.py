"""Agent configuration for Proactive Agent.

An Agent is a named composition of system instruction (SI), developer
instruction (DI), and tool declarations.  It is embodiment-agnostic —
the same Agent can be used with any embodiment.

Typical usage:

    agent = Agent.from_name("human")
    print(agent.system_instruction)
    print(agent.tools)
"""

import dataclasses
import os
from typing import Any

from prompt import si_builder
from tool import tools as tools_lib


def spot_simulated() -> bool:
  """True when Spot runs webcam-in-the-loop without a robot backend."""
  return os.getenv("SPOT_SIMULATED", "1").lower() in ("1", "true", "yes")


# System instruction shared by all physical-robot agents. Lives in the SI slot
# (not the DI) because general-purpose Live models otherwise keep the default
# "helpful AI assistant" identity and refuse to act ("I have no arm").
def embodiment_system_instruction() -> str:
  return _build_di("embodiment_si.md")


def native_audio_addendum() -> str:
  """DI addendum for models that produce speech natively (AUDIO modality)."""
  return _build_di("native_audio_addendum.md")


def _build_di(*files: str) -> str:
  """Concatenate instruction files from prompt/data in order."""
  parts = []
  for name in files:
    builder = si_builder.SIBuilder()
    builder.load_instruction_file(name)
    parts.append(builder.build().strip())
  return "\n\n".join(p for p in parts if p)


@dataclasses.dataclass
class Agent:
  """A named composition of prompt (system instruction) and tools."""

  name: str
  system_instruction: str
  developer_instruction: str
  tools: list[dict[str, Any]]

  # ---------------------------------------------------------------------------
  # Pre-built presets
  # ---------------------------------------------------------------------------

  @classmethod
  def human(cls) -> "Agent":
    """Local/browser mode with webcam."""
    return cls(
        name="human",
        system_instruction="",
        developer_instruction=_build_di("human_di.md", "plan_protocol.md"),
        tools=tools_lib.human_tools(),
    )

  @classmethod
  def spot(cls) -> "Agent":
    """Boston Dynamics Spot robot agent."""
    files = ["spot_di.md", "plan_protocol.md"]
    if spot_simulated():
      files.append("spot_sim_addendum.md")
    return cls(
        name="spot",
        system_instruction=embodiment_system_instruction(),
        developer_instruction=_build_di(*files),
        tools=tools_lib.spot_tools(),
    )

  @classmethod
  def tinybot(cls) -> "Agent":
    """Tinybot robot agent."""
    return cls(
        name="tinybot",
        system_instruction=embodiment_system_instruction(),
        developer_instruction=_build_di("tinybot_di.md", "plan_protocol.md"),
        tools=tools_lib.tinybot_tools(),
    )

  # ---------------------------------------------------------------------------
  # Name-based lookup
  # ---------------------------------------------------------------------------

  @classmethod
  def from_name(cls, name: str) -> "Agent":
    """Returns an Agent for the specified name string.

    Args:
      name: "human", "spot", or "tinybot" is supported for Lite.

    Returns:
      An Agent instance.

    Raises:
      ValueError: If the name is unknown.
    """
    presets = {
        "human": cls.human,
        "spot": cls.spot,
        "tinybot": cls.tinybot,
    }
    if name not in presets:
      raise ValueError(
          f"Unknown agent name: {name}. Available: {list(presets.keys())}"
      )
    return presets[name]()
