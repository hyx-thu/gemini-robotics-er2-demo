"""Agent configuration for Proactive Agent.

An Agent is a named composition of system instruction (SI), developer
instruction (DI), and tool declarations.  It is embodiment-agnostic —
the same Agent can be used with any embodiment.

Typical usage:

    agent = Agent.from_name("robot")
    print(agent.system_instruction)
    print(agent.tools)
"""

import dataclasses
import os
from typing import Any

from prompt import si_builder
from tool import tools as tools_lib


def spot_simulated() -> bool:
  """True when the robot runs webcam-in-the-loop without a robot backend."""
  return os.getenv("SPOT_SIMULATED", "1").lower() in ("1", "true", "yes")


def normalize_agent_name(name: str | None) -> str:
  """Normalize agent name, mapping legacy 'spot' alias to 'robot'."""
  cleaned = (name or "").strip().lower()
  if cleaned in ("robot", "spot"):
    return "robot"
  return (name or "").strip()


# System instruction shared by all physical-robot agents. Lives in the SI slot
# (not the DI) because general-purpose Live models otherwise keep the default
# "helpful AI assistant" identity and refuse to act ("I have no arm").
def embodiment_system_instruction() -> str:
  return _build_di("embodiment_si.md")


def native_audio_addendum() -> str:
  """DI addendum for models that produce speech natively (AUDIO modality)."""
  return _build_di("native_audio_addendum.md")


def affective_audio_addendum() -> str:
  """Spoken emotional prosody guidance when affective voice expressiveness is enabled."""
  return (
      "AFFECTIVE & EMOTIONALLY EXPRESSIVE SPOKEN DELIVERY:\n"
      "- Listen attentively to the speaker's vocal tone, pace, and mood, and"
      " express rich, natural human emotion in your spoken voice rather than"
      " speaking in a flat or monotone cadence.\n"
      "- When welcoming visitors, answering everyday questions, or completing a"
      " robot task, speak with a warm audible smile, lively pitch variation,"
      " and friendly, upbeat energy.\n"
      "- When a visitor sounds anxious, hurried, tired, or unsure, soften your"
      " speaking pace and reply with calm, gentle, reassuring empathy.\n"
      "- When the moment calls for light humor or playful concierge charm, let"
      " natural warmth and expressive conversational inflection shine through"
      " while keeping answers crisp and concise."
  )


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
  def robot(cls) -> "Agent":
    """Physical / simulated Robot agent."""
    files = ["spot_di.md", "plan_protocol.md"]
    if spot_simulated():
      files.append("spot_sim_addendum.md")
    return cls(
        name="robot",
        system_instruction=embodiment_system_instruction(),
        developer_instruction=_build_di(*files),
        tools=(
            tools_lib.spot_sim_tools()
            if spot_simulated()
            else tools_lib.spot_tools()
        ),
    )

  # Backward-compatible alias for internal callers.
  spot = robot

  # ---------------------------------------------------------------------------
  # Name-based lookup
  # ---------------------------------------------------------------------------

  @classmethod
  def from_name(cls, name: str) -> "Agent":
    """Returns an Agent for the specified name string.

    Args:
      name: "robot" (or legacy alias "spot").

    Returns:
      An Agent instance.

    Raises:
      ValueError: If the name is unknown.
    """
    normalized = normalize_agent_name(name)
    if normalized != "robot":
      raise ValueError(
          f"Unknown agent name: {name}. Available: ['robot']"
      )
    return cls.robot()
