"""System Instruction (SI) builder for Proactive Agent.

Composes the system instruction from modular, reusable sections.
Loads instruction files from a local ``data/`` subdirectory.

Typical usage:

    builder = SIBuilder()
    builder.set_persona("You are a safety monitor.")
    builder.add_section("observation", "You watch the scene for hazards.")
    instruction = builder.build()

Pre-built presets:

    builder = SIBuilder.human()
    builder = SIBuilder.apollo()
"""

import dataclasses
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class InstructionSection:
  """A named section of the system instruction."""

  name: str
  content: str
  priority: int = 0  # Lower values appear first in the final instruction.


# Directory containing optional instruction data files (markdown, etc.).
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_data_file(filename: str) -> str:
  """Load a text file from the data/ subdirectory.

  Args:
    filename: Relative filename inside data/.

  Returns:
    File content as a string.

  Raises:
    FileNotFoundError: If the file does not exist.
  """
  filepath = _DATA_DIR / filename
  with open(filepath, "r") as f:
    return f.read()


class SIBuilder:
  """Composes a system instruction from modular sections.

  Sections are ordered by priority (lower first), then by insertion order.
  The final instruction is the concatenation of all sections separated by
  double newlines.
  """

  def __init__(self) -> None:
    self._persona: str | None = None
    self._sections: list[InstructionSection] = []
    self._raw_instruction: str | None = None

  # ---- Persona (always first) ----

  def set_persona(self, persona: str) -> "SIBuilder":
    """Sets the agent persona.  Always appears at the top."""
    self._persona = persona
    return self

  # ---- Sections ----

  def add_section(
      self, name: str, content: str, priority: int = 0
  ) -> "SIBuilder":
    """Adds a named section.  Replaces any existing section with same name."""
    self._sections = [s for s in self._sections if s.name != name]
    self._sections.append(
        InstructionSection(name=name, content=content, priority=priority)
    )
    return self

  def remove_section(self, name: str) -> "SIBuilder":
    """Removes a section by name.  No-op if not found."""
    self._sections = [s for s in self._sections if s.name != name]
    return self

  def has_section(self, name: str) -> bool:
    return any(s.name == name for s in self._sections)

  # ---- Raw override ----

  def set_raw_instruction(self, content: str) -> "SIBuilder":
    """Sets a raw instruction that bypasses section-based building."""
    self._raw_instruction = content
    return self

  # ---- Build ----

  def build(self) -> str:
    """Builds the final system instruction string."""
    if self._raw_instruction is not None:
      return self._raw_instruction

    parts: list[str] = []
    if self._persona:
      parts.append("### PERSONA ###\n" + self._persona)

    sorted_sections = sorted(self._sections, key=lambda s: s.priority)
    for section in sorted_sections:
      parts.append(f"### {section.name.upper()} ###\n" + section.content)

    return "\n\n".join(parts)

  # ---- Load from data file ----

  def load_instruction_file(self, filename: str) -> "SIBuilder":
    """Loads a raw instruction from a data/ file, bypassing sections."""
    content = _load_data_file(filename)
    return self.set_raw_instruction(content)
