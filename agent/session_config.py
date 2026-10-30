"""Session configuration and model specification for Proactive Agent.

This module defines the per-session configuration and model abstraction layer.
ThinkingMode, ModelSpec, and KNOWN_MODELS live here because thinking mode is a
session-specific concern, not a server-level one.
"""

import dataclasses
import enum
import logging

logger = logging.getLogger(__name__)


class EndpointType(enum.Enum):
  GEMINI_LIVE_API = "gemini_live_api"


# Supported thinking levels for the Gemini API.
# "none" disables thinking (no thinkingConfig sent).
THINKING_LEVELS = ("none", "minimal", "low", "medium", "high")
DEFAULT_THINKING_LEVEL = "low"



@dataclasses.dataclass
class ModelSpec:
  """Specification for a model and its default attributes.

  Wraps the model name together with model-specific defaults (e.g. thinking
  budget). Use KNOWN_MODELS to register defaults for specific models. Users can
  still override individual attributes via CLI flags or ORCA eval query params.
  """

  name: str
  thinking_level: str = DEFAULT_THINKING_LEVEL
  # Whether the Live API accepts generationConfig.thinkingConfig for this
  # model. Sending it to a model that doesn't support it closes the socket
  # with 1007 "Thinking level is not supported for this model".
  supports_thinking: bool = True
  # Whether tool declarations may carry "behavior": "BLOCKING". Models
  # without support close the socket with 1007 "BLOCKING function calls are
  # not supported for this model"; for them the field is stripped from the
  # wire payload (the Live API default behaviour is blocking anyway, so the
  # agent-side scheduling semantics are unchanged).
  supports_blocking_tools: bool = True
  # Response modalities the model accepts. ER streaming models are text-only
  # (speech is produced by our TTS); requesting AUDIO from them is a 1007.
  response_modalities: tuple[str, ...] = ("AUDIO", "TEXT")
  # Whether the model was trained on the proactive "[HEARTBEAT] -> ack"
  # protocol (Robotics-ER family). General-purpose Live models are not: a
  # 2 s ack cadence quickly dominates their context and they start answering
  # the USER with ack too. For them the server slows the cadence, lengthens
  # the post-speech grace and uses a heartbeat text that spells out that
  # user requests take precedence over the timer.
  heartbeat_native: bool = True
  # Future attributes: temperature, max_output_tokens, etc.


# Registry of known models and their default attributes.
# Only public Gemini Live API models are included for Lite.
# thinking_level: "none" = disabled, "minimal"/"low"/"medium"/"high" = API thinking levels.
KNOWN_MODELS: dict[str, ModelSpec] = {
    "gemini-3.8-live": ModelSpec(
        name="gemini-3.8-live",
        thinking_level="none",
        supports_thinking=False,        # 1007: Thinking level is not supported
        supports_blocking_tools=False,  # 1007: BLOCKING function calls not supported
        heartbeat_native=False,
    ),
    "gemini-3.8-live-extended-thinking": ModelSpec(
        name="gemini-3.8-live-extended-thinking",
        thinking_level="low",
        supports_blocking_tools=False,  # 1007: BLOCKING function calls not supported
        heartbeat_native=False,
    ),
    "gemini-robotics-er-2-streaming-preview": ModelSpec(
        name="gemini-robotics-er-2-streaming-preview",
        thinking_level="low",
        response_modalities=("TEXT",),  # 1007 on AUDIO; speech comes from our TTS
    ),
    # Public Preview on Gemini Enterprise Agent Platform (Vertex AI global
    # generateContent); runs via CascadeClient (Vertex ASR -> ER2 REST -> TTS)
    # when selected directly, or via ErOrchestrator in gemini-3.8-live+er2.
    "gemini-robotics-er-2-preview": ModelSpec(
        name="gemini-robotics-er-2-preview",
        thinking_level="low",
        response_modalities=("TEXT",),
        heartbeat_native=True,
    ),
    # Cascade: Gemini ASR (gemini-3.5-transcribe-live) -> gemini-4-argon via
    # generateContent -> our TTS (model/cascade_client.py). Not a Live model.
    "gemini-4-argon": ModelSpec(
        name="gemini-4-argon",
        thinking_level="low",          # MINIMAL is rejected (400)
        response_modalities=("TEXT",),
        heartbeat_native=False,
    ),
    # Gemini 3.8 Live talks to the user; robot task orchestration is delegated
    # to gemini-robotics-er-2-preview (non-streaming) via the robot_task tool.
    "gemini-3.8-live+er2": ModelSpec(
        name="gemini-3.8-live+er2",
        thinking_level="none",
        supports_thinking=False,
        supports_blocking_tools=False,
        heartbeat_native=False,
    ),
    "robotics_er_live_text_only_2p0_no_safety_classifiers": ModelSpec(
        name="robotics_er_live_text_only_2p0_no_safety_classifiers",
        thinking_level="low",
        response_modalities=("TEXT",),
    ),
    "robotics_er_live_text_only_2p0_2_no_safety_classifiers": ModelSpec(
        name="robotics_er_live_text_only_2p0_2_no_safety_classifiers",
        thinking_level="low",
        response_modalities=("TEXT",),
    ),
    "robotics_er_live_text_only_optimized": ModelSpec(
        name="robotics_er_live_text_only_optimized",
        thinking_level="low",
        response_modalities=("TEXT",),
    ),
}

# Fallback spec for unknown models.
DEFAULT_MODEL_SPEC = ModelSpec(name="default", thinking_level=DEFAULT_THINKING_LEVEL)


@dataclasses.dataclass
class SessionConfig:
  """Configuration for a specific session/agent instance at runtime."""

  # Agent name (e.g., "human", "apollo", "custom")
  agent_name: str = "human"

  # Custom system instruction (used if agent_name is "custom")
  custom_si: str = ""

  # Custom developer instruction
  custom_di: str = ""

  # List of tool names to enable. If empty, all tools from ServerConfig are enabled.
  enabled_tools: list[str] = dataclasses.field(default_factory=list)

  # Response modality override (AUDIO or TEXT). If None, use ServerConfig default.
  response_modality: str | None = None

  # Model override. If None, use ServerConfig default.
  model: str | None = None

  # Enable TTS override. If None, use ServerConfig default.
  use_tts: bool | None = None

  # Thinking level override. None = use model default from KNOWN_MODELS.
  # "none" = disabled, "minimal"/"low"/"medium"/"high" = API thinking levels.
  thinking_level: str | None = None

  # Heartbeat text override. If None, use ServerConfig default or built-in default.
  heartbeat_text: str | None = None

  # Endpoint type (gemini_live_api)
  endpoint_type: str = "gemini_live_api"
