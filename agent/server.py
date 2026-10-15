# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "websockets",
#   "websocket-client",
#   "httpx",
#   "numpy",
#   "pillow",
#   "protobuf",
#   "grpcio",
#   "google-ai-generativelanguage",
# ]
# ///

r"""Proactive Agent — FastAPI server.

This is the main entry point for the agent. It provides:
  - WebSocket /ws endpoint for bidirectional Gemini Live API sessions
  - Static UI serving
  - Minimal REST API for camera proxy and inter-agent messaging

No external frameworks, SSOT, or internal RPC infrastructure.
"""

import argparse
import asyncio
import base64
import dataclasses
import json
import logging
import os
import re
import signal
import tempfile
import time
import zipfile

import fastapi
from fastapi import responses as fastapi_responses
from fastapi import staticfiles as fastapi_staticfiles
from fastapi.middleware import cors as cors_middleware
import uvicorn

import session_config
import session_manager
from agent import agent as agent_lib
from embodiment import human as human_embodiment_lib
from embodiment.spot import robot_client as spot_robot_client_lib
from embodiment.spot import spot_embodiment as spot_embodiment_lib
from embodiment.spot import spot_sim_embodiment as spot_sim_embodiment_lib
from embodiment.tinybot import tinybot_embodiment as tinybot_embodiment_lib
from model import live_api_client
from model import tts_client as tts_client_lib
from tool import tools as tools_lib


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ServerConfig:
  """Server-level configuration from CLI flags."""

  model: str = "models/gemini-robotics-er-2-streaming-preview"
  robot_url: str = "http://localhost:8888"
  api_key: str | None = None
  tts_api_key: str | None = None
  use_tts: bool = True
  tts_voice: str = "en-US-Chirp3-HD-Puck"
  tts_language_code: str = "en-US"
  tts_audio_gain: float = 1.0
  response_modality: str = "AUDIO"
  dump_video_dir: str = ""
  heartbeat_interval_seconds: float = 3.0
  heartbeat_min_delay_seconds: float = 2.0
  heartbeat_enabled: bool = True
  use_event_driven_heartbeat: bool = True
  media_resolution: str = "low"
  enable_send_message_to_user: bool = True
  mock_robot: bool = False
  agent_peers: dict[str, str] = dataclasses.field(default_factory=dict)
  custom_si: dict[str, str] = dataclasses.field(default_factory=dict)
  custom_di: dict[str, str] = dataclasses.field(default_factory=dict)
  custom_heartbeat_text: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  port: int = 8000


# ---------------------------------------------------------------------------
# UI directory resolution (handles .par packaging)
# ---------------------------------------------------------------------------
def _resolve_ui_dir():
  """Find the ui/ directory."""
  this_dir = os.path.dirname(os.path.abspath(__file__))
  candidate = os.path.join(this_dir, "ui")
  if os.path.isdir(candidate):
    logger.info("Found UI directory: %s", candidate)
    return candidate
  return None

# ---------------------------------------------------------------------------
# Per-model prompt / heartbeat adaptation
# ---------------------------------------------------------------------------
# Cadence used for models that were not trained on the ack heartbeat protocol.
_GENERIC_HEARTBEAT_INTERVAL_S = 6.0

# Composite "Live + ER2" mode: extra DI for the Live front-end.
_LIVE_FRONTEND_ADDENDUM = """### ROBOT TASKS ARE DELEGATED (LIVE + ER2 MODE) ###
- You are the robot's voice and conversation partner. You do NOT drive the arm
  yourself: for any physical manipulation request (move / fetch / pick up / put /
  sort / stack / hand over objects, multi-step robot tasks) call
  `robot_task(instruction=...)` once, passing the user's request faithfully in
  their own words. Gemini Robotics-ER 2 then plans and executes it with the
  camera and the arm.
- After calling it, tell the user in one short sentence that the robot is on it.
  Do not invent steps, coordinates or results.
- Messages that start with `[ROBOT]` are progress / outcome notes from the robot
  planner, not the user. Relay them to the user in ONE short sentence in the
  user's language, then wait. Never call robot_task because of a [ROBOT] note.
- If the user changes or cancels the task, call `robot_task` with the new request
  (it replaces the running one) or `stop`.
- Keep using your other tools (look, navigate, stand, sit, gestures, search,
  maps, location) directly as usual."""
_GENERIC_POST_SPEECH_GRACE_S = 2.5

# DI lines written for the text-only ER + TTS path that contradict native
# speech output. Removed (not just overridden) for AUDIO-modality models.
_TEXT_ONLY_COMMS_PATTERNS = (
    r"^.*You MUST use this tool to speak to the user.*$",
    r"^.*You MUST NOT reply directly with text or audio.*$",
    r"^.*then call `send_message\(target='user'.*$",
    r"^.*You (?:only|must only) talk (?:to the user )?via the `send_message` tool.*$",
    r"^.*`send_message\(target=\"user\".*$",
    r"^.*in the same turn as `send_message`.*$",
)

_TEXT_ONLY_INLINE_REPLACEMENTS = (
    (
        "In the same turn, call `send_message(target='user', message=...)` and briefly tell the user",
        "After calling `update_plan` and your first action tool (e.g. `detect`), briefly tell the user by voice",
    ),
    (
        "call `send_message(target='user', message=...)` and briefly tell the user",
        "briefly tell the user by voice",
    ),
    (
        "Always write the `send_message` text in the SAME language",
        "Always speak and reply in the SAME language",
    ),
    (
        "The TTS voice is multilingual",
        "Your voice is natively multilingual",
    ),
    ("via `send_message`", "by voice"),
    ("with `send_message`", "by voice"),
)


def _strip_text_only_comms_rules(di: str) -> str:
  out = di
  for pat in _TEXT_ONLY_COMMS_PATTERNS:
    out = re.sub(pat, "", out, flags=re.MULTILINE)
  for old, new in _TEXT_ONLY_INLINE_REPLACEMENTS:
    out = out.replace(old, new)
  return re.sub(r"\n{3,}", "\n\n", out)


# ---------------------------------------------------------------------------
# Thinking level resolution
# ---------------------------------------------------------------------------
def model_spec_for(model_name: str) -> session_config.ModelSpec:
  """Look up the ModelSpec for a (possibly "models/"-prefixed) model name."""
  normalized = model_name.removeprefix("models/")
  return session_config.KNOWN_MODELS.get(
      normalized, session_config.DEFAULT_MODEL_SPEC
  )


def resolve_response_modality(model_name: str, requested: str) -> str:
  """Clamp the requested response modality to what the model supports."""
  spec = model_spec_for(model_name)
  requested = (requested or "AUDIO").upper()
  if requested in spec.response_modalities:
    return requested
  fallback = spec.response_modalities[0]
  logger.warning(
      "Model %s does not support response modality %s; using %s",
      model_name, requested, fallback,
  )
  return fallback


def resolve_thinking_level(
    model_name: str,
    session_override: str | None = None,
) -> str:
  """Resolve thinking level: session override > model default > global default.

  Models that do not support thinkingConfig always resolve to "none",
  regardless of any override, so we never send a config the Live API
  rejects (close code 1007).
  """
  spec = model_spec_for(model_name)
  if not spec.supports_thinking:
    if session_override not in (None, "none"):
      logger.warning(
          "Model %s does not support thinking; ignoring requested level %r",
          model_name, session_override,
      )
    return "none"
  if session_override is not None:
    return session_override
  return spec.thinking_level


# ---------------------------------------------------------------------------
# Type conversion utility for query parameter parsing
# ---------------------------------------------------------------------------
def _convert_type(value: str, type_hint) -> object:
  """Convert a string value to the expected type hint."""
  if type_hint is bool or type_hint == "bool":
    return value.lower() in ("true", "1", "yes")
  if type_hint is int or type_hint == "int":
    return int(value)
  if type_hint is float or type_hint == "float":
    return float(value)
  return value


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
router = fastapi.APIRouter()


@router.get("/")
async def root(request: fastapi.Request):
  ui_dir = request.app.state.ui_dir
  if not ui_dir:
    return fastapi_responses.PlainTextResponse("UI not found")
  # Never let the shell page go stale: it carries the ?v= cache-busters for
  # the JS/CSS, so a cached index.html would keep loading old assets.
  return fastapi_responses.FileResponse(
      os.path.join(ui_dir, "index.html"),
      headers={"Cache-Control": "no-cache, must-revalidate"},
  )


@router.get("/api/camera")
async def api_camera(request: fastapi.Request):
  """Return the latest camera frame as MJPEG stream (robot camera proxy)."""
  poller = request.app.state.active_poller_ref
  if not poller:
    return fastapi.Response(status_code=204)

  async def _mjpeg_gen():
    async for frame in poller.get_stream():
      yield (
          b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
      )

  return fastapi.responses.StreamingResponse(
      _mjpeg_gen(),
      media_type="multipart/x-mixed-replace; boundary=frame",
      headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
  )


@router.post("/api/send")
async def api_send(request: fastapi.Request):
  """Inject text into the active session (inter-agent messaging)."""
  body = await request.json()
  text = body.get("text", "")
  active_session = request.app.state.active_session
  if not active_session or not text:
    return fastapi.responses.JSONResponse(
        status_code=400, content={"error": "No active session or empty text"}
    )
  await active_session.get_text_queue().put(text)
  return {"status": "ok"}


@router.get("/api/server_defaults")
async def api_server_defaults(request: fastapi.Request):
  """Return available models and server defaults for the UI."""
  config = request.app.state.config
  return {
      "models": list(session_config.KNOWN_MODELS.keys()),
      # Per-model thinking capability so the UI can sync its dropdown.
      "thinking": {
          name: {
              "default": spec.thinking_level,
              "supported": spec.supports_thinking,
          }
          for name, spec in session_config.KNOWN_MODELS.items()
      },
      "capabilities": {
          name: {
              "blocking_tools": spec.supports_blocking_tools,
              "response_modalities": list(spec.response_modalities),
          }
          for name, spec in session_config.KNOWN_MODELS.items()
      },
      "default_model": config.model,
      # Browser key restricted to the Maps Embed API + our domains (it is
      # visible in iframe URLs by design). Empty -> UI shows link-only cards.
      "maps_embed_key": os.environ.get("MAPS_EMBED_KEY", ""),
      "default_modality": config.response_modality,
      "use_tts": config.use_tts,
      "spot_simulated": agent_lib.spot_simulated(),
      # Agent pre-selected in the UI on page load (env DEFAULT_AGENT).
      "default_agent": os.environ.get("DEFAULT_AGENT", "spot").strip() or "spot",
      # Which video source the UI should use per agent.
      "video_sources": {
          "human": "computer",
          "spot": "computer" if agent_lib.spot_simulated() else "spot",
          "tinybot": "tinybot",
      },
  }


def _validate_agent_name(name: object) -> str:
  if not isinstance(name, str) or not name.strip():
    raise fastapi.HTTPException(status_code=400, detail="agent_name is required")
  name = name.strip()
  try:
    agent_lib.Agent.from_name(name)
  except ValueError as exc:
    raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
  return name


def _set_optional_override(
    overrides: dict[str, str],
    agent_name: str,
    value: object,
) -> None:
  if value is None or value == "":
    overrides.pop(agent_name, None)
    return
  if not isinstance(value, str):
    raise fastapi.HTTPException(status_code=400, detail="Instruction values must be strings")
  overrides[agent_name] = value


@router.post("/api/config/instructions")
async def update_instructions(request: fastapi.Request):
  """Update runtime instructions used when the next agent session starts."""
  body = await request.json()
  agent_name = _validate_agent_name(body.get("agent_name"))
  config = request.app.state.config

  _set_optional_override(
      config.custom_si, agent_name, body.get("system_instruction")
  )
  _set_optional_override(
      config.custom_di, agent_name, body.get("developer_instruction")
  )
  _set_optional_override(
      config.custom_heartbeat_text, agent_name, body.get("heartbeat_text")
  )

  for field in ("heartbeat_interval_seconds", "heartbeat_min_delay_seconds"):
    value = body.get(field)
    if value is None:
      continue
    try:
      value = float(value)
    except (TypeError, ValueError) as exc:
      raise fastapi.HTTPException(
          status_code=400, detail=f"{field} must be a number"
      ) from exc
    if value <= 0:
      raise fastapi.HTTPException(
          status_code=400, detail=f"{field} must be greater than zero"
      )
    setattr(config, field, value)

  if "use_event_driven_heartbeat" in body:
    value = body["use_event_driven_heartbeat"]
    if not isinstance(value, bool):
      raise fastapi.HTTPException(
          status_code=400,
          detail="use_event_driven_heartbeat must be a boolean",
      )
    config.use_event_driven_heartbeat = value

  return {
      "success": True,
      "agent_name": agent_name,
      "requires_reconnect": request.app.state.active_session is not None,
  }


@router.delete("/api/config/instructions")
async def reset_instructions(request: fastapi.Request, agent_name: str):
  """Reset one agent's overrides and heartbeat settings to startup values."""
  agent_name = _validate_agent_name(agent_name)
  config = request.app.state.config
  config.custom_si.pop(agent_name, None)
  config.custom_di.pop(agent_name, None)
  config.custom_heartbeat_text.pop(agent_name, None)

  defaults = request.app.state.instruction_defaults
  config.heartbeat_interval_seconds = defaults["heartbeat_interval_seconds"]
  config.heartbeat_min_delay_seconds = defaults["heartbeat_min_delay_seconds"]
  config.use_event_driven_heartbeat = defaults["use_event_driven_heartbeat"]
  return {
      "success": True,
      "agent_name": agent_name,
      "requires_reconnect": request.app.state.active_session is not None,
  }


@router.get("/api/agent_config/{name}")
async def api_agent_config(
    request: fastapi.Request,
    name: str,
    endpoint_type: str = "gemini_live_api",
    model: str | None = None,
):
  """Return the resolved agent configuration."""
  config = request.app.state.config
  try:
    agent = agent_lib.Agent.from_name(name)
    agent_si = config.custom_si.get(name, "")
    agent_di = config.custom_di.get(name, "")
    agent_hb = (
        config.custom_heartbeat_text.get(name, "")
        or session_manager.get_default_heartbeat_text()
    )
    if agent_si:
      agent.system_instruction = agent_si
    if agent_di:
      agent.developer_instruction = agent_di
  except ValueError:
    raise fastapi.HTTPException(
        status_code=400, detail=f"Unknown agent name: {name}"
    )

  resolved_tools = agent.tools


  decls = []
  for t in resolved_tools:
    if "functionDeclarations" in t:
      decls.extend(t["functionDeclarations"])
    else:
      decls.append(t)

  # Which backend serves the selected model (Vertex AI via OAuth, or the
  # Gemini Developer API via API key) — shown in Settings → Agent Config.
  if live_api_client.uses_vertex(config.model):
    service_address = (
        f"{os.environ.get('VERTEX_LOCATION', 'us-central1')}-aiplatform.googleapis.com"
        " (Vertex AI · OAuth)"
    )
  else:
    service_address = "generativelanguage.googleapis.com (Developer API · key)"

  return {
      "name": agent.name,
      "system_instruction": agent.system_instruction,
      "developer_instruction": agent.developer_instruction,
      "is_modified": bool(
          agent_si or agent_di or config.custom_heartbeat_text.get(name, "")
      ),
      "tools": decls,
      "model": config.model,
      "response_modality": config.response_modality,
      "use_tts": config.use_tts,
      "tts_voice": config.tts_voice,
      "tts_language_code": config.tts_language_code,
      "endpoint_type": endpoint_type,
      "service_address": service_address,
      "heartbeat_text": agent_hb,
      "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
      "heartbeat_min_delay_seconds": config.heartbeat_min_delay_seconds,
      "use_event_driven_heartbeat": config.use_event_driven_heartbeat,
      "thinking_level": resolve_thinking_level(model or config.model),
  }


# ---------------------------------------------------------------------------
# WebSocket endpoint — the heart of the system
# ---------------------------------------------------------------------------
@router.websocket("/ws")
async def websocket_endpoint(
    websocket: fastapi.WebSocket,
    agent_name: str = "human",
    custom_si: str = "",
    enabled_tools: str = "",
    response_modality: str | None = None,
    model: str | None = None,
    use_tts: bool | None = None,
    thinking_level: str | None = None,
    endpoint_type: str = "gemini_live_api",
):
  """WebSocket: Proactive Agent session with Gemini Live API."""
  app = websocket.app
  config = app.state.config

  await websocket.accept()
  logger.info("WebSocket connected")

  async def ui_callback(event):
    await websocket.send_json(event)

  async with app.state.session_lock:
    if app.state.active_session is not None:
      await websocket.close(
          code=1008, reason="Only one active session allowed"
      )
      return

    if agent_name == "spot" and agent_lib.spot_simulated():
      # Webcam-in-the-loop: browser camera/mic, simulated actuation.
      current_embodiment = spot_sim_embodiment_lib.SimulatedSpotEmbodiment()
      app.state.active_poller_ref = None
    elif agent_name == "spot":
      current_embodiment = spot_embodiment_lib.SpotEmbodiment(
          robot_url=config.robot_url
      )
      await current_embodiment.initialize()
      app.state.active_poller_ref = current_embodiment.poller
    elif agent_name == "tinybot":
      current_embodiment = tinybot_embodiment_lib.TinybotEmbodiment(
          robot_url=config.robot_url
      )
      app.state.active_poller_ref = current_embodiment.poller
    else:
      # Default to human (local webcam/mic) embodiment
      current_embodiment = human_embodiment_lib.HumanEmbodiment()

    # Build SessionConfig from query params
    session_cfg = session_config.SessionConfig(
        agent_name=agent_name,
        custom_si=custom_si,
        enabled_tools=enabled_tools.split(",") if enabled_tools else [],
        response_modality=response_modality,
        model=model or config.model,
        use_tts=use_tts,
        thinking_level=thinking_level,
        endpoint_type=endpoint_type,
    )
    logger.info("[SERVER] Resolved session config: %s", session_cfg)

    # Resolve agent
    custom_si_override = session_cfg.custom_si or config.custom_si.get(
        session_cfg.agent_name, ""
    )
    custom_di_override = session_cfg.custom_di or config.custom_di.get(
        session_cfg.agent_name, ""
    )
    heartbeat_text_override = (
        session_cfg.heartbeat_text
        or config.custom_heartbeat_text.get(session_cfg.agent_name, "")
    )
    resolved_model = session_cfg.model or config.model
    assert resolved_model is not None

    resolved_thinking_level = resolve_thinking_level(
        resolved_model, session_cfg.thinking_level
    )
    agent = agent_lib.Agent.from_name(
        session_cfg.agent_name,
    )
    if custom_si_override:
      agent.system_instruction = custom_si_override
    if custom_di_override:
      agent.developer_instruction = custom_di_override

    # Native-speech models (3.8-live family) talk directly; the DI's
    # "always use send_message / never reply with audio" rules were written
    # for text-only ER + our TTS, so override just the comms channel rules.
    modality = resolve_response_modality(
        resolved_model, session_cfg.response_modality or config.response_modality
    )
    # The text-only DI (send_message + TTS) is what the non-streaming ER
    # orchestrator needs in the composite mode; keep it before the AUDIO edit.
    er_developer_instruction = agent.developer_instruction
    orchestrator_model = live_api_client.orchestrator_model(resolved_model)
    if modality == "AUDIO":
      agent.developer_instruction = _strip_text_only_comms_rules(
          agent.developer_instruction
      ).rstrip() + "\n\n" + agent_lib.native_audio_addendum()

    # Heartbeat policy: models not trained on the ack protocol get a slower
    # cadence, a longer post-speech grace and an explicit heartbeat text.
    spec = model_spec_for(resolved_model)
    hb_interval = config.heartbeat_interval_seconds
    hb_min_delay = config.heartbeat_min_delay_seconds
    emb_hb_text = getattr(current_embodiment, "heartbeat_text", None)
    hb_text_default = emb_hb_text or session_manager.get_default_heartbeat_text()
    if not spec.heartbeat_native:
      hb_interval = max(hb_interval, _GENERIC_HEARTBEAT_INTERVAL_S)
      hb_min_delay = max(hb_min_delay, _GENERIC_POST_SPEECH_GRACE_S)
      hb_text_default = emb_hb_text or session_manager.get_generic_heartbeat_text()
      logger.info(
          "Model %s is not heartbeat-native: interval=%.1fs grace=%.1fs",
          resolved_model, hb_interval, hb_min_delay,
      )

    # Determine tools (agent tools, optionally filtered)
    agent_tools = (
        current_embodiment.get_tools()
        if agent_name in ("spot", "tinybot")
        else agent.tools
    )
    if session_cfg.enabled_tools:
      enabled = set(session_cfg.enabled_tools)
      agent_tools = [
          {
              **group,
              "functionDeclarations": [
                  declaration
                  for declaration in group.get("functionDeclarations", [])
                  if declaration.get("name") in enabled
              ],
          }
          for group in agent_tools
          if group.get("functionDeclarations")
      ]
      agent_tools = [
          group for group in agent_tools if group["functionDeclarations"]
      ]

    er_orchestrator_config = None
    heartbeat_enabled = config.heartbeat_enabled
    if orchestrator_model and agent_name in ("spot", "tinybot"):
      live_tools, er_tools = tools_lib.split_for_orchestration(agent_tools)
      agent_tools = live_tools
      agent.developer_instruction = (
          agent.developer_instruction.rstrip() + "\n\n" + _LIVE_FRONTEND_ADDENDUM
      )
      er_orchestrator_config = {
          "model": orchestrator_model,
          "tools": er_tools,
          "system_instruction": (
              agent.system_instruction + "\n\n" + er_developer_instruction
          ),
          "thinking_level": "low",
      }
      # The Live front-end does not run the robot observe->act loop; the ER
      # orchestrator sends its own heartbeats while a task is active.
      heartbeat_enabled = False
      logger.info(
          "Composite mode: %s talks, %s orchestrates robot tasks (%d tools)",
          live_api_client.wire_model(resolved_model), orchestrator_model,
          sum(len(g.get("functionDeclarations", [])) for g in er_tools),
      )

    resolved_use_tts = (
        False
        if modality == "AUDIO"
        else (
            session_cfg.use_tts
            if session_cfg.use_tts is not None
            else config.use_tts
        )
    )

    logger.info(
        "Session: agent=%s, model=%s, modality=%s, tts=%s, thinking_level=%s",
        session_cfg.agent_name,
        resolved_model,
        modality,
        resolved_use_tts,
        resolved_thinking_level,
    )

    app.state.active_session = session_manager.SessionManager(
        model=resolved_model,
        embodiment_instance=current_embodiment,
        tools=agent_tools,
        system_instruction=agent.system_instruction,
        developer_instruction=agent.developer_instruction,
        response_modality=modality,
        dump_video_dir=config.dump_video_dir or None,
        api_key=config.api_key,
        heartbeat_interval_seconds=hb_interval,
        heartbeat_enabled=heartbeat_enabled,
        heartbeat_min_delay_seconds=hb_min_delay,
        agent_peers=config.agent_peers,
        peer_name=session_cfg.agent_name,
        use_event_driven_heartbeat=config.use_event_driven_heartbeat,
        media_resolution=config.media_resolution,
        heartbeat_text=heartbeat_text_override or hb_text_default,
        enable_send_message_to_user=config.enable_send_message_to_user,
        endpoint_type=session_cfg.endpoint_type,
        thinking_level=resolved_thinking_level,
        supports_blocking_tools=spec.supports_blocking_tools,
        heartbeat_only_when_task_active=not spec.heartbeat_native,
        er_orchestrator_config=er_orchestrator_config,
    )
    app.state.session_agent_name = session_cfg.agent_name

  try:

    async def audio_output(data):
      try:
        await websocket.send_bytes(data)
      except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to send audio output to WebSocket")

    # TTS callback
    tts_client = None
    text_output_cb = None
    if resolved_use_tts:
      tts_client = tts_client_lib.Tts3pClient(
          voice_name=config.tts_voice,
          language_code=config.tts_language_code,
          api_key=config.tts_api_key or config.api_key,
          audio_gain=config.tts_audio_gain,
      )
      logger.info("TTS client initialized with voice: %s", config.tts_voice)

      async def text_output(text):
        filtered_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        if not filtered_text.strip():
          return

        logger.info("Synthesizing streaming TTS for: %.50s", filtered_text)
        total_bytes = 0
        first_chunk_sent = False
        t_start = 0.0
        try:
          async for chunk in tts_client.synthesize_stream(filtered_text):
            sess = app.state.active_session
            if (
                sess is not None
                and sess.decision_making is not None
                and sess.decision_making.user_speaking.is_set()
            ):
              logger.info("Stopping TTS chunk stream: user_speaking is set")
              return
            if not first_chunk_sent:
              t_start = time.time()
              first_chunk_sent = True
            if chunk:
              await audio_output(chunk)
              total_bytes += len(chunk)
        except asyncio.CancelledError:
          logger.info("TTS text_output task cancelled on barge-in")
          raise
        except Exception as e:
          logger.error("TTS streaming failed: %s", e)

        if total_bytes > 0 and t_start > 0.0:
          total_playback_secs = total_bytes / 48000.0
          elapsed_secs = time.time() - t_start
          remaining_secs = max(0.0, total_playback_secs - elapsed_secs)
          if remaining_secs > 0:
            await asyncio.sleep(remaining_secs)

      text_output_cb = text_output

    def _offer(queue: asyncio.Queue, item) -> None:
      """Enqueue a perishable chunk without ever blocking this loop.

      Audio/video are bounded queues; if the consumer is behind (e.g. the
      upstream Gemini stream is reconnecting) a blocking put() would stall
      this loop and with it *all* input from the browser, including typed
      text. Drop the oldest chunk instead.
      """
      while True:
        try:
          queue.put_nowait(item)
          return
        except asyncio.QueueFull:
          try:
            queue.get_nowait()
          except asyncio.QueueEmpty:
            pass

    async def receive_from_client():
      """Read audio/text/image from browser WebSocket."""
      rx = {"bytes": 0, "msgs": 0, "t0": time.monotonic(), "sizes": set()}
      try:
        while True:
          message = await websocket.receive()
          if message.get("type") == "websocket.disconnect":
            logger.info("WebSocket disconnect event received")
            break
          if message.get("bytes"):
            b = message["bytes"]
            rx["bytes"] += len(b)
            rx["msgs"] += 1
            if len(rx["sizes"]) < 6:
              rx["sizes"].add(len(b))
            dt = time.monotonic() - rx["t0"]
            if dt >= 5.0:
              logger.info(
                  "[audio-rate] received from browser %.0f B/s (%.2fx of"
                  " 16 kHz), %.1f msgs/s, sizes=%s",
                  rx["bytes"] / dt, rx["bytes"] / dt / 32000.0,
                  rx["msgs"] / dt, sorted(rx["sizes"]),
              )
              rx.update(bytes=0, msgs=0, t0=time.monotonic(), sizes=set())
            _offer(current_embodiment.get_audio_queue(), b)
          elif message.get("text"):
            text = message["text"]
            try:
              payload = json.loads(text)
              if isinstance(payload, dict):
                msg_type = payload.get("type")
                if msg_type == "image":
                  data = base64.b64decode(payload["data"])
                  _offer(current_embodiment.get_video_queue(), data)
                  continue
                if msg_type == "barge_in":
                  sess = app.state.active_session
                  if sess is not None and sess.decision_making is not None:
                    await sess.decision_making.interrupt_tts_for_user(
                        "client_vad_barge_in"
                    )
                  continue
                if msg_type == "video_source":
                  logger.info(
                      "Client video_source set to %s", payload.get("source")
                  )
                  continue
                if msg_type == "abort_task":
                  sess = app.state.active_session
                  if sess is not None and sess.tool_handler is not None:
                    await sess.tool_handler.cancel_task("stop button")
                  continue
                if msg_type == "geolocation":
                  sess = app.state.active_session
                  if sess is not None:
                    if payload.get("error"):
                      logger.info(
                          "Client geolocation unavailable: %s",
                          payload.get("error"),
                      )
                      sess.client_geolocation = None
                    else:
                      sess.client_geolocation = {
                          "latitude": payload.get("latitude"),
                          "longitude": payload.get("longitude"),
                          "accuracy": payload.get("accuracy"),
                          "timestamp": payload.get("timestamp"),
                          "source": "browser_geolocation",
                      }
                      logger.info(
                          "Client geolocation: %.5f, %.5f (±%sm)",
                          float(payload.get("latitude") or 0.0),
                          float(payload.get("longitude") or 0.0),
                          payload.get("accuracy"),
                      )
                  continue
            except json.JSONDecodeError:
              pass
            await current_embodiment.get_text_queue().put(text)
      except fastapi.WebSocketDisconnect:
        logger.info("WebSocket disconnected")
      except asyncio.CancelledError:
        pass
      except Exception as e:  # pylint: disable=broad-except
        logger.error("Receive error: %s", e)

    async def run_session():
      try:
        assert app.state.active_session is not None
        async for event in app.state.active_session.start_session(
            audio_output_callback=audio_output,
            text_output_callback=text_output_cb,
        ):
          if event:
            success = True
            try:
              await websocket.send_json(event)
            except (fastapi.WebSocketDisconnect, RuntimeError):
              logger.info("WebSocket disconnected during send_json")
              success = False

            if not success:
              break
      except Exception as e:
        logger.error("Error in run_session: %s", e)

    receive_task = asyncio.create_task(receive_from_client())
    session_task = asyncio.create_task(run_session())

    done, pending = await asyncio.wait(
        [session_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
      t.cancel()
    if pending:
      await asyncio.gather(*pending, return_exceptions=True)

  except Exception as e:  # pylint: disable=broad-except
    import traceback
    traceback.print_exc()
    logger.error("Session error: %s", e, exc_info=True)
  finally:
    app.state.active_poller_ref = None
    try:
      await current_embodiment.close()
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning("Failed to close embodiment: %s", exc)
    async with app.state.session_lock:
      app.state.active_session = None
    try:
      await websocket.close()
    except Exception:  # pylint: disable=broad-except
      pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(config: ServerConfig) -> fastapi.FastAPI:
  """Creates a FastAPI app instance with the given configuration."""
  ui_dir = _resolve_ui_dir()

  application = fastapi.FastAPI()
  
  @application.on_event("startup")
  def startup_event():
    # Force our package loggers to INFO and ensure they write to stdout
    loggers = [
        logging.getLogger(),
        logging.getLogger("__main__"),
    ]
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    for l in loggers:
      l.setLevel(logging.INFO)
      l.addHandler(ch)
      l.propagate = False  # Avoid duplicate logs if root logger starts working

  application.state.config = config
  application.state.ui_dir = ui_dir
  application.state.active_poller_ref = None
  application.state.active_session = None
  application.state.session_agent_name = None
  application.state.session_lock = asyncio.Lock()
  application.state.instruction_defaults = {
      "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
      "heartbeat_min_delay_seconds": config.heartbeat_min_delay_seconds,
      "use_event_driven_heartbeat": config.use_event_driven_heartbeat,
  }

  application.add_middleware(
      cors_middleware.CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  if ui_dir:
    application.mount(
        "/static",
        fastapi_staticfiles.StaticFiles(directory=ui_dir, follow_symlink=True),
        name="static",
    )
    logger.info("UI mounted from %s", ui_dir)
  else:
    logger.warning("UI directory not found.")

  application.include_router(router)
  return application


# ---------------------------------------------------------------------------
# Secrets handling
# ---------------------------------------------------------------------------
_SECRET_FIELDS = ("api_key", "tts_api_key")


def _load_dotenv(path: str | None = None) -> int:
  """Load KEY=VALUE pairs from a .env file into os.environ.

  Zero-dependency alternative to python-dotenv. Existing environment
  variables are NEVER overridden, so values injected by Cloud Run
  (e.g. via Secret Manager) always take precedence over the local file.
  Returns the number of variables loaded.
  """
  path = path or os.getenv(
      "AGENT_DOTENV", os.path.join(os.path.dirname(__file__), ".env")
  )
  if not os.path.isfile(path):
    return 0
  loaded = 0
  with open(path, "r", encoding="utf-8") as f:
    for raw in f:
      line = raw.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      if line.startswith("export "):
        line = line[len("export "):]
      key, value = line.split("=", 1)
      key = key.strip()
      value = value.strip()
      if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
      if key and key not in os.environ:
        os.environ[key] = value
        loaded += 1
  if loaded:
    logger.info("Loaded %d variable(s) from %s", loaded, path)
  return loaded


def _redact(value: str | None) -> str | None:
  """Mask a secret for logging, keeping only a short prefix/suffix."""
  if not value:
    return value
  if len(value) <= 10:
    return "***"
  return f"{value[:6]}…{value[-4:]}"


def _redacted_config(config: "ServerConfig") -> "ServerConfig":
  """Return a copy of the config with secret fields masked."""
  return dataclasses.replace(
      config,
      **{name: _redact(getattr(config, name)) for name in _SECRET_FIELDS},
  )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
  # Load local .env (gitignored) BEFORE argparse evaluates os.getenv defaults.
  _load_dotenv()

  parser = argparse.ArgumentParser(description="Proactive Agent Server")
  parser.add_argument(
      "--model", default="models/gemini-robotics-er-2-streaming-preview", help="Model name."
  )
  parser.add_argument(
      "--robot_url", default="http://localhost:8888", help="Robot URL."
  )
  parser.add_argument(
      "--api_key",
      default=os.getenv("GEMINI_API_KEY"),
      help="Gemini API key (or set GEMINI_API_KEY env var).",
  )
  parser.add_argument(
      "--tts_api_key",
      default=os.getenv("TTS_API_KEY"),
      help="TTS API key (or set TTS_API_KEY env var).",
  )
  parser.add_argument(
      "--use_tts", action="store_true", default=True, help="Enable TTS."
  )
  parser.add_argument("--no_tts", dest="use_tts", action="store_false")
  parser.add_argument(
      "--tts_voice",
      default="Puck",
      help="TTS voice name.",
  )
  parser.add_argument(
      "--tts_language_code", default="en-US", help="TTS language code."
  )
  parser.add_argument(
      "--tts_audio_gain", type=float, default=1.0, help="TTS audio gain."
  )
  parser.add_argument(
      "--response_modality", default="AUDIO", help="Response modality."
  )
  parser.add_argument("--dump_video_dir", default="", help="Dump video dir.")
  parser.add_argument("--port", type=int, default=8000, help="Server port.")
  parser.add_argument(
      "--heartbeat_interval_seconds",
      type=float,
      default=3.0,
      help="Heartbeat interval.",
  )
  parser.add_argument(
      "--heartbeat_min_delay_seconds",
      type=float,
      default=2.0,
      help="Min heartbeat delay.",
  )
  parser.add_argument(
      "--heartbeat_enabled", action="store_true", default=True
  )
  parser.add_argument(
      "--no_heartbeat", dest="heartbeat_enabled", action="store_false"
  )
  parser.add_argument(
      "--use_event_driven_heartbeat", action="store_true", default=True
  )
  parser.add_argument(
      "--media_resolution",
      default="low",
      choices=["low", "medium", "high", "ultra_high"],
  )
  parser.add_argument(
      "--enable_send_message_to_user", action="store_true", default=True
  )
  parser.add_argument(
      "--mock_robot", action="store_true", default=False, help="Mock robot."
  )
  parser.add_argument(
      "--agent_peers",
      default="",
      help="Comma-separated peers: name=url,name=url",
  )

  args = parser.parse_args()

  # Try to load API key from file if not provided
  api_key = args.api_key
  if not api_key:
    for p in [
        os.path.expanduser("~/.config/gemini/API_KEY"),
        os.path.expanduser("~/.config/safari_sdk/API_KEY"),
    ]:
      if os.path.isfile(p):
        with open(p, "r") as f:
          api_key = f.read().strip()
          logger.info("Loaded API key from %s", p)
          break

  # Parse agent peers
  agent_peers = {}
  if args.agent_peers:
    for pair in args.agent_peers.split(","):
      pair = pair.strip()
      if "=" in pair:
        name, url = pair.split("=", 1)
        agent_peers[name.strip()] = url.strip()

  config = ServerConfig(
      model=args.model,
      robot_url=args.robot_url,
      api_key=api_key,
      tts_api_key=args.tts_api_key,
      use_tts=args.use_tts,
      tts_voice=args.tts_voice,
      tts_language_code=args.tts_language_code,
      tts_audio_gain=args.tts_audio_gain,
      response_modality=args.response_modality,
      dump_video_dir=args.dump_video_dir,
      heartbeat_interval_seconds=args.heartbeat_interval_seconds,
      heartbeat_min_delay_seconds=args.heartbeat_min_delay_seconds,
      heartbeat_enabled=args.heartbeat_enabled,
      use_event_driven_heartbeat=args.use_event_driven_heartbeat,
      media_resolution=args.media_resolution,
      enable_send_message_to_user=args.enable_send_message_to_user,
      mock_robot=args.mock_robot,
      agent_peers=agent_peers,
      port=args.port,
  )

  if not config.api_key:
    logger.error(
        "No Gemini API key configured. Set GEMINI_API_KEY in the environment, "
        "in agent/.env (see .env.example), or pass --api_key."
    )
    raise SystemExit(2)

  logger.info("Server config: %s", _redacted_config(config))
  app_instance = create_app(config)

  uvicorn_config = uvicorn.Config(
      app_instance, host="0.0.0.0", port=config.port, log_level="info"
  )
  server = uvicorn.Server(uvicorn_config)

  def _handle_shutdown(sig, frame):
    del sig, frame
    logger.info("Received shutdown signal, stopping server...")
    server.should_exit = True

  signal.signal(signal.SIGINT, _handle_shutdown)
  signal.signal(signal.SIGTERM, _handle_shutdown)

  server.run()


if __name__ == "__main__":
  main()
