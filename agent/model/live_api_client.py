"""Gemini Live API client for bidirectional streaming via WebSockets.

This module provides a client that connects to the public Gemini Live API
(BidiGenerateContent) over WebSockets, enabling real-time voice and vision
interactions without requiring internal Google infrastructure (pywraprpc/LOAS).

All messages are sent and received as plain Python dicts (JSON), with no
protobuf dependency.

Usage:
  client = GeminiLiveApiClient(api_key="YOUR_API_KEY")
  stream = client.create_stream()

WebSocket endpoint (API key sent as the x-goog-api-key header):
  wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha
      .GenerativeService.BidiGenerateContent
"""

import logging
import threading
from typing import Any, Callable

import websocket

logger = logging.getLogger(__name__)

_LIVE_API_WSS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha"
    ".GenerativeService.BidiGenerateContent"
)


# pylint: disable=invalid-name


class GeminiLiveApiStream:
  """Wrapper around a WebSocket connection to the Gemini Live API.

  Provides the streaming interface (Start, Send, HalfClose, GetStatus) so
  that the SessionManager can use it transparently.
  """

  def __init__(self, ws):
    self._ws = ws
    self._on_message = None
    self._on_done = None
    self._read_thread = None
    # Why the socket ended, for diagnostics / reconnect decisions. ``closed``
    # flips once the read loop has exited; ``closed_locally`` when we hung up.
    self.close_code: int | None = None
    self.close_reason: str = ""
    self.closed: bool = False
    self.closed_locally: bool = False
    # Set by the owner to tell streams apart across reconnects.
    self.generation: int = 0
    # Vertex AI only: "projects/P/locations/L/publishers/google/models".
    self.model_resource_prefix: str | None = None

  def Start(
      self, on_message: Callable[[dict], None],
      on_done: Callable[[], None],
  ) -> None:
    """Start reading messages from the WebSocket in a background thread."""
    logger.debug("GeminiLiveApiStream.Start() called")
    self._on_message = on_message
    self._on_done = on_done
    self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
    self._read_thread.start()
    logger.debug("GeminiLiveApiStream read thread spawned and started")

  def _read_loop(self) -> None:
    """Reads messages from the WebSocket and dispatches to on_message."""
    logger.debug("GeminiLiveApiStream._read_loop() thread started")
    try:
      while True:
        opcode, data = self._ws.recv_data()
        if opcode == websocket.ABNF.OPCODE_CLOSE:
          import struct  # pylint: disable=g-import-not-at-top
          code = 1000
          reason = ""
          if len(data) >= 2:
            code = struct.unpack("!H", data[0:2])[0]
            reason = data[2:].decode("utf-8", errors="replace")
          self.close_code = code
          self.close_reason = reason
          logger.info(
              "WebSocket closed by server. code=%s, reason=%s", code, reason
          )
          break

        if opcode not in (
            websocket.ABNF.OPCODE_TEXT,
            websocket.ABNF.OPCODE_BINARY,
        ):
          continue

        raw = data

        # If it is bytes, it might still be a UTF-8 JSON string.
        if isinstance(raw, bytes):
          try:
            raw = raw.decode("utf-8")
          except UnicodeDecodeError:
            logger.warning("Received binary data but expected JSON")

        if isinstance(raw, str):
          import json
          try:
            parsed_msg = json.loads(raw)
            if "inlineData" in raw:
              ot = (
                  (parsed_msg.get("serverContent") or {})
                  .get("outputTranscription", {})
                  .get("text")
              )
              if ot:
                logger.info(
                    "Received audio+outputTranscription from Gemini Live API:"
                    " %r",
                    ot,
                )
            elif parsed_msg:
              logger.info(
                  "Received message from Gemini Live API: %s", parsed_msg
              )
          except json.JSONDecodeError as e:
            logger.warning("Failed to decode JSON: %s", e)
            continue
        else:
          logger.warning("Unexpected message type from WebSocket: %s", type(raw))
          continue

        if self._on_message:
          self._on_message(parsed_msg)
    except Exception as e:  # pylint: disable=broad-except
      if self.closed_locally:
        logger.info("GeminiLiveApiStream read loop ended after local close")
      else:
        logger.warning("Error in GeminiLiveApiStream read loop: %s", e)
        if not self.close_reason:
          self.close_reason = f"{type(e).__name__}: {e}"
    finally:
      self.closed = True
      logger.debug("GeminiLiveApiStream read loop exiting, calling on_done")
      if self._on_done:
        self._on_done()

  def Send(self, msg: Any) -> None:
    """Send a client message (must be a JSON-serializable dict)."""
    try:
      import json  # pylint: disable=g-import-not-at-top
      if self.model_resource_prefix and isinstance(msg, dict) and "setup" in msg:
        # Vertex AI wants a full publisher-model resource name, not models/X.
        setup = dict(msg["setup"])
        name = str(setup.get("model", "")).removeprefix("models/")
        if name and not name.startswith("projects/"):
          setup["model"] = f"{self.model_resource_prefix}/{name}"
        msg = {**msg, "setup": setup}
      json_str = json.dumps(msg)
      self._ws.send(json_str)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("GeminiLiveApiStream.Send failed: %s", e)
      raise e
    return None

  def HalfClose(self) -> None:
    """Signal that no more messages will be sent."""
    self.closed_locally = True
    try:
      self._ws.close()
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("GeminiLiveApiStream.HalfClose error: %s", e)

  def GetStatus(self) -> Any:
    return None

  def Shutdown(self) -> None:
    """Force-close the WebSocket connection."""
    self.closed_locally = True
    try:
      self._ws.close()
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("GeminiLiveApiStream.Shutdown error: %s", e)


class GeminiLiveApiClient:
  """Client for the public Gemini Live API (BidiGenerateContent over WSS).

  This client uses the websocket-client library to establish a persistent
  WebSocket connection to the Gemini Live API endpoint.
  """

  def __init__(self, api_key: str | None = None):
    if not api_key:
      raise ValueError("api_key is required for the Gemini Live API endpoint.")
    self._api_key = api_key
    # The key travels in the x-goog-api-key header, never in the URL, so it
    # cannot leak through URL logging, proxies or exception messages.
    self._url = _LIVE_API_WSS_URL
    logger.info(
        "GeminiLiveApiClient initialized -> %s",
        _LIVE_API_WSS_URL,
    )

  def create_stream(self) -> GeminiLiveApiStream:
    """Create a new WebSocket connection and return a stream wrapper."""
    ws = websocket.create_connection(
        self._url,
        header={
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        },
    )
    logger.info("WebSocket connection established to Gemini Live API")
    return GeminiLiveApiStream(ws)


_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class VertexLiveApiClient:
  """Client for the Vertex AI Live API (LlmBidiService over WSS).

  Auth is OAuth 2.0 via Application Default Credentials: on Cloud Run that is
  the service's runtime service account (needs roles/aiplatform.user); locally
  run ``gcloud auth application-default login``. No API key is involved.

  The wire protocol (setup / realtimeInput / toolResponse / serverContent) is
  the same JSON as the Gemini Developer API, so the same GeminiLiveApiStream
  wrapper is reused; only the URL, the Authorization header and the model
  resource name differ.
  """

  def __init__(
      self,
      project: str | None = None,
      location: str = "us-central1",
      token_provider: Callable[[], str] | None = None,
  ):
    self._location = location
    self._token_provider = token_provider
    self._credentials = None
    if token_provider is None:
      import google.auth  # pylint: disable=g-import-not-at-top
      self._credentials, adc_project = google.auth.default(scopes=_VERTEX_SCOPES)
      project = project or adc_project
    if not project:
      raise ValueError(
          "Vertex AI project is unknown: set VERTEX_PROJECT (or"
          " GOOGLE_CLOUD_PROJECT)."
      )
    self._project = project
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    self._url = (
        f"wss://{host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/"
        "BidiGenerateContent"
    )
    self.model_resource_prefix = (
        f"projects/{project}/locations/{location}/publishers/google/models"
    )
    logger.info(
        "VertexLiveApiClient initialized -> %s (project=%s, location=%s)",
        self._url, project, location,
    )

  def _access_token(self) -> str:
    if self._token_provider is not None:
      return self._token_provider()
    import google.auth.transport.requests  # pylint: disable=g-import-not-at-top
    creds = self._credentials
    if not creds.valid:
      creds.refresh(google.auth.transport.requests.Request())
    return creds.token

  def create_stream(self) -> GeminiLiveApiStream:
    """Open a new Vertex Live WebSocket (fresh token each time)."""
    ws = websocket.create_connection(
        self._url,
        header={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token()}",
        },
    )
    logger.info("WebSocket connection established to Vertex AI Live API")
    stream = GeminiLiveApiStream(ws)
    stream.model_resource_prefix = self.model_resource_prefix
    return stream


# Models served from Gemini Enterprise Agent Platform / Vertex AI (OAuth).
# Everything else stays on the Gemini Developer API (API key).
# gemini-3.8-live-extended-thinking is not a Vertex publisher model and Vertex
# rejects custom thinking levels on gemini-3.8-live unless allowlisted.
# If a Vertex Live model rejects setup (e.g. project not yet allowlisted for a
# preview Live model), disable_vertex_live() switches it to AI Studio for the
# remainder of the process.
VERTEX_LIVE_MODELS = frozenset({
    "gemini-3.8-live",
    "gemini-robotics-er-2-streaming-preview",
})

# Per-model Vertex Live location (gemini-robotics-er-2-streaming-preview is
# configured on global aiplatform.googleapis.com; gemini-3.8-live uses
# VERTEX_LOCATION, default us-central1).
VERTEX_LIVE_LOCATIONS: dict[str, str] = {
    "gemini-robotics-er-2-streaming-preview": "global",
}

_disabled_live_models: set[str] = set()

# Non-Live LLMs run as a cascade: Gemini ASR -> generateContent -> our TTS.
CASCADE_MODELS = frozenset({
    "gemini-4-argon",
    "gemini-robotics-er-2-preview",
})

# Composite modes: "<live model>+<tag>" = the Live model talks to the user and
# delegates robot task orchestration to a non-streaming model used as a tool.
ORCHESTRATED_MODELS: dict[str, tuple[str, str]] = {
    "gemini-3.8-live+er2": ("gemini-3.8-live", "gemini-robotics-er-2-preview"),
}


def wire_model(model: str) -> str:
  """The model name actually sent in the Live setup message."""
  name = model.removeprefix("models/")
  if name in ORCHESTRATED_MODELS:
    return ORCHESTRATED_MODELS[name][0]
  return name


def orchestrator_model(model: str) -> str | None:
  """Robot-task orchestrator model for composite modes (else None)."""
  pair = ORCHESTRATED_MODELS.get(model.removeprefix("models/"))
  return pair[1] if pair else None


def vertex_live_location(model: str) -> str:
  """Return the Vertex AI location for a Live model."""
  import os  # pylint: disable=g-import-not-at-top
  w = wire_model(model)
  if w in VERTEX_LIVE_LOCATIONS:
    return os.environ.get("VERTEX_ER_LIVE_LOCATION", VERTEX_LIVE_LOCATIONS[w])
  return os.environ.get("VERTEX_LOCATION", "us-central1")


def disable_vertex_live(model: str, reason: str) -> None:
  """Disable Vertex Live for ``model`` and fall back to AI Studio for this process."""
  w = wire_model(model)
  if w not in _disabled_live_models:
    _disabled_live_models.add(w)
    logger.warning(
        "Vertex Live disabled for %s -> AI Studio fallback (%s)", w, reason
    )


def uses_vertex(model: str) -> bool:
  """Whether this model should be served from Vertex AI."""
  import os  # pylint: disable=g-import-not-at-top
  if os.environ.get("LIVE_BACKEND", "").lower() == "aistudio":
    return False
  w = wire_model(model)
  return w in VERTEX_LIVE_MODELS and w not in _disabled_live_models


def make_client(model: str, api_key: str | None):
  """Return the Live client (Vertex, Developer API or cascade) for ``model``."""
  import os  # pylint: disable=g-import-not-at-top
  if model.removeprefix("models/") in CASCADE_MODELS:
    from model import cascade_client  # pylint: disable=g-import-not-at-top
    return cascade_client.CascadeClient(model, api_key)
  if uses_vertex(model):
    try:
      return VertexLiveApiClient(
          project=os.environ.get("VERTEX_PROJECT")
          or os.environ.get("GOOGLE_CLOUD_PROJECT"),
          location=vertex_live_location(model),
      )
    except Exception as e:  # pylint: disable=broad-except
      disable_vertex_live(model, f"init failed: {e}")
      if not api_key:
        raise
  return GeminiLiveApiClient(api_key=api_key)
