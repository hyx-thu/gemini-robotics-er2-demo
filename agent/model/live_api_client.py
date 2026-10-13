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
