"""Robotics Developer API client for Gemini bidirectional streaming via gRPC.

This module provides a client that connects to the Robotics Developer API
(RoboticsBidiGenerateContent) over gRPC. It translates incoming and outgoing
messages to/from JSON-compatible dictionaries, making it transparently
interchangeable with GeminiLiveApiClient.
"""

import logging
import os
import queue
import sys
import threading
from typing import Any, Callable

# Add path to the compiled proto packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robotics_developer"))

import grpc
try:
  from google.ai.generativelanguage_v1alpha.types import generative_service
  if not hasattr(generative_service, "BidiGenerateContentClientMessage"):
    from google.ai.generativelanguage_v1beta.types import generative_service
except (ImportError, AttributeError):
  from google.ai.generativelanguage_v1beta.types import generative_service
from google.protobuf import json_format
from google.robotics.developer.v1 import modelserving_pb2
from google.robotics.developer.v1 import modelserving_pb2_grpc

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class RequestIterator:
  """Iterator that yields requests from a queue."""

  def __init__(self):
    self._queue = queue.Queue()
    self._done = False

  def __iter__(self):
    return self

  def __next__(self):
    while True:
      if self._done and self._queue.empty():
        raise StopIteration
      try:
        return self._queue.get(timeout=0.1)
      except queue.Empty:
        if self._done:
          raise StopIteration
        continue

  def put(self, item):
    self._queue.put(item)

  def close(self):
    self._done = True


class GrpcBidiStreamWrapper:
  """Wraps a standard gRPC bidi stream to look like pywraprpc.MessageStream."""

  def __init__(self, grpc_stream, request_iterator):
    self._grpc_stream = grpc_stream
    self._request_iterator = request_iterator
    self._on_message = None
    self._on_done = None

  def Start(self, on_message: Callable, on_done: Callable) -> None:
    self._on_message = on_message
    self._on_done = on_done
    logger.info("GrpcBidiStreamWrapper.Start called")
    threading.Thread(target=self._read_loop, daemon=True).start()
    logger.info("GrpcBidiStreamWrapper thread started")

  def _read_loop(self):
    logger.info("GrpcBidiStreamWrapper._read_loop started")
    try:
      for response in self._grpc_stream:
        logger.info(
            "GrpcBidiStreamWrapper received response: %s",
            type(response).__name__,
        )
        if self._on_message:
          self._on_message(response)
    except grpc.RpcError as e:
      logger.error(
          "gRPC RpcError in read loop: code=%s details=%s trailing_metadata=%s",
          e.code(),
          e.details(),
          e.trailing_metadata(),
      )
    except Exception as e:
      logger.error("Error in gRPC read loop: %s (type=%s)", e, type(e).__name__)
    finally:
      logger.info("GrpcBidiStreamWrapper._read_loop exiting, calling on_done")
      if self._on_done:
        self._on_done()

  def Send(self, request: Any) -> None:
    logger.debug(
        "GrpcBidiStreamWrapper.Send: type=%s",
        type(request).__name__,
    )
    self._request_iterator.put(request)

  def HalfClose(self) -> None:
    self._request_iterator.close()

  def GetStatus(self) -> Any:
    return None


class RoboticsDeveloperStream:
  """Wrapper around GrpcBidiStreamWrapper matching GeminiLiveApiStream interface.

  It translates gRPC protobuf packets containing serialized inner protobufs to
  and from Python dictionaries conforming to the JSON Live API.
  """

  def __init__(self, stream: GrpcBidiStreamWrapper):
    self._stream = stream

  def Start(
      self,
      on_message: Callable[[dict], None],
      on_done: Callable[[], None],
  ) -> None:
    """Starts the stream and translates messages to JSON dicts on arrival."""
    def _on_robotics_message(msg):
      if msg is None:
        on_message(None)
        return
      
      try:
        # 1. Deserialize the outer gRPC response bytes to inner proto turn message
        server_msg = generative_service.BidiGenerateContentServerMessage.deserialize(
            msg.message
        )
        # 2. Convert inner proto to a JSON-compatible Python dict in camelCase
        server_msg_dict = json_format.MessageToDict(
            server_msg._pb,
            preserving_proto_field_name=False,
        )
        logger.info("Received message from Robotics Developer API: %s", server_msg_dict)
        on_message(server_msg_dict)
      except Exception as e:
        logger.exception("Failed to parse incoming gRPC message: %s", e)

    self._stream.Start(_on_robotics_message, on_done)

  def Send(self, msg: dict[str, Any]) -> None:
    """Translates a JSON dict to protobuf and sends it over gRPC."""
    try:
      # 1. Instantiate the inner proto-plus wrapper
      client_msg = generative_service.BidiGenerateContentClientMessage()
      # 2. Parse the camelCase dict into the underlying protobuf message descriptor
      json_format.ParseDict(msg, client_msg._pb, ignore_unknown_fields=True)
      # 3. Serialize to protobuf bytes
      serialized_bytes = generative_service.BidiGenerateContentClientMessage.serialize(
          client_msg
      )
      # 4. Wrap in the outer gRPC request message
      request = modelserving_pb2.RoboticsBidiGenerateContentRequest(
          message=serialized_bytes,
      )
      self._stream.Send(request)
    except Exception as e:
      logger.exception("Failed to send gRPC message: %s", e)
      raise e

  def HalfClose(self) -> None:
    self._stream.HalfClose()

  def GetStatus(self) -> Any:
    return self._stream.GetStatus()


class RoboticsDeveloperClient:
  """gRPC client for Robotics Developer API (ModelServing service)."""

  def __init__(self, api_key: str | None = None):
    self._api_key = api_key
    addr = "dns:///roboticsdeveloper.googleapis.com:443"
    channel_creds = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(
        addr,
        channel_creds,
        options=[("grpc.service_config_disable_resolution", 1)],
    )
    self._robotics_stub = modelserving_pb2_grpc.ModelServingStub(channel)
    logger.info("RoboticsDeveloperClient initialized via gRPC -> %s", addr)

  def create_stream(self) -> RoboticsDeveloperStream:
    metadata = []
    if self._api_key:
      metadata.append(("x-goog-api-key", self._api_key))

    request_iterator = RequestIterator()
    try:
      grpc_stream = self._robotics_stub.RoboticsBidiGenerateContent(
          request_iterator, metadata=metadata
      )
      logger.info("gRPC stream object created: %s", type(grpc_stream).__name__)
    except grpc.RpcError as e:
      logger.error(
          "Failed to create gRPC stream: code=%s details=%s",
          e.code(),
          e.details(),
      )
      raise
    wrapper = GrpcBidiStreamWrapper(grpc_stream, request_iterator)
    return RoboticsDeveloperStream(wrapper)
