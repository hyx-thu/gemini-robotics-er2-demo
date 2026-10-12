#!/usr/bin/env python3
# /// script
# dependencies = [
#   "grpcio>=1.60.0",
#   "protobuf>=4.25.0",
#   "google-ai-generativelanguage>=0.6.0",
#   "pillow>=10.0.0",
# ]
# ///

"""Standalone Example: ER2 Model Inference via Google Robotics Developer API.

===============================================================================
HIGH-LEVEL OVERVIEW & ARCHITECTURE
===============================================================================

This script demonstrates how to perform bidirectional streaming inference with
Gemini ER2 (Embodied Reasoning 2.0) models using Google's Robotics Developer
API over gRPC (`roboticsdeveloper.googleapis.com:443`).

Key Components & Protocol Flow:
--------------------------------
1. **gRPC Channel Setup**:
   Connects securely to `dns:///roboticsdeveloper.googleapis.com:443` passing
   the Gemini API key via the `x-goog-api-key` HTTP metadata header.

2. **In-Memory Protobuf & gRPC Definitions**:
   The script dynamically builds `RoboticsBidiGenerateContentRequest` and
   `RoboticsBidiGenerateContentResponse` descriptors in memory using
   `google.protobuf.descriptor_pool.DescriptorPool()`. This eliminates any
   dependency on local compiled `.proto` files or repository paths.

3. **Session Setup Packet (`setup`)**:
   Sends an initial setup message declaring the target model (e.g.
   `models/robotics_er_live_text_only_2p0_no_safety_classifiers`), system
   instructions (including silent thinking budget / effort level), output
   modalities (`TEXT`), and available robot tool function declarations (e.g.
   `run_instruction`).

4. **Synthetic Video / Image Frame Streaming (`realtimeInput`)**:
   Generates a synthetic random RGB JPEG image in memory and sends it to the
   model as a camera input frame (`realtimeInput.video`).

5. **User Turn & Model Response (`clientContent` / `serverContent`)**:
   Sends the user text prompt alongside the image frame in a completed turn.
   The model streams back response chunks, silent thinking tokens, and robot
   action tool calls (e.g., `run_instruction(instruction='...')`).

===============================================================================
USAGE INSTRUCTIONS
===============================================================================

Set your Gemini API Key and run with `uv`:

    export GEMINI_API_KEY="your_gemini_api_key"
    uv run agent/scripts/er2_developer_api_example.py

Or specify custom prompt, model, or effort level flags:

    uv run agent/scripts/er2_developer_api_example.py \
        --api_key "your_api_key" \
        --model robotics_er_live_text_only_2p0_no_safety_classifiers \
        --prompt "Pick up the blue cube and place it on the red tray." \
        --effort_level 0.75
"""

import argparse
import base64
import io
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from typing import Any

import grpc
from google.ai.generativelanguage_v1beta.types import generative_service
from google.protobuf import descriptor_pool
from google.protobuf import json_format
from google.protobuf.internal import builder as _builder
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# 1. In-Memory Protobuf & gRPC Service Definitions
# -----------------------------------------------------------------------------
# Serialized FileDescriptorProto for google/robotics/developer/v1/modelserving.proto
_DESCRIPTOR_BYTES = (
    b'\n/google/robotics/developer/v1/modelserving.proto\x12\x1cgoogle.robotics.developer.v1"5\n"RoboticsBidiGenerateContentRequest\x12\x0f\n\x07message\x18\x01'
    b' \x01(\x0c"6\n#RoboticsBidiGenerateContentResponse\x12\x0f\n\x07message\x18\x01'
    b' \x01(\x0c2\xb7\x01\n\x0cModelServing\x12\xa6\x01\n\x1bRoboticsBidiGenerateContent\x12@.google.robotics.developer.v1.RoboticsBidiGenerateContentRequest\x1aA.google.robotics.developer.v1.RoboticsBidiGenerateContentResponse(\x010\x01B$\n'
    b' com.google.robotics.developer.v1P\x01b\x06proto3'
)

# Build protobuf message descriptors in an isolated pool to avoid global symbol conflicts
_pool = descriptor_pool.DescriptorPool()
_file_descriptor = _pool.AddSerializedFile(_DESCRIPTOR_BYTES)
_proto_globals: dict[str, Any] = {}
_builder.BuildMessageAndEnumDescriptors(_file_descriptor, _proto_globals)
_builder.BuildTopDescriptorsAndMessages(
    _file_descriptor,
    "google.robotics.developer.v1.modelserving_pb2",
    _proto_globals,
)


class modelserving_pb2:  # pylint: disable=invalid-name
  """Container for the generated outer gRPC request/response protobuf classes."""

  RoboticsBidiGenerateContentRequest = _proto_globals[
      "RoboticsBidiGenerateContentRequest"
  ]
  RoboticsBidiGenerateContentResponse = _proto_globals[
      "RoboticsBidiGenerateContentResponse"
  ]


class _ModelServingStub:
  """gRPC stub for the ModelServing service (RoboticsBidiGenerateContent method)."""

  def __init__(self, channel: grpc.Channel):
    self.RoboticsBidiGenerateContent = channel.stream_stream(
        "/google.robotics.developer.v1.ModelServing/RoboticsBidiGenerateContent",
        request_serializer=(
            modelserving_pb2.RoboticsBidiGenerateContentRequest.SerializeToString
        ),
        response_deserializer=(
            modelserving_pb2.RoboticsBidiGenerateContentResponse.FromString
        ),
    )


class modelserving_pb2_grpc:  # pylint: disable=invalid-name
  """Container for gRPC client stubs."""

  ModelServingStub = _ModelServingStub


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("er2_example")


# -----------------------------------------------------------------------------
# 2. Synthetic Image Generator
# -----------------------------------------------------------------------------
def generate_random_image_bytes(width: int = 320, height: int = 240) -> bytes:
  """Generates a synthetic random RGB JPEG image in memory.

  Args:
      width: Image width in pixels (default: 320).
      height: Image height in pixels (default: 240).

  Returns:
      Encoded JPEG image as bytes.
  """
  bg_color = (
      random.randint(0, 255),
      random.randint(0, 255),
      random.randint(0, 255),
  )
  img = Image.new("RGB", (width, height), color=bg_color)
  draw = ImageDraw.Draw(img)

  # Draw a random colored rectangle to simulate an object in the scene
  rect_color = (
      random.randint(0, 255),
      random.randint(0, 255),
      random.randint(0, 255),
  )
  draw.rectangle([40, 40, width - 40, height - 40], fill=rect_color)

  buf = io.BytesIO()
  img.save(buf, format="JPEG")
  return buf.getvalue()


# -----------------------------------------------------------------------------
# 3. Thread-Safe gRPC Streaming Request Queue
# -----------------------------------------------------------------------------
class RequestIterator:
  """Thread-safe iterator yielding gRPC request messages to the streaming RPC."""

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

  def put_dict(self, msg_dict: dict[str, Any]) -> None:
    """Converts a JSON-compatible Python dict into a gRPC request packet."""
    client_msg = generative_service.BidiGenerateContentClientMessage()
    json_format.ParseDict(msg_dict, client_msg._pb, ignore_unknown_fields=True)
    serialized_bytes = (
        generative_service.BidiGenerateContentClientMessage.serialize(
            client_msg
        )
    )
    request = modelserving_pb2.RoboticsBidiGenerateContentRequest(
        message=serialized_bytes
    )
    self._queue.put(request)

  def close(self) -> None:
    self._done = True


# -----------------------------------------------------------------------------
# 4. Main Inference Function
# -----------------------------------------------------------------------------
def run_er2_inference(
    api_key: str,
    model: str,
    prompt: str,
    effort_level: float = 0.75,
    timeout_seconds: float = 15.0,
):
  """Establishes gRPC stream with ER2 model, sends inputs, and prints response.

  Args:
      api_key: Gemini API Key for authentication.
      model: Target ER2 model name.
      prompt: User prompt string.
      effort_level: Silent thinking effort budget (0.0 to 1.0).
      timeout_seconds: Max seconds to wait for model response completion.
  """
  model_name = model if model.startswith("models/") else f"models/{model}"
  server_addr = "dns:///roboticsdeveloper.googleapis.com:443"

  logger.info("Connecting to Robotics Developer API at %s...", server_addr)
  channel_creds = grpc.ssl_channel_credentials()
  channel = grpc.secure_channel(
      server_addr,
      channel_creds,
      options=[("grpc.service_config_disable_resolution", 1)],
  )
  stub = modelserving_pb2_grpc.ModelServingStub(channel)

  request_iterator = RequestIterator()

  # Step A: Construct session setup message
  system_instruction = (
      f"SPECIAL INSTRUCTION: think silently if needed. EFFORT LEVEL:"
      f" {effort_level:.2f}."
  )

  # Function declaration for robot instruction execution
  tools = [{
      "functionDeclarations": [{
          "name": "run_instruction",
          "description": (
              "Execute a natural language action/instruction for the robot."
          ),
          "parameters": {
              "type": "OBJECT",
              "properties": {
                  "instruction": {
                      "type": "STRING",
                      "description": (
                          "Atomic action instruction (e.g. 'pick up the blue"
                          " cube')"
                      ),
                  }
              },
              "required": ["instruction"],
          },
      }]
  }]

  setup_msg = {
      "setup": {
          "model": model_name,
          "systemInstruction": {
              "role": "system",
              "parts": [{"text": system_instruction}],
          },
          "generationConfig": {"responseModalities": ["TEXT"]},
          "tools": tools,
      }
  }

  # Step B: Generate synthetic camera frame
  image_bytes = generate_random_image_bytes(320, 240)
  image_b64 = base64.b64encode(image_bytes).decode("utf-8")
  logger.info("Generated random synthetic JPEG image (%d bytes)", len(image_bytes))

  realtime_image_msg = {
      "realtimeInput": {
          "video": {
              "mimeType": "image/jpeg",
              "data": image_b64,
          }
      }
  }

  # Step C: Construct user text & image turn
  user_turn_msg = {
      "clientContent": {
          "turns": [{
              "role": "user",
              "parts": [
                  {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                  {"text": prompt},
              ],
          }],
          "turnComplete": True,
      }
  }

  # Enqueue requests: setup -> image -> user turn
  request_iterator.put_dict(setup_msg)
  request_iterator.put_dict(realtime_image_msg)
  request_iterator.put_dict(user_turn_msg)

  metadata = [("x-goog-api-key", api_key)]
  logger.info("Sending setup, synthetic image frame, and prompt: %r", prompt)

  try:
    grpc_stream = stub.RoboticsBidiGenerateContent(
        request_iterator, metadata=metadata
    )
  except grpc.RpcError as e:
    logger.error("Failed to initiate gRPC stream: %s (%s)", e.details(), e.code())
    return

  # Listener thread to print incoming model responses
  def response_listener():
    try:
      for response in grpc_stream:
        server_msg = (
            generative_service.BidiGenerateContentServerMessage.deserialize(
                response.message
            )
        )
        msg_dict = json_format.MessageToDict(
            server_msg._pb, preserving_proto_field_name=False
        )
        print("\n--- Incoming Model Response Message ---")
        print(json.dumps(msg_dict, indent=2))

        # Log parsed tool calls if present
        if "toolCall" in msg_dict:
          calls = msg_dict["toolCall"].get("functionCalls", [])
          for fc in calls:
            logger.info(">>> ER2 Model Tool Call: %s(%s)", fc.get("name"), fc.get("args"))

        if msg_dict.get("serverContent", {}).get("turnComplete"):
          logger.info("Turn completed by ER2 model.")
          break
    except grpc.RpcError as e:
      logger.error("gRPC stream error: code=%s, details=%s", e.code(), e.details())
    except Exception as e:
      logger.exception("Error in response listener: %s", e)

  listener_thread = threading.Thread(target=response_listener, daemon=True)
  listener_thread.start()

  # Wait for turn completion or timeout
  start_time = time.time()
  while listener_thread.is_alive() and (time.time() - start_time < timeout_seconds):
    time.sleep(0.2)

  request_iterator.close()
  listener_thread.join(timeout=2.0)
  logger.info("Finished ER2 inference session.")


# -----------------------------------------------------------------------------
# 5. CLI Entry Point
# -----------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(
      description="Inference ER2 model via Google Robotics Developer API (gRPC)"
  )
  parser.add_argument(
      "--api_key",
      default=os.getenv("GEMINI_API_KEY", ""),
      help="Gemini API Key (or set GEMINI_API_KEY env var)",
  )
  parser.add_argument(
      "--model",
      default="robotics_er_live_text_only_2p0_no_safety_classifiers",
      help="ER2 model name",
  )
  parser.add_argument(
      "--prompt",
      default="Pick up the blue cube and place it on the red tray.",
      help="User input prompt",
  )
  parser.add_argument(
      "--effort_level",
      type=float,
      default=0.75,
      help="Thinking effort level (0.0 to 1.0)",
  )
  args = parser.parse_args()

  if not args.api_key:
    sys.stderr.write(
        "Error: Gemini API Key is required. Provide --api_key or set"
        " GEMINI_API_KEY environment variable.\n"
    )
    sys.exit(1)

  run_er2_inference(
      api_key=args.api_key,
      model=args.model,
      prompt=args.prompt,
      effort_level=args.effort_level,
  )


if __name__ == "__main__":
  main()
