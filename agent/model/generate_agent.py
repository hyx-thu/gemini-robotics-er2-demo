"""Turn-based Gemini agent over REST generateContent (non-Live models).

Shared engine for the two non-streaming model modes:

* ``gemini-4-argon`` cascade: Gemini ASR (Live transcription) -> Argon
  generateContent -> our TTS, wrapped as a Live-compatible stream
  (model/cascade_client.py) so the rest of the agent is unchanged.
* ``gemini-3.8-live+er2``: Gemini 3.8 Live talks to the user and delegates
  robot tasks to Gemini Robotics-ER 2 (non-streaming), which runs its own
  plan/act loop through this engine (core/er_orchestrator.py).

The engine only manages the conversation history and the HTTP call; tool
execution belongs to the caller. Models available on Vertex AI are
called there (ADC); the rest use the Gemini Developer API with the key in the
x-goog-api-key header, never in the URL (see model/vertex_rest.py).
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from model import vertex_rest

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
_RETRY_STATUS = (429, 500, 502, 503, 504)


def strip_declarations(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Live-only fields (``behavior``) are not part of generateContent."""
  out = []
  for group in tools or []:
    decls = group.get("functionDeclarations")
    if not decls:
      continue
    clean = []
    for d in decls:
      d = dict(d)
      d.pop("behavior", None)
      clean.append(d)
    out.append({"functionDeclarations": clean})
  return out


def image_part(jpeg_b64: str) -> dict[str, Any]:
  return {"inlineData": {"mimeType": "image/jpeg", "data": jpeg_b64}}


def jpeg_to_b64(chunk: bytes) -> str:
  return base64.b64encode(chunk).decode("utf-8")


class GenerateContentAgent:
  """Conversation state + generateContent calls for one model."""

  def __init__(
      self,
      model: str,
      api_key: str,
      system_instruction: str = "",
      tools: list[dict[str, Any]] | None = None,
      generation_config: dict[str, Any] | None = None,
      max_contents: int = 48,
      keep_images: int = 2,
      timeout_s: float = 120.0,
  ):
    self.model = model.removeprefix("models/")
    self._api_key = api_key
    self.system_instruction = system_instruction
    self.tools = strip_declarations(tools or [])
    self.generation_config = dict(generation_config or {})
    self.max_contents = max_contents
    self.keep_images = keep_images
    self.timeout_s = timeout_s
    self.history: list[dict[str, Any]] = []
    self.last_usage: dict[str, Any] = {}

  # ---- history ----

  def add(self, role: str, parts: list[dict[str, Any]]) -> None:
    if not parts:
      return
    if self.history and self.history[-1]["role"] == role:
      # Same-role neighbours (e.g. functionResponse + a new user text) are
      # merged into one content.
      self.history[-1]["parts"].extend(parts)
    else:
      self.history.append({"role": role, "parts": list(parts)})

  def add_tool_responses(
      self, responses: list[dict[str, Any]], frame_parts: list[dict[str, Any]]
  ) -> None:
    """Function responses, then the fresh camera frame as its OWN user turn.

    Vertex AI rejects a user content that mixes functionResponse parts with
    inlineData ("Requests ending with a model turn are not supported"), so
    the frame goes into a separate (consecutive) user content.
    """
    self.add("user", responses)
    if frame_parts:
      self.history.append(
          {"role": "user", "parts": [{"text": "[latest camera frame]"}] + list(frame_parts)}
      )

  def truncate(self, length: int) -> None:
    """Drop everything after ``length`` contents (e.g. an idle ack turn)."""
    del self.history[length:]

  def _prepared_contents(self) -> list[dict[str, Any]]:
    contents = copy.deepcopy(self.history)
    # Only the newest images stay; older frames become a short marker so a
    # long session does not re-bill hundreds of images every turn.
    seen = 0
    for content in reversed(contents):
      new_parts = []
      for part in reversed(content["parts"]):
        if "inlineData" in part:
          seen += 1
          if seen > self.keep_images:
            continue
        new_parts.append(part)
      content["parts"] = list(reversed(new_parts)) or [{"text": "[frame]"}]
    # Bound the history; always start at a user turn that is not a bare
    # functionResponse (a model functionCall must precede those).
    if len(contents) > self.max_contents:
      contents = contents[-self.max_contents:]
    while contents and (
        contents[0]["role"] != "user"
        or any("functionResponse" in p for p in contents[0]["parts"])
    ):
      contents.pop(0)
    return contents

  # ---- model call ----

  def generate(self) -> dict[str, Any]:
    """Call the model with the current history; append and return its turn.

    Returns ``{"parts": [...], "usage": {...}}``; parts may be empty on
    errors (logged), in which case nothing is appended.
    """
    body: dict[str, Any] = {"contents": self._prepared_contents()}
    if self.system_instruction:
      body["systemInstruction"] = {"parts": [{"text": self.system_instruction}]}
    if self.tools:
      body["tools"] = self.tools
    if self.generation_config:
      body["generationConfig"] = self.generation_config
    data = json.dumps(body).encode("utf-8")
    resp: dict[str, Any] | None = None
    t0 = time.monotonic()
    backend = ""
    for attempt in range(3):
      url, headers, backend = vertex_rest.endpoint(
          self.model, "generateContent", self._api_key
      )
      req = urllib.request.Request(url, data=data, headers=headers)
      try:
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
          resp = json.loads(r.read())
        break
      except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        logger.warning(
            "[%s] generateContent HTTP %s via %s (attempt %d): %s",
            self.model, e.code, backend, attempt + 1, detail,
        )
        if backend == "vertex" and vertex_rest.should_fallback(e):
          vertex_rest.disable(self.model, f"HTTP {e.code}")
          continue
        if e.code not in _RETRY_STATUS:
          break
      except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "[%s] generateContent failed (attempt %d): %s",
            self.model, attempt + 1, e,
        )
      time.sleep(1.5 * (attempt + 1))
    latency = time.monotonic() - t0
    if not resp:
      return {"parts": [], "usage": {}, "latency_s": latency, "error": True}
    cands = resp.get("candidates") or []
    content = (cands[0].get("content") if cands else None) or {}
    parts = [p for p in content.get("parts") or [] if p]
    self.last_usage = resp.get("usageMetadata") or {}
    logger.info(
        "[%s] generateContent (%s) %.1fs, %d parts, finish=%s",
        self.model, backend, latency, len(parts),
        cands[0].get("finishReason") if cands else None,
    )
    if parts:
      self.add("model", parts)
    return {"parts": parts, "usage": self.last_usage, "latency_s": latency}


def live_usage(usage: dict[str, Any]) -> dict[str, Any]:
  """generateContent usageMetadata -> Live API usageMetadata shape."""
  if not usage:
    return {}
  return {
      "promptTokenCount": usage.get("promptTokenCount", 0),
      "responseTokenCount": usage.get("candidatesTokenCount", 0),
      "thoughtsTokenCount": usage.get("thoughtsTokenCount", 0),
      "totalTokenCount": usage.get("totalTokenCount", 0),
      "promptTokensDetails": usage.get("promptTokensDetails", []),
  }
