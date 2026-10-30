"""Vertex-first routing for non-Live (REST generateContent) Gemini calls.

User preference: use the Gemini Enterprise Agent Platform (Vertex AI API,
aiplatform.googleapis.com) wherever the model is available there, and the
Gemini Developer API (AI Studio, API key) only as a fallback.

Verified on Gemini Enterprise Agent Platform / Vertex AI (project
elated-terminal-503102-u0) - all on the ``global`` location:
  gemini-robotics-er-2-preview (Public Preview on Gemini Platform),
  gemini-3.8-flash-tts, gemini-3.8-flash-lite-tts, gemini-4-argon,
  gemini-3.5-flash-lite, gemini-3.8-flash, gemini-3.5-transcribe.

Auth is ADC (Cloud Run service account, needs roles/aiplatform.user). If a
Vertex call fails with an auth / not-found style error, that model is
switched to AI Studio for the rest of the process (logged once).
Escape hatch: LIVE_BACKEND=aistudio routes everything to AI Studio.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.error

logger = logging.getLogger(__name__)

VERTEX_REST_MODELS = frozenset({
    "gemini-robotics-er-2-preview",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-4-argon",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-transcribe",
})

_AISTUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
# Auth / not-found style errors only: a 400 is a request problem that AI
# Studio would reject too (and a mid-conversation switch breaks thought
# signatures: "Corrupted thought signature").
_FALLBACK_CODES = (401, 403, 404)

_lock = threading.Lock()
_creds = None
_project: str | None = None
_disabled: set[str] = set()


def _vertex_project() -> str | None:
  return os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _location() -> str:
  return os.environ.get("VERTEX_GENAI_LOCATION", "global")


def uses_vertex(model: str) -> bool:
  name = model.removeprefix("models/")
  if os.environ.get("LIVE_BACKEND", "").lower() == "aistudio":
    return False
  return name in VERTEX_REST_MODELS and name not in _disabled and bool(_vertex_project())


def _token() -> str:
  global _creds, _project
  import google.auth  # pylint: disable=g-import-not-at-top
  import google.auth.transport.requests  # pylint: disable=g-import-not-at-top
  with _lock:
    if _creds is None:
      _creds, _project = google.auth.default(scopes=_SCOPES)
    if not _creds.valid:
      _creds.refresh(google.auth.transport.requests.Request())
    return _creds.token


def endpoint(
    model: str, method: str, api_key: str | None
) -> tuple[str, dict[str, str], str]:
  """(url, headers, backend) for ``model:method`` (method may carry ?alt=sse)."""
  name = model.removeprefix("models/")
  if uses_vertex(name):
    try:
      loc = _location()
      host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
      url = (
          f"https://{host}/v1/projects/{_vertex_project()}/locations/{loc}"
          f"/publishers/google/models/{name}:{method}"
      )
      return url, {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {_token()}",
      }, "vertex"
    except Exception as e:  # pylint: disable=broad-except
      disable(name, f"ADC unavailable: {e}")
  if not api_key:
    raise ValueError(f"No Vertex access and no API key for {name}")
  return f"{_AISTUDIO_BASE}{name}:{method}", {
      "Content-Type": "application/json",
      "x-goog-api-key": api_key,
  }, "aistudio"


def disable(model: str, reason: str) -> None:
  name = model.removeprefix("models/")
  if name not in _disabled:
    _disabled.add(name)
    logger.warning("Vertex disabled for %s -> AI Studio fallback (%s)", name, reason)


def should_fallback(err: Exception) -> bool:
  return isinstance(err, urllib.error.HTTPError) and err.code in _FALLBACK_CODES
