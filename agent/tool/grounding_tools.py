"""Information tools shared by every agent / model: location, web and maps.

Three function tools are declared here and executed server-side by
``ToolCallHandler`` (they are embodiment-agnostic):

* ``get_current_location`` - the browser's HTML5 Geolocation fix (GPS / Wi-Fi /
  IP based, whatever the device provides) that the UI reports over the
  WebSocket, enriched with a reverse-geocoded place description via Google
  Maps grounding when available.
* ``google_search`` - Google Search grounding (fresh web facts: weather, news,
  prices, opening hours, how-to questions ...).
* ``google_maps_search`` - Google Maps grounding (nearby places, directions,
  opening hours, ratings) anchored on the user's current location.

Grounding is executed through the ``generateContent`` REST API with a fast
Flash-Lite model, because the Live API session itself cannot combine custom
function tools with built-in grounding tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

GROUNDING_TOOL_NAMES = frozenset({
    "get_current_location",
    "google_search",
    "google_maps_search",
})

_DEFAULT_GROUNDING_MODEL = "gemini-3.5-flash-lite"
_REQUEST_TIMEOUT_S = 15


# ---------------------------------------------------------------------------
# Declarations (Gemini FunctionDeclaration JSON)
# ---------------------------------------------------------------------------


def _decl(
    name: str, description: str, properties: dict, required: list[str]
) -> dict:
  return {
      "name": name,
      "description": description,
      "parameters": {
          "type": "OBJECT",
          "properties": properties,
          "required": required,
      },
      "behavior": "BLOCKING",
  }


def get_current_location_tool() -> dict[str, Any]:
  return _decl(
      "get_current_location",
      "Get the user's / robot's CURRENT geographic location (latitude,"
      " longitude, accuracy) from the device GPS / browser geolocation, plus a"
      " human-readable place description (city, district, nearby landmark)."
      " Call this when the user asks where they are, what city / country this"
      " is, the local time zone, or before any 'nearby' / 'around here' maps"
      " query. Returns an error if the user has not granted location"
      " permission in the browser.",
      {},
      [],
  )


def google_search_tool() -> dict[str, Any]:
  return _decl(
      "google_search",
      "Search the web with Google Search grounding and return a concise,"
      " up-to-date, cited answer. Use it for anything that needs fresh or"
      " factual world knowledge you cannot see in the camera: weather, news,"
      " sports scores, prices, product specs, definitions, 'how to' facts,"
      " dates, people, companies, etc. Ask the question in the user's own"
      " language; the answer comes back in that language.",
      {
          "query": {
              "type": "STRING",
              "description": (
                  "The full natural-language question to answer, e.g."
                  " '今天上海的天气怎么样' or 'latest Boston Dynamics Spot"
                  " price'."
              ),
          },
      },
      ["query"],
  )


def google_maps_search_tool() -> dict[str, Any]:
  return _decl(
      "google_maps_search",
      "Search Google Maps (grounded) for places, businesses, addresses,"
      " directions, distances, travel times, opening hours and ratings."
      " Automatically anchored on the user's current location when it is"
      " known, so 'nearby' / '附近' queries work. Use it for questions like"
      " '附近有什么好吃的餐厅', 'how far is the nearest pharmacy', 'is the"
      " Apple Store open now', 'directions to Shanghai Hongqiao station'.",
      {
          "query": {
              "type": "STRING",
              "description": (
                  "Natural-language maps question in the user's language, e.g."
                  " '附近评价最好的咖啡馆' or 'nearest EV charging station'."
              ),
          },
          "latitude": {
              "type": "NUMBER",
              "description": (
                  "Optional latitude to anchor the search on (defaults to the"
                  " device's current location)."
              ),
          },
          "longitude": {
              "type": "NUMBER",
              "description": (
                  "Optional longitude to anchor the search on (defaults to the"
                  " device's current location)."
              ),
          },
      },
      ["query"],
  )


def grounding_tool_declarations() -> list[dict[str, Any]]:
  """All shared information tools, in a stable order."""
  return [
      get_current_location_tool(),
      google_search_tool(),
      google_maps_search_tool(),
  ]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _language_hint(user_language: str | None) -> str:
  if user_language:
    return (
        f" Answer in {user_language}; keep it concise (3-5 short sentences or"
        " a short list) and suitable for being read aloud."
    )
  return (
      " Answer in the same language as the question; keep it concise (3-5"
      " short sentences or a short list) and suitable for being read aloud."
  )


def _generate(
    api_key: str,
    prompt: str,
    tool: dict[str, Any],
    lat_lng: tuple[float, float] | None = None,
    model: str = _DEFAULT_GROUNDING_MODEL,
) -> dict[str, Any]:
  """Call generateContent with one built-in grounding tool."""
  payload: dict[str, Any] = {
      "contents": [{"parts": [{"text": prompt}]}],
      "tools": [tool],
      "generationConfig": {"temperature": 0.2},
  }
  if lat_lng is not None:
    payload["toolConfig"] = {
        "retrievalConfig": {
            "latLng": {"latitude": lat_lng[0], "longitude": lat_lng[1]}
        }
    }
  url = (
      "https://generativelanguage.googleapis.com/v1beta/models/"
      f"{model}:generateContent"
  )
  req = urllib.request.Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      # API key in a header, never in the URL (keeps it out of logs / proxies).
      headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
      method="POST",
  )
  t0 = time.monotonic()
  with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
    data = json.loads(resp.read().decode("utf-8"))
  cand = (data.get("candidates") or [{}])[0]
  parts = cand.get("content", {}).get("parts", [])
  text = "".join(p.get("text", "") for p in parts).strip()
  # Strip markdown emphasis so TTS / native speech reads it naturally.
  text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
  text = re.sub(r"^\s*[\*\-]\s+", "- ", text, flags=re.MULTILINE)
  meta = cand.get("groundingMetadata", {}) or {}
  sources: list[dict[str, str]] = []
  for chunk in meta.get("groundingChunks", []) or []:
    web = chunk.get("web") or {}
    maps = chunk.get("maps") or {}
    if web.get("uri"):
      sources.append({"title": web.get("title", ""), "uri": web["uri"]})
    elif maps.get("uri") or maps.get("title"):
      sources.append({
          "title": maps.get("title", ""),
          "uri": maps.get("uri", ""),
          "place_id": maps.get("placeId", ""),
      })
  return {
      "answer": text,
      "sources": sources[:6],
      "search_queries": (meta.get("webSearchQueries") or [])[:4],
      "latency_ms": int((time.monotonic() - t0) * 1000),
  }


def _coerce_float(value: Any) -> float | None:
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f


def run_grounding_tool(
    func_name: str,
    args: dict[str, Any],
    api_key: str | None,
    geolocation: dict[str, Any] | None,
    user_language: str | None = None,
) -> dict[str, Any]:
  """Execute one of the shared information tools (blocking; run in a thread)."""
  key = api_key or os.getenv("GEMINI_API_KEY")
  if not key:
    return {"error": "GEMINI_API_KEY is not configured on the server."}

  geo = geolocation or {}
  lat = _coerce_float(geo.get("latitude"))
  lng = _coerce_float(geo.get("longitude"))
  fix = (lat, lng) if lat is not None and lng is not None else None

  try:
    if func_name == "get_current_location":
      if fix is None:
        return {
            "available": False,
            "error": (
                "Device location is not available: the browser has not"
                " reported a geolocation fix (location permission denied,"
                " still acquiring, or unsupported)."
            ),
            "next_action": (
                "Tell the user (in their language) to allow location access"
                " in the browser, or ask them which city they are in."
            ),
        }
      ts = _coerce_float(geo.get("timestamp"))
      result: dict[str, Any] = {
          "available": True,
          "latitude": round(fix[0], 6),
          "longitude": round(fix[1], 6),
          "accuracy_m": _coerce_float(geo.get("accuracy")),
          "source": geo.get("source", "browser_geolocation"),
          "fix_age_s": round(time.time() - ts, 1) if ts else None,
          "google_maps_url": (
              f"https://www.google.com/maps?q={fix[0]:.6f},{fix[1]:.6f}"
          ),
      }
      try:
        place = _generate(
            key,
            "Describe the place at these coordinates for a user who asked"
            f" 'where am I?': latitude {fix[0]:.6f}, longitude {fix[1]:.6f}."
            " Give the nearest street/landmark, district, city, region and"
            " country, and the local time zone, in at most 2 short sentences."
            + _language_hint(user_language),
            {"google_maps": {}},
            lat_lng=fix,
        )
        result["place_description"] = place["answer"]
        if place["sources"]:
          result["sources"] = place["sources"][:3]
      except Exception as e:  # pylint: disable=broad-except
        logger.warning("Reverse geocoding via Maps grounding failed: %s", e)
        result["place_description"] = ""
      return result

    query = str(args.get("query") or "").strip()
    if not query:
      return {"error": "query is required"}

    if func_name == "google_search":
      out = _generate(
          key, query + _language_hint(user_language), {"google_search": {}}
      )
      out["query"] = query
      return out

    if func_name == "google_maps_search":
      lat_arg = _coerce_float(args.get("latitude"))
      lng_arg = _coerce_float(args.get("longitude"))
      anchor = (
          (lat_arg, lng_arg)
          if lat_arg is not None and lng_arg is not None
          else fix
      )
      prompt = query
      if anchor is not None:
        prompt += (
            f" (The user is currently at latitude {anchor[0]:.5f}, longitude"
            f" {anchor[1]:.5f}; interpret 'nearby' relative to that point and"
            " mention approximate distances.)"
        )
      else:
        prompt += (
            " (The user's exact location is unknown; if the question needs a"
            " location, say so briefly and ask which area to search.)"
        )
      out = _generate(
          key,
          prompt + _language_hint(user_language),
          {"google_maps": {}},
          lat_lng=anchor,
      )
      out["query"] = query
      out["anchored_on_device_location"] = anchor is not None and anchor == fix
      return out

    return {"error": f"Unknown information tool: {func_name}"}
  except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="replace")[:300]
    logger.error("%s HTTP %s: %s", func_name, e.code, detail)
    return {
        "error": f"{func_name} failed with HTTP {e.code}",
        "detail": detail,
        "next_action": (
            "Tell the user briefly that the lookup failed and offer to retry."
        ),
    }
  except Exception as e:  # pylint: disable=broad-except
    logger.exception("%s failed", func_name)
    return {"error": f"{func_name} failed: {e}"}
