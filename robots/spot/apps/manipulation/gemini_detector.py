from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from apps.manipulation.models import Detection2D


DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"


@dataclass(frozen=True)
class DetectionResult:
    detection: Detection2D
    prompt: str
    raw_response: str
    parsed_json: dict[str, Any]


class GeminiObjectDetector:
    """Language-conditioned object detector using Gemini Robotics ER style JSON output."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def detect(
        self,
        *,
        instruction: str,
        image_bytes: bytes,
        mime_type: str,
        width: int,
        height: int,
    ) -> Detection2D:
        return self.detect_with_result(
            instruction=instruction,
            image_bytes=image_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
        ).detection

    def detect_with_result(
        self,
        *,
        instruction: str,
        image_bytes: bytes,
        mime_type: str,
        width: int,
        height: int,
    ) -> DetectionResult:
        prompt = f"""
Point to {instruction}.

Output JSON only.
Return the target point as normalized image coordinates from 0 to 1000.
Use y first, then x.

Required schema:
{{"y": number, "x": number, "label": string, "confidence": number}}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw_response = response.text or ""
        parsed_json = _parse_json_object(raw_response)
        detection = Detection2D.from_json(parsed_json, width=width, height=height)
        return DetectionResult(
            detection=detection,
            prompt=prompt.strip(),
            raw_response=raw_response,
            parsed_json=parsed_json,
        )


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Gemini response did not contain a JSON object: {text!r}")
    return json.loads(match.group(0))
