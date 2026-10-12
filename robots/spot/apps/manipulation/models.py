from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection2D:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    grasp_px: tuple[float, float]
    normalized_yx: tuple[float, float] | None = None
    angle_rad: float | None = None
    rationale: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any], *, width: int | None = None, height: int | None = None) -> "Detection2D":
        point_y = data.get("y")
        point_x = data.get("x")
        if point_y is None and isinstance(data.get("point"), dict):
            point_y = data["point"].get("y")
            point_x = data["point"].get("x")
        if point_y is not None and point_x is not None:
            if width is None or height is None:
                raise ValueError("width and height are required for normalized y,x detections.")
            normalized_y = float(point_y)
            normalized_x = float(point_x)
            px = normalized_x * (width - 1) / 1000.0
            py = normalized_y * (height - 1) / 1000.0
            return cls(
                label=str(data.get("label", data.get("object", ""))),
                confidence=float(data.get("confidence", 1.0)),
                bbox_xyxy=(px, py, px, py),
                grasp_px=(px, py),
                normalized_yx=(normalized_y, normalized_x),
                angle_rad=None if data.get("angle_rad") is None else float(data["angle_rad"]),
                rationale=str(data.get("rationale", "")),
            )

        bbox = data.get("bbox_xyxy") or data.get("bbox") or data.get("box")
        grasp = data.get("grasp_px") or data.get("center_px") or data.get("point")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError("Detection JSON must include bbox_xyxy with four numbers.")
        if not isinstance(grasp, list | tuple) or len(grasp) != 2:
            raise ValueError("Detection JSON must include grasp_px with two numbers.")
        return cls(
            label=str(data.get("label", "")),
            confidence=float(data.get("confidence", 0.0)),
            bbox_xyxy=tuple(float(v) for v in bbox),
            grasp_px=(float(grasp[0]), float(grasp[1])),
            angle_rad=None if data.get("angle_rad") is None else float(data["angle_rad"]),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class Pose3D:
    frame_name: str
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


@dataclass(frozen=True)
class ImageObservation:
    color_response: Any
    depth_response: Any | None = None

    @property
    def color_bytes(self) -> bytes:
        return self.color_response.shot.image.data

    @property
    def color_mime_type(self) -> str:
        image_format = self.color_response.shot.image.format
        if image_format == self.color_response.shot.image.FORMAT_JPEG:
            return "image/jpeg"
        return "application/octet-stream"

    @property
    def width(self) -> int:
        return self.color_response.shot.image.cols

    @property
    def height(self) -> int:
        return self.color_response.shot.image.rows


@dataclass(frozen=True)
class ForceChange:
    baseline_norm: float
    current_norm: float
    delta_norm: float
    changed: bool
