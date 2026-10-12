"""Manipulation APIs for Spot arm and gripper workflows."""

from apps.manipulation.gemini_detector import GeminiObjectDetector
from apps.manipulation.models import Detection2D, ForceChange, ImageObservation, Pose3D
from apps.manipulation.spot_client import SpotManipulationClient

__all__ = [
    "Detection2D",
    "ForceChange",
    "GeminiObjectDetector",
    "ImageObservation",
    "Pose3D",
    "SpotManipulationClient",
]
