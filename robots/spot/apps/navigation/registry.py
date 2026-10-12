from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("keypoints.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Keypoint:
    name: str
    waypoint_id: str
    note: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_record(cls, name: str, record: dict[str, Any]) -> "Keypoint":
        return cls(
            name=name,
            waypoint_id=str(record["waypoint_id"]),
            note=str(record.get("note", "")),
            created_at=str(record.get("created_at", "")),
            updated_at=str(record.get("updated_at", "")),
        )


class KeypointRegistry:
    """JSON-backed mapping from human-friendly place names to GraphNav waypoint IDs."""

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "keypoints": {}}
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data.get("keypoints"), dict):
            raise ValueError(f"Invalid keypoint registry: {self.path}")
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, indent=2, sort_keys=True)
            file.write("\n")

    def all(self) -> list[Keypoint]:
        records = self._data["keypoints"]
        return [Keypoint.from_record(name, records[name]) for name in sorted(records)]

    def get(self, name: str) -> Keypoint:
        records = self._data["keypoints"]
        if name not in records:
            known = ", ".join(sorted(records)) or "none"
            raise KeyError(f"Unknown keypoint '{name}'. Known keypoints: {known}")
        return Keypoint.from_record(name, records[name])

    def register(self, name: str, waypoint_id: str, note: str = "", overwrite: bool = False) -> Keypoint:
        records = self._data["keypoints"]
        if name in records and not overwrite:
            raise ValueError(f"Keypoint '{name}' already exists. Use --overwrite to replace it.")

        timestamp = _now_iso()
        created_at = records.get(name, {}).get("created_at", timestamp)
        records[name] = {
            "waypoint_id": waypoint_id,
            "note": note,
            "created_at": created_at,
            "updated_at": timestamp,
        }
        self.save()
        return self.get(name)

    def remove(self, name: str) -> Keypoint:
        keypoint = self.get(name)
        del self._data["keypoints"][name]
        self.save()
        return keypoint

    def rename(self, old_name: str, new_name: str, overwrite: bool = False) -> Keypoint:
        records = self._data["keypoints"]
        if old_name not in records:
            raise KeyError(f"Unknown keypoint '{old_name}'.")
        if new_name in records and not overwrite:
            raise ValueError(f"Keypoint '{new_name}' already exists. Use --overwrite to replace it.")

        records[new_name] = records.pop(old_name)
        records[new_name]["updated_at"] = _now_iso()
        self.save()
        return self.get(new_name)

    def sync_from_waypoints(self, waypoints: dict[str, str], overwrite: bool = False) -> list[Keypoint]:
        synced: list[Keypoint] = []
        for name, waypoint_id in sorted(waypoints.items()):
            if not name:
                continue
            if name in self._data["keypoints"] and not overwrite:
                continue
            synced.append(self.register(name, waypoint_id, overwrite=True))
        return synced

    def prune_stale_waypoint_ids(self, valid_waypoint_ids: set[str]) -> list[Keypoint]:
        """Remove records that refer to waypoints absent from the live graph."""
        records = self._data["keypoints"]
        stale_names = [
            name
            for name, record in records.items()
            if str(record["waypoint_id"]) not in valid_waypoint_ids
        ]
        removed = [Keypoint.from_record(name, records[name]) for name in sorted(stale_names)]
        for name in stale_names:
            del records[name]
        if stale_names:
            self.save()
        return removed
