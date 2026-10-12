from __future__ import annotations

import os
from pathlib import Path


CONFIG_PATHS = (
    Path(".config"),
    Path(__file__).resolve().parents[2] / ".config",
)


def load_dot_config() -> dict[str, str]:
    """Load simple KEY=VALUE config files without overriding explicit env vars."""
    values: dict[str, str] = {}
    for path in CONFIG_PATHS:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            values[key] = value
            os.environ.setdefault(key, value)
    return values
