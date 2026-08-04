from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def require_config_paths(config: dict[str, Any], keys: list[str]) -> None:
    missing: list[str] = []
    for key in keys:
        value: Any = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                missing.append(key)
                break
            value = value[part]
        else:
            if value in (None, "", "TODO"):
                missing.append(key)
    if missing:
        raise ValueError("Missing required config values: " + ", ".join(missing))
