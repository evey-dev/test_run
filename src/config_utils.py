"""Utility helpers for reading YAML configuration files and creating output directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file with a friendly fallback to an empty dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory if it does not exist and return it."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
