"""Configuration loading."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config; env vars override MLflow settings."""
    cfg_path = Path(path or os.environ.get("TRAJCLF_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = uri
    return cfg
