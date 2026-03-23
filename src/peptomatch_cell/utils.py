"""Utility functions: config loading, logging, data cleaning."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("peptomatch_cell")

_ROOT = Path(__file__).resolve().parents[2]  # project root


def project_root() -> Path:
    return _ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_output_dir(name: str = "outputs") -> Path:
    d = _ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_numeric(val: Any, default: float = 0.0) -> float:
    """Convert various cell values to float."""
    if val is None or val == "":
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.upper() in ("N.D", "N.D.", "ND", "-", "N/A", "NA"):
        return 0.0
    if s.upper().startswith(("<LOQ", "< LOQ", "<0", "< 0")):
        return 0.0
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return default
