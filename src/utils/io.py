"""I/O helpers for YAML, JSON, Parquet, and directory creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialize an object to a UTF-8 JSON file."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    return out_path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet, ensuring parent directories exist."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)
    return out_path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path)
