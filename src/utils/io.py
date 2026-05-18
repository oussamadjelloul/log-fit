"""IO helpers for configs, manifests, and checkpoints.

Pure utilities — no methodology decisions. All path handling, YAML/JSON
serialization, and SHA256 checksums live here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file and return its dict representation."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: Path | str) -> None:
    """Write a dict to YAML, preserving key order and using readable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, indent=2)


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON file as a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Save data to JSON. Handles dataclasses via asdict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data):
        data = asdict(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for dataclasses and Path objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj).__name__} not JSON serializable")


def sha256_file(path: Path | str) -> str:
    """SHA256 checksum of a file. Used per Component 14 to checksum
    paragraphs.pkl, allocated_indices.json, fold_*.json."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_commit_hash() -> str:
    """Best-effort git commit hash for run provenance.

    Per Component 14: dataset_results.json includes git commit hash of code version.
    Returns 'unknown' if not in a git repo or git unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"
