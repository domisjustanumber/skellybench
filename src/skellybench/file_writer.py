"""Shared atomic writers for report JSON and HTML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._internal.io import atomic_write_json, atomic_write_text


def write_json_file(path: str | Path, payload: Any) -> Path:
    """Write one JSON report atomically."""
    return atomic_write_json(path, payload)


def write_html_file(path: str | Path, content: str) -> Path:
    """Write one HTML report atomically."""
    return atomic_write_text(path, content)


__all__ = ["write_html_file", "write_json_file"]
