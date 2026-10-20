"""Human-readable formatting helpers shared by HTML renderers."""

from __future__ import annotations

from typing import Any


def format_filesize(size_bytes: int | None) -> str:
    """Return a compact byte count such as ``1.4 MB``."""
    if size_bytes is None:
        return "n/a"
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def format_number(value: Any, digits: int = 2) -> str:
    """Format a numeric value, returning ``n/a`` for missing or non-numeric input."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
