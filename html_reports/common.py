"""Shared HTML escaping and table formatting."""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import Any


def escape(value: Any) -> str:
    """Escape a value for safe HTML text output."""
    return html.escape("" if value is None else str(value))


def table(caption: str, rows: Iterable[tuple[Any, Any]]) -> str:
    """Render a two-column HTML table."""
    body = "".join(
        f'<tr><th scope="row">{escape(name)}</th><td>{escape(value)}</td></tr>'
        for name, value in rows
    )
    return f"<table><caption>{escape(caption)}</caption><tbody>{body}</tbody></table>"


__all__ = ["escape", "table"]
