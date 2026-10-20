"""Atomic JSON/text I/O.

Reports are frequently rewritten by resumed or benchmark-only runs. A partially written JSON file
must never be mistaken for a complete report, so every write goes through a temporary file in the
destination directory followed by :func:`os.replace`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write *text* to *path* atomically, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Serialize *payload* as JSON and write it atomically."""
    return atomic_write_text(
        path,
        json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n",
    )


def read_json(path: str | Path) -> Any | None:
    """Read JSON from *path*, returning ``None`` when it is missing or unparseable.

    Report completeness checks treat an unreadable file as absent so a damaged report is rebuilt
    rather than trusted.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
