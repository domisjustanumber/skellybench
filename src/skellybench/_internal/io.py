"""Atomic and write-once JSON/text I/O.

HTML pages are rewritten often, so text writes go through a temporary file and :func:`os.replace`.
JSON reports are immutable, so :func:`write_json_once` never replaces an existing file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _temporary_sibling(destination: Path) -> tuple[int, Path]:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    return descriptor, Path(temp_name)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write *text* to *path* atomically, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = _temporary_sibling(destination)
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
    indent: int | None = 2,
    sort_keys: bool = False,
) -> Path:
    """Serialize *payload* as JSON and write it atomically. Replaces an existing file."""
    return atomic_write_text(path, json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n")


def _exclusive_write(destination: Path, text: str) -> None:
    with open(destination, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_text_once(path: str | Path, text: str) -> Path:
    """Write *text* to a new file. Raise :class:`FileExistsError` when *path* exists.

    The text goes to a temporary sibling first, then :func:`os.link` publishes it under the target
    name. ``os.link`` fails when the target exists, so the publish step is atomic and write-once.
    Some file systems (FAT/exFAT, some network shares and container bind mounts) do not support hard
    links. On those, the function falls back to an exclusive create. That is still write-once but not
    atomic: a crash can leave a partial file, which fails JSON parsing, so readers treat it as absent.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {destination}")
    descriptor, temp_path = _temporary_sibling(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite immutable file: {destination}") from None
        except OSError:
            try:
                _exclusive_write(destination, text)
            except FileExistsError:
                raise FileExistsError(
                    f"refusing to overwrite immutable file: {destination}"
                ) from None
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def write_json_once(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Serialize *payload* and write it with :func:`write_text_once`."""
    return write_text_once(path, json.dumps(payload, indent=indent) + "\n")


def read_json(path: str | Path) -> Any | None:
    """Read JSON from *path*, returning ``None`` when it is missing or unparseable.

    A damaged or partial file is treated as absent rather than trusted.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
