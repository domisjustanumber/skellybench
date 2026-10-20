"""Stable hashes and file descriptions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .formatting import format_filesize

_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the lowercase hex SHA-256 digest of UTF-8 *text*."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Describe a file without exposing its local path.

    Report payloads intentionally store a filename plus hash rather than an absolute path, so a
    report can move between machines without carrying local filesystem details.
    """
    file_path = Path(path)
    stat = file_path.stat()
    description: dict[str, Any] = {
        "filename": file_path.name,
        "size_bytes": stat.st_size,
        "filesize_human": format_filesize(stat.st_size),
    }
    if include_hash:
        description["sha256"] = sha256_file(file_path)
    return description
