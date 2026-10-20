"""UTC timestamp helpers used by every report format."""

from __future__ import annotations

from datetime import UTC, datetime

DEFAULT_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""Windows-safe filename stamp: no colons, no timezone offset."""


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def utc_stamp(value: datetime | None = None, *, fmt: str = DEFAULT_STAMP_FORMAT) -> str:
    """Format a UTC timestamp for use in a filename.

    Args:
        value: Timestamp to format. Defaults to now.
        fmt: ``strftime`` format. The default avoids ``:`` because Windows rejects it in
            filenames.
    """
    return (value or utc_now()).strftime(fmt)
