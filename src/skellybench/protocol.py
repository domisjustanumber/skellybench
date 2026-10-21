"""Benchmark protocol blocks and their stable identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schema import ACCURACY, PERFORMANCE, TIMING_SCOPES


def canonical_json(payload: Any) -> str:
    """Serialize a value with stable key order and separators for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fingerprint(payload: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of *payload*."""
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def protocol_block(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return *params* plus a ``protocol_id``: the first 16 hex characters of their SHA-256.

    *params* must hold ``benchmark_type`` and ``timing_scope``. Two benchmarks with the same
    ``protocol_id`` used the same timing and evaluation settings.
    """
    block = {key: value for key, value in params.items() if key != "protocol_id"}
    if block.get("benchmark_type") not in (ACCURACY, PERFORMANCE):
        raise ValueError("protocol benchmark_type must be 'accuracy' or 'performance'")
    if block.get("timing_scope") not in TIMING_SCOPES:
        raise ValueError(f"protocol timing_scope must be one of {list(TIMING_SCOPES)}")
    block["protocol_id"] = hashlib.sha256(canonical_json(block).encode("utf-8")).hexdigest()[:16]
    return block


@dataclass(frozen=True)
class PerformanceProtocol:
    """Adaptive steady-state timing settings for a performance benchmark."""

    images: int = 120
    warmup_passes: int = 1
    passes: int = 5
    min_pass_seconds: float = 1.0
    max_pass_seconds: float = 30.0
    cpu_threads: int = 0
    timing_scope: str = "session"
    excluded_stages: tuple[str, ...] = ()
    """Pipeline stages that the timing leaves out, for example ``("letterbox",)``."""
    version: int = 1

    def __post_init__(self) -> None:
        if self.images < 1:
            raise ValueError("images must be >= 1")
        if self.warmup_passes < 0 or self.passes < 1:
            raise ValueError("warmup_passes must be >= 0 and passes must be >= 1")
        if self.min_pass_seconds < 0 or self.max_pass_seconds < self.min_pass_seconds:
            raise ValueError("pass durations are invalid")
        if self.timing_scope not in TIMING_SCOPES:
            raise ValueError(f"timing_scope must be one of {list(TIMING_SCOPES)}")
        if isinstance(self.excluded_stages, str):
            raise ValueError("excluded_stages must be a sequence of stage names, not one string")
        if any(not isinstance(stage, str) or not stage for stage in self.excluded_stages):
            raise ValueError("excluded_stages must hold non-empty strings")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "images": self.images,
            "warmup_passes": self.warmup_passes,
            "passes": self.passes,
            "min_pass_seconds": self.min_pass_seconds,
            "max_pass_seconds": self.max_pass_seconds,
            "cpu_threads": self.cpu_threads,
            "timing_scope": self.timing_scope,
            "excluded_stages": list(self.excluded_stages),
        }

    def block(self, *, batch_size: int) -> dict[str, Any]:
        """Return the report ``protocol`` block for one batch size."""
        return protocol_block({"benchmark_type": PERFORMANCE, "batch_size": batch_size, **self.as_dict()})


def accuracy_protocol(*, batch_size: int, images: int, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the report ``protocol`` block for an accuracy benchmark."""
    return protocol_block(
        {
            "benchmark_type": ACCURACY,
            "version": 1,
            "batch_size": batch_size,
            "images": images,
            "timing_scope": "end_to_end",
            "evaluation": dict(evaluation),
        }
    )


__all__ = [
    "PerformanceProtocol",
    "accuracy_protocol",
    "canonical_json",
    "fingerprint",
    "protocol_block",
]
