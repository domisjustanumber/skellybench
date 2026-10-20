"""Canonical throughput protocol and derived identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import (
    BENCHMARK_PROTOCOL_VERSION,
    THROUGHPUT_SCENARIO,
    TIMING_SCOPES,
    validate_protocol,
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a mapping with stable key order and separators for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def input_signature_id(signature: Mapping[str, Any]) -> str:
    """Return the canonical hash of an input signature."""
    return hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Adaptive steady-state timing settings."""

    scenario: str = THROUGHPUT_SCENARIO
    version: int = BENCHMARK_PROTOCOL_VERSION
    images: int = 120
    warmup_passes: int = 1
    passes: int = 5
    min_pass_seconds: float = 1.0
    max_pass_seconds: float = 30.0
    cpu_threads: int = 0
    timing_scope: str = "session"

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
        }

    def payload(self, *, signature_id: str | None = None) -> dict[str, Any]:
        """Return the protocol block stored in reports."""
        protocol = self.as_dict()
        if signature_id is not None:
            protocol["input_signature_id"] = signature_id
        protocol["protocol_id"] = self.protocol_id(signature_id=signature_id)
        return protocol

    def protocol_id(self, *, signature_id: str | None = None) -> str:
        """Stable identifier over the protocol, scenario, and optional input signature."""
        payload = {"scenario": self.scenario, **self.as_dict()}
        if signature_id is not None:
            payload["input_signature_id"] = signature_id
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        if self.scenario not in {"throughput_adaptive", "accuracy_full_dataset"}:
            raise ValueError(f"unsupported scenario: {self.scenario!r}")
        if self.timing_scope not in TIMING_SCOPES:
            raise ValueError(f"unsupported timing scope: {self.timing_scope!r}")
        validate_protocol(self.payload())


def protocol_from_mapping(payload: Mapping[str, Any]) -> BenchmarkProtocol:
    """Build and validate a protocol from an input/report mapping."""
    validate_protocol(payload)
    return BenchmarkProtocol(
        scenario=payload.get("scenario", THROUGHPUT_SCENARIO),
        version=int(payload["version"]),
        images=int(payload["images"]),
        warmup_passes=int(payload["warmup_passes"]),
        passes=int(payload["passes"]),
        min_pass_seconds=float(payload["min_pass_seconds"]),
        max_pass_seconds=float(payload["max_pass_seconds"]),
        cpu_threads=int(payload["cpu_threads"]),
        timing_scope=str(payload["timing_scope"]),
    )


def comparison_key(
    *,
    artifact_sha256: str,
    hardware_id: str,
    environment_id: str,
    protocol_id: str,
    provider: str,
    mode: str,
) -> str:
    """Stable key for one comparable performance row."""
    payload = {
        "artifact_sha256": artifact_sha256,
        "hardware_id": hardware_id,
        "environment_id": environment_id,
        "protocol_id": protocol_id,
        "provider": provider,
        "mode": mode,
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "BenchmarkProtocol",
    "canonical_json",
    "comparison_key",
    "input_signature_id",
    "protocol_from_mapping",
]
