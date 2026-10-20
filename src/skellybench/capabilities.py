"""Machine-readable capabilities for skellybench."""

from __future__ import annotations

from typing import Any

from .schema import (
    BENCHMARK_TYPES,
    COCO_STAT_KEYS,
    MODES,
    PREDICTIONS_FORMAT,
    REPORT_KINDS,
    SCHEMA_VERSION,
    TIMING_SCOPES,
)


def capabilities() -> dict[str, Any]:
    """Return the capability contract as plain data."""
    return {
        "name": "skellybench",
        "schema_version": SCHEMA_VERSION,
        "report_kinds": list(REPORT_KINDS),
        "benchmark_types": list(BENCHMARK_TYPES),
        "performance": {"modes": list(MODES), "timing_scopes": list(TIMING_SCOPES)},
        "predictions": {"format": PREDICTIONS_FORMAT, "encoding": "gzip+base64", "separate_file": True},
        "metrics": {"metric_keys": list(COCO_STAT_KEYS), "tool": "faster-coco-eval"},
        "reports": {
            "json_is_write_once": True,
            "html_is_rebuilt_from_json": True,
            "one_benchmark_file_per_execution_provider": True,
            "run_summary_has_json": False,
        },
        "probes": {"local": "skellybench.probes.local", "docker": "skellybench.probes.docker"},
        "optional_dependencies": {"scoring": "faster-coco-eval"},
    }


def describe() -> str:
    """Return a short human-readable capability summary."""
    caps = capabilities()
    return (
        f"skellybench schema {caps['schema_version']}: "
        f"reports={caps['report_kinds']}; benchmark types={caps['benchmark_types']}; "
        f"modes={caps['performance']['modes']}"
    )


def supports(feature: str) -> bool:
    """Return a capability by dotted path, or ``False`` when it is absent."""
    value: Any = capabilities()
    for part in feature.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


__all__ = ["capabilities", "describe", "supports"]
