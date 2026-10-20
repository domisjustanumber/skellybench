"""Machine-readable capabilities for skellybench."""

from __future__ import annotations

from typing import Any

from .schema import (
    ACCURACY_SCENARIO,
    ACCURACY_STATUSES,
    BENCHMARK_OVERVIEW_KIND,
    BENCHMARK_REPORT_KIND,
    BENCHMARK_SUMMARY_KIND,
    COCO_STAT_KEYS,
    INPUT_BUNDLE_KIND,
    MODES,
    PASS_IDS,
    PASS_ORIGINS,
    PASS_STATUSES,
    PREDICTION_STORAGES,
    PROVIDER_KEYS,
    SCENARIOS,
    SCORING_MODES,
    SKELLYBENCH_SCHEMA_VERSION,
    THROUGHPUT_SCENARIO,
    TIMING_SCOPES,
)
from .model_reports.schema import MODEL_OVERVIEW_KIND, MODEL_REPORT_KIND


def capabilities() -> dict[str, Any]:
    """Return the complete capability contract.

    The output is intentionally data-only so another exporter can inspect it without importing
    skellybench internals. ``README.md`` and the schema constants are tested against this output.
    """
    return {
        "name": "skellybench",
        "schema_version": SKELLYBENCH_SCHEMA_VERSION,
        "report_kinds": [
            BENCHMARK_REPORT_KIND,
            BENCHMARK_SUMMARY_KIND,
            BENCHMARK_OVERVIEW_KIND,
            MODEL_REPORT_KIND,
            MODEL_OVERVIEW_KIND,
        ],
        "accuracy": {
            "passes": list(PASS_IDS),
            "pass_origins": list(PASS_ORIGINS),
            "scenarios": [ACCURACY_SCENARIO],
            "statuses": list(ACCURACY_STATUSES),
            "pass_statuses": list(PASS_STATUSES),
            "scoring_modes": list(SCORING_MODES),
            "scoring_requires_annotation": True,
            "supports_precomputed_metrics": True,
            "supports_precomputed_comparisons": True,
            "requires_internal_device_match": True,
            "writes_accuracy_metrics_to_benchmark_reports": False,
        },
        "performance": {
            "scenarios": list(SCENARIOS),
            "providers": list(PROVIDER_KEYS),
            "modes": list(MODES),
            "timing_scopes": list(TIMING_SCOPES),
            "throughput_is_images_per_second": True,
        },
        "predictions": {
            "storages": list(PREDICTION_STORAGES),
            "format": "coco_detection_results",
        },
        "environment": {
            "hardware_fields": ["id", "hardware_id", "os", "cpu", "gpus"],
            "library_fields": ["python.implementation", "python.version", "packages"],
            "package_key_format": "PEP 503 normalized distribution name",
        },
        "metrics": {
            "metric_keys": list(COCO_STAT_KEYS),
            "tool": "faster_coco_eval",
        },
        "reports": {
            "writes_benchmark_reports": True,
            "writes_model_reports": True,
            "input_kind": INPUT_BUNDLE_KIND,
            "embeds_raw_predictions": False,
            "throughput_only_benchmark_reports": True,
        },
        "optional_dependencies": {"scoring": "faster-coco-eval"},
        "scenarios": {
            ACCURACY_SCENARIO: "full-dataset apples-to-apples Python/ONNX accuracy run",
            THROUGHPUT_SCENARIO: "adaptive warmup/timed-pass throughput benchmark",
        },
    }


def describe() -> str:
    """Return a short human-readable capability summary."""
    caps = capabilities()
    return (
        "skellybench: model and benchmark reports; "
        f"accuracy passes={caps['accuracy']['passes']}; "
        f"providers={caps['performance']['providers']}; "
        f"modes={caps['performance']['modes']}; "
        f"scoring requires annotation={caps['accuracy']['scoring_requires_annotation']}; "
        f"precomputed metrics={caps['accuracy']['supports_precomputed_metrics']}"
    )


def supports(feature: str) -> bool:
    """Return a capability by dotted path, or ``False`` when it is absent.

    Boolean leaves are returned directly. A non-boolean leaf (for example a list) is considered
    supported when it is non-empty.
    """
    value: Any = capabilities()
    for part in feature.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


__all__ = ["capabilities", "describe", "supports"]
