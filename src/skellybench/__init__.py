"""skellybench: model and benchmark reports, COCO scoring, and environment probes.

Schema version 3. Producers build plain dictionaries with the helpers in :mod:`skellybench.blocks`
and :mod:`skellybench.reports`. JSON reports are written once. HTML pages are rendered from the JSON
by :mod:`skellybench.render` and can be rebuilt at any time.
"""

from __future__ import annotations

from . import blocks, paths, predictions, probes, render, reports
from ._internal.io import read_json, write_json_once, write_text_once
from .capabilities import capabilities, describe, supports
from .paths import (
    BenchmarkPaths,
    ModelPaths,
    benchmark_paths,
    benchmark_report_id,
    ep_slug,
    hostname_slug,
    make_run_id,
    model_paths,
    run_summary_path,
)
from .predictions import PredictionColumns, columns_from_records, encode_columns, read_predictions_file
from .protocol import PerformanceProtocol, accuracy_protocol, canonical_json, fingerprint, protocol_block
from .reports import (
    FamilyIndex,
    accuracy_entry,
    build_benchmark_report,
    build_model_report,
    read_model_report,
    report_id_for,
    scan_family,
    supersede_model_report,
    write_benchmark_predictions,
    write_benchmark_report,
    write_model_report,
)
from .schema import (
    ACCURACY,
    BENCHMARK_REPORT_KIND,
    COCO_STAT_KEYS,
    MODEL_REPORT_KIND,
    MODES,
    PERFORMANCE,
    PREDICTIONS_KIND,
    SCHEMA_VERSION,
    SkellybenchValidationError,
    validate_benchmark_report,
    validate_model_report,
)
from .scoring import (
    DEFAULT_FAIL_AP,
    DEFAULT_WARN_AP,
    ScoringError,
    ScoringResult,
    ScoringUnavailable,
    compare_metrics,
    score_columns,
    score_predictions,
)

__all__ = [
    "ACCURACY",
    "BENCHMARK_REPORT_KIND",
    "COCO_STAT_KEYS",
    "DEFAULT_FAIL_AP",
    "DEFAULT_WARN_AP",
    "MODEL_REPORT_KIND",
    "MODES",
    "PERFORMANCE",
    "PREDICTIONS_KIND",
    "SCHEMA_VERSION",
    "BenchmarkPaths",
    "FamilyIndex",
    "ModelPaths",
    "PerformanceProtocol",
    "PredictionColumns",
    "ScoringError",
    "ScoringResult",
    "ScoringUnavailable",
    "SkellybenchValidationError",
    "accuracy_entry",
    "accuracy_protocol",
    "benchmark_paths",
    "benchmark_report_id",
    "blocks",
    "build_benchmark_report",
    "build_model_report",
    "canonical_json",
    "capabilities",
    "columns_from_records",
    "compare_metrics",
    "describe",
    "encode_columns",
    "ep_slug",
    "fingerprint",
    "hostname_slug",
    "make_run_id",
    "model_paths",
    "paths",
    "predictions",
    "probes",
    "protocol_block",
    "read_json",
    "read_model_report",
    "read_predictions_file",
    "render",
    "report_id_for",
    "reports",
    "run_summary_path",
    "scan_family",
    "score_columns",
    "score_predictions",
    "supersede_model_report",
    "supports",
    "validate_benchmark_report",
    "validate_model_report",
    "write_benchmark_predictions",
    "write_benchmark_report",
    "write_json_once",
    "write_model_report",
    "write_text_once",
]
