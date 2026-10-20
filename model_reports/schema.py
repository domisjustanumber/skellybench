"""Schema, builders, validation, and merging for model reports.

A model report is the provenance-and-scoring document for one model artifact. It is deliberately
independent of RF-DETR, ONNX Runtime, and the benchmark implementation: callers supply plain
dictionaries and this module enforces the shared contract.

The schema is intentionally strict about the accuracy section. An apples-to-apples accuracy section
must contain exactly two passes, ``python`` and ``onnx``, on the same device, and that device must
record that it matches the source library's internal benchmark selection.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from report_core import iso_now

from ..hardware import DEFAULT_SOFTWARE_PACKAGES, detect_hardware, detect_libraries
from ..schema import (
    SKELLYBENCH_SCHEMA_VERSION,
    SkellybenchValidationError,
    validate_hardware,
    validate_libraries,
)

MODEL_REPORT_SCHEMA_VERSION = SKELLYBENCH_SCHEMA_VERSION
_LEGACY_MODEL_REPORT_SCHEMA_VERSION = 1
MODEL_REPORT_KIND = "model_report"
MODEL_OVERVIEW_KIND = "model_report_overview"

ACCURACY_SCENARIO = "accuracy_full_dataset"
THROUGHPUT_SCENARIO = "throughput_adaptive"
SCENARIOS = frozenset({ACCURACY_SCENARIO, THROUGHPUT_SCENARIO})

ACCURACY_STATUSES = frozenset({"measured", "unavailable", "skipped", "error"})
PASS_STATUSES = frozenset({"measured", "unavailable", "skipped", "error"})
PASS_IDS = ("python", "onnx")
PASS_ORIGINS = frozenset({"python_model", "onnx_model", "other"})
RUNTIMES = frozenset({"onnx", "rfdetr", "other"})
COMPARISON_STATUSES = frozenset({"ok", "warn", "fail", "not_available"})

REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_ID_PATTERN = REPORT_ID_PATTERN
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")

_LOCAL_PATH_KEYS = frozenset({"path", "local_path", "cache_path", "checkpoint_path"})
_RAW_PREDICTION_KEYS = frozenset(
    {"predictions", "raw_predictions", "prediction_records", "coco_predictions"}
)


class ModelReportValidationError(ValueError):
    """Raised when a model report does not satisfy the schema."""


@dataclass(frozen=True)
class ModelReportIdentity:
    """Fields used to decide whether an existing report is for the model being processed."""

    report_id: str
    model_id: str | None = None
    runtime: str | None = None
    precision: str | None = None
    batch_size: int | None = None
    artifact_sha256: str | None = None
    source_fp32_sha256: str | None = None
    source_md5: str | None = None


def _error(path: str, message: str) -> ModelReportValidationError:
    return ModelReportValidationError(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _require_non_empty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "must be a non-empty string")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "must be a boolean")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _error(path, f"must be an integer >= {minimum}")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_number(value: Any, path: str) -> float:
    if not _is_number(value):
        raise _error(path, "must be a finite number")
    return float(value)


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(path, "must be an array")
    return value


def _require_hex(value: Any, path: str, pattern: re.Pattern[str], label: str) -> str:
    text = _require_non_empty_str(value, path)
    if not pattern.fullmatch(text):
        raise _error(path, f"must be a lowercase hex {label}")
    return text


def _require_filename(value: Any, path: str) -> str:
    name = _require_non_empty_str(value, path)
    if Path(name).name != name:
        raise _error(path, "must be a filename, not a path")
    return name


def _require_timestamp(value: Any, path: str) -> str:
    text = _require_non_empty_str(value, path)
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise _error(path, "must be an ISO-8601 timestamp") from exc
    return text


def _reject_raw_predictions(value: Any, path: str = "snapshot") -> None:
    """Reject embedded raw prediction stores anywhere in a benchmark snapshot."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in _RAW_PREDICTION_KEYS:
                raise _error(child_path, "raw predictions must not be embedded in model reports")
            _reject_raw_predictions(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_raw_predictions(child, f"{path}[{index}]")


def _validate_source(source: Any, path: str) -> None:
    if source is None:
        return
    source_map = _require_mapping(source, path)
    for key in _LOCAL_PATH_KEYS:
        if key in source_map:
            raise _error(
                f"{path}.{key}",
                "local checkpoint paths are not provenance; use the upstream URL and md5",
            )
    if source_map.get("kind") == "checkpoint":
        _require_non_empty_str(source_map.get("url"), f"{path}.url")
        _require_hex(source_map.get("md5"), f"{path}.md5", MD5_PATTERN, "md5")


def _validate_model(model: Any) -> None:
    model_map = _require_mapping(model, "model")
    _require_non_empty_str(model_map.get("model_id"), "model.model_id")
    runtime = _require_non_empty_str(model_map.get("runtime"), "model.runtime")
    if runtime not in RUNTIMES:
        raise _error("model.runtime", f"must be one of {sorted(RUNTIMES)}")
    _require_non_empty_str(model_map.get("precision"), "model.precision")
    _require_int(model_map.get("batch_size"), "model.batch_size", minimum=1)
    _require_mapping(model_map.get("input_signature"), "model.input_signature")
    if "metadata" in model_map:
        _require_mapping(model_map["metadata"], "model.metadata")

    artifact = model_map.get("artifact")
    if artifact is not None:
        artifact_map = _require_mapping(artifact, "model.artifact")
        _require_filename(artifact_map.get("filename"), "model.artifact.filename")
        if "sha256" in artifact_map:
            _require_hex(artifact_map["sha256"], "model.artifact.sha256", SHA256_PATTERN, "sha256")
        if "source_fp32_sha256" in artifact_map:
            _require_hex(
                artifact_map["source_fp32_sha256"],
                "model.artifact.source_fp32_sha256",
                SHA256_PATTERN,
                "sha256",
            )
        if "size_bytes" in artifact_map:
            _require_int(artifact_map["size_bytes"], "model.artifact.size_bytes")
        if "filesize_human" in artifact_map:
            _require_non_empty_str(artifact_map["filesize_human"], "model.artifact.filesize_human")


def _validate_provenance(provenance: Any) -> None:
    provenance_map = _require_mapping(provenance, "provenance")
    origin = _require_non_empty_str(provenance_map.get("origin"), "provenance.origin")
    if origin not in PASS_ORIGINS:
        raise _error("provenance.origin", f"must be one of {sorted(PASS_ORIGINS)}")
    if "tools" in provenance_map:
        _require_mapping(provenance_map["tools"], "provenance.tools")
    if "config" in provenance_map:
        _require_mapping(provenance_map["config"], "provenance.config")
    if "environment" in provenance_map:
        _require_mapping(provenance_map["environment"], "provenance.environment")
    _validate_source(provenance_map.get("source"), "provenance.source")


def _validate_device(device: Any) -> str:
    device_map = _require_mapping(device, "accuracy.device")
    device_type = _require_non_empty_str(device_map.get("type"), "accuracy.device.type")
    if device_type not in {"cuda", "cpu"}:
        raise _error("accuracy.device.type", "must be 'cuda' or 'cpu'")
    if device_type == "cuda":
        _require_int(device_map.get("index"), "accuracy.device.index", minimum=0)
    torch_device = _require_non_empty_str(
        device_map.get("torch_device"), "accuracy.device.torch_device"
    )
    _require_non_empty_str(device_map.get("ort_provider"), "accuracy.device.ort_provider")
    matches = _require_bool(device_map.get("matches_internal"), "accuracy.device.matches_internal")
    if not matches:
        raise _error(
            "accuracy.device.matches_internal",
            "an apples-to-apples accuracy report must match the internal benchmark device",
        )
    _require_non_empty_str(device_map.get("selection"), "accuracy.device.selection")
    internal = _require_mapping(
        device_map.get("internal_benchmark"), "accuracy.device.internal_benchmark"
    )
    _require_non_empty_str(internal.get("tool"), "accuracy.device.internal_benchmark.tool")
    _require_non_empty_str(internal.get("device"), "accuracy.device.internal_benchmark.device")
    _require_non_empty_str(internal.get("provider"), "accuracy.device.internal_benchmark.provider")
    _require_non_empty_str(internal.get("selection"), "accuracy.device.internal_benchmark.selection")
    return torch_device


def _validate_evaluation(evaluation: Any) -> None:
    evaluation_map = _require_mapping(evaluation, "accuracy.evaluation")
    if evaluation_map.get("iou_type") != "bbox":
        raise _error("accuracy.evaluation.iou_type", "schema v1 supports only 'bbox'")
    _require_int(evaluation_map.get("max_dets"), "accuracy.evaluation.max_dets", minimum=1)
    _require_number(
        evaluation_map.get("score_threshold"), "accuracy.evaluation.score_threshold"
    )
    _require_int(evaluation_map.get("num_select"), "accuracy.evaluation.num_select", minimum=1)
    if "images_evaluated" in evaluation_map:
        _require_int(evaluation_map["images_evaluated"], "accuracy.evaluation.images_evaluated")


def _validate_predictions(predictions: Any, path: str) -> None:
    predictions_map = _require_mapping(predictions, path)
    _require_non_empty_str(predictions_map.get("format"), f"{path}.format")
    storage = _require_non_empty_str(predictions_map.get("storage"), f"{path}.storage")
    if storage not in {"inline", "json", "jsonl"}:
        raise _error(f"{path}.storage", "must be 'inline', 'json', or 'jsonl'")
    if "path" in predictions_map:
        raise _error(f"{path}.path", "prediction file paths are input-only and must not be reported")
    if storage != "inline":
        _require_hex(predictions_map.get("sha256"), f"{path}.sha256", SHA256_PATTERN, "sha256")
        _require_int(predictions_map.get("record_count"), f"{path}.record_count", minimum=0)
    if storage == "inline":
        _require_sequence(predictions_map.get("records"), f"{path}.records")


def _validate_metrics(metrics: Any, path: str) -> None:
    metrics_map = _require_mapping(metrics, path)
    _require_non_empty_str(metrics_map.get("tool"), f"{path}.tool")
    _require_non_empty_str(metrics_map.get("version"), f"{path}.version")
    values = _require_mapping(metrics_map.get("values"), f"{path}.values")
    for name, value in values.items():
        _require_number(value, f"{path}.values.{name}")


def _validate_pass(pass_entry: Any, *, expected_device: str, expected_provider: str) -> str:
    pass_map = _require_mapping(pass_entry, "accuracy.passes[]")
    pass_id = _require_non_empty_str(pass_map.get("pass_id"), "accuracy.passes[].pass_id")
    if pass_id not in PASS_IDS:
        raise _error("accuracy.passes[].pass_id", f"must be one of {list(PASS_IDS)}")
    _require_non_empty_str(pass_map.get("producer"), "accuracy.passes[].producer")
    origin = _require_non_empty_str(pass_map.get("origin"), "accuracy.passes[].origin")
    if origin not in PASS_ORIGINS:
        raise _error("accuracy.passes[].origin", f"must be one of {sorted(PASS_ORIGINS)}")
    status = _require_non_empty_str(pass_map.get("status"), "accuracy.passes[].status")
    if status not in PASS_STATUSES:
        raise _error("accuracy.passes[].status", f"must be one of {sorted(PASS_STATUSES)}")

    provenance = _require_mapping(pass_map.get("provenance"), "accuracy.passes[].provenance")
    pass_device = _require_non_empty_str(
        provenance.get("device"), "accuracy.passes[].provenance.device"
    )
    if pass_device != expected_device:
        raise _error(
            "accuracy.passes[].provenance.device",
            f"must match accuracy.device.torch_device ({expected_device})",
        )
    pass_provider = _require_non_empty_str(
        provenance.get("provider"), "accuracy.passes[].provenance.provider"
    )
    if pass_id == "onnx" and pass_provider != expected_provider:
        raise _error(
            "accuracy.passes[].provenance.provider",
            f"must match accuracy.device.ort_provider ({expected_provider})",
        )

    if status == "measured" and pass_map.get("metrics") is None and pass_map.get("predictions") is None:
        raise _error(
            "accuracy.passes[]",
            "a measured pass must contain 'metrics' or 'predictions'",
        )
    if pass_map.get("metrics") is not None:
        _validate_metrics(pass_map["metrics"], "accuracy.passes[].metrics")
    if pass_map.get("predictions") is not None:
        _validate_predictions(pass_map["predictions"], "accuracy.passes[].predictions")
    if "timing" in pass_map:
        _require_mapping(pass_map["timing"], "accuracy.passes[].timing")
    return pass_id


def _validate_comparisons(comparisons: Any) -> None:
    comparison_list = _require_sequence(comparisons, "accuracy.comparisons")
    for index, comparison in enumerate(comparison_list):
        path = f"accuracy.comparisons[{index}]"
        comparison_map = _require_mapping(comparison, path)
        left = _require_non_empty_str(comparison_map.get("left_pass"), f"{path}.left_pass")
        right = _require_non_empty_str(comparison_map.get("right_pass"), f"{path}.right_pass")
        if left not in PASS_IDS or right not in PASS_IDS or left == right:
            raise _error(path, "must compare two distinct passes from 'python' and 'onnx'")
        status = _require_non_empty_str(comparison_map.get("status"), f"{path}.status")
        if status not in COMPARISON_STATUSES:
            raise _error(f"{path}.status", f"must be one of {sorted(COMPARISON_STATUSES)}")
        if "deltas" in comparison_map:
            deltas = _require_mapping(comparison_map["deltas"], f"{path}.deltas")
            for name, value in deltas.items():
                _require_number(value, f"{path}.deltas.{name}")
        if "thresholds" in comparison_map:
            thresholds = _require_mapping(comparison_map["thresholds"], f"{path}.thresholds")
            for name, value in thresholds.items():
                _require_number(value, f"{path}.thresholds.{name}")


def _validate_accuracy(accuracy: Any) -> None:
    accuracy_map = _require_mapping(accuracy, "accuracy")
    if accuracy_map.get("scenario") != ACCURACY_SCENARIO:
        raise _error("accuracy.scenario", f"must be {ACCURACY_SCENARIO!r}")
    status = _require_non_empty_str(accuracy_map.get("status"), "accuracy.status")
    if status not in ACCURACY_STATUSES:
        raise _error("accuracy.status", f"must be one of {sorted(ACCURACY_STATUSES)}")
    if status != "measured":
        _require_non_empty_str(accuracy_map.get("reason"), "accuracy.reason")
        if "passes" in accuracy_map:
            _require_sequence(accuracy_map["passes"], "accuracy.passes")
        return

    expected_device = _validate_device(accuracy_map.get("device"))
    expected_provider = str(
        _require_mapping(accuracy_map["device"], "accuracy.device")["ort_provider"]
    )
    _validate_evaluation(accuracy_map.get("evaluation"))
    passes = _require_sequence(accuracy_map.get("passes"), "accuracy.passes")
    if len(passes) != 2:
        raise _error("accuracy.passes", "must contain exactly the 'python' and 'onnx' passes")
    pass_ids = [
        _validate_pass(
            pass_entry,
            expected_device=expected_device,
            expected_provider=expected_provider,
        )
        for pass_entry in passes
    ]
    if sorted(pass_ids) != sorted(PASS_IDS):
        raise _error("accuracy.passes", "must contain exactly one 'python' and one 'onnx' pass")
    comparisons = _require_sequence(accuracy_map.get("comparisons"), "accuracy.comparisons")
    if not comparisons:
        raise _error("accuracy.comparisons", "must contain the Python-vs-ONNX comparison")
    _validate_comparisons(comparisons)


def _validate_performance(performance: Any) -> None:
    performance_map = _require_mapping(performance, "performance")
    benchmarks = _require_sequence(performance_map.get("benchmarks"), "performance.benchmarks")
    for index, entry in enumerate(benchmarks):
        path = f"performance.benchmarks[{index}]"
        entry_map = _require_mapping(entry, path)
        run_id = _require_non_empty_str(entry_map.get("run_id"), f"{path}.run_id")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise _error(f"{path}.run_id", "must be Windows-safe (letters, digits, '.', '_', '-')")
        scenario = _require_non_empty_str(entry_map.get("scenario"), f"{path}.scenario")
        if scenario not in SCENARIOS:
            raise _error(f"{path}.scenario", f"must be one of {sorted(SCENARIOS)}")
        snapshot = _require_mapping(entry_map.get("snapshot"), f"{path}.snapshot")
        if "accuracy" in snapshot:
            raise _error(f"{path}.snapshot.accuracy", "benchmark snapshots are throughput-only")
        _reject_raw_predictions(snapshot, f"{path}.snapshot")
        snapshot_scenario = snapshot.get("scenario")
        if snapshot_scenario is not None and snapshot_scenario != scenario:
            raise _error(f"{path}.scenario", "must match the embedded snapshot scenario")
        if "report_id" in entry_map:
            report_id = _require_non_empty_str(entry_map["report_id"], f"{path}.report_id")
            if not REPORT_ID_PATTERN.fullmatch(report_id):
                raise _error(f"{path}.report_id", "must be Windows-safe")
        if "filename" in entry_map:
            _require_filename(entry_map["filename"], f"{path}.filename")
        if "sha256" in entry_map:
            _require_hex(entry_map["sha256"], f"{path}.sha256", SHA256_PATTERN, "sha256")


def validate_model_report(payload: Any) -> None:
    """Validate a model-report payload, raising :class:`ModelReportValidationError` on failure."""
    payload_map = _require_mapping(payload, "report")
    schema_version = _require_int(payload_map.get("schema_version"), "schema_version", minimum=1)
    if schema_version not in {MODEL_REPORT_SCHEMA_VERSION, _LEGACY_MODEL_REPORT_SCHEMA_VERSION}:
        raise _error(
            "schema_version",
            f"expected {MODEL_REPORT_SCHEMA_VERSION}, got {schema_version}",
        )
    if payload_map.get("kind") != MODEL_REPORT_KIND:
        raise _error("kind", f"must be {MODEL_REPORT_KIND!r}")
    report_id = _require_non_empty_str(payload_map.get("report_id"), "report_id")
    if not REPORT_ID_PATTERN.fullmatch(report_id):
        raise _error("report_id", "must be Windows-safe (letters, digits, '.', '_', '-')")
    _require_timestamp(payload_map.get("generated_at"), "generated_at")
    _validate_model(payload_map.get("model"))
    _validate_provenance(payload_map.get("provenance"))

    scoring = _require_mapping(payload_map.get("scoring"), "scoring")
    if not scoring:
        raise _error("scoring", "must contain at least an accuracy or performance section")
    if "accuracy" in scoring:
        _validate_accuracy(scoring["accuracy"])
    if "performance" in scoring:
        _validate_performance(scoring["performance"])
    if not any(key in scoring for key in ("accuracy", "performance")):
        raise _error("scoring", "must contain an accuracy or performance section")

    environment = _require_mapping(payload_map.get("environment"), "environment")
    if schema_version == _LEGACY_MODEL_REPORT_SCHEMA_VERSION:
        if "packages" in environment:
            _require_mapping(environment["packages"], "environment.packages")
    else:
        try:
            validate_hardware(environment.get("hardware"), "environment.hardware")
            validate_libraries(environment.get("libraries"), "environment.libraries")
        except SkellybenchValidationError as exc:
            raise ModelReportValidationError(str(exc)) from exc
    if "run_history" in payload_map:
        _require_sequence(payload_map["run_history"], "run_history")


def build_model_report(
    *,
    report_id: str,
    model: Mapping[str, Any],
    provenance: Mapping[str, Any],
    accuracy: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    tracked_packages: Iterable[str] | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Assemble and validate a model-report payload.

    Args:
        report_id: Windows-safe report identifier, unique for the model.
        model: Model metadata. Required keys are documented by :func:`validate_model_report`.
        provenance: Where the model came from and how it was produced.
        accuracy: Optional apples-to-apples accuracy section.
        performance: Optional benchmark-snapshot section. Defaults to an empty list.
        environment: Optional pre-captured environment. Captured locally when omitted.
        tracked_packages: Package names to track when *environment* is omitted.
        generated_at: Timestamp override for deterministic tests.

    Returns:
        A validated model-report payload.
    """
    if not REPORT_ID_PATTERN.fullmatch(str(report_id)):
        raise _error("report_id", "must be Windows-safe (letters, digits, '.', '_', '-')")

    if isinstance(generated_at, datetime):
        timestamp = generated_at.isoformat()
    elif generated_at is None:
        timestamp = iso_now()
    else:
        timestamp = str(generated_at)

    scoring: dict[str, Any] = {}
    if accuracy is not None:
        scoring["accuracy"] = copy.deepcopy(dict(accuracy))
    scoring["performance"] = copy.deepcopy(
        dict(performance) if performance is not None else {"benchmarks": []}
    )

    payload: dict[str, Any] = {
        "schema_version": MODEL_REPORT_SCHEMA_VERSION,
        "kind": MODEL_REPORT_KIND,
        "report_id": report_id,
        "generated_at": timestamp,
        "model": copy.deepcopy(dict(model)),
        "provenance": copy.deepcopy(dict(provenance)),
        "scoring": scoring,
        "environment": copy.deepcopy(dict(environment))
        if environment is not None
        else {
            "hardware": detect_hardware(),
            "libraries": detect_libraries(
                tracked_packages
                if tracked_packages is not None
                else DEFAULT_SOFTWARE_PACKAGES
            ),
        },
    }
    validate_model_report(payload)
    return payload


def is_measured_accuracy(accuracy: Any) -> bool:
    """Return whether an accuracy section contains a completed measurement."""
    return isinstance(accuracy, Mapping) and accuracy.get("status") == "measured"


def _deep_merge(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(existing))
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _upsert_benchmarks(existing: Any, incoming: Any) -> dict[str, Any]:
    existing_map = dict(existing) if isinstance(existing, Mapping) else {}
    incoming_map = dict(incoming) if isinstance(incoming, Mapping) else {}
    benchmarks = {
        str(entry.get("run_id")): copy.deepcopy(dict(entry))
        for entry in existing_map.get("benchmarks") or []
        if isinstance(entry, Mapping) and entry.get("run_id")
    }
    for entry in incoming_map.get("benchmarks") or []:
        if isinstance(entry, Mapping) and entry.get("run_id"):
            benchmarks[str(entry["run_id"])] = copy.deepcopy(dict(entry))
    return {"benchmarks": [benchmarks[key] for key in sorted(benchmarks)]}


def merge_model_report(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge *incoming* into *existing* without erasing a measured accuracy result.

    Benchmark snapshots are upserted by ``run_id``. An incoming skipped/unavailable accuracy
    section does not replace a measured one; a new measurement does replace the old one. The
    generate-once gate avoids this path unless a caller explicitly rebuilds a report.
    """
    if not existing:
        merged = copy.deepcopy(dict(incoming))
        validate_model_report(merged)
        return merged

    merged = copy.deepcopy(dict(existing))
    merged["schema_version"] = incoming.get("schema_version", MODEL_REPORT_SCHEMA_VERSION)
    merged["kind"] = incoming.get("kind", MODEL_REPORT_KIND)
    merged["generated_at"] = incoming.get("generated_at", merged.get("generated_at"))
    merged["report_id"] = incoming.get("report_id", merged.get("report_id"))
    merged["model"] = _deep_merge(merged.get("model") or {}, incoming.get("model") or {})
    merged["provenance"] = _deep_merge(
        merged.get("provenance") or {}, incoming.get("provenance") or {}
    )
    merged["environment"] = _deep_merge(
        merged.get("environment") or {}, incoming.get("environment") or {}
    )

    existing_scoring = merged.get("scoring") or {}
    incoming_scoring = incoming.get("scoring") or {}
    scoring = copy.deepcopy(dict(existing_scoring))
    if "accuracy" in incoming_scoring:
        incoming_accuracy = incoming_scoring["accuracy"]
        existing_accuracy = existing_scoring.get("accuracy")
        if is_measured_accuracy(existing_accuracy) and not is_measured_accuracy(incoming_accuracy):
            pass
        else:
            scoring["accuracy"] = copy.deepcopy(dict(incoming_accuracy))
    scoring["performance"] = _upsert_benchmarks(
        existing_scoring.get("performance"), incoming_scoring.get("performance")
    )
    merged["scoring"] = scoring

    history = list(merged.get("run_history") or [])
    history.extend(incoming.get("run_history") or [])
    if history:
        merged["run_history"] = history[-20:]

    validate_model_report(merged)
    return merged
