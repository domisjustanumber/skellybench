"""Schema version 3: constants and validators for every report skellybench writes.

skellybench writes three JSON kinds:

- ``benchmark_report``: one run x subject x benchmark type x execution provider.
- ``benchmark_predictions``: the columnar predictions of one accuracy benchmark report.
- ``model_report``: one model artifact, written once when the model is created.

Validators raise :class:`SkellybenchValidationError` with the JSON path of the first problem.
No report may hold an absolute local path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 3

BENCHMARK_REPORT_KIND = "benchmark_report"
PREDICTIONS_KIND = "benchmark_predictions"
MODEL_REPORT_KIND = "model_report"
REPORT_KINDS = (BENCHMARK_REPORT_KIND, PREDICTIONS_KIND, MODEL_REPORT_KIND)

ACCURACY = "accuracy"
PERFORMANCE = "performance"
BENCHMARK_TYPES = (ACCURACY, PERFORMANCE)
BENCHMARK_STATUSES = ("measured", "unavailable", "error")
COMPARISON_STATUSES = ("ok", "warn", "fail")
SCORED_BY = ("skellybench", "source_tool")
MODES = ("device", "host")
TIMING_SCOPES = ("session", "end_to_end")
PREDICTIONS_FORMAT = "coco_detections_columnar_v1"

COCO_STAT_KEYS = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
    "ar_small",
    "ar_medium",
    "ar_large",
)

SUBJECT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
"""A subject (artifact stem or ``<model_id>_source``). No ``.``, because ``.`` separates filename parts."""
EP_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HOSTNAME_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*$")
REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?!/))")
_PATH_KEYS = frozenset({"path", "filename", "file", "local"})


class SkellybenchValidationError(ValueError):
    """A report payload does not match schema version 3."""


def _error(path: str, message: str) -> SkellybenchValidationError:
    return SkellybenchValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _optional_mapping(value: Any, path: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, path)


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(path, "must be an array")
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "must be a non-empty string")
    return value


def _optional_str(value: Any, path: str) -> str | None:
    return None if value is None else _str(value, path)


def _int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, f"must be an integer >= {minimum}")
    return value


def _optional_int(value: Any, path: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _int(value, path, minimum=minimum)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise _error(path, "must be a finite number")
    return float(value)


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "must be a boolean")
    return value


def _optional_bool(value: Any, path: str) -> bool | None:
    return None if value is None else _bool(value, path)


def _pattern(value: Any, path: str, pattern: re.Pattern[str], label: str) -> str:
    text = _str(value, path)
    if not pattern.fullmatch(text):
        raise _error(path, f"must be {label}")
    return text


def _sha256(value: Any, path: str) -> str:
    return _pattern(value, path, SHA256_PATTERN, "a lowercase hex SHA-256")


def _optional_sha256(value: Any, path: str) -> str | None:
    return None if value is None else _sha256(value, path)


def _fingerprint(value: Any, path: str) -> str:
    return _pattern(value, path, FINGERPRINT_PATTERN, "'sha256:' plus 64 lowercase hex characters")


def _timestamp(value: Any, path: str) -> str:
    text = _str(value, path)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise _error(path, "must be an ISO-8601 timestamp") from None
    return text


def _one_of(value: Any, path: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise _error(path, f"must be one of {list(choices)}")
    return value


def reject_local_paths(value: Any, path: str = "$") -> None:
    """Raise when any string value in *value* is an absolute local path.

    URLs and relative paths are allowed. The ``data`` field of a predictions file is not checked.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "data":
                continue
            reject_local_paths(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_local_paths(item, f"{path}[{index}]")
    elif isinstance(value, str) and _ABSOLUTE_PATH.match(value):
        raise _error(path, f"must not hold an absolute local path ({value!r})")


def _header(payload: Any, kind: str) -> Mapping[str, Any]:
    document = _mapping(payload, "$")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise _error("$.schema_version", f"must be {SCHEMA_VERSION}")
    if document.get("kind") != kind:
        raise _error("$.kind", f"must be {kind!r}")
    return document


# --- shared blocks ---------------------------------------------------------------------------------


def validate_run(run: Any, path: str = "run") -> None:
    """Validate a ``run`` block."""
    block = _mapping(run, path)
    _pattern(block.get("run_id"), f"{path}.run_id", RUN_ID_PATTERN, "'<YYYYMMDDTHHMMSSZ>-<hostname slug>'")
    _timestamp(block.get("started_at"), f"{path}.started_at")
    if block.get("finished_at") is not None:
        _timestamp(block["finished_at"], f"{path}.finished_at")
    _str(block.get("command"), f"{path}.command")
    producer = _mapping(block.get("producer"), f"{path}.producer")
    _str(producer.get("name"), f"{path}.producer.name")
    _optional_str(producer.get("version"), f"{path}.producer.version")
    _optional_str(producer.get("git_commit"), f"{path}.producer.git_commit")
    _optional_bool(producer.get("git_dirty"), f"{path}.producer.git_dirty")
    _optional_sha256(producer.get("git_diff_sha256"), f"{path}.producer.git_diff_sha256")
    _optional_sha256(producer.get("lock_file_sha256"), f"{path}.producer.lock_file_sha256")


def validate_host(host: Any, path: str = "host") -> None:
    """Validate a ``host`` block."""
    block = _mapping(host, path)
    _str(block.get("hostname"), f"{path}.hostname")
    _mapping(block.get("os"), f"{path}.os")
    virtualization = _mapping(block.get("virtualization"), f"{path}.virtualization")
    for key in ("is_wsl", "is_container", "is_vm"):
        _optional_bool(virtualization.get(key), f"{path}.virtualization.{key}")
    cpu = _mapping(block.get("cpu"), f"{path}.cpu")
    _optional_str(cpu.get("name"), f"{path}.cpu.name")
    for index, gpu in enumerate(_sequence(block.get("gpus"), f"{path}.gpus")):
        entry = _mapping(gpu, f"{path}.gpus[{index}]")
        _optional_str(entry.get("name"), f"{path}.gpus[{index}].name")
        _optional_str(entry.get("uuid"), f"{path}.gpus[{index}].uuid")
    _optional_mapping(block.get("container"), f"{path}.container")
    _fingerprint(block.get("hardware_fingerprint"), f"{path}.hardware_fingerprint")


def validate_environment(environment: Any, path: str = "environment") -> None:
    """Validate an ``environment`` block.

    An external environment (see ``blocks.build_external_environment``) also has ``collected_by`` and
    ``container``. Both are optional here.
    """
    block = _mapping(environment, path)
    python = _mapping(block.get("python"), f"{path}.python")
    _str(python.get("version"), f"{path}.python.version")
    packages = _mapping(block.get("packages"), f"{path}.packages")
    for name, version in packages.items():
        if not isinstance(name, str) or name != name.lower():
            raise _error(f"{path}.packages", f"key {name!r} must be a lowercase distribution name")
        _str(version, f"{path}.packages.{name}")
    _mapping(block.get("package_sources", {}), f"{path}.package_sources")
    _optional_str(block.get("collected_by"), f"{path}.collected_by")
    container = _optional_mapping(block.get("container"), f"{path}.container")
    if container is not None:
        _str(container.get("image"), f"{path}.container.image")
        _optional_str(container.get("tag"), f"{path}.container.tag")
        if container.get("digest") is not None:
            _fingerprint(container["digest"], f"{path}.container.digest")
        _optional_sha256(container.get("dockerfile_sha256"), f"{path}.container.dockerfile_sha256")
        _bool(container.get("published"), f"{path}.container.published")
    _fingerprint(block.get("environment_fingerprint"), f"{path}.environment_fingerprint")


def validate_execution(execution: Any, path: str = "execution") -> None:
    """Validate an ``execution`` block."""
    block = _mapping(execution, path)
    _str(block.get("runtime"), f"{path}.runtime")
    provider = _mapping(block.get("provider"), f"{path}.provider")
    _str(provider.get("key"), f"{path}.provider.key")
    _pattern(provider.get("slug"), f"{path}.provider.slug", EP_SLUG_PATTERN, "a lowercase EP slug")
    _str(provider.get("ep_name"), f"{path}.provider.ep_name")
    device = _optional_mapping(block.get("device"), f"{path}.device")
    if device is not None:
        _one_of(device.get("kind"), f"{path}.device.kind", ("cpu", "gpu", "npu", "soc"))
        _optional_str(device.get("name"), f"{path}.device.name")
        _optional_str(device.get("host_gpu_uuid"), f"{path}.device.host_gpu_uuid")
    _mapping(block.get("drivers", {}), f"{path}.drivers")
    _str(block.get("label"), f"{path}.label")


def validate_subject(subject: Any, path: str = "subject") -> None:
    """Validate a benchmark ``subject`` block."""
    block = _mapping(subject, path)
    _pattern(block.get("subject_id"), f"{path}.subject_id", SUBJECT_PATTERN, "a subject without '.'")
    _str(block.get("model_report_id"), f"{path}.model_report_id")
    _str(block.get("model_id"), f"{path}.model_id")
    _str(block.get("format"), f"{path}.format")
    _str(block.get("precision"), f"{path}.precision")
    _int(block.get("batch_size"), f"{path}.batch_size", minimum=1)
    _bool(block.get("batch_agnostic"), f"{path}.batch_agnostic")
    artifact = _mapping(block.get("artifact"), f"{path}.artifact")
    _str(artifact.get("filename"), f"{path}.artifact.filename")
    _sha256(artifact.get("sha256"), f"{path}.artifact.sha256")
    _optional_int(artifact.get("size_bytes"), f"{path}.artifact.size_bytes")
    validate_input(block.get("input"), f"{path}.input")


def validate_input(value: Any, path: str = "input") -> None:
    """Validate an ``input`` block: image width and height."""
    block = _mapping(value, path)
    _int(block.get("width"), f"{path}.width", minimum=1)
    _int(block.get("height"), f"{path}.height", minimum=1)


def validate_dataset(dataset: Any, path: str = "dataset") -> None:
    """Validate a ``dataset`` block."""
    block = _mapping(dataset, path)
    _str(block.get("name"), f"{path}.name")
    _optional_sha256(block.get("annotation_sha256"), f"{path}.annotation_sha256")
    _sha256(block.get("image_list_sha256"), f"{path}.image_list_sha256")
    _optional_int(block.get("images_available"), f"{path}.images_available")
    _int(block.get("images_used"), f"{path}.images_used", minimum=1)
    _str(block.get("selection"), f"{path}.selection")
    _optional_int(block.get("limit"), f"{path}.limit", minimum=1)


def validate_protocol(protocol: Any, path: str = "protocol") -> None:
    """Validate a ``protocol`` block. It must carry a ``protocol_id``."""
    block = _mapping(protocol, path)
    _str(block.get("protocol_id"), f"{path}.protocol_id")
    _one_of(block.get("benchmark_type"), f"{path}.benchmark_type", BENCHMARK_TYPES)
    _one_of(block.get("timing_scope"), f"{path}.timing_scope", TIMING_SCOPES)


def validate_metrics(metrics: Any, path: str = "metrics") -> None:
    """Validate a COCO ``metrics`` block."""
    block = _mapping(metrics, path)
    _str(block.get("tool"), f"{path}.tool")
    _optional_str(block.get("version"), f"{path}.version")
    values = _mapping(block.get("values"), f"{path}.values")
    for key in COCO_STAT_KEYS:
        if key not in values:
            raise _error(f"{path}.values", f"missing {key!r}")
        _number(values[key], f"{path}.values.{key}")


def validate_method(method: Any, path: str = "method", *, metrics_tool: str | None = None) -> None:
    """Validate an ``accuracy.method`` block: ``{decode, scorer: {tool, version}, scored_by}``.

    ``decode`` is a free-form mapping. When *metrics_tool* is given, ``scorer.tool`` must equal it.
    """
    block = _mapping(method, path)
    _mapping(block.get("decode"), f"{path}.decode")
    scorer = _mapping(block.get("scorer"), f"{path}.scorer")
    tool = _str(scorer.get("tool"), f"{path}.scorer.tool")
    _optional_str(scorer.get("version"), f"{path}.scorer.version")
    _one_of(block.get("scored_by"), f"{path}.scored_by", SCORED_BY)
    if metrics_tool is not None and tool != metrics_tool:
        raise _error(f"{path}.scorer.tool", f"must equal accuracy.metrics.tool ({metrics_tool!r})")


def validate_predictions_reference(reference: Any, path: str = "predictions") -> None:
    """Validate the reference to a predictions file inside an accuracy block."""
    block = _mapping(reference, path)
    _str(block.get("filename"), f"{path}.filename")
    _sha256(block.get("file_sha256"), f"{path}.file_sha256")
    _sha256(block.get("data_sha256"), f"{path}.data_sha256")
    _one_of(block.get("format"), f"{path}.format", (PREDICTIONS_FORMAT,))
    _int(block.get("record_count"), f"{path}.record_count")


def validate_comparison(comparison: Any, path: str = "comparison") -> None:
    """Validate a self-vs-parent ``comparison`` block."""
    block = _mapping(comparison, path)
    _one_of(block.get("status"), f"{path}.status", COMPARISON_STATUSES)
    deltas = _mapping(block.get("deltas"), f"{path}.deltas")
    for key, value in deltas.items():
        _number(value, f"{path}.deltas.{key}")
    thresholds = _mapping(block.get("thresholds"), f"{path}.thresholds")
    _number(thresholds.get("warn_ap"), f"{path}.thresholds.warn_ap")
    _number(thresholds.get("fail_ap"), f"{path}.thresholds.fail_ap")


# --- benchmark report ------------------------------------------------------------------------------


def _validate_performance(performance: Any, path: str = "performance") -> None:
    block = _mapping(performance, path)
    records = _sequence(block.get("records"), f"{path}.records")
    for index, record in enumerate(records):
        entry_path = f"{path}.records[{index}]"
        entry = _mapping(record, entry_path)
        _one_of(entry.get("mode"), f"{entry_path}.mode", MODES)
        _one_of(entry.get("timing_scope"), f"{entry_path}.timing_scope", TIMING_SCOPES)
        timing = _mapping(entry.get("timing"), f"{entry_path}.timing")
        _number(timing.get("images_per_second"), f"{entry_path}.timing.images_per_second")
    for index, item in enumerate(_sequence(block.get("unavailable", []), f"{path}.unavailable")):
        entry = _mapping(item, f"{path}.unavailable[{index}]")
        _str(entry.get("reason"), f"{path}.unavailable[{index}].reason")


def _validate_accuracy(accuracy: Any, path: str = "accuracy") -> None:
    block = _mapping(accuracy, path)
    _mapping(block.get("evaluation"), f"{path}.evaluation")
    validate_metrics(block.get("metrics"), f"{path}.metrics")
    validate_method(block.get("method"), f"{path}.method", metrics_tool=block["metrics"]["tool"])
    validate_predictions_reference(block.get("predictions"), f"{path}.predictions")
    pair = _optional_mapping(block.get("pair"), f"{path}.pair")
    if pair is not None:
        _str(pair.get("partner_report_id"), f"{path}.pair.partner_report_id")


def validate_benchmark_report(payload: Any) -> None:
    """Validate one benchmark report."""
    document = _header(payload, BENCHMARK_REPORT_KIND)
    benchmark_type = _one_of(document.get("benchmark_type"), "$.benchmark_type", BENCHMARK_TYPES)
    status = _one_of(document.get("status"), "$.status", BENCHMARK_STATUSES)
    _pattern(document.get("report_id"), "$.report_id", REPORT_ID_PATTERN, "a filename-safe report id")
    _timestamp(document.get("generated_at"), "$.generated_at")
    validate_run(document.get("run"), "$.run")
    validate_host(document.get("host"), "$.host")
    validate_environment(document.get("environment"), "$.environment")
    validate_execution(document.get("execution"), "$.execution")
    validate_subject(document.get("subject"), "$.subject")
    if status == "measured":
        validate_dataset(document.get("dataset"), "$.dataset")
        validate_protocol(document.get("protocol"), "$.protocol")
        if document["protocol"]["benchmark_type"] != benchmark_type:
            raise _error("$.protocol.benchmark_type", "must equal $.benchmark_type")
        _validate_performance(document.get("performance"), "$.performance")
        if benchmark_type == ACCURACY:
            _validate_accuracy(document.get("accuracy"), "$.accuracy")
        elif document.get("accuracy") is not None:
            raise _error("$.accuracy", "must be absent in a performance report")
    else:
        _str(document.get("reason"), "$.reason")
        if document.get("accuracy") is not None:
            raise _error("$.accuracy", "must be absent when status is not 'measured'")
    reject_local_paths(document)


# --- predictions file ------------------------------------------------------------------------------


def validate_predictions_header(payload: Any) -> None:
    """Validate a predictions file, except the decoded content of ``data``."""
    document = _header(payload, PREDICTIONS_KIND)
    _pattern(document.get("report_id"), "$.report_id", REPORT_ID_PATTERN, "a filename-safe report id")
    _one_of(document.get("format"), "$.format", (PREDICTIONS_FORMAT,))
    image_ids = _sequence(document.get("image_ids"), "$.image_ids")
    for index, image_id in enumerate(image_ids):
        _int(image_id, f"$.image_ids[{index}]")
    per_image = document.get("per_image")
    counts = document.get("counts")
    record_count = _int(document.get("record_count"), "$.record_count")
    if (per_image is None) == (counts is None):
        raise _error("$.per_image", "exactly one of per_image and counts must be set")
    if per_image is not None:
        _int(per_image, "$.per_image")
        expected = per_image * len(image_ids)
    else:
        count_values = _sequence(counts, "$.counts")
        if len(count_values) != len(image_ids):
            raise _error("$.counts", "must have one entry per image id")
        expected = sum(_int(value, f"$.counts[{index}]") for index, value in enumerate(count_values))
    if expected != record_count:
        raise _error("$.record_count", f"must equal {expected}")
    _one_of(document.get("byte_order"), "$.byte_order", ("little",))
    _one_of(document.get("layout"), "$.layout", ("column-major",))
    _one_of(document.get("encoding"), "$.encoding", ("gzip+base64",))
    _sha256(document.get("data_sha256"), "$.data_sha256")
    if not isinstance(document.get("data"), str):
        raise _error("$.data", "must be a string")


# --- model report ----------------------------------------------------------------------------------


def _physical_device(snapshot: Mapping[str, Any]) -> str | None:
    execution = snapshot.get("execution") or {}
    device = execution.get("device") or {}
    if device.get("host_gpu_uuid"):
        return f"gpu:{device['host_gpu_uuid']}"
    host = snapshot.get("host") or {}
    fingerprint = host.get("hardware_fingerprint")
    cpu = (host.get("cpu") or {}).get("name")
    if device.get("kind") in (None, "cpu", "soc"):
        return f"cpu:{fingerprint}:{cpu}"
    return None


def validate_accuracy_pair(self_snapshot: Mapping[str, Any], parent_snapshot: Mapping[str, Any]) -> None:
    """Raise when two accuracy benchmarks are not a valid self/parent pair.

    They must share the dataset hashes, batch size, ``accuracy.evaluation``, and physical device. The
    runs may differ, because a model report can reuse a parent pass from an earlier run. The EP may
    differ (for example ``torch-cuda`` for a checkpoint and ``cuda`` for its ONNX export). The
    ``accuracy.method`` blocks may differ: nobody compares them.
    """
    checks = (
        ("dataset.annotation_sha256", lambda snapshot: (snapshot.get("dataset") or {}).get("annotation_sha256")),
        ("dataset.image_list_sha256", lambda snapshot: (snapshot.get("dataset") or {}).get("image_list_sha256")),
        ("subject.batch_size", lambda snapshot: (snapshot.get("subject") or {}).get("batch_size")),
        ("accuracy.evaluation", lambda snapshot: (snapshot.get("accuracy") or {}).get("evaluation")),
        ("physical device", _physical_device),
    )
    for label, getter in checks:
        left = getter(self_snapshot)
        right = getter(parent_snapshot)
        if left is None or left != right:
            raise _error("$.accuracy", f"self and parent differ in {label} ({left!r} != {right!r})")


def _validate_accuracy_entry(entry: Any, path: str) -> Mapping[str, Any]:
    block = _mapping(entry, path)
    link = _mapping(block.get("link"), f"{path}.link")
    _str(link.get("path"), f"{path}.link.path")
    _sha256(link.get("sha256"), f"{path}.link.sha256")
    snapshot = _mapping(block.get("snapshot"), f"{path}.snapshot")
    validate_benchmark_report(snapshot)
    if snapshot.get("benchmark_type") != ACCURACY or snapshot.get("status") != "measured":
        raise _error(f"{path}.snapshot", "must be a measured accuracy benchmark")
    return snapshot


def validate_model_report(payload: Any) -> None:
    """Validate one model report, including the self/parent accuracy pair."""
    document = _header(payload, MODEL_REPORT_KIND)
    report_id = _pattern(document.get("report_id"), "$.report_id", SUBJECT_PATTERN, "a subject without '.'")
    _pattern(document.get("created_run_id"), "$.created_run_id", RUN_ID_PATTERN, "a run id")
    _timestamp(document.get("generated_at"), "$.generated_at")

    model = _mapping(document.get("model"), "$.model")
    _str(model.get("model_id"), "$.model.model_id")
    _str(model.get("format"), "$.model.format")
    _str(model.get("precision"), "$.model.precision")
    _optional_int(model.get("batch_size"), "$.model.batch_size", minimum=1)
    variant = _mapping(model.get("variant"), "$.model.variant")
    _str(variant.get("family"), "$.model.variant.family")
    _str(variant.get("size"), "$.model.variant.size")
    _int(variant.get("size_rank"), "$.model.variant.size_rank")
    validate_input(model.get("input"), "$.model.input")
    artifact = _mapping(model.get("artifact"), "$.model.artifact")
    _str(artifact.get("filename"), "$.model.artifact.filename")
    _sha256(artifact.get("sha256"), "$.model.artifact.sha256")

    creation = _mapping(document.get("creation"), "$.creation")
    validate_run(creation.get("run"), "$.creation.run")
    if creation["run"]["run_id"] != document["created_run_id"]:
        raise _error("$.creation.run.run_id", "must equal $.created_run_id")
    validate_host(creation.get("host"), "$.creation.host")
    validate_environment(creation.get("environment"), "$.creation.environment")
    source = _mapping(creation.get("source"), "$.creation.source")
    _str(source.get("kind"), "$.creation.source.kind")
    _str(source.get("url"), "$.creation.source.url")
    if source.get("md5") is not None:
        _pattern(source["md5"], "$.creation.source.md5", MD5_PATTERN, "a lowercase hex MD5")
    _sha256(source.get("sha256"), "$.creation.source.sha256")
    parent = _optional_mapping(creation.get("parent"), "$.creation.parent")
    if parent is not None:
        _str(parent.get("report_id"), "$.creation.parent.report_id")
        _sha256(parent.get("sha256"), "$.creation.parent.sha256")
    sidecar = _optional_mapping(creation.get("config_sidecar"), "$.creation.config_sidecar")
    if sidecar is not None:
        _str(sidecar.get("url"), "$.creation.config_sidecar.url")
        _sha256(sidecar.get("sha256"), "$.creation.config_sidecar.sha256")
    steps = _sequence(creation.get("steps"), "$.creation.steps")
    if not steps:
        raise _error("$.creation.steps", "must not be empty")
    for index, step in enumerate(steps):
        entry = _mapping(step, f"$.creation.steps[{index}]")
        _str(entry.get("step"), f"$.creation.steps[{index}].step")
        _str(entry.get("tool"), f"$.creation.steps[{index}].tool")
        if parent is not None and entry.get("input_sha256") is not None:
            if entry["input_sha256"] != parent["sha256"]:
                raise _error(
                    f"$.creation.steps[{index}].input_sha256",
                    "must equal $.creation.parent.sha256",
                )
    last_output = steps[-1].get("output_sha256")
    if last_output is not None and last_output != artifact["sha256"]:
        raise _error("$.creation.steps", "the last step output_sha256 must equal $.model.artifact.sha256")

    accuracy = _mapping(document.get("accuracy"), "$.accuracy")
    self_snapshot = _validate_accuracy_entry(accuracy.get("self"), "$.accuracy.self")
    if (self_snapshot.get("subject") or {}).get("artifact", {}).get("sha256") != artifact["sha256"]:
        raise _error("$.accuracy.self.snapshot.subject.artifact.sha256", "must equal $.model.artifact.sha256")
    if self_snapshot["subject"].get("model_report_id") != report_id:
        raise _error("$.accuracy.self.snapshot.subject.model_report_id", "must equal $.report_id")
    if parent is None:
        if accuracy.get("parent") is not None or accuracy.get("comparison") is not None:
            raise _error("$.accuracy", "parent and comparison must be null for a model without a parent")
        if accuracy.get("parent_run_id") is not None:
            raise _error("$.accuracy.parent_run_id", "must be absent for a model without a parent")
    else:
        parent_snapshot = _validate_accuracy_entry(accuracy.get("parent"), "$.accuracy.parent")
        if parent_snapshot["subject"]["artifact"]["sha256"] != parent["sha256"]:
            raise _error("$.accuracy.parent.snapshot.subject.artifact.sha256", "must equal $.creation.parent.sha256")
        validate_accuracy_pair(self_snapshot, parent_snapshot)
        validate_comparison(accuracy.get("comparison"), "$.accuracy.comparison")
        parent_run_id = parent_snapshot["run"]["run_id"]
        if parent_run_id != document["created_run_id"]:
            if accuracy.get("parent_run_id") != parent_run_id:
                raise _error(
                    "$.accuracy.parent_run_id",
                    f"must equal the parent snapshot run id ({parent_run_id!r}) when it differs from $.created_run_id",
                )
        elif accuracy.get("parent_run_id") is not None:
            raise _error("$.accuracy.parent_run_id", "must be absent when the parent ran in $.created_run_id")
    reject_local_paths(document)


def payload_kind(payload: Any) -> str | None:
    """Return the ``kind`` of a schema-3 payload, or ``None`` for any other JSON."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        return None
    kind = payload.get("kind")
    return kind if kind in REPORT_KINDS else None


__all__ = [
    "ACCURACY",
    "BENCHMARK_REPORT_KIND",
    "BENCHMARK_STATUSES",
    "BENCHMARK_TYPES",
    "COCO_STAT_KEYS",
    "COMPARISON_STATUSES",
    "EP_SLUG_PATTERN",
    "FINGERPRINT_PATTERN",
    "HOSTNAME_SLUG_PATTERN",
    "MODEL_REPORT_KIND",
    "MODES",
    "PERFORMANCE",
    "PREDICTIONS_FORMAT",
    "PREDICTIONS_KIND",
    "REPORT_ID_PATTERN",
    "REPORT_KINDS",
    "RUN_ID_PATTERN",
    "SCHEMA_VERSION",
    "SCORED_BY",
    "SHA256_PATTERN",
    "SUBJECT_PATTERN",
    "SkellybenchValidationError",
    "TIMING_SCOPES",
    "payload_kind",
    "reject_local_paths",
    "validate_accuracy_pair",
    "validate_benchmark_report",
    "validate_comparison",
    "validate_dataset",
    "validate_environment",
    "validate_execution",
    "validate_host",
    "validate_input",
    "validate_method",
    "validate_metrics",
    "validate_model_report",
    "validate_predictions_header",
    "validate_predictions_reference",
    "validate_protocol",
    "validate_run",
    "validate_subject",
]
