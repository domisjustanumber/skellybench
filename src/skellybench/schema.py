"""Schema constants and validation for skellybench inputs and outputs.

The package is library-agnostic: it validates and renders plain dictionaries. Scoring is the only
optional dependency, and it is imported lazily by :mod:`skellybench.scoring`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SKELLYBENCH_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
BENCHMARK_PROTOCOL_VERSION = 1

INPUT_BUNDLE_KIND = "skellybench_run"
BENCHMARK_REPORT_KIND = "benchmark_report"
BENCHMARK_SUMMARY_KIND = "benchmark_summary"
BENCHMARK_OVERVIEW_KIND = "benchmark_overview"

ACCURACY_SCENARIO = "accuracy_full_dataset"
THROUGHPUT_SCENARIO = "throughput_adaptive"
SCENARIOS = (ACCURACY_SCENARIO, THROUGHPUT_SCENARIO)

ACCURACY_STATUSES = ("measured", "skipped", "unavailable", "error")
PASS_STATUSES = ("measured", "skipped", "unavailable", "error")
PASS_IDS = ("python", "onnx")
PASS_ORIGINS = ("python_model", "onnx_model", "other")
COMPARISON_STATUSES = ("ok", "warn", "fail", "not_available")
SCORING_MODES = ("recomputed", "precomputed", "unavailable")

MODES = ("device", "host")
TIMING_SCOPES = ("session", "end_to_end")
PROVIDER_KEYS = ("cpu", "openvino_gpu", "cuda", "tensorrt_rtx")

PREDICTION_STORAGES = ("inline", "json", "jsonl")
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

REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_ID_PATTERN = REPORT_ID_PATTERN
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
HASH_REF_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class SkellybenchValidationError(ValueError):
    """Raised when a bundle or report does not satisfy the skellybench schema."""


def _error(path: str, message: str) -> SkellybenchValidationError:
    return SkellybenchValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "must be a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "must be a boolean")
    return value


def _int(value: Any, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _error(path, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise _error(path, "must be a finite number")
    return float(value)


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(path, "must be an array")
    return value


def _hash(value: Any, path: str, pattern: re.Pattern[str], label: str) -> str:
    text = _str(value, path)
    if not pattern.fullmatch(text):
        raise _error(path, f"must be a lowercase hex {label}")
    return text


def _hash_ref(value: Any, path: str) -> str:
    text = _str(value, path)
    if not HASH_REF_PATTERN.fullmatch(text):
        raise _error(path, "must be a sha256 hash or sha256:<hex> reference")
    return text


def _filename(value: Any, path: str) -> str:
    name = _str(value, path)
    if Path(name).name != name:
        raise _error(path, "must be a filename, not a path")
    return name


def _timestamp(value: Any, path: str) -> str:
    text = _str(value, path)
    from datetime import datetime

    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise _error(path, "must be an ISO-8601 timestamp") from exc
    return text


def validate_prediction_record(record: Any, path: str = "prediction") -> None:
    """Validate one COCO detection result record."""
    record_map = _mapping(record, path)
    _int(record_map.get("image_id"), f"{path}.image_id")
    _int(record_map.get("category_id"), f"{path}.category_id")
    bbox = _sequence(record_map.get("bbox"), f"{path}.bbox")
    if len(bbox) != 4:
        raise _error(f"{path}.bbox", "must have exactly four numbers")
    for index, value in enumerate(bbox):
        _number(value, f"{path}.bbox[{index}]")
    if float(bbox[2]) < 0 or float(bbox[3]) < 0:
        raise _error(f"{path}.bbox", "width and height must be non-negative")
    _number(record_map.get("score"), f"{path}.score")


def validate_predictions_store(store: Any, path: str = "predictions") -> None:
    """Validate an inline/JSON/JSONL prediction store without reading it."""
    store_map = _mapping(store, path)
    _str(store_map.get("format"), f"{path}.format")
    storage = _str(store_map.get("storage"), f"{path}.storage")
    if storage not in PREDICTION_STORAGES:
        raise _error(f"{path}.storage", f"must be one of {list(PREDICTION_STORAGES)}")
    if storage == "inline":
        if "path" in store_map:
            raise _error(f"{path}.path", "inline predictions must not also point at a file")
        records = _sequence(store_map.get("records"), f"{path}.records")
        for index, record in enumerate(records):
            validate_prediction_record(record, f"{path}.records[{index}]")
        return
    if "records" in store_map:
        raise _error(f"{path}.records", "external predictions must not also embed records")
    _str(store_map.get("path"), f"{path}.path")
    _hash(store_map.get("sha256"), f"{path}.sha256", SHA256_PATTERN, "sha256")
    _int(store_map.get("record_count"), f"{path}.record_count", minimum=0)


def validate_metrics(metrics: Any, path: str = "metrics") -> None:
    """Validate a metrics block and known COCO detection values."""
    metrics_map = _mapping(metrics, path)
    _str(metrics_map.get("tool"), f"{path}.tool")
    _str(metrics_map.get("version"), f"{path}.version")
    values = _mapping(metrics_map.get("values"), f"{path}.values")
    for key, value in values.items():
        _number(value, f"{path}.values.{key}")


def _validate_device(device: Any) -> str:
    device_map = _mapping(device, "accuracy.device")
    device_type = _str(device_map.get("type"), "accuracy.device.type")
    if device_type not in {"cuda", "cpu"}:
        raise _error("accuracy.device.type", "must be 'cuda' or 'cpu'")
    if device_type == "cuda":
        _int(device_map.get("index"), "accuracy.device.index")
    torch_device = _str(device_map.get("torch_device"), "accuracy.device.torch_device")
    _str(device_map.get("ort_provider"), "accuracy.device.ort_provider")
    if not _bool(device_map.get("matches_internal"), "accuracy.device.matches_internal"):
        raise _error("accuracy.device.matches_internal", "must be true for apples-to-apples accuracy")
    _str(device_map.get("selection"), "accuracy.device.selection")
    internal = _mapping(device_map.get("internal_benchmark"), "accuracy.device.internal_benchmark")
    _str(internal.get("tool"), "accuracy.device.internal_benchmark.tool")
    _str(internal.get("device"), "accuracy.device.internal_benchmark.device")
    _str(internal.get("provider"), "accuracy.device.internal_benchmark.provider")
    _str(internal.get("selection"), "accuracy.device.internal_benchmark.selection")
    return torch_device


def _validate_evaluation(evaluation: Any) -> None:
    evaluation_map = _mapping(evaluation, "accuracy.evaluation")
    if evaluation_map.get("iou_type") != "bbox":
        raise _error("accuracy.evaluation.iou_type", "schema v1 supports only 'bbox'")
    _int(evaluation_map.get("max_dets"), "accuracy.evaluation.max_dets", minimum=1)
    _number(evaluation_map.get("score_threshold"), "accuracy.evaluation.score_threshold")
    _int(evaluation_map.get("num_select"), "accuracy.evaluation.num_select", minimum=1)
    if evaluation_map.get("limit") is not None:
        _int(evaluation_map["limit"], "accuracy.evaluation.limit", minimum=1)
    if "images_evaluated" in evaluation_map:
        _int(evaluation_map["images_evaluated"], "accuracy.evaluation.images_evaluated")


def _validate_pass(
    pass_entry: Any,
    path: str,
    *,
    expected_device: str | None,
    expected_provider: str | None,
) -> str:
    pass_map = _mapping(pass_entry, path)
    pass_id = _str(pass_map.get("pass_id"), f"{path}.pass_id")
    if pass_id not in PASS_IDS:
        raise _error(f"{path}.pass_id", f"must be one of {list(PASS_IDS)}")
    _str(pass_map.get("producer"), f"{path}.producer")
    origin = _str(pass_map.get("origin"), f"{path}.origin")
    if origin not in PASS_ORIGINS:
        raise _error(f"{path}.origin", f"must be one of {list(PASS_ORIGINS)}")
    status = _str(pass_map.get("status"), f"{path}.status")
    if status not in PASS_STATUSES:
        raise _error(f"{path}.status", f"must be one of {list(PASS_STATUSES)}")
    provenance = _mapping(pass_map.get("provenance"), f"{path}.provenance")
    device = _str(provenance.get("device"), f"{path}.provenance.device")
    if expected_device is not None and device != expected_device:
        raise _error(f"{path}.provenance.device", f"must match {expected_device!r}")
    provider = _str(provenance.get("provider"), f"{path}.provenance.provider")
    if pass_id == "onnx" and expected_provider is not None and provider != expected_provider:
        raise _error(f"{path}.provenance.provider", f"must match {expected_provider!r}")
    if status == "measured" and pass_map.get("predictions") is None and pass_map.get("metrics") is None:
        raise _error(f"{path}", "a measured pass must contain predictions or metrics")
    if status != "measured":
        _str(pass_map.get("reason"), f"{path}.reason")
    if pass_map.get("predictions") is not None:
        validate_predictions_store(pass_map["predictions"], f"{path}.predictions")
    if pass_map.get("metrics") is not None:
        validate_metrics(pass_map["metrics"], f"{path}.metrics")
    if "timing" in pass_map:
        _mapping(pass_map["timing"], f"{path}.timing")
    return pass_id


def validate_accuracy(accuracy: Any) -> None:
    """Validate an accuracy block, including the two-pass device contract."""
    accuracy_map = _mapping(accuracy, "accuracy")
    if accuracy_map.get("scenario") != ACCURACY_SCENARIO:
        raise _error("accuracy.scenario", f"must be {ACCURACY_SCENARIO!r}")
    status = _str(accuracy_map.get("status"), "accuracy.status")
    if status not in ACCURACY_STATUSES:
        raise _error("accuracy.status", f"must be one of {list(ACCURACY_STATUSES)}")
    if "scoring" in accuracy_map:
        scoring = _str(accuracy_map["scoring"], "accuracy.scoring")
        if scoring not in SCORING_MODES:
            raise _error("accuracy.scoring", f"must be one of {list(SCORING_MODES)}")
    if status != "measured":
        _str(accuracy_map.get("reason"), "accuracy.reason")
        if "passes" in accuracy_map:
            _sequence(accuracy_map["passes"], "accuracy.passes")
        return

    device = _validate_device(accuracy_map.get("device"))
    expected_provider = str(_mapping(accuracy_map["device"], "accuracy.device")["ort_provider"])
    _validate_evaluation(accuracy_map.get("evaluation"))
    passes = _sequence(accuracy_map.get("passes"), "accuracy.passes")
    if len(passes) != 2:
        raise _error("accuracy.passes", "must contain exactly the 'python' and 'onnx' passes")
    pass_ids = [
        _validate_pass(
            pass_entry,
            f"accuracy.passes[{index}]",
            expected_device=device,
            expected_provider=expected_provider,
        )
        for index, pass_entry in enumerate(passes)
    ]
    if sorted(pass_ids) != sorted(PASS_IDS):
        raise _error("accuracy.passes", "must contain exactly one 'python' and one 'onnx' pass")
    if "comparisons" in accuracy_map:
        _validate_comparisons(accuracy_map["comparisons"])


def _validate_comparisons(comparisons: Any) -> None:
    for index, comparison in enumerate(_sequence(comparisons, "accuracy.comparisons")):
        path = f"accuracy.comparisons[{index}]"
        comparison_map = _mapping(comparison, path)
        left = _str(comparison_map.get("left_pass"), f"{path}.left_pass")
        right = _str(comparison_map.get("right_pass"), f"{path}.right_pass")
        if left not in PASS_IDS or right not in PASS_IDS or left == right:
            raise _error(path, "must compare two distinct passes from 'python' and 'onnx'")
        status = _str(comparison_map.get("status"), f"{path}.status")
        if status not in COMPARISON_STATUSES:
            raise _error(f"{path}.status", f"must be one of {list(COMPARISON_STATUSES)}")
        if "deltas" in comparison_map:
            deltas = _mapping(comparison_map["deltas"], f"{path}.deltas")
            for key, value in deltas.items():
                _number(value, f"{path}.deltas.{key}")


def validate_protocol(protocol: Any, path: str = "protocol") -> None:
    """Validate a throughput protocol block."""
    protocol_map = _mapping(protocol, path)
    version = _int(protocol_map.get("version"), f"{path}.version", minimum=1)
    if version != BENCHMARK_PROTOCOL_VERSION:
        raise _error(f"{path}.version", f"must be {BENCHMARK_PROTOCOL_VERSION}")
    for key in ("images", "warmup_passes", "passes", "cpu_threads"):
        _int(protocol_map.get(key), f"{path}.{key}")
    for key in ("min_pass_seconds", "max_pass_seconds"):
        value = _number(protocol_map.get(key), f"{path}.{key}")
        if value < 0:
            raise _error(f"{path}.{key}", "must be non-negative")
    timing_scope = _str(protocol_map.get("timing_scope"), f"{path}.timing_scope")
    if timing_scope not in TIMING_SCOPES:
        raise _error(f"{path}.timing_scope", f"must be one of {list(TIMING_SCOPES)}")
    _str(protocol_map.get("protocol_id"), f"{path}.protocol_id")


def _validate_performance_record(record: Any, path: str) -> str:
    record_map = _mapping(record, path)
    record_id = _str(record_map.get("record_id"), f"{path}.record_id")
    status = _str(record_map.get("status"), f"{path}.status")
    if status not in PASS_STATUSES:
        raise _error(f"{path}.status", f"must be one of {list(PASS_STATUSES)}")
    provider = _str(record_map.get("provider"), f"{path}.provider")
    if provider not in PROVIDER_KEYS:
        raise _error(f"{path}.provider", f"must be one of {list(PROVIDER_KEYS)}")
    mode = _str(record_map.get("mode"), f"{path}.mode")
    if mode not in MODES:
        raise _error(f"{path}.mode", f"must be one of {list(MODES)}")
    if status == "measured":
        timing = _mapping(record_map.get("timing"), f"{path}.timing")
        _number(timing.get("images_per_second"), f"{path}.timing.images_per_second")
        _number(timing.get("ms_per_image"), f"{path}.timing.ms_per_image")
        for key, value in timing.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    _number(item, f"{path}.timing.{key}[{index}]")
            else:
                _number(value, f"{path}.timing.{key}")
    else:
        _str(record_map.get("reason"), f"{path}.reason")
    if "comparison_key" in record_map:
        _hash_ref(record_map["comparison_key"], f"{path}.comparison_key")
    return record_id


def validate_performance(performance: Any) -> None:
    """Validate a throughput-only performance block."""
    performance_map = _mapping(performance, "performance")
    scenario = _str(performance_map.get("scenario"), "performance.scenario")
    if scenario not in SCENARIOS:
        raise _error("performance.scenario", f"must be one of {list(SCENARIOS)}")
    validate_protocol(performance_map.get("protocol"))
    records = _sequence(performance_map.get("records"), "performance.records")
    record_ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = _validate_performance_record(record, f"performance.records[{index}]")
        if record_id in record_ids:
            raise _error(f"performance.records[{index}].record_id", "must be unique")
        record_ids.add(record_id)
    unavailable = _sequence(performance_map.get("unavailable"), "performance.unavailable")
    for index, item in enumerate(unavailable):
        item_map = _mapping(item, f"performance.unavailable[{index}]")
        _str(item_map.get("provider"), f"performance.unavailable[{index}].provider")
        _str(item_map.get("reason"), f"performance.unavailable[{index}].reason")
    if not records and not unavailable:
        raise _error("performance", "must contain a timing record or an unavailable entry")


def _validate_model_metadata(model: Any) -> None:
    model_map = _mapping(model, "model")
    _str(model_map.get("id"), "model.id")
    _str(model_map.get("task"), "model.task")
    _str(model_map.get("format"), "model.format")
    _str(model_map.get("precision"), "model.precision")
    artifact = _mapping(model_map.get("artifact"), "model.artifact")
    _filename(artifact.get("filename"), "model.artifact.filename")
    _hash(artifact.get("sha256"), "model.artifact.sha256", SHA256_PATTERN, "sha256")
    if "bytes" in artifact:
        _int(artifact["bytes"], "model.artifact.bytes")
    signature = _mapping(model_map.get("input_signature"), "model.input_signature")
    _hash_ref(model_map.get("input_signature_id"), "model.input_signature_id")
    if "batch_size" in signature:
        _int(signature["batch_size"], "model.input_signature.batch_size", minimum=1)
    if "shapes" in signature:
        for index, shape in enumerate(_sequence(signature["shapes"], "model.input_signature.shapes")):
            for dimension_index, dimension in enumerate(
                _sequence(shape, f"model.input_signature.shapes[{index}]")
            ):
                _int(
                    dimension,
                    f"model.input_signature.shapes[{index}][{dimension_index}]",
                    minimum=1,
                )
    if "metadata" in model_map:
        _mapping(model_map["metadata"], "model.metadata")


def validate_hardware(hardware: Any, path: str = "hardware") -> None:
    """Validate the standard hardware identity shared by all skellybench reports."""
    hardware_map = _mapping(hardware, path)
    _str(hardware_map.get("id"), f"{path}.id")
    _str(hardware_map.get("hardware_id"), f"{path}.hardware_id")
    os_info = _mapping(hardware_map.get("os"), f"{path}.os")
    for key in ("family", "name", "version", "arch"):
        _str(os_info.get(key), f"{path}.os.{key}")
    cpu = _mapping(hardware_map.get("cpu"), f"{path}.cpu")
    _str(cpu.get("model"), f"{path}.cpu.model")
    _str(cpu.get("arch"), f"{path}.cpu.arch")
    for key in ("physical_cores", "logical_cores"):
        if key in cpu and cpu[key] is not None:
            _int(cpu[key], f"{path}.cpu.{key}", minimum=1)
    if "memory" in hardware_map:
        memory = _mapping(hardware_map["memory"], f"{path}.memory")
        _int(memory.get("total_mb"), f"{path}.memory.total_mb", minimum=1)
    gpus = _sequence(hardware_map.get("gpus"), f"{path}.gpus")
    for index, gpu in enumerate(gpus):
        gpu_path = f"{path}.gpus[{index}]"
        gpu_map = _mapping(gpu, gpu_path)
        _int(gpu_map.get("index"), f"{gpu_path}.index")
        _str(gpu_map.get("id"), f"{gpu_path}.id")
        _str(gpu_map.get("name"), f"{gpu_path}.name")
        _str(gpu_map.get("vendor"), f"{gpu_path}.vendor")
        _str(gpu_map.get("driver"), f"{gpu_path}.driver")
        if gpu_map.get("vram_mb") is not None:
            _int(gpu_map["vram_mb"], f"{gpu_path}.vram_mb", minimum=1)


def validate_libraries(libraries: Any, path: str = "libraries") -> None:
    """Validate Python and distribution versions in the standard format."""
    libraries_map = _mapping(libraries, path)
    python = _mapping(libraries_map.get("python"), f"{path}.python")
    _str(python.get("implementation"), f"{path}.python.implementation")
    _str(python.get("version"), f"{path}.python.version")
    packages = _mapping(libraries_map.get("packages"), f"{path}.packages")
    for name, value in packages.items():
        package_name = _str(name, f"{path}.packages key")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", package_name):
            raise _error(
                f"{path}.packages key",
                "must use a lowercase normalized distribution name",
            )
        _str(value, f"{path}.packages.{name}")
    if "cuda" in libraries_map:
        cuda = _mapping(libraries_map["cuda"], f"{path}.cuda")
        for name in ("torch_cuda_build", "cudnn_runtime_version"):
            if name in cuda:
                _str(cuda[name], f"{path}.cuda.{name}")


def _validate_run(run: Any, *, schema_version: int = SKELLYBENCH_SCHEMA_VERSION) -> None:
    run_map = _mapping(run, "run")
    run_id = _str(run_map.get("run_id"), "run.run_id")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise _error("run.run_id", "must be Windows-safe (letters, digits, '.', '_', '-')")
    if schema_version == _LEGACY_SCHEMA_VERSION:
        machine = _mapping(run_map.get("machine"), "run.machine")
        _str(machine.get("id"), "run.machine.id")
        _str(machine.get("hardware_id"), "run.machine.hardware_id")
        software = _mapping(run_map.get("software"), "run.software")
        _str(software.get("python"), "run.software.python")
    else:
        validate_hardware(run_map.get("hardware"), "run.hardware")
        validate_libraries(run_map.get("libraries"), "run.libraries")
    runner = _mapping(run_map.get("runner"), "run.runner")
    _str(runner.get("user"), "run.runner.user")


def _validate_dataset(dataset: Any) -> None:
    dataset_map = _mapping(dataset, "dataset")
    _str(dataset_map.get("name"), "dataset.name")
    annotation = _mapping(dataset_map.get("annotation"), "dataset.annotation")
    _str(annotation.get("path"), "dataset.annotation.path")
    _hash(annotation.get("sha256"), "dataset.annotation.sha256", SHA256_PATTERN, "sha256")
    _int(annotation.get("image_count"), "dataset.annotation.image_count", minimum=1)
    if "images" in dataset_map:
        _mapping(dataset_map["images"], "dataset.images")


def _validate_benchmark_images(benchmark_images: Any) -> None:
    images_map = _mapping(benchmark_images, "benchmark_images")
    _str(images_map.get("dataset"), "benchmark_images.dataset")
    _int(images_map.get("count"), "benchmark_images.count", minimum=1)
    selection = _mapping(images_map.get("selection"), "benchmark_images.selection")
    _str(selection.get("method"), "benchmark_images.selection.method")
    _hash(selection.get("sha256"), "benchmark_images.selection.sha256", SHA256_PATTERN, "sha256")


def validate_input_bundle(bundle: Any) -> None:
    """Validate a complete skellybench input bundle."""
    bundle_map = _mapping(bundle, "bundle")
    version = _int(bundle_map.get("schema_version"), "schema_version", minimum=1)
    if version not in {_LEGACY_SCHEMA_VERSION, SKELLYBENCH_SCHEMA_VERSION}:
        raise _error("schema_version", f"must be {SKELLYBENCH_SCHEMA_VERSION}")
    if bundle_map.get("kind") != INPUT_BUNDLE_KIND:
        raise _error("kind", f"must be {INPUT_BUNDLE_KIND!r}")
    _validate_run(bundle_map.get("run"), schema_version=version)
    if "dataset" in bundle_map:
        _validate_dataset(bundle_map["dataset"])
    if "benchmark_images" in bundle_map:
        _validate_benchmark_images(bundle_map["benchmark_images"])

    models = _sequence(bundle_map.get("models"), "models")
    if not models:
        raise _error("models", "must contain at least one model")
    identities: set[tuple[str, str]] = set()
    any_measurement = False
    for index, entry in enumerate(models):
        path = f"models[{index}]"
        entry_map = _mapping(entry, path)
        _validate_model_metadata(entry_map.get("model"))
        model = entry_map["model"]
        identity = (str(model["id"]), str(model["artifact"]["sha256"]))
        if identity in identities:
            raise _error(path, "duplicate model.id/artifact.sha256 pair")
        identities.add(identity)
        has_accuracy = "accuracy" in entry_map
        has_performance = "performance" in entry_map
        if not has_accuracy and not has_performance:
            raise _error(path, "must contain accuracy or performance")
        any_measurement = True
        if has_accuracy:
            validate_accuracy(entry_map["accuracy"])
        if has_performance:
            validate_performance(entry_map["performance"])
    if not any_measurement:
        raise _error("models", "must contain at least one accuracy or performance section")


def validate_benchmark_report(payload: Any) -> None:
    """Validate a throughput-only benchmark report."""
    payload_map = _mapping(payload, "report")
    version = _int(payload_map.get("schema_version"), "schema_version", minimum=1)
    if version not in {_LEGACY_SCHEMA_VERSION, SKELLYBENCH_SCHEMA_VERSION}:
        raise _error("schema_version", f"must be {SKELLYBENCH_SCHEMA_VERSION}")
    if payload_map.get("kind") != BENCHMARK_REPORT_KIND:
        raise _error("kind", f"must be {BENCHMARK_REPORT_KIND!r}")
    report_id = _str(payload_map.get("report_id"), "report_id")
    if not REPORT_ID_PATTERN.fullmatch(report_id):
        raise _error("report_id", "must be Windows-safe")
    _timestamp(payload_map.get("generated_at"), "generated_at")
    _validate_run(payload_map.get("run"), schema_version=version)
    _validate_model_metadata(payload_map.get("model"))
    if "accuracy" in payload_map:
        raise _error("accuracy", "benchmark reports are throughput-only")
    validate_performance(payload_map.get("performance"))
    if "model_report" in payload_map:
        link = _mapping(payload_map["model_report"], "model_report")
        _str(link.get("report_id"), "model_report.report_id")
        if "filename" in link:
            _filename(link["filename"], "model_report.filename")
        if "sha256" in link:
            _hash(link["sha256"], "model_report.sha256", SHA256_PATTERN, "sha256")
    if "provenance" in payload_map:
        _mapping(payload_map["provenance"], "provenance")


def validate_benchmark_summary(payload: Any) -> None:
    """Validate a run summary document."""
    payload_map = _mapping(payload, "summary")
    version = _int(payload_map.get("schema_version"), "schema_version", minimum=1)
    if version not in {_LEGACY_SCHEMA_VERSION, SKELLYBENCH_SCHEMA_VERSION}:
        raise _error("schema_version", f"must be {SKELLYBENCH_SCHEMA_VERSION}")
    if payload_map.get("kind") != BENCHMARK_SUMMARY_KIND:
        raise _error("kind", f"must be {BENCHMARK_SUMMARY_KIND!r}")
    _str(payload_map.get("run_id"), "run_id")
    _timestamp(payload_map.get("generated_at"), "generated_at")
    _validate_run(payload_map.get("run"), schema_version=version)
    models = _sequence(payload_map.get("models"), "models")
    for index, entry in enumerate(models):
        entry_map = _mapping(entry, f"models[{index}]")
        _str(entry_map.get("model_id"), f"models[{index}].model_id")
        _filename(entry_map.get("report"), f"models[{index}].report")
        if "report_sha256" in entry_map:
            _hash(entry_map["report_sha256"], f"models[{index}].report_sha256", SHA256_PATTERN, "sha256")


def validate_benchmark_overview(payload: Any) -> None:
    """Validate a cross-machine index document."""
    payload_map = _mapping(payload, "overview")
    version = _int(payload_map.get("schema_version"), "schema_version", minimum=1)
    if version not in {_LEGACY_SCHEMA_VERSION, SKELLYBENCH_SCHEMA_VERSION}:
        raise _error("schema_version", f"must be {SKELLYBENCH_SCHEMA_VERSION}")
    if payload_map.get("kind") != BENCHMARK_OVERVIEW_KIND:
        raise _error("kind", f"must be {BENCHMARK_OVERVIEW_KIND!r}")
    _timestamp(payload_map.get("generated_at"), "generated_at")
    rows = _sequence(payload_map.get("rows"), "rows")
    _int(payload_map.get("row_count"), "row_count")
    if payload_map["row_count"] != len(rows):
        raise _error("row_count", "must equal len(rows)")
    for index, row in enumerate(rows):
        row_map = _mapping(row, f"rows[{index}]")
        _str(row_map.get("row_id"), f"rows[{index}].row_id")
        _filename(row_map.get("report"), f"rows[{index}].report")
        _str(row_map.get("provider"), f"rows[{index}].provider")
        _str(row_map.get("mode"), f"rows[{index}].mode")


__all__ = [
    "ACCURACY_SCENARIO",
    "ACCURACY_STATUSES",
    "BENCHMARK_OVERVIEW_KIND",
    "BENCHMARK_PROTOCOL_VERSION",
    "BENCHMARK_REPORT_KIND",
    "BENCHMARK_SUMMARY_KIND",
    "COCO_STAT_KEYS",
    "COMPARISON_STATUSES",
    "INPUT_BUNDLE_KIND",
    "MODES",
    "PASS_IDS",
    "PASS_ORIGINS",
    "PASS_STATUSES",
    "PREDICTION_STORAGES",
    "PROVIDER_KEYS",
    "REPORT_ID_PATTERN",
    "RUN_ID_PATTERN",
    "SCENARIOS",
    "SCORING_MODES",
    "SKELLYBENCH_SCHEMA_VERSION",
    "SkellybenchValidationError",
    "THROUGHPUT_SCENARIO",
    "TIMING_SCOPES",
    "validate_accuracy",
    "validate_benchmark_overview",
    "validate_benchmark_report",
    "validate_benchmark_summary",
    "validate_input_bundle",
    "validate_hardware",
    "validate_libraries",
    "validate_metrics",
    "validate_performance",
    "validate_prediction_record",
    "validate_predictions_store",
    "validate_protocol",
]
