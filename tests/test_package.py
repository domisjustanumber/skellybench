"""Hermetic tests for the standalone skellybench package."""

from __future__ import annotations

import ast
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import skellybench
from skellybench._internal.formatting import format_filesize, format_number
from skellybench._internal.hashing import describe_file, sha256_file, sha256_text
from skellybench._internal.io import atomic_write_json, atomic_write_text, read_json
from skellybench._internal.timestamps import iso_now, utc_now, utc_stamp
from skellybench import (
    ACCURACY_SCENARIO,
    BENCHMARK_OVERVIEW_KIND,
    BENCHMARK_REPORT_KIND,
    BENCHMARK_SUMMARY_KIND,
    INPUT_BUNDLE_KIND,
    PASS_IDS,
    PROVIDER_KEYS,
    SCENARIOS,
    SKELLYBENCH_SCHEMA_VERSION,
    THROUGHPUT_SCENARIO,
    BenchmarkProtocol,
    SkellybenchValidationError,
    benchmark_report_paths,
    build_benchmark_report,
    build_index,
    build_run_summary,
    capabilities,
    comparison_key,
    detect_hardware,
    detect_libraries,
    environment_id,
    index_paths,
    input_signature_id,
    protocol_from_mapping,
    render_benchmark_report_html,
    render_index_html,
    render_run_summary_html,
    score_accuracy,
    score_predictions,
    summary_paths,
    supports,
    validate_benchmark_report,
    validate_input_bundle,
    validate_prediction_record,
    validate_predictions_store,
    write_benchmark_report,
    write_index,
    write_run_summary,
)
from skellybench.schema import COCO_STAT_KEYS, REPORT_ID_PATTERN

STAMP = datetime(2026, 10, 19, 12, 34, 56, tzinfo=UTC)
PACKAGE_DIR = Path(skellybench.__file__).resolve().parent
README_PATH = Path(__file__).resolve().parents[1] / "README.md"
SHA = "a" * 64
SIGNATURE_SHA = "b" * 64
HARDWARE_SHA = "c" * 64
ENV_SHA = "d" * 64


def _metrics(ap: float = 0.5) -> dict[str, float]:
    return {key: ap for key in COCO_STAT_KEYS}


def _machine() -> dict[str, Any]:
    return {
        "id": "sha256:" + "1" * 64,
        "label": "test-machine",
        "hardware_id": "sha256:" + HARDWARE_SHA,
        "os": {
            "family": "windows",
            "name": "Windows 11",
            "version": "10.0.22631",
            "arch": "AMD64",
        },
        "cpu": {
            "vendor": "intel",
            "model": "Test CPU",
            "arch": "AMD64",
            "physical_cores": 8,
            "logical_cores": 16,
        },
        "memory": {"total_mb": 32768},
        "gpus": [
            {
                "index": 0,
                "id": "GPU-123",
                "name": "NVIDIA GeForce RTX 3070",
                "vendor": "nvidia",
                "driver": "580.88",
                "vram_mb": 8192,
            }
        ],
    }


def _libraries() -> dict[str, Any]:
    return {
        "python": {"implementation": "CPython", "version": "3.14.0"},
        "packages": {
            "onnxruntime": "1.31.0",
            "numpy": "2.5.4",
            "faster-coco-eval": "1.8.0",
            "pytest": "9.1.1",
        },
    }


def _run(run_id: str = "run-1") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": "2026-10-19T12:00:00Z",
        "finished_at": "2026-10-19T12:10:00Z",
        "command": {"argv": ["python", "export_all.py"], "display": "python export_all.py"},
        "runner": {"user": "Dom", "label": "Dom", "hostname": "TEST-PC"},
        "hardware": _machine(),
        "libraries": _libraries(),
    }


def _input_signature() -> dict[str, Any]:
    return {
        "batch_size": 1,
        "shapes": [[1, 3, 640, 640]],
        "dtype": "float32",
        "layout": "NCHW",
        "preprocessing_id": "imagenet_rgb_normalized",
    }


def _model_metadata() -> dict[str, Any]:
    signature = _input_signature()
    return {
        "id": "rfdetr-nano",
        "display_name": "RF-DETR Nano",
        "task": "object-detection",
        "format": "onnx",
        "precision": "fp32",
        "artifact": {
            "filename": "rfdetr-nano_b1_fp32.onnx",
            "sha256": SHA,
            "bytes": 1024,
        },
        "input_signature": signature,
        "input_signature_id": "sha256:" + input_signature_id(signature),
        "metadata": {"resolution": 640},
    }


def _device() -> dict[str, Any]:
    return {
        "type": "cuda",
        "index": 0,
        "torch_device": "cuda:0",
        "ort_provider": "CUDAExecutionProvider",
        "matches_internal": True,
        "selection": "source_library_internal_benchmark",
        "internal_benchmark": {
            "tool": "rfdetr.export.benchmark.main",
            "device": "cuda:0",
            "provider": "CUDAExecutionProvider",
            "selection": "hardcoded",
        },
    }


def _accuracy(*, with_predictions: bool = False, ap: float = 0.5) -> dict[str, Any]:
    passes = []
    for pass_id, origin, provider in (
        ("python", "python_model", "torch-cuda"),
        ("onnx", "onnx_model", "CUDAExecutionProvider"),
    ):
        entry: dict[str, Any] = {
            "pass_id": pass_id,
            "producer": "rf_detr" if pass_id == "python" else "onnx_runtime",
            "origin": origin,
            "status": "measured",
            "provenance": {"device": "cuda:0", "provider": provider},
        }
        if with_predictions:
            entry["predictions"] = {
                "format": "coco_detection_results",
                "storage": "inline",
                "records": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}],
            }
        else:
            entry["metrics"] = {
                "tool": "faster_coco_eval",
                "version": "1.8.0",
                "values": _metrics(ap),
            }
        passes.append(entry)
    return {
        "scenario": ACCURACY_SCENARIO,
        "status": "measured",
        "device": _device(),
        "evaluation": {
            "iou_type": "bbox",
            "max_dets": 100,
            "score_threshold": 0.0,
            "num_select": 300,
        },
        "passes": passes,
    }


def _performance() -> dict[str, Any]:
    protocol = BenchmarkProtocol().payload(signature_id=input_signature_id(_input_signature()))
    key = comparison_key(
        artifact_sha256=SHA,
        hardware_id="sha256:" + HARDWARE_SHA,
        environment_id="sha256:" + ENV_SHA,
        protocol_id=protocol["protocol_id"],
        provider="cuda",
        mode="host",
    )
    return {
        "scenario": THROUGHPUT_SCENARIO,
        "protocol": protocol,
        "records": [
            {
                "record_id": "cuda/host/GPU-123",
                "status": "measured",
                "provider": "cuda",
                "provider_label": "CUDA",
                "ep_name": "CUDAExecutionProvider",
                "ep_package": {"name": "onnxruntime-ep-cuda13", "version": "0.3.0"},
                "mode": "host",
                "device": {
                    "type": "gpu",
                    "index": 0,
                    "id": "GPU-123",
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": "nvidia",
                    "driver": "580.88",
                    "vram_mb": 8192,
                },
                "provider_options": {"device_id": 0},
                "timing": {
                    "images_per_second": 200.0,
                    "ms_per_image": 5.0,
                    "batch_ms_p50": 5.0,
                    "batch_ms_p95": 6.0,
                    "cv": 0.01,
                    "pass_throughputs": [199.0, 200.0, 201.0],
                    "pass_ms": [1000.0, 995.0, 990.0],
                },
                "session": {"create_ms": 10.0, "first_inference_ms": 20.0},
                "comparison_key": key,
                "hardware_id": "sha256:" + HARDWARE_SHA,
                "environment_id": "sha256:" + ENV_SHA,
            }
        ],
        "unavailable": [],
    }


def _dataset(tmp_path: Path | None = None) -> dict[str, Any]:
    path = (tmp_path or Path(".")) / "instances_val2017.json"
    return {
        "name": "coco_val2017",
        "split": "val2017",
        "annotation": {
            "path": str(path),
            "sha256": SHA,
            "image_count": 1,
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
    }


def _benchmark_images() -> dict[str, Any]:
    return {
        "dataset": "coco_val2017",
        "count": 1,
        "selection": {"method": "stride", "sha256": SIGNATURE_SHA},
    }


def _bundle(tmp_path: Path | None = None, **model_entry: Any) -> dict[str, Any]:
    entry = {"model": _model_metadata(), "accuracy": _accuracy(), "performance": _performance()}
    entry.update(model_entry)
    return {
        "schema_version": SKELLYBENCH_SCHEMA_VERSION,
        "kind": INPUT_BUNDLE_KIND,
        "run": _run(),
        "dataset": _dataset(tmp_path),
        "benchmark_images": _benchmark_images(),
        "models": [entry],
    }


def _annotation(tmp_path: Path) -> Path:
    path = tmp_path / "instances_val2017.json"
    path.write_text(
        json.dumps(
            {
                "info": {"description": "skellybench test"},
                "licenses": [],
                "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _benchmark_report(tmp_path: Path) -> dict[str, Any]:
    return build_benchmark_report(
        run=_run(),
        model=_model_metadata(),
        performance=_performance(),
        model_report={"report_id": "rfdetr-nano_b1_fp32.model", "filename": "rfdetr-nano_b1_fp32.model.report.json"},
        generated_at=STAMP,
    )


# --------------------------------------------------------------------------- capabilities / schema


def test_capabilities_contract_matches_schema() -> None:
    caps = capabilities()
    assert caps["accuracy"]["passes"] == list(PASS_IDS)
    assert caps["performance"]["providers"] == list(PROVIDER_KEYS)
    assert caps["performance"]["scenarios"] == list(SCENARIOS)
    assert caps["metrics"]["metric_keys"] == list(COCO_STAT_KEYS)
    assert caps["environment"]["hardware_fields"] == ["id", "hardware_id", "os", "cpu", "gpus"]
    assert "packages" in caps["environment"]["library_fields"]
    assert caps["reports"]["writes_model_reports"] is True
    assert caps["reports"]["throughput_only_benchmark_reports"] is True
    assert caps["accuracy"]["scoring_requires_annotation"] is True
    assert caps["accuracy"]["supports_precomputed_metrics"] is True
    assert caps["accuracy"]["requires_internal_device_match"] is True


def test_capability_probe_and_readme_are_consistent() -> None:
    caps = capabilities()
    readme = README_PATH.read_text(encoding="utf-8")
    for value in (
        *caps["accuracy"]["passes"],
        *caps["performance"]["providers"],
        *caps["performance"]["modes"],
        *caps["performance"]["timing_scopes"],
        *caps["predictions"]["storages"],
        *caps["report_kinds"],
    ):
        assert value in readme, f"{value!r} missing from README"
    assert supports("accuracy.supports_precomputed_metrics")
    assert supports("reports.writes_model_reports")
    assert not supports("missing.capability")


def test_protocol_id_includes_scenario_and_signature() -> None:
    throughput = BenchmarkProtocol()
    accuracy = BenchmarkProtocol(scenario=ACCURACY_SCENARIO)
    assert throughput.protocol_id(signature_id="sig") == throughput.protocol_id(signature_id="sig")
    assert throughput.protocol_id(signature_id="sig") != throughput.protocol_id(signature_id="other")
    assert throughput.protocol_id(signature_id="sig") != accuracy.protocol_id(signature_id="sig")
    assert throughput.payload(signature_id="sig")["protocol_id"] == throughput.protocol_id(signature_id="sig")
    assert protocol_from_mapping(throughput.payload(signature_id="sig")).images == 120


def test_input_bundle_validates() -> None:
    validate_input_bundle(_bundle())


def test_input_bundle_reads_legacy_machine_and_software_fields() -> None:
    bundle = _bundle()
    bundle["schema_version"] = 1
    bundle["run"]["machine"] = {
        "id": "legacy-machine",
        "hardware_id": "legacy-hardware",
    }
    bundle["run"]["software"] = {"python": "3.14.0"}
    del bundle["run"]["hardware"]
    del bundle["run"]["libraries"]
    validate_input_bundle(bundle)


def test_input_bundle_requires_exactly_two_accuracy_passes() -> None:
    bundle = _bundle()
    bundle["models"][0]["accuracy"]["passes"] = bundle["models"][0]["accuracy"]["passes"][:1]
    with pytest.raises(SkellybenchValidationError, match="exactly"):
        validate_input_bundle(bundle)


def test_input_bundle_requires_internal_device_match() -> None:
    bundle = _bundle()
    bundle["models"][0]["accuracy"]["device"]["matches_internal"] = False
    with pytest.raises(SkellybenchValidationError, match="matches_internal"):
        validate_input_bundle(bundle)


def test_input_bundle_requires_onnx_pass_provider_match() -> None:
    bundle = _bundle()
    bundle["models"][0]["accuracy"]["passes"][1]["provenance"]["provider"] = "CPUExecutionProvider"
    with pytest.raises(SkellybenchValidationError, match="provider"):
        validate_input_bundle(bundle)


def test_input_bundle_rejects_duplicate_models() -> None:
    bundle = _bundle()
    bundle["models"].append(copy.deepcopy(bundle["models"][0]))
    with pytest.raises(SkellybenchValidationError, match="duplicate"):
        validate_input_bundle(bundle)


def test_input_bundle_requires_measurement() -> None:
    bundle = _bundle()
    del bundle["models"][0]["accuracy"]
    del bundle["models"][0]["performance"]
    with pytest.raises(SkellybenchValidationError, match="accuracy or performance"):
        validate_input_bundle(bundle)


def test_prediction_store_requires_one_storage_form() -> None:
    validate_predictions_store(
        {
            "format": "coco_detection_results",
            "storage": "inline",
            "records": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
        }
    )
    with pytest.raises(SkellybenchValidationError, match="must not also"):
        validate_predictions_store(
            {
                "format": "coco_detection_results",
                "storage": "inline",
                "path": "x.json",
                "records": [],
            }
        )
    with pytest.raises(SkellybenchValidationError, match="must not also"):
        validate_predictions_store(
            {
                "format": "coco_detection_results",
                "storage": "json",
                "path": "x.json",
                "sha256": SHA,
                "record_count": 1,
                "records": [],
            }
        )


@pytest.mark.parametrize(
    "record",
    [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, -1, 1], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": float("nan")},
        {"image_id": "1", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5},
    ],
)
def test_prediction_record_validation(record: dict[str, Any]) -> None:
    with pytest.raises(SkellybenchValidationError):
        validate_prediction_record(record)


# --------------------------------------------------------------------------- scoring


def test_score_precomputed_metrics_without_annotation() -> None:
    scored = score_accuracy(_accuracy())
    assert scored["status"] == "measured"
    assert scored["scoring"] == "precomputed"
    assert scored["comparisons"][0]["status"] == "ok"
    assert scored["comparisons"][0]["deltas"]["delta_ap"] == 0.0


def test_score_predictions_without_annotation_is_unavailable_not_fabricated() -> None:
    scored = score_accuracy(_accuracy(with_predictions=True))
    assert scored["status"] == "unavailable"
    assert scored["scoring"] == "unavailable"
    assert "annotation" in scored["reason"]
    assert all(pass_entry["status"] == "unavailable" for pass_entry in scored["passes"])


def test_score_predictions_with_annotation_recomputes_and_compares(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    scored = score_accuracy(_accuracy(with_predictions=True), annotation_path=annotation)
    assert scored["status"] == "measured"
    assert scored["scoring"] == "recomputed"
    assert scored["evaluation"]["images_evaluated"] == 1
    assert all(pass_entry["metrics"]["values"]["ap"] == pytest.approx(1.0) for pass_entry in scored["passes"])
    assert scored["comparisons"][0]["deltas"]["delta_ap"] == pytest.approx(0.0)


def test_score_accuracy_restricts_evaluation_to_requested_images(tmp_path: Path) -> None:
    annotation = tmp_path / "two_images.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        ),
        encoding="utf-8",
    )
    scored = score_accuracy(
        _accuracy(with_predictions=True),
        annotation_path=annotation,
        image_ids=[1],
    )
    assert scored["status"] == "measured"
    assert scored["evaluation"]["images_evaluated"] == 1
    assert scored["passes"][0]["metrics"]["values"]["ap"] == pytest.approx(1.0)



def test_score_predictions_from_external_store_strips_local_path(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    records = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    store_path = tmp_path / "predictions.json"
    store_path.write_text(json.dumps(records), encoding="utf-8")
    accuracy = _accuracy()
    for pass_entry in accuracy["passes"]:
        pass_entry.pop("metrics")
        pass_entry["predictions"] = {
            "format": "coco_detection_results",
            "storage": "json",
            "path": str(store_path),
            "sha256": sha256_file(store_path),
            "record_count": 1,
        }
    scored = score_accuracy(accuracy, annotation_path=annotation, base_dir=tmp_path)
    assert scored["status"] == "measured"
    for pass_entry in scored["passes"]:
        assert "path" not in pass_entry["predictions"]
        assert pass_entry["predictions"]["filename"] == "predictions.json"


def test_score_precomputed_and_predictions_records_delta(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    accuracy = _accuracy()
    for pass_entry in accuracy["passes"]:
        pass_entry["predictions"] = {
            "format": "coco_detection_results",
            "storage": "inline",
            "records": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}],
        }
    scored = score_accuracy(accuracy, annotation_path=annotation)
    assert scored["scoring"] == "recomputed"
    assert "precomputed_delta" in scored["passes"][0]["metrics"]
    assert scored["passes"][0]["metrics"]["precomputed_delta"]["ap"] == pytest.approx(0.5)


def test_score_predictions_rejects_sha_mismatch(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    store_path = tmp_path / "predictions.json"
    store_path.write_text("[]", encoding="utf-8")
    accuracy = _accuracy()
    for pass_entry in accuracy["passes"]:
        pass_entry.pop("metrics")
        pass_entry["predictions"] = {
            "format": "coco_detection_results",
            "storage": "json",
            "path": str(store_path),
            "sha256": "e" * 64,
            "record_count": 0,
        }
    with pytest.raises(Exception, match="sha256 mismatch"):
        score_accuracy(accuracy, annotation_path=annotation)


def test_score_predictions_rejects_non_100_max_dets(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    with pytest.raises(Exception, match="max_dets=100"):
        score_predictions([], annotation, max_dets=50)


# --------------------------------------------------------------------------- reports


def test_benchmark_report_paths_and_write(tmp_path: Path) -> None:
    report = _benchmark_report(tmp_path)
    validate_benchmark_report(report)
    json_path, html_path = benchmark_report_paths(tmp_path, "rfdetr-nano_b1_fp32.onnx", "run-1")
    assert json_path.name == "rfdetr-nano_b1_fp32.onnx.benchmark-run-1.json"
    assert html_path.name == "rfdetr-nano_b1_fp32.onnx.benchmark-run-1.html"
    write_benchmark_report((json_path, html_path), report)
    assert json_path.is_file() and html_path.is_file()
    assert "Throughput records" in html_path.read_text(encoding="utf-8")
    assert "accuracy" not in json.loads(json_path.read_text(encoding="utf-8"))


def test_benchmark_report_rejects_accuracy() -> None:
    report = _benchmark_report(Path("."))
    report["accuracy"] = _accuracy()
    with pytest.raises(SkellybenchValidationError, match="throughput-only"):
        validate_benchmark_report(report)


def test_legacy_benchmark_report_keeps_environment_in_rendered_html() -> None:
    report = _benchmark_report(Path("."))
    report["schema_version"] = 1
    run = report["run"]
    run["machine"] = run.pop("hardware")
    run["software"] = {"python": "3.14.0", "onnxruntime": "1.31.0"}
    del run["libraries"]
    validate_benchmark_report(report)
    rendered = render_benchmark_report_html(report)
    assert "test-machine" in rendered
    assert "1.31.0" in rendered


def test_report_html_escapes_model_and_machine_values() -> None:
    report = _benchmark_report(Path("."))
    report["model"]["id"] = "<script>alert(1)</script>"
    rendered = render_benchmark_report_html(report)
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_run_summary_and_write(tmp_path: Path) -> None:
    report = _benchmark_report(tmp_path)
    summary = build_run_summary(run=_run(), reports=[report], generated_at=STAMP)
    assert summary["kind"] == BENCHMARK_SUMMARY_KIND
    assert summary["run_id"] == "run-1"
    assert summary["models"][0]["summary"]["images_per_second"] == 200.0
    assert render_run_summary_html(summary)

    json_path, html_path = benchmark_report_paths(tmp_path, "rfdetr-nano_b1_fp32.onnx", "run-1")
    write_benchmark_report((json_path, html_path), report)
    summary_json, summary_html = write_run_summary(
        tmp_path, run=_run(), report_files=[json_path], generated_at=STAMP
    )
    assert summary_json.is_file() and summary_html.is_file()
    assert summary_json.name == "benchmark-summary-run-1.json"


def test_summary_paths(tmp_path: Path) -> None:
    json_path, html_path = summary_paths(tmp_path, "run-1")
    assert json_path.name == "benchmark-summary-run-1.json"
    assert html_path.name == "benchmark-summary-run-1.html"


# --------------------------------------------------------------------------- index


def test_index_from_benchmark_report(tmp_path: Path) -> None:
    report = _benchmark_report(tmp_path)
    json_path, html_path = benchmark_report_paths(tmp_path, "rfdetr-nano_b1_fp32.onnx", "run-1")
    write_benchmark_report((json_path, html_path), report)

    index = build_index(tmp_path, generated_at=STAMP)
    assert index["kind"] == BENCHMARK_OVERVIEW_KIND
    assert index["row_count"] == 1
    row = index["rows"][0]
    assert row["provider"] == "cuda"
    assert row["mode"] == "host"
    assert row["images_per_second"] == 200.0
    assert row["comparison_key"].startswith("sha256:")
    assert render_index_html(index)
    assert build_index(tmp_path, generated_at=STAMP)["row_count"] == 1


def test_index_rebuilds_from_model_report_snapshot(tmp_path: Path) -> None:
    report = _benchmark_report(tmp_path)
    model_report = {
        "schema_version": 1,
        "kind": "model_report",
        "report_id": "rfdetr-nano_b1_fp32.model",
        "generated_at": STAMP.isoformat(),
        "scoring": {
            "performance": {
                "benchmarks": [
                    {
                        "run_id": "run-1",
                        "scenario": THROUGHPUT_SCENARIO,
                        "report_id": report["report_id"],
                        "filename": "rfdetr-nano_b1_fp32.onnx.benchmark-run-1.json",
                        "sha256": SHA,
                        "snapshot": report,
                    }
                ]
            }
        },
    }
    (tmp_path / "rfdetr-nano_b1_fp32.model.report.json").write_text(
        json.dumps(model_report), encoding="utf-8"
    )
    index = build_index(tmp_path, generated_at=STAMP)
    assert index["row_count"] == 1
    assert index["rows"][0]["report"] == "rfdetr-nano_b1_fp32.onnx.benchmark-run-1.json"


def test_write_index(tmp_path: Path) -> None:
    report = _benchmark_report(tmp_path)
    json_path, html_path = benchmark_report_paths(tmp_path, "rfdetr-nano_b1_fp32.onnx", "run-1")
    write_benchmark_report((json_path, html_path), report)
    index_json, index_html = write_index(tmp_path, generated_at=STAMP)
    assert index_json.is_file() and index_html.is_file()
    assert index_json.name == "benchmark-overview-20261019T123456Z.json"


# --------------------------------------------------------------------------- hardware and boundaries


def test_hardware_helpers_are_optional_and_stable() -> None:
    hardware = detect_hardware()
    assert hardware["id"].startswith("sha256:")
    assert hardware["hardware_id"].startswith("sha256:")
    libraries = detect_libraries(("pytest", "definitely-not-installed-xyz"))
    assert libraries["python"]["implementation"]
    assert libraries["python"]["version"]
    assert libraries["packages"]["pytest"] != "not installed"
    assert libraries["packages"]["definitely-not-installed-xyz"] == "not installed"
    assert environment_id(hardware, libraries).startswith("sha256:")


def test_input_bundle_rejects_unstructured_library_versions() -> None:
    bundle = _bundle()
    bundle["run"]["libraries"]["python"] = "3.14.0"
    with pytest.raises(SkellybenchValidationError, match="run.libraries.python"):
        validate_input_bundle(bundle)


def test_input_bundle_requires_normalized_package_names() -> None:
    bundle = _bundle()
    bundle["run"]["libraries"]["packages"]["Fast_Coco"] = "1.0"
    with pytest.raises(SkellybenchValidationError, match="normalized distribution name"):
        validate_input_bundle(bundle)


def test_skellybench_does_not_import_frameworks() -> None:
    banned = {"report_core", "rf_detr", "onnx_runtime", "torch", "onnxruntime"}
    sources = sorted(PACKAGE_DIR.rglob("*.py"))
    assert len(sources) >= 20, f"import scan found only {len(sources)} files"
    for source_file in sources:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & banned), f"{source_file}: forbidden imports {sorted(imported & banned)}"


def test_scoring_dependency_is_not_imported_at_module_scope() -> None:
    tree = ast.parse((PACKAGE_DIR / "scoring" / "coco.py").read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "faster_coco_eval" not in top_level


# --------------------------------------------------------------------------- internal helpers


def test_internal_atomic_writes_round_trip_and_clean_temp_files(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "report.txt"
    assert atomic_write_text(text_path, "hello\n") == text_path
    assert text_path.read_text(encoding="utf-8") == "hello\n"
    assert not list(text_path.parent.glob("*.tmp"))

    json_path = tmp_path / "report.json"
    payload = {"schema_version": 2, "values": [1, 2, 3]}
    atomic_write_json(json_path, payload)
    assert read_json(json_path) == payload


def test_internal_read_json_treats_missing_or_invalid_as_absent(tmp_path: Path) -> None:
    assert read_json(tmp_path / "missing.json") is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert read_json(invalid) is None


def test_internal_describe_file_hides_local_path(tmp_path: Path) -> None:
    artifact = tmp_path / "rfdetr-nano_b1_fp32.onnx"
    artifact.write_bytes(b"onnx")
    assert describe_file(artifact) == {
        "filename": "rfdetr-nano_b1_fp32.onnx",
        "size_bytes": 4,
        "filesize_human": "4 B",
        "sha256": sha256_file(artifact),
    }


def test_internal_timestamps_are_utc_and_windows_safe() -> None:
    assert utc_now().tzinfo is not None
    assert utc_stamp(STAMP) == "20261019T123456Z"
    assert ":" not in utc_stamp(STAMP)
    assert iso_now().endswith("+00:00")


def test_internal_format_helpers() -> None:
    assert format_filesize(None) == "n/a"
    assert format_filesize(1536) == "1.5 KB"
    assert format_number(1.2345, 2) == "1.23"
    assert format_number(None) == "n/a"


def test_model_report_contract_ships_with_the_package() -> None:
    from importlib.resources import files

    contract = files("skellybench.model_reports").joinpath("MODEL_REPORT_CONTRACT.md")
    assert contract.is_file()
    assert "model report" in contract.read_text(encoding="utf-8").lower()


def test_report_ids_are_windows_safe() -> None:
    assert REPORT_ID_PATTERN.fullmatch("rfdetr-nano_b1_fp32.onnx.benchmark-run-1")
    assert not REPORT_ID_PATTERN.fullmatch("bad:run")


def test_index_paths_unique_per_stamp(tmp_path: Path) -> None:
    first = index_paths(tmp_path, STAMP)
    second = index_paths(tmp_path, datetime(2026, 10, 19, 12, 34, 57, tzinfo=UTC))
    assert first != second
    assert first[0].name == "benchmark-overview-20261019T123456Z.json"
    assert sha256_text("skellybench") != sha256_text("other")
