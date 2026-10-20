"""Hermetic tests for model reports and the model-report overview."""

from __future__ import annotations

import ast
import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import skellybench
from skellybench import (
    ACCURACY_SCENARIO,
    MODEL_OVERVIEW_KIND,
    MODEL_REPORT_KIND,
    MODEL_REPORT_SCHEMA_VERSION,
    THROUGHPUT_SCENARIO,
    ModelReportIdentity,
    ModelReportValidationError,
    build_model_overview,
    build_model_report,
    check_model_report,
    ensure_model_report,
    merge_model_report,
    model_overview_paths,
    model_report_paths,
    read_model_report,
    render_model_report_file,
    rewrite_model_report_html,
    validate_model_report,
    write_model_overview,
    write_model_report,
)
from skellybench.html_reports.model import render_model_overview_html, render_model_report_html

STAMP = datetime(2026, 10, 19, 12, 34, 56, tzinfo=UTC)
PACKAGE_DIR = Path(skellybench.__file__).resolve().parent
SHA = "a" * 64
SOURCE_SHA = "c" * 64
MD5 = "b" * 32
REPORT_ID = "rfdetr-nano_b1_fp32"


def _metrics(ap: float) -> dict[str, float]:
    return {
        "ap": ap,
        "ap50": ap + 0.1,
        "ap75": ap - 0.05,
        "ar100": ap + 0.2,
    }


def _model(
    *,
    runtime: str = "onnx",
    precision: str = "fp32",
    batch_size: int = 1,
    model_id: str = "rfdetr-nano",
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "runtime": runtime,
        "precision": precision,
        "batch_size": batch_size,
        "input_signature": {"shape": [batch_size, 3, 640, 640], "dtype": "float32"},
        "artifact": artifact
        if artifact is not None
        else {
            "filename": "rfdetr-nano_b1_fp32.onnx",
            "sha256": SHA,
            "size_bytes": 1024,
            "filesize_human": "1.0 KB",
            "source_fp32_sha256": SOURCE_SHA,
        },
        "metadata": {"family": "rfdetr"},
    }


def _provenance(*, source_kind: str = "checkpoint") -> dict[str, Any]:
    return {
        "origin": "onnx_model",
        "source": {
            "kind": source_kind,
            "url": "https://storage.googleapis.com/rfdetr/rf-detr-nano.pth",
            "md5": MD5,
        },
        "tools": {"rfdetr": "1.11.2", "onnx": "1.23.2"},
        "command": {"display": "python export_all.py --sizes nano"},
        "config": {"batch_size": 1},
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


def _accuracy(ap: float = 0.5) -> dict[str, Any]:
    return {
        "scenario": ACCURACY_SCENARIO,
        "status": "measured",
        "device": _device(),
        "evaluation": {
            "iou_type": "bbox",
            "max_dets": 100,
            "score_threshold": 0.0,
            "num_select": 300,
            "images_evaluated": 5000,
        },
        "passes": [
            {
                "pass_id": "python",
                "producer": "rf_detr",
                "origin": "python_model",
                "status": "measured",
                "metrics": {
                    "tool": "faster_coco_eval",
                    "version": "1.8.0",
                    "values": _metrics(ap),
                },
                "provenance": {"device": "cuda:0", "provider": "torch-cuda"},
            },
            {
                "pass_id": "onnx",
                "producer": "onnx_runtime",
                "origin": "onnx_model",
                "status": "measured",
                "metrics": {
                    "tool": "faster_coco_eval",
                    "version": "1.8.0",
                    "values": _metrics(ap),
                },
                "provenance": {"device": "cuda:0", "provider": "CUDAExecutionProvider"},
            },
        ],
        "comparisons": [
            {
                "left_pass": "python",
                "right_pass": "onnx",
                "status": "ok",
                "deltas": {"delta_ap": 0.0, "delta_ap50": 0.0},
                "thresholds": {"warn_ap": 0.001, "fail_ap": 0.01},
            }
        ],
    }


def _performance(run_id: str = "run-1") -> dict[str, Any]:
    return {
        "benchmarks": [
            {
                "run_id": run_id,
                "scenario": THROUGHPUT_SCENARIO,
                "report_id": f"{REPORT_ID}.benchmark-{run_id}",
                "filename": f"{REPORT_ID}.benchmark-{run_id}.json",
                "sha256": SHA,
                "snapshot": {
                    "schema_version": 1,
                    "kind": "benchmark_report",
                    "scenario": THROUGHPUT_SCENARIO,
                    "performance": {
                        "records": [
                            {
                                "provider": "cuda",
                                "mode": "host",
                                "timing": {"images_per_second": 200.0},
                            }
                        ]
                    },
                },
            }
        ]
    }


def _environment() -> dict[str, Any]:
    return {
        "hardware": {
            "id": "sha256:" + "1" * 64,
            "label": "test-machine",
            "hardware_id": "sha256:" + "2" * 64,
            "os": {
                "family": "windows",
                "name": "Windows 11",
                "version": "10.0",
                "arch": "AMD64",
            },
            "cpu": {"model": "Test CPU", "arch": "AMD64", "logical_cores": 8},
            "gpus": [],
        },
        "libraries": {
            "python": {"implementation": "CPython", "version": "3.14.0"},
            "packages": {"pytest": "9.1.1"},
        },
    }


def _payload(
    *,
    report_id: str = REPORT_ID,
    accuracy: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_model_report(
        report_id=report_id,
        model=model or _model(),
        provenance=provenance or _provenance(),
        accuracy=accuracy if accuracy is not None else _accuracy(),
        performance=performance if performance is not None else _performance(),
        environment=_environment(),
        generated_at=STAMP,
    )


def _artifact(tmp_path: Path, name: str = "rfdetr-nano_b1_fp32.onnx") -> Path:
    path = tmp_path / "exports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"onnx")
    return path


# --------------------------------------------------------------------------- schema


def test_build_model_report_is_valid_and_contains_both_sections() -> None:
    payload = _payload()
    validate_model_report(payload)
    assert payload["kind"] == MODEL_REPORT_KIND
    assert payload["schema_version"] == MODEL_REPORT_SCHEMA_VERSION
    assert payload["report_id"] == REPORT_ID
    assert payload["scoring"]["accuracy"]["status"] == "measured"
    assert payload["scoring"]["performance"]["benchmarks"][0]["run_id"] == "run-1"
    assert set(payload["environment"]) == {"hardware", "libraries"}
    assert payload["environment"]["libraries"]["python"]["version"] == "3.14.0"


def test_model_report_reads_legacy_environment() -> None:
    payload = _payload()
    payload["schema_version"] = 1
    payload["environment"] = {
        "python": "3.14.0",
        "platform": "test",
        "packages": {"pytest": "9.1.1"},
    }
    validate_model_report(payload)


def test_build_without_accuracy_is_valid_but_not_measured() -> None:
    payload = build_model_report(
        report_id="rfdetr-nano_b1_source",
        model=_model(runtime="rfdetr", precision="source", artifact={"filename": "rf-detr-nano.pth"}),
        provenance=_provenance(),
        environment=_environment(),
        generated_at=STAMP,
    )
    assert "accuracy" not in payload["scoring"]
    assert payload["scoring"]["performance"] == {"benchmarks": []}


def test_measured_accuracy_requires_exactly_two_passes() -> None:
    accuracy = _accuracy()
    accuracy["passes"] = accuracy["passes"][:1]
    with pytest.raises(ModelReportValidationError, match="exactly"):
        _payload(accuracy=accuracy)


def test_measured_accuracy_requires_a_comparison() -> None:
    accuracy = _accuracy()
    accuracy["comparisons"] = []
    with pytest.raises(ModelReportValidationError, match="comparison"):
        _payload(accuracy=accuracy)


def test_accuracy_device_must_match_internal_benchmark() -> None:
    accuracy = _accuracy()
    accuracy["device"]["matches_internal"] = False
    with pytest.raises(ModelReportValidationError, match="matches_internal"):
        _payload(accuracy=accuracy)


def test_pass_device_must_match_accuracy_device() -> None:
    accuracy = _accuracy()
    accuracy["passes"][0]["provenance"]["device"] = "cpu"
    with pytest.raises(ModelReportValidationError, match="torch_device"):
        _payload(accuracy=accuracy)


def test_onnx_pass_provider_must_match_accuracy_provider() -> None:
    accuracy = _accuracy()
    accuracy["passes"][1]["provenance"]["provider"] = "CPUExecutionProvider"
    with pytest.raises(ModelReportValidationError, match="ort_provider"):
        _payload(accuracy=accuracy)


def test_measured_pass_requires_metrics_or_predictions() -> None:
    accuracy = _accuracy()
    del accuracy["passes"][0]["metrics"]
    with pytest.raises(ModelReportValidationError, match="metrics"):
        _payload(accuracy=accuracy)


def test_unavailable_accuracy_requires_reason_and_no_passes() -> None:
    payload = _payload(
        accuracy={
            "scenario": ACCURACY_SCENARIO,
            "status": "unavailable",
            "reason": "CUDAExecutionProvider is not available",
        }
    )
    assert payload["scoring"]["accuracy"]["status"] == "unavailable"


def test_unavailable_accuracy_without_reason_is_rejected() -> None:
    with pytest.raises(ModelReportValidationError, match="reason"):
        _payload(accuracy={"scenario": ACCURACY_SCENARIO, "status": "unavailable"})


def test_checkpoint_provenance_rejects_local_paths() -> None:
    provenance = _provenance()
    provenance["source"]["path"] = "model_cache/rf-detr-nano.pth"
    with pytest.raises(ModelReportValidationError, match="local checkpoint paths"):
        _payload(provenance=provenance)


def test_checkpoint_provenance_requires_url_and_md5() -> None:
    provenance = _provenance()
    del provenance["source"]["md5"]
    with pytest.raises(ModelReportValidationError, match="md5"):
        _payload(provenance=provenance)


def test_benchmark_snapshot_must_not_embed_raw_predictions() -> None:
    performance = _performance()
    performance["benchmarks"][0]["snapshot"]["predictions"] = [{"image_id": 1}]
    with pytest.raises(ModelReportValidationError, match="raw predictions"):
        _payload(performance=performance)


def test_benchmark_snapshot_must_not_contain_accuracy() -> None:
    performance = _performance()
    performance["benchmarks"][0]["snapshot"]["accuracy"] = {"status": "measured"}
    with pytest.raises(ModelReportValidationError, match="throughput-only"):
        _payload(performance=performance)


def test_benchmark_scenario_must_match_snapshot() -> None:
    performance = _performance()
    performance["benchmarks"][0]["scenario"] = ACCURACY_SCENARIO
    with pytest.raises(ModelReportValidationError, match="snapshot scenario"):
        _payload(performance=performance)


# --------------------------------------------------------------------------- paths and HTML


def test_model_report_paths_are_siblings_and_strip_existing_suffix(tmp_path: Path) -> None:
    paths = model_report_paths(tmp_path / "rfdetr-nano_b1_fp32.onnx")
    assert paths.json_path.name == "rfdetr-nano_b1_fp32.onnx.model.report.json"
    assert paths.html_path.name == "rfdetr-nano_b1_fp32.onnx.model.report.html"
    assert model_report_paths(paths.json_path) == paths
    assert model_report_paths(paths.html_path) == paths


def test_render_report_html_escapes_values() -> None:
    payload = _payload(model=_model(model_id="<script>alert(1)</script>"))
    rendered = render_model_report_html(payload)
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered


# --------------------------------------------------------------------------- writer and gate


def test_write_report_writes_json_and_html_then_skips(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    payload = _payload()

    first = write_model_report(paths, payload)
    assert first.written and not first.skipped
    assert paths.json_path.is_file() and paths.html_path.is_file()
    assert read_model_report(paths) == payload

    second = write_model_report(paths, payload)
    assert second.skipped and not second.written
    assert second.reason == "complete"


def test_force_rewrites_complete_report(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    payload = _payload()
    write_model_report(paths, payload)

    forced = write_model_report(paths, payload, force=True)
    assert forced.written and not forced.skipped
    assert forced.reason == "forced rewrite"


def test_ensure_does_not_measure_when_report_is_complete(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())
    calls = 0

    def factory() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _payload()

    result = ensure_model_report(paths, factory)
    assert result.skipped
    assert calls == 0

    forced = ensure_model_report(paths, factory, force=True)
    assert forced.written
    assert calls == 1


def test_missing_html_makes_report_incomplete_and_rebuildable(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())
    paths.html_path.unlink()

    check = check_model_report(paths)
    assert not check.complete
    assert "HTML" in check.reason

    result = write_model_report(paths, _payload())
    assert result.written
    assert check_model_report(paths).complete


def test_identity_mismatch_makes_report_incomplete(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())

    check = check_model_report(
        paths,
        identity=ModelReportIdentity(report_id=REPORT_ID, artifact_sha256="d" * 64),
    )
    assert not check.complete
    assert "sha256" in check.reason


def test_provenance_identity_matches(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())
    identity = ModelReportIdentity(
        report_id=REPORT_ID,
        model_id="rfdetr-nano",
        runtime="onnx",
        precision="fp32",
        batch_size=1,
        artifact_sha256=SHA,
        source_fp32_sha256=SOURCE_SHA,
    )
    assert check_model_report(paths, identity=identity).complete


def test_writer_rejects_payload_that_lacks_required_accuracy(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    payload = build_model_report(
        report_id=REPORT_ID,
        model=_model(),
        provenance=_provenance(),
        environment=_environment(),
        generated_at=STAMP,
    )
    with pytest.raises(ModelReportValidationError, match="accuracy section"):
        write_model_report(paths, payload, require_accuracy=True)


def test_required_accuracy_status_gate(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    payload = _payload(
        accuracy={
            "scenario": ACCURACY_SCENARIO,
            "status": "unavailable",
            "reason": "CUDA unavailable",
        }
    )
    write_model_report(paths, payload)
    assert check_model_report(paths, required_accuracy_status="unavailable").complete
    assert not check_model_report(paths, required_accuracy_status="measured").complete


def test_required_performance_gate(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    payload = build_model_report(
        report_id=REPORT_ID,
        model=_model(),
        provenance=_provenance(),
        accuracy=_accuracy(),
        environment=_environment(),
        generated_at=STAMP,
    )
    write_model_report(paths, payload)
    assert not check_model_report(paths, require_performance=True).complete


def test_render_model_report_file_does_not_need_the_artifact(tmp_path: Path) -> None:
    paths = model_report_paths(tmp_path / "missing" / "rfdetr-nano_b1_fp32.onnx")
    write_model_report(paths, _payload())
    paths.html_path.unlink()

    rendered = render_model_report_file(paths)
    assert rendered.is_file()
    assert "rfdetr-nano" in rendered.read_text(encoding="utf-8")


def test_rewrite_model_report_html(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())
    paths.html_path.unlink()
    assert rewrite_model_report_html(tmp_path) == 1
    assert paths.html_path.is_file()


# --------------------------------------------------------------------------- merge and overview


def test_merge_preserves_measured_accuracy_when_incoming_is_skipped() -> None:
    existing = _payload()
    incoming = _payload(
        accuracy={"scenario": ACCURACY_SCENARIO, "status": "skipped", "reason": "not run"}
    )
    merged = merge_model_report(existing, incoming)
    assert merged["scoring"]["accuracy"]["status"] == "measured"
    assert merged["scoring"]["accuracy"]["passes"][0]["pass_id"] == "python"


def test_merge_upserts_benchmark_snapshots_by_run_id() -> None:
    existing = _payload(performance=_performance("run-1"))
    incoming = _payload(performance=_performance("run-2"))
    merged = merge_model_report(existing, incoming)
    run_ids = [entry["run_id"] for entry in merged["scoring"]["performance"]["benchmarks"]]
    assert run_ids == ["run-1", "run-2"]


def test_overview_rescans_model_reports_and_ignores_benchmark_reports(tmp_path: Path) -> None:
    first = model_report_paths(_artifact(tmp_path, "rfdetr-nano_b1_fp32.onnx"))
    second = model_report_paths(_artifact(tmp_path, "rfdetr-nano_b1_fp16.onnx"))
    write_model_report(first, _payload(report_id="rfdetr-nano_b1_fp32"))
    write_model_report(
        second,
        _payload(
            report_id="rfdetr-nano_b1_fp16",
            model=_model(precision="fp16"),
            performance=_performance("run-2"),
        ),
    )
    benchmark_file = tmp_path / "exports" / "rfdetr-nano_b1_fp32.benchmark-run-1.json"
    benchmark_file.write_text("{}", encoding="utf-8")

    overview = build_model_overview(tmp_path, generated_at=STAMP)
    assert overview["kind"] == MODEL_OVERVIEW_KIND
    assert overview["report_count"] == 2
    assert overview["row_count"] == 2
    assert {row["report_id"] for row in overview["rows"]} == {
        "rfdetr-nano_b1_fp32",
        "rfdetr-nano_b1_fp16",
    }
    assert all(row["best_images_per_second"] == 200.0 for row in overview["rows"])


def test_overview_prefers_newest_report_for_duplicate_report_id(tmp_path: Path) -> None:
    first = model_report_paths(tmp_path / "a.onnx")
    second = model_report_paths(tmp_path / "b.onnx")
    older = _payload(report_id="same")
    older["generated_at"] = "2026-10-19T00:00:00+00:00"
    newer = _payload(report_id="same")
    newer["generated_at"] = "2026-10-19T01:00:00+00:00"
    write_model_report(first, older)
    write_model_report(second, newer)

    overview = build_model_overview(tmp_path, generated_at=STAMP)
    assert overview["report_count"] == 1
    assert overview["rows"][0]["generated_at"] == "2026-10-19T01:00:00+00:00"


def test_write_model_overview_writes_json_and_html(tmp_path: Path) -> None:
    paths = model_report_paths(_artifact(tmp_path))
    write_model_report(paths, _payload())
    json_path, html_path = write_model_overview(tmp_path, generated_at=STAMP)
    assert json_path.is_file() and html_path.is_file()
    assert "Model report overview" in html_path.read_text(encoding="utf-8")
    assert render_model_overview_html(build_model_overview(tmp_path, generated_at=STAMP))


# --------------------------------------------------------------------------- import boundary


def test_package_does_not_import_model_or_benchmark_frameworks() -> None:
    banned = {"rf_detr", "onnx_runtime", "torch", "onnxruntime", "numpy"}
    sources = sorted(PACKAGE_DIR.rglob("*.py"))
    assert len(sources) >= 20, f"import scan found only {len(sources)} files"
    imported: set[str] = set()
    for source_file in sources:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert not (imported & banned), f"forbidden imports: {sorted(imported & banned)}"


def test_fake_external_adapter_can_build_and_generate(tmp_path: Path) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.measurements = 0

        def report(self) -> dict[str, Any]:
            self.measurements += 1
            return _payload()

    adapter = FakeAdapter()
    paths = model_report_paths(_artifact(tmp_path))
    result = ensure_model_report(paths, adapter.report)
    assert result.written
    assert adapter.measurements == 1

    again = ensure_model_report(paths, adapter.report)
    assert again.skipped
    assert adapter.measurements == 1


def test_payload_is_deep_copied() -> None:
    accuracy = _accuracy()
    payload = _payload(accuracy=accuracy)
    accuracy["device"]["matches_internal"] = False
    assert payload["scoring"]["accuracy"]["device"]["matches_internal"] is True


def test_copy_helper_keeps_payload_independent() -> None:
    payload = _payload()
    duplicate = copy.deepcopy(payload)
    duplicate["model"]["batch_size"] = 3
    assert payload["model"]["batch_size"] == 1


def test_overview_paths_use_windows_safe_stamp(tmp_path: Path) -> None:
    json_path, html_path = model_overview_paths(tmp_path, STAMP)
    assert json_path.name == "model-report-overview-20261019T123456Z.json"
    assert html_path.name == "model-report-overview-20261019T123456Z.html"
    assert ":" not in json_path.name
