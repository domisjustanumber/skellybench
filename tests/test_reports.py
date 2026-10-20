"""Benchmark and model reports: write-once JSON, pairing rules, superseding, and HTML rebuilds."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from conftest import (
    RUN_ID,
    context,
    execution,
    onnx_model,
    sha,
    source_model,
    subject,
    write_accuracy,
    write_performance,
)
from skellybench import SkellybenchValidationError
from skellybench._internal.io import read_json
from skellybench.reports import scan_family, supersede_model_report, write_model_report
from skellybench.render import claim_run, read_run_state, rebuild_all, render_family, write_run_summary
from skellybench.schema import validate_model_report


def _lineage(family: Path, ctx: dict | None = None, *, fp16_ap: float = 0.5):
    ctx = ctx or context()
    host = ctx["host"]
    checkpoint = sha("checkpoint")
    fp32_sha, fp16_sha = sha("fp32"), sha("fp16")
    source_subject = subject("rfdetr-nano_source", checkpoint, batch_agnostic=True, fmt="pytorch_checkpoint")
    fp32_subject = subject("rfdetr-nano_b1_fp32", fp32_sha)
    fp16_subject = subject("rfdetr-nano_b1_fp16", fp16_sha, precision="fp16")
    source_json = write_accuracy(family, ctx, source_subject, execution(host, "torch-cuda"), 0.5)
    fp32_json = write_accuracy(family, ctx, fp32_subject, execution(host), 0.5)
    fp16_json = write_accuracy(family, ctx, fp16_subject, execution(host), fp16_ap)
    write_performance(family, ctx, fp32_subject, execution(host), 120.0)
    write_performance(family, ctx, fp32_subject, execution(host, "cpu"), 15.0)
    source = source_model(ctx, source_json, family, checkpoint)
    fp32 = onnx_model(ctx, "rfdetr-nano_b1_fp32", fp32_sha, "rfdetr-nano_source", checkpoint, fp32_json, source_json, family)
    fp16 = onnx_model(ctx, "rfdetr-nano_b1_fp16", fp16_sha, "rfdetr-nano_b1_fp32", fp32_sha, fp16_json, fp32_json, family,
                      precision="fp16", self_ap=fp16_ap)
    for report in (source, fp32, fp16):
        write_model_report(family, report)
    return {"source": source, "fp32": fp32, "fp16": fp16, "fp32_json": fp32_json, "fp16_json": fp16_json}


def test_benchmark_files_are_one_per_ep_and_named_by_rule(family: Path) -> None:
    _lineage(family)
    names = sorted(path.name for path in (family / "benchmarks" / "rfdetr-nano_b1_fp32").iterdir())
    assert f"rfdetr-nano_b1_fp32.performance.cuda.{RUN_ID}.json" in names
    assert f"rfdetr-nano_b1_fp32.performance.cpu.{RUN_ID}.json" in names
    assert f"rfdetr-nano_b1_fp32.accuracy.cuda.{RUN_ID}.predictions.json" in names
    source_names = [path.name for path in (family / "benchmarks" / "rfdetr-nano_source").iterdir()]
    assert f"rfdetr-nano_source.accuracy.torch-cuda.b1.{RUN_ID}.json" in source_names


def test_json_reports_are_write_once(family: Path) -> None:
    lineage = _lineage(family)
    with pytest.raises(FileExistsError):
        write_model_report(family, lineage["fp32"])
    with pytest.raises(FileExistsError):
        write_accuracy(family, context(), subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(context()["host"]), 0.5)


def test_pair_from_different_runs_is_rejected(family: Path) -> None:
    lineage = _lineage(family)
    broken = copy.deepcopy(lineage["fp16"])
    broken["accuracy"]["parent"]["snapshot"]["run"]["run_id"] = "20261021T000000Z-other"
    with pytest.raises(SkellybenchValidationError, match="run.run_id"):
        validate_model_report(broken)


def test_pair_on_different_gpu_is_rejected_but_different_ep_is_accepted(family: Path) -> None:
    lineage = _lineage(family)
    # fp32 pairs a torch-cuda source with the CUDA EP on the same GPU: valid.
    validate_model_report(lineage["fp32"])
    broken = copy.deepcopy(lineage["fp32"])
    broken["accuracy"]["parent"]["snapshot"]["execution"]["device"]["host_gpu_uuid"] = "GPU-other"
    with pytest.raises(SkellybenchValidationError, match="physical device"):
        validate_model_report(broken)


def test_parent_sha_must_match_conversion_input(family: Path) -> None:
    lineage = _lineage(family)
    broken = copy.deepcopy(lineage["fp16"])
    broken["creation"]["steps"][0]["input_sha256"] = sha("something else")
    with pytest.raises(SkellybenchValidationError, match="input_sha256"):
        validate_model_report(broken)


def test_reports_reject_absolute_paths(family: Path) -> None:
    lineage = _lineage(family)
    broken = copy.deepcopy(lineage["fp32"])
    broken["creation"]["run"]["command"] = "C:\\Users\\Dom\\uv.exe run"
    with pytest.raises(SkellybenchValidationError, match="absolute local path"):
        validate_model_report(broken)


def test_model_page_lists_later_benchmarks_and_ignores_other_artifacts(family: Path) -> None:
    _lineage(family)
    later = context("20261021T090000Z-gerald")
    write_performance(family, later, subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(later["host"]), 130.0)
    write_performance(family, later, subject("rfdetr-nano_b1_fp32", sha("different")), execution(later["host"], "cpu"), 9.0)
    render_family(family)
    html = (family / "rfdetr-nano_b1_fp32.model.html").read_text(encoding="utf-8")
    assert "20261021T090000Z-gerald" in html
    assert "130.00" in html
    assert "9.00" not in html
    # Chart marks link to performance benchmark pages, never to accuracy timings.
    assert f'href="benchmarks/rfdetr-nano_b1_fp32/rfdetr-nano_b1_fp32.performance.cuda.{RUN_ID}.html"' in html
    assert "<svg" in html and "40.00 images/s" not in html


def test_supersede_moves_model_and_links_still_resolve(family: Path) -> None:
    _lineage(family)
    moved = supersede_model_report(family, "rfdetr-nano_b1_fp32")
    assert moved is not None and moved.json.parent.name == RUN_ID
    assert not (family / "rfdetr-nano_b1_fp32.model.json").exists()
    index = scan_family(family)
    assert [entry.payload["report_id"] for entry in index.superseded] == ["rfdetr-nano_b1_fp32"]
    render_family(family)
    html = moved.html.read_text(encoding="utf-8")
    assert "superseded" in html
    assert 'href="../../benchmarks/rfdetr-nano_b1_fp32/' in html


def test_failed_comparison_banner_reaches_descendants(family: Path) -> None:
    _lineage(family, fp16_ap=0.4)
    render_family(family)
    fp16 = (family / "rfdetr-nano_b1_fp16.model.html").read_text(encoding="utf-8")
    assert "banner fail" in fp16
    assert "Cumulative" in fp16


def test_rebuild_skips_old_schema_json_and_never_opens_predictions(family: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _lineage(family)
    (family / "old.model.json").write_text(json.dumps({"schema_version": 2, "kind": "model_report"}), encoding="utf-8")
    (family / "benchmarks" / "rfdetr-nano_b1_fp32" / "junk.json").write_text("{", encoding="utf-8")
    import skellybench.reports as reports_module

    original = reports_module.read_json

    def guarded(path):
        assert not str(path).endswith(".predictions.json"), path
        return original(path)

    monkeypatch.setattr(reports_module, "read_json", guarded)
    exports_root = family.parents[2]
    counts = rebuild_all(exports_root)
    assert counts["model_pages"] == 3
    assert counts["benchmark_pages"] == 5


def test_html_rebuilds_from_json_alone(family: Path) -> None:
    _lineage(family)
    render_family(family)
    first = {path: path.read_text(encoding="utf-8") for path in family.rglob("*.html")}
    for path in first:
        path.unlink()
    render_family(family)
    second = {path: path.read_text(encoding="utf-8") for path in family.rglob("*.html")}
    assert first == second


def test_run_claim_and_summary_keep_state(family: Path) -> None:
    lineage = _lineage(family)
    exports_root = family.parents[2]
    state = {"run_id": "20261022T000000Z-gerald", "command": "uv run x", "hostname": "Gerald", "status": "in_progress"}
    path = claim_run(exports_root, state)
    with pytest.raises(FileExistsError):
        claim_run(exports_root, state)
    assert read_run_state(path)["status"] == "in_progress"

    summary_state = {"run_id": RUN_ID, "status": "finished", "command": "uv run x", "hostname": "Gerald",
                     "warnings": ["model report not written: --limit"], "exit_code": 0}
    summary = write_run_summary(exports_root, summary_state)
    html = summary.read_text(encoding="utf-8")
    assert "model report not written: --limit" in html
    assert lineage["fp32"]["report_id"] in html
    assert "../detectors/object_detectors/rfdetr/benchmarks/" in html
    summary.write_text(html.replace("Warnings", "Changed"), encoding="utf-8")
    rebuild_all(exports_root)
    assert read_run_state(summary)["warnings"] == ["model report not written: --limit"]
    assert "Warnings" in summary.read_text(encoding="utf-8")


def test_snapshots_hold_full_benchmark_json(family: Path) -> None:
    lineage = _lineage(family)
    snapshot = lineage["fp16"]["accuracy"]["self"]["snapshot"]
    assert snapshot == read_json(lineage["fp16_json"])
    assert "data" not in snapshot["accuracy"]["predictions"]
