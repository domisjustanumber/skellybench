"""The plain-data API for exporters that collect data in another process (version 0.3.0)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from conftest import context, execution, local_probe, metrics, sha, subject
from skellybench import SkellybenchValidationError, blocks
from skellybench.reports import build_benchmark_report
from skellybench.render import render_benchmark_html
from skellybench.schema import validate_environment

DIGEST = "sha256:" + "a" * 64
CONTAINER = {"image": "domscott/openmmlab-exporter", "tag": "df-0123456789ab", "digest": DIGEST,
             "dockerfile_sha256": sha("dockerfile"), "published": True}


def external(**overrides):
    values = {"python": {"implementation": "CPython", "version": "3.8.20"}, "packages": {"torch": "2.0.0"},
              "container": CONTAINER, "collected_by": "exporter:docker"}
    return blocks.build_external_environment(**{**values, **overrides})


# --- change 1: external environment ---------------------------------------------------------------


def test_external_environment_validates_and_fingerprint_covers_origin_and_container() -> None:
    environment = external()
    validate_environment(environment)
    assert environment["collected_by"] == "exporter:docker"
    assert environment["container"]["digest"] == DIGEST
    assert environment["package_sources"] == {}
    other_origin = external(collected_by="exporter:host")
    other_digest = external(container={**CONTAINER, "digest": "sha256:" + "b" * 64})
    no_container = external(container=None)
    fingerprints = {item["environment_fingerprint"] for item in (environment, other_origin, other_digest, no_container)}
    assert len(fingerprints) == 4
    assert external()["environment_fingerprint"] == environment["environment_fingerprint"]


def test_external_environment_rejects_bad_fields() -> None:
    with pytest.raises(ValueError):
        external(collected_by="")
    for key, value in (("digest", "sha256:short"), ("dockerfile_sha256", "xyz"), ("image", "")):
        broken = external(container={**CONTAINER, key: value})
        with pytest.raises(SkellybenchValidationError, match=key):
            validate_environment(broken)
    broken = external()
    broken["container"]["published"] = "yes"
    with pytest.raises(SkellybenchValidationError, match="published"):
        validate_environment(broken)
    broken = external(container={**CONTAINER, "image": "/home/user/image"})
    with pytest.raises(SkellybenchValidationError, match="absolute local path"):
        build_benchmark_report_with(broken)


def build_benchmark_report_with(environment):
    ctx = context()
    ctx["environment"] = environment
    return build_benchmark_report(
        benchmark_type="performance", status="unavailable", reason="test", execution=execution(ctx["host"]),
        subject=subject("rfdetr-nano_b1_fp32", sha("fp32")), **ctx)


def test_benchmark_page_shows_collector_and_container_image() -> None:
    report = build_benchmark_report_with(external())
    html = render_benchmark_html(report)
    assert "exporter:docker" in html
    assert "domscott/openmmlab-exporter:df-0123456789ab" in html
    assert DIGEST in html and "published" in html
    assert "Environment collected by" not in render_benchmark_html(build_benchmark_report_with(context()["environment"]))


# --- changes 2 and 3: method, shared evaluation and pairs across runs -----------------------------

from conftest import METHOD, onnx_model, source_model, write_accuracy  # noqa: E402
from skellybench.protocol import accuracy_protocol  # noqa: E402
from skellybench.reports import report_id_for, write_model_report  # noqa: E402
from skellybench.render import render_family  # noqa: E402
from skellybench.schema import validate_benchmark_report, validate_method  # noqa: E402

OLD_RUN = "20261019T100000Z-gerald"
SOURCE_METHOD = {"decode": {"library": "mmdet", "nms": "in-library"},
                 "scorer": {"tool": "faster-coco-eval", "version": "1.8.0"}, "scored_by": "source_tool"}


def test_method_is_required_and_checked() -> None:
    validate_method(METHOD, metrics_tool="faster-coco-eval")
    with pytest.raises(SkellybenchValidationError, match="scorer.tool"):
        validate_method(METHOD, metrics_tool="pycocotools")
    with pytest.raises(SkellybenchValidationError, match="scored_by"):
        validate_method({**METHOD, "scored_by": "me"})
    with pytest.raises(SkellybenchValidationError, match="decode"):
        validate_method({**METHOD, "decode": None})
    assert blocks.build_accuracy_method(decode={"a": 1}, scorer_tool="t", scorer_version=None,
                                        scored_by="skellybench")["scorer"] == {"tool": "t", "version": None}


def test_benchmark_without_method_is_rejected(family: Path) -> None:
    path = write_accuracy(family, context(), subject("a", sha("a")), execution(context()["host"]), 0.5)
    from skellybench._internal.io import read_json
    report = read_json(path)
    validate_benchmark_report(report)
    del report["accuracy"]["method"]
    with pytest.raises(SkellybenchValidationError, match="method"):
        validate_benchmark_report(report)
    report["accuracy"]["method"] = {**METHOD, "scorer": {"tool": "other", "version": None}}
    with pytest.raises(SkellybenchValidationError, match="metrics.tool"):
        validate_benchmark_report(report)


def test_protocol_id_ignores_method() -> None:
    evaluation = {"iou_type": "bbox", "max_dets": 100, "category_ids": [1]}
    assert accuracy_protocol(batch_size=1, images=2, evaluation=evaluation)["protocol_id"] == \
        accuracy_protocol(batch_size=1, images=2, evaluation=dict(evaluation))["protocol_id"]


def _reused_lineage(family: Path):
    old, new = context(OLD_RUN), context()
    host = new["host"]
    checkpoint, onnx_sha = sha("checkpoint"), sha("fp32")
    source_subject = subject("rfdetr-nano_source", checkpoint, batch_agnostic=True, fmt="pytorch_checkpoint")
    onnx_subject = subject("rfdetr-nano_b1_fp32", onnx_sha)
    source_json = write_accuracy(family, old, source_subject, execution(old["host"], "torch-cuda"), 0.5,
                                 method=SOURCE_METHOD)
    onnx_json = write_accuracy(family, new, onnx_subject, execution(host), 0.5)
    source = source_model(old, source_json, family, checkpoint)
    onnx = onnx_model(new, "rfdetr-nano_b1_fp32", onnx_sha, "rfdetr-nano_source", checkpoint, onnx_json,
                      source_json, family, parent_run_id=OLD_RUN)
    return source, onnx


def test_pair_across_runs_with_different_methods_is_valid_and_shown(family: Path) -> None:
    source, onnx = _reused_lineage(family)
    assert onnx["accuracy"]["parent_run_id"] == OLD_RUN
    assert source["accuracy"]["parent_run_id"] is None
    write_model_report(family, source)
    write_model_report(family, onnx)
    render_family(family)
    html = (family / "rfdetr-nano_b1_fp32.model.html").read_text(encoding="utf-8")
    assert f"Parent pass reused from run <code>{OLD_RUN}</code>" in html
    assert f"reused from run {OLD_RUN}" in html
    assert "source_tool" in html and "skellybench" in html and "mmdet" in html
    assert "Methods" in html


def test_pair_still_compares_evaluation(family: Path) -> None:
    _, onnx = _reused_lineage(family)
    broken = copy.deepcopy(onnx)
    broken["accuracy"]["parent"]["snapshot"]["accuracy"]["evaluation"]["category_ids"] = [1, 2]
    from skellybench.schema import validate_model_report
    with pytest.raises(SkellybenchValidationError, match="accuracy.evaluation"):
        validate_model_report(broken)


# --- change 4: category filter ---------------------------------------------------------------------


def _two_category_annotation(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "two.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [60.0, 60.0, 20.0, 20.0], "area": 400, "iscrowd": 0},
        ],
    }), encoding="utf-8")
    return path


def test_category_filter_changes_ap_and_reports_used_ids(tmp_path: Path) -> None:
    pytest.importorskip("faster_coco_eval")
    from skellybench.scoring import ScoringError, score_predictions

    annotation = _two_category_annotation(tmp_path)
    # The person is found. The bicycle is missed (the box is in the wrong place).
    records = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 5.0, 5.0], "score": 0.8},
    ]
    everything = score_predictions(records, annotation, image_ids=[1])
    person = score_predictions(records, annotation, image_ids=[1], category_ids=[1])
    bicycle = score_predictions(records, annotation, image_ids=[1], category_ids=[2])
    assert everything.category_ids == (1, 2)
    assert person.category_ids == (1,)
    assert person.metrics["ap"] == pytest.approx(1.0)
    assert everything.metrics["ap"] == pytest.approx(0.5)
    assert bicycle.metrics["ap"] == pytest.approx(0.0)
    with pytest.raises(ScoringError, match="not in the annotation"):
        score_predictions(records, annotation, image_ids=[1], category_ids=[9])
    with pytest.raises(ScoringError, match="empty"):
        score_predictions(records, annotation, image_ids=[1], category_ids=[])


def test_score_columns_takes_category_ids(tmp_path: Path) -> None:
    pytest.importorskip("faster_coco_eval")
    from skellybench.predictions import columns_from_records
    from skellybench.scoring import score_columns

    annotation = _two_category_annotation(tmp_path)
    records = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 5.0, 5.0], "score": 0.8},
    ]
    columns = columns_from_records(records, [1])
    assert score_columns(columns, annotation, category_ids=[1]).metrics["ap"] == pytest.approx(1.0)
    assert score_columns(columns, annotation).metrics["ap"] == pytest.approx(0.5)
