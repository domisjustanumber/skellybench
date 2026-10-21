"""The dataset blocks of benchmark and model reports (version 0.4.0)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from conftest import PERSON, context, dataset, execution, performance_dataset, sha, subject, write_accuracy
from skellybench import SkellybenchValidationError, blocks
from skellybench._internal.io import read_json
from skellybench.schema import validate_benchmark_report, validate_dataset, validate_model_report
from test_reports import _lineage


def test_accuracy_dataset_needs_annotation_file_categories_and_annotation_hash() -> None:
    validate_dataset(dataset(), benchmark_type="accuracy")
    for key in ("annotation_file", "categories", "annotation_sha256"):
        for value in (None, "absent"):
            broken = dataset()
            if value == "absent":
                del broken[key]
            else:
                broken[key] = value
            with pytest.raises(SkellybenchValidationError, match=key):
                validate_dataset(broken, benchmark_type="accuracy")


def test_categories_are_checked() -> None:
    cases = [
        ([], "must not be empty"),
        ([{"id": 1}], "categories\\[0\\].name"),
        ([{"id": -1, "name": "person"}], "categories\\[0\\].id"),
        ([{"id": "1", "name": "person"}], "categories\\[0\\].id"),
        ([{"id": 1, "name": "person"}, {"id": 1, "name": "other"}], "duplicate category id 1"),
        ("person", "must be an array"),
    ]
    for categories, match in cases:
        with pytest.raises(SkellybenchValidationError, match=match):
            validate_dataset({**dataset(), "categories": categories}, benchmark_type="accuracy")
    validate_dataset({**dataset(), "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}]},
                     benchmark_type="accuracy")


def test_performance_dataset_allows_null_annotations() -> None:
    validate_dataset(performance_dataset(), benchmark_type="performance")
    bare = {key: value for key, value in performance_dataset().items()
            if key not in ("annotation_file", "categories", "annotation_sha256", "split")}
    validate_dataset(bare, benchmark_type="performance")
    with pytest.raises(SkellybenchValidationError, match="categories"):
        validate_dataset({**performance_dataset(), "categories": []}, benchmark_type="performance")
    with pytest.raises(SkellybenchValidationError, match="annotation_file"):
        validate_dataset({**performance_dataset(), "annotation_file": ""}, benchmark_type="performance")
    with pytest.raises(SkellybenchValidationError, match="split"):
        validate_dataset({**performance_dataset(), "split": ""}, benchmark_type="performance")


def test_accuracy_report_without_categories_is_rejected(family: Path) -> None:
    report = read_json(write_accuracy(family, context(), subject("a", sha("a")), execution(context()["host"]), 0.5))
    validate_benchmark_report(report)
    report["dataset"]["categories"] = None
    with pytest.raises(SkellybenchValidationError, match=r"\$.dataset.categories"):
        validate_benchmark_report(report)


def test_annotation_file_must_not_be_an_absolute_path(family: Path) -> None:
    report = read_json(write_accuracy(family, context(), subject("a", sha("a")), execution(context()["host"]), 0.5))
    report["dataset"]["annotation_file"] = "C:\\coco\\annotations\\instances_val2017.json"
    with pytest.raises(SkellybenchValidationError, match="absolute local path"):
        validate_benchmark_report(report)


def test_model_report_dataset_is_built_from_its_accuracy_pass(family: Path) -> None:
    lineage = _lineage(family)
    model_dataset = lineage["fp16"]["dataset"]
    assert model_dataset == dataset()
    assert model_dataset["categories"] == PERSON
    assert "dataset" in lineage["source"]


def test_model_report_dataset_must_equal_the_self_dataset(family: Path) -> None:
    lineage = _lineage(family)
    for key, value in (("images_used", 1), ("categories", [{"id": 2, "name": "bicycle"}]),
                       ("annotation_file", "other.json"), ("split", "train2017"), ("name", "other")):
        broken = copy.deepcopy(lineage["fp16"])
        broken["dataset"][key] = value
        with pytest.raises(SkellybenchValidationError, match=rf"\$.dataset.{key}"):
            validate_model_report(broken)


def test_model_report_dataset_is_required_and_complete(family: Path) -> None:
    lineage = _lineage(family)
    broken = copy.deepcopy(lineage["fp32"])
    del broken["dataset"]
    with pytest.raises(SkellybenchValidationError, match=r"\$.dataset"):
        validate_model_report(broken)
    for key in ("split", "annotation_file", "categories", "image_list_sha256"):
        broken = copy.deepcopy(lineage["fp32"])
        del broken["dataset"][key]
        with pytest.raises(SkellybenchValidationError, match=rf"\$.dataset.{key}"):
            validate_model_report(broken)
    broken = copy.deepcopy(lineage["fp32"])
    broken["dataset"]["notes"] = "x"
    with pytest.raises(SkellybenchValidationError, match="'notes'"):
        validate_model_report(broken)


def test_build_model_dataset_copies_known_keys_and_takes_overrides() -> None:
    snapshot_dataset = {key: value for key, value in dataset().items() if key != "split"}
    snapshot_dataset["extra"] = "ignored"
    block = blocks.build_model_dataset(snapshot_dataset, split="val2017")
    assert block["split"] == "val2017" and "extra" not in block
    assert block["categories"] == PERSON
