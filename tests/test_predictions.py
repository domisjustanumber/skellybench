"""Columnar predictions files and COCO scoring of decoded columns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skellybench import SkellybenchValidationError
from skellybench.predictions import (
    columns_from_records,
    decode_payload,
    encode_columns,
    predictions_payload,
    read_predictions_file,
    write_predictions_file,
)
from skellybench.scoring import compare_metrics, score_columns, score_predictions

RECORDS = [
    {"image_id": 1, "category_id": 1, "bbox": [10.25, 10.5, 20.0, 20.0], "score": 0.9},
    {"image_id": 1, "category_id": 2, "bbox": [50.0, 50.0, 10.0, 10.0], "score": 0.25},
    {"image_id": 2, "category_id": 1, "bbox": [0.0, 0.0, 30.0, 30.0], "score": 0.75},
    {"image_id": 2, "category_id": 2, "bbox": [40.0, 40.0, 5.0, 5.0], "score": 0.125},
]


def _annotation(tmp_path: Path) -> Path:
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"},
                   {"id": 2, "width": 100, "height": 100, "file_name": "2.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0.0, 0.0, 30.0, 30.0], "area": 900, "iscrowd": 0},
        ],
    }), encoding="utf-8")
    return path


def test_encode_decode_round_trip_and_hashes(tmp_path: Path) -> None:
    columns = columns_from_records(RECORDS, [1, 2])
    assert columns.per_image == 2 and columns.counts is None
    reference = write_predictions_file(tmp_path / "p.predictions.json", "r1", columns)
    assert reference["record_count"] == 4
    decoded = read_predictions_file(tmp_path / "p.predictions.json")
    assert decoded.data == columns.data
    assert decoded.data_sha256 == reference["data_sha256"]
    assert decoded.to_records() == [
        {**record, "score": pytest.approx(record["score"])} for record in RECORDS
    ]


def test_variable_counts_round_trip() -> None:
    records = RECORDS[:3]
    columns = columns_from_records(records, [1, 2])
    assert columns.per_image is None and columns.counts == (2, 1)
    decoded = decode_payload(predictions_payload("r1", columns))
    assert decoded.row_image_ids() == [1, 1, 2]


def test_tampered_data_is_rejected() -> None:
    payload = predictions_payload("r1", columns_from_records(RECORDS, [1, 2]))
    payload["data_sha256"] = "0" * 64
    with pytest.raises(SkellybenchValidationError, match="data_sha256"):
        decode_payload(payload)


def test_numpy_columns_match_records() -> None:
    np = pytest.importorskip("numpy")
    columns = encode_columns(
        image_ids=[1, 2], per_image=2,
        category_ids=np.array([1, 2, 1, 2]),
        bboxes=np.array([record["bbox"] for record in RECORDS], dtype=np.float32),
        scores=np.array([record["score"] for record in RECORDS], dtype=np.float32),
    )
    assert columns.data == columns_from_records(RECORDS, [1, 2]).data
    array = columns.to_numpy()
    assert array.shape == (4, 7) and array[0, 0] == 1 and array[0, 6] == 1


def test_scoring_decoded_columns_equals_scoring_records(tmp_path: Path) -> None:
    pytest.importorskip("faster_coco_eval")
    annotation = _annotation(tmp_path)
    columns = decode_payload(predictions_payload("r1", columns_from_records(RECORDS, [1, 2])))
    from_columns = score_columns(columns, annotation)
    from_records = score_predictions(RECORDS, annotation, image_ids=[1, 2])
    assert from_columns.metrics == from_records.metrics
    assert from_columns.metrics["ap"] > 0.5


def test_compare_metrics_status() -> None:
    parent = {key: 0.5 for key in ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10",
                                   "ar100", "ar_small", "ar_medium", "ar_large")}
    assert compare_metrics(parent, parent)["status"] == "ok"
    assert compare_metrics(parent, {**parent, "ap": 0.495})["status"] == "warn"
    failed = compare_metrics(parent, {**parent, "ap": 0.48})
    assert failed["status"] == "fail" and failed["deltas"]["delta_ap"] == pytest.approx(-0.02)
