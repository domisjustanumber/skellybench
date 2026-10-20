"""Columnar predictions files for accuracy benchmarks.

One predictions file holds every COCO detection of one accuracy pass. The detections are stored as
three little-endian column blocks, in this order, then gzip, then base64:

1. ``category_id``: uint16, one per detection.
2. ``bbox_xywh``: float32, four per detection (x, y, width, height in original image pixels).
3. ``score``: float32, one per detection.

Detection ``i`` belongs to the image found from ``image_ids`` and ``per_image`` (a fixed count per
image) or ``counts`` (one count per image). float32 is lossless for model outputs that are float32.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import sys
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._internal.hashing import sha256_file
from ._internal.io import read_json, write_json_once
from .schema import (
    PREDICTIONS_FORMAT,
    PREDICTIONS_KIND,
    SCHEMA_VERSION,
    SkellybenchValidationError,
    validate_predictions_header,
)

COLUMNS: tuple[dict[str, Any], ...] = (
    {"name": "category_id", "dtype": "uint16"},
    {"name": "bbox_xywh", "dtype": "float32", "width": 4},
    {"name": "score", "dtype": "float32"},
)


def _little_endian(values: array) -> bytes:
    if sys.byteorder != "little":
        values = array(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def _column_bytes(values: Any, typecode: str, numpy_dtype: str) -> bytes:
    """Return little-endian bytes for a numpy array or any sequence of numbers."""
    if hasattr(values, "astype") and hasattr(values, "tobytes"):
        return values.astype(numpy_dtype, copy=False).reshape(-1).tobytes()
    if isinstance(values, (bytes, bytearray, memoryview)):
        return bytes(values)
    return _little_endian(array(typecode, values))


@dataclass(frozen=True)
class PredictionColumns:
    """Decoded or encoded columns of one accuracy pass."""

    image_ids: tuple[int, ...]
    per_image: int | None
    counts: tuple[int, ...] | None
    category_ids: bytes
    bboxes: bytes
    scores: bytes

    @property
    def record_count(self) -> int:
        return len(self.scores) // 4

    @property
    def data(self) -> bytes:
        return self.category_ids + self.bboxes + self.scores

    @property
    def data_sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def row_image_ids(self) -> list[int]:
        """Return the image id of every detection, in order."""
        if self.per_image is not None:
            return [image_id for image_id in self.image_ids for _ in range(self.per_image)]
        assert self.counts is not None
        return [image_id for image_id, count in zip(self.image_ids, self.counts, strict=True) for _ in range(count)]

    def category_array(self) -> array:
        values = array("H")
        values.frombytes(self.category_ids)
        if sys.byteorder != "little":
            values.byteswap()
        return values

    def float_array(self, data: bytes) -> array:
        values = array("f")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        return values

    def to_records(self) -> list[dict[str, Any]]:
        """Return COCO result records. Use :meth:`to_numpy` for large passes."""
        categories = self.category_array()
        boxes = self.float_array(self.bboxes)
        scores = self.float_array(self.scores)
        return [
            {
                "image_id": image_id,
                "category_id": int(categories[index]),
                "bbox": [float(value) for value in boxes[index * 4 : index * 4 + 4]],
                "score": float(scores[index]),
            }
            for index, image_id in enumerate(self.row_image_ids())
        ]

    def to_numpy(self) -> Any:
        """Return an ``N x 7`` float64 array ``[image_id, x, y, w, h, score, category_id]``.

        This is the array layout that ``COCO.loadRes`` accepts. numpy is imported here only.
        """
        import numpy as np

        count = self.record_count
        result = np.empty((count, 7), dtype=np.float64)
        if self.per_image is not None:
            result[:, 0] = np.repeat(np.asarray(self.image_ids, dtype=np.float64), self.per_image)
        else:
            result[:, 0] = np.repeat(np.asarray(self.image_ids, dtype=np.float64), np.asarray(self.counts))
        result[:, 1:5] = np.frombuffer(self.bboxes, dtype="<f4").reshape(count, 4)
        result[:, 5] = np.frombuffer(self.scores, dtype="<f4")
        result[:, 6] = np.frombuffer(self.category_ids, dtype="<u2")
        return result


def encode_columns(
    *,
    image_ids: Sequence[int],
    category_ids: Any,
    bboxes: Any,
    scores: Any,
    per_image: int | None = None,
    counts: Sequence[int] | None = None,
) -> PredictionColumns:
    """Pack detections into :class:`PredictionColumns`.

    Each column may be a numpy array, a sequence of numbers, or little-endian bytes. *bboxes* is
    flat (``x, y, w, h`` per detection) or ``N x 4``.
    """
    if (per_image is None) == (counts is None):
        raise ValueError("pass exactly one of per_image and counts")
    columns = PredictionColumns(
        image_ids=tuple(int(image_id) for image_id in image_ids),
        per_image=int(per_image) if per_image is not None else None,
        counts=tuple(int(count) for count in counts) if counts is not None else None,
        category_ids=_column_bytes(category_ids, "H", "<u2"),
        bboxes=_column_bytes(_flatten(bboxes), "f", "<f4"),
        scores=_column_bytes(scores, "f", "<f4"),
    )
    expected = (
        columns.per_image * len(columns.image_ids)
        if columns.per_image is not None
        else sum(columns.counts or ())
    )
    if (
        len(columns.category_ids) != 2 * expected
        or len(columns.bboxes) != 16 * expected
        or len(columns.scores) != 4 * expected
    ):
        raise ValueError(f"column lengths do not match {expected} detections")
    return columns


def _flatten(bboxes: Any) -> Any:
    if hasattr(bboxes, "reshape"):
        return bboxes.reshape(-1)
    if isinstance(bboxes, (bytes, bytearray, memoryview)):
        return bboxes
    flat: list[float] = []
    for item in bboxes:
        if isinstance(item, Iterable):
            flat.extend(float(value) for value in item)
        else:
            flat.append(float(item))
    return flat


def columns_from_records(records: Iterable[Mapping[str, Any]], image_ids: Sequence[int]) -> PredictionColumns:
    """Pack COCO result records. Records must be grouped by image in *image_ids* order."""
    position = {int(image_id): index for index, image_id in enumerate(image_ids)}
    counts = [0] * len(image_ids)
    categories: list[int] = []
    boxes: list[float] = []
    scores: list[float] = []
    last = -1
    for record in records:
        index = position[int(record["image_id"])]
        if index < last:
            raise ValueError("records must be grouped by image in image_ids order")
        last = index
        counts[index] += 1
        categories.append(int(record["category_id"]))
        boxes.extend(float(value) for value in record["bbox"])
        scores.append(float(record["score"]))
    uniform = len(set(counts)) == 1
    return encode_columns(
        image_ids=image_ids,
        category_ids=categories,
        bboxes=boxes,
        scores=scores,
        per_image=counts[0] if uniform and counts else None,
        counts=None if uniform and counts else counts,
    )


def predictions_payload(report_id: str, columns: PredictionColumns) -> dict[str, Any]:
    """Return the predictions file payload for one accuracy report."""
    compressed = gzip.compress(columns.data, compresslevel=6, mtime=0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREDICTIONS_KIND,
        "report_id": report_id,
        "format": PREDICTIONS_FORMAT,
        "image_ids": list(columns.image_ids),
        "per_image": columns.per_image,
        "counts": list(columns.counts) if columns.counts is not None else None,
        "record_count": columns.record_count,
        "columns": [dict(column) for column in COLUMNS],
        "byte_order": "little",
        "layout": "column-major",
        "encoding": "gzip+base64",
        "data_sha256": columns.data_sha256,
        "data": base64.b64encode(compressed).decode("ascii"),
    }
    validate_predictions_header(payload)
    return payload


def write_predictions_file(path: str | Path, report_id: str, columns: PredictionColumns) -> dict[str, Any]:
    """Write a predictions file once. Return the reference for ``accuracy.predictions``.

    The returned ``filename`` is the bare name. :func:`skellybench.reports.write_benchmark_report`
    replaces it with the family-relative path.
    """
    destination = Path(path)
    write_json_once(destination, predictions_payload(report_id, columns), indent=None)
    return {
        "filename": destination.name,
        "file_sha256": sha256_file(destination),
        "data_sha256": columns.data_sha256,
        "format": PREDICTIONS_FORMAT,
        "record_count": columns.record_count,
    }


def decode_payload(payload: Mapping[str, Any]) -> PredictionColumns:
    """Decode a predictions payload and check its data hash."""
    validate_predictions_header(payload)
    data = gzip.decompress(base64.b64decode(payload["data"]))
    if hashlib.sha256(data).hexdigest() != payload["data_sha256"]:
        raise SkellybenchValidationError("$.data: decoded bytes do not match data_sha256")
    count = int(payload["record_count"])
    if len(data) != count * 22:
        raise SkellybenchValidationError(f"$.data: expected {count * 22} bytes, got {len(data)}")
    return PredictionColumns(
        image_ids=tuple(int(image_id) for image_id in payload["image_ids"]),
        per_image=payload.get("per_image"),
        counts=tuple(payload["counts"]) if payload.get("counts") is not None else None,
        category_ids=data[: 2 * count],
        bboxes=data[2 * count : 18 * count],
        scores=data[18 * count :],
    )


def read_predictions_file(path: str | Path) -> PredictionColumns:
    """Read and decode a predictions file."""
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise SkellybenchValidationError(f"{path}: predictions file is missing or unparseable")
    return decode_payload(payload)


__all__ = [
    "COLUMNS",
    "PredictionColumns",
    "columns_from_records",
    "decode_payload",
    "encode_columns",
    "predictions_payload",
    "read_predictions_file",
    "write_predictions_file",
]
