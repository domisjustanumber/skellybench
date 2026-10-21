"""COCO bbox scoring for accuracy benchmarks.

Importing this module does not import ``faster_coco_eval``. Scoring imports it on first use and
raises :class:`ScoringUnavailable` when the optional ``scoring`` extra is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .._internal.hashing import sha256_file
from ..predictions import PredictionColumns
from ..schema import COCO_STAT_KEYS

DEFAULT_MAX_DETS = 100
DEFAULT_WARN_AP = 0.001
DEFAULT_FAIL_AP = 0.01
SCORING_TOOL = "faster-coco-eval"


class ScoringError(RuntimeError):
    """Raised when prediction data cannot be scored."""


class ScoringUnavailable(ScoringError):
    """Raised when the optional scoring back end is not installed."""


@dataclass(frozen=True)
class ScoringResult:
    """Metrics and provenance for one scored pass."""

    metrics: dict[str, float]
    images_evaluated: int
    annotation_sha256: str
    tool: str = SCORING_TOOL
    version: str = ""

    def as_scorer_block(self) -> dict[str, Any]:
        """Return the ``accuracy.method.scorer`` block: ``{tool, version}``."""
        return {"tool": self.tool, "version": self.version}

    def as_metrics_block(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "version": self.version,
            "values": {key: round(value, 6) for key, value in self.metrics.items()},
        }


def metrics_from_stats(stats: Sequence[float]) -> dict[str, float]:
    """Map ``COCOeval``'s twelve statistics onto explicit names."""
    if len(stats) < len(COCO_STAT_KEYS):
        raise ScoringError(f"expected {len(COCO_STAT_KEYS)} COCO stats, got {len(stats)}")
    return {key: float(value) for key, value in zip(COCO_STAT_KEYS, stats, strict=False)}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def score_predictions(
    predictions: Sequence[Mapping[str, Any]] | Any,
    annotation_path: str | Path,
    *,
    image_ids: Sequence[int] | None = None,
    max_dets: int = DEFAULT_MAX_DETS,
) -> ScoringResult:
    """Score COCO detection results against an annotation file.

    Args:
        predictions: COCO result records, or an ``N x 7`` numpy array
            ``[image_id, x, y, w, h, score, category_id]``.
        annotation_path: COCO ground-truth JSON.
        image_ids: restrict evaluation to these images (the images the run evaluated).
        max_dets: per-image detection cap. Must be ``100`` so the twelve statistics keep their slots.
    """
    if max_dets != DEFAULT_MAX_DETS:
        raise ScoringError("max_dets must be 100 for the twelve named COCO statistics")
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ScoringUnavailable("scoring requires faster-coco-eval: install skellybench[scoring]") from exc

    annotation = Path(annotation_path)
    if not annotation.is_file():
        raise ScoringError(f"annotation file is missing: {annotation}")
    try:
        coco_gt = COCO(str(annotation))
        is_array = hasattr(predictions, "shape")
        empty = (predictions.shape[0] == 0) if is_array else not predictions
        if empty:
            coco_dt = COCO()
            coco_dt.dataset = {"images": [], "annotations": [], "categories": []}
            coco_dt.createIndex()
        else:
            coco_dt = coco_gt.loadRes(predictions if is_array else [dict(record) for record in predictions])
        evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        allowed = (
            sorted({int(image_id) for image_id in image_ids})
            if image_ids is not None
            else [int(image_id) for image_id in coco_gt.getImgIds()]
        )
        evaluator.params.imgIds = allowed
        evaluator.params.catIds = [int(category_id) for category_id in coco_gt.getCatIds()]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = metrics_from_stats(evaluator.stats)
    except ScoringError:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific failures
        raise ScoringError(f"failed to score predictions: {exc}") from exc
    return ScoringResult(
        metrics=metrics,
        images_evaluated=len(allowed),
        annotation_sha256=sha256_file(annotation),
        version=_package_version(SCORING_TOOL),
    )


def score_columns(
    columns: PredictionColumns,
    annotation_path: str | Path,
    *,
    max_dets: int = DEFAULT_MAX_DETS,
) -> ScoringResult:
    """Score decoded predictions columns over exactly their own image ids."""
    return score_predictions(
        columns.to_numpy(),
        annotation_path,
        image_ids=columns.image_ids,
        max_dets=max_dets,
    )


def compare_metrics(
    parent: Mapping[str, float],
    child: Mapping[str, float],
    *,
    warn_ap: float = DEFAULT_WARN_AP,
    fail_ap: float = DEFAULT_FAIL_AP,
) -> dict[str, Any]:
    """Return the ``comparison`` block of a model report: ``child - parent`` for every metric.

    The status is ``fail`` when ``|delta AP| > fail_ap``, ``warn`` when it is above ``warn_ap``,
    else ``ok``.
    """
    missing = [key for key in COCO_STAT_KEYS if key not in parent or key not in child]
    if missing:
        raise ScoringError(f"metrics are missing {missing}")
    deltas = {f"delta_{key}": round(float(child[key]) - float(parent[key]), 6) for key in COCO_STAT_KEYS}
    delta_ap = abs(deltas["delta_ap"])
    status = "fail" if delta_ap > fail_ap else "warn" if delta_ap > warn_ap else "ok"
    return {"status": status, "deltas": deltas, "thresholds": {"warn_ap": warn_ap, "fail_ap": fail_ap}}


__all__ = [
    "DEFAULT_FAIL_AP",
    "DEFAULT_MAX_DETS",
    "DEFAULT_WARN_AP",
    "SCORING_TOOL",
    "ScoringError",
    "ScoringResult",
    "ScoringUnavailable",
    "compare_metrics",
    "metrics_from_stats",
    "score_columns",
    "score_predictions",
]
