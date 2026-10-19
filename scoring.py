"""Library-agnostic scoring for COCO detection results.

The scoring back end is optional: importing this module does not import ``faster_coco_eval``. When
predictions and annotations are available, COCO statistics are recomputed. When only precomputed
metrics are available (for example an imported bundle), those metrics are accepted as provenance
and never relabeled as locally recomputed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from report_core import sha256_file

from .schema import (
    COCO_STAT_KEYS,
    PASS_IDS,
    SCORING_MODES,
    SkellybenchValidationError,
    validate_accuracy,
    validate_prediction_record,
    validate_predictions_store,
)

DEFAULT_MAX_DETS = 100
DEFAULT_WARN_AP = 0.001
DEFAULT_FAIL_AP = 0.01


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
    tool: str = "faster_coco_eval"
    version: str = ""
    scoring_mode: str = "recomputed"

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
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _load_records(store: Mapping[str, Any], base_dir: Path | None) -> list[dict[str, Any]]:
    validate_predictions_store(store)
    storage = str(store["storage"])
    if storage == "inline":
        return [dict(record) for record in store["records"]]

    path = Path(str(store["path"]))
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ScoringError(f"prediction store is missing: {path}")
    real_sha = sha256_file(path)
    if real_sha != store["sha256"]:
        raise ScoringError(
            f"prediction store sha256 mismatch: expected {store['sha256']}, got {real_sha}"
        )
    if storage == "json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScoringError(f"cannot read prediction JSON: {path}") from exc
        if not isinstance(payload, list):
            raise ScoringError("prediction JSON must contain an array of COCO result records")
        records = payload
    else:
        records = []
        try:
            with open(path, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ScoringError(
                            f"invalid prediction JSONL at {path}:{line_number}"
                        ) from exc
        except OSError as exc:
            raise ScoringError(f"cannot read prediction JSONL: {path}") from exc

    if len(records) != int(store["record_count"]):
        raise ScoringError(
            f"prediction record_count mismatch: expected {store['record_count']}, got {len(records)}"
        )
    for index, record in enumerate(records):
        validate_prediction_record(record, f"prediction[{index}]")
    return [dict(record) for record in records]


def score_predictions(
    records: Sequence[Mapping[str, Any]],
    annotation_path: str | Path,
    *,
    max_dets: int = DEFAULT_MAX_DETS,
    score_threshold: float = 0.0,
    image_ids: Sequence[int] | None = None,
) -> ScoringResult:
    """Score COCO detection records against an annotation file.

    Args:
        records: COCO detection results (``image_id``, ``category_id``, ``bbox``, ``score``).
        annotation_path: COCO ground-truth JSON.
        max_dets: Per-image detection cap. Schema v1 expects ``100`` so the twelve named COCO
            statistics map to their conventional slots.
        score_threshold: Drop records below this score before evaluation.
        image_ids: Optional restriction to a set of image ids.

    Raises:
        ScoringUnavailable: ``faster_coco_eval`` is not installed.
        ScoringError: The annotation/predictions are invalid or cannot be scored.
    """
    if max_dets != DEFAULT_MAX_DETS:
        raise ScoringError("schema v1 requires max_dets=100 for the twelve named COCO statistics")
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ScoringUnavailable(
            "scoring requires faster-coco-eval; install it with `pip install faster-coco-eval`"
        ) from exc

    annotation = Path(annotation_path)
    if not annotation.is_file():
        raise ScoringError(f"annotation file is missing: {annotation}")

    filtered = [
        dict(record)
        for record in records
        if float(record.get("score", 0.0)) >= float(score_threshold)
    ]
    for index, record in enumerate(filtered):
        validate_prediction_record(record, f"prediction[{index}]")

    try:
        coco_gt = COCO(str(annotation))
        if filtered:
            coco_dt = coco_gt.loadRes(filtered, min_score=0.0)
        else:
            # loadRes([]) is not accepted by every pycocotools-compatible implementation.
            coco_dt = COCO()
            coco_dt.dataset = {"images": [], "annotations": [], "categories": []}
            coco_dt.createIndex()

        evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        if image_ids is not None:
            allowed = sorted({int(image_id) for image_id in image_ids})
        else:
            allowed = [int(image_id) for image_id in coco_gt.getImgIds()]
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
        version=_package_version("faster-coco-eval"),
    )


def _report_safe_predictions(store: Any) -> dict[str, Any] | None:
    """Strip input-only local paths from a prediction store."""
    if not isinstance(store, Mapping):
        return None
    result = copy.deepcopy(dict(store))
    path_value = result.pop("path", None)
    if path_value:
        result["filename"] = Path(str(path_value)).name
    return result


def _compare(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    warn_ap: float = DEFAULT_WARN_AP,
    fail_ap: float = DEFAULT_FAIL_AP,
) -> dict[str, Any]:
    common = [key for key in COCO_STAT_KEYS if key in left and key in right]
    deltas = {f"delta_{key}": float(right[key]) - float(left[key]) for key in common}
    delta_ap = deltas.get("delta_ap")
    if delta_ap is None:
        status = "not_available"
    elif abs(delta_ap) > fail_ap:
        status = "fail"
    elif abs(delta_ap) > warn_ap:
        status = "warn"
    else:
        status = "ok"
    return {
        "left_pass": "python",
        "right_pass": "onnx",
        "status": status,
        "deltas": deltas,
        "thresholds": {"warn_ap": warn_ap, "fail_ap": fail_ap},
    }


def score_accuracy(
    accuracy: Mapping[str, Any],
    *,
    annotation_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    warn_ap: float = DEFAULT_WARN_AP,
    fail_ap: float = DEFAULT_FAIL_AP,
) -> dict[str, Any]:
    """Score an accuracy block without requiring a local annotation file.

    Behavior:

    * predictions + annotation -> recompute metrics;
    * predictions only -> status becomes ``unavailable`` with a reason;
    * precomputed metrics only -> accepted as provenance;
    * neither -> status becomes ``unavailable``.

    The returned copy is report-safe: local prediction paths are replaced by filenames.
    """
    validate_accuracy(accuracy)
    result = copy.deepcopy(dict(accuracy))
    if result.get("status") != "measured":
        result.setdefault("scoring", "unavailable" if result.get("status") != "skipped" else "precomputed")
        return result

    base_path = Path(base_dir) if base_dir is not None else None
    annotation = Path(annotation_path) if annotation_path is not None else None
    annotation_available = annotation is not None and annotation.is_file()
    evaluation = result.get("evaluation") or {}
    max_dets = int(evaluation.get("max_dets", DEFAULT_MAX_DETS))
    score_threshold = float(evaluation.get("score_threshold", 0.0))

    modes: set[str] = set()
    reasons: list[str] = []
    for pass_entry in result.get("passes") or []:
        if pass_entry.get("status") != "measured":
            if not pass_entry.get("reason"):
                raise ScoringError(f"pass {pass_entry.get('pass_id')!r} is not measured but has no reason")
            modes.add("unavailable")
            continue

        pass_id = str(pass_entry["pass_id"])
        precomputed = pass_entry.get("metrics")
        predictions = pass_entry.get("predictions")
        pass_entry["predictions"] = _report_safe_predictions(predictions)

        if predictions is not None and annotation_available:
            records = _load_records(predictions, base_path)
            try:
                scored = score_predictions(
                    records,
                    annotation,
                    max_dets=max_dets,
                    score_threshold=score_threshold,
                )
            except ScoringUnavailable as exc:
                if precomputed is not None:
                    pass_entry["scoring"] = "precomputed"
                    pass_entry["scoring_reason"] = str(exc)
                    modes.add("precomputed")
                    continue
                pass_entry["status"] = "unavailable"
                pass_entry["reason"] = str(exc)
                pass_entry.pop("metrics", None)
                modes.add("unavailable")
                reasons.append(f"{pass_id}: {exc}")
                continue
            metrics_block = scored.as_metrics_block()
            if precomputed is not None:
                pre_values = (precomputed.get("values") or {})
                metrics_block["precomputed_delta"] = {
                    key: float(scored.metrics[key]) - float(pre_values[key])
                    for key in COCO_STAT_KEYS
                    if key in pre_values
                }
            pass_entry["metrics"] = metrics_block
            pass_entry["scoring"] = "recomputed"
            modes.add("recomputed")
            result["evaluation"] = {**evaluation, "images_evaluated": scored.images_evaluated}
        elif precomputed is not None:
            pass_entry["scoring"] = "precomputed"
            modes.add("precomputed")
        elif predictions is not None:
            pass_entry["status"] = "unavailable"
            pass_entry["reason"] = "annotation file is unavailable; predictions were not scored"
            pass_entry.pop("metrics", None)
            modes.add("unavailable")
            reasons.append(f"{pass_id}: annotation file is unavailable")
        else:
            pass_entry["status"] = "unavailable"
            pass_entry["reason"] = "no predictions or metrics were supplied"
            modes.add("unavailable")
            reasons.append(f"{pass_id}: no predictions or metrics")

    measured = [entry for entry in result.get("passes") or [] if entry.get("status") == "measured"]
    if any(entry.get("status") == "error" for entry in result.get("passes") or []):
        result["status"] = "error"
    elif len(measured) == len(PASS_IDS):
        result["status"] = "measured"
    elif result.get("passes") and all(entry.get("status") == "skipped" for entry in result["passes"]):
        result["status"] = "skipped"
    else:
        result["status"] = "unavailable"
    if result["status"] != "measured":
        result["reason"] = "; ".join(reasons) or "accuracy passes were not all measured"

    if result["status"] == "measured":
        by_id = {entry["pass_id"]: entry for entry in result["passes"]}
        left_values = (by_id["python"].get("metrics") or {}).get("values") or {}
        right_values = (by_id["onnx"].get("metrics") or {}).get("values") or {}
        result["comparisons"] = [_compare(left_values, right_values, warn_ap=warn_ap, fail_ap=fail_ap)]
        if "recomputed" in modes:
            result["scoring"] = "recomputed"
        else:
            result["scoring"] = "precomputed"
    else:
        result["scoring"] = "unavailable"
        result["comparisons"] = []

    assert result["scoring"] in SCORING_MODES
    return result


__all__ = [
    "DEFAULT_FAIL_AP",
    "DEFAULT_MAX_DETS",
    "DEFAULT_WARN_AP",
    "ScoringError",
    "ScoringResult",
    "ScoringUnavailable",
    "metrics_from_stats",
    "score_accuracy",
    "score_predictions",
]
