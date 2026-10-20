"""Filesystem layout for model reports and their overview."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._internal.timestamps import utc_stamp

MODEL_REPORT_JSON_SUFFIX = ".model.report.json"
MODEL_REPORT_HTML_SUFFIX = ".model.report.html"
MODEL_OVERVIEW_STEM = "model-report-overview"


@dataclass(frozen=True)
class ModelReportPaths:
    """The base name and the two files that make up a model report."""

    base: Path
    json_path: Path
    html_path: Path


def _strip_known_suffix(base: Path) -> Path:
    name = base.name
    for suffix in (MODEL_REPORT_JSON_SUFFIX, MODEL_REPORT_HTML_SUFFIX):
        if name.endswith(suffix):
            return base.with_name(name[: -len(suffix)])
    return base


def model_report_paths(base: str | Path) -> ModelReportPaths:
    """Return the JSON/HTML paths for a report base.

    The base may be an artifact path (``.../rfdetr-nano_b1_fp32.onnx``), a checkpoint path, or a
    synthetic per-batch source stem. Passing an existing report path is also accepted and stripped
    back to its base, which makes ``--render-reports`` idempotent.
    """
    path = _strip_known_suffix(Path(base))
    return ModelReportPaths(
        base=path,
        json_path=path.with_name(path.name + MODEL_REPORT_JSON_SUFFIX),
        html_path=path.with_name(path.name + MODEL_REPORT_HTML_SUFFIX),
    )


def model_overview_paths(
    exports_root: str | Path,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Return ``(json_path, html_path)`` for the model-report overview."""
    root = Path(exports_root)
    stem = f"{MODEL_OVERVIEW_STEM}-{utc_stamp(generated_at)}"
    return root / f"{stem}.json", root / f"{stem}.html"
