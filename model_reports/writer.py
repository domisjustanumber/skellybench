"""Writing and rebuilding model reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from report_core import read_json

from ..file_writer import write_html_file, write_json_file
from ..html_reports.model import render_model_report_html
from .completeness import (
    check_model_report,
    model_report_requirement_error,
    read_model_report,
)
from .paths import MODEL_REPORT_HTML_SUFFIX, MODEL_REPORT_JSON_SUFFIX, ModelReportPaths
from .schema import (
    ModelReportIdentity,
    ModelReportValidationError,
    validate_model_report,
)


@dataclass(frozen=True)
class ModelReportWriteResult:
    """Outcome of a guarded model-report write."""

    paths: ModelReportPaths
    written: bool
    skipped: bool
    reason: str
    payload: dict[str, Any] | None = None


def write_model_report(
    paths: ModelReportPaths,
    payload: dict[str, Any],
    *,
    force: bool = False,
    identity: ModelReportIdentity | None = None,
    require_accuracy: bool = False,
    require_performance: bool = False,
    required_accuracy_status: str | None = None,
) -> ModelReportWriteResult:
    """Write a model report unless a complete one already exists.

    Args:
        paths: Target JSON/HTML paths.
        payload: Validated model-report payload.
        force: Overwrite a complete existing report.
        identity: Provenance fields that must match an existing report to count as complete.
        require_accuracy: Require an accuracy section before skipping.
        require_performance: Require at least one benchmark snapshot before skipping.
        required_accuracy_status: Require a specific accuracy status before skipping.

    Returns:
        A result indicating whether files were written or skipped.
    """
    validate_model_report(payload)
    requirement = model_report_requirement_error(
        payload,
        require_accuracy=require_accuracy,
        require_performance=require_performance,
        required_accuracy_status=required_accuracy_status,
    )
    if requirement is not None:
        raise ModelReportValidationError(requirement)
    check = check_model_report(
        paths,
        identity=identity,
        require_accuracy=require_accuracy,
        require_performance=require_performance,
        required_accuracy_status=required_accuracy_status,
    )
    if check.complete and not force:
        return ModelReportWriteResult(
            paths=paths,
            written=False,
            skipped=True,
            reason=check.reason,
            payload=check.payload,
        )

    write_json_file(paths.json_path, payload)
    write_html_file(paths.html_path, render_model_report_html(payload))
    return ModelReportWriteResult(
        paths=paths,
        written=True,
        skipped=False,
        reason="forced rewrite" if force and check.complete else "written",
        payload=payload,
    )


def ensure_model_report(
    paths: ModelReportPaths,
    factory: Callable[[], dict[str, Any]],
    *,
    force: bool = False,
    identity: ModelReportIdentity | None = None,
    require_accuracy: bool = False,
    require_performance: bool = False,
    required_accuracy_status: str | None = None,
) -> ModelReportWriteResult:
    """Generate a model report only when needed.

    *factory* is not called when a complete report already exists and *force* is false. This is the
    entry point the export pipeline uses so measurement work can be skipped before it starts.
    """
    check = check_model_report(
        paths,
        identity=identity,
        require_accuracy=require_accuracy,
        require_performance=require_performance,
        required_accuracy_status=required_accuracy_status,
    )
    if check.complete and not force:
        return ModelReportWriteResult(
            paths=paths,
            written=False,
            skipped=True,
            reason=check.reason,
            payload=check.payload,
        )

    payload = factory()
    return write_model_report(
        paths,
        payload,
        force=force,
        identity=identity,
        require_accuracy=require_accuracy,
        require_performance=require_performance,
        required_accuracy_status=required_accuracy_status,
    )


def render_model_report_file(paths: ModelReportPaths) -> Path:
    """Re-render only the HTML side of an existing report.

    The model artifact does not need to exist: everything rendered comes from the report JSON.
    """
    payload = read_model_report(paths)
    if payload is None:
        raise ModelReportValidationError(f"{paths.json_path}: model report JSON is missing or unparseable")
    validate_model_report(payload)
    return write_html_file(paths.html_path, render_model_report_html(payload))


def rewrite_model_report_html(exports_root: str | Path) -> int:
    """Re-render every model report's HTML below *exports_root*, returning the count written."""
    root = Path(exports_root)
    written = 0
    for report_file in sorted(root.rglob(f"*{MODEL_REPORT_JSON_SUFFIX}")) if root.is_dir() else []:
        payload = read_json(report_file)
        if not isinstance(payload, dict):
            continue
        try:
            validate_model_report(payload)
        except ModelReportValidationError:
            continue
        html_path = report_file.with_name(
            report_file.name[: -len(MODEL_REPORT_JSON_SUFFIX)] + MODEL_REPORT_HTML_SUFFIX
        )
        write_html_file(html_path, render_model_report_html(payload))
        written += 1
    return written
