"""Generate-once completeness checks for model reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._internal.io import read_json

from .paths import ModelReportPaths
from .schema import (
    ModelReportIdentity,
    ModelReportValidationError,
    validate_model_report,
)


@dataclass(frozen=True)
class ModelReportCheck:
    """Result of inspecting a model report on disk."""

    paths: ModelReportPaths
    complete: bool
    reason: str
    payload: dict[str, Any] | None = None


def read_model_report(paths: ModelReportPaths) -> dict[str, Any] | None:
    """Read a model-report JSON, returning ``None`` when absent or unparseable."""
    payload = read_json(paths.json_path)
    return payload if isinstance(payload, dict) else None


def _identity_reason(
    payload: dict[str, Any],
    identity: ModelReportIdentity | None,
) -> str | None:
    if identity is None:
        return None
    if payload.get("report_id") != identity.report_id:
        return f"report_id differs (expected {identity.report_id!r})"

    model = payload.get("model") or {}
    checks = (
        ("model_id", identity.model_id),
        ("runtime", identity.runtime),
        ("precision", identity.precision),
        ("batch_size", identity.batch_size),
    )
    for field, expected in checks:
        if expected is not None and model.get(field) != expected:
            return f"model.{field} differs (expected {expected!r})"

    artifact = model.get("artifact") or {}
    if identity.artifact_sha256 is not None and artifact.get("sha256") != identity.artifact_sha256:
        return "artifact sha256 differs"
    if (
        identity.source_fp32_sha256 is not None
        and artifact.get("source_fp32_sha256") != identity.source_fp32_sha256
    ):
        return "source fp32 sha256 differs"
    if identity.source_md5 is not None:
        source = (payload.get("provenance") or {}).get("source") or {}
        if source.get("md5") != identity.source_md5:
            return "source checkpoint md5 differs"
    return None


def model_report_requirement_error(
    payload: dict[str, Any],
    *,
    require_accuracy: bool = False,
    require_performance: bool = False,
    required_accuracy_status: str | None = None,
) -> str | None:
    """Return a reason when a payload lacks a caller-required section, else ``None``."""
    scoring = payload.get("scoring") or {}
    if require_accuracy and "accuracy" not in scoring:
        return "accuracy section is missing"
    if required_accuracy_status is not None:
        actual_status = (scoring.get("accuracy") or {}).get("status")
        if actual_status != required_accuracy_status:
            return f"accuracy status is {actual_status!r}, expected {required_accuracy_status!r}"
    if require_performance:
        benchmarks = (scoring.get("performance") or {}).get("benchmarks") or []
        if not benchmarks:
            return "performance snapshots are missing"
    return None


def check_model_report(
    paths: ModelReportPaths,
    *,
    identity: ModelReportIdentity | None = None,
    require_accuracy: bool = False,
    require_performance: bool = False,
    required_accuracy_status: str | None = None,
) -> ModelReportCheck:
    """Decide whether a model report exists, is valid, and satisfies the requested sections.

    The check never requires the model artifact itself to be present. A report is a self-contained
    provenance/scoring record, so it can be re-rendered or audited on a machine that does not have
    the ``.onnx``/``.pth`` file.
    """
    if not paths.json_path.is_file() or paths.json_path.stat().st_size == 0:
        return ModelReportCheck(paths, False, "model report JSON is missing or empty")
    if not paths.html_path.is_file() or paths.html_path.stat().st_size == 0:
        return ModelReportCheck(paths, False, "model report HTML is missing or empty")

    payload = read_model_report(paths)
    if payload is None:
        return ModelReportCheck(paths, False, "model report JSON is unparseable")

    try:
        validate_model_report(payload)
    except ModelReportValidationError as exc:
        return ModelReportCheck(paths, False, f"model report is invalid: {exc}", payload)

    reason = _identity_reason(payload, identity)
    if reason is not None:
        return ModelReportCheck(paths, False, reason, payload)

    requirement = model_report_requirement_error(
        payload,
        require_accuracy=require_accuracy,
        require_performance=require_performance,
        required_accuracy_status=required_accuracy_status,
    )
    if requirement is not None:
        return ModelReportCheck(paths, False, requirement, payload)

    return ModelReportCheck(paths, True, "complete", payload)
