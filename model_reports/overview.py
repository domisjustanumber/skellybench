"""Cross-machine overview built by rescanning model reports on disk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from report_core import describe_file, read_json, utc_now

from ..file_writer import write_html_file, write_json_file
from ..html_reports.model import render_model_overview_html
from .paths import (
    MODEL_OVERVIEW_STEM,
    MODEL_REPORT_JSON_SUFFIX,
    model_overview_paths,
)
from .schema import (
    MODEL_OVERVIEW_KIND,
    MODEL_REPORT_SCHEMA_VERSION,
    ModelReportValidationError,
    validate_model_report,
)


def _iter_model_reports(exports_root: Path) -> list[Path]:
    return sorted(exports_root.rglob(f"*{MODEL_REPORT_JSON_SUFFIX}"))


def _best_images_per_second(snapshot: dict[str, Any]) -> float | None:
    performance = snapshot.get("performance")
    container = performance if isinstance(performance, dict) else snapshot
    records = container.get("records") if isinstance(container, dict) else None
    values: list[float] = []
    iterable = records.values() if isinstance(records, dict) else records if isinstance(records, list) else []
    for record in iterable:
        if not isinstance(record, dict):
            continue
        value = (record.get("timing") or {}).get("images_per_second")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return max(values) if values else None


def _report_row(
    payload: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    model = payload.get("model") or {}
    artifact = model.get("artifact") or {}
    scoring = payload.get("scoring") or {}
    accuracy = scoring.get("accuracy") or {}
    performance = scoring.get("performance") or {}

    measured_passes = [
        pass_entry
        for pass_entry in accuracy.get("passes") or []
        if isinstance(pass_entry, dict) and pass_entry.get("status") == "measured"
    ]
    ap = None
    if measured_passes:
        ap_pass = next(
            (entry for entry in measured_passes if entry.get("pass_id") == "onnx"),
            measured_passes[0],
        )
        values = (ap_pass.get("metrics") or {}).get("values") or {}
        ap = values.get("ap")

    delta_ap = None
    for comparison in accuracy.get("comparisons") or []:
        if (
            isinstance(comparison, dict)
            and comparison.get("left_pass") == "python"
            and comparison.get("right_pass") == "onnx"
        ):
            delta_ap = (comparison.get("deltas") or {}).get("delta_ap")
            break

    benchmarks = performance.get("benchmarks") or []
    best_values = [
        value
        for entry in benchmarks
        if isinstance(entry, dict)
        for value in [_best_images_per_second(entry.get("snapshot") or {})]
        if value is not None
    ]

    report_file = describe_file(report_path)
    return {
        "report_id": payload.get("report_id"),
        "model_id": model.get("model_id"),
        "runtime": model.get("runtime"),
        "precision": model.get("precision"),
        "batch_size": model.get("batch_size"),
        "accuracy_status": accuracy.get("status"),
        "ap": ap,
        "delta_ap": delta_ap,
        "benchmark_count": len(benchmarks),
        "best_images_per_second": max(best_values) if best_values else None,
        "artifact_filename": artifact.get("filename"),
        "artifact_sha256": artifact.get("sha256"),
        "generated_at": payload.get("generated_at"),
        "report": report_file["filename"],
        "report_sha256": report_file.get("sha256"),
    }


def build_model_overview(
    exports_root: str | Path,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an overview payload by rescanning every model report under *exports_root*.

    Invalid or unparseable reports are skipped rather than aborting the whole overview. When the
    same ``report_id`` appears more than once, the newest ``generated_at`` wins.
    """
    root = Path(exports_root)
    reports: dict[str, tuple[str, dict[str, Any], Path]] = {}
    for report_path in _iter_model_reports(root) if root.is_dir() else []:
        payload = read_json(report_path)
        if not isinstance(payload, dict):
            continue
        try:
            validate_model_report(payload)
        except ModelReportValidationError:
            continue
        report_id = str(payload.get("report_id"))
        generated = str(payload.get("generated_at") or "")
        previous = reports.get(report_id)
        if previous is None or generated >= previous[0]:
            reports[report_id] = (generated, payload, report_path)

    rows = [
        _report_row(payload, report_path=report_path)
        for _generated, payload, report_path in reports.values()
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("model_id") or ""),
            row.get("batch_size") or 0,
            str(row.get("precision") or ""),
            str(row.get("runtime") or ""),
        )
    )

    return {
        "schema_version": MODEL_REPORT_SCHEMA_VERSION,
        "kind": MODEL_OVERVIEW_KIND,
        "generated_at": (generated_at or utc_now()).isoformat(),
        "report_count": len(reports),
        "row_count": len(rows),
        "rows": rows,
    }


def write_model_overview(
    exports_root: str | Path,
    *,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Rebuild the model overview from disk and write its JSON and HTML files."""
    payload = build_model_overview(exports_root, generated_at=generated_at)
    json_path, html_path = model_overview_paths(exports_root, generated_at)
    write_json_file(json_path, payload)
    write_html_file(html_path, render_model_overview_html(payload))
    return json_path, html_path


__all__ = [
    "MODEL_OVERVIEW_STEM",
    "build_model_overview",
    "write_model_overview",
]
