"""Per-run benchmark summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .._internal.hashing import describe_file
from .._internal.io import read_json
from .._internal.timestamps import utc_now

from ..file_writer import write_html_file, write_json_file
from ..html_reports.summary import render_run_summary_html
from ..schema import (
    BENCHMARK_SUMMARY_KIND,
    SKELLYBENCH_SCHEMA_VERSION,
    validate_benchmark_report,
    validate_benchmark_summary,
)


def summary_paths(output_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    """Return ``(json_path, html_path)`` for a run summary."""
    root = Path(output_dir)
    stem = f"benchmark-summary-{run_id}"
    return root / f"{stem}.json", root / f"{stem}.html"


def _best_images_per_second(performance: Mapping[str, Any]) -> float | None:
    values = [
        float((record.get("timing") or {}).get("images_per_second"))
        for record in performance.get("records") or []
        if (record.get("timing") or {}).get("images_per_second") is not None
    ]
    return max(values) if values else None


def build_run_summary(
    *,
    run: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    report_paths: Mapping[str, Path] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a run summary from already-validated benchmark report payloads."""
    models: list[dict[str, Any]] = []
    unavailable: dict[str, dict[str, Any]] = {}
    for report in reports:
        validate_benchmark_report(report)
        model = report.get("model") or {}
        performance = report.get("performance") or {}
        report_id = str(report.get("report_id"))
        path = (report_paths or {}).get(report_id)
        report_description = describe_file(path) if path and path.is_file() else {}
        best = _best_images_per_second(performance)
        model_entry = {
            "model_id": model.get("id"),
            "artifact": {
                "filename": (model.get("artifact") or {}).get("filename"),
                "sha256": (model.get("artifact") or {}).get("sha256"),
            },
            "report": path.name if path else f"{report_id}.json",
            "model_report": (report.get("model_report") or {}).get("filename"),
            "performance_status": "measured" if performance.get("records") else "unavailable",
            "summary": {
                "images_per_second": best,
                "record_count": len(performance.get("records") or []),
                "unavailable_count": len(performance.get("unavailable") or []),
            },
        }
        if report_description.get("sha256"):
            model_entry["report_sha256"] = report_description["sha256"]
        models.append(model_entry)
        for item in performance.get("unavailable") or []:
            unavailable[str(item.get("provider"))] = dict(item)

    models.sort(key=lambda entry: str(entry.get("model_id") or ""))
    payload = {
        "schema_version": SKELLYBENCH_SCHEMA_VERSION,
        "kind": BENCHMARK_SUMMARY_KIND,
        "run_id": str(run.get("run_id")),
        "generated_at": (generated_at or utc_now()).isoformat(),
        "run": dict(run),
        "models": models,
        "unavailable": [unavailable[key] for key in sorted(unavailable)],
    }
    validate_benchmark_summary(payload)
    return payload


def write_run_summary(
    output_dir: str | Path,
    *,
    run: Mapping[str, Any],
    report_files: Sequence[str | Path],
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Read benchmark reports, build a summary, and write JSON/HTML."""
    paths = [Path(path) for path in report_files]
    report_paths: dict[str, Path] = {}
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        validate_benchmark_report(payload)
        reports.append(payload)
        report_paths[str(payload["report_id"])] = path
    summary = build_run_summary(
        run=run,
        reports=reports,
        report_paths=report_paths,
        generated_at=generated_at,
    )
    json_path, html_path = summary_paths(output_dir, str(run["run_id"]))
    write_json_file(json_path, summary)
    write_html_file(html_path, render_run_summary_html(summary))
    return json_path, html_path


def rewrite_summary_html(output_root: str | Path) -> int:
    """Re-render every run-summary HTML below *output_root*."""
    root = Path(output_root)
    written = 0
    for summary_file in sorted(root.rglob("benchmark-summary-*.json")) if root.is_dir() else []:
        payload = read_json(summary_file)
        if not isinstance(payload, dict) or payload.get("kind") != BENCHMARK_SUMMARY_KIND:
            continue
        try:
            validate_benchmark_summary(payload)
        except Exception:
            continue
        write_html_file(summary_file.with_suffix(".html"), render_run_summary_html(payload))
        written += 1
    return written


__all__ = [
    "build_run_summary",
    "rewrite_summary_html",
    "summary_paths",
    "write_run_summary",
]
