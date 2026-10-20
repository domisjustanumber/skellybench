"""Benchmark report identity, construction, and report-file lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .._internal.io import read_json
from .._internal.timestamps import utc_now

from ..file_writer import write_html_file, write_json_file
from ..html_reports.benchmark import render_benchmark_report_html
from ..schema import (
    BENCHMARK_REPORT_KIND,
    SKELLYBENCH_SCHEMA_VERSION,
    validate_benchmark_report,
)


def benchmark_report_id(artifact: str | Path, run_id: str) -> str:
    """Stable report id for one artifact/run pair."""
    return f"{Path(artifact).name}.benchmark-{run_id}"


def benchmark_report_paths(
    output_dir: str | Path,
    artifact: str | Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Return ``(json_path, html_path)`` for a benchmark report."""
    root = Path(output_dir)
    stem = benchmark_report_id(artifact, run_id)
    return root / f"{stem}.json", root / f"{stem}.html"


def build_benchmark_report(
    *,
    run: Mapping[str, Any],
    model: Mapping[str, Any],
    performance: Mapping[str, Any],
    report_id: str | None = None,
    generated_at: datetime | str | None = None,
    model_report: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a throughput-only benchmark report payload."""
    artifact = (model.get("artifact") or {}).get("filename")
    if not artifact:
        raise ValueError("model.artifact.filename is required to derive a benchmark report id")
    timestamp = (
        generated_at.isoformat()
        if isinstance(generated_at, datetime)
        else str(generated_at or utc_now().isoformat())
    )
    payload: dict[str, Any] = {
        "schema_version": SKELLYBENCH_SCHEMA_VERSION,
        "kind": BENCHMARK_REPORT_KIND,
        "report_id": report_id or benchmark_report_id(artifact, str(run["run_id"])),
        "generated_at": timestamp,
        "run": dict(run),
        "model": dict(model),
        "performance": dict(performance),
    }
    if model_report is not None:
        payload["model_report"] = dict(model_report)
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    validate_benchmark_report(payload)
    return payload


def write_benchmark_report(
    paths: tuple[Path, Path],
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Validate and write JSON and HTML benchmark report files."""
    validate_benchmark_report(payload)
    json_path, html_path = paths
    write_json_file(json_path, payload)
    write_html_file(html_path, render_benchmark_report_html(payload))
    return json_path, html_path


def rewrite_benchmark_report_html(output_root: str | Path) -> int:
    """Re-render every benchmark report HTML below *output_root*."""
    root = Path(output_root)
    written = 0
    for report_file in sorted(root.rglob("*.json")) if root.is_dir() else []:
        payload = read_json(report_file)
        if not isinstance(payload, dict) or payload.get("kind") != BENCHMARK_REPORT_KIND:
            continue
        try:
            validate_benchmark_report(payload)
        except Exception:
            continue
        write_html_file(report_file.with_suffix(".html"), render_benchmark_report_html(payload))
        written += 1
    return written


__all__ = [
    "benchmark_report_id",
    "benchmark_report_paths",
    "build_benchmark_report",
    "rewrite_benchmark_report_html",
    "write_benchmark_report",
]
