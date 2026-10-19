"""Per-run benchmark summaries."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from report_core import atomic_write_json, atomic_write_text, describe_file, read_json, utc_now

from .schema import (
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


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_run_summary_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained run summary."""
    run = payload.get("run") or {}
    models = payload.get("models") or []
    header = (
        "<tr><th scope=\"col\">Model</th><th scope=\"col\">Artifact</th>"
        "<th scope=\"col\">Report</th><th scope=\"col\">Status</th>"
        "<th scope=\"col\">Best images/s</th><th scope=\"col\">Records</th></tr>"
    )
    rows = []
    for entry in models:
        summary = entry.get("summary") or {}
        cells = (
            entry.get("model_id"),
            (entry.get("artifact") or {}).get("filename"),
            entry.get("report"),
            entry.get("performance_status"),
            summary.get("images_per_second"),
            summary.get("record_count"),
        )
        rows.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
    unavailable = payload.get("unavailable") or []
    unavailable_html = ""
    if unavailable:
        unavailable_html = (
            "<h2>Unavailable providers</h2><ul>"
            + "".join(
                f"<li>{_escape(item.get('provider'))}: {_escape(item.get('reason'))}</li>"
                for item in unavailable
            )
            + "</ul>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark summary {_escape(payload.get("run_id", ""))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 2rem auto; max-width: 78rem; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border-bottom: 1px solid rgba(128,128,128,.35); padding: .35rem .55rem; text-align: left; }}
  th {{ background: Canvas; }}
  .muted {{ opacity: .7; }}
</style>
</head>
<body>
<h1>Benchmark summary</h1>
<p class="muted">Run {_escape(payload.get('run_id', ''))} · {_escape(run.get('machine', {}).get('label', ''))} · generated {_escape(payload.get('generated_at', ''))}</p>
<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>
{unavailable_html}
</body>
</html>
"""


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
    atomic_write_json(json_path, summary)
    atomic_write_text(html_path, render_run_summary_html(summary))
    return json_path, html_path


__all__ = [
    "build_run_summary",
    "render_run_summary_html",
    "summary_paths",
    "write_run_summary",
]
