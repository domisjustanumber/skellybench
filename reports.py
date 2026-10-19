"""Throughput-only benchmark reports and their HTML renderer."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from report_core import atomic_write_json, atomic_write_text, format_number, utc_now

from .schema import (
    BENCHMARK_REPORT_KIND,
    SKELLYBENCH_SCHEMA_VERSION,
    validate_benchmark_report,
)


def benchmark_report_id(artifact: str | Path, run_id: str) -> str:
    """Stable report id for one artifact/run pair."""
    name = Path(artifact).name
    return f"{name}.benchmark-{run_id}"


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
    timestamp = generated_at.isoformat() if isinstance(generated_at, datetime) else str(
        generated_at or utc_now().isoformat()
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


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _table(caption: str, rows: list[tuple[Any, Any]]) -> str:
    body = "".join(
        f'<tr><th scope="row">{_escape(name)}</th><td>{_escape(value)}</td></tr>'
        for name, value in rows
    )
    return f"<table><caption>{_escape(caption)}</caption><tbody>{body}</tbody></table>"


def _performance_rows(performance: Mapping[str, Any]) -> str:
    records = performance.get("records") or []
    unavailable = performance.get("unavailable") or []
    header = (
        "<tr><th scope=\"col\">Provider</th><th scope=\"col\">Mode</th>"
        "<th scope=\"col\">Images/s</th><th scope=\"col\">ms/image</th>"
        "<th scope=\"col\">p50 batch ms</th><th scope=\"col\">p95 batch ms</th>"
        "<th scope=\"col\">CV</th><th scope=\"col\">Hardware</th></tr>"
    )
    body: list[str] = []
    for record in records:
        timing = record.get("timing") or {}
        cells = (
            record.get("provider"),
            record.get("mode"),
            format_number(timing.get("images_per_second"), 2),
            format_number(timing.get("ms_per_image"), 2),
            format_number(timing.get("batch_ms_p50"), 2),
            format_number(timing.get("batch_ms_p95"), 2),
            f"{float(timing.get('cv')) * 100.0:.1f}%" if timing.get("cv") is not None else "n/a",
            record.get("hardware_id"),
        )
        body.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
    records_table = (
        "<table><caption>Throughput records</caption>"
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
    )
    unavailable_rows = "".join(
        f"<tr><td>{_escape(item.get('provider'))}</td>"
        f"<td colspan=\"7\">{_escape(item.get('reason'))}</td></tr>"
        for item in unavailable
    )
    unavailable_table = (
        "<table><caption>Unavailable providers</caption>"
        "<thead><tr><th scope=\"col\">Provider</th><th scope=\"col\">Reason</th></tr></thead>"
        f"<tbody>{unavailable_rows}</tbody></table>"
        if unavailable_rows
        else ""
    )
    return records_table + unavailable_table


def render_benchmark_report_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained throughput report."""
    model = payload.get("model") or {}
    run = payload.get("run") or {}
    machine = run.get("machine") or {}
    software = run.get("software") or {}
    performance = payload.get("performance") or {}
    protocol = performance.get("protocol") or {}
    title = f"{model.get('id', 'model')} {model.get('precision', '')} benchmark {run.get('run_id', '')}".strip()

    model_rows = [
        ("Model", model.get("id")),
        ("Task", model.get("task")),
        ("Format", model.get("format")),
        ("Precision", model.get("precision")),
        ("Artifact", (model.get("artifact") or {}).get("filename")),
        ("Artifact sha256", (model.get("artifact") or {}).get("sha256")),
        ("Input signature", json.dumps(model.get("input_signature") or {}, sort_keys=True)),
        ("Input signature id", model.get("input_signature_id")),
    ]
    run_rows = [
        ("Run ID", run.get("run_id")),
        ("Generated", payload.get("generated_at")),
        ("Scenario", performance.get("scenario")),
        ("Machine", machine.get("label") or machine.get("id")),
        ("Runner", (run.get("runner") or {}).get("user")),
        ("Python", software.get("python")),
        ("ONNX Runtime", software.get("onnxruntime")),
    ]
    protocol_rows = [
        ("Protocol ID", protocol.get("protocol_id")),
        ("Images", protocol.get("images")),
        ("Warmup passes", protocol.get("warmup_passes")),
        ("Timed passes", protocol.get("passes")),
        ("Min pass seconds", protocol.get("min_pass_seconds")),
        ("Max pass seconds", protocol.get("max_pass_seconds")),
        ("CPU threads", protocol.get("cpu_threads")),
        ("Timing scope", protocol.get("timing_scope")),
    ]
    model_report = payload.get("model_report") or {}
    model_report_rows = [
        ("Report ID", model_report.get("report_id")),
        ("Filename", model_report.get("filename")),
        ("sha256", model_report.get("sha256")),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }}
  h1 {{ margin-bottom: .25rem; }}
  h2 {{ margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }}
  th, td {{ border-bottom: 1px solid rgba(128,128,128,.35); padding: .4rem .6rem; text-align: left; vertical-align: top; }}
  th[scope=row] {{ width: 18rem; font-weight: 600; }}
  caption {{ text-align: left; font-weight: 600; padding-bottom: .4rem; }}
  pre {{ overflow: auto; padding: .75rem; background: rgba(128,128,128,.12); }}
  .muted {{ opacity: .7; }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p class="muted">Throughput-only benchmark report · schema {_escape(payload.get("schema_version"))}</p>
<h2>Model</h2>
{_table("Artifact", model_rows)}
<h2>Run</h2>
{_table("Run and machine", run_rows)}
<h2>Protocol</h2>
{_table("Measurement protocol", protocol_rows)}
<h2>Results</h2>
{_performance_rows(performance)}
<h2>Linked model report</h2>
{_table("Model report reference", model_report_rows) if model_report else "<p>Not linked.</p>"}
</body>
</html>
"""


def write_benchmark_report(paths: tuple[Path, Path], payload: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write JSON and HTML benchmark report files."""
    validate_benchmark_report(payload)
    json_path, html_path = paths
    atomic_write_json(json_path, payload)
    atomic_write_text(html_path, render_benchmark_report_html(payload))
    return json_path, html_path


__all__ = [
    "benchmark_report_id",
    "benchmark_report_paths",
    "build_benchmark_report",
    "render_benchmark_report_html",
    "write_benchmark_report",
]
