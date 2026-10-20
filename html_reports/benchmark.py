"""HTML rendering for throughput benchmark reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from report_core import format_number

from .common import escape, table


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
        cv = timing.get("cv")
        cells = (
            record.get("provider"),
            record.get("mode"),
            format_number(timing.get("images_per_second"), 2),
            format_number(timing.get("ms_per_image"), 2),
            format_number(timing.get("batch_ms_p50"), 2),
            format_number(timing.get("batch_ms_p95"), 2),
            f"{float(cv) * 100.0:.1f}%" if cv is not None else "n/a",
            record.get("hardware_id"),
        )
        body.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    records_table = (
        "<table><caption>Throughput records</caption>"
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
    )
    unavailable_rows = "".join(
        f"<tr><td>{escape(item.get('provider'))}</td>"
        f"<td colspan=\"7\">{escape(item.get('reason'))}</td></tr>"
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
    hardware = run.get("hardware") or run.get("machine") or {}
    libraries = run.get("libraries") or {}
    legacy_software = run.get("software") or {}
    python = libraries.get("python") or {}
    package_versions = libraries.get("packages") or {}
    performance = payload.get("performance") or {}
    protocol = performance.get("protocol") or {}
    title = (
        f"{model.get('id', 'model')} {model.get('precision', '')} "
        f"benchmark {run.get('run_id', '')}"
    ).strip()

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
        ("Machine", hardware.get("label") or hardware.get("id")),
        ("Runner", (run.get("runner") or {}).get("user")),
        ("Python", python.get("version") or legacy_software.get("python")),
        (
            "ONNX Runtime",
            package_versions.get("onnxruntime")
            or package_versions.get("onnxruntime-gpu")
            or legacy_software.get("onnxruntime"),
        ),
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
<title>{escape(title)}</title>
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
<h1>{escape(title)}</h1>
<p class="muted">Throughput-only benchmark report · schema {escape(payload.get("schema_version"))}</p>
<h2>Model</h2>
{table("Artifact", model_rows)}
<h2>Run</h2>
{table("Run and machine", run_rows)}
<h2>Protocol</h2>
{table("Measurement protocol", protocol_rows)}
<h2>Results</h2>
{_performance_rows(performance)}
<h2>Linked model report</h2>
{table("Model report reference", model_report_rows) if model_report else "<p>Not linked.</p>"}
</body>
</html>
"""


__all__ = ["render_benchmark_report_html"]
