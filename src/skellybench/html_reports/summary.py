"""HTML rendering for per-run benchmark summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import escape


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
        rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    unavailable = payload.get("unavailable") or []
    unavailable_html = ""
    if unavailable:
        unavailable_html = (
            "<h2>Unavailable providers</h2><ul>"
            + "".join(
                f"<li>{escape(item.get('provider'))}: {escape(item.get('reason'))}</li>"
                for item in unavailable
            )
            + "</ul>"
        )
    hardware = run.get("hardware") or run.get("machine") or {}
    cpu = hardware.get("cpu") or {}
    libraries = run.get("libraries") or {}
    cuda = libraries.get("cuda") or {}
    environment_details = [
        (label, value)
        for label, value in (
            ("CPU", cpu.get("model")),
            ("PyTorch CUDA build", cuda.get("torch_cuda_build")),
            ("cuDNN runtime", cuda.get("cudnn_runtime_version")),
        )
        if value is not None
    ]
    environment_html = (
        "<p class=\"muted\">"
        + " · ".join(
            f"{escape(label)}: {escape(value)}" for label, value in environment_details
        )
        + "</p>"
        if environment_details
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark summary {escape(payload.get("run_id", ""))}</title>
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
<p class="muted">Run {escape(payload.get('run_id', ''))} · {escape(hardware.get('label', ''))} · generated {escape(payload.get('generated_at', ''))}</p>
{environment_html}
<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>
{unavailable_html}
</body>
</html>
"""


__all__ = ["render_run_summary_html"]
