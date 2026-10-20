"""Self-contained HTML renderers for model reports and the model overview."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from report_core import format_filesize, format_number

from ..model_reports.schema import MODEL_OVERVIEW_KIND
from .common import escape as _escape
from .common import table as _table


def _json_details(value: Any, summary: str) -> str:
    rendered = _escape(json.dumps(value, indent=2, sort_keys=True))
    return f"<details><summary>{_escape(summary)}</summary><pre>{rendered}</pre></details>"


def _model_rows(payload: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    model = payload.get("model") or {}
    artifact = model.get("artifact") or {}
    rows: list[tuple[Any, Any]] = [
        ("Report ID", payload.get("report_id")),
        ("Model", model.get("model_id")),
        ("Runtime", model.get("runtime")),
        ("Precision", model.get("precision")),
        ("Batch size", model.get("batch_size")),
        ("Input signature", json.dumps(model.get("input_signature") or {}, sort_keys=True)),
    ]
    if artifact:
        rows.extend(
            [
                ("Artifact", artifact.get("filename")),
                ("Artifact sha256", artifact.get("sha256")),
                ("Artifact size", artifact.get("filesize_human") or format_filesize(artifact.get("size_bytes"))),
                ("Source fp32 sha256", artifact.get("source_fp32_sha256")),
            ]
        )
    return rows


def _provenance_rows(payload: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    provenance = payload.get("provenance") or {}
    source = provenance.get("source") or {}
    command = provenance.get("command") or {}
    rows: list[tuple[Any, Any]] = [
        ("Origin", provenance.get("origin")),
        ("Source kind", source.get("kind")),
        ("Source URL", source.get("url")),
        ("Source md5", source.get("md5")),
        ("Command", command.get("display") if isinstance(command, Mapping) else command),
    ]
    tools = provenance.get("tools")
    if tools:
        rows.append(("Tools", json.dumps(tools, sort_keys=True)))
    return rows


def _pass_rows(accuracy: Mapping[str, Any]) -> str:
    passes = accuracy.get("passes") or []
    if not passes:
        return "<p>No measured passes.</p>"
    header = (
        "<tr><th scope=\"col\">Pass</th><th scope=\"col\">Origin</th>"
        "<th scope=\"col\">Status</th><th scope=\"col\">AP</th>"
        "<th scope=\"col\">AP50</th><th scope=\"col\">AP75</th>"
        "<th scope=\"col\">AR100</th><th scope=\"col\">Tool</th></tr>"
    )
    body: list[str] = []
    for pass_entry in passes:
        metrics = pass_entry.get("metrics") or {}
        values = metrics.get("values") or {}
        tool = f"{metrics.get('tool', '')} {metrics.get('version', '')}".strip()
        cells = (
            pass_entry.get("pass_id"),
            pass_entry.get("origin"),
            pass_entry.get("status"),
            format_number(values.get("ap"), 4),
            format_number(values.get("ap50"), 4),
            format_number(values.get("ap75"), 4),
            format_number(values.get("ar100"), 4),
            tool,
        )
        body.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
    return (
        "<table><caption>Accuracy passes</caption>"
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
    )


def _comparison_rows(accuracy: Mapping[str, Any]) -> str:
    comparisons = accuracy.get("comparisons") or []
    if not comparisons:
        return ""
    header = (
        "<tr><th scope=\"col\">Left</th><th scope=\"col\">Right</th>"
        "<th scope=\"col\">Status</th><th scope=\"col\">ΔAP</th>"
        "<th scope=\"col\">ΔAP50</th><th scope=\"col\">Thresholds</th></tr>"
    )
    body: list[str] = []
    for comparison in comparisons:
        deltas = comparison.get("deltas") or {}
        cells = (
            comparison.get("left_pass"),
            comparison.get("right_pass"),
            comparison.get("status"),
            format_number(deltas.get("delta_ap"), 6),
            format_number(deltas.get("delta_ap50"), 6),
            json.dumps(comparison.get("thresholds") or {}, sort_keys=True),
        )
        body.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
    return (
        "<table><caption>Accuracy comparison</caption>"
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
    )


def _device_rows(accuracy: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    device = accuracy.get("device") or {}
    internal = device.get("internal_benchmark") or {}
    rows: list[tuple[Any, Any]] = [
        ("Accuracy status", accuracy.get("status")),
        ("Scenario", accuracy.get("scenario")),
        ("Device type", device.get("type")),
        ("Device index", device.get("index")),
        ("Torch device", device.get("torch_device")),
        ("ORT provider", device.get("ort_provider")),
        ("Matches internal benchmark", device.get("matches_internal")),
        ("Internal benchmark tool", internal.get("tool")),
        ("Internal benchmark device", internal.get("device")),
        ("Internal benchmark provider", internal.get("provider")),
    ]
    if accuracy.get("reason"):
        rows.append(("Reason", accuracy["reason"]))
    evaluation = accuracy.get("evaluation") or {}
    if evaluation:
        rows.extend(
            [
                ("Images evaluated", evaluation.get("images_evaluated")),
                ("IoU type", evaluation.get("iou_type")),
                ("Max detections", evaluation.get("max_dets")),
                ("Score threshold", evaluation.get("score_threshold")),
                ("Num select", evaluation.get("num_select")),
            ]
        )
    return rows


def _snapshot_records(snapshot: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    performance = snapshot.get("performance")
    container = performance if isinstance(performance, Mapping) else snapshot
    records = container.get("records") if isinstance(container, Mapping) else None
    if isinstance(records, Mapping):
        return [record for record in records.values() if isinstance(record, Mapping)]
    if isinstance(records, list):
        return [record for record in records if isinstance(record, Mapping)]
    return []


def _best_images_per_second(snapshot: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for record in _snapshot_records(snapshot):
        value = (record.get("timing") or {}).get("images_per_second")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return max(values) if values else None


def _performance_tables(performance: Mapping[str, Any]) -> str:
    benchmarks = performance.get("benchmarks") or []
    if not benchmarks:
        return "<p>No benchmark snapshots.</p>"
    header = (
        "<tr><th scope=\"col\">Run</th><th scope=\"col\">Scenario</th>"
        "<th scope=\"col\">Report</th><th scope=\"col\">Best images/s</th></tr>"
    )
    body: list[str] = []
    details: list[str] = []
    for entry in benchmarks:
        snapshot = entry.get("snapshot") or {}
        cells = (
            entry.get("run_id"),
            entry.get("scenario"),
            entry.get("filename") or entry.get("report_id"),
            format_number(_best_images_per_second(snapshot), 2),
        )
        body.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
        details.append(_json_details(snapshot, f"Snapshot {entry.get('run_id', '')}"))
    return (
        "<table><caption>Benchmark snapshots</caption>"
        f"<thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"
        + "".join(details)
    )


def render_model_report_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained model-report HTML document."""
    model = payload.get("model") or {}
    title = " ".join(
        str(part)
        for part in (
            model.get("model_id", "model"),
            model.get("runtime", ""),
            model.get("precision", ""),
            f"b{model.get('batch_size', '?')}",
        )
        if part
    )
    precision_rows = _model_rows(payload)
    device_rows = _provenance_rows(payload)
    scoring = payload.get("scoring") or {}
    accuracy = scoring.get("accuracy") or {}
    performance = scoring.get("performance") or {}

    accuracy_html = ""
    if accuracy:
        accuracy_html = (
            _table("Accuracy device and evaluation", _device_rows(accuracy))
            + _pass_rows(accuracy)
            + _comparison_rows(accuracy)
        )
    else:
        accuracy_html = "<p>Accuracy not measured.</p>"

    environment = payload.get("environment") or {}
    hardware = environment.get("hardware") or {}
    libraries = environment.get("libraries") or {}
    if libraries:
        python = libraries.get("python") or {}
        os_info = hardware.get("os") or {}
        cpu = hardware.get("cpu") or {}
        environment_rows = [
            ("Python", f"{python.get('implementation', '')} {python.get('version', '')}".strip()),
            ("Machine", hardware.get("label") or hardware.get("id")),
            ("Hardware ID", hardware.get("hardware_id")),
            ("Operating system", " ".join(str(os_info.get(key) or "") for key in ("name", "version", "arch")).strip()),
            ("CPU", cpu.get("model")),
        ]
        memory = hardware.get("memory") or {}
        if memory.get("total_mb") is not None:
            environment_rows.append(("Memory (MB)", memory["total_mb"]))
        for index, gpu in enumerate(hardware.get("gpus") or []):
            environment_rows.append(
                (
                    f"GPU {index}",
                    " · ".join(
                        str(gpu.get(key) or "")
                        for key in ("name", "driver", "vram_mb")
                        if gpu.get(key) is not None
                    ),
                )
            )
        packages = libraries.get("packages") or {}
    else:
        packages = environment.get("packages") or {}
        environment_rows = [
            ("Python", environment.get("python")),
            ("Platform", environment.get("platform")),
            ("Machine", environment.get("machine")),
        ]
    environment_rows.extend(
        (str(name), str(version)) for name, version in sorted(packages.items())
    )

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
  details {{ margin: .5rem 0; }}
  .muted {{ opacity: .7; }}
  .warn {{ color: #b45309; }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p class="muted">Generated {_escape(payload.get("generated_at", ""))} · schema {_escape(payload.get("schema_version"))}</p>
<h2>Model</h2>
{_table("Artifact", precision_rows)}
<h2>Provenance</h2>
{_table("Origin and tools", device_rows)}
<h2>Accuracy</h2>
{accuracy_html}
<h2>Performance</h2>
{_performance_tables(performance)}
<h2>Environment</h2>
{_table("Hardware and library versions", environment_rows)}
</body>
</html>
"""


def render_model_overview_html(overview: Mapping[str, Any]) -> str:
    """Render the cross-machine model-report overview as a static table."""
    if overview.get("kind") != MODEL_OVERVIEW_KIND:
        raise ValueError(f"not a {MODEL_OVERVIEW_KIND} payload")
    columns = (
        ("model_id", "Model"),
        ("runtime", "Runtime"),
        ("precision", "Precision"),
        ("batch_size", "Batch"),
        ("accuracy_status", "Accuracy"),
        ("ap", "AP"),
        ("delta_ap", "ΔAP"),
        ("benchmark_count", "Benchmarks"),
        ("best_images_per_second", "Best img/s"),
    )
    header = "".join(f'<th scope="col">{_escape(label)}</th>' for _key, label in columns)
    rows: list[str] = []
    for row in overview.get("rows") or []:
        cells = [row.get(key) for key, _label in columns]
        rendered = [
            format_number(cell, 4) if key in {"ap", "delta_ap"} else
            format_number(cell, 2) if key == "best_images_per_second" else
            "" if cell is None else str(cell)
            for (key, _label), cell in zip(columns, cells, strict=True)
        ]
        rows.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in rendered) + "</tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model report overview</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 2rem auto; max-width: 78rem; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border-bottom: 1px solid rgba(128,128,128,.35); padding: .35rem .55rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ position: sticky; top: 0; background: Canvas; }}
  tbody tr:hover {{ background: color-mix(in srgb, CanvasText 6%, Canvas); }}
  .muted {{ opacity: .7; }}
</style>
</head>
<body>
<h1>Model report overview</h1>
<p class="muted">{overview.get('report_count', 0)} reports · {overview.get('row_count', 0)} rows · generated {_escape(overview.get('generated_at', ''))}</p>
<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>
</body>
</html>
"""
