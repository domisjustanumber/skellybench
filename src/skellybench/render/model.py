"""Model report HTML: overview, accuracy table, performance chart, and benchmark links."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .benchmark import host_rows, reproduction_rows
from .common import (
    SORT_SCRIPT,
    cards,
    comparison_banner,
    data_table,
    esc,
    fmt,
    fmt_bytes,
    json_details,
    kv_table,
    link,
    page,
)

MODE_SHAPES = {"device": "circle", "host": "square"}


@dataclass(frozen=True)
class LinkedBenchmark:
    """A benchmark payload plus the link to its HTML page from the page being rendered."""

    payload: Mapping[str, Any]
    href: str


@dataclass(frozen=True)
class ChartPoint:
    row: str
    value: float
    mode: str
    run_id: str
    hostname: str
    href: str
    newest: bool = False


def _row_labels(benchmarks: Sequence[LinkedBenchmark]) -> dict[str, str]:
    """Map report id -> chart row label. A label shared by two hosts gets the hostname added."""
    hosts_by_label: dict[str, set[str]] = {}
    for item in benchmarks:
        label = item.payload["execution"]["label"]
        hosts_by_label.setdefault(label, set()).add(item.payload["host"]["hostname"])
    labels = {}
    for item in benchmarks:
        label = item.payload["execution"]["label"]
        host = item.payload["host"]["hostname"]
        labels[item.payload["report_id"]] = f"{label} · {host}" if len(hosts_by_label[label]) > 1 else label
    return labels


def chart_points(benchmarks: Sequence[LinkedBenchmark]) -> list[ChartPoint]:
    """Return one point per measured performance record. Accuracy timings are never charted."""
    performance = [
        item
        for item in benchmarks
        if item.payload["benchmark_type"] == "performance" and item.payload["status"] == "measured"
    ]
    labels = _row_labels(performance)
    points: list[ChartPoint] = []
    for item in performance:
        for record in (item.payload.get("performance") or {}).get("records") or []:
            value = (record.get("timing") or {}).get("images_per_second")
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            points.append(
                ChartPoint(
                    row=labels[item.payload["report_id"]],
                    value=float(value),
                    mode=str(record.get("mode")),
                    run_id=str(item.payload["run"]["run_id"]),
                    hostname=str(item.payload["host"]["hostname"]),
                    href=item.href,
                )
            )
    newest = {}
    for point in points:
        newest[point.row] = max(newest.get(point.row, ""), point.run_id)
    return [
        ChartPoint(**{**point.__dict__, "newest": point.run_id == newest[point.row]})
        for point in points
    ]


def _ticks(low: float, high: float, log: bool) -> list[float]:
    if log:
        ticks = []
        exponent = math.floor(math.log10(low))
        while 10**exponent <= high * 1.0001:
            for factor in (1, 2, 5):
                value = factor * 10**exponent
                if low <= value <= high:
                    ticks.append(value)
            exponent += 1
        return ticks or [low, high]
    span = high - low or 1.0
    step = 10 ** math.floor(math.log10(span / 5))
    for factor in (1, 2, 5, 10):
        if span / (step * factor) <= 6:
            step *= factor
            break
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-6:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _tick_label(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:g}"


def render_chart(points: Sequence[ChartPoint]) -> str:
    """Render a horizontal dot plot of images/s per execution label as inline SVG."""
    if not points:
        return '<p class="muted">No performance benchmarks yet.</p>'
    rows = sorted({point.row for point in points}, key=lambda row: -max(p.value for p in points if p.row == row))
    values = [point.value for point in points]
    high = max(values) * 1.08
    low_value = min(values)
    log = low_value > 0 and max(values) / low_value > 20
    low = 10 ** math.floor(math.log10(low_value)) if log else 0.0
    label_width = min(300, max(120, 7 * max(len(row) for row in rows) + 16))
    width = 900
    row_height = 30
    top = 10
    plot_width = width - label_width - 24
    height = top + row_height * len(rows) + 36

    def x(value: float) -> float:
        if log:
            return label_width + plot_width * (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        return label_width + plot_width * (value - low) / (high - low)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Images per second by execution provider and device">'
    ]
    axis_y = top + row_height * len(rows)
    for tick in _ticks(max(low, 1e-9) if log else low, high, log):
        tick_x = x(tick)
        parts.append(f'<line class="gridline" x1="{tick_x:.1f}" y1="{top}" x2="{tick_x:.1f}" y2="{axis_y}"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{axis_y + 16}" text-anchor="middle">{_tick_label(tick)}</text>')
    parts.append(f'<line class="axis" x1="{label_width}" y1="{axis_y}" x2="{width - 24}" y2="{axis_y}"/>')
    parts.append(
        f'<text x="{label_width + plot_width / 2:.1f}" y="{axis_y + 32}" text-anchor="middle">'
        f"images per second{' (log scale)' if log else ''}</text>"
    )
    for index, row in enumerate(rows):
        center = top + row_height * index + row_height / 2
        parts.append(f'<text x="{label_width - 10}" y="{center + 4:.1f}" text-anchor="end">{esc(row)}</text>')
        parts.append(f'<line class="gridline" x1="{label_width}" y1="{center:.1f}" x2="{width - 24}" y2="{center:.1f}"/>')
        for point in sorted((p for p in points if p.row == row), key=lambda p: p.newest):
            css = f"{esc(point.mode)}{'' if point.newest else ' hollow'}"
            title = f"{point.run_id} · {point.hostname} · {point.mode} · {point.value:.2f} images/s"
            px = x(point.value)
            if MODE_SHAPES.get(point.mode) == "square":
                shape = f'<rect class="{css}" x="{px - 5:.1f}" y="{center - 5:.1f}" width="10" height="10"/>'
            else:
                shape = f'<circle class="{css}" cx="{px:.1f}" cy="{center:.1f}" r="6"/>'
            parts.append(f'<a href="{esc(point.href)}"><title>{esc(title)}</title>{shape}</a>')
    parts.append("</svg>")
    legend = (
        '<div class="legend"><span class="swatch" style="background:var(--accent);border-radius:50%"></span>device I/O'
        '<span class="swatch" style="background:var(--accent-2)"></span>host I/O'
        " · filled = newest run in the row · hollow = older runs · click a mark to open its report</div>"
    )
    return legend + "".join(parts)


def _metric(snapshot: Mapping[str, Any], key: str) -> float | None:
    return (((snapshot.get("accuracy") or {}).get("metrics") or {}).get("values") or {}).get(key)


def _method_table(accuracy: Mapping[str, Any]) -> str:
    """Return the self and parent accuracy methods side by side."""
    columns = [("Self", accuracy["self"]["snapshot"])]
    if accuracy.get("parent"):
        columns.append(("Parent", accuracy["parent"]["snapshot"]))

    def cells(render: Any) -> list[str]:
        return [render((snapshot.get("accuracy") or {}).get("method") or {}) for _, snapshot in columns]

    rows = [
        ["Decode", *cells(lambda method: f"<code>{esc(method.get('decode'))}</code>")],
        ["Scorer", *cells(lambda method: f"{esc((method.get('scorer') or {}).get('tool'))} "
                                         f"{esc((method.get('scorer') or {}).get('version'))}")],
        ["Scored by", *cells(lambda method: esc(method.get("scored_by")))],
    ]
    rows.append(["Run", *[esc(snapshot["run"]["run_id"]) for _, snapshot in columns]])
    return data_table(["Method", *[name for name, _ in columns]], rows)


def _accuracy_rows(
    report: Mapping[str, Any],
    benchmarks: Sequence[LinkedBenchmark],
    parent_ap_by_run: Mapping[str, float],
    parent_link: LinkedBenchmark | None,
) -> list[list[Any]]:
    parent_run_id = report["accuracy"].get("parent_run_id")
    accuracy = report["accuracy"]
    self_id = accuracy["self"]["snapshot"]["report_id"]
    rows = []
    items = [item for item in benchmarks if item.payload["benchmark_type"] == "accuracy" and item.payload["status"] == "measured"]
    if parent_link is not None:
        items.append(parent_link)
    for item in sorted(items, key=lambda entry: entry.payload["run"]["run_id"], reverse=True):
        snapshot = item.payload
        run_id = snapshot["run"]["run_id"]
        is_parent = parent_link is not None and item is parent_link
        role = (
            f"parent (at creation, reused from run {parent_run_id})" if is_parent and parent_run_id
            else "parent (at creation)" if is_parent
            else "self (at creation)" if snapshot["report_id"] == self_id else ""
        )
        ap = _metric(snapshot, "ap")
        parent_ap = parent_ap_by_run.get(run_id)
        delta = None if is_parent or ap is None or parent_ap is None else ap - parent_ap
        rows.append([
            link(item.href, run_id),
            esc(snapshot["host"]["hostname"]),
            esc(snapshot["execution"]["label"]),
            esc(snapshot["subject"]["subject_id"]) + (f" <span class='muted'>({esc(role)})</span>" if role else ""),
            *[(fmt(_metric(snapshot, key), 4), _metric(snapshot, key)) for key in ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar100")],
            (fmt(delta, 4), delta if delta is not None else ""),
        ])
    return rows


def render_model_html(
    report: Mapping[str, Any],
    *,
    benchmarks: Sequence[LinkedBenchmark],
    parent_href: str | None = None,
    parent_benchmark: LinkedBenchmark | None = None,
    parent_ap_by_run: Mapping[str, float] | None = None,
    ancestor_comparisons: Sequence[tuple[str, Mapping[str, Any]]] = (),
    current_href: str | None = None,
    superseded: bool = False,
) -> str:
    """Render one model report page.

    Args:
        benchmarks: every benchmark of this model's artifact, with links from this page.
        parent_href: link to the parent model page.
        parent_benchmark: the creation-time parent accuracy benchmark, with a link.
        parent_ap_by_run: parent AP per run id, for the delta column.
        ancestor_comparisons: ``(model id, comparison)`` of each ancestor, nearest first. A failed
            ancestor shows a banner, and the deltas add up to a cumulative delta.
        current_href: for a superseded page, the link to the current model page.
    """
    model = report["model"]
    creation = report["creation"]
    accuracy = report["accuracy"]
    artifact = model.get("artifact") or {}
    variant = model.get("variant") or {}
    title = f"{model.get('display_name') or model.get('model_id')} — {report['report_id']}"
    parts = [f"<h1>{esc(title)}</h1>"]
    parts.append(
        f'<p class="subtitle">Created in run <code>{esc(report["created_run_id"])}</code> on '
        f"{esc(creation['host'].get('hostname'))}</p>"
    )
    if superseded:
        parts.append(f'<div class="banner info">This model report is superseded. {link(current_href, "Open the current report")}.</div>')
    parts.append(comparison_banner(accuracy.get("comparison")))
    for ancestor_id, comparison in ancestor_comparisons:
        if comparison.get("status") == "fail":
            parts.append(
                f'<div class="banner fail">Ancestor {esc(ancestor_id)} failed its accuracy comparison, '
                "so the sidecar leaves out this model too.</div>"
            )
    source = creation.get("source") or {}
    parent = creation.get("parent") or {}
    model_input = model.get("input") or {}
    parts.append(cards([
        ("Model", esc(model.get("model_id"))),
        ("Format · precision", f"{esc(model.get('format'))} · {esc(model.get('precision'))}"),
        ("Batch size", esc(model.get("batch_size") or "any")),
        ("Input", f"{esc(model_input.get('width'))} × {esc(model_input.get('height'))}"),
        ("File", f"{esc(artifact.get('filename'))}<br><span class='muted'>{fmt_bytes(artifact.get('size_bytes'))}</span>"),
        ("SHA-256", f"<code>{esc(artifact.get('sha256'))}</code>"),
        ("Parent", link(parent_href, parent.get("report_id")) if parent else "none (source model)"),
        ("Source", link(source.get("url"), source.get("filename"))),
        ("Family · size", f"{esc(variant.get('family'))} · {esc(variant.get('size'))}"),
    ]))

    parts.append("<h2>Accuracy</h2>")
    if accuracy.get("parent_run_id"):
        parts.append(f'<div class="banner info">Parent pass reused from run <code>{esc(accuracy["parent_run_id"])}</code>.</div>')
    comparison = accuracy.get("comparison")
    if comparison:
        deltas = comparison.get("deltas") or {}
        cumulative = deltas.get("delta_ap")
        for _, ancestor in ancestor_comparisons:
            value = (ancestor.get("deltas") or {}).get("delta_ap")
            cumulative = None if cumulative is None or value is None else cumulative + value
        summary = [
            ("Δ AP against the parent (at creation)", f"{fmt(deltas.get('delta_ap'), 4)} — {esc(comparison.get('status'))}"),
            ("Thresholds", f"warn {fmt((comparison.get('thresholds') or {}).get('warn_ap'), 4)}, "
             f"fail {fmt((comparison.get('thresholds') or {}).get('fail_ap'), 4)}"),
        ]
        if ancestor_comparisons:
            summary.append(("Cumulative Δ AP against the source model", fmt(cumulative, 4)))
        parts.append(kv_table(summary))
    parts.append("<h3>Methods</h3>")
    parts.append(_method_table(accuracy))
    parts.append('<p class="muted">The two passes can decode and score in different ways. '
                 "The shared evaluation settings are equal.</p>")
    parts.append("<h3>Results</h3>")
    rows = _accuracy_rows(report, benchmarks, parent_ap_by_run or {}, parent_benchmark)
    parts.append(data_table(
        ["Run", "Host", "Device", "Subject", "AP", "AP50", "AP75", "AP small", "AP medium", "AP large", "AR100", "Δ AP vs parent"],
        rows,
        sortable=True,
        numeric=tuple(range(4, 12)),
    ))
    parts.append('<p class="muted">Click a column header to sort. Δ AP compares against the parent model '
                 "in the same run.</p>")

    parts.append("<h2>Performance</h2>")
    points = chart_points(benchmarks)
    parts.append(render_chart(points))
    if points:
        parts.append(data_table(
            ["Device", "Mode", "Images/s", "Run", "Host"],
            [
                [esc(point.row), esc(point.mode), (fmt(point.value), point.value), link(point.href, point.run_id), esc(point.hostname)]
                for point in sorted(points, key=lambda p: (p.row, p.mode, p.run_id))
            ],
            sortable=True,
            numeric=(2,),
        ))

    if report.get("accuracy_notes") or report.get("performance_notes"):
        parts.append("<h2>Notes</h2>")
        for key, heading in (("accuracy_notes", "Accuracy"), ("performance_notes", "Performance")):
            if report.get(key):
                parts.append(f"<h3>{heading}</h3><p>{esc(report[key])}</p>")

    parts.append("<h2>Benchmark reports</h2>")
    for benchmark_type, heading in (("accuracy", "Accuracy"), ("performance", "Performance")):
        items = sorted(
            (item for item in benchmarks if item.payload["benchmark_type"] == benchmark_type),
            key=lambda item: item.payload["run"]["run_id"],
            reverse=True,
        )
        parts.append(f"<h3>{heading}</h3>")
        if not items:
            parts.append('<p class="muted">None.</p>')
            continue
        parts.append('<ul class="links">' + "".join(
            f"<li>{link(item.href, item.payload['report_id'])} — {esc(item.payload['execution']['label'])}, "
            f"{esc(item.payload['status'])}</li>"
            for item in items
        ) + "</ul>")

    parts.append("<h2>Creation</h2>")
    step_rows = [
        [
            esc(step.get("step")),
            f"{esc(step.get('tool'))} {esc(step.get('version'))}",
            f"<code>{esc(step.get('params'))}</code>",
            f"<code>{esc(step.get('output_sha256'))}</code>",
        ]
        for step in creation.get("steps") or []
    ]
    parts.append(data_table(["Step", "Tool", "Parameters", "Output SHA-256"], step_rows))
    parts.append(kv_table([
        ("Source URL", link(source.get("url"), source.get("url"))),
        ("Source MD5", f"<code>{esc(source.get('md5'))}</code>"),
        ("Source SHA-256", f"<code>{esc(source.get('sha256'))}</code>"),
        ("Config sidecar", link((creation.get("config_sidecar") or {}).get("url"), (creation.get("config_sidecar") or {}).get("url"))
         if creation.get("config_sidecar") else "none"),
    ]))
    parts.append("<h3>Reproduce the export</h3>")
    parts.append(kv_table(reproduction_rows(creation["run"], creation["environment"], None, extra_packages=("rfdetr", "onnxconverter-common"))))
    parts.append("<h3>Export host</h3>")
    parts.append(kv_table(host_rows(creation["host"])))
    parts.append(json_details("Full creation environment", creation["environment"]))
    return page(title, "".join(parts), scripts=(SORT_SCRIPT,))


__all__ = ["ChartPoint", "LinkedBenchmark", "chart_points", "render_chart", "render_model_html"]
