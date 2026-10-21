"""Shared HTML building blocks for report pages.

Pages are self-contained: no external scripts, stylesheets, or fonts, so they open from disk and
from a zip file. Colours are CSS custom properties with a dark-mode set.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import TIMED_STAGES

STYLE = """
:root {
  --bg: #ffffff; --surface: #f6f7f9; --border: #dde1e6; --text: #1b1f24; --muted: #5b6470;
  --accent: #2f6fde; --accent-2: #c2581c; --ok: #1f8a4c; --warn: #b7791f; --fail: #c53030;
  --ok-bg: #e7f6ed; --warn-bg: #fdf3e1; --fail-bg: #fde8e8; --grid: #e9ecef;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171b; --surface: #1c2026; --border: #323842; --text: #e6e9ed; --muted: #9aa4b1;
    --accent: #6ea0ff; --accent-2: #f08c4a; --ok: #4cc27f; --warn: #e0a640; --fail: #f07171;
    --ok-bg: #16301f; --warn-bg: #352a12; --fail-bg: #3a1a1a; --grid: #2a2f36;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
h3 { font-size: 14px; margin: 18px 0 6px; }
a { color: var(--accent); }
.subtitle { color: var(--muted); margin: 0 0 16px; }
.mono, code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 12.5px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.card .label { color: var(--muted); font-size: 12px; }
.card .value { font-size: 15px; word-break: break-word; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { background: var(--surface); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.kv th { width: 220px; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th[aria-sort="ascending"]::after { content: " \\25B2"; font-size: 10px; }
table.sortable th[aria-sort="descending"]::after { content: " \\25BC"; font-size: 10px; }
.banner { border-radius: 8px; padding: 10px 14px; margin: 12px 0; border: 1px solid var(--border); }
.banner.fail { background: var(--fail-bg); border-color: var(--fail); }
.banner.warn { background: var(--warn-bg); border-color: var(--warn); }
.banner.ok { background: var(--ok-bg); border-color: var(--ok); }
.banner.info { background: var(--surface); }
.pill { display: inline-block; padding: 0 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); }
.pill.ok { color: var(--ok); border-color: var(--ok); }
.pill.warn { color: var(--warn); border-color: var(--warn); }
.pill.fail, .pill.error { color: var(--fail); border-color: var(--fail); }
.pill.unavailable { color: var(--muted); }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--muted); }
pre { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px;
  overflow-x: auto; font-size: 12px; }
ul.links { padding-left: 18px; }
svg text { fill: var(--text); font-size: 12px; }
svg .axis { stroke: var(--border); }
svg .gridline { stroke: var(--grid); }
svg .device { fill: var(--accent); stroke: var(--accent); }
svg .host { fill: var(--accent-2); stroke: var(--accent-2); }
svg .hollow { fill: var(--bg); stroke-width: 2; }
.legend { color: var(--muted); font-size: 12px; margin: 4px 0 8px; }
.legend .swatch { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 12px; vertical-align: middle; }
.muted { color: var(--muted); }
ul.kv-list { list-style: none; margin: 0; padding: 0; }
ul.kv-list ul.kv-list { padding-left: 16px; border-left: 1px solid var(--border); margin: 2px 0 2px 2px; }
ul.kv-list .key { color: var(--muted); }
"""

SORT_SCRIPT = """
document.querySelectorAll('table.sortable').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, column) {
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var ascending = th.getAttribute('aria-sort') !== 'ascending';
      table.querySelectorAll('th').forEach(function (other) { other.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
      rows.sort(function (a, b) {
        var x = a.cells[column].getAttribute('data-sort') || a.cells[column].textContent;
        var y = b.cells[column].getAttribute('data-sort') || b.cells[column].textContent;
        var nx = parseFloat(x), ny = parseFloat(y);
        var result = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
        return ascending ? result : -result;
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
});
"""


def esc(value: Any) -> str:
    """HTML-escape a value. ``None`` becomes ``n/a``."""
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 2) -> str:
    """Format a number with *digits* decimals. Non-numbers are escaped as text."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return esc(value)


NOT_MEASURED = '<span class="muted" title="not measured">—</span>'
"""The cell for a stage that the pass did not measure. It is never shown as 0."""


def fmt_ms(value: Any, digits: int = 2) -> str:
    """Format a time in ms. ``None`` (not measured) becomes "—"."""
    if value is None:
        return NOT_MEASURED
    return fmt(float(value), digits)


def _scalar_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def value_html(value: Any) -> str:
    """Render a JSON value as a readable key/value list. Nested objects are indented.

    The output never holds a Python ``repr``: a list of plain values is joined with commas, and an
    empty object or list is "none".
    """
    if isinstance(value, Mapping):
        if not value:
            return '<span class="muted">none</span>'
        items = "".join(
            f'<li><span class="key">{esc(key)}</span>: {value_html(item)}</li>' for key, item in value.items()
        )
        return f'<ul class="kv-list">{items}</ul>'
    if isinstance(value, (list, tuple)):
        if not value:
            return '<span class="muted">none</span>'
        if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
            return esc(", ".join(_scalar_text(item) for item in value))
        return '<ul class="kv-list">' + "".join(f"<li>{value_html(item)}</li>" for item in value) + "</ul>"
    return esc(_scalar_text(value))


STAGE_HEADERS = ("Preprocess ms/image", "Inference ms/image", "Postprocess ms/image", "Other ms/image")
TIMING_HEADERS = ("Images/s", "ms/image", *STAGE_HEADERS)


@dataclass(frozen=True)
class StageTiming:
    """The end-to-end time of one performance record and its stage breakdown."""

    images_per_second: float | None
    ms_per_image: float | None
    stages: dict[str, float | None]
    other: float | None
    providers: dict[str, str] = field(default_factory=dict)
    """The provider of each stage that differs from the provider of the pass."""
    excluded: dict[str, float] = field(default_factory=dict)

    def cells(self) -> list[tuple[str, Any]]:
        """Return the cells for :data:`TIMING_HEADERS`."""
        values = [self.images_per_second, self.ms_per_image, *(self.stages[stage] for stage in TIMED_STAGES), self.other]
        cells = [(fmt(self.images_per_second), self.images_per_second if self.images_per_second is not None else "")]
        cells += [(fmt_ms(value), value if value is not None else "") for value in values[1:]]
        return cells

    def providers_html(self) -> str:
        return esc("; ".join(f"{stage}: {name}" for stage, name in self.providers.items()))

    def excluded_text(self) -> str:
        return ", ".join(f"{stage} {value:.2f} ms/image" for stage, value in self.excluded.items())

    def breakdown(self) -> str:
        """Return plain text such as ``3.92 ms/image (preprocess 1.81, inference 1.51, …)``."""

        def text(value: float | None) -> str:
            return "—" if value is None else f"{value:.2f}"

        parts = [f"{stage} {text(self.stages[stage])}" for stage in TIMED_STAGES] + [f"other {text(self.other)}"]
        result = f"{text(self.ms_per_image)} ms/image ({', '.join(parts)} ms)"
        if self.providers:
            result += " · " + "; ".join(f"{stage} on {name}" for stage, name in self.providers.items())
        if self.excluded:
            result += f" · excluded: {self.excluded_text()}"
        return result


def provider_names(execution: Mapping[str, Any] | None) -> set[str]:
    """Return the names of the pass's provider: EP name, key, slug, and labels."""
    execution = execution or {}
    provider = execution.get("provider") or {}
    names = {provider.get(key) for key in ("ep_name", "key", "slug", "label")} | {execution.get("label")}
    return {str(name) for name in names if name}


def stage_timing(timing: Mapping[str, Any] | None, execution: Mapping[str, Any] | None = None) -> StageTiming:
    """Read the stage breakdown of one ``timing`` block.

    ``ms_per_image`` falls back to ``1000 / images_per_second``. "Other" is ``ms_per_image`` minus the
    sum of the measured stages. It is ``None`` when any stage is not measured.
    """
    timing = timing or {}
    ips = timing.get("images_per_second")
    ips = float(ips) if isinstance(ips, (int, float)) and not isinstance(ips, bool) else None
    ms = timing.get("ms_per_image")
    if ms is None and ips:
        ms = 1000.0 / ips
    ms = float(ms) if ms is not None else None
    raw = timing.get("stages_ms_per_image") or {}
    stages = {stage: (float(raw[stage]) if raw.get(stage) is not None else None) for stage in TIMED_STAGES}
    other = None
    if ms is not None and all(value is not None for value in stages.values()):
        other = ms - sum(value for value in stages.values() if value is not None)
    own = provider_names(execution)
    providers = {stage: str(name) for stage, name in (timing.get("stage_providers") or {}).items()
                 if name and str(name) not in own}
    excluded = {str(stage): float(value) for stage, value in (timing.get("excluded_ms_per_image") or {}).items()}
    return StageTiming(ips, ms, stages, other, providers, excluded)


def excluded_note(records: Sequence[Mapping[str, Any]], execution: Mapping[str, Any] | None = None) -> str:
    """Return a note with the excluded stage times of each record, or an empty string."""
    parts = []
    for record in records:
        text = stage_timing(record.get("timing"), execution).excluded_text()
        if text:
            parts.append(f"{text} ({record.get('mode')})")
    if not parts:
        return ""
    return f'<p class="muted">Not in ms/image (excluded and timed separately): {esc("; ".join(parts))}.</p>'


def dataset_line(dataset: Mapping[str, Any] | None) -> str:
    """Return one line of HTML that names a dataset: name, split, annotation file, categories, images."""
    if not dataset:
        return '<span class="muted">not recorded</span>'
    parts = [esc(dataset.get("name"))]
    if dataset.get("split"):
        parts.append(f"split {esc(dataset['split'])}")
    if dataset.get("annotation_file"):
        parts.append(f"<code>{esc(dataset['annotation_file'])}</code>")
    else:
        parts.append('<span class="muted">no annotations read</span>')
    categories = dataset.get("categories")
    if categories:
        parts.append("categories " + esc(", ".join(f"{item.get('id')} {item.get('name')}" for item in categories)))
    images = f"{fmt(dataset.get('images_used'))} of {fmt(dataset.get('images_available'))} images"
    if dataset.get("selection"):
        images += f" ({esc(dataset['selection'])})"
    parts.append(images)
    return " · ".join(parts)


def stage_definitions_table(protocol: Mapping[str, Any] | None) -> str:
    """Return the stage definitions of a protocol as a table."""
    definitions = (protocol or {}).get("stage_definitions") or {}
    if not definitions:
        return '<p class="muted">The protocol defines no stages.</p>'
    rows = [
        [esc(mode), esc(stage), esc(text)]
        for mode, stages in definitions.items()
        for stage, text in stages.items()
    ]
    return data_table(["Mode", "Stage", "Definition"], rows)


def fmt_bytes(size: Any) -> str:
    if not isinstance(size, (int, float)):
        return "n/a"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def link(href: str | None, text: Any) -> str:
    return f'<a href="{esc(href)}">{esc(text)}</a>' if href else esc(text)


def pill(status: Any) -> str:
    text = esc(status)
    return f'<span class="pill {text}">{text}</span>'


def kv_table(rows: Iterable[tuple[str, str]]) -> str:
    """Render key/value rows. Values are already HTML."""
    body = "".join(f"<tr><th>{esc(key)}</th><td>{value}</td></tr>" for key, value in rows)
    return f'<div class="table-wrap"><table class="kv"><tbody>{body}</tbody></table></div>'


def data_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str | tuple[str, Any]]],
    *,
    sortable: bool = False,
    numeric: Sequence[int] = (),
) -> str:
    """Render a table. A cell is HTML, or ``(html, sort_value)``."""
    head = "".join(
        f'<th class="num">{esc(header)}</th>' if index in numeric else f"<th>{esc(header)}</th>"
        for index, header in enumerate(headers)
    )
    body_rows = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            content, sort_value = cell if isinstance(cell, tuple) else (cell, None)
            attributes = ' class="num"' if index in numeric else ""
            if sort_value is not None:
                attributes += f' data-sort="{esc(sort_value)}"'
            cells.append(f"<td{attributes}>{content}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    css = ' class="sortable"' if sortable else ""
    return (
        f'<div class="table-wrap"><table{css}><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def cards(items: Iterable[tuple[str, str]]) -> str:
    """Render label/value cards. Values are already HTML."""
    body = "".join(
        f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="grid">{body}</div>'


def json_details(summary: str, value: Any) -> str:
    return f"<details><summary>{esc(summary)}</summary><pre>{esc(json.dumps(value, indent=2))}</pre></details>"


def data_island(element_id: str, value: Any) -> str:
    """Embed JSON in the page. ``</`` is escaped so the data cannot close the script tag."""
    text = json.dumps(value, sort_keys=True).replace("</", "<\\/")
    return f'<script type="application/json" id="{esc(element_id)}">{text}</script>'


def page(title: str, body: str, *, scripts: Sequence[str] = (), head: str = "") -> str:
    """Return a complete HTML document."""
    script_tags = "".join(f"<script>{script}</script>" for script in scripts)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{STYLE}</style>{head}</head>"
        f"<body><main>{body}</main>{script_tags}</body></html>\n"
    )


def comparison_banner(comparison: Mapping[str, Any] | None, *, prefix: str = "") -> str:
    if not comparison:
        return ""
    status = comparison.get("status")
    delta = (comparison.get("deltas") or {}).get("delta_ap")
    text = f"{prefix}Accuracy comparison against the parent: {esc(status)} (Δ AP {fmt(delta, 4)})."
    if status == "fail":
        text += " The sidecar leaves out this model and every model made from it."
    if status in ("warn", "fail"):
        return f'<div class="banner {esc(status)}">{text}</div>'
    return ""


__all__ = [
    "NOT_MEASURED",
    "SORT_SCRIPT",
    "STAGE_HEADERS",
    "STYLE",
    "TIMING_HEADERS",
    "StageTiming",
    "cards",
    "comparison_banner",
    "data_island",
    "data_table",
    "dataset_line",
    "esc",
    "excluded_note",
    "fmt",
    "fmt_bytes",
    "fmt_ms",
    "json_details",
    "kv_table",
    "link",
    "page",
    "pill",
    "provider_names",
    "stage_definitions_table",
    "stage_timing",
    "value_html",
]
