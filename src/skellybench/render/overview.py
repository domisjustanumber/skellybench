"""Family overview HTML: which variant and execution provider to use on given hardware.

One page per model family, ``<family>.overview.html``. It answers two questions:

- "I have this hardware and this batch size. What is the fastest way to run this model?"
- "Which variant gives the best balance of accuracy and speed?"

The page is HTML only. Rows come from the current model reports and their performance benchmarks.
The rows go into the page as a JSON data island. A small inline script draws the cards and charts.
Without JavaScript, the page still shows the full table.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._internal.io import atomic_write_text
from ..paths import href, overview_path, resolve_family_path
from ..reports import FamilyIndex, scan_family
from .common import SORT_SCRIPT, data_island, data_table, esc, fmt, link, page

ROWS_ELEMENT_ID = "overview-rows"


def _metric(payload: Mapping[str, Any], key: str) -> float | None:
    return (((payload.get("accuracy") or {}).get("metrics") or {}).get("values") or {}).get(key)


def overview_rows(index: FamilyIndex, page_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(rows, unavailable)`` for the overview.

    One row per subject x machine (hardware fingerprint) x execution label x mode. When several runs
    exist for a key, the row uses the newest run and counts the others. Only artifacts with a
    current model report appear, because the variant and the accuracy come from that report.
    """
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    unavailable: list[dict[str, Any]] = []
    for subject, model in index.models.items():
        report = model.payload
        if report["model"].get("format") == "pytorch_checkpoint":
            continue
        variant = report["model"]["variant"]
        accuracy = report["accuracy"]
        self_snapshot = accuracy["self"]["snapshot"]
        comparison = accuracy.get("comparison") or {}
        accuracy_page = resolve_family_path(index.family_dir, accuracy["self"]["link"]["path"]).with_suffix(".html")
        for entry in index.benchmarks_for(subject, report["model"]["artifact"]["sha256"]):
            payload = entry.payload
            if payload["benchmark_type"] != "performance":
                continue
            host = payload["host"]
            execution = payload["execution"]
            if payload["status"] != "measured":
                unavailable.append({
                    "subject": subject,
                    "size": variant["size"],
                    "precision": variant["precision"],
                    "batch_size": variant["batch_size"],
                    "label": execution["label"],
                    "host": host["hostname"],
                    "fingerprint": host["hardware_fingerprint"],
                    "reason": payload.get("reason"),
                    "href": href(page_path, entry.html_path),
                })
                continue
            for record in (payload.get("performance") or {}).get("records") or []:
                timing = record.get("timing") or {}
                key = (subject, host["hardware_fingerprint"], execution["label"], record.get("mode"))
                run_id = payload["run"]["run_id"]
                row = {
                    "subject": subject,
                    "size": variant["size"],
                    "size_rank": variant["size_rank"],
                    "precision": variant["precision"],
                    "batch_size": variant["batch_size"],
                    "label": execution["label"],
                    "host": host["hostname"],
                    "fingerprint": host["hardware_fingerprint"],
                    "mode": record.get("mode"),
                    "ips": timing.get("images_per_second"),
                    "p50": timing.get("batch_ms_p50"),
                    "p95": timing.get("batch_ms_p95"),
                    "run_id": run_id,
                    "runs": 1,
                    "ap": _metric(self_snapshot, "ap"),
                    "ap50": _metric(self_snapshot, "ap50"),
                    "ap75": _metric(self_snapshot, "ap75"),
                    "delta_ap": (comparison.get("deltas") or {}).get("delta_ap"),
                    "comparison": comparison.get("status"),
                    "accuracy_notes": report.get("accuracy_notes"),
                    "performance_notes": report.get("performance_notes"),
                    "accuracy_label": self_snapshot["execution"]["label"],
                    "performance_href": href(page_path, entry.html_path),
                    "accuracy_href": href(page_path, accuracy_page),
                    "model_href": href(page_path, model.html_path),
                }
                previous = rows.get(key)
                if previous is None:
                    rows[key] = row
                else:
                    runs = previous["runs"] + 1
                    rows[key] = {**(row if run_id > previous["run_id"] else previous), "runs": runs}
    ordered = sorted(
        rows.values(),
        key=lambda row: (row["host"], row["label"], row["batch_size"] or 0, row["size_rank"], row["precision"], row["mode"]),
    )
    return ordered, unavailable


def _speed(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get("p50") if metric == "latency" else row.get("ips")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _faster(left: float, right: float, metric: str) -> bool:
    return left < right if metric == "latency" else left > right


def pareto_frontier(rows: list[Mapping[str, Any]], metric: str = "throughput") -> list[Mapping[str, Any]]:
    """Return the rows no other row beats on both speed and AP, fastest first.

    A row is on the frontier when every faster row has a lower AP. Ties in speed keep the higher AP.
    """
    usable = [row for row in rows if _speed(row, metric) is not None and row.get("ap") is not None]
    ordered = sorted(
        usable,
        key=lambda row: ((_speed(row, metric) if metric == "latency" else -_speed(row, metric)), -row["ap"]),
    )
    frontier: list[Mapping[str, Any]] = []
    best_ap = float("-inf")
    for row in ordered:
        if row["ap"] > best_ap:
            frontier.append(row)
            best_ap = row["ap"]
    return frontier


def knee(frontier: list[Mapping[str, Any]], metric: str = "throughput") -> Mapping[str, Any] | None:
    """Return the frontier row with the largest AP gain per halving of speed against its faster neighbour.

    With one frontier row, return it. With none, return ``None``.
    """
    if not frontier:
        return None
    import math

    best, best_gain = frontier[0], float("-inf")
    for faster_row, slower_row in zip(frontier, frontier[1:], strict=False):
        fast, slow = _speed(faster_row, metric), _speed(slower_row, metric)
        halvings = abs(math.log2(fast / slow)) if fast and slow else 0.0
        gain = (slower_row["ap"] - faster_row["ap"]) / halvings if halvings > 0 else float("inf")
        if gain > best_gain:
            best, best_gain = slower_row, gain
    return best


def best_for_target(rows: list[Mapping[str, Any]], target: float, metric: str = "throughput") -> Mapping[str, Any] | None:
    """Return the highest-AP row that meets *target* (images/s, or max p50 ms for latency), or ``None``."""
    meeting = [
        row
        for row in rows
        if row.get("ap") is not None
        and (speed := _speed(row, metric)) is not None
        and (speed <= target if metric == "latency" else speed >= target)
    ]
    return max(meeting, key=lambda row: row["ap"], default=None)


def group_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row[key]) for key in ("fingerprint", "label", "batch_size", "mode"))


def frontier_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return ``{group key: {metric: {"frontier": [row id], "knee": row id}}}``.

    Each row must carry ``id``, its position in *rows*. The page script reads these groups, so the
    frontier and knee rules live in Python, where tests cover them.
    """
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(group_key(row), []).append(index)
    result: dict[str, dict[str, Any]] = {}
    for key, indexes in groups.items():
        members = [rows[index] for index in indexes]
        result[key] = {}
        for metric in ("throughput", "latency"):
            front = pareto_frontier(members, metric)
            best = knee(front, metric)
            result[key][metric] = {
                "frontier": [row["id"] for row in front],
                "knee": best["id"] if best is not None else None,
            }
    return result


OVERVIEW_STYLE = """
<style>
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 12px 0 4px; }
.controls label { display: flex; flex-direction: column; font-size: 12px; color: var(--muted); gap: 2px; }
.controls select, .controls input { font: inherit; padding: 4px 6px; background: var(--bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px; min-width: 120px; }
.rec .value { font-size: 16px; font-weight: 600; }
.rec .detail { color: var(--muted); font-size: 12px; }
:root { --c1: #2f6fde; --c2: #1f8a4c; --c3: #c2581c; --c4: #8e44ad; }
@media (prefers-color-scheme: dark) { :root { --c1: #6ea0ff; --c2: #4cc27f; --c3: #f08c4a; --c4: #c18ae6; } }
svg .size-0 { fill: var(--c1); stroke: var(--c1); } svg .size-1 { fill: var(--c2); stroke: var(--c2); }
svg .size-2 { fill: var(--c3); stroke: var(--c3); } svg .size-3 { fill: var(--c4); stroke: var(--c4); }
svg .faded { opacity: 0.25; }
svg .frontier { fill: none; stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 3; }
svg .target { stroke: var(--fail); stroke-width: 1.5; }
svg .line { fill: none; stroke-width: 2; }
.note-marker { cursor: help; color: var(--accent); font-size: 12px; }
table.heat td.cell { text-align: right; font-variant-numeric: tabular-nums; }
table.heat td.best { outline: 2px solid var(--accent); outline-offset: -2px; font-weight: 600; }
table.heat td.empty { color: var(--muted); }
</style>
"""

OVERVIEW_SCRIPT = r"""
(function () {
  var rows = JSON.parse(document.getElementById('overview-rows').textContent);
  var data = rows.rows, unavailable = rows.unavailable, groups = rows.groups;
  if (!data.length) { return; }
  var $ = function (id) { return document.getElementById(id); };
  var svgNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent, text) {
    var node = document.createElementNS(svgNS, tag);
    for (var key in attrs) { node.setAttribute(key, attrs[key]); }
    if (text !== undefined) { node.textContent = text; }
    if (parent) { parent.appendChild(node); }
    return node;
  }
  function uniq(list) { return list.filter(function (v, i) { return list.indexOf(v) === i; }); }
  function fmt(v, d) { return (v === null || v === undefined) ? 'n/a' : Number(v).toFixed(d); }
  function option(select, value, text) { var o = document.createElement('option'); o.value = value; o.textContent = text; select.appendChild(o); }

  var machines = uniq(data.map(function (r) { return r.fingerprint; }));
  var hostOf = {}; data.forEach(function (r) { hostOf[r.fingerprint] = r.host; });
  var counts = {}; data.forEach(function (r) { var k = r.fingerprint + '|' + r.label; counts[k] = (counts[k] || 0) + 1; });
  var best = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; })[0].split('|');
  machines.forEach(function (m) { option($('machine'), m, hostOf[m] + ' (' + m.slice(7, 15) + ')'); });
  $('machine').value = best[0];
  function fillLabels() {
    var select = $('label'); var previous = select.value; select.innerHTML = '';
    uniq(data.filter(function (r) { return r.fingerprint === $('machine').value; }).map(function (r) { return r.label; }))
      .forEach(function (l) { option(select, l, l); });
    if ([].some.call(select.options, function (o) { return o.value === previous; })) { select.value = previous; }
    else if ($('machine').value === best[0]) { select.value = best[1]; }
  }
  uniq(data.map(function (r) { return r.batch_size; })).sort(function (a, b) { return a - b; })
    .forEach(function (b) { option($('batch'), b, b); });
  if ([].some.call($('batch').options, function (o) { return o.value === '1'; })) { $('batch').value = '1'; }
  uniq(data.map(function (r) { return r.mode; })).forEach(function (m) { option($('mode'), m, m); });
  if ([].some.call($('mode').options, function (o) { return o.value === 'device'; })) { $('mode').value = 'device'; }
  fillLabels();

  function latency() { return $('metric').value === 'latency'; }
  function speed(r) { return latency() ? r.p50 : r.ips; }
  function faster(a, b) { return latency() ? a < b : a > b; }
  function selected() {
    return data.filter(function (r) {
      return r.fingerprint === $('machine').value && r.label === $('label').value &&
        String(r.batch_size) === $('batch').value && r.mode === $('mode').value && speed(r) && r.ap !== null;
    });
  }
  function group() {
    var key = [$('machine').value, $('label').value, $('batch').value, $('mode').value].join('|');
    return (groups[key] || {})[$('metric').value] || { frontier: [], knee: null };
  }
  function frontier() { return group().frontier.map(function (id) { return data[id]; }); }
  function describe(r) { return r.size + ' ' + r.precision + ' (b' + r.batch_size + ')'; }
  function card(title, r, rule) {
    if (!r) { return '<div class="card rec"><div class="label">' + title + '</div><div class="value">none</div><div class="detail">' + rule + '</div></div>'; }
    return '<div class="card rec"><div class="label">' + title + '</div><div class="value"><a href="' + r.model_href + '">' + describe(r) +
      '</a></div><div class="detail"><a href="' + r.performance_href + '">' + fmt(r.ips, 1) + ' images/s</a> · p50 ' + fmt(r.p50, 1) +
      ' ms · AP ' + fmt(r.ap, 3) + '</div><div class="detail">' + rule + '</div></div>';
  }
  function recommendations(list) {
    if (!list.length) { $('cards').innerHTML = '<p class="muted">No measured rows for this filter.</p>'; return; }
    var fastest = list.slice().sort(function (a, b) { return faster(speed(a), speed(b)) ? -1 : 1; })[0];
    var accurate = list.slice().sort(function (a, b) { return b.ap - a.ap; })[0];
    var target = parseFloat($('target').value), balance, rule;
    if (!isNaN(target) && target > 0) {
      var meets = list.filter(function (r) { return latency() ? r.p50 <= target : r.ips >= target; });
      balance = meets.sort(function (a, b) { return b.ap - a.ap; })[0];
      rule = latency() ? 'Highest AP with p50 latency at or under ' + target + ' ms.' : 'Highest AP at or above ' + target + ' images/s.';
    } else {
      balance = group().knee === null ? null : data[group().knee];
      rule = 'Pareto-frontier row with the largest AP gain per halving of speed (the knee).';
    }
    $('cards').innerHTML = card('Fastest', fastest, latency() ? 'Lowest p50 latency.' : 'Highest images/s.') +
      card('Most accurate', accurate, 'Highest AP (accuracy device: ' + accurate.accuracy_label + ').') + card('Best balance', balance, rule);
  }
  function scatter(list) {
    var svg = $('scatter'); while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    if (!list.length) { return; }
    var W = 900, H = 380, L = 64, R = 20, T = 16, B = 44;
    var xs = list.map(speed), ys = list.map(function (r) { return r.ap; });
    var x0 = Math.min.apply(null, xs) / 1.3, x1 = Math.max.apply(null, xs) * 1.3;
    var y0 = Math.max(0, Math.min.apply(null, ys) - 0.02), y1 = Math.min(1, Math.max.apply(null, ys) + 0.02);
    function X(v) { return L + (W - L - R) * (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)); }
    function Y(v) { return T + (H - T - B) * (1 - (v - y0) / (y1 - y0 || 1)); }
    for (var e = Math.floor(Math.log10(x0)); Math.pow(10, e) <= x1; e++) {
      [1, 2, 5].forEach(function (f) { var v = f * Math.pow(10, e); if (v >= x0 && v <= x1) {
        el('line', { 'class': 'gridline', x1: X(v), x2: X(v), y1: T, y2: H - B }, svg);
        el('text', { x: X(v), y: H - B + 16, 'text-anchor': 'middle' }, svg, String(v)); } });
    }
    for (var k = 0; k <= 4; k++) { var v = y0 + (y1 - y0) * k / 4;
      el('line', { 'class': 'gridline', x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, svg);
      el('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }, svg, v.toFixed(3)); }
    el('text', { x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle' }, svg, latency() ? 'p50 batch latency in ms (log scale, left is faster)' : 'images per second (log scale)');
    el('text', { x: 14, y: (T + H - B) / 2, transform: 'rotate(-90 14 ' + (T + H - B) / 2 + ')', 'text-anchor': 'middle' }, svg, 'COCO AP');
    var target = parseFloat($('target').value);
    if (!isNaN(target) && target > x0 && target < x1) { el('line', { 'class': 'target', x1: X(target), x2: X(target), y1: T, y2: H - B }, svg); }
    var front = frontier();
    el('polyline', { 'class': 'frontier', points: front.map(function (r) { return X(speed(r)) + ',' + Y(r.ap); }).join(' ') }, svg);
    list.forEach(function (r) {
      var onFront = front.indexOf(r) >= 0;
      var meets = isNaN(target) || (latency() ? r.p50 <= target : r.ips >= target);
      var a = el('a', { href: r.performance_href }, svg);
      el('title', {}, a, describe(r) + ' · ' + fmt(r.ips, 1) + ' images/s · p50 ' + fmt(r.p50, 1) + ' ms · AP ' + fmt(r.ap, 3));
      var cls = 'size-' + (r.size_rank % 4) + ((onFront && meets) ? '' : ' faded');
      var cx = X(speed(r)), cy = Y(r.ap);
      if (r.precision === 'fp16') { el('polygon', { 'class': cls, points: cx + ',' + (cy - 7) + ' ' + (cx - 6) + ',' + (cy + 5) + ' ' + (cx + 6) + ',' + (cy + 5) }, a); }
      else { el('circle', { 'class': cls, cx: cx, cy: cy, r: 6 }, a); }
    });
  }
  function heatmap() {
    var machine = $('machine').value, batch = $('batch').value, mode = $('mode').value;
    var list = data.filter(function (r) { return r.fingerprint === machine && String(r.batch_size) === batch && r.mode === mode; });
    var labels = uniq(data.filter(function (r) { return r.fingerprint === machine; }).map(function (r) { return r.label; })).sort();
    var variants = uniq(list.map(function (r) { return r.size_rank + '|' + r.size + '|' + r.precision; })).sort();
    var html = '<div class="table-wrap"><table class="heat"><thead><tr><th>Variant</th>' + labels.map(function (l) { return '<th class="num">' + l + '</th>'; }).join('') + '</tr></thead><tbody>';
    variants.forEach(function (v) {
      var parts = v.split('|'), cells = labels.map(function (l) { return list.filter(function (r) { return r.label === l && r.size === parts[1] && r.precision === parts[2]; })[0]; });
      var values = cells.filter(Boolean).map(speed), top = latency() ? Math.min.apply(null, values) : Math.max.apply(null, values);
      html += '<tr><th>' + parts[1] + ' ' + parts[2] + '</th>' + cells.map(function (c, i) {
        if (!c) {
          var why = unavailable.filter(function (u) { return u.fingerprint === machine && u.label === labels[i] && u.size === parts[1] && u.precision === parts[2] && String(u.batch_size) === batch; })[0];
          return '<td class="cell empty" title="' + (why ? String(why.reason).replace(/"/g, '&quot;') : 'not measured') + '">' + (why ? 'unavailable' : 'not measured') + '</td>';
        }
        return '<td class="cell' + (speed(c) === top ? ' best' : '') + '"><a href="' + c.performance_href + '">' + (latency() ? fmt(c.p50, 1) + ' ms' : fmt(c.ips, 1)) + '</a></td>';
      }).join('') + '</tr>';
    });
    $('heatmap').innerHTML = html + '</tbody></table></div>';
  }
  function scaling() {
    var svg = $('scaling'); while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var list = data.filter(function (r) { return r.fingerprint === $('machine').value && r.label === $('label').value && r.mode === $('mode').value && speed(r); });
    if (!list.length) { return; }
    var W = 900, H = 300, L = 64, R = 140, T = 16, B = 40;
    var batches = uniq(list.map(function (r) { return r.batch_size; })).sort(function (a, b) { return a - b; });
    var top = Math.max.apply(null, list.map(speed)) * 1.1;
    function X(b) { return batches.length === 1 ? (L + W - R) / 2 : L + (W - L - R) * batches.indexOf(b) / (batches.length - 1); }
    function Y(v) { return T + (H - T - B) * (1 - v / top); }
    batches.forEach(function (b) { el('text', { x: X(b), y: H - B + 16, 'text-anchor': 'middle' }, svg, 'b' + b); });
    for (var k = 0; k <= 4; k++) { var v = top * k / 4; el('line', { 'class': 'gridline', x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, svg);
      el('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }, svg, v.toFixed(0)); }
    el('text', { x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle' }, svg, 'batch size');
    uniq(list.map(function (r) { return r.size_rank + '|' + r.size + '|' + r.precision; })).sort().forEach(function (v) {
      var parts = v.split('|');
      var points = list.filter(function (r) { return r.size === parts[1] && r.precision === parts[2]; }).sort(function (a, b) { return a.batch_size - b.batch_size; });
      var cls = 'size-' + (Number(parts[0]) % 4);
      el('polyline', { 'class': 'line ' + cls, style: 'fill:none' + (parts[2] === 'fp16' ? ';stroke-dasharray:6 4' : ''), points: points.map(function (r) { return X(r.batch_size) + ',' + Y(speed(r)); }).join(' ') }, svg);
      points.forEach(function (r) { var a = el('a', { href: r.performance_href }, svg); el('title', {}, a, describe(r) + ': ' + fmt(speed(r), 1));
        el('circle', { 'class': cls, cx: X(r.batch_size), cy: Y(speed(r)), r: 4 }, a); });
      var last = points[points.length - 1]; el('text', { x: W - R + 8, y: Y(speed(last)) + 4 }, svg, parts[1] + ' ' + parts[2]);
    });
  }
  function update() {
    $('target-label').firstChild.textContent = latency() ? 'Max p50 latency (ms)' : 'Target images/s';
    var list = selected(); recommendations(list); scatter(list); heatmap(); scaling();
  }
  $('machine').addEventListener('change', function () { fillLabels(); update(); });
  ['label', 'batch', 'mode', 'metric'].forEach(function (id) { $(id).addEventListener('change', update); });
  $('target').addEventListener('input', update);
  $('views').hidden = false; $('nojs').hidden = true;
  update();
})();
"""


def _notes_marker(row: Mapping[str, Any]) -> str:
    """Return a small marker whose hover text holds the model notes, or an empty string."""
    notes = [f"{label}: {row[key]}" for key, label in (("accuracy_notes", "Accuracy"), ("performance_notes", "Performance"))
             if row.get(key)]
    if not notes:
        return ""
    return f' <span class="note-marker" title="{esc(chr(10).join(notes))}" aria-label="This model has notes">ⓘ</span>'


def render_overview_html(family: str, rows: list[dict[str, Any]], unavailable: list[dict[str, Any]]) -> str:
    """Render the family overview page."""
    rows = [{**row, "id": index} for index, row in enumerate(rows)]
    title = f"{family} overview"
    table_rows = [
        [
            (link(row["model_href"], row["subject"]) + _notes_marker(row), row["subject"]),
            esc(row["size"]),
            esc(row["precision"]),
            (esc(row["batch_size"]), row["batch_size"]),
            esc(row["label"]),
            esc(row["host"]),
            esc(row["mode"]),
            (link(row["performance_href"], fmt(row["ips"])), row["ips"]),
            (fmt(row["p50"], 2), row["p50"]),
            (link(row["accuracy_href"], fmt(row["ap"], 4)), row["ap"]),
            (fmt(row["delta_ap"], 4), row["delta_ap"] if row["delta_ap"] is not None else ""),
            (esc(row["runs"]), row["runs"]),
        ]
        for row in rows
    ]
    body = [
        f"<h1>{esc(title)}</h1>",
        '<p class="subtitle">Pick your hardware and batch size to see the fastest variant and the best '
        "accuracy/speed balance. Every mark and cell links to its benchmark report.</p>",
    ]
    if not rows:
        body.append('<p class="muted">No performance benchmarks for models with a model report yet.</p>')
    else:
        body.append(
            '<div class="controls">'
            '<label>Machine<select id="machine"></select></label>'
            '<label>Hardware<select id="label"></select></label>'
            '<label>Batch size<select id="batch"></select></label>'
            '<label>Mode<select id="mode"></select></label>'
            '<label>Speed metric<select id="metric"><option value="throughput">throughput (images/s)</option>'
            '<option value="latency">latency (p50 ms per batch)</option></select></label>'
            '<label id="target-label">Target images/s<input id="target" type="number" min="0" step="any" '
            'placeholder="for example 30 × cameras"></label>'
            "</div>"
            '<p id="nojs" class="muted">The charts need JavaScript. The full table below has every row.</p>'
            '<div id="views" hidden>'
            '<h2>Recommendations</h2><div id="cards" class="grid"></div>'
            "<h2>Accuracy vs speed</h2>"
            '<div class="legend">Colour: model size · circle fp32, triangle fp16 · dashed line: Pareto frontier · '
            "faded: not on the frontier, or misses the target · accuracy is measured on the accuracy device "
            "(CUDA), not on the selected hardware</div>"
            '<svg id="scatter" viewBox="0 0 900 380" width="100%" role="img" aria-label="AP against speed"></svg>'
            "<h2>Fastest hardware for each variant</h2>"
            '<p class="muted">The best cell in each row has an outline.</p><div id="heatmap"></div>'
            "<h2>Batch scaling</h2>"
            '<p class="muted">Throughput by batch size on the selected hardware. Dashed lines are fp16.</p>'
            '<svg id="scaling" viewBox="0 0 900 300" width="100%" role="img" aria-label="Throughput by batch size"></svg>'
            "</div>"
        )
    body.append("<h2>All rows</h2>")
    body.append(data_table(
        ["Model", "Size", "Precision", "Batch", "Hardware", "Host", "Mode", "Images/s", "p50 ms", "AP", "Δ AP vs parent", "Runs"],
        table_rows,
        sortable=True,
        numeric=(3, 7, 8, 9, 10, 11),
    ))
    if unavailable:
        body.append("<h2>Unavailable benchmarks</h2>")
        body.append(data_table(
            ["Model", "Hardware", "Host", "Reason"],
            [[link(item["href"], item["subject"]), esc(item["label"]), esc(item["host"]), esc(item["reason"])]
             for item in unavailable],
        ))
    body.append(data_island(ROWS_ELEMENT_ID, {"rows": rows, "unavailable": unavailable, "groups": frontier_groups(rows)}))
    scripts = (SORT_SCRIPT, OVERVIEW_SCRIPT) if rows else (SORT_SCRIPT,)
    return page(title, "".join(body), scripts=scripts, head=OVERVIEW_STYLE)


def render_family_overview(family_dir: str | Path, *, index: FamilyIndex | None = None) -> Path | None:
    """Write ``<family>.overview.html``. Return ``None`` when the family has no model reports."""
    index = index or scan_family(family_dir)
    families = sorted({entry.payload["model"]["variant"]["family"] for entry in index.models.values()})
    if not families:
        return None
    path = overview_path(family_dir, families[0])
    rows, unavailable = overview_rows(index, path)
    return atomic_write_text(path, render_overview_html(families[0], rows, unavailable))


__all__ = [
    "best_for_target",
    "frontier_groups",
    "knee",
    "overview_rows",
    "pareto_frontier",
    "render_family_overview",
    "render_overview_html",
]
