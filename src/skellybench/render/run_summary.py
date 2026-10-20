"""Run summary HTML.

A run summary has no JSON of its own. Its benchmark and model rows come from the schema-3 JSON
reports of the run. Its run state (status, warnings, error, exit code) lives in a JSON data island
inside the page, so a rebuild keeps it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import data_island, data_table, esc, fmt, kv_table, link, page, pill

STATE_ELEMENT_ID = "run-state"
_STATE_PATTERN = re.compile(
    r'<script type="application/json" id="' + STATE_ELEMENT_ID + r'">(.*?)</script>', re.DOTALL
)
RUN_STATUSES = ("in_progress", "finished", "failed")


def read_run_state(path: str | Path) -> dict[str, Any] | None:
    """Return the run state stored in a run summary page, or ``None``."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = _STATE_PATTERN.search(text)
    if match is None:
        return None
    try:
        state = json.loads(match.group(1).replace("<\\/", "</"))
    except json.JSONDecodeError:
        return None
    return state if isinstance(state, dict) else None


def render_run_summary_html(
    state: Mapping[str, Any],
    *,
    benchmarks: Sequence[tuple[Mapping[str, Any], str]],
    models: Sequence[tuple[Mapping[str, Any], str]],
) -> str:
    """Render the run summary.

    Args:
        state: ``run_id``, ``status``, ``started_at``, ``finished_at``, ``command``, ``hostname``,
            ``producer``, ``exit_code``, ``error``, ``warnings``.
        benchmarks: benchmark payloads of this run with links from the summary page.
        models: model payloads created in this run with links from the summary page.
    """
    run_id = state["run_id"]
    status = state.get("status", "finished")
    title = f"Run {run_id}"
    producer = state.get("producer") or {}
    parts = [f"<h1>{esc(title)}</h1>", f'<p class="subtitle">{pill(status)}</p>']
    if status == "in_progress":
        parts.append('<div class="banner info">This run is in progress, or it stopped without a summary. '
                     "Run <code>--render-reports</code> to rebuild this page from the reports on disk.</div>")
    if status == "failed":
        parts.append(f'<div class="banner fail">The run failed: {esc(state.get("error"))}</div>')
    failures = [
        (payload, href)
        for payload, href in models
        if ((payload.get("accuracy") or {}).get("comparison") or {}).get("status") == "fail"
    ]
    if failures:
        parts.append(
            '<div class="banner fail"><strong>Accuracy comparison failures.</strong> The sidecar leaves out '
            "these models and every model made from them:<ul>"
            + "".join(
                f"<li>{link(href, payload['report_id'])} — Δ AP "
                f"{fmt((payload['accuracy']['comparison'].get('deltas') or {}).get('delta_ap'), 4)}</li>"
                for payload, href in failures
            )
            + "</ul></div>"
        )
    parts.append(kv_table([
        ("Command", f"<code>{esc(state.get('command'))}</code>"),
        ("Host", esc(state.get("hostname"))),
        ("Started", esc(state.get("started_at"))),
        ("Finished", esc(state.get("finished_at"))),
        ("Exit code", esc(state.get("exit_code"))),
        ("Git commit", f"<code>{esc(producer.get('git_commit'))}</code>"
         + (' <span class="pill warn">dirty</span>' if producer.get("git_dirty") else "")),
    ]))
    warnings = state.get("warnings") or []
    if warnings:
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{esc(item)}</li>" for item in warnings) + "</ul>")

    parts.append("<h2>Models created</h2>")
    if models:
        parts.append(data_table(
            ["Model", "Precision", "Batch", "Δ AP vs parent", "Comparison"],
            [
                [
                    link(href, payload["report_id"]),
                    esc(payload["model"].get("precision")),
                    esc(payload["model"].get("batch_size") or "any"),
                    fmt(((payload["accuracy"].get("comparison") or {}).get("deltas") or {}).get("delta_ap"), 4),
                    pill((payload["accuracy"].get("comparison") or {}).get("status") or "n/a"),
                ]
                for payload, href in sorted(models, key=lambda item: item[0]["report_id"])
            ],
            numeric=(3,),
        ))
    else:
        parts.append('<p class="muted">None.</p>')

    accuracy = [(payload, href) for payload, href in benchmarks if payload["benchmark_type"] == "accuracy"]
    parts.append("<h2>Accuracy benchmarks</h2>")
    if accuracy:
        parts.append(data_table(
            ["Report", "Device", "Status", "AP", "AP50"],
            [
                [
                    link(href, payload["subject"]["subject_id"]),
                    esc(payload["execution"]["label"]),
                    pill(payload["status"]),
                    fmt((((payload.get("accuracy") or {}).get("metrics") or {}).get("values") or {}).get("ap"), 4),
                    fmt((((payload.get("accuracy") or {}).get("metrics") or {}).get("values") or {}).get("ap50"), 4),
                ]
                for payload, href in sorted(accuracy, key=lambda item: item[0]["report_id"])
            ],
            numeric=(3, 4),
        ))
    else:
        parts.append('<p class="muted">None.</p>')

    performance = [(payload, href) for payload, href in benchmarks if payload["benchmark_type"] == "performance"]
    parts.append("<h2>Performance benchmarks</h2>")
    if not performance:
        parts.append('<p class="muted">None.</p>')
    by_label: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    for payload, href in performance:
        by_label.setdefault(payload["execution"]["label"], []).append((payload, href))
    for label in sorted(by_label):
        parts.append(f"<h3>{esc(label)}</h3>")
        rows = []
        for payload, href in sorted(by_label[label], key=lambda item: item[0]["report_id"]):
            records = (payload.get("performance") or {}).get("records") or []
            best = max((float((record.get("timing") or {}).get("images_per_second") or 0) for record in records), default=None)
            rows.append([
                link(href, payload["subject"]["subject_id"]),
                pill(payload["status"]),
                (fmt(best), best if best is not None else ""),
                esc(payload.get("reason") or ""),
            ])
        parts.append(data_table(["Artifact", "Status", "Best images/s", "Reason"], rows, numeric=(2,)))

    unavailable = [(payload, href) for payload, href in benchmarks if payload["status"] != "measured"]
    if unavailable:
        parts.append("<h2>Unavailable or failed benchmarks</h2>")
        parts.append(data_table(
            ["Report", "Status", "Reason"],
            [[link(href, payload["report_id"]), pill(payload["status"]), esc(payload.get("reason"))]
             for payload, href in unavailable],
        ))
    parts.append(data_island(STATE_ELEMENT_ID, dict(state)))
    return page(title, "".join(parts))


__all__ = ["RUN_STATUSES", "STATE_ELEMENT_ID", "read_run_state", "render_run_summary_html"]
