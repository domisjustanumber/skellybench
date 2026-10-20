"""Render every HTML page from the JSON reports on disk.

Nothing here writes JSON. Every page can be deleted and rebuilt with :func:`rebuild_all`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .._internal.io import atomic_write_text, write_text_once
from ..paths import RUNS_DIR, href, resolve_family_path, run_summary_path
from ..reports import FamilyIndex, ModelEntry, find_family_dirs, scan_family
from .benchmark import render_benchmark_html
from .model import LinkedBenchmark, render_model_html
from .run_summary import read_run_state, render_run_summary_html


def _find_model(index: FamilyIndex, report_id: str, sha256: str) -> ModelEntry | None:
    for entry in index.all_models():
        if entry.payload["report_id"] == report_id and entry.payload["model"]["artifact"]["sha256"] == sha256:
            return entry
    return None


def render_benchmark_pages(family_dir: str | Path, *, index: FamilyIndex | None = None,
                           report_ids: Iterable[str] | None = None) -> int:
    """Render benchmark HTML pages. Return the count written."""
    index = index or scan_family(family_dir)
    wanted = set(report_ids) if report_ids is not None else None
    written = 0
    for entry in index.benchmarks:
        if wanted is not None and entry.payload["report_id"] not in wanted:
            continue
        model = index.model_for_benchmark(entry.payload)
        predictions = (entry.payload.get("accuracy") or {}).get("predictions") or {}
        predictions_href = None
        if predictions.get("filename"):
            try:
                predictions_href = href(entry.html_path, resolve_family_path(family_dir, predictions["filename"]))
            except ValueError:
                predictions_href = None
        atomic_write_text(
            entry.html_path,
            render_benchmark_html(
                entry.payload,
                model_href=href(entry.html_path, model.html_path) if model else None,
                predictions_href=predictions_href,
            ),
        )
        written += 1
    return written


def _ancestor_comparisons(index: FamilyIndex, report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    chain: list[tuple[str, Mapping[str, Any]]] = []
    parent = report["creation"].get("parent")
    seen = {report["report_id"]}
    while parent and parent.get("report_id") not in seen:
        seen.add(parent["report_id"])
        entry = _find_model(index, parent["report_id"], parent["sha256"])
        if entry is None:
            break
        comparison = entry.payload["accuracy"].get("comparison")
        if comparison:
            chain.append((entry.payload["report_id"], comparison))
        parent = entry.payload["creation"].get("parent")
    return chain


def render_model_page(index: FamilyIndex, entry: ModelEntry) -> Path:
    """Render one model page (current or superseded)."""
    report = entry.payload
    page_path = entry.html_path
    sha = report["model"]["artifact"]["sha256"]
    benchmarks = [
        LinkedBenchmark(item.payload, href(page_path, item.html_path))
        for item in index.benchmarks_for(report["report_id"], sha)
    ]
    parent = report["creation"].get("parent")
    parent_href = parent_benchmark = None
    parent_ap_by_run: dict[str, float] = {}
    if parent:
        parent_entry = _find_model(index, parent["report_id"], parent["sha256"])
        if parent_entry is not None:
            parent_href = href(page_path, parent_entry.html_path)
        for item in index.benchmarks_for(parent["report_id"], parent["sha256"]):
            values = (((item.payload.get("accuracy") or {}).get("metrics") or {}).get("values") or {})
            if item.payload["benchmark_type"] == "accuracy" and "ap" in values:
                parent_ap_by_run[item.payload["run"]["run_id"]] = float(values["ap"])
        parent_accuracy = report["accuracy"].get("parent")
        if parent_accuracy:
            snapshot = parent_accuracy["snapshot"]
            values = ((snapshot.get("accuracy") or {}).get("metrics") or {}).get("values") or {}
            if "ap" in values:
                parent_ap_by_run.setdefault(snapshot["run"]["run_id"], float(values["ap"]))
            target = resolve_family_path(index.family_dir, parent_accuracy["link"]["path"]).with_suffix(".html")
            parent_benchmark = LinkedBenchmark(snapshot, href(page_path, target))
    current_href = None
    if entry.superseded:
        current = index.models.get(report["report_id"])
        current_href = href(page_path, current.html_path) if current else None
    html = render_model_html(
        report,
        benchmarks=benchmarks,
        parent_href=parent_href,
        parent_benchmark=parent_benchmark,
        parent_ap_by_run=parent_ap_by_run,
        ancestor_comparisons=_ancestor_comparisons(index, report),
        current_href=current_href,
        superseded=entry.superseded,
    )
    return atomic_write_text(page_path, html)


def render_model_pages(family_dir: str | Path, *, index: FamilyIndex | None = None,
                       subjects: Iterable[str] | None = None) -> int:
    """Render model HTML pages (current and superseded). Return the count written."""
    index = index or scan_family(family_dir)
    wanted = set(subjects) if subjects is not None else None
    written = 0
    for entry in index.all_models():
        if wanted is not None and entry.payload["report_id"] not in wanted:
            continue
        render_model_page(index, entry)
        written += 1
    return written


def render_family(family_dir: str | Path) -> dict[str, int]:
    """Render every benchmark and model page of one family."""
    index = scan_family(family_dir)
    counts = {
        "benchmark_pages": render_benchmark_pages(family_dir, index=index),
        "model_pages": render_model_pages(family_dir, index=index),
    }
    try:
        from .overview import render_family_overview
    except ImportError:  # pragma: no cover - overview is optional
        return counts
    counts["overview_pages"] = 1 if render_family_overview(family_dir, index=index) else 0
    return counts


# --- run summaries ---------------------------------------------------------------------------------


def _run_entries(exports_root: Path, run_id: str, summary: Path) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]]]:
    benchmarks: list[tuple[Any, str]] = []
    models: list[tuple[Any, str]] = []
    for family_dir in find_family_dirs(exports_root):
        index = scan_family(family_dir)
        for entry in index.benchmarks:
            if entry.payload["run"]["run_id"] == run_id:
                benchmarks.append((entry.payload, href(summary, entry.html_path)))
        for model in index.all_models():
            if model.payload["created_run_id"] == run_id:
                models.append((model.payload, href(summary, model.html_path)))
    return benchmarks, models


def claim_run(exports_root: str | Path, state: Mapping[str, Any]) -> Path:
    """Claim a run id: create ``runs/<run_id>.html`` with exclusive create.

    Raise :class:`FileExistsError` when the run id is taken. The placeholder says "run in progress".
    """
    path = run_summary_path(exports_root, str(state["run_id"]))
    placeholder = {**dict(state), "status": "in_progress"}
    write_text_once(path, render_run_summary_html(placeholder, benchmarks=[], models=[]))
    return path


def write_run_summary(exports_root: str | Path, state: Mapping[str, Any]) -> Path:
    """Render the run summary from the run's JSON reports and *state*, replacing the placeholder."""
    root = Path(exports_root)
    path = run_summary_path(root, str(state["run_id"]))
    benchmarks, models = _run_entries(root, str(state["run_id"]), path)
    return atomic_write_text(path, render_run_summary_html(state, benchmarks=benchmarks, models=models))


def rebuild_run_summaries(exports_root: str | Path) -> int:
    """Re-render every run summary below ``<exports_root>/runs`` from its stored state."""
    root = Path(exports_root)
    written = 0
    for path in sorted((root / RUNS_DIR).glob("*.html")):
        state = read_run_state(path)
        if not state or state.get("run_id") != path.stem:
            continue
        write_run_summary(root, state)
        written += 1
    return written


def rebuild_all(exports_root: str | Path) -> dict[str, int]:
    """Re-render every HTML page below *exports_root* from JSON. Never writes JSON."""
    totals: dict[str, int] = {"families": 0}
    for family_dir in find_family_dirs(exports_root):
        totals["families"] += 1
        for key, value in render_family(family_dir).items():
            totals[key] = totals.get(key, 0) + value
    totals["run_summaries"] = rebuild_run_summaries(exports_root)
    return totals


__all__ = [
    "claim_run",
    "rebuild_all",
    "rebuild_run_summaries",
    "render_benchmark_pages",
    "render_family",
    "render_model_page",
    "render_model_pages",
    "write_run_summary",
]
