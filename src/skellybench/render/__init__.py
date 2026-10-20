"""HTML rendering for schema-3 reports. Pages are rebuilt from JSON and never hold the only copy of data."""

from .benchmark import render_benchmark_html
from .model import LinkedBenchmark, render_model_html
from .run_summary import read_run_state, render_run_summary_html
from .site import (
    claim_run,
    rebuild_all,
    rebuild_run_summaries,
    render_benchmark_pages,
    render_family,
    render_model_page,
    render_model_pages,
    write_run_summary,
)

__all__ = [
    "LinkedBenchmark",
    "claim_run",
    "read_run_state",
    "rebuild_all",
    "rebuild_run_summaries",
    "render_benchmark_html",
    "render_benchmark_pages",
    "render_family",
    "render_model_html",
    "render_model_page",
    "render_model_pages",
    "render_run_summary_html",
    "write_run_summary",
]
