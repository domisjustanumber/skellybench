"""Family overview: row flattening, Pareto frontier, knee and target rules, and the page."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import context, execution, sha, subject, write_performance
from skellybench.render import render_family
from skellybench.render.overview import best_for_target, frontier_groups, knee, overview_rows, pareto_frontier
from skellybench.reports import build_benchmark_report, scan_family, write_benchmark_report
from test_reports import _lineage


def _row(name: str, ips: float, ap: float, p50: float | None = None) -> dict:
    return {"subject": name, "ips": ips, "p50": p50 if p50 is not None else 1000 / ips, "ap": ap}


def test_pareto_frontier_keeps_rows_no_other_row_beats() -> None:
    rows = [_row("fast", 300, 0.40), _row("mid", 150, 0.48), _row("dominated", 140, 0.45), _row("slow", 60, 0.53)]
    assert [row["subject"] for row in pareto_frontier(rows)] == ["fast", "mid", "slow"]
    assert [row["subject"] for row in pareto_frontier(rows, "latency")] == ["fast", "mid", "slow"]


def test_pareto_ties_and_single_row() -> None:
    tie = [_row("a", 100, 0.5), _row("b", 100, 0.6)]
    assert [row["subject"] for row in pareto_frontier(tie)] == ["b"]
    single = [_row("only", 50, 0.3)]
    assert knee(pareto_frontier(single)) is single[0]
    assert knee([]) is None


def test_knee_picks_largest_ap_gain_per_halving() -> None:
    # fast -> mid: +0.08 AP for 1 halving; mid -> slow: +0.05 AP for ~1.32 halvings.
    frontier = pareto_frontier([_row("fast", 300, 0.40), _row("mid", 150, 0.48), _row("slow", 60, 0.53)])
    assert knee(frontier)["subject"] == "mid"


def test_target_rule() -> None:
    rows = [_row("fast", 300, 0.40), _row("mid", 150, 0.48), _row("slow", 60, 0.53)]
    assert best_for_target(rows, 100)["subject"] == "mid"
    assert best_for_target(rows, 1000) is None
    assert best_for_target(rows, 10, "latency")["subject"] == "mid"


def test_rows_use_newest_run_and_count_others(family: Path) -> None:
    _lineage(family)
    later = context("20261021T090000Z-gerald")
    write_performance(family, later, subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(later["host"]), 140.0)
    rows, _ = overview_rows(scan_family(family), family / "rfdetr.overview.html")
    cuda_device = [row for row in rows if row["label"] == "CUDA (RTX 3070)" and row["mode"] == "device"]
    assert len(cuda_device) == 1
    assert cuda_device[0]["ips"] == 140.0 and cuda_device[0]["runs"] == 2
    assert not [row for row in rows if row["subject"] == "rfdetr-nano_source"]
    assert all(row["ap"] == 0.5 for row in rows)


def test_two_hosts_with_the_same_gpu_label_stay_separate(family: Path) -> None:
    _lineage(family)
    other = context("20261021T090000Z-other", hostname="Other")
    other["host"]["hardware_fingerprint"] = "sha256:" + "f" * 64
    write_performance(family, other, subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(other["host"]), 90.0)
    rows, _ = overview_rows(scan_family(family), family / "rfdetr.overview.html")
    hosts = {row["host"] for row in rows if row["label"] == "CUDA (RTX 3070)" and row["mode"] == "device"}
    assert hosts == {"Gerald", "Other"}


def test_page_embeds_groups_and_lists_every_row_without_javascript(family: Path) -> None:
    _lineage(family)
    ctx = context()
    report = build_benchmark_report(
        benchmark_type="performance", status="unavailable", reason="onnxruntime-ep-openvino is not installed",
        execution={**execution(ctx["host"], "cpu"), "provider": {**execution(ctx["host"], "cpu")["provider"],
                                                                 "key": "openvino_gpu", "slug": "openvino-gpu"}},
        subject=subject("rfdetr-nano_b1_fp32", sha("fp32")), **ctx,
    )
    write_benchmark_report(family, report)
    render_family(family)
    html = (family / "rfdetr.overview.html").read_text(encoding="utf-8")
    island = re.search(r'id="overview-rows">(.*?)</script>', html, re.DOTALL).group(1)
    data = json.loads(island.replace("<\\/", "</"))
    assert len(data["rows"]) == 4
    assert all("throughput" in group and "latency" in group for group in data["groups"].values())
    assert data["unavailable"][0]["reason"] == "onnxruntime-ep-openvino is not installed"
    assert html.count("<tr>") >= len(data["rows"]) + 1
    assert 'href="benchmarks/rfdetr-nano_b1_fp32/' in html


def test_groups_match_python_rules(family: Path) -> None:
    rows = [{**_row(name, ips, ap), "id": index, "fingerprint": "f", "label": "CUDA", "batch_size": 1, "mode": "device"}
            for index, (name, ips, ap) in enumerate([("fast", 300, 0.4), ("mid", 150, 0.48), ("slow", 60, 0.53)])]
    groups = frontier_groups(rows)
    assert groups["f|CUDA|1|device"]["throughput"] == {"frontier": [0, 1, 2], "knee": 1}
