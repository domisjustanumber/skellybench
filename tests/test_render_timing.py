"""Pages show the end-to-end time, the stage breakdown, the dataset and readable decode blocks (0.4.0)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import (
    NULL_STAGES,
    RUN_ID,
    context,
    execution,
    performance_dataset,
    sha,
    subject,
    write_accuracy,
    write_performance,
)
from skellybench.protocol import PerformanceProtocol
from skellybench.render import render_benchmark_html, render_family, write_run_summary
from skellybench.render.common import stage_timing, value_html
from skellybench.reports import build_benchmark_report
from test_reports import _lineage

NESTED_METHOD = {"decode": {"kind": "rtmdet", "score_threshold": 0.001, "nms": {"iou_threshold": 0.65, "max_per_image": 300},
                            "classes": [0], "agnostic": False, "pre_nms": None},
                 "scorer": {"tool": "faster-coco-eval", "version": "1.8.0"}, "scored_by": "skellybench"}
LATER = "20261021T090000Z-gerald"


def _page(family: Path, subject_id: str, kind: str, provider: str = "cuda", run_id: str = RUN_ID) -> str:
    path = family / "benchmarks" / subject_id / f"{subject_id}.{kind}.{provider}.{run_id}.html"
    return path.read_text(encoding="utf-8")


def test_stage_timing_computes_other_and_hides_the_pass_provider() -> None:
    exe = execution(context()["host"])
    timing = {"images_per_second": 250.0, "ms_per_image": 4.0,
              "stages_ms_per_image": {"preprocess": 1.8, "inference": 1.5, "postprocess": 0.3},
              "stage_providers": {"preprocess": None, "inference": "CUDAExecutionProvider", "postprocess": "CPUExecutionProvider"},
              "excluded_ms_per_image": {"letterbox": 0.9}}
    stages = stage_timing(timing, exe)
    assert round(stages.other, 6) == 0.4
    assert stages.providers == {"postprocess": "CPUExecutionProvider"}
    assert stages.breakdown() == ("4.00 ms/image (preprocess 1.80, inference 1.50, postprocess 0.30, other 0.40 ms)"
                                  " · postprocess on CPUExecutionProvider · excluded: letterbox 0.90 ms/image")
    partial = stage_timing({"images_per_second": 200.0, "stages_ms_per_image": {**NULL_STAGES, "inference": 2.0}})
    assert partial.ms_per_image == 5.0 and partial.other is None


def test_value_html_is_a_nested_key_value_list() -> None:
    html = value_html(NESTED_METHOD["decode"])
    assert "{'" not in html and "{&#x27;" not in html
    assert '<span class="key">nms</span>: <ul class="kv-list"><li><span class="key">iou_threshold</span>: 0.65</li>' in html
    assert '<span class="key">classes</span>: 0' in html
    assert '<span class="key">agnostic</span>: false' in html and '<span class="key">pre_nms</span>: none' in html
    assert value_html({}) == '<span class="muted">none</span>'


def test_benchmark_pages_show_stages_providers_letterbox_definitions_and_dataset(family: Path) -> None:
    _lineage(family)
    later = context(LATER)
    write_performance(family, later, subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(later["host"]), 120.0,
                      decode_provider="CPUExecutionProvider")
    render_family(family)
    html = _page(family, "rfdetr-nano_b1_fp32", "performance", run_id=LATER)
    for text in ("Images/s", "ms/image", "Preprocess ms/image", "Inference ms/image", "Postprocess ms/image",
                 "Other ms/image", "Stage providers"):
        assert f">{text}</th>" in html, text
    assert ">8.33</td>" in html and ">3.75</td>" in html and ">0.42</td>" in html
    assert "postprocess: CPUExecutionProvider" in html
    assert "letterbox 0.50 ms/image (device)" in html
    assert "<h3>Stage definitions</h3>" in html and "the host-to-device copy of the input" in html
    assert "no annotations read" in html and "none (this pass reads no annotations)" in html
    assert "stride sample of 120" in html
    assert "{'" not in html
    same = _page(family, "rfdetr-nano_b1_fp32", "performance")
    assert "postprocess: " not in same


def test_null_stages_render_as_not_measured(family: Path) -> None:
    ctx = context()
    timing = {"images_per_second": 40.0, "stages_ms_per_image": dict(NULL_STAGES)}
    report = build_benchmark_report(
        benchmark_type="performance", status="measured", execution=execution(ctx["host"]),
        subject=subject("rfdetr-nano_b1_fp32", sha("fp32")), dataset=performance_dataset(),
        protocol=PerformanceProtocol().block(batch_size=1),
        performance={"records": [{"mode": "host", "timing_scope": "end_to_end", "timing": timing}]}, **ctx)
    html = render_benchmark_html(report)
    row = re.search(r"<tbody><tr><td>host</td>.*?</tr>", html).group(0)
    assert row.count('title="not measured">—</span>') == 4
    assert ">25.00</td>" in row
    assert ">0.00<" not in row and ">0<" not in row
    assert "The protocol defines no stages." in html


def test_accuracy_pages_render_decode_as_key_value_list(family: Path) -> None:
    ctx = context()
    write_accuracy(family, ctx, subject("rfdetr-nano_b1_fp32", sha("fp32")), execution(ctx["host"]), 0.5,
                   method=NESTED_METHOD)
    render_family(family)
    html = _page(family, "rfdetr-nano_b1_fp32", "accuracy")
    assert "{'" not in html
    assert '<span class="key">iou_threshold</span>: 0.65' in html
    assert "<code>instances_val2017.json</code>" in html and "categories 1 person" in html
    assert "split val2017" in html


def test_model_page_shows_dataset_stage_columns_and_readable_methods(family: Path) -> None:
    _lineage(family)
    render_family(family)
    html = (family / "rfdetr-nano_b1_fp32.model.html").read_text(encoding="utf-8")
    assert "{'" not in html
    line = "coco-val2017 · split val2017 · <code>instances_val2017.json</code> · categories 1 person · 2 of 2 images (all)"
    assert f'<p class="subtitle">Accuracy dataset: {line}</p>' in html
    assert f"<p>Dataset: {line}</p>" in html
    for text in ("ms/image", "Preprocess ms/image", "Inference ms/image", "Postprocess ms/image", "Other ms/image"):
        assert f">{text}</th>" in html, text
    assert "8.33 ms/image (preprocess 3.75, inference 3.33, postprocess 0.83, other 0.42 ms)" in html
    assert "letterbox (0.50 ms/image)" in html
    assert '<span class="key">score_threshold</span>: 0.0' in html


def test_overview_shows_dataset_stage_columns_and_batch_latency_name(family: Path) -> None:
    _lineage(family)
    render_family(family)
    html = (family / "rfdetr.overview.html").read_text(encoding="utf-8")
    assert "Accuracy dataset: coco-val2017 · split val2017" in html
    assert html.count("Accuracy dataset:") == 1
    for text in ("ms/image", "Preprocess ms/image", "Other ms/image", "p50 ms per batch"):
        assert f">{text}</th>" in html, text
    assert ">p50 ms</th>" not in html
    island = re.search(r'id="overview-rows">(.*?)</script>', html, re.DOTALL).group(1)
    rows = json.loads(island.replace("<\\/", "</"))["rows"]
    device = next(row for row in rows if row["label"] == "CUDA (RTX 3070)" and row["mode"] == "device")
    assert round(device["ms"], 2) == 8.33 and round(device["other"], 2) == 0.42
    assert device["breakdown"].startswith("8.33 ms/image (preprocess 3.75")
    assert "r.breakdown" in html


def test_run_summary_shows_images_per_second_and_ms_per_image(family: Path) -> None:
    _lineage(family)
    summary = write_run_summary(family.parents[2], {"run_id": RUN_ID, "status": "finished", "command": "x",
                                                    "hostname": "Gerald"})
    html = summary.read_text(encoding="utf-8")
    assert ">Images/s</th>" in html and ">ms/image</th>" in html
    assert ">120.00</td>" in html and ">8.33</td>" in html
    assert ">96.00</td>" in html and ">10.42</td>" in html
