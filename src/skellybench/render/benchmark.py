"""Benchmark report HTML."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..blocks import describe_container_limits
from ..schema import COCO_STAT_KEYS
from .common import (
    SORT_SCRIPT,
    TIMING_HEADERS,
    cards,
    data_table,
    dataset_line,
    esc,
    excluded_note,
    fmt,
    fmt_bytes,
    json_details,
    kv_table,
    link,
    page,
    pill,
    stage_definitions_table,
    stage_timing,
    value_html,
)

KEY_PACKAGES = ("onnxruntime", "onnxruntime-gpu", "torch", "torchvision", "numpy", "onnx", "faster-coco-eval")


def reproduction_rows(run: Mapping[str, Any], environment: Mapping[str, Any], execution: Mapping[str, Any] | None,
                      *, extra_packages: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Rows that tell a reader how to reproduce a run."""
    producer = run.get("producer") or {}
    packages = environment.get("packages") or {}
    python = environment.get("python") or {}
    rows = [
        ("Command", f'<code>{esc(run.get("command"))}</code>'),
        ("Producer", f"{esc(producer.get('name'))} {esc(producer.get('version'))}"),
        ("Git commit", f'<code>{esc(producer.get("git_commit"))}</code>'
         + (' <span class="pill warn">dirty</span>' if producer.get("git_dirty") else "")),
    ]
    if producer.get("git_dirty"):
        rows.append(("Uncommitted diff SHA-256", f'<code>{esc(producer.get("git_diff_sha256"))}</code>'))
    rows.append(("Lock file SHA-256", f'<code>{esc(producer.get("lock_file_sha256"))}</code>'))
    if environment.get("collected_by"):
        rows.append(("Environment collected by", esc(environment["collected_by"])))
    container = environment.get("container")
    if container:
        tag = f":{container['tag']}" if container.get("tag") else ""
        digest = f" <code>{esc(container['digest'])}</code>" if container.get("digest") else ""
        rows.append(("Container image", f"{esc(container.get('image'))}{esc(tag)}{digest}"
                     f" · {'published' if container.get('published') else 'not published'}"))
        if container.get("dockerfile_sha256"):
            rows.append(("Dockerfile SHA-256", f"<code>{esc(container['dockerfile_sha256'])}</code>"))
    rows.append(("Python", f"{esc(python.get('implementation'))} {esc(python.get('version'))}"))
    names = [name for name in (*extra_packages, *KEY_PACKAGES) if name in packages]
    if names:
        rows.append(("Key packages", ", ".join(f"{esc(name)} {esc(packages[name])}" for name in dict.fromkeys(names))))
    sources = environment.get("package_sources") or {}
    for name, source in sources.items():
        detail = source.get("commit") or source.get("local") or source.get("url")
        rows.append((f"Source of {name}", f'<code>{esc(detail)}</code>'))
    if execution:
        drivers = execution.get("drivers") or {}
        for key, value in drivers.items():
            if key == "loaded_libraries" and isinstance(value, Mapping):
                text = ", ".join(f"{esc(name)} {esc(version)}" for name, version in value.items()) or "none"
                rows.append(("Loaded native libraries", text))
            else:
                rows.append((key.replace("_", " ").capitalize(), esc(value)))
    return rows


def host_rows(host: Mapping[str, Any]) -> list[tuple[str, str]]:
    os_info = host.get("os") or {}
    cpu = host.get("cpu") or {}
    virtualization = host.get("virtualization") or {}
    rows = [
        ("Hostname", esc(host.get("hostname"))),
        ("OS", f"{esc(os_info.get('distribution') or os_info.get('name'))} ({esc(os_info.get('arch'))})"),
        ("CPU", f"{esc(cpu.get('name'))} — {fmt(cpu.get('physical_cores'))} cores / {fmt(cpu.get('logical_cores'))} threads"),
        ("Memory", f"{fmt((host.get('memory') or {}).get('total_mb'))} MB"),
    ]
    for gpu in host.get("gpus") or []:
        memory = f", {fmt(gpu.get('vram_mb'))} MB" if gpu.get("vram_mb") else ""
        rows.append(("GPU", f"{esc(gpu.get('name'))} — driver {esc(gpu.get('driver'))}{memory}"))
    flags = [
        name
        for name, key in (("WSL", "is_wsl"), ("container", "is_container"), ("VM", "is_vm"))
        if virtualization.get(key)
    ]
    rows.append(("Virtualization", esc(", ".join(flags) or "none detected")
                 + (f" ({esc(virtualization.get('hypervisor'))})" if virtualization.get("hypervisor") else "")))
    container = host.get("container")
    if container:
        rows.append(("Container image", f"{esc(container.get('image'))} <code>{esc(container.get('image_digest'))}</code>"))
        limits = describe_container_limits(host)
        rows.append(("Container limits", esc(", ".join(limits) or "none")))
    if host.get("cuda_device_order"):
        rows.append(("CUDA_DEVICE_ORDER", esc(host["cuda_device_order"])))
    return rows


_STAGE_NOTE = ('<p class="muted">ms/image is the end-to-end time. "Other" is ms/image minus the measured '
               "stages. — means that the stage is not measured. The stage definitions are in the protocol "
               "section. Stage providers show only a stage that ran on another provider than the pass.</p>")


def _performance_table(performance: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    records = performance.get("records") or []
    if not records:
        return '<p class="muted">No measured records.</p>'
    rows = []
    for record in records:
        timing = record.get("timing") or {}
        session = record.get("session") or {}
        stages = stage_timing(timing, execution)
        rows.append([
            esc(record.get("mode")),
            esc(record.get("timing_scope")),
            *stages.cells(),
            stages.providers_html(),
            (fmt(timing.get("batch_ms_p50"), 3), timing.get("batch_ms_p50")),
            (fmt(timing.get("batch_ms_p95"), 3), timing.get("batch_ms_p95")),
            (fmt(timing.get("cv"), 4), timing.get("cv")),
            (fmt(session.get("create_ms"), 1), session.get("create_ms")),
        ])
    table = data_table(
        ["Mode", "Timing scope", *TIMING_HEADERS, "Stage providers", "Batch p50 ms", "Batch p95 ms", "CV",
         "Session create ms"],
        rows,
        numeric=(2, 3, 4, 5, 6, 7, 9, 10, 11, 12),
    )
    return table + excluded_note(records, execution) + _STAGE_NOTE


def _end_to_end_table(performance: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    records = performance.get("records") or []
    rows = []
    for record in records:
        stages = stage_timing(record.get("timing"), execution)
        rows.append([esc(record.get("mode")), *stages.cells(), stages.providers_html()])
    table = data_table(["Mode", *TIMING_HEADERS, "Stage providers"], rows, numeric=(1, 2, 3, 4, 5, 6))
    return table + excluded_note(records, execution) + _STAGE_NOTE


def render_benchmark_html(
    report: Mapping[str, Any],
    *,
    model_href: str | None = None,
    predictions_href: str | None = None,
) -> str:
    """Render one benchmark report."""
    subject = report["subject"]
    execution = report["execution"]
    run = report["run"]
    benchmark_type = report["benchmark_type"]
    title = f"{execution['label']} {benchmark_type} — {subject['subject_id']}"
    artifact = subject.get("artifact") or {}
    parts = [
        f"<h1>{esc(title)}</h1>",
        f'<p class="subtitle">Run <code>{esc(run["run_id"])}</code> on {esc(report["host"].get("hostname"))} · '
        f"generated {esc(report.get('generated_at'))} · {pill(report['status'])}</p>",
    ]
    if report["status"] != "measured":
        parts.append(f'<div class="banner fail">{esc(report["status"])}: {esc(report.get("reason"))}</div>')
    device = execution.get("device") or {}
    parts.append(cards([
        ("Model", link(model_href, subject.get("model_report_id"))),
        ("Artifact", f"{esc(artifact.get('filename'))}<br><span class='muted'>{fmt_bytes(artifact.get('size_bytes'))}</span>"),
        ("Precision · batch", f"{esc(subject.get('precision'))} · {esc(subject.get('batch_size'))}"),
        ("Input", f"{esc((subject.get('input') or {}).get('width'))} × {esc((subject.get('input') or {}).get('height'))}"),
        ("Execution provider", f"{esc(execution['provider'].get('ep_name'))}<br><span class='muted'>{esc(execution.get('runtime'))} {esc(execution.get('runtime_version'))}</span>"),
        ("Device", f"{esc(device.get('name'))}"),
    ]))

    if report["status"] == "measured":
        accuracy = report.get("accuracy")
        if accuracy:
            values = (accuracy.get("metrics") or {}).get("values") or {}
            parts.append("<h2>Accuracy</h2>")
            parts.append(data_table(
                [key.upper().replace("_", " ") for key in COCO_STAT_KEYS],
                [[fmt(values.get(key), 4) for key in COCO_STAT_KEYS]],
                numeric=tuple(range(len(COCO_STAT_KEYS))),
            ))
            metrics = accuracy.get("metrics") or {}
            predictions = accuracy.get("predictions") or {}
            parts.append(kv_table([
                ("Scored with", f"{esc(metrics.get('tool'))} {esc(metrics.get('version'))}"),
                ("Scored by", esc((accuracy.get("method") or {}).get("scored_by"))),
                ("Decode", value_html((accuracy.get("method") or {}).get("decode"))),
                ("Shared evaluation", value_html(accuracy.get("evaluation"))),
                ("Predictions", link(predictions_href, predictions.get("filename"))
                 + f" — {fmt(predictions.get('record_count'))} detections"),
                ("Predictions file SHA-256", f"<code>{esc(predictions.get('file_sha256'))}</code>"),
                ("Predictions data SHA-256", f"<code>{esc(predictions.get('data_sha256'))}</code>"),
                ("Paired with", esc((accuracy.get("pair") or {}).get("partner_report_id"))),
            ]))
            parts.append("<h2>End-to-end timing</h2>")
            parts.append('<p class="muted">Timing of the accuracy pass, for reference only. '
                         "Performance benchmarks give the comparable speed numbers.</p>")
            parts.append(_end_to_end_table(report.get("performance") or {}, execution))
        else:
            parts.append("<h2>Performance</h2>")
            parts.append(_performance_table(report.get("performance") or {}, execution))
        unavailable = (report.get("performance") or {}).get("unavailable") or []
        if unavailable:
            parts.append("<h3>Unavailable modes</h3>")
            parts.append(data_table(["Mode", "Reason"], [[esc(item.get("mode")), esc(item.get("reason"))] for item in unavailable]))
        dataset = report.get("dataset") or {}
        protocol = report.get("protocol") or {}
        categories = dataset.get("categories")
        no_annotations = '<span class="muted">none (this pass reads no annotations)</span>'
        parts.append("<h2>Dataset and protocol</h2>")
        parts.append(f"<p>Dataset: {dataset_line(dataset)}</p>")
        parts.append(kv_table([
            ("Dataset", esc(dataset.get("name"))),
            ("Split", esc(dataset.get("split"))),
            ("Annotation file", f"<code>{esc(dataset['annotation_file'])}</code>" if dataset.get("annotation_file")
             else no_annotations),
            ("Categories", esc(", ".join(f"{item.get('id')} {item.get('name')}" for item in categories))
             if categories else no_annotations),
            ("Images used", f"{fmt(dataset.get('images_used'))} of {fmt(dataset.get('images_available'))} ({esc(dataset.get('selection'))})"),
            ("Limit", esc(dataset.get("limit")) if dataset.get("limit") else "none"),
            ("Annotation SHA-256", f"<code>{esc(dataset.get('annotation_sha256'))}</code>"),
            ("Image list SHA-256", f"<code>{esc(dataset.get('image_list_sha256'))}</code>"),
            ("Excluded stages", esc(", ".join(protocol.get("excluded_stages") or []) or "none")),
            ("Protocol", value_html({key: value for key, value in protocol.items() if key != "stage_definitions"})),
        ]))
        parts.append("<h3>Stage definitions</h3>")
        parts.append(stage_definitions_table(protocol))

    parts.append("<h2>Reproduce</h2>")
    parts.append(kv_table(reproduction_rows(run, report["environment"], execution)))
    parts.append("<h2>Host</h2>")
    parts.append(kv_table(host_rows(report["host"])))
    parts.append(json_details("Execution provider options", execution.get("provider")))
    parts.append(json_details("Full environment", report["environment"]))
    return page(title, "".join(parts), scripts=(SORT_SCRIPT,))


__all__ = ["host_rows", "render_benchmark_html", "reproduction_rows"]
