"""Cross-machine benchmark index rebuilt from report files or model-report snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .._internal.io import read_json
from .._internal.timestamps import utc_now, utc_stamp

from ..file_writer import write_html_file, write_json_file
from ..html_reports.index import render_index_html
from ..protocol import comparison_key
from ..schema import (
    BENCHMARK_OVERVIEW_KIND,
    BENCHMARK_REPORT_KIND,
    SKELLYBENCH_SCHEMA_VERSION,
    validate_benchmark_overview,
    validate_benchmark_report,
)

_MODEL_REPORT_KIND = "model_report"


def index_paths(output_dir: str | Path, generated_at: datetime | None = None) -> tuple[Path, Path]:
    """Return ``(json_path, html_path)`` for the cross-machine index."""
    root = Path(output_dir)
    stem = f"benchmark-overview-{utc_stamp(generated_at)}"
    return root / f"{stem}.json", root / f"{stem}.html"


def _report_from_snapshot(
    entry: dict[str, Any],
    *,
    model_report: dict[str, Any],
) -> dict[str, Any] | None:
    snapshot = entry.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("accuracy") is not None:
        return None
    if "run" not in snapshot or "model" not in snapshot or "performance" not in snapshot:
        return None
    payload = dict(snapshot)
    payload.setdefault("schema_version", SKELLYBENCH_SCHEMA_VERSION)
    payload["kind"] = BENCHMARK_REPORT_KIND
    report_id = entry.get("report_id") or payload.get("report_id") or "unknown"
    payload["report_id"] = report_id
    payload["_report_filename"] = entry.get("filename") or f"{report_id}.json"
    payload.setdefault("generated_at", model_report.get("generated_at"))
    if "model_report" not in payload:
        payload["model_report"] = {
            "report_id": model_report.get("report_id"),
            "filename": entry.get("filename"),
        }
    return payload


def _collect_reports(root: Path) -> dict[str, tuple[dict[str, Any], Path | None]]:
    reports: dict[str, tuple[dict[str, Any], Path | None]] = {}
    if not root.is_dir():
        return reports
    for path in sorted(root.rglob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        kind = payload.get("kind")
        if kind == BENCHMARK_REPORT_KIND:
            payload["_report_filename"] = path.name
            try:
                validate_benchmark_report(payload)
            except Exception:
                continue
            reports[str(payload["report_id"])] = (payload, path)
            continue
        if kind != _MODEL_REPORT_KIND:
            continue
        scoring = payload.get("scoring") or {}
        performance = scoring.get("performance") or {}
        for entry in performance.get("benchmarks") or []:
            if not isinstance(entry, dict):
                continue
            report = _report_from_snapshot(entry, model_report=payload)
            if report is None:
                continue
            try:
                validate_benchmark_report(report)
            except Exception:
                continue
            report_id = str(report["report_id"])
            # A real benchmark file wins; a snapshot is only a rebuild source.
            if report_id not in reports or reports[report_id][1] is None:
                reports[report_id] = (report, None)
    return reports


def _row_from_record(
    report: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    run = report.get("run") or {}
    machine = run.get("hardware") or run.get("machine") or {}
    os_info = machine.get("os") or {}
    cpu = machine.get("cpu") or {}
    memory = machine.get("memory") or {}
    model = report.get("model") or {}
    performance = report.get("performance") or {}
    protocol = performance.get("protocol") or {}
    record_device = record.get("device") or {}
    provider = str(record.get("provider"))
    mode = str(record.get("mode"))
    row_id = f"{report['report_id']}#{record.get('record_id') or f'{provider}/{mode}'}"
    record_hardware_id = str(record.get("hardware_id") or machine.get("hardware_id") or "")
    record_environment_id = str(record.get("environment_id") or "")
    key = record.get("comparison_key")
    if not key:
        key = comparison_key(
            artifact_sha256=str((model.get("artifact") or {}).get("sha256") or ""),
            hardware_id=record_hardware_id,
            environment_id=record_environment_id,
            protocol_id=str(protocol.get("protocol_id") or ""),
            provider=provider,
            mode=mode,
        )
    timing = record.get("timing") or {}
    return {
        "row_id": row_id,
        "report": str(report.get("_report_filename") or f"{report['report_id']}.json"),
        "run_id": run.get("run_id"),
        "model_id": model.get("id"),
        "task": model.get("task"),
        "precision": model.get("precision"),
        "input_signature_id": model.get("input_signature_id"),
        "artifact_sha256": (model.get("artifact") or {}).get("sha256"),
        "scenario": performance.get("scenario"),
        "provider": provider,
        "provider_label": record.get("provider_label"),
        "ep_name": record.get("ep_name"),
        "ep_package": (record.get("ep_package") or {}).get("name"),
        "mode": mode,
        "gpu_id": record_device.get("id"),
        "gpu_name": record_device.get("name"),
        "gpu_vendor": record_device.get("vendor"),
        "gpu_driver": record_device.get("driver"),
        "gpu_vram_mb": record_device.get("vram_mb"),
        "os_family": os_info.get("family"),
        "os_name": os_info.get("name"),
        "os_version": os_info.get("version"),
        "os_arch": os_info.get("arch"),
        "cpu_vendor": cpu.get("vendor"),
        "cpu_model": cpu.get("model"),
        "cpu_physical_cores": cpu.get("physical_cores"),
        "cpu_logical_cores": cpu.get("logical_cores"),
        "ram_mb": memory.get("total_mb"),
        "machine_id": machine.get("id"),
        "machine_label": machine.get("label"),
        "hostname": (run.get("runner") or {}).get("hostname"),
        "runner_user": (run.get("runner") or {}).get("user"),
        "runner_label": (run.get("runner") or {}).get("label"),
        "protocol_id": protocol.get("protocol_id"),
        "hardware_id": record_hardware_id,
        "environment_id": record_environment_id,
        "comparison_key": key,
        "images_per_second": timing.get("images_per_second"),
        "ms_per_image": timing.get("ms_per_image"),
        "batch_ms_p50": timing.get("batch_ms_p50"),
        "batch_ms_p95": timing.get("batch_ms_p95"),
        "cv": timing.get("cv"),
        "stale": bool(record.get("stale")),
    }


def build_index(
    output_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a flat cross-machine index from benchmark reports and model-report snapshots."""
    reports = _collect_reports(Path(output_dir))
    rows: list[dict[str, Any]] = []
    unavailable: dict[str, dict[str, Any]] = {}
    for report, _path in reports.values():
        performance = report.get("performance") or {}
        for record in performance.get("records") or []:
            if record.get("status") != "measured":
                continue
            rows.append(_row_from_record(report, record))
        for item in performance.get("unavailable") or []:
            unavailable[str(item.get("provider"))] = dict(item)

    rows.sort(
        key=lambda row: (
            str(row.get("model_id") or ""),
            str(row.get("provider") or ""),
            str(row.get("mode") or ""),
            str(row.get("machine_label") or ""),
        )
    )
    dimensions = {
        "providers": sorted({str(row.get("provider")) for row in rows if row.get("provider")}),
        "modes": sorted({str(row.get("mode")) for row in rows if row.get("mode")}),
        "scenarios": sorted({str(row.get("scenario")) for row in rows if row.get("scenario")}),
        "models": sorted({str(row.get("model_id")) for row in rows if row.get("model_id")}),
        "machines": sorted(
            {str(row.get("machine_label") or row.get("machine_id")) for row in rows if row.get("machine_label") or row.get("machine_id")}
        ),
        "gpus": sorted({str(row.get("gpu_name")) for row in rows if row.get("gpu_name")}),
    }
    payload = {
        "schema_version": SKELLYBENCH_SCHEMA_VERSION,
        "kind": BENCHMARK_OVERVIEW_KIND,
        "generated_at": (generated_at or utc_now()).isoformat(),
        "row_count": len(rows),
        "dimensions": dimensions,
        "rows": rows,
        "unavailable": [unavailable[key] for key in sorted(unavailable)],
    }
    validate_benchmark_overview(payload)
    return payload


def write_index(
    output_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Rebuild and write the cross-machine index."""
    payload = build_index(output_dir, generated_at=generated_at)
    json_path, html_path = index_paths(output_dir, generated_at)
    write_json_file(json_path, payload)
    write_html_file(html_path, render_index_html(payload))
    return json_path, html_path


__all__ = ["build_index", "index_paths", "write_index"]
