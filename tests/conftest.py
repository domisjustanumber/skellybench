"""Shared factories for schema-3 report tests. No network, no model weights, no COCO data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from skellybench import blocks
from skellybench.predictions import encode_columns
from skellybench.protocol import PerformanceProtocol, accuracy_protocol
from skellybench.reports import (
    accuracy_entry,
    build_benchmark_report,
    build_model_report,
    write_benchmark_predictions,
    write_benchmark_report,
    write_model_report,
)
from skellybench.scoring import compare_metrics
from skellybench.schema import COCO_STAT_KEYS

RUN_ID = "20261020T153239Z-gerald"
GPU_UUID = "GPU-23465ec5-1fad-462f-0ca0-f27b67213a4b"
EVALUATION = {"iou_type": "bbox", "max_dets": 100, "category_ids": [1]}
METHOD = {"decode": {"score_threshold": 0.0, "num_select": 300},
          "scorer": {"tool": "faster-coco-eval", "version": "1.8.0"}, "scored_by": "skellybench"}
_HOST_STAGES = {
    "preprocess": "Normalization and NCHW stacking of the letterboxed batch.",
    "inference": "session.run only. It holds the internal input and output copies of ONNX Runtime.",
    "postprocess": "The sidecar decode: NMS, unprojection and clamping.",
    "letterbox": "Excluded. Timed separately.",
}
STAGE_DEFINITIONS = {
    "host": _HOST_STAGES,
    "device": {
        **_HOST_STAGES,
        "preprocess": "Normalization and NCHW stacking of the letterboxed batch, the IO binding set-up and "
                      "the host-to-device copy of the input.",
        "inference": "The session run only.",
        "postprocess": "The device-to-host copy of the outputs, then the sidecar decode.",
    },
}
NULL_STAGES = {"preprocess": None, "inference": None, "postprocess": None}


def stage_timing(ips: float, *, letterbox: bool = False, decode_provider: str | None = None) -> dict[str, Any]:
    """Return a performance ``timing`` block whose stages leave about 5 % as "other"."""
    ms = 1000.0 / ips
    timing: dict[str, Any] = {
        "images_per_second": ips,
        "ms_per_image": ms,
        "stages_ms_per_image": {"preprocess": 0.45 * ms, "inference": 0.4 * ms, "postprocess": 0.1 * ms},
        "batch_ms_p50": ms,
        "batch_ms_p95": ms * 1.1,
    }
    if decode_provider is not None:
        timing["stage_providers"] = {"preprocess": None, "inference": None, "postprocess": decode_provider}
    if letterbox:
        timing["excluded_ms_per_image"] = {"letterbox": 0.5}
    return timing


def performance_protocol(**overrides: Any) -> PerformanceProtocol:
    return PerformanceProtocol(**{"excluded_stages": ("letterbox",), "stage_definitions": STAGE_DEFINITIONS, **overrides})


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def local_probe(hostname: str = "Gerald") -> dict[str, Any]:
    return {
        "hostname": hostname,
        "os": {"name": "Windows", "version": "10.0.26300", "distribution": "Windows-11", "arch": "AMD64"},
        "cpu": {"name": "Intel(R) Core(TM) i7-14700K", "physical_cores": 20, "logical_cores": 28},
        "memory_total_mb": 65244,
        "gpus": [
            {"uuid": GPU_UUID, "vendor": "nvidia", "pci_bus_id": "00000000:01:00.0",
             "name": "NVIDIA GeForce RTX 3070", "driver": "610.88", "vram_mb": 8192},
        ],
        "virtualization": {"is_wsl": False, "is_vm": False, "hypervisor": None},
        "python": {"implementation": "CPython", "version": "3.14.2", "build": "main"},
        "packages": {"numpy": "2.5.4", "onnxruntime-gpu": "1.31.0"},
        "package_sources": {},
        "git": {"commit": "b34e351", "dirty": False, "diff_sha256": None},
        "lock_file_sha256": sha("lock"),
        "command": "uv run python export_all.py --sizes nano",
        "cuda_device_order": None,
    }


def context(run_id: str = RUN_ID, hostname: str = "Gerald") -> dict[str, Any]:
    local = local_probe(hostname)
    host = blocks.build_host(local, None)
    environment = blocks.build_environment(local, host)
    run = blocks.build_run(
        run_id=run_id,
        started_at="2026-10-20T15:32:39+00:00",
        command=local["command"],
        producer=blocks.build_producer(name="roboflow-exporter", version="0.1.0", git=local["git"],
                                       lock_file_sha256=local["lock_file_sha256"]),
    )
    return {"run": run, "host": host, "environment": environment}


def execution(host: dict[str, Any], provider: str = "cuda") -> dict[str, Any]:
    if provider == "cpu":
        return blocks.build_execution(host=host, runtime="onnxruntime", runtime_version="1.31.0",
                                      provider_key="cpu", provider_slug="cpu", provider_label="CPU",
                                      ep_name="CPUExecutionProvider", device_kind="cpu")
    if provider == "torch-cuda":
        return blocks.build_execution(host=host, runtime="pytorch", runtime_version="2.14.1",
                                      provider_key="torch_cuda", provider_slug="torch-cuda", provider_label="PyTorch CUDA",
                                      ep_name="torch", device_kind="gpu", gpu_uuid=GPU_UUID, ep_device_id=0)
    return blocks.build_execution(host=host, runtime="onnxruntime", runtime_version="1.31.0",
                                  provider_key="cuda", provider_slug="cuda", provider_label="CUDA",
                                  ep_name="CUDAExecutionProvider", device_kind="gpu", gpu_uuid=GPU_UUID,
                                  ep_device_id=0, loaded_libraries={"cudnn64_9.dll": "9.10.0"})


def subject(subject_id: str, artifact_sha: str, *, batch_agnostic: bool = False, precision: str = "fp32",
            fmt: str = "onnx") -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "model_report_id": subject_id,
        "model_id": "rfdetr-nano",
        "format": fmt,
        "precision": precision,
        "batch_size": 1,
        "batch_agnostic": batch_agnostic,
        "artifact": {"filename": f"{subject_id}.onnx", "sha256": artifact_sha, "size_bytes": 1000},
        "input": {"width": 384, "height": 384},
    }


def dataset() -> dict[str, Any]:
    return {"name": "coco-val2017", "annotation_sha256": sha("annotations"), "image_list_sha256": sha("images"),
            "images_available": 2, "images_used": 2, "selection": "all", "limit": None}


def metrics(ap: float) -> dict[str, Any]:
    return {"tool": "faster-coco-eval", "version": "1.8.0", "values": {key: ap for key in COCO_STAT_KEYS}}


def write_accuracy(family_dir: Path, ctx: dict[str, Any], subj: dict[str, Any], exe: dict[str, Any], ap: float,
                   partner: str | None = None, method: dict[str, Any] | None = None) -> Path:
    from skellybench.reports import report_id_for

    report_id = report_id_for(subj, "accuracy", exe, ctx["run"]["run_id"])
    columns = encode_columns(image_ids=[1, 2], per_image=2, category_ids=[1, 2, 3, 4],
                             bboxes=[[0, 0, 1, 1]] * 4, scores=[0.9, 0.8, 0.7, 0.6])
    reference = write_benchmark_predictions(family_dir, subject_id=subj["subject_id"], report_id=report_id, columns=columns)
    report = build_benchmark_report(
        benchmark_type="accuracy", status="measured", execution=exe, subject=subj, dataset=dataset(),
        protocol=accuracy_protocol(batch_size=1, images=2, evaluation=EVALUATION),
        performance={"records": [{"mode": "host", "timing_scope": "end_to_end",
                                  "timing": {"images_per_second": 40.0, "ms_per_image": 25.0,
                                             "stages_ms_per_image": dict(NULL_STAGES)}}]},
        accuracy={"evaluation": EVALUATION, "method": method or METHOD, "metrics": metrics(ap), "predictions": reference,
                  "pair": {"run_id": ctx["run"]["run_id"], "partner_report_id": partner or report_id}},
        **ctx,
    )
    return write_benchmark_report(family_dir, report).json


def write_performance(family_dir: Path, ctx: dict[str, Any], subj: dict[str, Any], exe: dict[str, Any],
                      ips: float = 100.0, protocol: PerformanceProtocol | None = None,
                      decode_provider: str | None = None) -> Path:
    protocol = protocol or performance_protocol()
    letterbox = "letterbox" in protocol.excluded_stages
    report = build_benchmark_report(
        benchmark_type="performance", status="measured", execution=exe, subject=subj, dataset=dataset(),
        protocol=protocol.block(batch_size=1),
        performance={"records": [
            {"mode": "device", "timing_scope": "session",
             "timing": stage_timing(ips, letterbox=letterbox, decode_provider=decode_provider)},
            {"mode": "host", "timing_scope": "session",
             "timing": stage_timing(ips * 0.8, letterbox=letterbox, decode_provider=decode_provider)},
        ]},
        **ctx,
    )
    return write_benchmark_report(family_dir, report).json


def source_model(ctx: dict[str, Any], accuracy_json: Path, family_dir: Path, checkpoint_sha: str) -> dict[str, Any]:
    return build_model_report(
        report_id="rfdetr-nano_source",
        created_run_id=ctx["run"]["run_id"],
        model={"model_id": "rfdetr-nano", "display_name": "RF-DETR Nano", "format": "pytorch_checkpoint",
               "precision": "fp32", "batch_size": None,
               "variant": {"family": "rfdetr", "size": "nano", "size_rank": 0, "precision": "fp32", "batch_size": None},
               "input": {"width": 384, "height": 384},
               "artifact": {"filename": "rf-detr-nano.pth", "sha256": checkpoint_sha, "size_bytes": 10}},
        creation={**ctx, "source": {"kind": "checkpoint", "filename": "rf-detr-nano.pth", "url": "https://example.com/nano.pth",
                                    "md5": "0" * 32, "sha256": checkpoint_sha},
                  "parent": None, "config_sidecar": None,
                  "steps": [{"step": "download", "tool": "rfdetr", "version": "1.11.2", "output_sha256": checkpoint_sha}]},
        accuracy_self=accuracy_entry(family_dir, accuracy_json),
    )


def onnx_model(ctx: dict[str, Any], report_id: str, artifact_sha: str, parent_id: str, parent_sha: str,
               self_json: Path, parent_json: Path, family_dir: Path, precision: str = "fp32",
               self_ap: float = 0.5, parent_ap: float = 0.5, parent_run_id: str | None = None,
               **extra: Any) -> dict[str, Any]:
    return build_model_report(
        report_id=report_id,
        created_run_id=ctx["run"]["run_id"],
        model={"model_id": "rfdetr-nano", "display_name": "RF-DETR Nano", "format": "onnx", "precision": precision,
               "batch_size": 1,
               "variant": {"family": "rfdetr", "size": "nano", "size_rank": 0, "precision": precision, "batch_size": 1},
               "input": {"width": 384, "height": 384},
               "artifact": {"filename": f"{report_id}.onnx", "sha256": artifact_sha, "size_bytes": 10}},
        creation={**ctx, "source": {"kind": "checkpoint", "filename": "rf-detr-nano.pth", "url": "https://example.com/nano.pth",
                                    "md5": "0" * 32, "sha256": sha("checkpoint")},
                  "parent": {"report_id": parent_id, "format": "onnx", "filename": f"{parent_id}.onnx", "sha256": parent_sha},
                  "config_sidecar": None,
                  "steps": [{"step": "convert", "tool": "tool", "version": "1", "input_sha256": parent_sha,
                             "output_sha256": artifact_sha}]},
        accuracy_self=accuracy_entry(family_dir, self_json),
        accuracy_parent=accuracy_entry(family_dir, parent_json),
        comparison=compare_metrics(metrics(parent_ap)["values"], metrics(self_ap)["values"]),
        parent_run_id=parent_run_id,
        **extra,
    )


@pytest.fixture
def family(tmp_path: Path) -> Path:
    directory = tmp_path / "exports" / "detectors" / "object_detectors" / "rfdetr"
    directory.mkdir(parents=True)
    return directory
