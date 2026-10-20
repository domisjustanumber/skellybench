# Report API and data structures (schema version 3)

This document uses ASD-STE100 Simplified Technical English.

The JSON payloads are the source data. HTML pages are rendered from them.
Examples abbreviate values with `…`.

## Typical producer flow

```python
from datetime import UTC, datetime
from skellybench import (
    PerformanceProtocol, accuracy_entry, accuracy_protocol, blocks, build_benchmark_report,
    build_model_report, compare_metrics, encode_columns, make_run_id, read_predictions_file,
    report_id_for, score_columns, write_benchmark_predictions, write_benchmark_report, write_model_report,
)
from skellybench.probes import docker, local
from skellybench.render import claim_run, render_family, write_run_summary

machine = local.collect(repository=".", lock_file="uv.lock")
host = blocks.build_host(machine, docker.collect())
environment = blocks.build_environment(machine, host)
run_id = make_run_id(host["hostname"], datetime.now(UTC))
run = blocks.build_run(run_id=run_id, started_at=…, command=machine["command"],
                       producer=blocks.build_producer(name="my-exporter", version="1.0", git=machine["git"],
                                                      lock_file_sha256=machine["lock_file_sha256"]))
claim_run(exports_root, {"run_id": run_id, "command": run["command"], "hostname": host["hostname"]})

# Accuracy: write predictions first, score the stored data, then write the report.
report_id = report_id_for(subject, "accuracy", execution, run_id)
reference = write_benchmark_predictions(family_dir, subject_id=subject["subject_id"], report_id=report_id,
                                        columns=encode_columns(image_ids=…, per_image=300, category_ids=…,
                                                               bboxes=…, scores=…))
scored = score_columns(read_predictions_file(family_dir / reference["filename"]), annotation_file)
report = build_benchmark_report(benchmark_type="accuracy", status="measured", run=run, host=host,
                                environment=environment, execution=execution, subject=subject, dataset=…,
                                protocol=accuracy_protocol(batch_size=1, images=5000, evaluation=…),
                                performance=…, accuracy={"evaluation": …, "metrics": scored.as_metrics_block(),
                                                         "predictions": reference, "pair": …})
write_benchmark_report(family_dir, report)

render_family(family_dir)
write_run_summary(exports_root, {"run_id": run_id, "status": "finished", …})
```

## Shared blocks

### `run`

```json
{
  "run_id": "20261020T153239Z-gerald",
  "started_at": "2026-10-20T15:32:39+00:00",
  "command": "uv run python export_all.py --sizes nano",
  "producer": {"name": "roboflow-exporter", "version": "0.1.0", "git_commit": "b34e351…",
               "git_dirty": false, "git_diff_sha256": null, "lock_file_sha256": "…"}
}
```

`run_id` is `<UTC YYYYMMDDTHHMMSSZ>-<hostname slug>`. The slug is lowercase `[a-z0-9-]`.

### `host`

```json
{
  "hostname": "Gerald",
  "os": {"name": "Windows", "version": "10.0.26300", "release": "11", "distribution": "Windows-11-…", "arch": "AMD64"},
  "virtualization": {"is_wsl": false, "is_container": false, "container_runtime": null, "is_vm": false, "hypervisor": null},
  "cpu": {"name": "Intel(R) Core(TM) i7-14700K", "short_name": "i7-14700K", "physical_cores": 20, "logical_cores": 28},
  "memory": {"total_mb": 65244},
  "gpus": [{"uuid": "GPU-…", "vendor": "nvidia", "pci_bus_id": "00000000:01:00.0", "name": "NVIDIA GeForce RTX 3070",
            "short_name": "RTX 3070", "driver": "610.88", "vram_mb": 8192}],
  "container": null,
  "cuda_device_order": null,
  "hardware_fingerprint": "sha256:…"
}
```

In a container, `container` holds `runtime`, `container_id`, `image`, `image_digest`, `base_os`,
`limits` (`cpu_quota_cores`, `cpuset`, `memory_limit_mb`), `gpu_runtime`, and `host_hostname`.

### `environment`

```json
{
  "python": {"implementation": "CPython", "version": "3.14.2", "build": "…", "compiler": "…"},
  "packages": {"numpy": "2.5.4", "onnxruntime-gpu": "1.31.0", "…": "…"},
  "package_sources": {"skellybench": {"vcs": "git", "url": "https://github.com/…", "commit": "7eb6bde…"}},
  "environment_fingerprint": "sha256:…"
}
```

`packages` holds every installed distribution. `package_sources` holds only packages that were not
installed from an index.

### `execution`

```json
{
  "runtime": "onnxruntime", "runtime_version": "1.31.0",
  "provider": {"key": "cuda", "slug": "cuda", "label": "CUDA", "ep_name": "CUDAExecutionProvider", "options": {…}},
  "device": {"kind": "gpu", "name": "NVIDIA GeForce RTX 3070", "short_name": "RTX 3070",
             "host_gpu_uuid": "GPU-…", "ep_device_id": 0},
  "drivers": {"nvidia_driver": "610.88", "loaded_libraries": {"cudnn64_9.dll": "9.24.0"}},
  "label": "CUDA (RTX 3070)"
}
```

`label` is `"<provider label> (<device short name>)"`. Charts use it as the category.

## Benchmark report

```json
{
  "schema_version": 3, "kind": "benchmark_report",
  "benchmark_type": "accuracy", "status": "measured",
  "report_id": "rfdetr-nano_b1_fp32.accuracy.cuda.20261020T153239Z-gerald",
  "generated_at": "…",
  "run": {…}, "host": {…}, "environment": {…}, "execution": {…},
  "subject": {"subject_id": "rfdetr-nano_b1_fp32", "model_report_id": "rfdetr-nano_b1_fp32", "model_id": "rfdetr-nano",
              "format": "onnx", "precision": "fp32", "batch_size": 1, "batch_agnostic": false,
              "artifact": {"filename": "rfdetr-nano_b1_fp32.onnx", "sha256": "…", "size_bytes": 107824324},
              "input": {"width": 384, "height": 384}},
  "dataset": {"name": "coco-val2017", "annotation_sha256": "…", "image_list_sha256": "…",
              "images_available": 5000, "images_used": 5000, "selection": "all", "limit": null},
  "protocol": {"benchmark_type": "accuracy", "timing_scope": "end_to_end", "protocol_id": "…", …},
  "performance": {"records": [{"mode": "host", "timing_scope": "end_to_end", "timing": {"images_per_second": …}}]},
  "accuracy": {
    "evaluation": {…},
    "metrics": {"tool": "faster-coco-eval", "version": "1.8.0", "values": {"ap": …, "ap50": …, …}},
    "predictions": {"filename": "benchmarks/…/….predictions.json", "file_sha256": "…", "data_sha256": "…",
                    "format": "coco_detections_columnar_v1", "record_count": 1500000},
    "pair": {"run_id": "…", "partner_report_id": "…"}
  }
}
```

A performance report has `benchmark_type: "performance"`, no `accuracy` block, and one record per
mode (`device` or `host`). A report whose `status` is `unavailable` or `error` has a `reason`.

## Predictions file

```json
{
  "schema_version": 3, "kind": "benchmark_predictions", "report_id": "…",
  "format": "coco_detections_columnar_v1",
  "image_ids": [139, 285, …], "per_image": 300, "counts": null, "record_count": 1500000,
  "columns": [{"name": "category_id", "dtype": "uint16"}, {"name": "bbox_xywh", "dtype": "float32", "width": 4},
              {"name": "score", "dtype": "float32"}],
  "byte_order": "little", "layout": "column-major", "encoding": "gzip+base64",
  "data_sha256": "…", "data": "H4sI…"
}
```

Set `per_image` when every image has the same count. Else set `counts`, one count per image.

## Model report

```json
{
  "schema_version": 3, "kind": "model_report",
  "report_id": "rfdetr-nano_b1_fp16", "created_run_id": "20261020T153239Z-gerald", "generated_at": "…",
  "model": {"model_id": "rfdetr-nano", "display_name": "RF-DETR Nano", "task": "object-detection",
            "format": "onnx", "precision": "fp16", "batch_size": 1,
            "variant": {"family": "rfdetr", "size": "nano", "size_rank": 0, "precision": "fp16", "batch_size": 1},
            "input": {"width": 384, "height": 384}, "outputs": […],
            "artifact": {"filename": "rfdetr-nano_b1_fp16.onnx", "sha256": "…", "size_bytes": …}, "decode": {…}},
  "creation": {
    "run": {…}, "host": {…}, "environment": {…},
    "source": {"kind": "checkpoint", "filename": "rf-detr-nano.pth", "url": "https://…", "md5": "…", "sha256": "…"},
    "parent": {"report_id": "rfdetr-nano_b1_fp32", "format": "onnx", "filename": "rfdetr-nano_b1_fp32.onnx", "sha256": "…"},
    "config_sidecar": null,
    "steps": [{"step": "fp16_convert", "tool": "onnxconverter-common", "version": "1.16.0",
               "params": {…}, "repairs": {…}, "input_sha256": "…", "output_sha256": "…"}]
  },
  "accuracy": {
    "self": {"link": {"path": "benchmarks/…json", "sha256": "…"}, "snapshot": {…full benchmark report…}},
    "parent": {"link": {…}, "snapshot": {…}},
    "comparison": {"status": "ok", "deltas": {"delta_ap": …, …}, "thresholds": {"warn_ap": 0.001, "fail_ap": 0.01}}
  }
}
```

`model.variant` lets the family overview group models of any family.

## API

| Function | Module | Purpose |
|---|---|---|
| `local.collect()`, `docker.collect()` | `probes` | Probe the machine and the container |
| `build_run`, `build_producer`, `build_host`, `build_environment`, `build_execution` | `blocks` | Build the shared blocks |
| `make_run_id`, `benchmark_report_id`, `ep_slug` | `paths` | Names |
| `PerformanceProtocol`, `accuracy_protocol` | `protocol` | Protocol blocks with stable IDs |
| `encode_columns`, `columns_from_records`, `read_predictions_file` | `predictions` | Predictions files |
| `score_columns`, `score_predictions`, `compare_metrics` | `scoring` | COCO metrics and comparisons |
| `build_benchmark_report`, `write_benchmark_predictions`, `write_benchmark_report` | `reports` | Benchmark reports |
| `accuracy_entry`, `build_model_report`, `write_model_report`, `supersede_model_report` | `reports` | Model reports |
| `scan_family`, `read_model_report` | `reports` | Read the reports on disk |
| `claim_run`, `write_run_summary`, `render_family`, `rebuild_all` | `render` | HTML pages |
