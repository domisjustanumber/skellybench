# Report API and data structures

This document describes the public APIs and JSON payloads for model reports, benchmark reports,
and per-run summaries. The JSON payloads are the source data; each writer also renders a matching
HTML file.

All builders return schema-version-2 dictionaries and validate them before returning. Schema
version 1 reports remain readable for compatibility. The structures below describe current
schema-version-2 output.

Examples are structural and may abbreviate nested values; `{}` and placeholder strings stand for
fully populated objects where the related contract requires fields.

## Shared data structures

### Run

Benchmark reports and run summaries receive the run metadata object:

```json
{
  "run_id": "run-20261020-101500",
  "started_at": "2026-10-20T10:15:00Z",
  "finished_at": "2026-10-20T10:16:00Z",
  "command": {"argv": ["python", "benchmark.py"], "display": "python benchmark.py"},
  "runner": {"user": "operator"},
  "hardware": {
    "id": "sha256:<64 lowercase hex characters>",
    "hardware_id": "sha256:<64 lowercase hex characters>",
    "os": {"family": "windows", "name": "Windows", "version": "11", "arch": "AMD64"},
    "cpu": {"model": "CPU model", "arch": "AMD64"},
    "gpus": []
  },
  "libraries": {
    "python": {"implementation": "CPython", "version": "3.x"},
    "packages": {"onnxruntime": "1.x"}
  }
}
```

`run_id`, `runner.user`, hardware identity/OS/CPU, the `gpus` array (which may be empty), and
library identity are required. Hardware can also include `memory: {"total_mb": 32768}` and GPU
entries with `index`, `id`, `name`, `vendor`, `driver`, and optional `vram_mb`. `command`,
timestamps, and additional runner fields are provenance metadata.

CUDA-enabled exporter reports can also include
`libraries.cuda.torch_cuda_build` and `libraries.cuda.cudnn_runtime_version`. The values record the
CUDA build and cuDNN runtime versions reported by PyTorch. For example, the values can be `"13.0"`
and `"9.24.0"`. Model, benchmark, and run-summary HTML reports show these values when present.

### Model metadata

Benchmark reports use this model metadata shape:

```json
{
  "id": "detector-small",
  "task": "object-detection",
  "format": "onnx",
  "precision": "fp32",
  "artifact": {
    "filename": "detector-small.onnx",
    "sha256": "<64 lowercase hex characters>",
    "bytes": 123456
  },
  "input_signature": {"batch_size": 1, "shapes": [[1, 3, 640, 640]]},
  "input_signature_id": "sha256:<64 lowercase hex characters>"
}
```

The `id`, `task`, `format`, `precision`, artifact filename and SHA-256, `input_signature`, and
`input_signature_id` are required for benchmark report model metadata. `bytes`, display fields,
and `metadata` may be included. Model reports use a related but distinct shape, documented below.
Filenames in reports are basenames, not local paths.

### Performance

Both report families use a throughput performance object:

```json
{
  "scenario": "throughput_adaptive",
  "protocol": {
    "version": 1,
    "images": 120,
    "warmup_passes": 1,
    "passes": 5,
    "min_pass_seconds": 1.0,
    "max_pass_seconds": 30.0,
    "cpu_threads": 0,
    "timing_scope": "session",
    "protocol_id": "stable-protocol-id"
  },
  "records": [{
    "record_id": "cuda/host/GPU-0",
    "status": "measured",
    "provider": "cuda",
    "mode": "host",
    "timing": {"images_per_second": 200.0, "ms_per_image": 5.0}
  }],
  "unavailable": []
}
```

`scenario` is `throughput_adaptive` or `accuracy_full_dataset`. Protocol values are validated;
optional `input_signature_id` and other protocol metadata may also be present. Each record requires
a unique `record_id`, `status`, provider (`cpu`, `openvino_gpu`, `cuda`, or `tensorrt_rtx`), and
mode (`device` or `host`). A measured record requires finite `timing.images_per_second` and
`timing.ms_per_image`; additional finite timing values are allowed. A non-measured record instead
requires `reason`. `unavailable` entries require `provider` and `reason`. At least one record or
unavailable entry is required. Performance report data is throughput-only; accuracy belongs in
model reports.

## Model report

### Build input

```python
build_model_report(
    *,
    report_id: str,
    model: Mapping[str, Any],
    provenance: Mapping[str, Any],
    accuracy: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    tracked_packages: Iterable[str] | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]
```

The required `model` fields are:

| Field | Structure |
|---|---|
| `model_id` | Non-empty model identifier |
| `runtime` | `onnx`, `rfdetr`, or `other` |
| `precision` | Non-empty precision label |
| `batch_size` | Positive integer |
| `input_signature` | Object describing model inputs |

Optional `model.artifact` can contain `filename`, `sha256`, `source_fp32_sha256`, `size_bytes`, and
`filesize_human`. Optional `metadata` is an opaque object. `provenance` requires `origin`
(`python_model`, `onnx_model`, or `other`) and may include `tools`, `config`, `environment`, and
`source`. A checkpoint `source` uses its upstream `url` and lowercase MD5 `md5`; do not include a
local checkpoint path.

`accuracy` and `performance` are independently optional, but at least one scoring section must be
present in the resulting payload. If omitted, `performance` defaults to `{"benchmarks": []}`.
`environment` defaults to detected hardware and library versions; when supplied, its current
shape is:

```json
{
  "hardware": {"id": "...", "hardware_id": "...", "os": {}, "cpu": {}, "gpus": []},
  "libraries": {
    "python": {"implementation": "CPython", "version": "3.x"},
    "packages": {},
    "cuda": {"torch_cuda_build": "13.0", "cudnn_runtime_version": "9.24.0"}
  }
}
```

`libraries.cuda` is optional. It records the CUDA build and cuDNN runtime versions when the
exporter detects them.

`tracked_packages` applies when the environment is detected by the builder. `generated_at` accepts
an ISO-8601 string or `datetime`; when omitted, the current timestamp is used.

For measured accuracy, `accuracy` has `scenario: "accuracy_full_dataset"`, a valid `device`,
evaluation settings, exactly two passes (`python` and `onnx`), and a Python-vs-ONNX comparison.
Both passes must use the selected device; the ONNX pass must match the selected ONNX Runtime
provider. Each measured pass contains `metrics` or `predictions`. Metrics are `{tool, version,
values}`. Prediction data is represented by a store descriptor: inline stores contain COCO
detection `records`; JSON/JSONL stores contain a SHA-256 and record count. Report output does not
include local prediction paths or raw prediction records. Non-measured accuracy statuses
(`unavailable`, `skipped`, or `error`) require a `reason`.

### JSON output

```json
{
  "schema_version": 2,
  "kind": "model_report",
  "report_id": "detector-small-fp32",
  "generated_at": "2026-10-20T10:15:00Z",
  "model": {
    "model_id": "detector-small",
    "runtime": "onnx",
    "precision": "fp32",
    "batch_size": 1,
    "input_signature": {"shapes": [[1, 3, 640, 640]]},
    "artifact": {"filename": "detector-small.onnx", "sha256": "<sha256>"}
  },
  "provenance": {"origin": "onnx_model"},
  "scoring": {
    "accuracy": {"scenario": "accuracy_full_dataset", "status": "unavailable", "reason": "..."},
    "performance": {
      "benchmarks": [{
        "run_id": "run-20261020-101500",
        "scenario": "throughput_adaptive",
        "report_id": "detector-small.onnx.benchmark-run-20261020-101500",
        "filename": "detector-small.onnx.benchmark-run-20261020-101500.json",
        "sha256": "<sha256>",
        "snapshot": {"kind": "benchmark_report", "performance": {"records": []}}
      }]
    }
  },
  "environment": {
    "hardware": {"id": "...", "hardware_id": "...", "os": {}, "cpu": {}, "gpus": []},
    "libraries": {"python": {"implementation": "CPython", "version": "3.x"}, "packages": {}}
  }
}
```

The `accuracy` section may be absent; `scoring` must contain at least one of `accuracy` or
`performance`. Each performance benchmark entry requires `run_id`, `scenario`, and throughput-only
`snapshot`; report id, filename, and hash are optional. Snapshots must not contain accuracy or raw
prediction data. reports are written as a JSON/HTML pair next to the model artifact, using
`<artifact-basename>.model.report.json` and `.model.report.html`; the `model_reports` API also
exposes guarded write and re-render operations.

## Benchmark report

### Build input

```python
build_benchmark_report(
    *,
    run: Mapping[str, Any],
    model: Mapping[str, Any],
    performance: Mapping[str, Any],
    report_id: str | None = None,
    generated_at: datetime | str | None = None,
    model_report: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]
```

`run`, `model`, and `performance` use the shared structures above. `model.artifact.filename` is
required because it is used to derive the default report id. `report_id` overrides that default.
`model_report`, when provided, links a model report by `report_id` and may include its JSON
`filename` and SHA-256. `provenance`, when provided, is an object. `generated_at` accepts an ISO
timestamp string or `datetime`.

### Example from a real run

This example uses accuracy run `20261020T014403Z-c37089e4-accuracy` and throughput reports from
the same run. It ran RF-DETR Nano with batch size 1 and both fp32 and fp16 models. The run used
CUDA device 0 on an NVIDIA GeForce RTX 3070. It evaluated all 5,000 COCO `val2017` images. The run
used `rfdetr` 1.11.2 and `faster-coco-eval` 1.8.0.

The exporter records the CPU model as `Intel(R) Core(TM) i7-14700K`. It records the PyTorch CUDA
build as `13.0` and the cuDNN runtime as `9.24.0`.

The input below shows fields from the actual `model` argument for each report. The builder receives
one model and its own `performance` object per call. The run metadata and performance arguments are
not shown in full. The example omits hardware fingerprints.

```json
[
  {
    "model": {
      "id": "rfdetr-nano",
      "task": "object-detection",
      "format": "onnx",
      "precision": "fp32",
      "artifact": {
        "filename": "rfdetr-nano_b1_fp32.onnx",
        "sha256": "68d00b8d1583b02daa385dc91c681c9659760afab7c385a46381725ba6412c6a",
        "bytes": 107824324
      },
      "input_signature": {
        "batch_size": 1,
        "shapes": [[1, 3, 384, 384]],
        "dtype": "float32",
        "layout": "NCHW",
        "preprocessing_id": "imagenet_rgb_normalized"
      },
      "input_signature_id": "sha256:81772153f578ac742b505a09610aaffc81b43ae3717229ee3dc3d0f6a14dd972"
    }
  },
  {
    "model": {
      "id": "rfdetr-nano",
      "task": "object-detection",
      "format": "onnx",
      "precision": "fp16",
      "artifact": {
        "filename": "rfdetr-nano_b1_fp16.onnx",
        "sha256": "f21f896b2a0b375da4e6a1bdc2a051bb019b17fb342cb70c83e24f2be22a508e",
        "bytes": 54223621
      },
      "input_signature": {
        "batch_size": 1,
        "shapes": [[1, 3, 384, 384]],
        "dtype": "float32",
        "layout": "NCHW",
        "preprocessing_id": "imagenet_rgb_normalized"
      },
      "input_signature_id": "sha256:81772153f578ac742b505a09610aaffc81b43ae3717229ee3dc3d0f6a14dd972"
    }
  }
]
```

The output below shows measured values from the two reports. The input uses float32 images for both
models. The `fp16` label describes the model weights, not the input tensor.

| Precision | CUDA device throughput | CUDA device latency | Python AP | ONNX AP |
|---|---:|---:|---:|---:|
| fp32 | 129.537 images/s | 7.7198 ms/image | 0.480397 | 0.480287 |
| fp16 | 128.532 images/s | 7.7802 ms/image | 0.480397 | 0.480230 |

The `run` input used run id `20261020T014403Z-c37089e4-throughput`. It started at
`2026-10-20T01:44:03.632775+00:00`. The accuracy run ended at `2026-10-20T02:04:42.150587+00:00`.
The throughput `performance` inputs used these measured CUDA device records:

| Precision | Scenario | Images | Warmup passes | Passes | Provider | Mode | Images/s | ms/image |
|---|---|---:|---:|---:|---|---|---:|---:|
| fp32 | throughput_adaptive | 120 | 1 | 5 | cuda | device | 129.537 | 7.7198 |
| fp16 | throughput_adaptive | 120 | 1 | 5 | cuda | device | 128.532 | 7.7802 |

Each report also contains other provider and mode records.

Each accuracy pass stored 1,500,000 COCO detections in a JSONL file. The report stores each
prediction file's name, SHA-256, and record count. It does not include the prediction records.

This excerpt shows the fp32 benchmark report output. It omits the full run metadata and other
provider records. It replaces the machine-specific GPU fingerprint in `record_id` with
`gpu-0`. The measurement values and other shown fields are from the report.

```json
{
  "schema_version": 2,
  "kind": "benchmark_report",
  "report_id": "rfdetr-nano_b1_fp32.onnx.benchmark-20261020T014403Z-c37089e4-throughput",
  "generated_at": "2026-10-20T01:44:03.667159+00:00",
  "run": {
    "run_id": "20261020T014403Z-c37089e4-throughput"
  },
  "model": {
    "id": "rfdetr-nano",
    "task": "object-detection",
    "format": "onnx",
    "precision": "fp32",
    "artifact": {
      "filename": "rfdetr-nano_b1_fp32.onnx",
      "sha256": "68d00b8d1583b02daa385dc91c681c9659760afab7c385a46381725ba6412c6a",
      "bytes": 107824324
    },
    "input_signature": {
      "batch_size": 1,
      "shapes": [[1, 3, 384, 384]],
      "dtype": "float32",
      "layout": "NCHW",
      "preprocessing_id": "imagenet_rgb_normalized"
    },
    "input_signature_id": "sha256:81772153f578ac742b505a09610aaffc81b43ae3717229ee3dc3d0f6a14dd972"
  },
  "performance": {
    "scenario": "throughput_adaptive",
    "protocol": {
      "version": 1,
      "images": 120,
      "warmup_passes": 1,
      "passes": 5,
      "min_pass_seconds": 1.0,
      "max_pass_seconds": 30.0,
      "cpu_threads": 0,
      "timing_scope": "session",
      "input_signature_id": "81772153f578ac742b505a09610aaffc81b43ae3717229ee3dc3d0f6a14dd972",
      "protocol_id": "f09db357871f17c0"
    },
    "records": [{
      "record_id": "cuda/device/gpu-0",
      "status": "measured",
      "provider": "cuda",
      "mode": "device",
      "timing": {
        "images_per_second": 129.537,
        "ms_per_image": 7.7198
      }
    }],
    "unavailable": []
  },
  "model_report": {
    "report_id": "rfdetr-nano_b1_fp32",
    "filename": "rfdetr-nano_b1_fp32.onnx.model.report.json"
  }
}
```

### JSON output and files

The validated output contains:

```json
{
  "schema_version": 2,
  "kind": "benchmark_report",
  "report_id": "detector-small.onnx.benchmark-run-20261020-101500",
  "generated_at": "2026-10-20T10:15:00Z",
  "run": {"run_id": "run-20261020-101500", "runner": {}, "hardware": {}, "libraries": {}},
  "model": {"id": "detector-small", "artifact": {"filename": "detector-small.onnx"}},
  "performance": {"scenario": "throughput_adaptive", "protocol": {}, "records": [], "unavailable": []},
  "model_report": {"report_id": "detector-small-fp32", "filename": "detector-small-fp32.json"},
  "provenance": {}
}
```

`model_report` and `provenance` are optional. Accuracy is not allowed at the top level or inside
the performance snapshot. `write_benchmark_report((json_path, html_path), payload)` validates and
writes both files. The standard path names are
`<artifact-filename>.benchmark-<run_id>.json` and `.html`.

## Run summary

### Build input

```python
build_run_summary(
    *,
    run: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    report_paths: Mapping[str, Path] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]
```

`run` is the shared run object. `reports` is a sequence of validated benchmark report payloads.
Optionally, `report_paths` maps each benchmark `report_id` to its JSON `Path`; an existing file is
hashed and the filename/hash are included in the summary. `generated_at` overrides the timestamp.

`write_run_summary(output_dir, *, run, report_files, generated_at=None)` is the file-based
alternative: it reads and validates the listed report JSON files, builds the summary, and writes
the output pair.

### JSON output and files

```json
{
  "schema_version": 2,
  "kind": "benchmark_summary",
  "run_id": "run-20261020-101500",
  "generated_at": "2026-10-20T10:16:00Z",
  "run": {"run_id": "run-20261020-101500", "runner": {}, "hardware": {}, "libraries": {}},
  "models": [{
    "model_id": "detector-small",
    "artifact": {"filename": "detector-small.onnx", "sha256": "<sha256>"},
    "report": "detector-small.onnx.benchmark-run-20261020-101500.json",
    "model_report": "detector-small-fp32.json",
    "performance_status": "measured",
    "summary": {
      "images_per_second": 200.0,
      "record_count": 1,
      "unavailable_count": 0
    },
    "report_sha256": "<sha256>"
  }],
  "unavailable": []
}
```

There is one `models` entry for each input benchmark report, sorted by model id. `artifact.sha256`,
`model_report`, and `report_sha256` may be absent when the source data is absent. `performance_status`
is `measured` when records exist and `unavailable` otherwise. The summary uses the fastest recorded
`images_per_second`; it reports record and unavailable counts. The top-level `unavailable` list
combines provider unavailability entries from the reports, keyed by provider. The standard files
are `benchmark-summary-<run_id>.json` and `benchmark-summary-<run_id>.html`.

## Public validation and write functions

Import these operations from `skellybench`:

| Report | Build | Validate | Write |
|---|---|---|---|
| Model report | `build_model_report` | `validate_model_report` | `write_model_report` |
| Benchmark report | `build_benchmark_report` | `validate_benchmark_report` | `write_benchmark_report` |
| Run summary | `build_run_summary` | `validate_benchmark_summary` | `write_run_summary` |

Validation failures raise `ModelReportValidationError` for model reports and
`SkellybenchValidationError` for benchmark reports and summaries.
