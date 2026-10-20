# skellybench

`skellybench` is the library-agnostic scoring and reporting layer for the Roboflow Exporter.
Producers (`onnx_runtime`, `rf_detr`, and future exporters) hand it plain dictionaries. It validates
the bundle, optionally recomputes COCO metrics, and writes model reports, benchmark reports, run
summaries, and cross-machine overviews.

## Package layout

| Path | Responsibility |
|---|---|
| `scoring/` | COCO scoring, prediction sanitizing, and score results |
| `model_reports/` | Model-report schemas, validation, identity checks, and report lifecycle |
| `benchmark_reports/` | Benchmark reports, run summaries, and cross-machine benchmark overviews |
| `file_writer.py` | Shared atomic JSON and HTML file writes |
| `html_reports/` | HTML report rendering and shared escaping/table formatting |
| `schema.py` | Shared input, accuracy, performance, and report validation |
| `protocol.py` | Input signatures, benchmark protocols, and comparison identities |

Import public operations from `skellybench`. Keep report-specific rules in their report package.
Keep HTML generation in `html_reports/`. Use `file_writer.py` for report file writes.

## Design constraints

- No imports of `report_core`, `rf_detr`, `onnx_runtime`, `torch`, or `onnxruntime`.
- `faster-coco-eval` is optional and imported lazily, only when scoring is requested.
- Reports never contain raw prediction records or local input paths.
- Benchmark report artifacts are throughput-only. Accuracy metrics live in model reports.
- Model-report features are provided by `model_reports/` and exported at package root.
- Input paths are runtime inputs; reports use filenames, URLs, hashes, and counts.

## Capability probe

```powershell
uv run python -m skellybench capabilities
uv run python -m skellybench capabilities --format text
```

The capability contract is the single source used by the README and schema tests.

| Capability | Value |
|---|---|
| Schema version | `2` |
| Accuracy passes | `python`, `onnx` |
| Accuracy scenario | `accuracy_full_dataset` |
| Performance scenarios | `accuracy_full_dataset`, `throughput_adaptive` |
| Providers | `cpu`, `openvino_gpu`, `cuda`, `tensorrt_rtx` |
| Timing modes | `device`, `host` |
| Timing scopes | `session`, `end_to_end` |
| Prediction storages | `inline`, `json`, `jsonl` |
| Metric tool | `faster_coco_eval` |
| Scoring requires annotation | `true` |
| Supports precomputed metrics | `true` |
| Supports precomputed comparisons | `true` |
| Requires internal device match | `true` |
| Writes benchmark reports | `true` |
| Writes model reports | `true` |

## Input bundle

```json
{
  "schema_version": 2,
  "kind": "skellybench_run",
  "run": { "...": "..." },
  "dataset": { "...": "..." },
  "benchmark_images": { "...": "..." },
  "models": [{ "...": "..." }]
}
```

`run` requires `run_id`, `hardware`, `libraries`, and `runner.user`. `models` must be non-empty and
each model must include accuracy, performance, or both.

### Standard hardware and library versions

`run.hardware` records the machine identity, operating system, CPU, memory, and GPUs. Use `id` for
the machine identity and `hardware_id` for the hardware fingerprint. Each GPU records its index,
id, name, vendor, driver version, and VRAM when known.

`run.libraries` records Python and package versions:

```json
{
  "python": {"implementation": "CPython", "version": "3.14.0"},
  "packages": {
    "numpy": "2.5.4",
    "onnxruntime": "1.31.0",
    "onnxruntime-gpu": "not installed"
  }
}
```

Use distribution names as package keys. Use version strings as values. Use `"not installed"` when a
tracked package is absent. Hardware and library version data use the same shape in model reports.
Schema v1 input and reports remain readable. Writers produce schema v2.

### Model entry

Each model records its artifact filename/sha256, task, format, precision, canonical input signature,
and `input_signature_id`. `metadata` is opaque and is rendered but never interpreted.

## Accuracy

Accuracy is an apples-to-apples comparison of exactly two passes, `python` and `onnx`, over the same
images, preprocessing, thresholds, and device. There is no library-internal accuracy pass.

The device is selected from the source library's internal benchmark device and recorded in
`accuracy.device`:

- `type` is `cuda` or `cpu`.
- CUDA requires `index`; `torch_device` and `ort_provider` are the concrete selectors.
- `internal_benchmark` records the source tool, device, provider, and selection source.
- `matches_internal` must be `true`.
- Both pass-level `provenance.device`/`provider` values must match the accuracy device.
- CPU is used only when the source benchmark uses CPU. If CUDA is required but unavailable, accuracy
  becomes `unavailable` with a reason; it is never silently replaced with CPU.

### Scoring without an annotation file

| Input available | `accuracy.scoring` | Result |
|---|---|---|
| Predictions + annotation | `recomputed` | Metrics are computed and compared. |
| Predictions only, no annotation | `unavailable` | `accuracy.status = "unavailable"` with a reason; report-only rendering still works. |
| Precomputed metrics, no annotation | `precomputed` | Metrics are accepted as provenance; no recomputation. |
| Neither | `unavailable` | Status and reason only; metrics are never fabricated. |

Precomputed metrics and comparisons are accepted. When both predictions and metrics are supplied,
the local recomputation is recorded alongside the precomputed delta.

### COCO result record

```json
{"image_id": 39769, "category_id": 18, "bbox": [12.5, 40.0, 100.0, 220.0], "score": 0.91}
```

`bbox` is `[x, y, width, height]` in source-image pixels. Width and height must be non-negative.
Schema v1 supports COCO detection results only.

### Prediction stores

| Storage | Meaning |
|---|---|
| `inline` | `predictions.records` is an array of COCO result records. |
| `json` | `predictions.path` points to a JSON array of records. |
| `jsonl` | `predictions.path` points to one record per line. |

External stores require `sha256` and `record_count`; skellybench verifies both. The local path is
input-only and is replaced by a filename in report output.

## Performance

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
    "protocol_id": "..."
  },
  "records": [{ "...": "..." }],
  "unavailable": [{"provider": "openvino_gpu", "reason": "EP not available"}]
}
```

`scenario` is `throughput_adaptive` for adaptive warmup/timed-pass benchmarks, or
`accuracy_full_dataset` for end-to-end values logged during the full-dataset accuracy run.
`protocol_id` includes the scenario and input signature. Timings from different scenarios or timing
scopes are never merged.

Measured records require `timing.images_per_second` and `timing.ms_per_image`. All timing values are
non-negative finite numbers. Throughput is images per second, never batches per second.

## Report outputs

### Benchmark report

Report kind: `benchmark_report`

`<artifact>.benchmark-<run_id>.json` / `.html`

```json
{
  "schema_version": 2,
  "kind": "benchmark_report",
  "report_id": "<artifact>.benchmark-<run_id>",
  "generated_at": "...",
  "run": { "...": "..." },
  "model": { "...": "..." },
  "performance": { "...": "..." },
  "model_report": {"report_id": "...", "filename": "...", "sha256": "..."},
  "provenance": { "...": "..." }
}
```

A benchmark report never contains an `accuracy` section.

### Run summary

Report kind: `benchmark_summary`

`benchmark-summary-<run_id>.json` / `.html`

One row per model report: artifact identity, report filename/sha256, linked model report,
performance status, and best images/s.

### Cross-machine index

Report kind: `benchmark_overview`

`benchmark-overview-<UTC>.json` / `.html`

Flat rows per provider/mode/scenario with machine, OS, CPU, GPU, protocol, hardware/environment ids,
and comparison key. The index is rebuilt by rescanning report files. If a benchmark report file was
deleted but its payload is embedded in a model report, the snapshot is used as a rebuild source.

## Report lifecycle

- A benchmark report is immutable for one `run_id`.
- Each invocation must use a new run ID.
- An explicit `--benchmark-run-id` with existing report files must fail.
- A benchmark-only run must not change a model report.
- Do not merge timings from different scenarios or timing scopes.
- Do not write local paths to a report.
- Do not embed raw prediction records.
- The cross-machine index can rebuild rows from model report snapshots.

## Validation rules

- `schema_version` is positive; `kind` matches the expected document.
- `run_id` and report ids are Windows-safe.
- Models are non-empty and unique by `model.id` + `artifact.sha256`.
- A measured accuracy pass has predictions or metrics.
- Prediction stores use exactly one storage form and validate every record.
- Metrics values are finite; the canonical keys are `ap`, `ap50`, `ap75`, `ap_small`, `ap_medium`,
  `ap_large`, `ar1`, `ar10`, `ar100`, `ar_small`, `ar_medium`, `ar_large`.
- Performance record ids are unique; measured records have non-negative timings; unavailable
  records have a reason and no timing.
- Hashes are lowercase hex SHA-256.
- Report output must not contain local input paths or raw prediction records.

## Model reports

Report kind: `model_report`

Model reports contain model identity, provenance, accuracy results, benchmark snapshots, and the
standard hardware and library version blocks. The package writes JSON and HTML, checks report
identity before it skips generation, and rebuilds model-report overviews from disk.

Report kind: `model_report_overview`

A model report references each benchmark report by `report_id`, filename, `sha256`, and
`comparison_key`. It embeds the complete benchmark report payload, including run metadata, protocol,
hardware and library versions, provider details, and timing records. It never embeds raw prediction
records or local paths.
