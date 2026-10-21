# Report Contract

This document uses ASD-STE100 Simplified Technical English.

## Scope

This contract applies to every report that skellybench writes: schema version 3.
[`docs/api_documentation.md`](../../docs/api_documentation.md) gives the full data format.

## Immutability

- Write each JSON report once with `write_json_once`. Do not replace it. Do not edit it.
- `write_json_once` publishes with `os.link`, which fails when the target exists.
  On a file system without hard links, it uses an exclusive create.
- Render HTML pages from JSON. You can rewrite an HTML page at any time.
- Do not rebuild a JSON report from an HTML page or from a model report snapshot.

## Benchmark reports

- Write one benchmark report for each run, subject, benchmark type, and execution provider.
- Name it `<subject>.<type>.<ep>[.b<batch>].<run_id>.json`.
  Add `.b<batch>` only for a batch-agnostic subject, for example a source checkpoint.
- A subject must not contain `.`.
- An accuracy report has COCO metrics, the end-to-end timing of the pass, and a predictions reference.
- A performance report has no accuracy block.
- An accuracy report has `accuracy.evaluation` and `accuracy.method`.
  `evaluation` holds only the settings that both passes of a pair share.
  `method` holds `decode`, `scorer` (`tool`, `version`), and `scored_by` (`skellybench` or `source_tool`).
  `method.scorer.tool` must equal `accuracy.metrics.tool`.
  The protocol id does not cover `method`.
- A report with `status` other than `measured` has a `reason` and no accuracy block.
- A performance protocol has `excluded_stages`: the pipeline stages that the timing leaves out.
  The list is part of the protocol id. The pages show it.
- The `execution` block links the execution provider to the physical device.
  A GPU device carries the UUID of a `host.gpus` entry.

## Environments

- Build the `environment` block of a local process from the probes with `build_environment`.
- Build the `environment` block of another process, for example a Docker container, with
  `build_external_environment`.
- An external environment has `collected_by` and `container`.
  `container` holds `image`, `tag`, `digest`, `dockerfile_sha256`, and `published`.
- The fingerprint of an external environment covers `collected_by` and `container`.
- `container.digest` is `sha256:` plus 64 hex characters. `container.dockerfile_sha256` is 64 hex characters.

## Predictions files

- Write one predictions file for each accuracy report: `<report_id>.predictions.json`.
- Store the detections as little-endian columns: `category_id` (uint16), `bbox_xywh` (4 x float32),
  and `score` (float32). Then gzip, then base64.
- Record `data_sha256` of the decoded bytes.
- Write the predictions file before you score. Score the decoded file content.
- Renderers do not open predictions files.
- To score a subset of categories, pass `category_ids` to the scorer.
  Record `ScoringResult.category_ids` in `accuracy.evaluation.category_ids`.

## Model reports

- Write one model report for each model artifact, when the model is created.
- A model report holds the creation record: run, host, environment, source, parent, and steps.
- A model report holds full copies of two accuracy benchmark reports: `self` and `parent`.
  A model without a parent has `parent: null` and `comparison: null`.
- `self` and `parent` must share the dataset hashes, the batch size, `accuracy.evaluation`,
  and the physical device. The execution provider can differ. The run can differ.
- If the parent snapshot ran in another run than `created_run_id`, set `accuracy.parent_run_id` to the
  run id of the parent snapshot. Otherwise leave it empty.
- Nobody compares `accuracy.method` between `self` and `parent`.
- A model report can have `accuracy_notes` and `performance_notes`. Each one is a non-empty string.
  Use them to explain a known difference, for example a different decode in the two passes.
- A creation step with `input_sha256` must name the parent SHA-256.
- The last creation step must output the artifact SHA-256.
- Move a replaced model report to `superseded/<created_run_id>/`. Do not overwrite it.

## Paths

- Do not store an absolute local path in a report.
- Store every path relative to the family directory.
- A benchmark report does not store a path to its model report. The renderer finds the model
  report by `subject.model_report_id` and the artifact SHA-256.

## Run summary and family overview

- The run summary and the family overview are HTML only.
- The run summary keeps its run state in a JSON data island, so a rebuild keeps the state.
- The family overview reads current model reports and their performance benchmarks.

## Readers

- Skip a JSON file whose `schema_version` is not 3 or whose `kind` is unknown.
- Skip a JSON file that fails validation. Treat a partial file as absent.

## Probes

- Only `probes/local.py` and `probes/docker.py` detect hardware, versions, or containers.
- The probe files use the standard library only and do not import each other.
- A probe that cannot run returns `None`. It does not guess.
- `cuda_devices()` uses the CUDA driver API through `ctypes`. It does not import torch.
