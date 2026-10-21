# Changelog

This file uses ASD-STE100 Simplified Technical English.
Every release can break the previous one. Update each consumer project when you upgrade.

## 0.4.0

Stage timings and the dataset of each report.
The new fields are required. Reports that version 0.3.0 wrote fail validation, and readers skip them.
There is no migration. Write the reports again.

### Breaking changes since 0.2.0

- 0.3.0: an accuracy report needs the block `accuracy.method` (`decode`, `scorer`, `scored_by`).
  `accuracy.evaluation` holds only the settings that both passes of a pair share.
- 0.3.0: the performance protocol always has `excluded_stages`. Every performance `protocol_id` changed.
- 0.4.0: each performance record needs `timing.stages_ms_per_image`. This also applies to the record of
  an accuracy report.
- 0.4.0: the validator rejects `timing.stage_ms_per_image`. Write `stages_ms_per_image` with the key
  `postprocess`. The key `decode` is not a stage name.
- 0.4.0: the performance protocol always has `stage_definitions`. Every performance `protocol_id`
  changes again, also when the definitions are empty.
- 0.4.0: a measured stage needs `protocol.stage_definitions[<mode>]` for the mode of its record.
- 0.4.0: the dataset of an accuracy report needs `annotation_file`, `categories` and `annotation_sha256`.
- 0.4.0: a model report needs the top-level `dataset` block.

### Changes

- A performance record has `timing.stages_ms_per_image = {preprocess, inference, postprocess}`.
  Each value is a number >= 0, or `null` when the stage is not measured.
  The stages do not need to add up to `ms_per_image`.
- A performance record can have `timing.stage_providers = {preprocess, inference, postprocess}`.
  Each value is the name of the execution provider of the stage, or `null`.
- A performance record can have `timing.excluded_ms_per_image`, for example `{"letterbox": 0.85}`.
  Each key must be in `protocol.excluded_stages`. Each value is a number >= 0.
- The validator checks `timing.ms_per_image` when it is present. It must be a number >= 0.
- `PerformanceProtocol` takes `stage_definitions`: `{mode: {stage: text}}`.
  Each mode defines `preprocess`, `inference` and `postprocess`. A mode can also define a stage of
  `excluded_stages`. The definitions are part of `protocol_id`.
- `accuracy_protocol` takes the optional `stage_definitions`. Without it, the block and its
  `protocol_id` do not change.
- The dataset block of a benchmark report can have `split`, `annotation_file` and
  `categories` (`[{id, name}]`). A performance report reads no annotations, so `annotation_file`,
  `categories` and `annotation_sha256` can be `null` there.
- A model report has a top-level `dataset` block: `name`, `split`, `annotation_file`, `categories`,
  `images_used`, `images_available`, `selection`, `limit`, `annotation_sha256` and
  `image_list_sha256`. Each key that it shares with the dataset of `accuracy.self` must be equal.
  `build_model_report` takes `dataset`. When you do not give it, the function builds it from the dataset
  of `accuracy.self` with the new `blocks.build_model_dataset`.
- The benchmark page, the model page and the family overview show images/s, the end-to-end ms/image,
  the preprocess, inference and postprocess ms/image, and the rest as "other".
  They show the provider of a stage when it is not the provider of the pass, and the excluded
  letterbox time. A stage that is not measured shows "—". The run summary shows images/s and ms/image
  for each artifact and mode.
- The benchmark page shows the stage definitions of the protocol.
- The model page and the family overview show the dataset in the header. The model page also shows it in
  the accuracy section. The benchmark page shows the split, the annotation file and the categories.
- The family overview calls the batch latency column "p50 ms per batch".
- The pages show `decode`, `accuracy.evaluation`, the protocol and the step parameters as key/value
  lists. Nested blocks are indented.
- `python -m skellybench capabilities` reports `performance.stage_timings`, `performance.stages`,
  `performance.stage_providers`, `performance.excluded_ms_per_image`, `protocol.stage_definitions`,
  `dataset.annotation_file`, `dataset.categories`, `model_report.dataset`, `render.stage_breakdown` and
  `render.key_value_blocks`.

## 0.3.0

Plain-data API for exporters that collect data in another process.

- Add `blocks.build_external_environment` for an environment that a caller collected, for example in a
  Docker container. The block has `collected_by` and `container`. The validator checks both.
  The benchmark page shows them.
- Add the required block `accuracy.method` to accuracy reports: `decode`, `scorer`, `scored_by`.
  `accuracy.evaluation` now holds only the settings that both passes of a pair share.
  Add `blocks.build_accuracy_method` and `ScoringResult.as_scorer_block`.
  The model page shows the methods of `self` and `parent` side by side.
- `validate_accuracy_pair` no longer requires an equal `run.run_id`.
  `build_model_report` takes `parent_run_id`, stored as `accuracy.parent_run_id`.
  The validator requires it when the parent ran in another run. The model page marks a reused parent.
- `score_columns` and `score_predictions` take `category_ids`. The scorer sets `catIds` to them.
  `ScoringResult.category_ids` holds the ids that it used.
- Model reports take the optional `accuracy_notes` and `performance_notes`.
  `build_model_report` has the matching arguments. The model page has a "Notes" section.
  The family overview marks a model that has notes.
- Add `probes.local.cuda_devices()`. It uses the CUDA driver API through `ctypes` and returns
  `[{cuda_index, uuid, name, pci_bus_id}]`, or `None` without a driver. `local.collect()` includes it as
  `cuda_devices`.
- `PerformanceProtocol` takes `excluded_stages`. The protocol block always has the list, and it
  changes `protocol_id`. The performance chart, the performance table, and the benchmark page show it.
  This changes the `protocol_id` of every performance protocol, also with an empty list.
- `python -m skellybench capabilities` reports `environment.external`, `accuracy.method`,
  `pairs.cross_run`, `scoring.category_ids`, `model_report.notes`, `probes.cuda_devices`, and
  `protocol.excluded_stages`. It also reports the package version.
