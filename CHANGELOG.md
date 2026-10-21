# Changelog

This file uses ASD-STE100 Simplified Technical English.
Every release can break the previous one. Update each consumer project when you upgrade.

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
