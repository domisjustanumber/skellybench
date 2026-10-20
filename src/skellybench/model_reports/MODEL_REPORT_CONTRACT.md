# Model Report Contract

This document uses ASD-STE100 Simplified Technical English.

## Scope

This contract applies to model reports in `skellybench.model_reports`.

## Generation

- Generate a model report once.
- Skip a model report when the JSON and HTML files exist and the schema is valid.
- Also skip only when the report identity matches the artifact or source checkpoint.
- Use `--force` to regenerate a model report.
- Do not edit a generated report by hand.

## Content

A model report must contain:

- model identity
- runtime
- precision
- batch size
- input signature
- artifact filename
- artifact SHA-256 for ONNX reports
- source URL and MD5
- tool versions
- standard hardware and library version data
- command and config
- accuracy passes
- benchmark snapshots

## Accuracy

- Use exactly two passes: `python` and `onnx`.
- Use the same device, images, preprocessing, and settings for both passes.
- Read [`rf_detr/ACCURACY_CONTRACT.md`](../../rf_detr/ACCURACY_CONTRACT.md) for the device rules.

## Report safety

- Do not write local paths to a report.
- Do not embed raw prediction records.
- Store hardware under `environment.hardware`.
- Store Python and package versions under `environment.libraries`.
- Use distribution names as package keys and version strings as values.
- CUDA-enabled exporter reports can include `environment.libraries.cuda.torch_cuda_build` and
  `environment.libraries.cuda.cudnn_runtime_version`.
- Store both CUDA library versions as strings.
- Use `"not installed"` when a tracked package is absent.
- Use filenames, hashes, and record counts for prediction stores.
- A benchmark snapshot must not contain an `accuracy` section.
- A benchmark snapshot must not contain raw predictions.

## Rebuild

- Use `--render-reports` to rebuild report HTML and indexes.
- Use `--render-reports` to restore missing benchmark files from model report snapshots.
- Do not use `--render-reports` to remeasure a model.

## Schema

Write schema version 2. Read schema version 1 reports for compatibility. Version 2 uses the same
hardware and library version format as skellybench run inputs.
