# Changelog

This file uses ASD-STE100 Simplified Technical English.
Every release can break the previous one. Update each consumer project when you upgrade.

## 0.3.0

Plain-data API for exporters that collect data in another process.

- Add `blocks.build_external_environment` for an environment that a caller collected, for example in a
  Docker container. The block has `collected_by` and `container`. The validator checks both.
  The benchmark page shows them.
