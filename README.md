# skellybench

This document uses ASD-STE100 Simplified Technical English.

`skellybench` writes model and benchmark reports, scores COCO detections, and probes the environment.
Producers (for example an exporter) give it plain dictionaries.
It validates them, writes the JSON reports once, and renders HTML pages from the JSON.

## Install

```powershell
uv add skellybench
uv add "skellybench[scoring]"
```

The `scoring` extra installs `faster-coco-eval`.
The rest of the package uses the standard library only.

## Reports

Schema version 3 has three JSON kinds and two HTML-only pages.

| Report | JSON | HTML | One per |
|---|---|---|---|
| Benchmark report | yes, write-once | yes | run, subject, benchmark type, execution provider |
| Predictions file | yes, write-once | no | accuracy benchmark report |
| Model report | yes, write-once | yes | model artifact |
| Run summary | no | yes | run |
| Family overview | no | yes | model family |

The JSON files are the source of truth.
You can delete any HTML page and rebuild it with `skellybench.render.rebuild_all`.
Read [`src/skellybench/REPORT_CONTRACT.md`](src/skellybench/REPORT_CONTRACT.md) for the rules and
[`docs/api_documentation.md`](docs/api_documentation.md) for the data format and the API.

## Package layout

Paths are relative to `src/skellybench/`.

| Path | Responsibility |
|---|---|
| `schema.py` | Schema version 3 constants and validators |
| `paths.py` | File layout, naming, run IDs, and the link rule |
| `blocks.py` | Builds the `run`, `host`, `environment` (also for another process), `execution`, and `method` blocks |
| `protocol.py` | Protocol blocks and stable IDs |
| `predictions.py` | Columnar gzip+base64 predictions files |
| `reports.py` | Builds, writes, supersedes, and scans reports |
| `scoring/` | COCO scoring and self-vs-parent comparison |
| `render/` | HTML pages: benchmark, model, run summary, family overview |
| `probes/local.py` | Standalone probes for this machine and its Python environment |
| `probes/docker.py` | Standalone probes for the container that the process runs in |

## Environment probes

`probes/local.py` and `probes/docker.py` are standalone.
Each file uses the standard library only and imports no other skellybench module.
You can import them, or copy one file into another project.

```python
from skellybench.probes import docker, local

machine = local.collect(repository=".", lock_file="uv.lock")
container = docker.collect()  # None outside a container
```

Each public function does one probe and returns JSON-ready data.
A probe that cannot run returns `None` for its value.
Each probe takes optional arguments (filesystem root, command runner, environment) so that tests can
replace the system.
`local.cuda_devices()` lists the CUDA devices in CUDA order, with the UUID, the name, and the PCI bus ID.
It uses the CUDA driver API through `ctypes`, not torch.
It returns `None` when the driver is missing.
`local.collect()` returns it as `cuda_devices`.
The module docstrings list every function and the platforms that it supports.

In a container, set these variables, because a container cannot see its own image digest:

| Variable | Meaning |
|---|---|
| `SKELLYBENCH_IMAGE` | The image reference |
| `SKELLYBENCH_IMAGE_DIGEST` | The image digest |
| `SKELLYBENCH_HOST_HOSTNAME` | The host name. Use it in the run ID. |

## Capability probe

```powershell
python -m skellybench capabilities
python -m skellybench capabilities --format text
```

The JSON lists the features of this version. Check one with `skellybench.supports("pairs.cross_run")`.
Version 0.3.0 reports `environment.external`, `accuracy.method`, `pairs.cross_run`,
`scoring.category_ids`, `model_report.notes`, `probes.cuda_devices`, and `protocol.excluded_stages`.

## Development

```powershell
uv sync
uv run pytest
uv build
```

The tests do not need network access, model weights, a GPU, or COCO data.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later.
See [LICENSE](LICENSE) for the full text.
