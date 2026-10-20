"""File layout, naming, and link rules for schema-3 reports.

Layout below a family directory (the folder that holds the model files)::

    <subject>.model.json / .model.html
    benchmarks/<subject>/<subject>.<type>.<ep>[.b<batch>].<run_id>.json / .html / .predictions.json
    superseded/<created_run_id>/<subject>.model.json / .model.html
    <family>.overview.html

Run summaries live at ``<exports root>/runs/<run_id>.html``.

Every path stored in a JSON report is relative to the family directory and uses ``/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from ._internal.timestamps import utc_stamp
from .schema import (
    BENCHMARK_TYPES,
    EP_SLUG_PATTERN,
    HOSTNAME_SLUG_PATTERN,
    RUN_ID_PATTERN,
    SUBJECT_PATTERN,
)

BENCHMARKS_DIR = "benchmarks"
SUPERSEDED_DIR = "superseded"
RUNS_DIR = "runs"
MODEL_JSON_SUFFIX = ".model.json"
MODEL_HTML_SUFFIX = ".model.html"
PREDICTIONS_SUFFIX = ".predictions.json"
OVERVIEW_SUFFIX = ".overview.html"

_BENCHMARK_NAME = re.compile(
    r"^(?P<subject>[A-Za-z0-9_-]+)\.(?P<type>accuracy|performance)\.(?P<ep>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"(?:\.b(?P<batch>[1-9][0-9]*))?\.(?P<run_id>\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"(?P<predictions>\.predictions)?\.(?P<ext>json|html)$"
)


def hostname_slug(hostname: str) -> str:
    """Return a lowercase ``[a-z0-9-]`` slug for a hostname. It never contains ``.``."""
    slug = re.sub(r"[^a-z0-9]+", "-", hostname.lower()).strip("-")
    if not slug or not HOSTNAME_SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"hostname {hostname!r} has no usable characters for a run id")
    return slug


def make_run_id(hostname: str, when: datetime | None = None) -> str:
    """Return ``<UTC YYYYMMDDTHHMMSSZ>-<hostname slug>``."""
    return f"{utc_stamp(when)}-{hostname_slug(hostname)}"


def ep_slug(provider_key: str) -> str:
    """Return the filename slug for a provider key, for example ``openvino_gpu`` -> ``openvino-gpu``."""
    slug = re.sub(r"[^a-z0-9]+", "-", provider_key.lower()).strip("-")
    if not EP_SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"provider key {provider_key!r} has no usable characters for a slug")
    return slug


def check_subject(subject: str) -> str:
    """Return *subject* or raise when it contains ``.`` or other unsafe characters."""
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise ValueError(f"subject {subject!r} must match {SUBJECT_PATTERN.pattern} (no '.')")
    return subject


def benchmark_report_id(
    subject: str,
    benchmark_type: str,
    ep: str,
    run_id: str,
    *,
    batch_size: int | None = None,
) -> str:
    """Return ``<subject>.<type>.<ep>[.b<batch>].<run_id>``.

    Pass *batch_size* only for a batch-agnostic subject, such as a source checkpoint.
    """
    check_subject(subject)
    if benchmark_type not in BENCHMARK_TYPES:
        raise ValueError(f"benchmark type must be one of {list(BENCHMARK_TYPES)}")
    if not EP_SLUG_PATTERN.fullmatch(ep):
        raise ValueError(f"EP slug {ep!r} must match {EP_SLUG_PATTERN.pattern}")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} must match {RUN_ID_PATTERN.pattern}")
    batch = f".b{int(batch_size)}" if batch_size is not None else ""
    return f"{subject}.{benchmark_type}.{ep}{batch}.{run_id}"


def parse_benchmark_filename(name: str) -> dict[str, object] | None:
    """Parse a benchmark JSON/HTML/predictions filename. Return ``None`` when it does not match."""
    match = _BENCHMARK_NAME.fullmatch(name)
    if match is None:
        return None
    return {
        "subject": match["subject"],
        "benchmark_type": match["type"],
        "ep": match["ep"],
        "batch_size": int(match["batch"]) if match["batch"] else None,
        "run_id": match["run_id"],
        "predictions": match["predictions"] is not None,
        "ext": match["ext"],
    }


@dataclass(frozen=True)
class BenchmarkPaths:
    """The three files of one benchmark report."""

    json: Path
    html: Path
    predictions: Path


def benchmark_dir(family_dir: str | Path, subject: str) -> Path:
    return Path(family_dir) / BENCHMARKS_DIR / check_subject(subject)


def benchmark_paths(family_dir: str | Path, subject: str, report_id: str) -> BenchmarkPaths:
    """Return the JSON, HTML, and predictions paths for a benchmark report id."""
    directory = benchmark_dir(family_dir, subject)
    return BenchmarkPaths(
        json=directory / f"{report_id}.json",
        html=directory / f"{report_id}.html",
        predictions=directory / f"{report_id}{PREDICTIONS_SUFFIX}",
    )


@dataclass(frozen=True)
class ModelPaths:
    """The two files of one model report."""

    json: Path
    html: Path


def model_paths(family_dir: str | Path, subject: str) -> ModelPaths:
    directory = Path(family_dir)
    check_subject(subject)
    return ModelPaths(json=directory / f"{subject}{MODEL_JSON_SUFFIX}", html=directory / f"{subject}{MODEL_HTML_SUFFIX}")


def superseded_model_paths(family_dir: str | Path, subject: str, created_run_id: str) -> ModelPaths:
    return model_paths(Path(family_dir) / SUPERSEDED_DIR / created_run_id, subject)


def run_summary_path(exports_root: str | Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} must match {RUN_ID_PATTERN.pattern}")
    return Path(exports_root) / RUNS_DIR / f"{run_id}.html"


def overview_path(family_dir: str | Path, family: str) -> Path:
    return Path(family_dir) / f"{family}{OVERVIEW_SUFFIX}"


def family_relative(family_dir: str | Path, path: str | Path) -> str:
    """Return *path* relative to *family_dir*, with ``/`` separators."""
    relative = Path(path).resolve().relative_to(Path(family_dir).resolve())
    return PurePosixPath(*relative.parts).as_posix()


def resolve_family_path(family_dir: str | Path, relative: str) -> Path:
    """Resolve a family-relative path. Raise when it escapes the family directory."""
    base = Path(family_dir).resolve()
    target = (base / PurePosixPath(relative)).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"path {relative!r} escapes the family directory")
    return target


def href(from_file: str | Path, to_file: str | Path) -> str:
    """Return a relative link from the HTML file *from_file* to *to_file*, with ``/`` separators."""
    source = Path(from_file).resolve().parent
    target = Path(to_file).resolve()
    common = Path(*_common_parts(source.parts, target.parts))
    ups = len(source.parts) - len(common.parts)
    rest = target.parts[len(common.parts):]
    return PurePosixPath(*([".."] * ups), *rest).as_posix()


def _common_parts(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    common: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a.lower() != b.lower():
            break
        common.append(a)
    return tuple(common)


__all__ = [
    "BENCHMARKS_DIR",
    "BenchmarkPaths",
    "MODEL_HTML_SUFFIX",
    "MODEL_JSON_SUFFIX",
    "ModelPaths",
    "OVERVIEW_SUFFIX",
    "PREDICTIONS_SUFFIX",
    "RUNS_DIR",
    "SUPERSEDED_DIR",
    "benchmark_dir",
    "benchmark_paths",
    "benchmark_report_id",
    "check_subject",
    "ep_slug",
    "family_relative",
    "hostname_slug",
    "href",
    "make_run_id",
    "model_paths",
    "overview_path",
    "parse_benchmark_filename",
    "resolve_family_path",
    "run_summary_path",
    "superseded_model_paths",
]
