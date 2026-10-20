"""Build, write, supersede, and scan schema-3 benchmark and model reports.

JSON reports are written once (:func:`skellybench._internal.io.write_json_once`). HTML pages are
rendered from the JSON by :mod:`skellybench.render` and may be rewritten at any time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ._internal.hashing import sha256_file
from ._internal.io import read_json, write_json_once
from ._internal.timestamps import iso_now
from .paths import (
    BENCHMARKS_DIR,
    MODEL_JSON_SUFFIX,
    PREDICTIONS_SUFFIX,
    SUPERSEDED_DIR,
    BenchmarkPaths,
    ModelPaths,
    benchmark_paths,
    benchmark_report_id,
    family_relative,
    model_paths,
    superseded_model_paths,
)
from .predictions import PredictionColumns, write_predictions_file
from .schema import (
    BENCHMARK_REPORT_KIND,
    MODEL_REPORT_KIND,
    SCHEMA_VERSION,
    payload_kind,
    validate_benchmark_report,
    validate_model_report,
)


def _timestamp(value: datetime | str | None) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value or iso_now()


# --- benchmark reports -----------------------------------------------------------------------------


def report_id_for(subject: Mapping[str, Any], benchmark_type: str, execution: Mapping[str, Any], run_id: str) -> str:
    """Return the benchmark report id for a subject, type, execution block, and run."""
    return benchmark_report_id(
        str(subject["subject_id"]),
        benchmark_type,
        str(execution["provider"]["slug"]),
        run_id,
        batch_size=int(subject["batch_size"]) if subject.get("batch_agnostic") else None,
    )


def build_benchmark_report(
    *,
    benchmark_type: str,
    status: str,
    run: Mapping[str, Any],
    host: Mapping[str, Any],
    environment: Mapping[str, Any],
    execution: Mapping[str, Any],
    subject: Mapping[str, Any],
    dataset: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    accuracy: Mapping[str, Any] | None = None,
    reason: str | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build and validate one benchmark report."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": BENCHMARK_REPORT_KIND,
        "benchmark_type": benchmark_type,
        "status": status,
        "report_id": report_id_for(subject, benchmark_type, execution, str(run["run_id"])),
        "generated_at": _timestamp(generated_at),
        "run": dict(run),
        "host": dict(host),
        "environment": dict(environment),
        "execution": dict(execution),
        "subject": dict(subject),
        "dataset": dict(dataset) if dataset is not None else None,
        "protocol": dict(protocol) if protocol is not None else None,
        "performance": dict(performance) if performance is not None else None,
    }
    if accuracy is not None:
        payload["accuracy"] = dict(accuracy)
    if reason is not None:
        payload["reason"] = reason
    validate_benchmark_report(payload)
    return payload


def write_benchmark_predictions(
    family_dir: str | Path,
    *,
    subject_id: str,
    report_id: str,
    columns: PredictionColumns,
) -> dict[str, Any]:
    """Write the predictions file of an accuracy report. Return its ``accuracy.predictions`` reference.

    Call this before scoring and before the report is built, so the stored data is the scored data.
    """
    paths = benchmark_paths(family_dir, subject_id, report_id)
    reference = write_predictions_file(paths.predictions, report_id, columns)
    reference["filename"] = family_relative(family_dir, paths.predictions)
    return reference


def write_benchmark_report(family_dir: str | Path, report: Mapping[str, Any]) -> BenchmarkPaths:
    """Validate and write the benchmark JSON once. Render the HTML with :mod:`skellybench.render`."""
    validate_benchmark_report(report)
    paths = benchmark_paths(family_dir, str(report["subject"]["subject_id"]), str(report["report_id"]))
    write_json_once(paths.json, dict(report))
    return paths


# --- model reports ---------------------------------------------------------------------------------


def accuracy_entry(family_dir: str | Path, benchmark_json: str | Path) -> dict[str, Any]:
    """Return ``{"link": {"path", "sha256"}, "snapshot"}`` for one accuracy benchmark JSON file."""
    path = Path(benchmark_json)
    snapshot = read_json(path)
    if not isinstance(snapshot, dict):
        raise ValueError(f"{path}: benchmark JSON is missing or unparseable")
    return {
        "link": {"path": family_relative(family_dir, path), "sha256": sha256_file(path)},
        "snapshot": snapshot,
    }


def build_model_report(
    *,
    report_id: str,
    created_run_id: str,
    model: Mapping[str, Any],
    creation: Mapping[str, Any],
    accuracy_self: Mapping[str, Any],
    accuracy_parent: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build and validate one model report."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": MODEL_REPORT_KIND,
        "report_id": report_id,
        "created_run_id": created_run_id,
        "generated_at": _timestamp(generated_at),
        "model": dict(model),
        "creation": dict(creation),
        "accuracy": {
            "self": dict(accuracy_self),
            "parent": dict(accuracy_parent) if accuracy_parent is not None else None,
            "comparison": dict(comparison) if comparison is not None else None,
        },
    }
    validate_model_report(payload)
    return payload


def write_model_report(family_dir: str | Path, report: Mapping[str, Any]) -> ModelPaths:
    """Validate and write the model JSON once. Raise :class:`FileExistsError` when one exists."""
    validate_model_report(report)
    paths = model_paths(family_dir, str(report["report_id"]))
    write_json_once(paths.json, dict(report))
    return paths


def read_model_report(family_dir: str | Path, subject: str) -> dict[str, Any] | None:
    """Return the current valid model report for *subject*, or ``None``."""
    payload = read_json(model_paths(family_dir, subject).json)
    if payload_kind(payload) != MODEL_REPORT_KIND:
        return None
    try:
        validate_model_report(payload)
    except ValueError:
        return None
    return payload


def supersede_model_report(family_dir: str | Path, subject: str) -> ModelPaths | None:
    """Move the current model JSON and HTML to ``superseded/<created_run_id>/``.

    Return the new paths, or ``None`` when there is no model JSON. A model JSON that is not a valid
    schema-3 report moves to ``superseded/unknown/``.
    """
    current = model_paths(family_dir, subject)
    if not current.json.exists():
        return None
    payload = read_json(current.json)
    created = payload.get("created_run_id") if isinstance(payload, dict) else None
    target = superseded_model_paths(family_dir, subject, str(created or "unknown"))
    if target.json.exists():
        raise FileExistsError(f"superseded model report already exists: {target.json}")
    target.json.parent.mkdir(parents=True, exist_ok=True)
    os.replace(current.json, target.json)
    if current.html.exists():
        os.replace(current.html, target.html)
    return target


# --- scanning --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkEntry:
    """One benchmark report on disk."""

    json_path: Path
    html_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class ModelEntry:
    """One model report on disk (current or superseded)."""

    json_path: Path
    html_path: Path
    payload: dict[str, Any]
    superseded: bool


@dataclass
class FamilyIndex:
    """Every schema-3 report below one family directory."""

    family_dir: Path
    models: dict[str, ModelEntry] = field(default_factory=dict)
    superseded: list[ModelEntry] = field(default_factory=list)
    benchmarks: list[BenchmarkEntry] = field(default_factory=list)

    def benchmarks_for(self, subject: str, artifact_sha256: str | None = None) -> list[BenchmarkEntry]:
        return [
            entry
            for entry in self.benchmarks
            if entry.payload["subject"]["subject_id"] == subject
            and (artifact_sha256 is None or entry.payload["subject"]["artifact"]["sha256"] == artifact_sha256)
        ]

    def model_for_benchmark(self, payload: Mapping[str, Any]) -> ModelEntry | None:
        """Return the model report (current or superseded) for a benchmark's artifact SHA-256."""
        subject = payload["subject"]
        sha = subject["artifact"]["sha256"]
        current = self.models.get(str(subject["model_report_id"]))
        if current is not None and current.payload["model"]["artifact"]["sha256"] == sha:
            return current
        for entry in self.superseded:
            if (
                entry.payload["report_id"] == subject["model_report_id"]
                and entry.payload["model"]["artifact"]["sha256"] == sha
            ):
                return entry
        return None

    def all_models(self) -> list[ModelEntry]:
        return [*self.models.values(), *self.superseded]


def _valid(payload: Any, kind: str) -> dict[str, Any] | None:
    if payload_kind(payload) != kind:
        return None
    try:
        (validate_model_report if kind == MODEL_REPORT_KIND else validate_benchmark_report)(payload)
    except ValueError:
        return None
    return payload


def scan_family(family_dir: str | Path) -> FamilyIndex:
    """Read every valid schema-3 report below *family_dir*.

    Files whose ``schema_version`` is not 3, whose ``kind`` is unknown, or that fail validation are
    skipped. Predictions files are never opened.
    """
    root = Path(family_dir)
    index = FamilyIndex(family_dir=root)
    if not root.is_dir():
        return index
    for path in sorted(root.glob(f"*{MODEL_JSON_SUFFIX}")):
        payload = _valid(read_json(path), MODEL_REPORT_KIND)
        if payload is not None:
            index.models[payload["report_id"]] = ModelEntry(
                path, path.with_name(path.name[: -len(".json")] + ".html"), payload, False
            )
    for path in sorted((root / SUPERSEDED_DIR).glob(f"*/*{MODEL_JSON_SUFFIX}")):
        payload = _valid(read_json(path), MODEL_REPORT_KIND)
        if payload is not None:
            index.superseded.append(
                ModelEntry(path, path.with_name(path.name[: -len(".json")] + ".html"), payload, True)
            )
    for path in sorted((root / BENCHMARKS_DIR).glob("*/*.json")):
        if path.name.endswith(PREDICTIONS_SUFFIX):
            continue
        payload = _valid(read_json(path), BENCHMARK_REPORT_KIND)
        if payload is not None:
            index.benchmarks.append(BenchmarkEntry(path, path.with_suffix(".html"), payload))
    return index


def find_family_dirs(exports_root: str | Path) -> list[Path]:
    """Return every folder below *exports_root* that holds model reports or a ``benchmarks`` folder."""
    root = Path(exports_root)
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for path in root.rglob(f"*{MODEL_JSON_SUFFIX}"):
        if SUPERSEDED_DIR not in path.relative_to(root).parts:
            found.add(path.parent)
    for path in root.rglob(BENCHMARKS_DIR):
        if path.is_dir():
            found.add(path.parent)
    return sorted(found)


__all__ = [
    "BenchmarkEntry",
    "FamilyIndex",
    "ModelEntry",
    "accuracy_entry",
    "build_benchmark_report",
    "build_model_report",
    "find_family_dirs",
    "read_model_report",
    "report_id_for",
    "scan_family",
    "supersede_model_report",
    "write_benchmark_predictions",
    "write_benchmark_report",
    "write_model_report",
]
