"""The plain-data API for exporters that collect data in another process (version 0.3.0)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from conftest import context, execution, local_probe, metrics, sha, subject
from skellybench import SkellybenchValidationError, blocks
from skellybench.reports import build_benchmark_report
from skellybench.render import render_benchmark_html
from skellybench.schema import validate_environment

DIGEST = "sha256:" + "a" * 64
CONTAINER = {"image": "domscott/openmmlab-exporter", "tag": "df-0123456789ab", "digest": DIGEST,
             "dockerfile_sha256": sha("dockerfile"), "published": True}


def external(**overrides):
    values = {"python": {"implementation": "CPython", "version": "3.8.20"}, "packages": {"torch": "2.0.0"},
              "container": CONTAINER, "collected_by": "exporter:docker"}
    return blocks.build_external_environment(**{**values, **overrides})


# --- change 1: external environment ---------------------------------------------------------------


def test_external_environment_validates_and_fingerprint_covers_origin_and_container() -> None:
    environment = external()
    validate_environment(environment)
    assert environment["collected_by"] == "exporter:docker"
    assert environment["container"]["digest"] == DIGEST
    assert environment["package_sources"] == {}
    other_origin = external(collected_by="exporter:host")
    other_digest = external(container={**CONTAINER, "digest": "sha256:" + "b" * 64})
    no_container = external(container=None)
    fingerprints = {item["environment_fingerprint"] for item in (environment, other_origin, other_digest, no_container)}
    assert len(fingerprints) == 4
    assert external()["environment_fingerprint"] == environment["environment_fingerprint"]


def test_external_environment_rejects_bad_fields() -> None:
    with pytest.raises(ValueError):
        external(collected_by="")
    for key, value in (("digest", "sha256:short"), ("dockerfile_sha256", "xyz"), ("image", "")):
        broken = external(container={**CONTAINER, key: value})
        with pytest.raises(SkellybenchValidationError, match=key):
            validate_environment(broken)
    broken = external()
    broken["container"]["published"] = "yes"
    with pytest.raises(SkellybenchValidationError, match="published"):
        validate_environment(broken)
    broken = external(container={**CONTAINER, "image": "/home/user/image"})
    with pytest.raises(SkellybenchValidationError, match="absolute local path"):
        build_benchmark_report_with(broken)


def build_benchmark_report_with(environment):
    ctx = context()
    ctx["environment"] = environment
    return build_benchmark_report(
        benchmark_type="performance", status="unavailable", reason="test", execution=execution(ctx["host"]),
        subject=subject("rfdetr-nano_b1_fp32", sha("fp32")), **ctx)


def test_benchmark_page_shows_collector_and_container_image() -> None:
    report = build_benchmark_report_with(external())
    html = render_benchmark_html(report)
    assert "exporter:docker" in html
    assert "domscott/openmmlab-exporter:df-0123456789ab" in html
    assert DIGEST in html and "published" in html
    assert "Environment collected by" not in render_benchmark_html(build_benchmark_report_with(context()["environment"]))
