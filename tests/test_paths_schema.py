"""Paths, run ids, write-once files, blocks, and protocol ids."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import GPU_UUID, context, execution, local_probe
from skellybench import blocks
from skellybench._internal.io import write_json_once
from skellybench.paths import (
    benchmark_report_id,
    ep_slug,
    family_relative,
    hostname_slug,
    make_run_id,
    parse_benchmark_filename,
    resolve_family_path,
)
from skellybench.protocol import PerformanceProtocol, accuracy_protocol


def test_run_id_uses_utc_stamp_and_hostname_slug() -> None:
    when = datetime(2026, 10, 20, 15, 32, 39, tzinfo=UTC)
    assert make_run_id("Gerald.local", when) == "20261020T153239Z-gerald-local"
    assert hostname_slug("DESKTOP_AB.12") == "desktop-ab-12"
    with pytest.raises(ValueError):
        hostname_slug("...")


@pytest.mark.parametrize("ep", ["cpu", "cuda", "tensorrt-rtx", "openvino-gpu", "coreml", "torch-cuda", "torch-cpu"])
@pytest.mark.parametrize("batch", [None, 3])
def test_filename_round_trip(ep: str, batch: int | None) -> None:
    report_id = benchmark_report_id("rfdetr-nano_b1_fp32", "performance", ep, "20261020T153239Z-gerald", batch_size=batch)
    for suffix, predictions in ((".json", False), (".html", False), (".predictions.json", True)):
        parsed = parse_benchmark_filename(report_id + suffix)
        assert parsed is not None
        assert parsed["ep"] == ep and parsed["batch_size"] == batch and parsed["predictions"] is predictions


def test_subject_with_dot_is_rejected() -> None:
    with pytest.raises(ValueError, match="no '.'"):
        benchmark_report_id("yolox-s_v0.1", "accuracy", "cuda", "20261020T153239Z-gerald")
    assert ep_slug("openvino_gpu") == "openvino-gpu"


def test_family_paths_are_relative_and_cannot_escape(tmp_path: Path) -> None:
    target = tmp_path / "benchmarks" / "a" / "b.json"
    assert family_relative(tmp_path, target) == "benchmarks/a/b.json"
    with pytest.raises(ValueError):
        resolve_family_path(tmp_path, "../outside.json")


def test_write_json_once_refuses_overwrite_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    write_json_once(target, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json_once(target, {"a": 2})
    assert target.read_text(encoding="utf-8").strip() == '{\n  "a": 1\n}'
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


def test_write_json_once_falls_back_without_hard_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_links(*_args, **_kwargs):
        raise PermissionError("hard links are not supported")

    monkeypatch.setattr(os, "link", no_links)
    target = tmp_path / "report.json"
    write_json_once(target, {"a": 1})
    assert target.exists()
    with pytest.raises(FileExistsError):
        write_json_once(target, {"a": 2})
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


def test_blocks_build_fixed_fingerprints_short_names_and_labels() -> None:
    first = context()
    second = context()
    assert first["host"]["hardware_fingerprint"] == second["host"]["hardware_fingerprint"]
    assert first["environment"]["environment_fingerprint"] == second["environment"]["environment_fingerprint"]
    assert first["host"]["cpu"]["short_name"] == "i7-14700K"
    assert first["host"]["gpus"][0]["short_name"] == "RTX 3070"
    assert execution(first["host"])["label"] == "CUDA (RTX 3070)"
    assert execution(first["host"], "cpu")["label"] == "CPU (i7-14700K)"
    assert first["run"]["producer"]["name"] == "roboflow-exporter"


@pytest.mark.parametrize(
    ("name", "short"),
    [
        ("AMD Ryzen 9 7950X 16-Core Processor", "Ryzen 9 7950X"),
        ("Apple M5", "M5"),
        ("Intel(R) Core(TM) Ultra 7 155H", "Ultra 7 155H"),
    ],
)
def test_short_cpu_names(name: str, short: str) -> None:
    assert blocks.short_cpu_name(name) == short


def test_short_gpu_names() -> None:
    assert blocks.short_gpu_name("Intel(R) UHD Graphics 770") == "UHD 770"


def test_container_hostname_and_limits_enter_host_and_fingerprint() -> None:
    local = local_probe("3f2a9c")
    container = {"runtime": "docker", "image_digest": "sha256:" + "a" * 64, "host_hostname": "gerald",
                 "limits": {"cpu_quota_cores": 8.0, "cpuset": "0-7", "memory_limit_mb": 1024}}
    host = blocks.build_host(local, container)
    assert host["hostname"] == "gerald"
    assert host["virtualization"]["is_container"] is True
    limited = blocks.build_environment(local, host)
    container["limits"] = {"cpu_quota_cores": None, "cpuset": None, "memory_limit_mb": None}
    unlimited = blocks.build_environment(local, blocks.build_host(local, container))
    assert limited["environment_fingerprint"] != unlimited["environment_fingerprint"]


def test_execution_maps_gpu_by_uuid() -> None:
    host = context()["host"]
    block = execution(host)
    assert block["device"]["host_gpu_uuid"] == GPU_UUID
    assert block["drivers"]["nvidia_driver"] == "610.88"
    assert block["drivers"]["loaded_libraries"] == {"cudnn64_9.dll": "9.10.0"}


def test_protocol_ids_are_stable_and_type_specific() -> None:
    assert PerformanceProtocol().block(batch_size=1)["protocol_id"] == PerformanceProtocol().block(batch_size=1)["protocol_id"]
    assert PerformanceProtocol().block(batch_size=1)["protocol_id"] != PerformanceProtocol().block(batch_size=2)["protocol_id"]
    accuracy = accuracy_protocol(batch_size=1, images=10, evaluation={"max_dets": 100})
    assert accuracy["benchmark_type"] == "accuracy" and len(accuracy["protocol_id"]) == 16
    with pytest.raises(ValueError):
        PerformanceProtocol(passes=0)
