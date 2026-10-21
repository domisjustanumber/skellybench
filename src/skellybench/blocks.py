"""Build the shared ``run``, ``host``, ``environment`` and ``execution`` report blocks.

The probes in :mod:`skellybench.probes` return plain data. This module adds what reports need on
top: short display names, chart labels, and fingerprints.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .protocol import fingerprint
from .schema import MODEL_DATASET_KEYS

_TRADEMARKS = re.compile(r"\((?:R|TM|tm|r)\)")
_CPU_NOISE = re.compile(
    r"\bCPU\b|\bProcessor\b|\b\d+-Core\b|@\s*[\d.]+\s*GHz|\bIntel\b|\bAMD\b|\bApple\b|\bCore\b(?=\s+(?:i\d|Ultra|\d))"
)
_GPU_NOISE = re.compile(r"\bNVIDIA\b|\bGeForce\b|\bIntel\b|\bGraphics\b|\bAMD\b|\bRadeon\b|\bApple\b")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _short(name: str | None, noise: re.Pattern[str]) -> str | None:
    if not name:
        return None
    plain = _squash(_TRADEMARKS.sub(" ", name))
    return _squash(noise.sub(" ", plain)) or plain


def short_cpu_name(name: str | None) -> str | None:
    """Return a display name such as ``i7-14700K`` or ``M5``. The full name stays the record."""
    return _short(name, _CPU_NOISE)


def short_gpu_name(name: str | None) -> str | None:
    """Return a display name such as ``RTX 3070`` or ``UHD 770``."""
    return _short(name, _GPU_NOISE)


def execution_label(provider_label: str, device_short_name: str | None) -> str:
    """Return ``"<EP label> (<device short name>)"``, the chart category for a benchmark."""
    return f"{provider_label} ({device_short_name})" if device_short_name else provider_label


def build_producer(
    *,
    name: str,
    version: str | None,
    git: Mapping[str, Any] | None = None,
    lock_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Return ``run.producer``: the tool that ran, with its source revision."""
    git = git or {}
    return {
        "name": name,
        "version": version,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "git_diff_sha256": git.get("diff_sha256"),
        "lock_file_sha256": lock_file_sha256,
    }


def build_run(
    *,
    run_id: str,
    started_at: str,
    command: str,
    producer: Mapping[str, Any],
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Return a ``run`` block."""
    run: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "command": command,
        "producer": dict(producer),
    }
    if finished_at is not None:
        run["finished_at"] = finished_at
    return run


def _gpu_entry(gpu: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(gpu)
    entry["short_name"] = short_gpu_name(gpu.get("name"))
    return entry


def resolved_hostname(local: Mapping[str, Any], container: Mapping[str, Any] | None) -> str:
    """Return the host hostname: ``SKELLYBENCH_HOST_HOSTNAME`` in a container, else the local name."""
    if container and container.get("host_hostname"):
        return str(container["host_hostname"])
    return str(local.get("hostname") or "unknown")


def build_host(local: Mapping[str, Any], container: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a ``host`` block from ``probes.local.collect()`` and ``probes.docker.collect()``."""
    cpu = dict(local.get("cpu") or {})
    cpu["short_name"] = short_cpu_name(cpu.get("name"))
    gpus = [_gpu_entry(gpu) for gpu in local.get("gpus") or []]
    local_virtualization = local.get("virtualization") or {}
    virtualization = {
        "is_wsl": local_virtualization.get("is_wsl"),
        "is_container": container is not None,
        "container_runtime": (container or {}).get("runtime"),
        "is_vm": local_virtualization.get("is_vm"),
        "hypervisor": local_virtualization.get("hypervisor"),
    }
    memory = {"total_mb": local.get("memory_total_mb")}
    hardware = {
        "cpu": [cpu.get("name"), cpu.get("physical_cores"), cpu.get("logical_cores")],
        "memory_mb": memory["total_mb"],
        "gpus": [[gpu.get("uuid"), gpu.get("name")] for gpu in gpus],
    }
    return {
        "hostname": resolved_hostname(local, container),
        "os": dict(local.get("os") or {}),
        "virtualization": virtualization,
        "cpu": cpu,
        "memory": memory,
        "gpus": gpus,
        "container": dict(container) if container is not None else None,
        "cuda_device_order": local.get("cuda_device_order"),
        "hardware_fingerprint": fingerprint(hardware),
    }


def build_environment(local: Mapping[str, Any], host: Mapping[str, Any]) -> dict[str, Any]:
    """Return an ``environment`` block. Native libraries go in ``execution.drivers``, not here."""
    container = host.get("container") or {}
    environment = {
        "python": dict(local.get("python") or {}),
        "packages": dict(local.get("packages") or {}),
        "package_sources": dict(local.get("package_sources") or {}),
    }
    environment["environment_fingerprint"] = fingerprint(
        {
            "os": host.get("os"),
            "virtualization": host.get("virtualization"),
            "gpu_drivers": [[gpu.get("uuid"), gpu.get("driver")] for gpu in host.get("gpus") or []],
            "python": environment["python"],
            "packages": environment["packages"],
            "package_sources": environment["package_sources"],
            "container": {
                "image_digest": container.get("image_digest"),
                "limits": container.get("limits"),
            },
        }
    )
    return environment


def build_external_environment(
    *,
    python: Mapping[str, Any],
    packages: Mapping[str, str],
    package_sources: Mapping[str, Any] | None = None,
    container: Mapping[str, Any] | None = None,
    collected_by: str,
) -> dict[str, Any]:
    """Return an ``environment`` block for an environment that the caller collected in another process.

    Use it for a Docker container, where :mod:`skellybench.probes` did not run. The block has the
    normal ``python``, ``packages`` and ``package_sources`` fields, and two more:

    - ``collected_by``: who collected the data, for example ``"exporter:docker"``.
    - ``container``: ``{image, tag, digest, dockerfile_sha256, published}``, or ``None``.

    ``environment_fingerprint`` covers every field.
    """
    if not collected_by or not isinstance(collected_by, str):
        raise ValueError("collected_by must be a non-empty string")
    entry: dict[str, Any] | None = None
    if container is not None:
        entry = {
            "image": container.get("image"),
            "tag": container.get("tag"),
            "digest": container.get("digest"),
            "dockerfile_sha256": container.get("dockerfile_sha256"),
            "published": bool(container.get("published", False)),
        }
    environment: dict[str, Any] = {
        "python": dict(python),
        "packages": {str(name).lower(): str(version) for name, version in packages.items()},
        "package_sources": dict(package_sources or {}),
        "collected_by": collected_by,
        "container": entry,
    }
    environment["environment_fingerprint"] = fingerprint(environment)
    return environment


def build_accuracy_method(
    *,
    decode: Mapping[str, Any],
    scorer_tool: str,
    scorer_version: str | None,
    scored_by: str,
) -> dict[str, Any]:
    """Return an ``accuracy.method`` block: how this pass decoded its output and who scored it.

    *scorer_tool* must equal ``accuracy.metrics.tool``. *scored_by* is ``"skellybench"`` or
    ``"source_tool"``.
    """
    return {
        "decode": dict(decode),
        "scorer": {"tool": scorer_tool, "version": scorer_version},
        "scored_by": scored_by,
    }


def build_model_dataset(snapshot_dataset: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return the model report ``dataset`` block from the ``dataset`` block of its accuracy pass.

    The block has the keys of :data:`skellybench.schema.MODEL_DATASET_KEYS` that *snapshot_dataset*
    has. *overrides* add or replace keys, for example ``split="val2017"`` when the accuracy pass did
    not record the split. The validator checks that each shared key equals the accuracy pass.
    """
    block = {key: snapshot_dataset[key] for key in MODEL_DATASET_KEYS if key in snapshot_dataset}
    block.update(overrides)
    return block


def find_gpu(host: Mapping[str, Any], uuid: str | None) -> Mapping[str, Any] | None:
    """Return the ``host.gpus`` entry with *uuid*, or ``None``."""
    if not uuid:
        return None
    for gpu in host.get("gpus") or []:
        if str(gpu.get("uuid") or "").lower() == uuid.lower():
            return gpu
    return None


def build_execution(
    *,
    host: Mapping[str, Any],
    runtime: str,
    runtime_version: str | None,
    provider_key: str,
    provider_slug: str,
    provider_label: str,
    ep_name: str,
    device_kind: str,
    gpu_uuid: str | None = None,
    ep_device_id: int | None = None,
    device_name: str | None = None,
    options: Mapping[str, Any] | None = None,
    plugin: Mapping[str, Any] | None = None,
    loaded_libraries: Mapping[str, str | None] | None = None,
    extra_drivers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an ``execution`` block that links an EP to the physical device it ran on.

    For a GPU, pass *gpu_uuid*; the device name and driver come from the matching ``host.gpus``
    entry. For a CPU, the device name is ``host.cpu.name``.
    """
    drivers: dict[str, Any] = {}
    short_name: str | None = None
    if device_kind == "gpu":
        gpu = find_gpu(host, gpu_uuid)
        if gpu is not None:
            device_name = device_name or gpu.get("name")
            short_name = gpu.get("short_name") or short_gpu_name(device_name)
            if gpu.get("driver"):
                drivers[f"{gpu.get('vendor') or 'gpu'}_driver"] = gpu["driver"]
        else:
            short_name = short_gpu_name(device_name)
    else:
        cpu = host.get("cpu") or {}
        device_name = device_name or cpu.get("name")
        short_name = cpu.get("short_name") or short_cpu_name(device_name)
    if loaded_libraries:
        drivers["loaded_libraries"] = dict(loaded_libraries)
    drivers.update(dict(extra_drivers or {}))
    device: dict[str, Any] = {"kind": device_kind, "name": device_name, "short_name": short_name}
    if device_kind == "gpu":
        device["host_gpu_uuid"] = gpu_uuid
        device["ep_device_id"] = ep_device_id
    provider: dict[str, Any] = {
        "key": provider_key,
        "slug": provider_slug,
        "label": provider_label,
        "ep_name": ep_name,
        "options": dict(options or {}),
    }
    if plugin:
        provider["plugin"] = dict(plugin)
    return {
        "runtime": runtime,
        "runtime_version": runtime_version,
        "provider": provider,
        "device": device,
        "drivers": drivers,
        "label": execution_label(provider_label, short_name),
    }


def describe_container_limits(host: Mapping[str, Any]) -> Sequence[str]:
    """Return human-readable container limits for HTML pages."""
    container = host.get("container") or {}
    limits = container.get("limits") or {}
    parts = []
    if limits.get("cpu_quota_cores") is not None:
        parts.append(f"CPU quota {limits['cpu_quota_cores']} cores")
    if limits.get("cpuset"):
        parts.append(f"cpuset {limits['cpuset']}")
    if limits.get("memory_limit_mb") is not None:
        parts.append(f"memory {limits['memory_limit_mb']} MB")
    return parts


__all__ = [
    "build_accuracy_method",
    "build_environment",
    "build_execution",
    "build_external_environment",
    "build_host",
    "build_producer",
    "build_run",
    "describe_container_limits",
    "execution_label",
    "find_gpu",
    "resolved_hostname",
    "short_cpu_name",
    "short_gpu_name",
]
