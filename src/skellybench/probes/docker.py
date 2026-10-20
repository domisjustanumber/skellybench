"""Probes for the container that this process runs in.

This module uses the Python standard library only.
It does not import other skellybench modules, so you can copy it into another project.
Each public function does one probe and returns plain JSON-ready data.
A probe that cannot find a value returns ``None`` for it. It never guesses.

A process in a container cannot see its own image digest. Set these variables in the image or in
``docker run`` so the probe can record them:

- ``SKELLYBENCH_IMAGE``: the image reference, for example ``ghcr.io/org/exporter:2026-10-20``.
- ``SKELLYBENCH_IMAGE_DIGEST``: the image digest, for example ``sha256:...``.
- ``SKELLYBENCH_HOST_HOSTNAME``: the hostname of the machine that runs the container.

Public functions:

- ``is_container()``: ``True``, ``False``, or ``None`` when no check can run.
- ``container_runtime()``: ``docker``, ``podman``, ``containerd``, ``unknown``, or ``None``.
- ``container_id()``: the container ID from ``/proc/self/mountinfo`` or ``/proc/self/cgroup``.
- ``image()``: image reference and digest from the variables above, or from podman's
  ``/run/.containerenv``.
- ``base_os()``: ``PRETTY_NAME`` from ``/etc/os-release``.
- ``cgroup_limits()``: CPU quota, cpuset, and memory limit (cgroup v2, with a v1 fallback).
- ``gpu_runtime()``: NVIDIA Container Toolkit variables.
- ``host_hostname()``: ``SKELLYBENCH_HOST_HOSTNAME``.
- ``collect()``: ``None`` outside a container, else all the probes above in one mapping.

Optional arguments let a test replace the system: ``root`` (filesystem root), ``environ``
(environment mapping), ``system`` (``platform.system()`` value), and ``registry`` (a callable that
returns the Windows ``ContainerType`` value).
"""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

IMAGE_VARIABLE = "SKELLYBENCH_IMAGE"
IMAGE_DIGEST_VARIABLE = "SKELLYBENCH_IMAGE_DIGEST"
HOST_HOSTNAME_VARIABLE = "SKELLYBENCH_HOST_HOSTNAME"

_CONTAINER_ID = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
_CGROUP_MARKERS = (
    ("libpod", "podman"),
    ("docker", "docker"),
    ("containerd", "containerd"),
    ("kubepods", "containerd"),
    ("lxc", "lxc"),
)
# cgroup v1 reports "no limit" as a very large page-aligned number.
_V1_UNLIMITED_BYTES = 1 << 60


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return environ if environ is not None else os.environ


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _windows_container_type() -> int | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control") as key:
            value, _ = winreg.QueryValueEx(key, "ContainerType")
    except OSError:
        return None
    return int(value) if isinstance(value, int) else None


def _containerenv(root: Path) -> dict[str, str]:
    text = _read_text(root / "run" / ".containerenv")
    values: dict[str, str] = {}
    for line in (text or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip().strip('"')
    return values


def container_runtime(
    *,
    root: Path = Path("/"),
    system: str | None = None,
    registry: Callable[[], int | None] | None = None,
) -> str | None:
    """Return the container runtime name, or ``None`` outside a container."""
    current = system if system is not None else platform.system()
    if current == "Windows":
        container_type = (registry or _windows_container_type)()
        return "docker" if container_type else None
    if (root / "run" / ".containerenv").exists():
        return "podman"
    if (root / ".dockerenv").exists():
        return "docker"
    cgroup = _read_text(root / "proc" / "1" / "cgroup") or ""
    for marker, runtime in _CGROUP_MARKERS:
        if marker in cgroup:
            return runtime
    return None


def is_container(
    *,
    root: Path = Path("/"),
    system: str | None = None,
    registry: Callable[[], int | None] | None = None,
) -> bool | None:
    """Return ``True`` in a container, ``False`` outside one, ``None`` when no check can run."""
    current = system if system is not None else platform.system()
    if container_runtime(root=root, system=current, registry=registry) is not None:
        return True
    if current == "Windows":
        return False
    if (root / "proc" / "1" / "cgroup").exists() or (root / "proc").exists():
        return False
    return None


def container_id(*, root: Path = Path("/")) -> str | None:
    """Return the 64-hex container ID, or ``None``."""
    for relative in (("proc", "self", "mountinfo"), ("proc", "self", "cgroup"), ("proc", "1", "cgroup")):
        text = _read_text(root.joinpath(*relative)) or ""
        for line in text.splitlines():
            if not any(marker in line for marker in ("docker", "containers", "libpod", "containerd", "kubepods")):
                continue
            match = _CONTAINER_ID.search(line)
            if match:
                return match.group(1)
    podman = _containerenv(root).get("id")
    return podman or None


def image(*, root: Path = Path("/"), environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Return ``{"image", "image_digest"}`` from the ``SKELLYBENCH_*`` variables or podman."""
    values = _environ(environ)
    podman = _containerenv(root)
    reference = values.get(IMAGE_VARIABLE) or podman.get("image") or None
    digest = values.get(IMAGE_DIGEST_VARIABLE) or None
    if digest is None and podman.get("imageid"):
        digest = f"sha256:{podman['imageid']}" if ":" not in podman["imageid"] else podman["imageid"]
    return {"image": reference, "image_digest": digest}


def base_os(*, root: Path = Path("/")) -> str | None:
    """Return ``PRETTY_NAME`` from ``/etc/os-release``."""
    text = _read_text(root / "etc" / "os-release") or ""
    match = re.search(r'^PRETTY_NAME="?([^"\n]*)"?$', text, re.MULTILINE)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _first_text(root: Path, *candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        text = _read_text(root.joinpath(*candidate))
        if text is not None and text.strip():
            return text.strip()
    return None


def cgroup_limits(*, root: Path = Path("/")) -> dict[str, Any]:
    """Return ``cpu_quota_cores``, ``cpuset``, and ``memory_limit_mb``. ``None`` means not limited."""
    cgroup = root / "sys" / "fs" / "cgroup"
    cpu_quota: float | None = None
    memory_mb: int | None = None

    cpu_max = _read_text(cgroup / "cpu.max")
    if cpu_max is not None:
        quota, _, period = cpu_max.strip().partition(" ")
        if quota != "max" and quota and period:
            try:
                cpu_quota = round(int(quota) / int(period), 3)
            except (ValueError, ZeroDivisionError):
                cpu_quota = None
    else:
        quota_text = _first_text(root, ("sys", "fs", "cgroup", "cpu", "cpu.cfs_quota_us"),
                                 ("sys", "fs", "cgroup", "cpu,cpuacct", "cpu.cfs_quota_us"))
        period_text = _first_text(root, ("sys", "fs", "cgroup", "cpu", "cpu.cfs_period_us"),
                                  ("sys", "fs", "cgroup", "cpu,cpuacct", "cpu.cfs_period_us"))
        try:
            if quota_text is not None and period_text is not None and int(quota_text) > 0:
                cpu_quota = round(int(quota_text) / int(period_text), 3)
        except (ValueError, ZeroDivisionError):
            cpu_quota = None

    cpuset = _first_text(
        root,
        ("sys", "fs", "cgroup", "cpuset.cpus.effective"),
        ("sys", "fs", "cgroup", "cpuset", "cpuset.effective_cpus"),
        ("sys", "fs", "cgroup", "cpuset", "cpuset.cpus"),
    )

    memory_max = _read_text(cgroup / "memory.max")
    if memory_max is not None:
        text = memory_max.strip()
        if text and text != "max":
            try:
                memory_mb = int(text) // (1024 * 1024)
            except ValueError:
                memory_mb = None
    else:
        limit_text = _first_text(root, ("sys", "fs", "cgroup", "memory", "memory.limit_in_bytes"))
        try:
            if limit_text is not None and int(limit_text) < _V1_UNLIMITED_BYTES:
                memory_mb = int(limit_text) // (1024 * 1024)
        except ValueError:
            memory_mb = None
    return {"cpu_quota_cores": cpu_quota, "cpuset": cpuset, "memory_limit_mb": memory_mb}


def gpu_runtime(*, environ: Mapping[str, str] | None = None) -> dict[str, str | None] | None:
    """Return NVIDIA Container Toolkit variables, or ``None`` when none is set."""
    values = _environ(environ)
    result = {
        "nvidia_visible_devices": values.get("NVIDIA_VISIBLE_DEVICES") or None,
        "nvidia_driver_capabilities": values.get("NVIDIA_DRIVER_CAPABILITIES") or None,
        "image_cuda_version": values.get("CUDA_VERSION") or None,
    }
    return result if any(value is not None for value in result.values()) else None


def host_hostname(*, environ: Mapping[str, str] | None = None) -> str | None:
    """Return ``SKELLYBENCH_HOST_HOSTNAME``, or ``None``."""
    return _environ(environ).get(HOST_HOSTNAME_VARIABLE) or None


def collect(
    *,
    root: Path = Path("/"),
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    registry: Callable[[], int | None] | None = None,
) -> dict[str, Any] | None:
    """Return the container block, or ``None`` when the process does not run in a container."""
    runtime = container_runtime(root=root, system=system, registry=registry)
    if runtime is None:
        return None
    image_info = image(root=root, environ=environ)
    return {
        "runtime": runtime,
        "container_id": container_id(root=root),
        "image": image_info["image"],
        "image_digest": image_info["image_digest"],
        "base_os": base_os(root=root),
        "limits": cgroup_limits(root=root),
        "gpu_runtime": gpu_runtime(environ=environ),
        "host_hostname": host_hostname(environ=environ),
    }


__all__ = [
    "HOST_HOSTNAME_VARIABLE",
    "IMAGE_DIGEST_VARIABLE",
    "IMAGE_VARIABLE",
    "base_os",
    "cgroup_limits",
    "collect",
    "container_id",
    "container_runtime",
    "gpu_runtime",
    "host_hostname",
    "image",
    "is_container",
]
