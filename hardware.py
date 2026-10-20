"""Optional producer-side hardware and library-version detection.

These helpers are conveniences for producers. Reports never substitute the local machine for a
supplied identity, and imported reports are validated rather than re-detected.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from report_core import sha256_text

DEFAULT_SOFTWARE_PACKAGES = (
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "numpy",
    "faster-coco-eval",
    "torch",
    "torchvision",
)


def _distribution_name(name: str) -> str:
    """Return a PEP 503-style package key."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _run(command: Sequence[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _gpu_inventory() -> list[dict[str, Any]]:
    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        inventory: list[dict[str, Any]] = []
        for line in nvidia.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            inventory.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else len(inventory),
                    "id": parts[1] or "unknown",
                    "name": parts[2] or "unknown",
                    "driver": parts[3] or "unknown",
                    "vram_mb": int(parts[4].split()[0]) if parts[4].split() and parts[4].split()[0].isdigit() else None,
                    "vendor": "nvidia",
                }
            )
        return inventory
    return []


def _memory_total_mb() -> int | None:
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys // (1024 * 1024))
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (AttributeError, OSError, ValueError):
        return None


def detect_hardware(*, label: str | None = None) -> dict[str, Any]:
    """Detect local hardware. Missing values are omitted, never guessed."""
    gpus = _gpu_inventory()
    cpu_model = platform.processor() or platform.machine()
    logical = os.cpu_count()
    identity = {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model,
        "logical_cores": logical,
        "memory_mb": _memory_total_mb(),
        "gpus": [{"id": gpu.get("id"), "name": gpu.get("name")} for gpu in gpus],
    }
    machine_id = "sha256:" + sha256_text(str(identity["node"]))
    hardware_id = "sha256:" + sha256_text(
        repr(
            (
                identity["cpu"],
                identity["logical_cores"],
                identity["memory_mb"],
                tuple((gpu.get("id"), gpu.get("name")) for gpu in gpus),
            )
        )
    )
    machine: dict[str, Any] = {
        "id": machine_id,
        "label": label or platform.node(),
        "hardware_id": hardware_id,
        "os": {
            "family": platform.system().lower(),
            "name": platform.system(),
            "version": platform.version(),
            "arch": platform.machine(),
            "distribution": platform.platform(),
            "is_wsl": "microsoft" in platform.release().lower(),
        },
        "cpu": {
            "model": cpu_model,
            "arch": platform.machine(),
            "logical_cores": logical,
        },
        "gpus": gpus,
    }
    memory_mb = identity["memory_mb"]
    if memory_mb is not None:
        machine["memory"] = {"total_mb": memory_mb}
    return machine


def detect_libraries(packages: Iterable[str] = DEFAULT_SOFTWARE_PACKAGES) -> dict[str, Any]:
    """Capture Python and installed package versions in the standard format."""
    package_versions: dict[str, str] = {}
    for name in packages:
        try:
            package_versions[_distribution_name(name)] = version(name)
        except PackageNotFoundError:
            package_versions[_distribution_name(name)] = "not installed"
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": package_versions,
    }


def runner_identity(*, label: str | None = None, hardware_id: str | None = None) -> dict[str, Any]:
    """Return the runner identity used in a report."""
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    identity: dict[str, Any] = {
        "user": user,
        "label": label or user,
        "hostname": platform.node(),
    }
    if hardware_id:
        identity["hardware_id"] = hardware_id
    return identity


def environment_id(hardware: Mapping[str, Any], libraries: Mapping[str, Any]) -> str:
    """Stable hash of OS, GPU driver, runtime, and EP versions."""
    payload = {
        "os": hardware.get("os"),
        "gpus": hardware.get("gpus"),
        "libraries": libraries,
    }
    return "sha256:" + sha256_text(repr(payload))


__all__ = [
    "DEFAULT_SOFTWARE_PACKAGES",
    "detect_hardware",
    "detect_libraries",
    "environment_id",
    "runner_identity",
]
