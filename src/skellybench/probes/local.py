"""Probes for this machine and its Python environment.

This module uses the Python standard library only.
It does not import other skellybench modules, so you can copy it into another project.
Each public function does one probe and returns plain JSON-ready data.
A probe that cannot run returns ``None`` for its value. It does not raise for a missing tool or file.
The output contains no absolute local paths.

Public functions:

- ``hostname()``: the network name of the machine. All platforms.
- ``os_info()``: OS name, version, distribution, and architecture. All platforms.
- ``cpu_info()``: CPU name and core counts. Windows, Linux, macOS.
- ``memory_total_mb()``: total physical memory. Windows, Linux, macOS.
- ``gpus()``: GPUs with UUID, PCI bus ID, name, driver, and memory. NVIDIA through ``nvidia-smi``.
  Other adapters through WMI (Windows), ``/sys/class/drm`` (Linux), or ``system_profiler`` (macOS).
- ``cuda_devices()``: CUDA devices in CUDA enumeration order, with index, UUID, name, and PCI bus ID.
  Uses the CUDA driver API through ``ctypes``. No torch. Windows and Linux.
- ``virtualization()``: WSL and virtual machine detection. Windows, Linux, macOS.
- ``python_info()``: interpreter implementation, version, and build.
- ``installed_packages()``: every installed distribution and its version.
- ``package_sources()``: the PEP 610 source of each package that was not installed from an index.
- ``loaded_native_libraries(patterns)``: native libraries loaded into this process, with versions.
  Windows, Linux, macOS.
- ``git_revision(path)``: commit, dirty flag, and diff hash of a git checkout.
- ``file_sha256(path)``: SHA-256 of one file, for example a lock file.
- ``launch_command()``: the command line that started this process, with a ``uv`` launcher if one
  started it. Windows, Linux, macOS.
- ``environment_variable(name)``: the value of one environment variable.
- ``collect()``: all the probes above in one mapping.

Optional arguments let a test replace the system: ``root`` (filesystem root), ``run`` (command
runner with the ``subprocess.run`` signature), ``environ`` (environment mapping), ``system``
(``platform.system()`` value), and ``windows_info`` (the parsed Windows CIM snapshot).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

Runner = Callable[..., Any]

DEFAULT_NATIVE_PATTERNS: tuple[str, ...] = (
    "cudart",
    "cudnn",
    "cublas",
    "nvinfer",
    "tensorrt_rtx",
    "openvino",
)
"""Library name prefixes that :func:`loaded_native_libraries` reports by default."""

UV_LAUNCHERS = frozenset({"uv", "uvx"})
WINDOWS_CIM_TIMEOUT_SECONDS = 15.0
_VENDORS_BY_PCI_ID = {"0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd", "0x106b": "apple"}
_HYPERVISOR_VENDORS: tuple[tuple[str, str], ...] = (
    ("vmware", "vmware"),
    ("virtualbox", "virtualbox"),
    ("innotek", "virtualbox"),
    ("qemu", "kvm"),
    ("kvm", "kvm"),
    ("xen", "xen"),
    ("parallels", "parallels"),
    ("amazon ec2", "kvm"),
    ("google compute engine", "kvm"),
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")
_SONAME_VERSION = re.compile(r"\.so\.(\d+(?:\.\d+)*)$")

_windows_info_cache: dict[int, dict[str, Any]] = {}


# --- helpers ---------------------------------------------------------------------------------------


def _system(system: str | None) -> str:
    return system if system is not None else platform.system()


def _run_text(run: Runner | None, command: Sequence[str], *, timeout: float = 5.0) -> str | None:
    runner = run or subprocess.run
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    output = getattr(completed, "stdout", None)
    return output.strip() if isinstance(output, str) else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_absolute_path(token: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE.match(token)) or token.startswith("/")


def _basename(token: str) -> str:
    return re.split(r"[\\/]", token.rstrip("\\/"))[-1] or token


def reduce_path_token(token: str) -> str:
    """Return *token* with an absolute path (or ``--opt=path`` value) reduced to its basename."""
    stripped = token.strip('"')
    if _is_absolute_path(stripped):
        return _basename(stripped)
    left, separator, right = stripped.partition("=")
    if separator and left and _is_absolute_path(right.strip('"')):
        return f"{left}={_basename(right.strip(chr(34)))}"
    return stripped


# --- identity and OS -------------------------------------------------------------------------------


def hostname() -> str | None:
    """Return the network name of this machine."""
    name = platform.node()
    return name or None


def os_info(*, system: str | None = None) -> dict[str, Any]:
    """Return OS name, version, distribution string, and architecture."""
    name = _system(system)
    return {
        "name": name or None,
        "version": platform.version() or None,
        "release": platform.release() or None,
        "distribution": platform.platform() or None,
        "arch": platform.machine() or None,
    }


# --- Windows CIM snapshot --------------------------------------------------------------------------

_WINDOWS_CIM_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$chain = @()
$p = __PID__
for ($i = 0; $i -lt 8; $i++) {
  $x = Get-CimInstance Win32_Process -Filter "ProcessId=$p"
  if (-not $x) { break }
  $chain += [pscustomobject]@{ pid = $x.ProcessId; ppid = $x.ParentProcessId; name = $x.Name; command = $x.CommandLine }
  if ($x.ParentProcessId -eq $p -or $x.ParentProcessId -eq 0) { break }
  $p = $x.ParentProcessId
}
[pscustomobject]@{
  video = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, PNPDeviceID, AdapterCompatibility, AdapterRAM)
  system = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model
  processor = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
  processes = $chain
} | ConvertTo-Json -Depth 4 -Compress
"""


def windows_cim_snapshot(*, run: Runner | None = None, pid: int | None = None) -> dict[str, Any]:
    """Return one cached CIM snapshot: video adapters, computer system, processors, process chain.

    The standard library has no WMI access, so this makes one PowerShell call with a timeout.
    The result is cached for the process. On failure it returns an empty mapping.
    """
    start_pid = pid if pid is not None else os.getpid()
    if run is None and start_pid in _windows_info_cache:
        return _windows_info_cache[start_pid]
    script = _WINDOWS_CIM_SCRIPT.replace("__PID__", str(int(start_pid)))
    output = None
    for shell in ("powershell", "pwsh"):
        output = _run_text(
            run,
            [shell, "-NoProfile", "-NonInteractive", "-Command", script],
            timeout=WINDOWS_CIM_TIMEOUT_SECONDS,
        )
        if output:
            break
    snapshot: dict[str, Any] = {}
    if output:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            snapshot = parsed
    for key in ("video", "processor", "processes"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            snapshot[key] = [value]
        elif not isinstance(value, list):
            snapshot[key] = []
    if not isinstance(snapshot.get("system"), dict):
        snapshot["system"] = {}
    if run is None:
        _windows_info_cache[start_pid] = snapshot
    return snapshot


def _windows_snapshot(windows_info: Mapping[str, Any] | None, run: Runner | None) -> Mapping[str, Any]:
    return windows_info if windows_info is not None else windows_cim_snapshot(run=run)


# --- CPU and memory --------------------------------------------------------------------------------


def _windows_cpu_name() -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def cpu_info(
    *,
    system: str | None = None,
    root: Path = Path("/"),
    run: Runner | None = None,
    windows_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the CPU name and physical/logical core counts."""
    name: str | None = None
    physical: int | None = None
    logical = os.cpu_count()
    current = _system(system)
    if current == "Windows":
        name = _windows_cpu_name()
        processors = _windows_snapshot(windows_info, run).get("processor") or []
        cores = [_int_or_none(item.get("NumberOfCores")) for item in processors if isinstance(item, dict)]
        if cores and all(value is not None for value in cores):
            physical = sum(value for value in cores if value is not None)
        if name is None and processors and isinstance(processors[0], dict):
            name = str(processors[0].get("Name") or "").strip() or None
    elif current == "Darwin":
        name = _run_text(run, ["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = _int_or_none(_run_text(run, ["sysctl", "-n", "hw.physicalcpu"]))
    else:
        text = _read_text(root / "proc" / "cpuinfo")
        if text:
            cores: set[tuple[str, str]] = set()
            physical_id = core_id = ""
            for line in text.splitlines():
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "model name" and name is None:
                    name = value or None
                elif key == "physical id":
                    physical_id = value
                elif key == "core id":
                    core_id = value
                elif not line.strip():
                    if core_id:
                        cores.add((physical_id, core_id))
                    physical_id = core_id = ""
            if core_id:
                cores.add((physical_id, core_id))
            physical = len(cores) or None
    if name is None:
        name = platform.processor() or None
    return {"name": name, "physical_cores": physical, "logical_cores": logical}


def memory_total_mb(*, system: str | None = None, root: Path = Path("/"), run: Runner | None = None) -> int | None:
    """Return total physical memory in MiB."""
    current = _system(system)
    if current == "Windows":
        try:
            import ctypes

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
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys // (1024 * 1024))
        except (OSError, AttributeError, ImportError):
            return None
        return None
    if current == "Darwin":
        value = _int_or_none(_run_text(run, ["sysctl", "-n", "hw.memsize"]))
        return value // (1024 * 1024) if value else None
    text = _read_text(root / "proc" / "meminfo")
    if text:
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
        if match:
            return int(match.group(1)) // 1024
    return None


# --- GPUs ------------------------------------------------------------------------------------------


def nvidia_gpus(*, run: Runner | None = None) -> list[dict[str, Any]]:
    """Return NVIDIA GPUs from ``nvidia-smi``, or an empty list."""
    output = _run_text(
        run,
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )
    gpus: list[dict[str, Any]] = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        uuid, bus_id, name, driver, memory = parts[:5]
        gpus.append(
            {
                "uuid": uuid or None,
                "vendor": "nvidia",
                "pci_bus_id": bus_id.lower() if bus_id else None,
                "name": name or None,
                "driver": driver or None,
                "vram_mb": _int_or_none(memory),
            }
        )
    return gpus


def _vendor_from_name(name: str) -> str | None:
    lowered = name.lower()
    for vendor in ("nvidia", "intel", "amd", "apple"):
        if vendor in lowered:
            return vendor
    if "radeon" in lowered or "advanced micro devices" in lowered:
        return "amd"
    return None


def _windows_other_gpus(windows_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for adapter in windows_info.get("video") or []:
        if not isinstance(adapter, dict):
            continue
        name = str(adapter.get("Name") or "").strip()
        compatibility = str(adapter.get("AdapterCompatibility") or "")
        vendor = _vendor_from_name(f"{compatibility} {name}")
        if vendor == "nvidia" or not name:
            continue
        # Virtual and remote display adapters have a ROOT\ or SWD\ device ID, not PCI\.
        if not str(adapter.get("PNPDeviceID") or "").upper().startswith("PCI\\"):
            continue
        found.append(
            {
                "uuid": str(adapter.get("PNPDeviceID") or "") or None,
                "vendor": vendor,
                "pci_bus_id": None,
                "name": name,
                "driver": str(adapter.get("DriverVersion") or "") or None,
                "vram_mb": None,
            }
        )
    return found


def _linux_other_gpus(root: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    drm = root / "sys" / "class" / "drm"
    try:
        cards = sorted(path for path in drm.iterdir() if re.fullmatch(r"card\d+", path.name))
    except OSError:
        return found
    for card in cards:
        device = card / "device"
        vendor_id = (_read_text(device / "vendor") or "").strip().lower()
        vendor = _VENDORS_BY_PCI_ID.get(vendor_id)
        if vendor in (None, "nvidia"):
            continue
        device_id = (_read_text(device / "device") or "").strip().lower()
        uevent = _read_text(device / "uevent") or ""
        slot = re.search(r"^PCI_SLOT_NAME=(.+)$", uevent, re.MULTILINE)
        driver = re.search(r"^DRIVER=(.+)$", uevent, re.MULTILINE)
        bus_id = slot.group(1).strip().lower() if slot else None
        found.append(
            {
                "uuid": f"pci:{bus_id}" if bus_id else None,
                "vendor": vendor,
                "pci_bus_id": bus_id,
                "name": f"{vendor.upper() if vendor == 'amd' else vendor.title()} GPU [{vendor_id[2:]}:{device_id[2:]}]",
                "driver": driver.group(1).strip() if driver else None,
                "vram_mb": None,
            }
        )
    return found


def _macos_gpus(run: Runner | None) -> list[dict[str, Any]]:
    output = _run_text(run, ["system_profiler", "SPDisplaysDataType", "-json"], timeout=20.0)
    if not output:
        return []
    try:
        displays = json.loads(output).get("SPDisplaysDataType") or []
    except (json.JSONDecodeError, AttributeError):
        return []
    found = []
    for display in displays:
        if not isinstance(display, dict):
            continue
        name = str(display.get("sppci_model") or display.get("_name") or "").strip()
        if not name:
            continue
        found.append(
            {
                "uuid": None,
                "vendor": _vendor_from_name(name) or "apple",
                "pci_bus_id": None,
                "name": name,
                "driver": None,
                "vram_mb": None,
            }
        )
    return found


def gpus(
    *,
    system: str | None = None,
    root: Path = Path("/"),
    run: Runner | None = None,
    windows_info: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return every GPU found. NVIDIA GPUs come first, in ``nvidia-smi`` order."""
    current = _system(system)
    found = nvidia_gpus(run=run)
    if current == "Windows":
        found.extend(_windows_other_gpus(_windows_snapshot(windows_info, run)))
    elif current == "Darwin":
        found.extend(_macos_gpus(run))
    else:
        found.extend(_linux_other_gpus(root))
    return found


# --- virtualization --------------------------------------------------------------------------------


def _hypervisor_from_text(text: str) -> str | None:
    lowered = text.lower()
    if "microsoft corporation" in lowered and "virtual machine" in lowered:
        return "hyper-v"
    for needle, name in _HYPERVISOR_VENDORS:
        if needle in lowered:
            return name
    return None


def virtualization(
    *,
    system: str | None = None,
    root: Path = Path("/"),
    run: Runner | None = None,
    windows_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``is_wsl``, ``is_vm`` and ``hypervisor``. A value is ``None`` when the check cannot run."""
    current = _system(system)
    result: dict[str, Any] = {"is_wsl": False, "is_vm": None, "hypervisor": None}
    if current == "Windows":
        info = _windows_snapshot(windows_info, run).get("system") or {}
        if not info:
            return result
        # Do not use HypervisorPresent: it is true on bare metal with Hyper-V or VBS turned on.
        hypervisor = _hypervisor_from_text(f"{info.get('Manufacturer', '')} {info.get('Model', '')}")
        result.update(is_vm=hypervisor is not None, hypervisor=hypervisor)
        return result
    if current == "Darwin":
        value = _run_text(run, ["sysctl", "-n", "kern.hv_vmm_present"])
        if value is not None:
            result["is_vm"] = value.strip() == "1"
            result["hypervisor"] = "unknown" if result["is_vm"] else None
        return result

    release = (_read_text(root / "proc" / "sys" / "kernel" / "osrelease") or platform.release()).lower()
    result["is_wsl"] = "microsoft" in release
    if result["is_wsl"]:
        wsl2 = "wsl2" in release or "microsoft-standard" in release
        result.update(is_vm=wsl2, hypervisor="hyper-v" if wsl2 else None)
        return result
    # systemd-detect-virt exits 1 and prints "none" on bare metal, so a failed call falls through
    # to the DMI and CPU-flag checks.
    detected = _run_text(run, ["systemd-detect-virt", "--vm"])
    if detected:
        result.update(is_vm=detected != "none", hypervisor=None if detected == "none" else detected)
        return result
    cpuinfo = _read_text(root / "proc" / "cpuinfo")
    vendor = _read_text(root / "sys" / "class" / "dmi" / "id" / "sys_vendor") or ""
    product = _read_text(root / "sys" / "class" / "dmi" / "id" / "product_name") or ""
    hypervisor = _hypervisor_from_text(f"{vendor} {product}")
    if hypervisor is not None:
        result.update(is_vm=True, hypervisor=hypervisor)
    elif cpuinfo is not None:
        flagged = bool(re.search(r"^flags\s*:.*\bhypervisor\b", cpuinfo, re.MULTILINE))
        result.update(is_vm=flagged, hypervisor="unknown" if flagged else None)
    return result


# --- Python environment ----------------------------------------------------------------------------


def _distribution_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_info() -> dict[str, Any]:
    """Return interpreter implementation, version, and build string."""
    build_number, build_date = platform.python_build()
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "build": f"{build_number}, {build_date}",
        "compiler": platform.python_compiler() or None,
    }


def _distributions(distributions: Iterable[Any] | None) -> list[Any]:
    return list(distributions) if distributions is not None else list(metadata.distributions())


def installed_packages(*, distributions: Iterable[Any] | None = None) -> dict[str, str]:
    """Return every installed distribution as ``{normalized name: version}``, sorted by name."""
    packages: dict[str, str] = {}
    for distribution in _distributions(distributions):
        name = distribution.metadata.get("Name") if distribution.metadata else None
        if not name:
            continue
        packages[_distribution_key(str(name))] = str(distribution.version)
    return dict(sorted(packages.items()))


def package_sources(*, distributions: Iterable[Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the PEP 610 source of each package that was not installed from an index.

    Git installs record the URL and commit. Local and editable installs record the folder basename,
    never the local path.
    """
    sources: dict[str, dict[str, Any]] = {}
    for distribution in _distributions(distributions):
        name = distribution.metadata.get("Name") if distribution.metadata else None
        if not name:
            continue
        try:
            text = distribution.read_text("direct_url.json")
        except (OSError, UnicodeDecodeError):
            text = None
        if not text:
            continue
        try:
            direct = json.loads(text)
        except json.JSONDecodeError:
            continue
        url = str(direct.get("url") or "")
        entry: dict[str, Any] = {}
        vcs = direct.get("vcs_info")
        if isinstance(vcs, dict):
            entry.update(
                vcs=vcs.get("vcs"),
                url=url,
                commit=vcs.get("commit_id"),
                requested_revision=vcs.get("requested_revision"),
            )
        elif url.startswith("file:"):
            entry["local"] = _basename(url.removeprefix("file://").removeprefix("file:"))
            if (direct.get("dir_info") or {}).get("editable"):
                entry["editable"] = True
        else:
            entry["url"] = url
            archive = direct.get("archive_info")
            if isinstance(archive, dict) and archive.get("hashes"):
                entry["hashes"] = archive.get("hashes")
        sources[_distribution_key(str(name))] = {key: value for key, value in entry.items() if value is not None}
    return dict(sorted(sources.items()))


# --- loaded native libraries -----------------------------------------------------------------------


def _windows_loaded_modules() -> list[str]:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return []
    try:
        psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    process = kernel32.GetCurrentProcess()
    count = 1024
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            return []
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64
    total = needed.value // ctypes.sizeof(wintypes.HMODULE)
    buffer = ctypes.create_unicode_buffer(32768)
    paths: list[str] = []
    for module in modules[:total]:
        if module and kernel32.GetModuleFileNameW(module, buffer, len(buffer)):
            paths.append(buffer.value)
    return paths


def _windows_file_version(path: str) -> str | None:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        version_dll = ctypes.WinDLL("version")  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return None
    version_dll.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    version_dll.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    version_dll.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    version_dll.GetFileVersionInfoW.restype = wintypes.BOOL
    version_dll.VerQueryValueW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.UINT),
    ]
    version_dll.VerQueryValueW.restype = wintypes.BOOL
    size = version_dll.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not version_dll.GetFileVersionInfoW(path, 0, size, data):
        return None
    pointer = ctypes.c_void_p()
    length = wintypes.UINT()
    if not version_dll.VerQueryValueW(data, "\\", ctypes.byref(pointer), ctypes.byref(length)) or not pointer:
        return None

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in (
            "dwSignature", "dwStrucVersion", "dwFileVersionMS", "dwFileVersionLS",
            "dwProductVersionMS", "dwProductVersionLS", "dwFileFlagsMask", "dwFileFlags",
            "dwFileOS", "dwFileType", "dwFileSubtype", "dwFileDateMS", "dwFileDateLS",
        )]

    info = ctypes.cast(pointer, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    if info.dwSignature != 0xFEEF04BD:
        return None
    parts = (
        info.dwFileVersionMS >> 16,
        info.dwFileVersionMS & 0xFFFF,
        info.dwFileVersionLS >> 16,
        info.dwFileVersionLS & 0xFFFF,
    )
    return ".".join(str(part) for part in parts)


def _linux_loaded_modules(root: Path) -> list[str]:
    text = _read_text(root / "proc" / "self" / "maps")
    if not text:
        return []
    paths: list[str] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and ".so" in fields[5]:
            path = fields[5].strip()
            if path not in paths:
                paths.append(path)
    return paths


def _macos_loaded_modules() -> list[str]:
    try:
        import ctypes

        system = ctypes.CDLL(None)
        system._dyld_image_count.restype = ctypes.c_uint32
        system._dyld_get_image_name.restype = ctypes.c_char_p
        system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return [
            system._dyld_get_image_name(index).decode("utf-8", "replace")
            for index in range(system._dyld_image_count())
        ]
    except (OSError, AttributeError):
        return []


def _decode_cuda_version(value: int, *, major_factor: int = 1000) -> str:
    major, rest = divmod(value, major_factor)
    return f"{major}.{rest // 10}"


def _query_library_version(path: str, stem: str) -> str | None:
    """Ask a loaded library for its own version through ``ctypes``. Return ``None`` on failure."""
    try:
        import ctypes

        library = ctypes.CDLL(path)
    except (OSError, ImportError):
        return None
    try:
        if stem.startswith("cudart") and hasattr(library, "cudaRuntimeGetVersion"):
            value = ctypes.c_int()
            if library.cudaRuntimeGetVersion(ctypes.byref(value)) == 0 and value.value > 0:
                return _decode_cuda_version(value.value)
        elif stem.startswith("cudnn") and hasattr(library, "cudnnGetVersion"):
            library.cudnnGetVersion.restype = ctypes.c_size_t
            value = int(library.cudnnGetVersion())
            if value >= 90000:
                major, rest = divmod(value, 10000)
                minor, patch = divmod(rest, 100)
                return f"{major}.{minor}.{patch}"
            if value > 0:
                major, rest = divmod(value, 1000)
                minor, patch = divmod(rest, 100)
                return f"{major}.{minor}.{patch}"
        elif stem.startswith("cublas") and hasattr(library, "cublasGetProperty"):
            parts = []
            for property_type in (0, 1, 2):
                value = ctypes.c_int()
                if library.cublasGetProperty(property_type, ctypes.byref(value)) != 0:
                    return None
                parts.append(str(value.value))
            return ".".join(parts)
        elif stem.startswith("nvinfer") and hasattr(library, "getInferLibVersion"):
            library.getInferLibVersion.restype = ctypes.c_int32
            value = int(library.getInferLibVersion())
            if value > 0:
                major, rest = divmod(value, 10000)
                minor, patch = divmod(rest, 100)
                return f"{major}.{minor}.{patch}"
    except (OSError, ValueError, AttributeError):
        return None
    return None


def loaded_native_libraries(
    patterns: Sequence[str] = DEFAULT_NATIVE_PATTERNS,
    *,
    system: str | None = None,
    root: Path = Path("/"),
    modules: Sequence[str] | None = None,
    query_versions: bool = True,
) -> dict[str, str | None]:
    """Return ``{library basename: version}`` for loaded libraries whose name starts with a pattern.

    The version comes from, in order: the library's own version function, the Windows file version
    resource, the real file name behind a Linux soname link, and otherwise ``None``.
    Call this after the runtime has loaded its libraries (for example after the first inference).
    """
    current = _system(system)
    if modules is None:
        if current == "Windows":
            modules = _windows_loaded_modules()
        elif current == "Darwin":
            modules = _macos_loaded_modules()
        else:
            modules = _linux_loaded_modules(root)
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    found: dict[str, str | None] = {}
    for path in modules:
        name = _basename(path)
        lowered = name.lower().removeprefix("lib")
        if not lowered.startswith(lowered_patterns):
            continue
        version: str | None = None
        if query_versions:
            version = _query_library_version(path, lowered)
        if version is None and current == "Windows" and query_versions:
            version = _windows_file_version(path)
        if version is None:
            real = os.path.realpath(path) if query_versions else path
            match = _SONAME_VERSION.search(_basename(real)) or _SONAME_VERSION.search(name)
            if match:
                version = match.group(1)
        found[name] = version
    return dict(sorted(found.items()))


# --- source revision and files ---------------------------------------------------------------------


def git_revision(path: str | Path, *, run: Runner | None = None) -> dict[str, Any] | None:
    """Return the commit, dirty flag, and diff hash of the git checkout at *path*.

    The diff hash covers ``git diff HEAD`` plus the names of untracked files. It is ``None`` for a
    clean tree. Return ``None`` when *path* is not in a git checkout.
    """
    folder = str(path)
    commit = _run_text(run, ["git", "-C", folder, "rev-parse", "HEAD"])
    if not commit:
        return None
    status = _run_text(run, ["git", "-C", folder, "status", "--porcelain"])
    dirty = bool(status)
    diff_sha256 = None
    if dirty:
        diff = _run_text(run, ["git", "-C", folder, "diff", "HEAD"], timeout=30.0) or ""
        untracked = _run_text(run, ["git", "-C", folder, "ls-files", "--others", "--exclude-standard"]) or ""
        diff_sha256 = hashlib.sha256(f"{diff}\n--untracked--\n{untracked}".encode()).hexdigest()
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha256}


def file_sha256(path: str | Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def environment_variable(name: str, *, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the value of one environment variable, or ``None``."""
    value = (environ if environ is not None else os.environ).get(name)
    return value or None


# --- launch command --------------------------------------------------------------------------------


def _process_name(command_or_name: str) -> str:
    name = _basename(command_or_name.strip().strip('"')).lower()
    return name.removesuffix(".exe")


def _split_command(command: str, *, system: str) -> list[str]:
    if system == "Windows":
        return [token.strip('"') for token in re.findall(r'"[^"]*"|\S+', command)]
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _format_command(tokens: Sequence[str]) -> str:
    reduced = [reduce_path_token(token) for token in tokens]
    if reduced and reduced[0].lower().endswith(".exe"):
        reduced[0] = reduced[0][: -len(".exe")]
    return shlex.join(reduced)


def _linux_process_chain(start: int, root: Path, max_depth: int) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    pid = start
    for _ in range(max_depth + 1):
        stat = _read_text(root / "proc" / str(pid) / "stat")
        raw = None
        try:
            raw = (root / "proc" / str(pid) / "cmdline").read_bytes()
        except OSError:
            pass
        if not stat or raw is None:
            break
        ppid = _int_or_none(stat.rpartition(")")[2].split()[1]) if ")" in stat else None
        tokens = [token.decode("utf-8", "replace") for token in raw.split(b"\0") if token]
        chain.append({"pid": pid, "ppid": ppid, "name": tokens[0] if tokens else "", "tokens": tokens})
        if not ppid or ppid == pid:
            break
        pid = ppid
    return chain


def _macos_process_chain(start: int, run: Runner | None, max_depth: int) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    pid = start
    for _ in range(max_depth + 1):
        output = _run_text(run, ["ps", "-o", "ppid=,command=", "-p", str(pid)])
        if not output:
            break
        ppid_text, _, command = output.strip().partition(" ")
        tokens = _split_command(command.strip(), system="Darwin")
        ppid = _int_or_none(ppid_text)
        chain.append({"pid": pid, "ppid": ppid, "name": tokens[0] if tokens else "", "tokens": tokens})
        if not ppid or ppid == pid:
            break
        pid = ppid
    return chain


def _windows_process_chain(windows_info: Mapping[str, Any], max_depth: int) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    for process in (windows_info.get("processes") or [])[: max_depth + 1]:
        if not isinstance(process, dict):
            continue
        command = str(process.get("command") or "")
        chain.append(
            {
                "pid": process.get("pid"),
                "ppid": process.get("ppid"),
                "name": str(process.get("name") or ""),
                "tokens": _split_command(command, system="Windows") if command else [],
            }
        )
    return chain


def launch_command(
    *,
    max_depth: int = 5,
    system: str | None = None,
    root: Path = Path("/"),
    run: Runner | None = None,
    windows_info: Mapping[str, Any] | None = None,
    orig_argv: Sequence[str] | None = None,
    pid: int | None = None,
) -> str:
    """Return the full command line that started this process, as one string.

    The function walks up the process tree, at most *max_depth* levels, to the nearest ``uv`` or
    ``uvx`` process, and returns that command line. On Windows the chain is ``uv.exe`` -> venv
    ``python.exe`` launcher -> real ``python.exe``, so ``uv`` is the grandparent.
    Without a ``uv`` ancestor, it returns ``python`` plus ``sys.orig_argv``.
    Absolute paths are reduced to their basename.
    """
    current = _system(system)
    start = pid if pid is not None else os.getpid()
    if current == "Windows":
        info = windows_info if windows_info is not None else windows_cim_snapshot(run=run, pid=start)
        chain = _windows_process_chain(info, max_depth)
    elif current == "Darwin":
        chain = _macos_process_chain(start, run, max_depth)
    else:
        chain = _linux_process_chain(start, root, max_depth)
    for process in chain[1:]:
        if _process_name(process["name"]) in UV_LAUNCHERS and process["tokens"]:
            return _format_command(process["tokens"])
    argv = list(orig_argv if orig_argv is not None else getattr(sys, "orig_argv", sys.argv))
    return _format_command(["python", *argv[1:]])


# --- CUDA driver API -------------------------------------------------------------------------------

CudaLoader = Callable[[str], Any]
_CUDA_NAME_BYTES = 256


def _load_cuda_library(file_name: str, system: str) -> Any:
    import ctypes

    return ctypes.WinDLL(file_name) if system == "Windows" else ctypes.CDLL(file_name)


def _format_cuda_uuid(raw: bytes) -> str:
    text = raw.hex()
    return f"GPU-{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def cuda_devices(*, loader: CudaLoader | None = None, system: str | None = None) -> list[dict[str, Any]] | None:
    """Return the CUDA devices in CUDA enumeration order, or ``None`` without a working driver.

    Each entry is ``{"cuda_index", "uuid", "name", "pci_bus_id"}``. The UUID is ``"GPU-xxxxxxxx-..."``,
    the same text as ``nvidia-smi``. The probe calls ``cuInit``, ``cuDeviceGetCount``, ``cuDeviceGet``,
    ``cuDeviceGetUuid_v2`` (``cuDeviceGetUuid`` on an old driver), ``cuDeviceGetName`` and
    ``cuDeviceGetPCIBusId`` from ``nvcuda.dll`` (Windows) or ``libcuda.so.1`` (Linux).
    ``None`` means the library is missing or ``cuInit`` failed. A value that one call cannot give is
    ``None``. The probe never raises.

    Args:
        loader: a function that takes the library file name and returns a library object. A test
            passes a fake. The default loads the real library with ``ctypes``.
        system: a ``platform.system()`` value.
    """
    import ctypes

    current = _system(system)
    if current not in ("Windows", "Linux"):
        return None
    file_name = "nvcuda.dll" if current == "Windows" else "libcuda.so.1"
    try:
        library = loader(file_name) if loader is not None else _load_cuda_library(file_name, current)
        if library is None or library.cuInit(0) != 0:
            return None
        count = ctypes.c_int(0)
        if library.cuDeviceGetCount(ctypes.pointer(count)) != 0:
            return None
        devices: list[dict[str, Any]] = []
        for index in range(count.value):
            handle = ctypes.c_int(0)
            if library.cuDeviceGet(ctypes.pointer(handle), index) != 0:
                devices.append({"cuda_index": index, "uuid": None, "name": None, "pci_bus_id": None})
                continue
            raw_uuid = (ctypes.c_ubyte * 16)()
            uuid_call = getattr(library, "cuDeviceGetUuid_v2", None) or getattr(library, "cuDeviceGetUuid", None)
            uuid = None
            if uuid_call is not None and uuid_call(raw_uuid, handle) == 0:
                uuid = _format_cuda_uuid(bytes(raw_uuid))
            name_buffer = ctypes.create_string_buffer(_CUDA_NAME_BYTES)
            name = None
            if library.cuDeviceGetName(name_buffer, _CUDA_NAME_BYTES, handle) == 0:
                name = name_buffer.value.decode("utf-8", "replace").strip() or None
            bus_buffer = ctypes.create_string_buffer(_CUDA_NAME_BYTES)
            bus_id = None
            if library.cuDeviceGetPCIBusId(bus_buffer, _CUDA_NAME_BYTES, handle) == 0:
                bus_id = bus_buffer.value.decode("ascii", "replace").strip().lower() or None
            devices.append({"cuda_index": index, "uuid": uuid, "name": name, "pci_bus_id": bus_id})
        return devices
    except Exception:  # noqa: BLE001 - a probe never raises
        return None


# --- collect ---------------------------------------------------------------------------------------


def collect(
    *,
    repository: str | Path | None = None,
    lock_file: str | Path | None = None,
    system: str | None = None,
    root: Path = Path("/"),
    run: Runner | None = None,
    environ: Mapping[str, str] | None = None,
    windows_info: Mapping[str, Any] | None = None,
    cuda_loader: CudaLoader | None = None,
) -> dict[str, Any]:
    """Run every probe in this module and return one mapping.

    Args:
        repository: optional git checkout for :func:`git_revision`.
        lock_file: optional lock file for :func:`file_sha256`. Only the hash is returned.
    """
    current = _system(system)
    info = windows_info
    if current == "Windows" and info is None:
        info = windows_cim_snapshot(run=run)
    return {
        "hostname": hostname(),
        "os": os_info(system=current),
        "cpu": cpu_info(system=current, root=root, run=run, windows_info=info),
        "memory_total_mb": memory_total_mb(system=current, root=root, run=run),
        "gpus": gpus(system=current, root=root, run=run, windows_info=info),
        "virtualization": virtualization(system=current, root=root, run=run, windows_info=info),
        "python": python_info(),
        "packages": installed_packages(),
        "package_sources": package_sources(),
        "git": git_revision(repository, run=run) if repository is not None else None,
        "lock_file_sha256": file_sha256(lock_file) if lock_file is not None else None,
        "command": launch_command(system=current, root=root, run=run, windows_info=info),
        "cuda_device_order": environment_variable("CUDA_DEVICE_ORDER", environ=environ),
        "cuda_devices": cuda_devices(loader=cuda_loader, system=current),
    }


__all__ = [
    "DEFAULT_NATIVE_PATTERNS",
    "collect",
    "cpu_info",
    "cuda_devices",
    "environment_variable",
    "file_sha256",
    "git_revision",
    "gpus",
    "hostname",
    "installed_packages",
    "launch_command",
    "loaded_native_libraries",
    "memory_total_mb",
    "nvidia_gpus",
    "os_info",
    "package_sources",
    "python_info",
    "reduce_path_token",
    "virtualization",
    "windows_cim_snapshot",
]
