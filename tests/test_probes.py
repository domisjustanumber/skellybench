"""Local and docker probes with fake systems."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from skellybench.probes import docker, local

PROBES = Path(__file__).resolve().parents[1] / "src" / "skellybench" / "probes"


def _runner(outputs: dict[str, str | None]):
    def run(command, **_kwargs):
        key = " ".join(command[:2])
        for prefix, output in outputs.items():
            if " ".join(command).startswith(prefix):
                if output is None:
                    return SimpleNamespace(returncode=1, stdout="")
                return SimpleNamespace(returncode=0, stdout=output)
        raise FileNotFoundError(key)

    return run


@pytest.mark.parametrize("name", ["local.py", "docker.py"])
def test_probe_files_use_standard_library_only(name: str) -> None:
    tree = ast.parse((PROBES / name).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{name} uses a relative import"
            modules.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert modules <= allowed, modules - allowed


def test_nvidia_smi_parsing_reads_uuid_and_bus_id() -> None:
    run = _runner({"nvidia-smi": "GPU-abc, 00000000:01:00.0, NVIDIA GeForce RTX 3070, 610.88, 8192\n"})
    assert local.nvidia_gpus(run=run) == [{
        "uuid": "GPU-abc", "vendor": "nvidia", "pci_bus_id": "00000000:01:00.0",
        "name": "NVIDIA GeForce RTX 3070", "driver": "610.88", "vram_mb": 8192,
    }]


def test_windows_snapshot_gives_gpus_cpu_and_vm() -> None:
    snapshot = {
        "video": [
            {"Name": "NVIDIA GeForce RTX 3070", "AdapterCompatibility": "NVIDIA", "PNPDeviceID": "PCI\\VEN_10DE"},
            {"Name": "Intel(R) UHD Graphics 770", "AdapterCompatibility": "Intel Corporation",
             "PNPDeviceID": "PCI\\VEN_8086&DEV_A780", "DriverVersion": "32.0.101.7088"},
            {"Name": "Parsec Virtual Display Adapter", "PNPDeviceID": "ROOT\\DISPLAY\\0000"},
        ],
        "system": {"Manufacturer": "Microsoft Corporation", "Model": "Virtual Machine"},
        "processor": [{"Name": "Intel CPU", "NumberOfCores": 20}],
        "processes": [],
    }
    run = _runner({"nvidia-smi": None})
    found = local.gpus(system="Windows", run=run, windows_info=snapshot)
    assert [gpu["name"] for gpu in found] == ["Intel(R) UHD Graphics 770"]
    assert local.virtualization(system="Windows", windows_info=snapshot) == {
        "is_wsl": False, "is_vm": True, "hypervisor": "hyper-v"}
    assert local.cpu_info(system="Windows", windows_info=snapshot)["physical_cores"] == 20


def test_windows_bare_metal_is_not_a_vm() -> None:
    snapshot = {"system": {"Manufacturer": "ASUS", "Model": "System Product Name"}}
    assert local.virtualization(system="Windows", windows_info=snapshot)["is_vm"] is False
    assert local.virtualization(system="Windows", windows_info={"system": {}})["is_vm"] is None


def test_windows_cim_snapshot_failure_gives_empty_values() -> None:
    def run(*_args, **_kwargs):
        raise TimeoutError("powershell timed out")

    snapshot = local.windows_cim_snapshot(run=_runner({}), pid=1)
    assert snapshot == {"video": [], "processor": [], "processes": [], "system": {}}


def test_linux_virtualization_and_wsl(tmp_path: Path) -> None:
    (tmp_path / "proc" / "sys" / "kernel").mkdir(parents=True)
    (tmp_path / "proc" / "sys" / "kernel" / "osrelease").write_text("5.15.0-microsoft-standard-WSL2")
    assert local.virtualization(system="Linux", root=tmp_path, run=_runner({})) == {
        "is_wsl": True, "is_vm": True, "hypervisor": "hyper-v"}
    (tmp_path / "proc" / "sys" / "kernel" / "osrelease").write_text("6.8.0-generic")
    assert local.virtualization(system="Linux", root=tmp_path, run=_runner({"systemd-detect-virt": "kvm"}))["hypervisor"] == "kvm"
    (tmp_path / "proc" / "cpuinfo").write_text("flags\t: fpu vme hypervisor\n")
    assert local.virtualization(system="Linux", root=tmp_path, run=_runner({"systemd-detect-virt": None}))["is_vm"] is True


def test_macos_vm_flag() -> None:
    assert local.virtualization(system="Darwin", run=_runner({"sysctl": "1"}))["is_vm"] is True
    assert local.virtualization(system="Darwin", run=_runner({}))["is_vm"] is None


def test_launch_command_finds_uv_grandparent_on_windows() -> None:
    snapshot = {"processes": [
        {"pid": 3, "ppid": 2, "name": "python.exe", "command": r'"C:\py\python.exe" export_all.py'},
        {"pid": 2, "ppid": 1, "name": "python.exe", "command": r'"C:\repo\.venv\Scripts\python.exe" export_all.py'},
        {"pid": 1, "ppid": 0, "name": "uv.exe",
         "command": r'"C:\Users\Dom\.local\bin\uv.exe" run python export_all.py --coco-root D:\data\coco --sizes nano'},
    ]}
    command = local.launch_command(system="Windows", windows_info=snapshot)
    assert command == "uv run python export_all.py --coco-root coco --sizes nano"


def test_launch_command_falls_back_to_orig_argv(tmp_path: Path) -> None:
    command = local.launch_command(system="Linux", root=tmp_path, orig_argv=["/usr/bin/python3", "/home/x/export_all.py", "--fp32"])
    assert command == "python export_all.py --fp32"


def test_launch_command_walks_linux_proc(tmp_path: Path) -> None:
    for pid, ppid, argv in ((30, 20, ["/venv/bin/python", "export_all.py"]), (20, 1, ["/home/u/.cargo/bin/uv", "run", "export_all.py"])):
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True)
        (directory / "stat").write_text(f"{pid} (x) S {ppid} 0 0")
        (directory / "cmdline").write_bytes(b"\0".join(token.encode() for token in argv) + b"\0")
    assert local.launch_command(system="Linux", root=tmp_path, pid=30) == "uv run export_all.py"


def test_loaded_libraries_use_module_lists_not_torch(tmp_path: Path) -> None:
    modules = ["/opt/lib/libcudnn.so.9", "/opt/lib/libcudart.so.13.0.48", "/opt/lib/libc.so.6"]
    found = local.loaded_native_libraries(system="Linux", modules=modules, query_versions=False)
    assert found == {"libcudart.so.13.0.48": "13.0.48", "libcudnn.so.9": "9"}


def test_loaded_libraries_resolve_soname_link(tmp_path: Path) -> None:
    real = tmp_path / "libcudnn.so.9.10.0"
    real.write_bytes(b"")
    link = tmp_path / "libcudnn.so.9"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks are not available")
    found = local.loaded_native_libraries(system="Linux", modules=[str(link)], query_versions=True)
    assert found == {"libcudnn.so.9": "9.10.0"}


class _Distribution:
    def __init__(self, name: str, version: str, direct_url: dict | None = None) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str):
        return json.dumps(self._direct_url) if filename == "direct_url.json" and self._direct_url else None


def test_package_sources_record_git_commit_and_hide_local_paths() -> None:
    distributions = [
        _Distribution("skellybench", "0.1.0", {"url": "https://github.com/x/skellybench.git",
                                                "vcs_info": {"vcs": "git", "commit_id": "7eb6bde"}}),
        _Distribution("My_Lib", "1.0", {"url": "file:///C:/Users/Dom/GitHub/my-lib", "dir_info": {"editable": True}}),
        _Distribution("numpy", "2.5.4"),
    ]
    assert local.installed_packages(distributions=distributions) == {"my-lib": "1.0", "numpy": "2.5.4", "skellybench": "0.1.0"}
    sources = local.package_sources(distributions=distributions)
    assert sources["skellybench"]["commit"] == "7eb6bde"
    assert sources["my-lib"] == {"local": "my-lib", "editable": True}
    assert "numpy" not in sources


def test_git_revision_hashes_a_dirty_diff() -> None:
    run = _runner({"git -C . rev-parse": "abc123", "git -C . status": " M file.py", "git -C . diff": "diff text",
                   "git -C . ls-files": "new.py"})
    revision = local.git_revision(".", run=run)
    assert revision["commit"] == "abc123" and revision["dirty"] is True and len(revision["diff_sha256"]) == 64
    assert local.git_revision(".", run=_runner({})) is None


def test_local_probe_functions_return_json_when_inputs_are_missing(tmp_path: Path) -> None:
    run = _runner({})
    values = [
        local.cpu_info(system="Linux", root=tmp_path, run=run),
        local.memory_total_mb(system="Linux", root=tmp_path, run=run),
        local.gpus(system="Linux", root=tmp_path, run=run),
        local.virtualization(system="Linux", root=tmp_path, run=run),
        local.loaded_native_libraries(system="Linux", root=tmp_path),
        local.file_sha256(tmp_path / "missing.lock"),
    ]
    json.dumps(values)


def _container_root(tmp_path: Path, *, v2: bool) -> Path:
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "1").mkdir(parents=True)
    container = "a" * 64
    (tmp_path / "proc" / "self" / "mountinfo").write_text(
        f"100 90 0:50 /var/lib/docker/containers/{container}/hostname /etc/hostname rw\n")
    (tmp_path / "proc" / "1" / "cgroup").write_text("0::/\n")
    (tmp_path / ".dockerenv").write_text("")
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "os-release").write_text('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n')
    cgroup = tmp_path / "sys" / "fs" / "cgroup"
    if v2:
        cgroup.mkdir(parents=True)
        (cgroup / "cpu.max").write_text("800000 100000\n")
        (cgroup / "cpuset.cpus.effective").write_text("0-7\n")
        (cgroup / "memory.max").write_text(str(32768 * 1024 * 1024))
    else:
        for name, value in (("cpu/cpu.cfs_quota_us", "800000"), ("cpu/cpu.cfs_period_us", "100000"),
                            ("cpuset/cpuset.cpus", "0-7"), ("memory/memory.limit_in_bytes", str(32768 * 1024 * 1024))):
            path = cgroup / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
    return tmp_path


def test_docker_probe_reads_container_block(tmp_path: Path) -> None:
    root = _container_root(tmp_path, v2=True)
    environ = {"SKELLYBENCH_IMAGE": "ghcr.io/x/exporter:1", "SKELLYBENCH_IMAGE_DIGEST": "sha256:" + "b" * 64,
               "SKELLYBENCH_HOST_HOSTNAME": "gerald", "NVIDIA_VISIBLE_DEVICES": "all", "CUDA_VERSION": "13.0.1"}
    block = docker.collect(root=root, environ=environ, system="Linux")
    assert block == {
        "runtime": "docker", "container_id": "a" * 64, "image": "ghcr.io/x/exporter:1",
        "image_digest": "sha256:" + "b" * 64, "base_os": "Ubuntu 24.04.1 LTS",
        "limits": {"cpu_quota_cores": 8.0, "cpuset": "0-7", "memory_limit_mb": 32768},
        "gpu_runtime": {"nvidia_visible_devices": "all", "nvidia_driver_capabilities": None, "image_cuda_version": "13.0.1"},
        "host_hostname": "gerald",
    }


def test_cgroup_v1_and_v2_give_the_same_limits(tmp_path: Path) -> None:
    v2 = docker.cgroup_limits(root=_container_root(tmp_path / "v2", v2=True))
    v1 = docker.cgroup_limits(root=_container_root(tmp_path / "v1", v2=False))
    assert v1 == v2
    unlimited = tmp_path / "unlimited" / "sys" / "fs" / "cgroup"
    unlimited.mkdir(parents=True)
    (unlimited / "cpu.max").write_text("max 100000")
    (unlimited / "memory.max").write_text("max")
    assert docker.cgroup_limits(root=tmp_path / "unlimited") == {"cpu_quota_cores": None, "cpuset": None, "memory_limit_mb": None}


def test_podman_containerenv_gives_image(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / ".containerenv").write_text('engine="podman-5"\nimage="quay.io/x/y:1"\nimageid="' + "c" * 64 + '"\n')
    assert docker.container_runtime(root=tmp_path, system="Linux") == "podman"
    assert docker.image(root=tmp_path, environ={}) == {"image": "quay.io/x/y:1", "image_digest": "sha256:" + "c" * 64}


def test_not_in_a_container(tmp_path: Path) -> None:
    (tmp_path / "proc" / "1").mkdir(parents=True)
    (tmp_path / "proc" / "1" / "cgroup").write_text("0::/init.scope\n")
    assert docker.collect(root=tmp_path, environ={}, system="Linux") is None
    assert docker.is_container(root=tmp_path, system="Linux") is False
    assert docker.collect(system="Windows", registry=lambda: None) is None
    assert docker.container_runtime(system="Windows", registry=lambda: 2) == "docker"
