"""Print hardware, scheduler, and Python accelerator information as JSON.

This probe is read-only and safe to run in a short interactive or batch job.
It does not print the full environment, credentials, or tokens.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


ENV_KEYS = (
    "TMPDIR",
    "SLURM_JOB_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "LOADEDMODULES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CONDA_PREFIX",
)
PACKAGES = (
    "cmdstanpy",
    "jax",
    "numpyro",
    "nutpie",
    "numpy",
    "pymc",
    "pymc-extras",
    "tensorflow",
    "tensorflow-metal",
    "tensorflow-probability",
    "tf-keras",
    "torch",
    "mpi4py",
)


def run_command(args: list[str], timeout: int = 15) -> dict[str, Any]:
    if shutil.which(args[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout[:1200].strip(),
            "stderr": result.stderr[:300].strip(),
        }
    except Exception as exc:  # noqa: BLE001 - report probe failure, never abort
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            versions[package] = f"error: {type(exc).__name__}: {exc}"
    return versions


def tensorflow_probe() -> dict[str, Any]:
    try:
        import tensorflow as tf

        devices = tf.config.list_logical_devices()
        result: dict[str, Any] = {
            "version": tf.__version__,
            "devices": [f"{device.name} ({device.device_type})" for device in devices],
        }
        accelerators = [device.name for device in devices if device.device_type == "GPU"]
        if accelerators:
            try:
                tf.config.set_soft_device_placement(False)
                with tf.device(accelerators[0]):
                    value = tf.linalg.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
                    value.numpy()
                result["matmul_device"] = value.device
                result["matmul_ok"] = True
            except Exception as exc:  # noqa: BLE001 - probe must continue
                result["matmul_ok"] = False
                result["matmul_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["matmul_ok"] = None
        return result
    except Exception as exc:  # noqa: BLE001 - optional framework
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def jax_probe() -> dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        result: dict[str, Any] = {
            "version": jax.__version__,
            "devices": [str(device) for device in devices],
        }
        accelerator = next((device for device in devices if device.platform != "cpu"), None)
        if accelerator is not None:
            try:
                with jax.default_device(accelerator):
                    value = jnp.matmul(jnp.ones((16, 16)), jnp.ones((16, 16)))
                    value.block_until_ready()
                result["matmul_device"] = str(value.device)
                result["matmul_ok"] = True
            except Exception as exc:  # noqa: BLE001 - probe must continue
                result["matmul_ok"] = False
                result["matmul_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["matmul_ok"] = None
        return result
    except Exception as exc:  # noqa: BLE001 - optional framework
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def torch_probe() -> dict[str, Any]:
    try:
        import torch

        result: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            try:
                value = torch.ones((16, 16), device="cuda") @ torch.ones(
                    (16, 16), device="cuda"
                )
                torch.cuda.synchronize()
                result["device_name"] = torch.cuda.get_device_name()
                result["matmul_device"] = str(value.device)
                result["matmul_ok"] = True
            except Exception as exc:  # noqa: BLE001 - probe must continue
                result["matmul_ok"] = False
                result["matmul_error"] = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - optional framework
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "environment": {key: os.environ.get(key) for key in ENV_KEYS if key in os.environ},
        "packages": package_versions(),
        "commands": {
            "nvidia_smi": run_command(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
            ),
            "rocminfo": run_command(["rocminfo"], timeout=30),
            "mpi_version": run_command(["mpirun", "--version"]),
        },
        "tensorflow": tensorflow_probe(),
        "jax": jax_probe(),
        "torch": torch_probe(),
    }
    try:
        from cmdstanpy import cmdstan_path

        report["cmdstan_path"] = cmdstan_path()
    except Exception as exc:  # noqa: BLE001 - CmdStan may not be installed yet
        report["cmdstan_path"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        from mpi4py import MPI

        report["mpi4py"] = {
            "version": importlib.metadata.version("mpi4py"),
            "library": MPI.Get_library_version().strip(),
            "world_size": MPI.COMM_WORLD.Get_size(),
            "world_rank": MPI.COMM_WORLD.Get_rank(),
        }
    except Exception as exc:  # noqa: BLE001 - optional MPI
        report["mpi4py"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
