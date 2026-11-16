#!/usr/bin/env python3
"""Compare fresh MPS32 and parallel CPU64 production fits of 312 BASIC cells."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

import dlmmc
from dlm_lat_alt_mpi_run import DATASET, _load_regressors
from dlmmc.profiles import ExecutionProfile


ROOT = Path(__file__).resolve().parent
MODEL = "vanilla_ar1"
PROFILE = dlmmc.PRODUCTION_PROFILE


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_basic_grid() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, dict]:
    """Load every non-placeholder cell in pressure-major, latitude-minor order."""
    with xr.open_dataset(DATASET, engine="netcdf4") as source:
        data = source["o3"].load()
        uncertainty = source["o3_sigma"].load()
    dates = pd.to_datetime(data.time.values, unit="D", origin="julian")
    data = data.assign_coords(time=dates)
    uncertainty = uncertainty.assign_coords(time=dates)
    regressors = _load_regressors(data.time, data.latitude.to_numpy())
    x = regressors.to_array("regressor").broadcast_like(data)
    ids = np.array([
        f"p{p:02d}-lat{lat:02d}"
        for p in range(data.sizes["pressure"])
        for lat in range(data.sizes["latitude"])
    ])
    y = data.stack(panel=("pressure", "latitude")).reset_index("panel")
    s = uncertainty.stack(panel=("pressure", "latitude")).reset_index("panel")
    x = x.stack(panel=("pressure", "latitude")).reset_index("panel")
    placeholders = ((y == 0) & (s == 0)).all("time").values
    keep = np.flatnonzero(~placeholders)
    if len(ids) != 324 or len(keep) != 312 or len(dates) != 396:
        raise ValueError("Expected BASIC 324 locations, 12 zero placeholders, 396 months")
    y = y.isel(panel=keep).assign_coords(panel=ids[keep]).transpose("time", "panel")
    s = s.isel(panel=keep).assign_coords(panel=ids[keep]).transpose("time", "panel")
    x = x.isel(panel=keep).assign_coords(panel=ids[keep]).transpose("panel", "time", "regressor")
    if not bool(np.isfinite(x).all()):
        raise ValueError("BASIC regressors must cover every observation month")
    if not (np.isfinite(y.values) & np.isfinite(s.values)).any(axis=0).all():
        raise ValueError("Every selected cell needs observations and uncertainties")
    provenance = {
        "panels": len(keep), "months": len(dates),
        "panel_ids": ids[keep].tolist(),
        "excluded_zero_placeholders": ids[placeholders].tolist(),
        "time_start": str(dates[0]), "time_end": str(dates[-1]),
        "regressors": list(regressors.data_vars),
        "input_sha256": {
            str(path): sha256(path)
            for path in [DATASET, *sorted(Path("regressors").glob("*.txt"))]
        },
    }
    return y, s, x, provenance


def sampling_timings(folder: Path, profile: ExecutionProfile, panels: int,
                     parameters: int) -> dict[str, float]:
    """Validate the entire checkpoint window and sum synchronized block timings."""
    target = profile.warmup + profile.retained_sweeps
    latest = read_json(folder / "latest.json")
    if (latest["status"] != "SAMPLING_COMPLETE_NOT_CONVERGENCE"
            or latest["iterations"] != [target] or latest["target_iteration"] != target):
        raise RuntimeError("Incomplete fit: refusing to report a completed benchmark")
    burnin = retained = 0.0
    cursor = 0
    for path in sorted((folder / "run-0").glob("block-*.npz")):
        _, start, end = path.stem.split("-")
        start, end = int(start), int(end)
        if start != cursor or not start < end <= target:
            raise ValueError(f"Noncontiguous sampling checkpoint: {path}")
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            steps = np.arange(max(start, profile.warmup) + 1, end + 1)
            draws = z["draws"]
            if (metadata["iteration"] != end or metadata["warmup"] != profile.warmup
                    or not np.array_equal(z["retained_iterations"], steps)
                    or draws.shape != (len(steps), 2 * profile.half_walkers, panels, parameters)
                    or not np.isfinite(draws).all() or not np.isfinite(z["draw_logp"]).all()):
                raise ValueError(f"Invalid retained checkpoint: {path}")
            seconds = float(z["sampling_seconds"])
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid sampling timing: {path}")
        if end <= profile.warmup:
            burnin += seconds
        elif start >= profile.warmup:
            retained += seconds
        else:
            raise ValueError("A block straddles burn-in; separate timings are unavailable")
        cursor = end
    if cursor != target:
        raise RuntimeError("Incomplete checkpoint sequence")
    return {
        "burnin_seconds": burnin,
        "retained_sampling_seconds": retained,
        "sampling_including_burnin_seconds": burnin + retained,
    }


def flag_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(flag for row in rows for flag in row["flags"]))


def run_backend(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    y, s, x, inputs = load_basic_grid()
    loading = time.perf_counter() - started
    print(f"{args.backend}: {inputs['panels']} cells x {inputs['months']} months", flush=True)
    dtype = "float32" if args.backend == "mps" else "float64"
    selection = {
        "scope": "independent_production", "explicit_user_response": args.approval,
        "panel_ids": inputs["panel_ids"], "model": MODEL, "backend": args.backend,
        "dtype": dtype, "profile": PROFILE.to_dict(),
    }
    write_json(folder / "inputs.json", inputs)
    write_json(folder / "selection.json", selection)
    before = time.perf_counter()
    fit = dlmmc.fit_production(
        y, x, uncertainty=s, backend=args.backend, model=MODEL, profile=PROFILE,
        selection=selection, output_dir=folder / "fit", seed=args.seed,
        numba_threads=args.numba_threads, wall_seconds=args.wall_seconds,
    )
    if args.backend == "mps":
        torch.mps.synchronize()
    fit_wall = time.perf_counter() - before
    if (fit.prepared.cells != 312 or str(fit.prepared.y.dtype) != f"torch.{dtype}"
            or fit.prepared.y.device.type != ("mps" if args.backend == "mps" else "cpu")
            or fit.config["threads"] != {"torch": 1, "numba": args.numba_threads}):
        raise RuntimeError("Actual fit layout, precision or threads differ from requested settings")
    sample = sampling_timings(fit.path, PROFILE, 312, len(fit.prepared.parameter_names))
    compilation = read_json(fit.path / "compilation.json")
    initialization = read_json(fit.path / "initialization.json")
    before = time.perf_counter()
    diagnostics = dlmmc.diagnose_production(fit)
    write_json(folder / "diagnostics.json", diagnostics)
    monitoring = time.perf_counter() - before
    before = time.perf_counter()
    output = dlmmc.summarize_production_trends(
        fit, output_dir=folder / "trends", anchor_month="1998-01",
        seed=args.seed + 1, numba_threads=args.numba_threads,
        wall_seconds=args.wall_seconds,
    )
    if args.backend == "mps":
        torch.mps.synchronize()
    output_wall = time.perf_counter() - before
    if (output["panels"] != 312 or output["full_batch_shape"] != [2048, 312]
            or output["parameter_iterations"] != [1, 256, 128, 312]
            or output["cap_exceeded"]):
        raise RuntimeError("Incomplete or unexpected production trend output")
    if [row["panel"] for row in output["cells"]] != inputs["panel_ids"]:
        raise RuntimeError("Trend output panel order differs from input")
    for row in output["cells"]:
        path = folder / "trends" / f"cell-{row['cell']:04d}.npz"
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Output checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            for name in ("mean", "sd", "change_mean", "change_sd"):
                if z[name].shape != (396,) or not np.isfinite(z[name]).all():
                    raise ValueError(f"Invalid {name}: {path}")
    elapsed = time.perf_counter() - started
    compile_seconds = compilation["compilation_or_cache_load_seconds"]
    warm_seconds = compilation["disposable_exact_shape_warmup_seconds"]
    output_compile = output["compilation_or_cache_load_seconds"]
    output_warm = output["disposable_exact_shape_warmup_seconds"]
    result = {
        "status": "COMPLETE_BENCHMARK_WITH_DIAGNOSTIC_FLAGS",
        "qualification_eligible": False, "backend": args.backend, "dtype": dtype,
        "profile": PROFILE.to_dict(), "model": MODEL, "seed": args.seed,
        "output_seed": args.seed + 1, "threads": fit.config["threads"],
        "inputs": inputs, "prepared_checksum": fit.prepared.checksum,
        "fit_provenance": fit.config["provenance"],
        "timings": {
            **sample, "input_loading_seconds": loading, "fit_wall_seconds": fit_wall,
            "initialization_seconds": initialization["seconds"],
            "fit_compilation_or_cache_load_seconds": compile_seconds,
            "fit_disposable_warmup_seconds": warm_seconds,
            "monitoring_seconds": monitoring, "trend_wall_seconds": output_wall,
            "trend_compilation_or_cache_load_seconds": output_compile,
            "trend_disposable_warmup_seconds": output_warm,
            "trend_compute_seconds": output["smoothing_and_reduction_seconds"],
            "trend_post_compile_seconds": output["post_compile_output_seconds"],
            "pipeline_wall_seconds": elapsed,
            "pipeline_post_compile_seconds": elapsed - compile_seconds - warm_seconds - output_compile - output_warm,
        },
        "parameter_flag_counts": flag_counts(diagnostics["cells"]),
        "trend_flag_counts": flag_counts(output["cells"]),
        "unsuccessful_optimizer_cells": [
            row["cell"] for row in initialization["optimizations"] if not row["success"]
        ],
    }
    write_json(folder / "timing.json", result)
    print(json.dumps(result["timings"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--approval", required=True, help="Actual user approval for this full-grid comparison")
    parser.add_argument("--numba-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026111801)
    parser.add_argument("--wall-seconds", type=float, default=5400, help="Cap for each fit and each trend summary")
    parser.add_argument("--backend", choices=("mps", "numba"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.approval.strip():
        parser.error("--approval must record explicit full-grid permission")
    import numba
    if not 1 <= args.numba_threads <= numba.config.NUMBA_NUM_THREADS:
        parser.error("--numba-threads exceeds the available Numba thread pool")
    if not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error("--wall-seconds must be finite and positive")
    if not 0 <= args.seed < 2**32 - 10000:
        parser.error("--seed must be between 0 and 2**32 - 10001")
    args.output_dir = args.output_dir.resolve()
    os.chdir(ROOT)
    if args.backend:
        run_backend(args)
        return
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Native MPS with torch.mps.compile_shader is required; no CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0")
    record = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "interpreter": sys.executable, "python": sys.version,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "logical_cpus": os.cpu_count(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("dlmmc", "affine", "torch", "numpy", "scipy", "numba", "pandas", "xarray", "netCDF4")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": sha256(Path(__file__)),
        "mps_environment": {name: env[name] for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")},
        "interpretation": (
            "One sequential run per backend, same model/input/budget, MPS32 versus CPU64. "
            "Not equal-precision convergence or speedup at matched ESS. Existing compiler caches may be reused. "
            "Pipeline includes input loading, fit/checkpoints, monitoring, trend output and verification; no plotting. "
            "Post-compile timings exclude reported compilation/cache loading and disposable warmups, not burn-in."
        ),
        "runs": {},
    }
    report_path = args.output_dir / "comparison.json"
    write_json(report_path, record)
    for backend in ("mps", "numba"):
        log_path = args.output_dir / f"{backend}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--backend", backend, "--output-dir", str(args.output_dir / backend),
            "--approval", args.approval, "--numba-threads", str(args.numba_threads),
            "--seed", str(args.seed), "--wall-seconds", str(args.wall_seconds),
        ]
        print(f"Starting {backend}; progress log: {log_path}", flush=True)
        before = time.perf_counter()
        with log_path.open("x") as log:
            try:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                record["status"] = f"FAILED_{backend.upper()}"
                write_json(report_path, record)
                print(f"{backend} failed; see {log_path}", file=sys.stderr, flush=True)
                raise
        seconds = time.perf_counter() - before
        result = read_json(args.output_dir / backend / "timing.json")
        result["timings"]["process_wall_seconds"] = seconds
        record["runs"][backend] = result
        write_json(report_path, record)
        print(f"Completed {backend} in {seconds:.2f}s including process startup", flush=True)
    cpu, gpu = record["runs"]["numba"], record["runs"]["mps"]
    if cpu["inputs"] != gpu["inputs"] or cpu["prepared_checksum"] != gpu["prepared_checksum"]:
        raise RuntimeError("CPU and MPS inputs differ; timing comparison rejected")
    record["cpu_over_mps"] = {
        key: cpu["timings"][key] / gpu["timings"][key]
        for key in ("sampling_including_burnin_seconds", "trend_post_compile_seconds",
                    "pipeline_post_compile_seconds", "process_wall_seconds")
    }
    record["status"] = "COMPLETE_BENCHMARK_WITH_DIAGNOSTIC_FLAGS"
    write_json(report_path, record)
    print("\nPhase                                  CPU64(s)   MPS32(s)   CPU/MPS")
    for key, ratio in record["cpu_over_mps"].items():
        print(f"{key:38s} {cpu['timings'][key]:9.2f} {gpu['timings'][key]:10.2f} {ratio:9.2f}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
