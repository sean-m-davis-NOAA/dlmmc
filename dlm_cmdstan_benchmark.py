#!/usr/bin/env python3
"""Measure saturated CmdStan subset throughput; estimate, never fit, 312 cells."""

from __future__ import annotations

import os

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "STAN_NUM_THREADS",
)
if __name__ == "__main__":
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

import dlmmc
from dlm_grid_benchmark import ROOT, MODEL, load_basic_grid, read_json, sha256, write_json
from dlmmc.diagnostics import rank_split_rhat


PARAMETERS = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


def select_cells(total: int, cells: int, seed: int) -> np.ndarray:
    """Choose one cell per flat-order stratum, before inspecting any fit results."""
    if not 1 <= cells < total:
        raise ValueError("Select a proper subset, not the full grid")
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(group) for group in np.array_split(np.arange(total), cells)])


def extrapolate(batch_seconds: float, task_seconds: list[float],
                stan_seconds: list[float], *, cells: int, workers: int,
                setup_seconds: float, summary_seconds: float = 0, total: int = 312) -> dict:
    if (not 1 <= cells < total or workers < 1 or not task_seconds
            or len(task_seconds) != len(stan_seconds)
            or len(task_seconds) % cells
            or not np.isfinite([batch_seconds, setup_seconds, summary_seconds, *task_seconds, *stan_seconds]).all()
            or min(batch_seconds, *task_seconds, *stan_seconds) <= 0
            or min(setup_seconds, summary_seconds) < 0):
        raise ValueError("Complete positive subset timings required")
    scale = total / cells
    return {
        "kind": "EXTRAPOLATED_NOT_MEASURED_FULL_GRID",
        "full_grid_cells": total, "subset_cells": cells, "scale": scale,
        "batch_scaled_seconds": batch_seconds * scale,
        "batch_scaled_plus_one_setup_seconds": batch_seconds * scale + setup_seconds,
        "pipeline_scaled_plus_one_setup_seconds": (batch_seconds + summary_seconds) * scale + setup_seconds,
        "ideal_load_balance_seconds": sum(task_seconds) * scale / workers,
        "stan_reported_time_at_ideal_load_balance_seconds": sum(stan_seconds) * scale / workers,
        "busy_worker_fraction": sum(task_seconds) / (workers * batch_seconds),
        "assumptions": (
            "The selected cells represent full-grid costs at the same worker count. "
            "Batch scaling repeats the observed startup/drain penalty; ideal load balance "
            "assumes every worker stays occupied. Neither is a confidence bound or a "
            "convergence-qualified speedup. No 312-cell CmdStan fit was run."
        ),
    }


def run_chain(index: int, panel: str, chain: int, y, s, x,
              args: argparse.Namespace, epoch: float, cache: Path) -> dict:
    import cmdstanpy

    started = time.perf_counter()
    folder = args.output_dir / panel / f"chain-{chain}"
    folder.mkdir(parents=True, exist_ok=False)
    seed = args.seed + index * args.chains + chain
    before = time.perf_counter()
    posterior = dlmmc.fit(
        y.isel(panel=index), x.isel(panel=index), uncertainty=s.isel(panel=index),
        model=MODEL, chains=1, chain_ids=[chain], parallel_chains=1,
        threads_per_chain=1, force_one_process_per_chain=True,
        warmup=args.warmup, samples=args.samples, seed=seed,
        adapt_delta=0.8, max_treedepth=10, show_progress=False, refresh=100,
        cache_dir=cache, output_dir=folder / "stan", timeout=args.chain_timeout,
    )
    fit_seconds = time.perf_counter() - before
    if dict(posterior.sizes) != {"chain": 1, "draw": args.samples, "time": 396, "regressor": 5}:
        raise RuntimeError(f"Unexpected posterior dimensions for {panel}, chain {chain}")
    csv_files = list((folder / "stan").glob("*.csv"))
    if len(csv_files) != 1:
        raise RuntimeError("Exactly one completed CSV required per single-chain task")
    stan = cmdstanpy.from_csv(csv_files, method="sample")
    if not isinstance(stan, cmdstanpy.CmdStanMCMC):
        raise RuntimeError("Expected completed CmdStan MCMC output")
    if stan.num_draws_sampling != args.samples or stan.chains != 1:
        raise RuntimeError("Incomplete CmdStan retained window")
    method = stan.method_variables()
    if not all(np.isfinite(value).all() for value in method.values()):
        raise ValueError("Nonfinite CmdStan sampler diagnostics")
    params = np.column_stack([posterior[name].values[0] for name in PARAMETERS])
    trend = posterior["trend"].values[0]
    anchor = np.flatnonzero(posterior.time.values.astype("datetime64[M]") == np.datetime64("1998-01"))
    if len(anchor) != 1 or not np.isfinite(params).all() or not np.isfinite(trend).all():
        raise ValueError("Finite parameter/trend draws and the January 1998 anchor required")
    change = trend - trend[:, anchor]
    path = folder / "summary.npz"
    np.savez(path, parameters=params, time=posterior.time.values,
             mean=trend.mean(0), variance=trend.var(0),
             change_mean=change.mean(0), change_variance=change.var(0))
    result = {
        "panel": panel, "index": index, "chain": chain, "seed": seed,
        "draws": args.samples, "fit_seconds": fit_seconds,
        "stan_time": stan.time[0],
        "divergences": int(method["divergent__"].sum()),
        "max_treedepth_hits": int((method["treedepth__"] >= 10).sum()),
        "mean_leapfrog_steps": float(method["n_leapfrog__"].mean()),
        "start_seconds": started - epoch,
        "end_seconds": time.perf_counter() - epoch,
        "summary_sha256": sha256(path),
        "csv": str(csv_files[0].relative_to(args.output_dir)),
        "csv_bytes": csv_files[0].stat().st_size,
    }
    result["task_seconds"] = result["end_seconds"] - result["start_seconds"]
    write_json(folder / "timing.json", result)
    return result


def summarize_cells(args: argparse.Namespace, panel_ids: list[str]) -> list[dict]:
    rows = []
    for panel in panel_ids:
        stats = []
        for chain in range(1, args.chains + 1):
            with np.load(args.output_dir / panel / f"chain-{chain}/summary.npz") as z:
                stats.append({key: z[key].copy() for key in z.files})
        means = np.stack([z["mean"] for z in stats])
        changes = np.stack([z["change_mean"] for z in stats])
        mean, change = means.mean(0), changes.mean(0)
        sd = np.sqrt(np.mean([z["variance"] + (z["mean"] - mean)**2 for z in stats], axis=0))
        change_sd = np.sqrt(np.mean([
            z["change_variance"] + (z["change_mean"] - change)**2 for z in stats
        ], axis=0))
        np.savez(args.output_dir / panel / "trends.npz", time=stats[0]["time"],
                 mean=mean, sd=sd, change_mean=change, change_sd=change_sd)
        rhat = None
        if args.chains >= 2 and args.samples >= 4:
            rhat = rank_split_rhat(np.stack([z["parameters"] for z in stats]))
            if not np.isfinite(rhat).all():
                raise ValueError(f"Undefined parameter Rhat for {panel}")
        rows.append({
            "panel": panel, "parameters": list(PARAMETERS),
            "rank_split_rhat": None if rhat is None else rhat.tolist(),
            "rhat_over_1_01": None if rhat is None else bool((rhat > 1.01).any()),
            "qualification_eligible": False,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--affine-report", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026111601)
    parser.add_argument("--chain-timeout", type=float, default=1800,
                        help="Seconds per chain; timeout fails the benchmark, never scales partial results")
    args = parser.parse_args()
    if not 1 <= args.cells < 312 or args.chains < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("Positive subset/chains/samples and nonnegative warmup required")
    if not 1 <= args.workers <= min(os.cpu_count() or 1, args.cells * args.chains):
        parser.error("Workers must not exceed available CPUs or chain tasks")
    if not 0 <= args.seed < 2**32 - 312 * args.chains:
        parser.error("Seed outside supported per-chain range")
    if not np.isfinite(args.chain_timeout) or args.chain_timeout <= 0:
        parser.error("Chain timeout must be finite and positive")
    args.output_dir = args.output_dir.resolve()
    args.affine_report = args.affine_report.resolve()
    os.chdir(ROOT)
    import cmdstanpy
    import psutil

    started = time.perf_counter()
    baseline = read_json(args.affine_report)
    if baseline["status"] != "COMPLETE_BENCHMARK_WITH_DIAGNOSTIC_FLAGS":
        raise ValueError("Completed affine baseline required")
    y, s, x, inputs = load_basic_grid()
    for backend in ("numba", "mps"):
        if inputs != baseline["runs"][backend]["inputs"] or baseline["runs"][backend]["model"] != MODEL:
            raise ValueError("BASIC data/model differs from the affine baseline")
    indices = select_cells(312, args.cells, args.seed)
    panels = [inputs["panel_ids"][int(i)] for i in indices]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache = args.output_dir / "stan-cache"
    report = {
        "status": "PREPARING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "interpreter": sys.executable,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "logical_cpus": os.cpu_count(), "workers": args.workers,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        "cmdstan_path": cmdstanpy.cmdstan_path(), "cmdstan_version": cmdstanpy.cmdstan_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("dlmmc", "cmdstanpy", "numpy", "pandas", "xarray", "psutil")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": sha256(Path(__file__)), "inputs": inputs,
        "selection_method": "One seeded random cell per contiguous flat-order stratum",
        "selected_indices": indices.tolist(), "selected_panel_ids": panels,
        "settings": {
            "model": MODEL, "chains_per_cell": args.chains, "warmup_per_chain": args.warmup,
            "samples_per_chain": args.samples, "adapt_delta": 0.8, "max_treedepth": 10,
            "parallel_chains_per_task": 1, "threads_per_chain": 1, "seed": args.seed,
            "chain_timeout": args.chain_timeout, "dtype": "float64",
        },
        "affine_report_sha256": sha256(args.affine_report),
        "affine_baseline": {
            backend: {key: baseline["runs"][backend][key] for key in
                      ("model", "profile", "dtype", "threads", "timings", "parameter_flag_counts")}
            for backend in ("numba", "mps")
        },
        "interpretation": (
            "CmdStan public API default budget versus fixed-budget affine production, not matched ESS. "
            "Stan retained-step timing includes conditional state draws/generated quantities and CSV output; "
            "affine computes analytic trend moments separately. Thread workers launch independent single-threaded "
            "Stan processes, not MPI; this omits MPI rank-0 NetCDF aggregation. No full-grid Stan fit."
        ),
        "tasks": [], "concurrency_samples": [],
    }
    report_path = args.output_dir / "comparison.json"
    write_json(report_path, report)
    print(f"Preselected cells: {panels}", flush=True)
    before = time.perf_counter()
    executable = Path(dlmmc.compile_models([MODEL], cache_dir=cache)[MODEL]).resolve()
    report["compile_seconds"] = time.perf_counter() - before
    report["executable_sha256"] = sha256(executable)
    report["stan_source_sha256"] = sha256(executable.with_suffix(".stan"))
    report["setup_seconds"] = time.perf_counter() - started
    report["status"] = "RUNNING"
    write_json(report_path, report)
    process = psutil.Process()
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_chain, int(index), panel, chain, y, s, x, args, batch_start, cache)
            for chain in range(1, args.chains + 1)
            for index, panel in zip(indices, panels)
        }
        try:
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                active = 0
                for child in process.children(recursive=True):
                    try:
                        active += Path(child.exe()).resolve() == executable
                    except psutil.NoSuchProcess:
                        continue
                report["concurrency_samples"].append({
                    "seconds": time.perf_counter() - batch_start, "active_stan_processes": active,
                })
                for future in done:
                    result = future.result()
                    report["tasks"].append(result)
                    print(f"Completed {result['panel']} chain {result['chain']}: "
                          f"{result['task_seconds']:.1f}s, {result['divergences']} divergences", flush=True)
                write_json(report_path, report)
        except Exception as error:
            report["status"] = "FAILED_NO_EXTRAPOLATION"
            report["error"] = f"{type(error).__name__}: {error}"
            write_json(report_path, report)
            for future in pending:
                future.cancel()
            raise
    batch = time.perf_counter() - batch_start
    expected = {(panel, chain) for panel in panels for chain in range(1, args.chains + 1)}
    if {(row["panel"], row["chain"]) for row in report["tasks"]} != expected:
        raise RuntimeError("Incomplete chain set; extrapolation refused")
    before = time.perf_counter()
    report["panel_diagnostics"] = summarize_cells(args, panels)
    report["summary_seconds"] = time.perf_counter() - before
    report["batch_seconds"] = batch
    report["observed_peak_stan_processes"] = max(
        row["active_stan_processes"] for row in report["concurrency_samples"]
    )
    report["extrapolation"] = extrapolate(
        batch, [row["task_seconds"] for row in report["tasks"]],
        [row["stan_time"]["total"] for row in report["tasks"]],
        cells=args.cells, workers=args.workers, setup_seconds=report["setup_seconds"],
        summary_seconds=report["summary_seconds"],
    )
    report["subset_whole_process_seconds"] = time.perf_counter() - started
    report["status"] = "COMPLETE_SUBSET_WITH_DIAGNOSTIC_FLAGS"
    write_json(report_path, report)
    print(json.dumps({
        key: report[key] for key in ("batch_seconds", "observed_peak_stan_processes", "extrapolation")
    }, indent=2), flush=True)
    print(f"Full report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
