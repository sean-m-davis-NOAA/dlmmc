"""Benchmark CmdStan grid throughput over explicitly selected MPI rank counts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "dlm_lat_alt_mpi_run.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--parallel-chains", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument(
        "--launcher-arg",
        action="append",
        default=[],
        help="Extra launcher argument; repeat for multiple arguments (for example, --launcher-arg=--mpi=pmix).",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, float, str]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, time.perf_counter() - started, result.stdout + result.stderr


def count_statuses(path: Path) -> dict[str, int]:
    with xr.open_dataset(path) as result:
        values, counts = np.unique(result["fit_status"].to_numpy(), return_counts=True)
    meanings = {-1: "failed", 0: "skipped", 1: "complete"}
    return {meanings[int(value)]: int(count) for value, count in zip(values, counts)}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if any(rank < 1 for rank in args.ranks):
        parser.error("all rank counts must be positive")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("rank counts must be unique")
    if args.cells < 1 or args.repeats < 1:
        parser.error("--cells and --repeats must be positive")
    if args.chains < 1 or not 1 <= args.parallel_chains <= args.chains:
        parser.error("require --chains >= 1 and 1 <= --parallel-chains <= --chains")
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")

    launcher = shutil.which(args.launcher)
    if launcher is None:
        parser.error(f"MPI launcher not found: {args.launcher}")
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / f"mpi-rank-sweep-{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, env.get("PYTHONPATH", "")) if item
    )

    precompile_command = [
        sys.executable,
        "-c",
        "import dlmmc; dlmmc.compile_models(['vanilla_ar1'], cache_dir=%r)"
        % (str(args.cache_dir) if args.cache_dir else None),
    ]
    return_code, precompile_seconds, precompile_log = run_command(
        precompile_command, REPO_ROOT, env
    )
    precompile_log_path = output_dir / "precompile.log"
    precompile_log_path.write_text(precompile_log)
    if return_code:
        raise SystemExit(
            f"CmdStan precompile failed; see {precompile_log_path} (exit {return_code})."
        )

    runner_args = [
        "--cell-count",
        str(args.cells),
        "--chains",
        str(args.chains),
        "--parallel-chains",
        str(args.parallel_chains),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
    ]
    if args.cache_dir is not None:
        runner_args.extend(["--cache-dir", str(args.cache_dir)])

    trials: list[dict[str, Any]] = []
    for ranks in sorted(args.ranks):
        fitting_workers = 1 if ranks == 1 else ranks - 1
        if fitting_workers > args.cells:
            print(
                f"Note: {ranks} ranks provide {fitting_workers} workers for only "
                f"{args.cells} cells; some workers will be idle."
            )
        for repeat in range(1, args.repeats + 1):
            label = f"ranks-{ranks}-repeat-{repeat}"
            output_path = output_dir / f"{label}.nc"
            log_path = output_dir / f"{label}.log"
            command = [
                launcher,
                *args.launcher_arg,
                "-n",
                str(ranks),
                sys.executable,
                str(RUNNER),
                "--output",
                str(output_path),
                *runner_args,
            ]
            return_code, elapsed, log = run_command(command, REPO_ROOT, env)
            log_path.write_text(log)
            trial: dict[str, Any] = {
                "ranks": ranks,
                "fitting_workers": fitting_workers,
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "return_code": return_code,
                "command": command,
                "log": str(log_path),
            }
            if return_code == 0 and output_path.exists():
                statuses = count_statuses(output_path)
                complete = statuses.get("complete", 0)
                trial["comparison_eligible"] = (
                    complete == args.cells
                    and statuses.get("failed", 0) == 0
                    and statuses.get("skipped", 0) == 0
                )
                trial["statuses"] = statuses
                trial["cells_per_hour"] = complete * 3600 / elapsed if elapsed else None
                trial["output"] = str(output_path)
            else:
                trial["error"] = "MPI run failed or did not create its NetCDF output."
            trials.append(trial)
            print(
                f"ranks={ranks} workers={fitting_workers} repeat={repeat} "
                f"elapsed={elapsed:.2f}s return_code={return_code} "
                f"statuses={trial.get('statuses', trial.get('error'))} "
                f"comparison_eligible={trial.get('comparison_eligible', False)}"
            )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        if trial.get("comparison_eligible"):
            grouped.setdefault(trial["ranks"], []).append(trial)
    rank_medians = {
        ranks: median(trial["elapsed_seconds"] for trial in runs)
        for ranks, runs in grouped.items()
    }
    best_ranks = min(rank_medians, key=rank_medians.get) if rank_medians else None
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "precompile_seconds_excluded_from_trials": precompile_seconds,
        "precompile_log": str(precompile_log_path),
        "cells": args.cells,
        "chains": args.chains,
        "parallel_chains": args.parallel_chains,
        "warmup": args.warmup,
        "samples": args.samples,
        "repeats": args.repeats,
        "rank_median_seconds": rank_medians,
        "fastest_measured_ranks_not_a_scientific_optimum": best_ranks,
        "trials": trials,
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"summary={report_path}")
    if best_ranks is not None:
        print(f"fastest_measured_total_ranks={best_ranks} (provisional; inspect repeats and diagnostics)")


if __name__ == "__main__":
    main()