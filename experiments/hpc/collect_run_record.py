"""Append a run-record row for a finished HERA performance job.

Combines ``runs/<job>/run-metadata.txt``, ``sacct`` accounting and the grid
NetCDF into one row matching ``hera_cpu_mpi_run_record.csv.example``. Per-cell
R-hat/ESS rows are also written to ``runs/<job>/cell-diagnostics.csv``.

``wall_seconds`` and ``peak_memory_mib`` come from the longest ``srun`` step,
which is the grid fit; environment setup and the MPI check are excluded.
``ess_per_second`` is the sum over fitted cells of each cell's minimum bulk
ESS divided by that wall time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = Path("/scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work")
OUTPUT_NAMES = ("hera-mpi-perf.nc", "hera-mpi-smoke.nc")
MODEL_NAME = "vanilla_ar1"


def _load_cell_diagnostics():
    spec = importlib.util.spec_from_file_location("cell_diagnostics", HERE / "cell_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_fields() -> list[str]:
    with (HERE / "hera_cpu_mpi_run_record.csv.example").open() as handle:
        return next(csv.reader(handle))


def parse_metadata(text: str) -> dict[str, object]:
    """Parse key=value lines plus the git status, module, and version blocks."""
    meta: dict[str, object] = {}
    lines = text.splitlines()
    block = None
    git_status: list[str] = []
    modules: list[str] = []
    for line in lines:
        if line in ("git_status:", "module_list:", "srun_mpi_plugins:"):
            block = line[:-1]
            continue
        if "=" in line and not line.startswith(" ") and line.split("=", 1)[0].isidentifier():
            block = None
            key, value = line.split("=", 1)
            meta.setdefault(key, value)
            continue
        head, _, rest = line.partition(" ")
        if head in ("python", "cmdstanpy", "cmdstan_version", "mpi4py", "mpi_library"):
            block = None
            meta[head] = rest.strip()
            continue
        if block == "git_status" and line.strip():
            git_status.append(line)
        elif block == "module_list":
            modules.extend((int(n), name) for n, name in re.findall(r"(\d+)\)\s+(\S+)", line))
    meta["git_dirty"] = bool(git_status)
    meta["loaded_modules"] = " ".join(name for _, name in sorted(modules))
    return meta


def parse_rss_mib(value: str) -> float | None:
    if not value:
        return None
    units = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0**2}
    if value[-1] in units:
        return float(value[:-1]) * units[value[-1]]
    return float(value) / 1024**2


def sacct_rows(job_id: str) -> list[dict[str, str]]:
    fields = ["JobID", "JobName", "State", "Start", "ElapsedRaw", "NNodes", "NodeList", "NCPUS", "MaxRSS"]
    result = subprocess.run(
        ["sacct", "-j", job_id, "-P", "-n", "--format=" + ",".join(fields)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [dict(zip(fields, line.split("|"))) for line in result.stdout.splitlines() if line]


def fit_step(rows: list[dict[str, str]]) -> dict[str, str]:
    steps = [r for r in rows if "." in r["JobID"] and r["JobID"].rsplit(".", 1)[-1].isdigit()]
    if not steps:
        raise ValueError("No numbered srun steps found in sacct output")
    return max(steps, key=lambda r: int(r["ElapsedRaw"] or 0))


def worker_cell_counts(cell_ids: list[int], ntasks: int) -> list[int]:
    """Cells per fitting rank when fit_grid_mpi deals the cells round-robin in flat order.

    Runs made before resume support (commit d4b116f) assigned ``flat_index % workers``
    instead; for the evenly spread cell sets in TESTING.md the counts are identical,
    and for the 312-cell full grid they differ by at most one cell per worker.
    """
    if ntasks == 1:
        return [len(cell_ids)]
    workers = ntasks - 1
    counts = Counter(position % workers for position in range(len(set(cell_ids))))
    return [counts.get(w, 0) for w in range(workers)]


def _summary(values: np.ndarray, worst: str) -> str:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return ""
    edge = values.max() if worst == "max" else values.min()
    return f"{worst}={edge:.4g};median={np.median(values):.4g}"


def build_record(job_id: str, work_root: Path, configuration: str | None) -> tuple[dict[str, object], list[dict]]:
    run_dir = work_root / "runs" / job_id
    meta = parse_metadata((run_dir / "run-metadata.txt").read_text())
    output = next((run_dir / n for n in OUTPUT_NAMES if (run_dir / n).exists()), None)
    if output is None:
        raise FileNotFoundError(f"No grid NetCDF in {run_dir}")

    rows = sacct_rows(job_id)
    job = next(r for r in rows if r["JobID"] == job_id)
    step = fit_step(rows)
    wall = int(step["ElapsedRaw"])
    peak = max((parse_rss_mib(r["MaxRSS"]) or 0.0) for r in rows)

    diag = _load_cell_diagnostics()
    with xr.open_dataset(output) as ds:
        status = Counter(int(v) for v in ds["fit_status"].values.ravel())
        cells = diag.cell_diagnostics(ds, job_id)
    for row in cells:
        row["run_id"] = job_id

    def column(name):
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in cells])

    rhat, bulk, tail = column("max_rhat"), column("min_ess_bulk"), column("min_ess_tail")
    divergences = column("divergences")
    divergence_summary = (
        f"total={int(np.nansum(divergences))};cells_with_divergences={int(np.sum(divergences > 0))}"
        if np.isfinite(divergences).any() else ""
    )
    fitted = status.get(1, 0)
    ntasks = int(meta["ntasks"])
    cell_ids = [int(c) for c in str(meta["cell_indices"]).split()]
    per_worker = worker_cell_counts(cell_ids, ntasks)

    cache_state = ""
    cache_dir = Path(str(meta.get("cache_dir", "")))
    executables = [p for p in cache_dir.glob(f"*{MODEL_NAME}-*") if p.suffix != ".stan"] if cache_dir.is_dir() else []
    if executables and job["Start"] not in ("", "Unknown", "None"):
        start = datetime.fromisoformat(job["Start"]).timestamp()
        cache_state = "warm" if min(p.stat().st_mtime for p in executables) < start else "cold"

    record = {
        "run_id": job_id,
        "job_id": job_id,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_state": job["State"],
        "configuration": configuration or job["JobName"],
        "git_commit": meta.get("git_commit", ""),
        "git_dirty": meta["git_dirty"],
        "node_count": job["NNodes"],
        "allocated_nodes": job["NodeList"],
        "ntasks": ntasks,
        "cpus_per_task": meta.get("cpus_per_task", ""),
        "allocated_cpus": job["NCPUS"],
        "model": meta.get("model", ""),
        "priors": "default",
        "cell_ids": " ".join(map(str, cell_ids)),
        "chains": meta.get("chains", ""),
        "parallel_chains": meta.get("parallel_chains", ""),
        "warmup": meta.get("warmup", ""),
        "samples_per_chain": meta.get("samples", ""),
        "seed": meta.get("seed", ""),
        "python_version": str(meta.get("python", "")).split(" ")[0],
        "cmdstan_version": meta.get("cmdstan_version", ""),
        "cmdstanpy_version": meta.get("cmdstanpy", ""),
        "mpi4py_version": meta.get("mpi4py", ""),
        "mpi_library": meta.get("mpi_library", ""),
        "srun_mpi_plugin": meta.get("srun_mpi_plugin", ""),
        "loaded_modules": meta["loaded_modules"],
        "compile_cache_state": cache_state,
        "wall_seconds": wall,
        "cells_per_hour": f"{fitted / wall * 3600:.3f}" if wall else "",
        "fit_status_counts": ";".join(f"{k}:{v}" for k, v in sorted(status.items())),
        "divergences": divergence_summary,
        "rhat_summary": _summary(rhat, "max"),
        "ess_bulk_summary": _summary(bulk, "min"),
        "ess_tail_summary": _summary(tail, "min"),
        "ess_per_second": f"{np.nansum(bulk) / wall:.4g}" if wall and np.isfinite(bulk).any() else "",
        "peak_memory_mib": f"{peak:.0f}",
        "rank_utilization": (
            f"cells_per_fitting_rank={'/'.join(map(str, per_worker))};"
            f"busy_fraction={sum(per_worker) / (max(per_worker) * ntasks):.3f}"
        ),
        "output_bytes": output.stat().st_size,
        "output_path": str(output),
        "tmpdir": meta.get("tmpdir", ""),
        "notes": (
            f"job_elapsed_s={job['ElapsedRaw']}; no_progress={meta.get('no_progress', '0')}"
            + ("" if divergence_summary else "; divergences not stored in grid NetCDF")
        ),
    }
    return record, cells


def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="+")
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("DLMMC_WORK_ROOT", DEFAULT_WORK_ROOT)))
    parser.add_argument("--configuration", help="Label for a single job; defaults to the Slurm job name")
    parser.add_argument("--output", type=Path, required=True, help="Run record CSV to append to")
    args = parser.parse_args(argv)
    if args.configuration and len(args.job_ids) > 1:
        parser.error("--configuration applies to one job at a time")

    fields = _record_fields()
    diag_fields = _load_cell_diagnostics().FIELDS
    for job_id in args.job_ids:
        record, cells = build_record(job_id, args.work_root, args.configuration)
        _append(args.output, fields, [record])
        cell_path = args.work_root / "runs" / job_id / "cell-diagnostics.csv"
        cell_path.unlink(missing_ok=True)
        _append(cell_path, diag_fields, cells)
        print(f"{job_id}: {record['configuration']} wall={record['wall_seconds']}s "
              f"cells/h={record['cells_per_hour']} ess/s={record['ess_per_second']} "
              f"rhat[{record['rhat_summary']}] -> {cell_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
