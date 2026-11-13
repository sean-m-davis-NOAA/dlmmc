# HPC files

Scripts and Slurm templates for running and testing `dlmmc` MPI grid fits on
HPC systems. Site-specific recipes, templates, and test records live in one
folder per site; the tools in this folder are shared.

## Sites

- **[hera/](hera/)**: NOAA RDHPCS HERA CPU nodes. To run dlmmc there, follow
  [hera/README.md](hera/README.md); the validation and benchmark record is
  [hera/TESTING.md](hera/TESTING.md).

| File | Purpose |
|---|---|
| [hera/hera_production.sbatch.example](hera/hera_production.sbatch.example) | Production grid fit: one node, 20 ranks × 2 CPUs, 2 parallel chains. Replace the `YOUR_*` placeholders. |
| [hera/hera_cpu_mpi.sbatch.example](hera/hera_cpu_mpi.sbatch.example) | Three-rank MPI smoke test of two cells. |
| [hera/hera_cpu_fit_baseline.sbatch.example](hera/hera_cpu_fit_baseline.sbatch.example) | Single-cell `dlmmc.fit()` baseline without MPI. |
| [hera/hera_cpu_mpi_performance.sbatch.example](hera/hera_cpu_mpi_performance.sbatch.example) | Benchmark template configured by environment variables (see [hera/TESTING.md](hera/TESTING.md#performance-test-plan)). |
| [hera/hera_cpu_mpi_multinode.sbatch.example](hera/hera_cpu_mpi_multinode.sbatch.example) | Two-node wrapper around the benchmark template. |

The HERA benchmark templates contain the paths and account used for testing;
the production template uses placeholders.

## Shared tools

Run these from the repository root, for example
`python experiments/hpc/summarize_grid_output.py OUTPUT.nc`.

| File | Purpose |
|---|---|
| [summarize_grid_output.py](summarize_grid_output.py) | Fit-status counts and divergences for an output file; exits 1 if any cell failed. |
| [cell_diagnostics.py](cell_diagnostics.py) | Per-cell R-hat, bulk/tail ESS, and divergences as CSV (needs `arviz`). |
| [check_mpi.py](check_mpi.py) | Verifies MPI initialization and collectives inside a Slurm launch. |
| [check_fit_status.py](check_fit_status.py) | Verifies that exactly the selected cells were fitted. |
| [collect_run_record.py](collect_run_record.py) | Builds a run-record row from a job's metadata, `sacct`, and output. Pass `--work-root` or set `DLMMC_WORK_ROOT`. |
| [hera_cpu_mpi_run_record.csv.example](hera_cpu_mpi_run_record.csv.example), [hera_cpu_mpi_cell_diagnostics.csv.example](hera_cpu_mpi_cell_diagnostics.csv.example) | Column templates for the run record and per-cell diagnostics; `collect_run_record.py` reads the run-record template. |
