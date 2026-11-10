# HPC files

Scripts and Slurm templates for running and testing `dlmmc` MPI grid fits on
NOAA RDHPCS HERA.

- **To run dlmmc on HERA**, follow [HERA.md](../../HERA.md).
- **For the validation and benchmark record**, see [TESTING.md](TESTING.md).

| File | Purpose |
|---|---|
| [hera_production.sbatch.example](hera_production.sbatch.example) | Production grid fit: one node, 20 ranks × 2 CPUs, 2 parallel chains. Replace the `YOUR_*` placeholders. |
| [summarize_grid_output.py](summarize_grid_output.py) | Fit-status counts and divergences for an output file; exits 1 if any cell failed. |
| [cell_diagnostics.py](cell_diagnostics.py) | Per-cell R-hat, bulk/tail ESS, and divergences as CSV (needs `arviz`). |
| [hera_cpu_mpi.sbatch.example](hera_cpu_mpi.sbatch.example) | Three-rank MPI smoke test of two cells. |
| [hera_cpu_fit_baseline.sbatch.example](hera_cpu_fit_baseline.sbatch.example) | Single-cell `dlmmc.fit()` baseline without MPI. |
| [hera_cpu_mpi_performance.sbatch.example](hera_cpu_mpi_performance.sbatch.example) | Benchmark template configured by environment variables (see [TESTING.md](TESTING.md#performance-test-plan)). |
| [hera_cpu_mpi_multinode.sbatch.example](hera_cpu_mpi_multinode.sbatch.example) | Two-node wrapper around the benchmark template. |
| [check_mpi.py](check_mpi.py) | Verifies MPI initialization and collectives inside a Slurm launch. |
| [check_fit_status.py](check_fit_status.py) | Verifies that exactly the selected cells were fitted. |
| [collect_run_record.py](collect_run_record.py) | Builds a run-record row from a job's metadata, `sacct`, and output. |
| [hera_cpu_mpi_run_record.csv.example](hera_cpu_mpi_run_record.csv.example), [hera_cpu_mpi_cell_diagnostics.csv.example](hera_cpu_mpi_cell_diagnostics.csv.example) | Column templates for the run record and per-cell diagnostics. |

The benchmark templates contain the paths and account used for testing; the
production template uses placeholders.
