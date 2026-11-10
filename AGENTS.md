# Agent notes for dlmmc

`dlmmc` fits dynamic linear models with CmdStanPy. `dlmmc.fit()` fits one time series; `dlmmc.fit_grid_mpi()` ([src/dlmmc/mpi.py](src/dlmmc/mpi.py)) fits a grid with MPI, where rank 0 writes the NetCDF and the other ranks fit the cells that need fitting, dealt round-robin in flat order. `--resume` / `resume=True` continues an existing output file.

## Tests

On HERA, run the suite with the project environment:

```bash
/home/Sean.M.Davis/miniforge3/envs/dlmmc-hera/bin/python -m pytest -q
```

## HERA CPU/MPI validation

- New-user recipe: [HERA.md](HERA.md). Testing record (measurements, interpretation, full-grid result, open items): [experiments/hpc/TESTING.md](experiments/hpc/TESTING.md). File index: [experiments/hpc/README.md](experiments/hpc/README.md). Keep recommendations in HERA.md and measurements in TESTING.md.
- The layout work is done (20 ranks × 2 CPUs, `--parallel-chains 2`, about 570 s/cell). Remaining work is listed under "Open items" in TESTING.md.
- Repository checkout: `/scratch3/BMC/chimera/Sean.M.Davis/dlmmc` (branch `hpc/hera-cmdstanpy-mpi`).
- Run outputs, Slurm logs, and the run-record CSV: `/scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work` (`runs/JOBID/`, `hera-cpu-mpi-run-record.csv`). This is Work storage: not backed up and subject to purge.
- Environment: Miniforge at `/home/Sean.M.Davis/miniforge3`, env `dlmmc-hera` (Python 3.12, CmdStan 2.40, `mpi4py` built with `MPICC=mpiicx` against `impi/2024.2.1`).
- Slurm account `chimera`, partition `hera`, `srun --mpi=pmi2`. Submit jobs from the repository root; templates are `experiments/hpc/*.sbatch.example`. Confirm with the user before submitting jobs, since they charge the allocation.
- Collect finished jobs with `python experiments/hpc/collect_run_record.py --output /scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work/hera-cpu-mpi-run-record.csv JOBID ...`.

HERA pitfalls already solved in the templates:

- Initialize Lmod with `source "$(dirname "$LMOD_CMD")/../init/bash"`; `$MODULESHOME/init/bash` breaks it.
- Wrap `conda activate` in `set +u` / `set -u`; the CmdStan activation hook references unset variables.
- Do not initialize MPI on a login node; check MPI only inside a Slurm launch.
- `*.csv` is git-ignored, so CSV templates use the `.csv.example` suffix.
