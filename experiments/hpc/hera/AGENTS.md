# Agent notes for HERA work

General repository notes are in the root [AGENTS.md](../../../AGENTS.md).

## Tests

On HERA, run the suite from the repository root with the project environment:

```bash
/home/Sean.M.Davis/miniforge3/envs/dlmmc-hera/bin/python -m pytest -q
```

## HERA CPU/MPI validation

- New-user recipe: [README.md](README.md). Testing record (measurements, interpretation, full-grid result, open items): [TESTING.md](TESTING.md). File index: [experiments/hpc/README.md](../README.md). Keep recommendations in README.md and measurements in TESTING.md.
- The layout work is done (20 ranks × 2 CPUs, `--parallel-chains 2`, about 570 s/cell), and every item under "Open items" in TESTING.md is closed.
- Repository checkout: `/scratch3/BMC/chimera/Sean.M.Davis/dlmmc` (branch `main`; the HERA work was developed on `hpc/hera-cmdstanpy-mpi`).
- Run outputs, Slurm logs, and the run-record CSV: `/scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work` (`runs/JOBID/`, `hera-cpu-mpi-run-record.csv`). This is Work storage: not backed up and subject to purge.
- Environment: Miniforge at `/home/Sean.M.Davis/miniforge3`, env `dlmmc-hera` (Python 3.12, CmdStan 2.40, `mpi4py` built with `MPICC=mpiicx` against `impi/2024.2.1`).
- Slurm account `chimera`, partition `hera`, `srun --mpi=pmi2`. Submit jobs from the repository root; templates are `experiments/hpc/hera/*.sbatch.example`, and they call the shared tools in `experiments/hpc/`. Confirm with the user before submitting jobs, since they charge the allocation.
- Collect finished jobs with `python experiments/hpc/collect_run_record.py --work-root /scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work --output /scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work/hera-cpu-mpi-run-record.csv JOBID ...`.

HERA pitfalls already solved in the templates:

- Initialize Lmod with `source "$(dirname "$LMOD_CMD")/../init/bash"`; `$MODULESHOME/init/bash` breaks it.
- Wrap `conda activate` in `set +u` / `set -u`; the CmdStan activation hook references unset variables.
- Do not initialize MPI on a login node; check MPI only inside a Slurm launch.
- `*.csv` is git-ignored, so CSV templates use the `.csv.example` suffix.
