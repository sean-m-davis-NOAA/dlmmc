# Agent notes for dlmmc

`dlmmc` fits dynamic linear models with CmdStanPy. `dlmmc.fit()` fits one time series; `dlmmc.fit_grid_mpi()` ([src/dlmmc/mpi.py](src/dlmmc/mpi.py)) fits a grid with MPI, where rank 0 writes the NetCDF and the other ranks fit the cells that need fitting, dealt round-robin in flat order. `--resume` / `resume=True` continues an existing output file.

## Tests

From the repository root, with `dlmmc` installed in the active environment:

```bash
python -m pytest -q
```

## HPC

HPC recipes, Slurm templates, and test records live under [experiments/hpc/](experiments/hpc/README.md): shared tools at the top level and one folder per site. For HERA work, read [experiments/hpc/hera/AGENTS.md](experiments/hpc/hera/AGENTS.md) first.
