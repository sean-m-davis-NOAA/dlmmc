## Running the grid example with MPI

`dlm_lat_alt_mpi_run.py` fits one DLM for each pressure/latitude cell in the example ozone dataset through `dlmmc.fit_grid_mpi()`. The package uses xarray for inputs and writes a NetCDF result. Install the optional MPI dependencies with `python -m pip install 'dlmmc[mpi]'` (or `python -m pip install -e '.[mpi]'` from a source checkout). You also need a working MPI implementation compatible with `mpi4py`.

The package contains the Stan sources and compiles the selected model on rank 0 before workers start. To compile all packaged models before launching, run `python compile_stan_models.py` from the repository root.

On a local machine with Open MPI, launch four ranks from the repository root with:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The script calls `dlmmc.fit_grid_mpi()`. With multiple ranks, rank 0 creates and writes `results/BASIC_V1_2017.nc`; all other ranks fit assigned grid cells and send each result to rank 0 as soon as the fit completes. Rank 0 is reserved for writing, so four ranks means three fitting workers. This single-writer design avoids concurrent NetCDF/HDF5 writes and does not require parallel-I/O-enabled libraries. Running the script directly with one process also works, but performs the fits serially.

The output is created as a new file; by default, an existing output path is not overwritten. Use `--overwrite` to replace it; runs do not resume or append. The result is readable with xarray. `fit_status` is 1 for completed fits, 0 for skipped cells, and -1 for cells where fitting raised an error.

The defaults are one chain, one warmup iteration, and two sampling draws per cell. They are only a pipeline smoke test and do not produce converged posterior estimates. For longer runs, set the `--chains`, `--warmup`, and `--samples` options. Choose `--output` to set a different output path. Existing output is not overwritten unless `--overwrite` is specified. Short smoke-test chains can emit Stan warnings; check `fit_status` and the output file rather than treating convergence warnings as a pipeline failure.

Verify that the output opens and inspect the grid dimensions and fit-status counts:

```bash
python -c "import numpy as np, xarray as xr; ds=xr.open_dataset('results/BASIC_V1_2017.nc'); print(ds.sizes); print(dict(zip(*np.unique(ds.fit_status.values, return_counts=True))))"
```

On multi-node HPC systems, pass `--cache-dir` pointing to a shared filesystem so the worker nodes can use the executable compiled by rank 0. Launch within a scheduler allocation using the site's MPI launcher and ensure `mpi4py` is built against that MPI implementation. Do not mix a site-provided MPI library with a different Conda MPI implementation. For larger production runs, adjust the sampling settings and number of ranks for the available resources.

### Temporary storage

Workers remove each successful fit's temporary Stan output after loading its
posterior and diagnostics into memory, rather than retaining CSV files for the
entire grid run. Failed or interrupted fits retain their output and log its
location for troubleshooting; repeated failures can still fill temporary storage.
The NetCDF result and compiled-model cache are not removed.

Set `TMPDIR` to a writable scratch directory with sufficient space and quota
before launching MPI. Ensure the setting reaches every worker and the directory
exists on each node. A single fit can still require substantial temporary space.
See [INSTALL.md](INSTALL.md#temporary-storage) for setup instructions.

If calling `fit_grid_mpi()` directly, an explicit `output_dir` is forwarded to
each fit and is never removed by dlmmc. Avoid pointing concurrent fits at the same
output directory because their filenames can collide.
