## Running the grid example with MPI

`dlm_lat_alt_mpi_run.py` fits one DLM for each pressure/latitude cell in the example ozone dataset. It uses CmdStanPy for sampling, xarray to read the input, and NetCDF for the result file. The Python environment needs NumPy, SciPy, xarray, netCDF4, CmdStanPy, tqdm, and mpi4py, plus a working MPI implementation.

From the repository root, generate the CmdStanPy model files and canonical markers expected by the example:

```bash
python compile_stan_models.py
```

On a local machine with Open MPI, launch four ranks with:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

With multiple ranks, rank 0 creates and writes `results/BASIC_V1_2017.nc`; all other ranks fit assigned grid cells and send each result to rank 0 as soon as the fit completes. Rank 0 is reserved for writing, so four ranks means three fitting workers. This single-writer design avoids concurrent NetCDF/HDF5 writes and does not require parallel-I/O-enabled libraries. Running the script directly with one process also works, but performs the fits serially.

The output is recreated at the start of each run; the script does not resume or append to an existing file. It is readable with xarray. `fit_status` is 1 for completed fits, 0 for skipped cells, and -1 for cells where fitting raised an error.

The defaults (`n_chains = 1`, `warmup = 1`, `iterations = 3`) are only a pipeline smoke test and produce two sampling draws per fit. They do not produce converged posterior estimates. Increase warmup and sampling iterations in the script for scientific runs. Short smoke-test chains can emit Stan warnings; check `fit_status` and the output file rather than treating convergence warnings as a pipeline failure.

Verify that the output opens and inspect the grid dimensions and fit-status counts:

```bash
python -c "import numpy as np, xarray as xr; ds=xr.open_dataset('results/BASIC_V1_2017.nc'); print(ds.sizes); print(dict(zip(*np.unique(ds.fit_status.values, return_counts=True))))"
```

On HPC systems, launch within a scheduler allocation using the site's MPI launcher and ensure `mpi4py` is built against that MPI implementation. Do not mix a site-provided MPI library with a different Conda MPI implementation. For larger production runs, adjust the sampling settings and number of ranks for the available resources.
