## Running the grid example with MPI

For a HERA-first CPU workflow, environment checks, and a Slurm smoke template,
see [experiments/hpc/README.md](experiments/hpc/README.md). Its HERA-specific
module names, MPI plugin, account, and storage paths are placeholders that must
be verified on the live system.

`dlm_lat_alt_mpi_run.py` fits one DLM for each pressure/latitude cell in the example ozone dataset through `dlmmc.fit_grid_mpi()`. The package uses xarray for inputs and writes a NetCDF result. Install the optional MPI dependencies with `python -m pip install 'dlmmc[mpi]'` (or `python -m pip install -e '.[mpi]'` from a source checkout). You also need a working MPI implementation compatible with `mpi4py`.

The package contains the Stan sources and compiles the selected model on rank 0 before workers start. To compile all packaged models before launching, run `python compile_stan_models.py` from the repository root.

On a local machine with Open MPI, launch four ranks from the repository root with:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

To capture all standard output and warnings in a file:

```bash
mkdir -p results && mpirun -np 4 python dlm_lat_alt_mpi_run.py > results/mpi_run.log 2>&1
```

The script calls `dlmmc.fit_grid_mpi()`. With multiple ranks, rank 0 creates and writes `results/BASIC_V1_2017.nc`; all other ranks fit assigned grid cells and send each result to rank 0 as soon as the fit completes. Rank 0 is reserved for writing, so four ranks means three fitting workers. This single-writer design avoids concurrent NetCDF/HDF5 writes and does not require parallel-I/O-enabled libraries. Running the script directly with one process also works, but performs the fits serially.

The output is created as a new file; by default, an existing output path is not overwritten. Use `--overwrite` to replace it; runs do not resume or append. The result is readable with xarray. `fit_status` is 1 for completed fits, 0 for skipped cells, and -1 for cells where fitting raised an error.

The defaults are one chain, one warmup iteration, and two sampling draws per cell. They are only a pipeline smoke test and do not produce converged posterior estimates. For longer runs, set the `--chains`, `--warmup`, and `--samples` options. Choose `--output` to set a different output path. Existing output is not overwritten unless `--overwrite` is specified. Short smoke-test chains can emit Stan warnings; check `fit_status` and the output file rather than treating convergence warnings as a pipeline failure.

Use `--parallel-chains` to limit concurrent chains per cell and allocate at least that many CPUs per fitting worker. `--cell-indices` selects flat row-major `(pressure, latitude)` indices; unselected cells remain in the output grid with skipped status `0`.

### Previewing a run (`--dry-run`)

Add `--dry-run` (or `-n`) to preview a run without fitting, compiling, or
starting MPI, so it is safe on a login node. Give the total rank count (including
the rank-0 writer) with `--ranks`; inside a Slurm job it defaults to
`$SLURM_NTASKS`. The report lists how many cells will be fitted, how the static
assignment spreads them across workers (the busiest worker sets the run time),
cells with constant data, which usually fail or are meaningless, and the
uncompressed output size.

```bash
python dlm_lat_alt_mpi_run.py --dry-run --ranks 20 --chains 4 --warmup 500 --samples 500 \
    --seconds-per-cell 570 --time-limit 4:00:00
```

To estimate the run time, pass `--seconds-per-cell`: the wall seconds per cell
measured for the same model, chain count, iterations, `--parallel-chains`, and
node layout. dlmmc does not predict this value, because it depends on the data,
model, and hardware. A good measurement is a previous run's fitting wall time
divided by its busiest worker's cell count. As a rough guide, time per cell scales
with `chains * (warmup + samples) / parallel_chains`, but contention on a busy
node can make it longer. The estimate is the busiest worker's cell count times
`--seconds-per-cell`, multiplied by `--safety-factor` (default 1.25). With
`--time-limit` (any Slurm `--time` format), the report warns if the limit is
shorter than the estimate with margin. Model compilation and startup add time
that the margin should cover. The same preview is available from Python as
`dlmmc.estimate_grid_run()`.

### Output variables and precision

Edit `OUTPUT_VARIABLES` near the top of `dlm_lat_alt_mpi_run.py` to choose which
posterior variables are saved and their precision. Remove an entry to omit that
variable; use `"single"` for float32 or `"double"` for float64. For example:

```python
OUTPUT_VARIABLES = {
    "trend": "single",
    "sigma_trend": "double",
    "sigma_AR": "double",
    "rhoAR1": "double",
    "sigma_seas": "double",
    "beta": "single",
}
```

This example omits `slope`, `ar`, and `seasonal`. The supplied configuration saves
every posterior variable in single precision (float32), which halves the
uncompressed storage per value compared with double precision; compressed file
sizes vary. Earlier versions saved all posterior variables as float64; set entries
to `"double"` to reproduce that. Coordinates keep their existing precision
(including double-precision timestamps), and `fit_status` remains an integer.

Direct API callers can pass the same mapping as
`fit_grid_mpi(..., output_variables=...)`. `None` (the API default) saves all
posterior variables in single precision; `{}` saves only coordinates,
`fit_status`, and `divergent_transitions`. Unknown names or precisions fail during setup, before an existing
file is replaced. Only selected variables are transferred to rank 0, converted
to the requested precision. Sampling itself runs in double precision and is
unchanged; existing NetCDF files are not modified.

Every grid file also contains `fit_status` and an integer `divergent_transitions`
variable over the grid dimensions (missing for skipped or failed cells), plus
global attributes describing the model, sampler settings, priors, and software
versions.

By default, each non-time-varying parameter also gets a per-cell posterior mean,
named `<name>_mean` (for example `sigma_AR_mean`, `rhoAR1_mean`, and, for static
regressor models, `beta_mean` with a `regressor` dimension). Means are stored in
double precision, are computed from the full-precision draws, and are written
even when that parameter's draws are omitted from `OUTPUT_VARIABLES`, so you can
drop large draw arrays and keep a compact summary. Time-varying variables
(`trend`, `slope`, `ar`, `seasonal`, dynamic `beta`) do not get means. Pass
`posterior_means=False` to `fit_grid_mpi()` to omit them. See [Output metadata and sampler diagnostics](README.md#output-metadata-and-sampler-diagnostics)
for what these mean and how to interpret divergences.

The HERA Slurm templates invoke this runner, so they use the same configuration.
The R-hat/ESS collector can inspect only retained non-time-varying parameters;
omitting them narrows diagnostic coverage, and omitting all of them prevents
that collector from computing diagnostics.

### Checking output

Verify that the output opens and inspect the grid dimensions and fit-status counts:

```bash
python -c "import numpy as np, xarray as xr; ds=xr.open_dataset('results/BASIC_V1_2017.nc'); print(ds.sizes); print(dict(zip(*np.unique(ds.fit_status.values, return_counts=True))))"
```

On multi-node HPC systems, pass `--cache-dir` pointing to a shared filesystem so the worker nodes can use the executable compiled by rank 0. Launch within a scheduler allocation using the site's MPI launcher and ensure `mpi4py` is built against that MPI implementation. Do not mix a site-provided MPI library with a different Conda MPI implementation. For larger production runs, adjust the sampling settings and number of ranks for the available resources.

## HERA CPU/MPI evaluation

To run grid fits on NOAA RDHPCS HERA, follow [HERA.md](HERA.md): environment
setup with Intel MPI, the recommended 20 ranks × 2 CPUs layout, wall-time
planning with `--dry-run`, and a production Slurm template. The validation and
benchmark record behind those recommendations is in
[experiments/hpc/TESTING.md](experiments/hpc/TESTING.md), and
[experiments/hpc/README.md](experiments/hpc/README.md) lists the scripts and
templates.

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
