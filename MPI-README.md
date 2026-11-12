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

The script calls `dlmmc.fit_grid_mpi()`. With multiple ranks, rank 0 creates and writes `results/BASIC_V1_2017.nc`; all other ranks fit assigned grid cells and send each result to rank 0 as soon as the fit completes. Rank 0 is reserved for writing, so four ranks means three fitting workers. Cells with data are dealt round-robin in flat (row-major) order to the workers, which fit their cells one after another. This single-writer design avoids concurrent NetCDF/HDF5 writes and does not require parallel-I/O-enabled libraries. Running the script directly with one process also works, but performs the fits serially.

The output is created as a new file; by default, an existing output path is not overwritten. Use `--overwrite` to replace it, or `--resume` to continue it (see [Splitting a run across jobs](#splitting-a-run-across-jobs---resume)). The result is readable with xarray. `fit_status` is 1 for completed fits, 0 for skipped (or not yet fitted) cells, and -1 for cells where fitting raised an error.

The defaults are one chain, one warmup iteration, and two sampling draws per cell. They are only a pipeline smoke test and do not produce converged posterior estimates. For longer runs, set the `--chains`, `--warmup`, and `--samples` options. Use `--cell-indices` to fit only some cells (row-major flat IDs), or `--max-cells N` to fit only the first N cells not yet complete. Choose `--output` to set a different output path. Existing output is not overwritten unless `--overwrite` is specified. Short smoke-test chains can emit Stan warnings; check `fit_status` and the output file rather than treating convergence warnings as a pipeline failure.

### Previewing a run (`--dry-run`)

Add `--dry-run` (or `-n`) to preview a run without fitting, compiling, or
starting MPI, so it is safe on a login node. Give the total rank count (including
the rank-0 writer) with `--ranks`; inside a Slurm job it defaults to
`$SLURM_NTASKS`. The report lists how many cells will be fitted, how the round-robin
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

### Splitting a run across jobs (`--resume`)

A grid that needs more time than one batch job allows can be fitted in several
jobs. Add `--resume` to every submission, with the same output path and
settings. Each job fits only the cells whose `fit_status` is not yet 1
(complete), then exits. Failed cells (-1) are retried. If the output file does
not exist yet, `--resume` starts a new run, so every job can use the same
command. A job that hits its time limit loses only the cells in progress,
because rank 0 writes each result to the file as it arrives.

Before fitting, the existing file is checked against the requested run: model,
grid, chains, warmup, samples, seed, sampler options, priors, and output
variables and precision must all match. Otherwise the run stops with a message
listing the differences. Each resume appends a line to the file's `history`
attribute. With `--dry-run`, `--resume` counts only the remaining cells, so you
can check whether one more job will finish the grid.

For example, on a workstation, fit the first half of the 324-cell example grid
(27 pressures × 12 latitudes) in one run, then the rest in a second. Pass the
same output path and sampler settings both times:

```bash
# Run 1: fit flat cell IDs 0-161 only; the other cells stay unfitted (fit_status 0).
mpirun -np 4 python dlm_lat_alt_mpi_run.py --resume --chains 4 --warmup 500 --samples 500 \
    --cell-indices $(seq 0 161)

# Optional: check how many cells remain.
python dlm_lat_alt_mpi_run.py --dry-run --ranks 4 --resume --chains 4 --warmup 500 --samples 500

# Run 2: without --cell-indices, fit every cell that is not yet complete (IDs 162-323).
mpirun -np 4 python dlm_lat_alt_mpi_run.py --resume --chains 4 --warmup 500 --samples 500
```

`--cell-indices` takes row-major flat IDs (`pressure_index * 12 + latitude_index`)
and marks every other cell as skipped for that run.

Alternatively, give only a cell count with `--max-cells N`: each run fits the
first N cells (in flat order) that are not yet complete, starting from the
beginning of the grid when the file is new. Repeating the same command works
through the grid in batches of N; once every cell is done, a further run fits
nothing. Failed cells (-1) are retried, so they count toward N again in each
run. `--max-cells` and `--cell-indices` cannot be combined. For example, the
same 324-cell grid in two equal runs:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py --resume --max-cells 162 --chains 4 --warmup 500 --samples 500
mpirun -np 4 python dlm_lat_alt_mpi_run.py --resume --max-cells 162 --chains 4 --warmup 500 --samples 500
```

Neither option is one of the settings that must match, so they can differ between
runs. With `--dry-run`, they are applied the same way, and the report shows how
many cells are left for later runs.

An interrupted run (Ctrl-C, or a batch job that hits its time limit) resumes the
same way: repeat its command with `--resume`. On a batch scheduler, put `--resume` in the job script and submit the
same script repeatedly. With Slurm, `--dependency=afterany:JOBID` starts each job
after the previous one ends, even if that job ran out of time.

A job that starts after the grid is complete fits nothing and exits after
setup. From Python, use `fit_grid_mpi(..., resume=True, max_cells=N)` and
`estimate_grid_run(..., resume_from=path, max_cells=N)`.

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

For the example's `vanilla_ar1` model, saving **all draws** means saving:

- The time-varying components `trend`, `slope`, `ar`, and `seasonal`.
- The parameters `sigma_trend`, `sigma_AR`, `rhoAR1`, `sigma_seas`, and `beta`.
  The `beta` draws contain one coefficient for each regressor: ENSO, SOLAR,
  QBO30, QBO50, and SAOD in the supplied example.

The mapping above is **trend plus parameter draws**: it retains `trend` and all
five parameter variables, but omits `slope`, `ar`, and `seasonal`. Omitting
those time-varying components means their posterior summaries cannot be
computed from that file; they do not have automatically saved mean variables.
The same model and number of samples are fitted in either case.

The supplied configuration saves every posterior variable in single precision
(float32), which halves the uncompressed storage per value compared with double
precision; compressed file sizes vary. Earlier versions saved all posterior
variables as float64; set entries to `"double"` to reproduce that. Coordinates
keep their existing precision (including double-precision timestamps), and
`fit_status` remains an integer.

New output files use cell-aligned compressed chunks, so writing one cell does
not rewrite chunks containing other cells. Older files can contain substantial
unused HDF5 space from repeated chunk rewrites; changing precision alone does
not recover that space. With the NetCDF command-line tools installed, compact a
closed file into a new file without changing its values:

```bash
nccopy results/BASIC_V1_2017.nc results/BASIC_V1_2017_compact.nc
```

Keep the original until the copy is checked. Existing files retain their chunk
layout on resume. To both compact and align an older pressure/latitude grid for
future cell writes, use `nccopy -c pressure/1,latitude/1 input.nc output.nc`
(substitute your grid dimension names).

Choose variables according to the analyses you need:

- Keep all draws for component-by-component posterior uncertainty.
- Keep `trend` and the scalar/regressor parameters to save trend uncertainty
  and parameter diagnostics without `slope`, `ar`, or `seasonal` draws.
- Keep only `sigma_trend`, `sigma_AR`, `rhoAR1`, `sigma_seas`, and `beta` for
  parameter distributions and R-hat/ESS diagnostics, but no latent time series.
- Use `{}` for means and fit/divergence counts only (with the default
  `posterior_means=True`). Means alone cannot recover credible intervals or
  R-hat/ESS. Omitting outputs reduces storage and MPI transfers, not sampling work.

Direct API callers can pass the same mapping as
`fit_grid_mpi(..., output_variables=...)`. `None` (the API default) saves all
posterior variables in single precision; `{}` saves no posterior draws, but
still writes coordinates, `fit_status`, `divergent_transitions`, and the default
non-time-varying posterior means. Unknown names or precisions fail during setup,
before an existing file is replaced. Only selected variables are transferred to
rank 0, converted to the requested precision. Sampling itself runs in double
precision and is unchanged; existing NetCDF files are not modified.

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

### Checking output

Verify that the output opens and inspect the grid dimensions and fit-status counts:

```bash
python -c "import numpy as np, xarray as xr; ds=xr.open_dataset('results/BASIC_V1_2017.nc'); print(ds.sizes); print(dict(zip(*np.unique(ds.fit_status.values, return_counts=True))))"
```

On multi-node HPC systems, pass `--cache-dir` pointing to a shared filesystem so the worker nodes can use the executable compiled by rank 0. Launch within a scheduler allocation using the site's MPI launcher and ensure `mpi4py` is built against that MPI implementation. Do not mix a site-provided MPI library with a different Conda MPI implementation. For larger production runs, adjust the sampling settings and number of ranks for the available resources.

## HERA CPU/MPI evaluation

For the HERA-first CPU workflow, site-specific setup checks, and a Slurm smoke
template, see [experiments/hpc/README.md](experiments/hpc/README.md) and
[experiments/hpc/hera_cpu_mpi.sbatch.example](experiments/hpc/hera_cpu_mpi.sbatch.example).
The example is intentionally not submit-ready: verify the project account,
Miniforge prefix, exact MPI module versions, `mpi4py` linkage, `srun` MPI plugin,
and Work/TMPDIR paths on HERA first. The initial template uses one node, rank 0
as writer, and two fitting workers; it is a pipeline smoke test, not a speed or
convergence result.

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
