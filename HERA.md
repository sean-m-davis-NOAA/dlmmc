# Running dlmmc on HERA

A recipe for fitting a `dlmmc` grid with MPI on NOAA RDHPCS HERA CPU nodes.
For how these recommendations were measured, see
[experiments/hpc/TESTING.md](experiments/hpc/TESTING.md). For the general MPI
interface, see [MPI-README.md](MPI-README.md).

**In short:** build a Miniforge environment with `mpi4py` compiled against
HERA's Intel MPI, then run one whole node as 20 MPI ranks × 2 CPUs with
`--parallel-chains 2`. Plan about 570 s per cell for 4 chains × (500 warmup +
500 draws); a 312-cell grid takes about 2 h 45 min.

## 1. One-time setup

Install Miniforge in your home or project directory by following
[Installing Miniforge](https://docs.rdhpcs.noaa.gov/software/python/miniforge.html).
RDHPCS does not allow Anaconda or Miniconda. Then, on a login node:

```bash
CONDA_BASE=$HOME/miniforge3            # your Miniforge prefix
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -n dlmmc-hera -c conda-forge --strict-channel-priority \
    python=3.12 numpy scipy pandas platformdirs xarray netcdf4 cmdstanpy \
    "arviz>=1.3,<2" pip setuptools wheel -y

# Build mpi4py against HERA's Intel MPI, not a conda MPI.
source "$(dirname "$LMOD_CMD")/../init/bash"
module load intel/2024.2.1 impi/2024.2.1
conda activate dlmmc-hera
MPICC=mpiicx python -m pip install --no-binary=mpi4py mpi4py

# Install dlmmc from a checkout on shared storage (the compute nodes need it).
git clone https://github.com/sean-m-davis-NOAA/dlmmc.git
python -m pip install --no-deps -e ./dlmmc
python -c 'import mpi4py, cmdstanpy; mpi4py.rc.initialize=False; from mpi4py import MPI; print(MPI.Get_library_version()); print(cmdstanpy.cmdstan_path())'
```

The last command should report Intel MPI and a CmdStan path inside the
environment; conda installs CmdStan along with `cmdstanpy`. `arviz` is needed only for
the R-hat check in step 5.

Notes:

- Use `mpiicx`. On HERA, `mpiicc` calls the classic `icc` compiler, which is no
  longer available.
- If the modules change, check with `module spider impi` and rebuild `mpi4py`.
  Always load the same `intel` and `impi` modules in batch jobs that you used
  for the build.
- Don't start MPI on a login node. The import check above deliberately avoids
  it, and `--dry-run` never initializes MPI.

## 2. Prepare the run script

[dlm_lat_alt_mpi_run.py](dlm_lat_alt_mpi_run.py) fits the example
pressure/latitude grid. For your own data, copy it and edit `DATASET`, the
regressors, and the model. Also review:

- **`OUTPUT_VARIABLES`**: which posterior variables to save and whether each is
  saved as `'single'` (default) or `'double'` precision. See
  [Output variables and precision](MPI-README.md#output-variables-and-precision).
  The latent time series (`trend`, `ar`, `seasonal`, ...) dominate the file
  size. Per-cell posterior means of the scalar parameters are always written.
  The compacted test grid was 3.01 GB with all float32 draws, or 550 MB with
  only `trend` and parameter draws. New outputs use cell-aligned chunks to avoid
  unused HDF5 space growth; older files can be compacted with `nccopy`. See the
  [storage investigation](experiments/hpc/TESTING.md#storage-investigation-and-output-choices-2026-11-12).
- **`--cell-indices`**: fit only these row-major flat `(pressure, latitude)`
  cells. Cells with no finite data are skipped automatically, but constant
  cells (for example all zeros) are not. In the example dataset, flat IDs 3–8
  and 15–20 are all zeros, so exclude them:

  ```bash
  CELL_INDICES="$(python -c 'print(*[i for i in range(324) if not (3 <= i <= 8 or 15 <= i <= 20)])')"
  ```

## 3. Choose the layout and wall time

HERA allocates whole 40-CPU nodes, so fill the node. The recommended layout is:

| Setting | Value |
|---|---|
| Nodes | 1 |
| `--ntasks` (MPI ranks) | 20: rank 0 writes the NetCDF, 19 ranks fit cells |
| `--cpus-per-task` | 2 |
| `--parallel-chains` | 2 |

With 4 chains, this ran about 116 cells/hour. A layout of 10 ranks × 4 CPUs with
`--parallel-chains 4` was practically as fast. Always set `--parallel-chains` to
the CPUs per task: with 2 CPUs but 1 parallel chain, the second CPU sits idle and
each cell takes about 70% longer.

Workers fit cells one after another, so the run time depends on the busiest
worker. Preview the plan with `--dry-run` (`-n`). It is safe on a login node
and needs only the data files:

```bash
python dlm_lat_alt_mpi_run.py -n --ranks 20 --chains 4 --samples 500 \
    --seconds-per-cell 570 --time-limit 4:00:00 --cell-indices $CELL_INDICES
```

```text
Cells to fit: 312 of 324 (12 skipped: no finite data or not selected)
Ranks: 20 (19 fitting workers; rank 0 writes)
Cells per worker: 16-17
Uncompressed output size: 4.13 GB (compression usually makes the file smaller)
Estimated fitting time: 2:41:30 (busiest worker), 3:21:52 with safety margin
Time limit 4:00:00: OK.
```

The 570 s/cell figure was measured for the example data and model with
4 chains × (500 warmup + 500 draws) under the 20 × 2 layout. Time per cell
scales roughly with warmup + draws and with the length of the series. For a
different model, data, or layout, measure it once: run a few cells (one per
worker) and divide the wall time by the busiest worker's cell count.

## 4. Submit

Copy [experiments/hpc/hera_production.sbatch.example](experiments/hpc/hera_production.sbatch.example),
replace the `YOUR_*` placeholders (account, Work directory, Miniforge prefix),
and set `--time` from the dry run. Submit from the repository root:

```bash
CELL_INDICES="$CELL_INDICES" sbatch --time=04:00:00 my-dlmmc-grid.sbatch
```

The template loads the modules, activates the environment, puts `TMPDIR` and the
CmdStan cache on Work storage, records a dry-run preview in the job log, runs
the fit with `srun --mpi=pmi2`, and summarizes the output. Override `CHAINS`,
`PARALLEL_CHAINS`, `WARMUP`, `SAMPLES`, `SEED`, or `SECONDS_PER_CELL` from the
environment when submitting.

Keep outputs and temporary files on your project's Work filesystem, which is not
backed up and is subject to purge; copy results you want to keep elsewhere. The
model compiles on the first run (about a minute) and is cached in the
`cmdstan-cache` directory afterwards.

### Splitting a long grid across jobs

If the dry run needs more time than you want to request (or than the QoS
allows), queue several shorter jobs that continue the same output file. Each
job fits only the cells that are not yet complete. A job killed at its time
limit loses only the cells in progress, and the next job picks up the rest:

```bash
export OUTPUT_FILE=/path/on/work/dlmmc-grid.nc RESUME=1 CELL_INDICES
JOB=$(sbatch --parsable --time=02:00:00 my-dlmmc-grid.sbatch)
JOB=$(sbatch --parsable --time=02:00:00 --dependency=afterany:$JOB my-dlmmc-grid.sbatch)
sbatch --time=02:00:00 --dependency=afterany:$JOB my-dlmmc-grid.sbatch
```

Keep every setting the same across the jobs. A resumed run refuses to start if
the chains, warmup, samples, seed, priors, or output variables differ from the
file's. Check how many cells remain, and whether one more job will finish them:

```bash
python dlm_lat_alt_mpi_run.py -n --resume --output "$OUTPUT_FILE" --ranks 20 --chains 4 \
    --samples 500 --seconds-per-cell 570 --time-limit 2:00:00 --cell-indices $CELL_INDICES
```

A job that starts after the grid is finished exits after setup without fitting.

To give every job the same, predictable amount of work instead, set `MAX_CELLS`
(and leave `CELL_INDICES` unset; the two cannot be combined). Each job then fits
the first `MAX_CELLS` cells that are not yet complete. A multiple of the 19
fitting workers balances the load: at about 570 s per cell, `MAX_CELLS=114` is 6
cells per worker, about 57 minutes of fitting (72 minutes with the dry run's
margin). Submit the same job until the dry run reports no cells left. Failed cells
are retried and count toward `MAX_CELLS` in every job, so cells that always fail
(such as constant-data cells) are better excluded with `CELL_INDICES`.

```bash
export OUTPUT_FILE=/path/on/work/dlmmc-grid.nc RESUME=1 MAX_CELLS=114
sbatch --time=01:30:00 my-dlmmc-grid.sbatch
```

## 5. Check the results

The job ends by running
[summarize_grid_output.py](experiments/hpc/summarize_grid_output.py), which you
can also run on any output file. Example output:

```text
Cells: 324 (12 skipped, 312 fitted)
Divergent transitions: 0 total, 0 fitted cells with any (max 0 of 2000 draws)
```

It exits with an error if any cell failed (`fit_status` -1). Then:

- **Divergent transitions:** a few in a cell call for a closer look; many mean
  the cell's posterior is unreliable. See
  [Output metadata and sampler diagnostics](README.md#output-metadata-and-sampler-diagnostics).
- **R-hat and effective sample size** per cell (needs `arviz`):

  ```bash
  python experiments/hpc/cell_diagnostics.py OUTPUT.nc --run-id JOBID --output cells.csv
  ```

  Re-fit cells with `max_rhat` above 1.01 using more warmup and draws, for
  example with `--cell-indices` and `--warmup 1000 --samples 1000`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module` fails or loads nothing in a batch job | Initialize Lmod with `source "$(dirname "$LMOD_CMD")/../init/bash"`; `$MODULESHOME/init/bash` does not work here. |
| `unbound variable` during `conda activate` in a `set -u` script | Activation hooks reference unset variables; wrap the call in `set +u` / `set -u`. |
| MPI hangs or every rank reports rank 0 | Launch with `srun --mpi=pmi2`, and check that `mpi4py` was built with `mpiicx` against the loaded `impi` module. |
| Job killed at the time limit | Resubmit the same job with `RESUME=1` and the same `OUTPUT_FILE` to fit the remaining cells (see [Splitting a long grid across jobs](#splitting-a-long-grid-across-jobs)), or rerun `--dry-run` with your measured `--seconds-per-cell` and raise `--time`. |
| `TMPDIR` fills up | Point `TMPDIR` at Work storage. Successful fits clean up after themselves; failed fits keep their CmdStan files for debugging. |
