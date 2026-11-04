# HERA CPU/MPI Validation

This is the HERA-first checklist for validating `dlmmc.fit_grid_mpi()` on NOAA RDHPCS CPU nodes. It is an experiment workflow, not a supported HERA installation recipe or a claim that any configuration has been tested on HERA. Keep the production API and CmdStanPy backend unchanged during this evaluation.

## Verified Site Facts

The current [HERA User Guide](https://docs.rdhpcs.noaa.gov/systems/hera_user_guide.html) describes 40-core Intel Skylake regular compute nodes and the `hera` partition, with `batch`, `windfall`, `debug`, `urgent`, and `long` QoS choices. It lists Intel MPI and MVAPICH2; confirm the live module versions and site recommendation with `module spider` before choosing a stack. The guide directs MPI launches through Slurm and states that HERA storage is shared with Ursa.

RDHPCS Python guidance requires Miniforge/conda-forge and prohibits Anaconda/Miniconda distributions. Create the environment in project home, use an approved conda-forge channel, and keep run outputs and temporary files on the project-approved Work filesystem. Work storage is not backed up and is subject to purge policy. Check the live storage paths and quotas with site guidance and `saccount_params`; do not copy an assumed `/scratchN` path from another system.

Useful official references:

- [HERA User Guide](https://docs.rdhpcs.noaa.gov/systems/hera_user_guide.html)
- [Slurm Overview](https://docs.rdhpcs.noaa.gov/slurm/overview.html)
- [Python on RDHPCS](https://docs.rdhpcs.noaa.gov/software/python/index.html)
- [Installing Miniforge](https://docs.rdhpcs.noaa.gov/software/python/miniforge.html)
- [Modules and Lmod](https://docs.rdhpcs.noaa.gov/software/modules/index.html)
- [Storage Areas](https://docs.rdhpcs.noaa.gov/data/storage.html)

## Environment and MPI Contract

1. On a HERA login or approved service node, record `module spider python`, `module spider intel`, `module spider impi`, and `srun --mpi=list`. Record the available Miniforge and MPI module versions, compiler wrapper, Slurm MPI plugin, Python version, CmdStan version, and `mpi4py.MPI.Get_library_version()`.
2. Create a dedicated Miniforge environment in project home using conda-forge packages. Avoid Anaconda/Miniconda and do not install packages into `base`.
3. Build or install `mpi4py` against the exact MPI library used by the HERA launcher. Do not mix a Conda MPICH/OpenMPI runtime with the site Intel MPI or MVAPICH2 launcher. If compiling `mpi4py`, load the matching site MPI module and use its compiler wrapper, then check `MPI.Get_library_version()` inside a Slurm-launched job.
4. Install the project in that environment from the checked-out source and resolve all core dependencies before submitting compute jobs. Do not rely on internet access from compute nodes.
5. Compile the CmdStan model on HERA (or in an architecture-compatible shared cache) and set `--cache-dir` to a directory visible to all ranks. Keep `TMPDIR` writable and set it to site-approved temporary storage. The starter example is single-node so each rank can use that node's local temporary path; rework this explicitly before multi-node runs.

An initial environment can be created in project home after Miniforge is installed and the site compiler/MPI modules have been verified:

```bash
CONDA_ENV=/path/to/project-home/envs/dlmmc-hera
conda create -p "$CONDA_ENV" -c conda-forge --strict-channel-priority \
	python=3.12 numpy scipy pandas platformdirs xarray netcdf4 cmdstanpy \
	pip setuptools wheel -y

# Replace these with exact module versions confirmed using module spider.
INTEL_MODULE="intel/REPLACE_WITH_VERSION"
IMPI_MODULE="impi/REPLACE_WITH_VERSION"
module load "$INTEL_MODULE" "$IMPI_MODULE"
conda activate "$CONDA_ENV"
MPICC=mpiicc python -m pip install --no-binary=mpi4py mpi4py

# In a checkout reachable from the compute nodes; dependencies were installed above.
python -m pip install --no-deps -e /path/to/dlmmc
python -c 'from mpi4py import MPI; import cmdstanpy; print(MPI.Get_library_version()); print(cmdstanpy.cmdstan_path())'
```

The module names/wrappers above follow the documented Intel MPI stack; replace them if `module spider`, `mpiicc -show`, or site support indicates a different current stack. Building `mpi4py` from source avoids silently bringing in an unrelated Conda MPI runtime. Record the `mpi4py` build command and `MPI.Get_library_version()` with each run.

## First Smoke Test

Start with [hera_cpu_mpi.sbatch.example](hera_cpu_mpi.sbatch.example), after replacing every placeholder and confirming the account, module versions, MPI plugin, and Work path. Submit from a project-Work checkout (or edit the Slurm log paths to a pre-created Work directory). It requests one node and three MPI ranks: rank 0 is the NetCDF writer and two ranks fit cells. Each worker uses one CmdStan chain. It checks MPI initialization, fits two cells with tiny warmup/sample counts, and verifies both `fit_status` values. This only tests setup and output plumbing; it is not a scientific fit.

Before a long run, inspect the job's resource limits with `sbatch-limits` and use `sinfo -O partition`. If the job needs to pull packages or sources, do that from an approved login/service workflow before compute; do not install dependencies in the batch script.

If the MPI smoke fails, retain the batch output, `module list`, `srun --mpi=list`, `mpi4py` library version, and `sacct` job state. Do not change several environment variables at once. Confirm that `srun` selected the same MPI implementation against which `mpi4py` was built before trying another stack/plugin.

## Performance Test Plan

After the smoke succeeds, use fixed cell IDs, data, seeds, and sampling settings for all configurations. The source-grid example has 312 valid nonconstant cells; select a reproducible diverse subset and record the actual flat IDs. Run from a project Work directory and write NetCDF, logs, CmdStan cache, and any retained artifacts outside the source checkout.

1. Measure a one-cell `dlmmc.fit()` baseline and a one-rank `fit_grid_mpi()` baseline to separate model compilation, single-cell sampling, and grid setup.
2. Measure the same 16- or 32-cell workload with Slurm task counts such as 3, 5, and 9 (rank 0 writer included). Keep one chain per worker initially.
3. With the same task allocation, compare `--parallel-chains 1` versus 2 only when each worker has enough allocated CPUs. A first eight-chain-process configuration is 5 MPI ranks with 2 CPUs per task and `--parallel-chains 2`: four workers times two chains, plus rank 0. Compare with 9 ranks, 1 CPU per task, and `--parallel-chains 1`: eight workers plus rank 0. Keep within a single 40-core node for the first sweep.
4. Once single-node behavior is understood, request multiple nodes and explicitly set `--nodes`, `--ntasks-per-node`, and `--cpus-per-task`. Preserve rank 0 as the sole NetCDF writer; workers fit assigned cells sequentially.
5. Use scientific warmup and retained draw counts for performance claims. The runner defaults of 1 warmup and 2 samples are pipeline smoke settings only.

For every run record the git commit, HERA node/CPU allocation, loaded modules, MPI implementation and launcher/plugin, Python/CmdStan/CmdStanPy/mpi4py versions, cell IDs, model and priors, chain settings, seed, wall time, `fit_status` counts, CmdStan R-hat and bulk/tail ESS, and peak memory. Include model compilation/cache state, cells per hour, rank utilization, output size, and `TMPDIR`/Work path. Inspect per-cell diagnostics; aggregate throughput must not hide a poorly mixing cell. Compare end-to-end wall time and ESS/sec, not only likelihood timings.

## Decision Gate

Use CmdStanPy on HERA as the scientific and performance reference. Do not recommend a new backend based on these tests: they evaluate the existing CmdStanPy/MPI path and its resource configuration. First establish a reproducible HERA baseline, then select rank/chain settings that improve throughput without worsening per-cell convergence, memory, or fit reliability. Keep site-specific assumptions in this HERA document until verified; update host-neutral [MPI-README.md](../../MPI-README.md) only for generally reusable guidance.
