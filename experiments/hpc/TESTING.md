# HERA CPU/MPI Testing Record

This is the record of how `dlmmc.fit_grid_mpi()` was validated and benchmarked on NOAA RDHPCS HERA CPU nodes, and how the recommendations in [HERA.md](../../HERA.md) were reached. Use HERA.md to run dlmmc; use this document to reproduce or extend the measurements. These are performance and pipeline tests on the example dataset, not a scientific convergence study.

Run outputs, Slurm logs, and the run-record CSV are on project Work storage (`WORK_ROOT`, `/scratch3/BMC/chimera/Sean.M.Davis/dlmmc-work`), which is not backed up and is subject to purge.

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

## Tested Environment

The setup steps are in [HERA.md](../../HERA.md#1-one-time-setup). All runs below used:

- Miniforge environment `dlmmc-hera` from conda-forge: Python 3.12.16, CmdStanPy 1.3.2, CmdStan 2.40.0.
- `mpi4py` 4.1.2 built with `MPICC=mpiicx` under `intel/2024.2.1` and `impi/2024.2.1` (Intel MPI 2021.13), launched with `srun --mpi=pmi2`.
- Slurm account `chimera`, partition `hera`, QoS `batch`; the model was compiled on HERA into a shared cache on Work.

On the HERA login node, `mpiicc` points to the unavailable classic `icc`, while `mpiicx` uses the available `icx`. Building `mpi4py` from source avoids silently bringing in an unrelated Conda MPI runtime. Each job's `runs/JOBID/run-metadata.txt` records the module list, MPI plugins, and library versions.

## First Smoke Test

Start with [hera_cpu_mpi.sbatch.example](hera_cpu_mpi.sbatch.example), after confirming its Miniforge installation, account, module versions, MPI plugin, and Work path. The current Intel MPI stack was checked with the `pmi2` Slurm plugin; the template uses it by default. It requests one node and three MPI ranks: rank 0 is the NetCDF writer and two ranks fit one fixed cell each (flat IDs 0 and 33). It checks MPI initialization, runs tiny warmup/sample counts, and verifies those two `fit_status` values. This only tests setup and output plumbing; it is not a scientific fit.

Before a long run, inspect the job's resource limits with `sbatch-limits` and use `sinfo -O partition`. If the job needs to pull packages or sources, do that from an approved login/service workflow before compute; do not install dependencies in the batch script.

If the MPI smoke fails, retain the batch output, `module list`, `srun --mpi=list`, `mpi4py` library version, and `sacct` job state. Do not change several environment variables at once. Confirm that `srun` selected the same MPI implementation against which `mpi4py` was built before trying another stack/plugin.

On 2026-11-04, HERA job `34633280` completed the two-cell smoke using three ranks, Intel MPI 2021.13, `pmi2`, and CmdStan 2.40.0. Both selected cells completed and 322 were skipped; the 73-second wall time included a cold model compilation. CmdStan reported non-fatal NaN-scale and divergence warnings in these tiny chains, so this is not a convergence or performance result. The NetCDF and a populated run record are under `WORK_ROOT/runs/34633280`.

## Performance Test Plan

After the smoke succeeds, use fixed cell IDs, data, seeds, and sampling settings for all configurations. The source-grid example has 312 valid nonconstant cells. The performance template uses 16 IDs approximately spread across the grid: `0 22 43 65 86 108 129 151 172 194 215 237 258 280 301 323`. For a 32-cell workload, use `0 10 21 31 42 52 62 73 83 94 104 115 125 135 146 156 167 177 188 198 208 219 229 240 250 261 271 281 292 302 313 323`. Flat IDs are row-major over `(pressure, latitude)`. Both sets are balanced modulo eight: the flat-index modulo assignment used for these runs gave each of eight workers two or four cells respectively, and is also balanced for two or four workers. (dlmmc now deals the cells to fit round-robin, which gives the same counts for these sets.) The runner masks unselected cells, which are written with skipped status `0`; the output remains the full grid.

1. Use [hera_cpu_fit_baseline.sbatch.example](hera_cpu_fit_baseline.sbatch.example) for a one-cell `dlmmc.fit()` baseline (flat ID 0 is pressure index 0, latitude index 0). Use the MPI performance template with one rank and `PERF_CELL_INDICES=0` for a matching one-cell `fit_grid_mpi()` baseline. The performance template's 16-cell, one-rank run then measures grid overhead separately.
2. Measure the same 16-cell workload with Slurm task counts 3, 5, and 9 (rank 0 writer included), initially using one CPU per task and `--parallel-chains 1`. Keep chains, warmup, draws, seed, and cell IDs fixed.
3. With the same five-task allocation, compare `--parallel-chains 1` and 2 only when each worker has enough allocated CPUs. For an eight-chain-process configuration, use 5 MPI ranks with 2 CPUs per task and `--parallel-chains 2` (four workers, two concurrent chains per worker, plus rank 0). Compare with 9 ranks, 1 CPU per task, and `--parallel-chains 1` (eight workers plus rank 0). Keep within a single 40-core node for the first sweep.
4. Once single-node behavior is understood, try [hera_cpu_mpi_multinode.sbatch.example](hera_cpu_mpi_multinode.sbatch.example). It requests two nodes and nine ranks (eight workers plus rank 0) and invokes the same performance script. It uses a per-job `TMPDIR` on shared project Work so every node sees the directory; confirm Work and the selected MPI plugin are accessible on all nodes. This configuration completed on 2026-11-05 as job `34659299`: all 16 selected cells finished across two nodes in 1,453 seconds. Preserve rank 0 as the sole NetCDF writer; workers fit assigned cells sequentially.
5. Use scientific warmup and retained draw counts for performance claims. The runner defaults of 1 warmup and 2 samples are pipeline smoke settings only.

Configure [hera_cpu_mpi_performance.sbatch.example](hera_cpu_mpi_performance.sbatch.example) per run from the submit command rather than copying it: `sbatch` options override its `#SBATCH` defaults, and the exported environment variables `PERF_CELL_INDICES` (space-separated flat IDs), `CHAINS`, `PARALLEL_CHAINS`, `WARMUP`, `SAMPLES`, and `SEED` override its fit settings. Keep the Slurm allocation consistent with the fit settings. From the repository root, the sweep is:

```bash
T=experiments/hpc/hera_cpu_mpi_performance.sbatch.example
sbatch -J dlmmc-hera-fit experiments/hpc/hera_cpu_fit_baseline.sbatch.example
PERF_CELL_INDICES=0 sbatch -J perf-r1-c1 --ntasks=1 --time=01:00:00 "$T"
sbatch -J perf-r1-c16 --ntasks=1 --time=04:00:00 "$T"
sbatch -J perf-r3 --ntasks=3 --time=03:00:00 "$T"
sbatch -J perf-r5 --ntasks=5 --time=02:00:00 "$T"
sbatch -J perf-r9 --ntasks=9 --time=01:30:00 "$T"
PARALLEL_CHAINS=2 sbatch -J perf-r5x2-pc2 --ntasks=5 --cpus-per-task=2 --time=01:30:00 "$T"
sbatch -J perf-r9-2node --time=01:30:00 experiments/hpc/hera_cpu_mpi_multinode.sbatch.example
```

Set `CHAINS`, `WARMUP`, `SAMPLES`, and `SEED` identically across comparisons. The multi-node wrapper passes its environment to the performance template. The baseline and performance templates use shared Work for temporary output, so account for its I/O behavior when comparing configurations.

After the jobs finish, collect one run-record row per job and the per-cell diagnostics with [collect_run_record.py](collect_run_record.py). It needs the `diagnostics` extra (`conda install -c conda-forge "arviz>=1.3,<2"` in the HERA environment):

```bash
python experiments/hpc/collect_run_record.py --output "$WORK_ROOT/hera-cpu-mpi-run-record.csv" JOBID [JOBID ...]
```

Each row merges `runs/JOBID/run-metadata.txt`, `sacct`, and the grid NetCDF. `wall_seconds` and `peak_memory_mib` use the grid-fit `srun` step, excluding environment setup and the MPI check (whole-job elapsed time is in `notes`). `ess_per_second` is the sum over fitted cells of each cell's minimum bulk ESS divided by that wall time. `rank_utilization` reports the cells planned per fitting rank and the resulting busy fraction of all ranks; rank 0 counts as idle when it is only the writer. Per-cell rows are written to `runs/JOBID/cell-diagnostics.csv` by [cell_diagnostics.py](cell_diagnostics.py), which computes ArviZ rank-normalized R-hat and bulk/tail ESS over the non-time-varying parameters (`sigma_*`, `rhoAR1`, `beta`) and reports each cell's worst value. The [run record CSV template](hera_cpu_mpi_run_record.csv.example) and [cell diagnostics CSV template](hera_cpu_mpi_cell_diagnostics.csv.example) define the columns. Inspect per-cell diagnostics; aggregate throughput must not hide a poorly mixing cell. Compare end-to-end wall time and ESS/sec, not only likelihood timings.

The grid NetCDF contains posterior draws and, for runs made after per-cell divergence storage was added, a `divergent_transitions` variable that fills the divergence columns. It does not retain per-cell CmdStan CSVs. The initial sweep and first full-grid run predate divergence storage; later runs below include it. Latent time series (`trend`, `slope`, `ar`, `seasonal`) are excluded from the per-cell summary.

Edit `OUTPUT_VARIABLES` in [dlm_lat_alt_mpi_run.py](../../dlm_lat_alt_mpi_run.py)
to choose retained posterior variables and single/double storage precision;
see [Output variables and precision](../../MPI-README.md#output-variables-and-precision).
The default now saves every posterior variable in single precision (float32).
All measurements below, including full-grid job `34668570`, were recorded with
the earlier all-double (float64) output, so their output sizes and transfer and
write costs are not directly comparable with single-precision runs. Keep the
same output configuration when comparing layouts. The diagnostics collector can
assess only retained non-time-varying parameters and requires at least one such
parameter.

### Initial measurements (2026-11-05)

All MPI measurements below used 4 chains, 500 warmup iterations, 500 retained draws, seed `20261104`, and the listed cell set. The one-cell runs used flat ID 0; the grid runs used the fixed 16-cell set above.

| Job | Configuration | Wall time (s) | Cells/hour | Max R-hat | ESS/s |
|---|---|---:|---:|---:|---:|
| `34659225` | Plain `dlmmc.fit()`, 1 cell | 460 | — | not measured | — |
| `34659293` | MPI, 1 rank, 1 cell, 1 CPU | 559 | 6.440 | 1.005 | 1.904 |
| `34659679` | MPI, 1 rank, 1 cell, 2 CPUs | 487 | 7.392 | 1.005 | 2.185 |
| `34659297` | MPI, 9 ranks, 1 node, 1 CPU/task | 1,557 | 36.994 | 1.009 | 12.20 |
| `34659298` | MPI, 5 ranks, 2 CPUs/task, 2 parallel chains | 1,411 | 40.822 | 1.009 | 13.46 |
| `34659299` | MPI, 9 ranks, 2 nodes, 1 CPU/task | 1,453 | 39.642 | 1.009 | 13.07 |
| `34659296` | MPI, 5 ranks, 1 node, 1 CPU/task | 2,734 | 21.068 | 1.009 | 6.948 |
| `34659976` | MPI, 1 rank, 1 cell, 1 CPU, progress off | 554 | 6.498 | 1.005 | 1.921 |
| `34659294` | MPI, 1 rank, 16 cells, 1 CPU/task | 9,134 | 6.306 | 1.009 | 2.080 |
| `34659295` | MPI, 3 ranks, 1 node, 1 CPU/task | 4,713 | 12.222 | 1.009 | 4.031 |
| `34660583` | MPI, 9 ranks, 1 node, 2 CPUs/task | 1,368 | 42.105 | 1.009 | 13.89 |
| `34660584` | MPI, 5 ranks, 1 node, 2 CPUs/task, 2 parallel chains | 1,411 | 40.822 | 1.009 | 13.46 |
| `34665311` | MPI, 5 ranks, 1 node, 2 CPUs/task, 1 parallel chain | 2,364 | 24.365 | 1.009 | 8.036 |
| `34665312` | MPI, 20 ranks, 1 node, 2 CPUs/task, 2 parallel chains, 32 cells | 1,040 | 110.769 | 1.010 | 32.43 |
| `34665313` | MPI, 10 ranks, 1 node, 4 CPUs/task, 4 parallel chains, 32 cells | 1,060 | 108.679 | 1.010 | 31.82 |
| `34666833` | MPI, 40 ranks, 1 node, 1 CPU/task, 1 parallel chain, 39 cells | 1,371 | 102.407 | 1.010 | 30.88 |

The multi-node job completed every selected cell. Turning off progress bars changed the one-cell wall time by less than 1% relative to job `34659293`, so it did not recover the one-CPU performance difference in this comparison. R-hat was computed for the MPI runs only; the plain `dlmmc.fit()` baseline recorded 0 divergences but did not retain draws for an R-hat calculation. These are exploratory measurements, not convergence guarantees.

All jobs in the initial sweep and both instrumented reruns completed with exit code 0. Across MPI runs, the maximum per-cell R-hat was 1.009; the one-cell baseline's R-hat was not measured. The 9-rank, 2-CPU/task rerun had the shortest wall time (1,368 seconds), 43 seconds ahead of the 5-rank, 2-CPU/task, 2-parallel-chain run. Every single-node job was allocated and billed a whole 40-CPU node (`sacct` reports `cpu=40` and the same billing TRES), so a layout requesting fewer CPUs per node is not cheaper; it leaves cores idle.

The instrumented reruns logged fit durations for all 16 cells. Per-cell times ranged from 442–702 seconds (mean 584) for job `34660583` (9 ranks, 2 CPUs/task, `parallel_chains=1`) and 242–383 seconds (mean 327) for job `34660584` (5 ranks, 2 CPUs/task, `parallel_chains=2`). Per-cell diagnostics are under `WORK_ROOT/runs/JOBID/cell-diagnostics.csv`; the complete run records are in `WORK_ROOT/hera-cpu-mpi-run-record.csv`.

A "cell" is one `(pressure, latitude)` grid point, i.e. one single-time-series DLM fit (4 chains × (500 warmup + 500 draws) here). `cells/hour` is fitted cells divided by the grid-fit `srun` step's wall time; environment setup and the MPI check are excluded.

### Interpretation and recommended layout

The 16-cell totals mostly reflect how many cells each worker received (2 with eight workers, 4 with four), so they cannot rank layouts on their own. Per-cell `fit_seconds` from the job's `.err` log (`grep 'dlmmc.mpi cell' WORK_ROOT/dlmmc-hera-perf-JOBID.err`) and steady-state node throughput (fitting workers ÷ mean per-cell time) are more informative.

Layout tests completed on 2026-11-05 (all exit code 0, one whole 40-CPU node each). `Node throughput` below is the measured `cells/hour` from the run record: fitted cells divided by the grid-fit step's wall time. It is not calculated from mean per-cell sampler time.

| Job | Layout | Cells | Mean (min–max) per-cell fit time (s) | Node throughput (cells/h) | Wall time (s) | Peak RSS/task (MiB) |
|---|---|---:|---|---:|---:|---:|
| `34665311` | 5 × 2 CPUs, `PARALLEL_CHAINS=1` | 16 | 552 (425–657) | 26 | 2,364 | 1,390 |
| `34660584` | 5 × 2 CPUs, `PARALLEL_CHAINS=2` | 16 | 327 (242–383) | 44 | 1,411 | — |
| `34665312` | 20 × 2 CPUs, `PARALLEL_CHAINS=2` | 32 | 462 (301–661) | 111 | 1,040 | 1,698 |
| `34665313` | 10 × 4 CPUs, `PARALLEL_CHAINS=4` | 32 | 229 (166–325) | 109 | 1,060 | 1,387 |
| `34666833` | 40 × 1 CPU, `PARALLEL_CHAINS=1` | 39 | 941 (698–1,052) | 102 | 1,371 | 2,327 |

- **Parallel chains matter.** In the controlled pair `34665311`/`34660584` (same allocation, only `PARALLEL_CHAINS` differs), two parallel chains cut mean per-cell time by 41% (552 → 327 s) and wall time by 40%. With `PARALLEL_CHAINS=1` the second CPU per rank is mostly idle.
- **Fill the node.** Both full-node layouts fit the 32 cells in about 17.5 minutes, about 3.4 times the throughput of the best 16-cell partial-node layout.
- **The full-node layouts are close.** 20 × 2 recorded 111 cells/hour, 10 × 4 recorded 109, and 40 × 1 recorded 102. Differences of this size from single runs are not conclusive, but 40 × 1 did not improve throughput despite using all 39 workers. The measured wall throughput includes sampling, data transfer, and output writing; mean sampler time alone does not capture those costs.
- **40 × 1 trades lower worker count per cell rate for more concurrent fits.** Its 39 cells ran one per fitting worker, with a mean logged fit time of 941 s (compared with 462 s under 20 × 2 and 229 s under 10 × 4). The overall wall throughput was 102 cells/hour.
- **Diagnostics were consistent.** The 39-cell 40 × 1 run had maximum R-hat 1.0099 and minimum bulk ESS 623. The worst 32-cell point from the full-node comparison, ID 281, had R-hat 1.0104, minimum bulk ESS 653, and minimum tail ESS 282, slightly above the common 1.01 threshold; review that cell or use more draws before relying on its full-grid results. These runs predate per-cell divergence storage, so their divergences are unknown.
- **Memory is not a constraint.** The peak was below 1.7 GiB per task, or under 34 GiB for 20 tasks on a 92 GB node; rank 0 writes each cell as it arrives.

**Recommendation: one node, 20 ranks × 2 CPUs/task, `PARALLEL_CHAINS=2`.** It had the highest measured wall throughput, but only marginally; regard it and 10 × 4 as practically tied. The 40 × 1 layout is a viable alternative but was about 8% slower in measured cells/hour. At the observed 102–111 cells/hour, a 312-cell grid is roughly 2.8–3.1 hours before allowance for static-assignment imbalance (each of 19 workers gets 16 or 17 cells). The full-grid run below used a 4-hour request and finished in 2 h 41 min (116 cells/hour).

### Full-grid run (2026-11-06)

Job `34668570` fitted every valid cell with the recommended layout: one node, 20 ranks × 2 CPUs/task, `PARALLEL_CHAINS=2`, 4 chains × (500 warmup + 500 draws), seed `20261104`, and all 312 valid flat IDs passed explicitly (IDs 3–8 and 15–20 are constant zeros). It was submitted from the repository root with:

```bash
T=experiments/hpc/hera_cpu_mpi_performance.sbatch.example
ALL_VALID="$(python -c 'print(" ".join(str(i) for i in range(324) if not (3 <= i <= 8 or 15 <= i <= 20)))')"
PERF_CELL_INDICES="$ALL_VALID" PARALLEL_CHAINS=2 sbatch -J full-r20x2-pc2 --ntasks=20 --cpus-per-task=2 --time=04:00:00 "$T"
```

| Metric | Value |
|---|---|
| Job state | `COMPLETED`; `fit_status` 312 fitted, 12 skipped, 0 failed |
| Grid-fit wall time | 9,663 s (2 h 41 min); whole job 9,673 s |
| Throughput | 116.2 cells/hour, about 570 s of wall time per cell per worker |
| Cells per fitting worker | 16 or 17; 91.8% of rank time busy (rank 0 counts as idle) |
| R-hat (per-cell worst parameter) | median 1.004, max 1.017 |
| Bulk / tail ESS (per-cell minimum) | min 532 / 180, median 1,057 / 599 |
| Peak memory (largest task) | 6,994 MiB |
| Output | 7.43 GB, all variables in double precision |

Four cells had a worst-parameter R-hat above 1.01: flat IDs 132 (1.0174), 90 (1.0158), 26 (1.0107), and 281 (1.0104). This job predates per-cell divergence storage, so divergences are unknown. The throughput matched the 32-cell layout test (111 cells/hour), so 570 s/cell is the calibration used for `--dry-run` in HERA.md; the dry run predicts 2:41:30 for this grid.

### Single-precision and resume smoke test (2026-11-10)

Jobs `34827576` and `34827577` ran [hera_production.sbatch.example](hera_production.sbatch.example) (20 ranks × 2 CPUs, `PARALLEL_CHAINS=2`, 4 chains × (200 warmup + 200 draws), seed `20261104`) with `RESUME=1` and one fixed `OUTPUT_FILE`. The first job fitted the first 19 IDs of the 32-cell set in 5 min 41 s. The second, chained with `--dependency=afterany`, was given all 32 IDs, reported "13 of 324 (19 already complete ...)", and fitted only the 13 new cells. The resulting file had float32 posterior draws, float64 `*_mean` variables (matching the draw means to about 6e-8), int32 `divergent_transitions` (0 in every cell), the prior and sampler attributes, and a `history` entry for the resume. It was 262 MB for 32 cells. Maximum R-hat was 1.03, as expected for 200 draws.

### High-R-hat re-fit (2026-11-10)

Job `34827902` re-fitted the four flat IDs above (132, 90, 26, 281) with 4 chains × (1000 warmup + 1000 draws), `PARALLEL_CHAINS=4`, and 5 tasks × 4 CPUs. It finished in 5 min 55 s. Every cell now passes:

| Flat ID | Max R-hat (500+500) | Max R-hat (1000+1000) | Min bulk ESS | Min tail ESS | Divergences |
| --- | --- | --- | --- | --- | --- |
| 26 | 1.0107 | 1.0011 | 1423 | 1063 | 0 |
| 90 | 1.0158 | 1.0009 | 1956 | 868 | 0 |
| 132 | 1.0174 | 1.0012 | 2065 | 1025 | 0 |
| 281 | 1.0104 | 1.0014 | 2402 | 1332 | 0 |

Increasing warmup and retained draws resolved these flags in this test; it does not by itself establish model adequacy. Use 1000+1000 or re-fit flagged cells, and check the resulting diagnostics rather than assuming an iteration count guarantees convergence.

### Single-precision full-grid run (2026-11-10 to 2026-11-11)

Job `34827903` fitted the 312 valid cells with the same model, seed, and layout as the earlier full-grid job: one node, 20 ranks × 2 CPUs/task, `PARALLEL_CHAINS=2`, 4 chains × (500 warmup + 500 draws), seed `20261104`. It completed successfully in 2 h 24 min 46 s (129.3 cells/hour); the dry-run estimate was 2 h 41 min 30 s before its safety margin. All 312 selected cells completed, and the 12 constant-zero cells were skipped.

| Metric | Value |
|---|---|
| R-hat (per-cell worst parameter) | median 1.004, max 1.017 |
| Bulk / tail ESS (per-cell minimum) | min 532 / 180 |
| Divergences | 5 transitions across 5 cells (IDs 58, 92, 112, 141, 146; max 1 per cell) |
| Peak memory (largest task) | 6.19 GiB |
| Output variable storage types | posterior draws float32; `*_mean` float64; divergence counts int32 |
| NetCDF file size | 7,695,494,207 bytes (7.70 GB) |

The four cells flagged in the earlier full-grid run again had R-hat above 1.01: IDs 132 (1.0174), 90 (1.0158), 26 (1.0107), and 281 (1.0104). The higher-draw re-fit in job `34827902` brought these cells below 1.0015, but that separate re-fit is not incorporated into this full-grid file.

**Initial storage finding:** despite storing posterior draws as float32, this file is about 3.5% larger than the earlier all-float64 file (7,434,190,077 bytes). Both use zlib level 4 with shuffle enabled. The investigation below identified unused HDF5 space, rather than ineffective float32 compression, as the cause.

### Storage investigation and output choices (2026-11-12)

`h5stat -S` separated live compressed data from unused/unaccounted file space:

| File | Total bytes | Live data bytes | Unaccounted bytes |
|---|---:|---:|---:|
| Earlier float64 grid | 7,434,190,077 | 6,755,832,495 | 678,176,408 |
| Float32 grid before compaction | 7,695,494,207 | 3,022,328,911 | 4,672,961,915 |
| Float32 grid after `nccopy` | 3,014,575,403 | 3,014,350,754 | 4,625 |

The remainder in each row is file metadata. Float32 did reduce the live compressed data, but the per-cell writer exposed an HDF5 allocation problem: default chunks span multiple cells (for the latent states, 9 pressures by 4 latitudes). Updating one cell rewrites shared compressed chunks. Closing and reopening after each result, which keeps partial results readable on interruption, can leave the old allocations unused.

A four-variable, 16-cell reproducer with zlib level 4 grew to 15.79 MB with shared chunks, versus 2.78 MB with cell-aligned chunks. The shared-chunk case took 1.72 s to write versus 0.19 s for cell-aligned chunks; these are small synthetic measurements, not full-grid performance claims.

The fix on `main` and both HPC branches gives each compressed chunk a grid extent of one cell and caps its uncompressed size at 4 MiB. It applies to draws and posterior means, preserves the close-after-each-cell behavior, and still accepts older files on resume. Existing files retain their chunk layout; compact them with `nccopy`, or also rechunk them for further writes with `nccopy -c pressure/1,latitude/1 input.nc output.nc`. See [MPI-README.md](../../MPI-README.md#output-variables-and-precision).

The full-grid compacted copy is in `<WORK_ROOT>/storage-analysis/full-single-nccopy.nc`. Every variable, value, dtype, dimension, and NetCDF attribute was compared with the original and matched exactly. The original is unchanged. The copy saves 4.68 GB (60.8%).

Compression-level comparison on the four latent variables for real cells 0, 26, 132, and 281, using cell-aligned chunks:

| zlib level | File size | Write time |
|---|---:|---:|
| 1 | 37.26 MB | 1.21 s |
| 4 | 36.97 MB | 1.44 s |
| 9 | 36.50 MB | 9.06 s |

Keep level 4: level 9 saved only 1.3% in this sample but cost over six times as much write time. Changing the chunk layout is the important fix.

The four latent variables account for over 99% of the compacted grid's live data. Copies retaining different variable sets measured:

| Saved posterior variables | File size | What remains available |
|---|---:|---|
| All (current default) | 3.01 GB | All component posterior draws |
| `trend` plus scalar/regressor parameters | 549.73 MB | Trend uncertainty and parameter R-hat/ESS |
| Scalar/regressor parameters only | 18.88 MB | Parameter distributions and R-hat/ESS, but no latent time-series draws |
| Means only (`OUTPUT_VARIABLES = {}`) | 61.65 kB | Posterior means, fit status, and divergences; no credible intervals or R-hat/ESS |

Every profile keeps coordinates, status/divergence counts, and default posterior means. Sizes describe copies of this run, not guarantees for other data or chunk layouts. Keep all variables as the package default to preserve analyses; recommend `trend` plus `sigma_trend`, `sigma_AR`, `rhoAR1`, `sigma_seas`, and `beta` when the user does not need the other latent components. Omitting outputs saves storage and MPI traffic, not fitting work.

### High-R-hat refit with corrected chunks (2026-11-12)

Job `34877319` repeated the four high-R-hat cells (26, 90, 132, 281) with 4 chains x (1000 warmup + 1000 draws), 5 ranks x 4 CPUs, and `PARALLEL_CHAINS=4`. This both fulfills the requested refit and validates the corrected writer in a real MPI job. It completed with exit code 0 in 5 min 58 s.

All four cells have zero divergences. Their R-hat and ESS values match the earlier higher-draw refit table exactly (maximum R-hat 1.0014, minimum bulk ESS 1422.6, minimum tail ESS 868.4). Every saved draw and posterior mean for the fitted cells matches the earlier refit exactly, while every compressed posterior/mean variable now has cell-aligned chunks.

The new file is 73,902,011 bytes (73.90 MB), versus 197,475,838 bytes before the fix: a 62.6% reduction with unchanged values. `h5stat -S` reports 73,826,288 bytes of live data, 69,749 bytes of metadata, and only 5,974 unaccounted bytes. Results and diagnostics are in `<WORK_ROOT>/refit-rhat-cellchunks/`. They remain a separate higher-draw output, not replacements in the 500+500 full grid.

### Divergent-cell refit (2026-11-12)

Job `34877828` re-fitted the five cells that each had one divergence in the 500+500 full grid. It used the same model and seed (`20261104`), 4 chains x (1000 warmup + 1000 draws), 6 ranks x 4 CPUs, and `PARALLEL_CHAINS=4`. The job completed successfully in 6 min 32 s. The table reports each cell's worst R-hat and minimum ESS across the saved non-time-varying parameters, not the latent time series.

| Flat ID | Divergences, 500+500 | Divergences, 1000+1000 | Max R-hat, 1000+1000 | Min bulk ESS | Min tail ESS |
|---|---:|---:|---:|---:|---:|
| 58 | 1 | 0 | 1.0031 | 1944.4 | 1193.8 |
| 92 | 1 | 0 | 1.0023 | 2365.1 | 1501.6 |
| 112 | 1 | 0 | 1.0024 | 1515.9 | 1009.9 |
| 141 | 1 | 0 | 1.0018 | 2043.4 | 1206.4 |
| 146 | 1 | 0 | 1.0016 | 2292.1 | 1408.1 |

These are encouraging diagnostics, but zero divergences in one longer refit does not prove that the posterior has no problematic regions. The higher-draw results are in `<WORK_ROOT>/divergent-cells-refit/` and are separate from the original grid.

**Numerical warning:** the refit log contains five CmdStanPy "Non-fatal error during sampling" blocks reporting a NaN scale in `normal_lpdf` at the [Kalman likelihood](../../src/dlmmc/_stan_models.py#L179), which uses `sqrt(C_y[t])`. The original full-grid log contains 312 such blocks for 312 successful fits. The aggregated messages do not establish the affected sampling phase, cause, or impact, and are distinct from the stored divergence counts. Investigate these warnings before treating the output as scientifically final.

## Conclusions

- The CmdStanPy/MPI path works on HERA with Intel MPI and `srun --mpi=pmi2`, single- and multi-node, with rank 0 as the sole writer.
- Use one whole node as 20 ranks × 2 CPUs with `--parallel-chains` equal to the CPUs per task. 10 × 4 is practically tied; 40 × 1 is about 8% slower.
- Peak memory was 6.19 GiB in the latest full-grid run. Its compacted float32 output is 3.01 GB; the writer now uses cell-aligned chunks to avoid the unused-space growth found in that run.
- Run time is set by the busiest worker, which `--dry-run` reports. dlmmc deals the cells to fit round-robin to the workers; the earlier layout tests used `flat_index % (nranks - 1)`, which gives the same or nearly the same counts.
- Grids that need more than one job's wall time can be split across jobs with `--resume` (see HERA.md).

## Open items

1. ~~Smoke-test the single-precision output, divergences, means, and resume in real MPI jobs.~~ Done: jobs `34827576` and `34827577`.
2. ~~Re-fit flat IDs 132, 90, 26, and 281 with more warmup and draws.~~ Done: job `34827902`; all below R-hat 1.0015.
3. ~~Rerun the full grid with single-precision output to measure the file size and confirm the run time.~~ Done: job `34827903`; it completed in 2 h 24 min 46 s, but its file was 3.5% larger than the prior double-precision file.
4. ~~Investigate compression and output-variable choices.~~ Done: unused HDF5 space caused the growth; the chunk-layout fix is on `main` and both HPC branches. Keep zlib level 4 and all variables by default; optional slimmer profiles are measured above.
5. ~~Validate the corrected writer in a real MPI refit of high-R-hat IDs 26, 90, 132, and 281.~~ Done: job `34877319`; max R-hat 1.0014, no divergences, unchanged draws, and a 62.6% smaller file.
6. ~~Re-fit the five divergent cells (IDs 58, 92, 112, 141, 146) with more warmup and draws.~~ Done: job `34877828`; all have zero divergences and worst R-hat at most 1.0031. This does not establish that the posterior has no problematic regions.
7. Investigate the recurring non-fatal NaN likelihood-scale warnings in the original and refit logs, including when they occur and whether they affect inference. Preserve the relevant raw sampler logs in a targeted follow-up if needed.
8. Decide how to use the successful higher-draw refits alongside the original grid. They are separate outputs: different iteration counts cannot be resumed into the 500+500 file because resume requires matching settings.

## Decision Gate

Use CmdStanPy on HERA as the scientific and performance reference. Do not recommend a new backend based on these tests: they evaluate the existing CmdStanPy/MPI path and its resource configuration. Keep site-specific recommendations in [HERA.md](../../HERA.md); update host-neutral [MPI-README.md](../../MPI-README.md) only for generally reusable guidance.
