# Colleague testing

Install the pinned sampler and this checkout using [INSTALL.md](INSTALL.md). Start from the repository root with the task-local environment active.

```sh
python -m pip check
python -m pytest tests -m 'not stan' -q
python -m pytest .deps/affine/tests/test_torch.py .deps/affine/tests/test_torch_stream.py -q
```

On an Apple Silicon machine, enable native tests explicitly in the process environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python -m pytest tests -m 'not stan' -q
```

Tests skip unavailable hardware; a skip is not a qualification pass. CUDA tests should be run on an actual CUDA GPU before claiming CUDA support has been validated. The optional `stan` integration test requires an installed CmdStan toolchain and is separate from the default command. CI currently covers CPU pull requests and manual workflow dispatch; it does not claim hardware or scientific posterior qualification.

For a bounded fresh-kernel tutorial check (all four notebooks, no new long scientific fits):

```sh
python scripts/check_notebooks.py --output-dir results/notebook-smoke
```

The script executes a fresh kernel per notebook, preserves the checked-in notebooks, writes executed copies/plots and records checksums and timings. Inspect the generated plots. Long-fit switches must remain false, even when historical scope selections are present. The script uses the active interpreter; install the `notebooks` extra first. New scripts or science jobs should write to new directories rather than overwrite checkpoint evidence.

For scientific exploration use the README single-panel example, then the five-panel tensor tutorial. The default independent production profile is 128 walkers /128 burn-in /256 retained with one ensemble. Its output is an estimate with monitoring flags, not a convergence certificate. Keep qualification at five independent panels maximum and four-cell joint references. Spatial and lag post-pilot scopes require separate explicit selections after their respective runtime estimates. No full-grid benchmark, long coupled-chain continuation or 200-replication campaign is part of these commands.

Report the commit, platform, Python/Torch/Numba versions, backend/dtype, model, seed, exact command, input checksum and full traceback or discrepancy. Never treat walkers as independent chains. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for known limitations before interpreting a discrepancy as a new regression.

## Opt-in full BASIC CPU/MPS timing comparison

[dlm_grid_benchmark.py](dlm_grid_benchmark.py) runs the 312 non-placeholder cells
(27 pressure levels x 12 latitudes, excluding 12 all-zero ozone/uncertainty
locations), with all 396 months from 1985 through 2017 and the five supplied
regressors. It reuses the MPI example's regressor loader, not the cached fits.
Install the CPU dependencies from [INSTALL.md](INSTALL.md), plus the NetCDF reader:

```sh
conda activate dlmmc-tensor
python -m pip install 'netCDF4>=1.6'
python dlm_grid_benchmark.py --output-dir results/basic-cpu-mps-new \
  --numba-threads 8 \
  --approval "Run the full 312-cell BASIC comparison on MPS and parallel CPU"
```

Run only with explicit full-grid approval, recorded verbatim by `--approval`.
The script starts fresh MPS and Numba processes **sequentially**, with MPS
fallback/fast math disabled. It uses `vanilla_ar1`, one ensemble, 128 walkers,
128 burn-in and 256 retained sweeps, then 2,048 selected draws per cell for
analytic trend summaries anchored at January 1998. Both backends use the same
seeds, input order, model and budgets; CPU uses float64 and MPS float32. This is
not a matched-ESS benchmark or a convergence/equivalence certificate.

The new output directory contains per-backend logs, checkpoints, diagnostic
flags, 312 trend files per backend, and `comparison.json`. Timings separate
burn-in/retained sampling, initialization, compilation/cache loading, disposable
warmup, monitoring and output. Whole-process time includes startup and all
setup; post-compile pipeline time excludes compilation and disposable warmup
but **includes burn-in**. The CPU smoother's full-size disposable warmup can
be substantial. There is no plotting, MPI, Stan or coupled spatial fit.
Existing compiler caches may be reused; one run per backend is a local
measurement, not a hardware-independent speedup claim. Existing output is
never overwritten, and incomplete fits fail rather than report success.

## Saturated CmdStan subset benchmark

[dlm_cmdstan_benchmark.py](dlm_cmdstan_benchmark.py) estimates full-grid CmdStan
cost without fitting all 312 cells. It reuses the full BASIC loader above and
calls the same public `dlmmc.fit` API used by the MPI grid runner. Python thread
workers launch independent, single-threaded CmdStan **processes**; Python is
not performing the sampling. Chains are queued independently, avoiding nested
parallelism and keeping workers occupied when cell costs differ.

With the existing `dlmmc-tensor` environment and a working CmdStan toolchain:

```sh
conda activate dlmmc-tensor
python -m pip install 'cmdstanpy>=1.2' psutil
python dlm_cmdstan_benchmark.py \
  --output-dir results/basic-cmdstan-new \
  --affine-report results/basic-cpu-mps-20261115/comparison.json \
  --cells 16 --workers 14 --chains 4 --warmup 1000 --samples 1000 \
  --seed 2026111601 --chain-timeout 1800
```

`--affine-report` must point to a completed local CPU/MPS benchmark report; the
driver verifies input provenance and model agreement before fitting. Existing
output directories are rejected. The driver chooses one seeded random cell
from each of 16 contiguous pressure-major grid strata and records the named
selection before sampling. All 396 months and all five regressors are used.
Its default Stan budget is the public API default, **not** the affine budget.
The worker limit cannot exceed the available logical CPUs, and numerical-library
thread counts are limited to one. A timed-out or incomplete chain fails the
benchmark rather than contributing a success-shaped extrapolation.

The report retains every chain's timing, CSV, diagnostics and trend moments,
plus sampled counts of active Stan processes. `batch_seconds` includes the
parallel fits, CSV loading, per-chain checks and summary-file writes. The
separately measured pooled trend summaries combine all four chains. Compilation
is performed once before the parallel batch. Interpreter imports and MPI
rank-0 NetCDF aggregation are not timed by this driver.

## Laptop timing results, November 15-16, 2026

**Apple M4 Pro, 14 CPU cores, 24 GiB memory**, Python 3.13.18. CmdStan 2.39.0 /
CmdStanPy 1.3.0; Torch 2.15.0 / Numba 0.68.1. The versioned
[compact measurement record](docs/laptop-backend-timings-20261116.json) preserves
settings, selected cells, diagnostics and hashes of the full local reports.
Raw output remains in the git-ignored results directories identified there.

| Approach | Scope actually measured | Time for 312 cells | Relative to MPS total |
|---|---|---:|---:|
| Affine MPS float32 | All 312 cells | **281.44 s / 4m 41s** | 1.00x |
| Affine Numba CPU float64, 8 threads | All 312 cells | **1,273.53 s / 21m 14s** | 4.53x |
| CmdStan float64, 14 chain workers | 16 cells, 64 chains | **18,443.37 s / 5h 07m, estimated** | **65.53x, estimated** |

These are **configured-workflow cost ratios, not equal-precision or
convergence-qualified sampler speedups**. The affine runs used one ensemble,
128 walkers, 128 burn-in and 256 retained sweeps, followed by analytic trend
summaries using 2,048 integration draws per cell. CmdStan used four independent
NUTS chains per cell with 1,000 warmup and 1,000 retained draws each. Its retained
iterations also generate state trajectories and write CSV; the output contracts
and achieved precision are different. CmdStan had 14 CPU workers whereas the
earlier Numba measurement used eight threads. See
[Sampling budgets and effective sample size](#sampling-budgets-and-effective-sample-size)
for ESS-normalized ratios.

### Measured CmdStan saturation and extrapolation

- All **64/64 chains** completed, yielding **64,000 retained draws** over 16
  cells. The observed peak was **14 simultaneous Stan processes**. Busy-worker
  occupancy was **92.7%**, including worker-side parsing/I/O; this is not a
  measurement of CPU utilization.
- Parallel subset batch: **944.73 s (15m 45s)**. One-time setup: **12.59 s**,
  including **11.97 s** compilation. Pooled trend summaries: **0.44 s**.
  The measured driver body took **957.76 s (15m 58s)**.
- Direct extrapolation:
  `(944.7263 + 0.4419) * (312 / 16) + 12.5872 = 18,443.37 s`, or about
  **5h 07m**. Compilation/setup is charged once, not scaled by 19.5.
- An ideal-load-balance calculation, `sum(chain-task seconds) / 14 * 19.5`,
  gives **17,079.95 s**, or about **4h 45m**, before one-time setup and pooled
  summaries. The direct extrapolation repeats the small batch's startup/drain
  penalty; a long dynamically scheduled run may amortize it better.
- Thus **roughly 4.75-5.12 hours** is a useful planning estimate under these
  assumptions, **not a confidence interval**. The direct estimate is about
  **14.48x** the measured affine CPU total and **65.53x** the MPS total.
  No full-grid CmdStan fit was performed.

Individual chain-task times ranged from **116.64 to 262.15 s** (median
**196.46 s**). The extrapolation assumes this preselected subset represents the
other cells' costs. It does not establish a rigorous runtime bound; machine
load, thermal conditions, cell difficulty and output-storage throughput can
change the result. This is not a measured MPI speedup or an MPI wall-time
prediction including its writer.

### Setup costs and diagnostic limitations

The measured affine sampling times, including burn-in, were **160.99 s MPS**
and **540.93 s CPU**. Excluding compilation/cache loading and disposable
warmups, their full pipelines took **277.96 s** and **923.10 s**, respectively.
The CPU total in the table includes a **343.65 s full-size disposable smoother
warmup**. This is why total elapsed ratios differ from warmed-pipeline ratios.

The CmdStan run had **one post-warmup divergence** (`p13-lat02`, chain 2) and
**zero maximum-treedepth hits**. The largest rank-normalized split Rhat over
`sigma_trend`, `sigma_seas`, `sigma_AR` and `rhoAR1` was **1.00513**; this is not
a check of every scientific summary. **56/64 chain logs contained 84 nonfatal
NaN-scale messages from `normal_lpdf`**. The chains completed, and retained
sampler diagnostics, monitored parameters and trend arrays were checked for
finiteness; the warnings were retained, not suppressed or called a clean
qualification pass. The affine single-ensemble/short-window flags also remain.
None of these timing measurements certifies convergence or matched ESS.

### Sampling budgets and effective sample size

The approaches did not produce the same number of draws per cell:

| Per cell | Affine MPS and Numba | CmdStan |
|---|---|---|
| Discarded tuning | 128 sweeps x 128 walkers = 16,384 walker states | 4 chains x 1,000 = 4,000 warmup iterations |
| Retained sampler output | 256 sweeps x 128 walkers = 32,768 correlated walker states | 4 chains x 1,000 = **4,000 draws** |
| Draws used for trend summaries | **2,048** (8 random walkers per sweep); states integrated analytically | **4,000** sampled trend trajectories |

Raw counts are not comparable because autocorrelation differs. For the same 16
cells, the minimum posterior-mean ESS over `sigma_trend`, `sigma_seas`,
`sigma_AR` and `rhoAR1` was (median over cells, with range):

| Approach | Hyperparameter ESS | Trend MCSE / posterior SD, median (worst) |
|---|---:|---:|
| CmdStan | **3,007** (1,766-3,805) | 0.0161 (0.0181) |
| Affine MPS | **845** (366-3,096) | 0.0040 (0.0130) |
| Affine Numba | **760** (349-1,389) | 0.0044 (0.0126) |

CmdStan ESS and MCSE use ArviZ `method="mean"` over the four chains' retained
draws; trend values are from the `trend` columns of the retained CSVs. Affine values
are `mean_ess_nominal` from `diagnose_production` and `mcse_over_posterior_sd`
from the trend receipts. The affine estimates use a walker-averaged
autocorrelation time and are provisional (single ensemble, short window). Trend
MCSE is evaluated at January 1985, July 2001 and December 2017.

CmdStan therefore obtained about **3.6-4.0x more hyperparameter ESS** per cell.
Two linear projections to matched hyperparameter ESS keep burn-in/warmup fixed
and assume ESS is proportional to retained iterations:

| Full-grid projection | CmdStan / MPS | CmdStan / Numba |
|---|---:|---:|
| Configured workflows (above) | 65.5x | 14.5x |
| Affine retained sweeps raised to CmdStan ESS (MPS 556 s, Numba 2,330 s) | **33.2x** | **7.9x** |
| CmdStan retained draws cut to affine ESS (12,580 s and 12,348 s) | **44.7x** | **9.7x** |

For trend means, the comparison reverses: analytic state integration
(Rao-Blackwellization) gives affine about 4x smaller median MCSE relative to
posterior SD (about 1.8x at each cell's worst checked month), despite fewer
output draws. Matching affine's trend precision would require roughly 3.4-16x
more retained CmdStan draws, or about 2.0-7.8x its total chain time, so the
configured-workflow ratios understate affine's advantage for trend means. These
projections are estimates from 16 cells, not measurements; tail ESS was not
compared because the affine production diagnostics do not report it.
