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
