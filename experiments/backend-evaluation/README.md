- `python experiments/backend-evaluation/hpc_environment_probe.py` emits a JSON hardware/software/device report without installing anything; `--output path.json` keeps framework logs separate.
# Experimental Backend Evaluation

This directory contains scratch-to-share validation code for comparing CmdStanPy, PyMC, and TensorFlow Probability (TFP). It is exploratory only; the production package remains CmdStanPy-based.

See [GPU_RUNBOOK.md](GPU_RUNBOOK.md) for the portable GPU evaluation workflow and [AGENTS.md](AGENTS.md) for experiment constraints and the staged roadmap.

## Environments

The tests were run in separate Conda environments to avoid dependency conflicts:

- `dlmmc-cmdstanpy2`: CmdStanPy 1.3.0, ArviZ 1.3.0
- `dlmmc-pymc-eval`: PyMC 6.3.2, pymc-extras 0.15.1, nutpie 0.16.11, JAX 0.12.0.1, NumPy 2.5.4
- `dlmmc-tfp-eval`: TensorFlow 2.18.1, TFP 0.25.0, TensorFlow Metal on an Apple M4 Pro
- `dlmmc-local-optimizations`: Python 3.12.16, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.1, emcee 3.1.6, mpi4py 4.1.2 with MPICH 5.0.2
- `dlmmc-torch-mps`: Python 3.12.16, PyTorch 2.9.1 with the MPS backend

Run commands from the repository root. CmdStan requires a working C++ toolchain.
To recreate the exploratory environments, use:

```bash
conda env create -f experiments/backend-evaluation/environment-pymc-eval.yml
conda env create -f experiments/backend-evaluation/environment-tfp-metal.yml
```

For local CPU JIT testing, an isolated environment can be created with:

```bash
conda create -n dlmmc-local-optimizations -c conda-forge --strict-channel-priority \
	python=3.12 numpy scipy numba pip
```

The latest PyPI PyTorch wheel tested (2.15.0) collided with the conda-forge OpenMP runtime in the Numba environment. The vendor MPS build is installed in a separate environment:

```bash
conda create -n dlmmc-torch-mps --override-channels -c defaults \
	python=3.12 'pytorch=2.9.1=*gpu_mps*'
```

These are local evaluation environments, not HERA-certified specifications. In particular, `environment-tfp-metal.yml` is for macOS/Apple Metal only. RDHPCS policy requires Miniforge with conda-forge rather than Anaconda/Miniconda. For HERA, first verify available GPU nodes, scheduler requests, site modules, and supported GPU runtime on an allocation; create a separate environment using the site-approved setup. See [AGENTS.md](AGENTS.md) for the HPC evaluation workflow and official RDHPCS references.

## Preparing an Ursa GPU Probe

The public [Ursa User Guide](https://docs.rdhpcs.noaa.gov/systems/ursa_user_guide.html) documents `u1-h100` (two NVIDIA H100 GPUs per node, `gpu` or `gpuwf` QoS), plus exploratory `u1-gh` and `u1-mi300x` partitions (`gpuwf`). Start with one H100 GPU using windfall access if the project does not have a dedicated GPU allocation. Confirm current access, module versions, and driver/runtime on Ursa before choosing a framework environment; the macOS Metal environment above is not suitable there.

The probe script prints a JSON report of Python/package versions, loaded modules, scheduler environment, MPI, and available TensorFlow/JAX/PyTorch devices. It also performs a tiny device-placement matrix operation. From an Ursa allocation, first inspect modules with `module spider`, then edit the activation block, account placeholder, and GPU request directive in `ursa_h100_probe.sbatch.example` to match the live site configuration before submitting it. The template is intentionally not directly submittable. Keep environments in project home and job outputs in the site-approved work/scratch area; Ursa HPFS scratch is not backed up.

The same staged procedure applies to other workstations and clusters, but the GPU runtime and environment must be selected for each host. The current package has no production GPU backend: use CmdStanPy for the CPU/MPI control and treat the TFP scripts below as fixed-parameter experiments. See [GPU_RUNBOOK.md](GPU_RUNBOOK.md) for the parity gates and fair end-to-end comparison method.

## Reproduce The Real-Cell Fit Comparison

The comparison uses pressure index 22, latitude index 8, 396 monthly observations, the five regressors used by `dlm_benchmark.py`, and known time-varying `o3_sigma`. Both runs use four chains, 500 warmup iterations, 500 retained draws, and the same model priors.

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/cmdstan_real_cell.py
conda run -n dlmmc-pymc-eval python experiments/backend-evaluation/pymc_real_cell.py
```

The CmdStan script writes its inputs, draws, and a fixed-parameter likelihood/gradient reference to `/tmp/dlmmc-backend-eval-real-cell-cmdstan.npz` by default; the PyMC script reads that file and writes `/tmp/dlmmc-backend-eval-real-cell-pymc.npz`.

For the NumPyro comparison, generate a CmdStan reference and then run the CPU NumPyro model from the user-provided `jax_dlm` environment:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/cmdstan_real_cell.py \
	--output /tmp/dlmmc-backend-eval-real-cell-cmdstan-numpyro.npz
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	XLA_FLAGS=--xla_force_host_platform_device_count=4 \
	python experiments/backend-evaluation/numpyro_real_cell.py \
	--reference /tmp/dlmmc-backend-eval-real-cell-cmdstan-numpyro.npz \
	--chain-method parallel --chains 4 --warmup 500 --samples 500
```

NumPyro checks fixed likelihood/gradient parity and skips sampling if it fails. The four forced JAX CPU devices are logical devices for mapping parallel chains, not additional physical cores. It also reproduces the simulation-smoothed `trend`, `slope`, `seasonal`, `ar`, and `beta` draws in bounded batches.

For a matched multi-cell CPU comparison, create one fixed reference set, then run CmdStan MPI and vectorized NumPyro on the same cell IDs:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/tfp_real_grid_reference.py \
	--cells 8 --output /tmp/dlmmc-torch-real-grid-cmdstan.npz
PYTHONPATH=src conda run -n dlmmc-cmdstanpy2 mpiexec -n 4 \
	python dlm_lat_alt_mpi_run.py \
	--output /tmp/dlmmc-cmdstan-multicell-grid-pc4.nc \
	--cell-indices 0 14 28 42 56 70 84 98 \
	--chains 4 --parallel-chains 4 --warmup 500 --samples 500 \
	--seed 20261102 --overwrite
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	XLA_FLAGS=--xla_force_host_platform_device_count=4 \
	python experiments/backend-evaluation/numpyro_real_grid.py \
	--reference /tmp/dlmmc-torch-real-grid-cmdstan.npz \
	--cmdstan-reference /tmp/dlmmc-cmdstan-multicell-grid-pc4.nc \
	--chains 4 --chain-method parallel --warmup 500 --samples 500 \
	--seed 20261102 --output /tmp/dlmmc-numpyro-multicell-posterior-full.npz
```

The CmdStan run uses three fitting workers, each sampling four chains in parallel; the NumPyro run vectorizes the independent-cell likelihood inside each of four parallel CPU chains. Both save all public smoother outputs. This is a fixed eight-cell workload, not the full grid.

Numba 0.68.1 is installed in `dlmmc-local-optimizations`. To benchmark its forward likelihood kernel against the same fixed CmdStan reference:

```bash
conda run -n dlmmc-local-optimizations python experiments/backend-evaluation/numba_real_cell.py \
	--reference /tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz --repeats 100
```

This compiles the CPU kernel on its first call and reports cold compile-plus-call separately from warm likelihood calls. It also checks a finite-difference gradient against CmdStan; that gradient is a parity diagnostic, not an autodiff sampler backend.

The single-cell Torch check is:

```bash
conda run -n dlmmc-torch-mps python experiments/backend-evaluation/torch_real_cell.py \
	--reference /tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz --device both
```

To compare batched independent cells, first create a CmdStan reference, then run:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/tfp_real_grid_reference.py \
	--cells 32 --output /tmp/dlmmc-torch-real-grid32-cmdstan.npz
conda run -n dlmmc-torch-mps python experiments/backend-evaluation/torch_real_grid.py \
	--reference /tmp/dlmmc-torch-real-grid32-cmdstan.npz --device cpu --threads 1
```

The Torch scripts use CPU float64 and MPS float32 because MPS rejects float64. They report numerical errors explicitly; this is a fixed likelihood/gradient test, not a posterior sampler or full fit. Use `--threads 1`, `4`, or `10` to compare CPU thread settings.

## Reproduce The Real-Grid Likelihood/Gradient Comparison

The reference script chooses 32 approximately evenly spaced valid cells from the 27-by-12 ozone grid, skipping all-missing and constant series. CmdStanPy's diagnostic `log_prob` evaluates the same constrained parameters in each cell and saves the real data, references, and gradients. The TFP script batches those cells and compares CPU and Metal execution.

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/tfp_real_grid_reference.py
conda run -n dlmmc-tfp-eval python experiments/backend-evaluation/tfp_real_grid_benchmark.py
```

The reference is saved at `/tmp/dlmmc-tfp-real-stan-reference.npz`.

## Results So Far

### PyMC Real-Cell Fit

CmdStanPy completed in 142.4 s; PyMC/nutpie completed in 180.5 s. Both had zero divergences. CmdStan R-hat values were about 1.00, with bulk ESS around 1,200-2,100. PyMC's largest R-hat was about 1.010, with bulk ESS around 670-4,000. Posterior means were close:

| Parameter | CmdStanPy | PyMC |
| --- | ---: | ---: |
| `sigma_trend` | 0.000155 | 0.000154 |
| `sigma_seas` / `sigma_season` | 0.00208 | 0.00211 |
| `sigma_AR` / `sigma_ar` | 0.0910 | 0.0912 |
| `rhoAR1` / `params_ar` | 0.329 | 0.330 |
| `beta[ENSO]` | 0.164 | 0.166 |
| `beta[SOLAR]` | 0.101 | 0.102 |
| `beta[QBO30]` | 0.081 | 0.081 |
| `beta[QBO50]` | 0.020 | 0.020 |
| `beta[SAOD]` | 0.012 | 0.016 |

PyMC's R-hat for `sigma_season` was about 1.01 and its bulk ESS about 670; treat this as encouraging one-cell evidence, not definitive convergence. The leading missing observation in PyMC is intentional: it represents the unobserved prior-time state before Stan scores the first real observation. PyMC emitted the expected state-space imputation warning for that leading missing value. CmdStanPy emitted non-fatal NaN-scale warnings after sampling despite zero divergences; those should be investigated separately.

### Numba CPU Likelihood Kernel

In `dlmmc-local-optimizations` with Numba 0.68.1, the `vanilla_ar1` forward likelihood matched the fixed CmdStan value with absolute error `1.1e-10`. Finite-difference gradient maximum relative error was `1.7e-7`. First-call JIT compile plus evaluation took `2.46 s`; median warm evaluation took `0.413 ms`.

That is a kernel result, not a fit speedup: Numba does not provide the autodiff gradient needed for NUTS here, nor smoothing, sampling, or posterior output. CmdStan is already compiled C++; a realistic backend comparison still needs an end-to-end sampler.

### Numba Parallel Cell Batch

The `numba_real_grid.py` benchmark applies the likelihood and finite-difference gradient independently across cells using `numba.prange`:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/tfp_real_grid_reference.py \
	--cells 32 --output /tmp/dlmmc-numba-grid32-cmdstan.npz
conda run -n dlmmc-local-optimizations python experiments/backend-evaluation/numba_real_grid.py \
	--reference /tmp/dlmmc-numba-grid32-cmdstan.npz --threads 1 4 10 --repeats 5
```

For 32 cells, likelihood time fell from 12.35 ms at one thread to 1.94 ms at 10 threads (about 6.4x); finite-difference gradient time fell from 99.7 ms to 13.8 ms. Likelihood error was at most `4.4e-9`; maximum finite-difference gradient relative error was `3.9e-6`. This is a fixed-kernel result, not a posterior sampler or end-to-end fit.
### Numba-Vectorized emcee Pilot

`emcee` calls the Numba `prange` log-posterior with `vectorize=True`; the sampler itself uses neither JAX nor PyMC. Install emcee and an MPI implementation in the isolated local environment:

```bash
conda run -n dlmmc-local-optimizations python -m pip install emcee
conda install -n dlmmc-local-optimizations -c conda-forge mpich mpi4py
conda run -n dlmmc-local-optimizations python experiments/backend-evaluation/numba_emcee_real_cell.py \
	--walkers 32 --threads 10 --warmup 1000 --samples 5000 --seed 20261102
```

Two one-cell runs took 13.7-13.9 s including JIT, warmup, and sampling. Acceptance was 0.584-0.586, with estimated ESS of about 2,945-3,735 per parameter. A deterministic subset of 2,528 draws from each seed was passed through the existing JAX CPU simulation smoother; state posterior-mean RMSEs were 2.4-4.2% of CmdStan posterior SD.

The matched eight-cell pilot used the cells `0, 14, 28, 42, 56, 70, 84, 98`, four fitting workers, two Numba threads per worker, 32 walkers, 1,000 warmup steps, and 5,000 retained steps per walker. On macOS, the tested MPICH setup requires `FI_PROVIDER=tcp`:

```bash
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-torch-real-grid-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid-mpi-pilot.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 5000
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	python experiments/backend-evaluation/numba_emcee_grid_smoother_check.py
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-torch-real-grid-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid-mpi-repeat-seed20261103.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 5000 --seed 20261103
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	python experiments/backend-evaluation/numba_emcee_grid_compare.py
time PYTHONPATH=src conda run -n dlmmc-cmdstanpy2 mpiexec -n 5 \
	python dlm_lat_alt_mpi_run.py \
	--output /tmp/dlmmc-cmdstan-multicell-grid-pc8-long.nc \
	--cell-indices 0 14 28 42 56 70 84 98 \
	--chains 4 --parallel-chains 2 --warmup 1000 --samples 2000 --seed 20261104
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	python experiments/backend-evaluation/numba_emcee_grid_compare.py \
	--cmdstan-reference /tmp/dlmmc-cmdstan-multicell-grid-pc8-long.nc \
	--output /tmp/dlmmc-numba-emcee-grid-comparison-pc8-long.json
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-torch-real-grid-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid-mpi-long-seed20261104.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 10000 --seed 20261104
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	python experiments/backend-evaluation/numba_emcee_grid_compare.py \
	--pilots /tmp/dlmmc-numba-emcee-grid-mpi-pilot.npz \
	/tmp/dlmmc-numba-emcee-grid-mpi-repeat-seed20261103.npz \
	/tmp/dlmmc-numba-emcee-grid-mpi-long-seed20261104.npz \
	--cmdstan-reference /tmp/dlmmc-cmdstan-multicell-grid-pc8-long.nc \
	--output /tmp/dlmmc-numba-emcee-grid-comparison-3seeds.json
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-numba-grid32-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid32-long-seed20261105.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 10000 --seed 20261105
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-numba-grid32-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid32-long-seed20261106.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 10000 --seed 20261106
time PYTHONPATH=src conda run -n dlmmc-cmdstanpy2 mpiexec -n 5 \
	python dlm_lat_alt_mpi_run.py \
	--output /tmp/dlmmc-cmdstan-grid32-pc8-long.nc \
	--cell-indices 0 13 27 34 41 48 54 61 68 75 82 89 96 103 109 116 \
	123 130 137 144 151 158 164 171 178 185 192 199 206 213 219 226 \
	--chains 4 --parallel-chains 2 --warmup 1000 --samples 2000 --seed 20261105
conda run -n jax_dlm env JAX_PLATFORMS=cpu \
	python experiments/backend-evaluation/numba_emcee_grid_compare.py \
	--pilots /tmp/dlmmc-numba-emcee-grid32-long-seed20261105.npz \
	/tmp/dlmmc-numba-emcee-grid32-long-seed20261106.npz \
	--reference /tmp/dlmmc-numba-grid32-cmdstan.npz \
	--cmdstan-reference /tmp/dlmmc-cmdstan-grid32-pc8-long.nc \
	--output /tmp/dlmmc-numba-emcee-grid32-comparison-two-seeds.json
```

Two independent emcee seeds took 94.1 s and 85.1 s for sampling/JIT/output; required JAX CPU smoothing added 19.0 s and 18.5 s, for 113.1 s and 103.6 s end to end. Mean acceptance was 0.556 and 0.555. Minimum estimated ESS by parameter across cells ranged from 2,403 to 3,176. Against the longer CmdStan posterior, the largest parameter mean error was 3.5%, SD error 3.1%, and 5th/50th/95th percentile error 9.7% of CmdStan posterior SD. For states, average posterior-mean error across cells was 2.1-2.6%, with worst-cell error 4.4%; average posterior-SD error was 1.7-2.9%, with worst-cell error 4.7%. Each smoother check used 2,528 draws per cell.

The longer CmdStan run used the same eight concurrent compute slots (four MPI workers, two chains per worker), 1,000 warmup and 2,000 retained draws per chain, and took 695.8 s end to end. It completed every cell, with maximum hyperparameter R-hat 1.0017 and minimum bulk ESS 3,084. ArviZ bulk ESS/sec summed over cells was 49.6-71.1 by parameter. Emcee's integrated-autocorrelation ESS/sec including state smoothing was 204.7-258.8, or 3.1-4.7x the CmdStan rate, across the two seeds. This is promising evidence, not a production decision: emcee's walkers are dependent and do not provide independent-chain R-hat, its ESS relies on finite-chain autocorrelation estimates, the worst parameter quantile discrepancy reached 9.7%, and CmdStan retained the known non-fatal Stan NaN-scale warnings. Repeat longer runs and check per-cell diagnostics before scaling beyond eight cells. CmdStan remains the production backend.

A third seed with 10,000 retained steps per walker completed sampling in 159.7 s; JAX CPU smoothing 5,008 draws per cell added 34.8 s, for 194.6 s end to end. Mean acceptance was 0.556. Minimum emcee ESS across cells was 4,456-5,214, above the long CmdStan reference's 3,084 minimum bulk ESS. The maximum parameter mean, SD, and 5th/50th/95th percentile errors were 2.4%, 2.0%, and 5.4% of CmdStan SD. State posterior-mean error averaged 1.7-1.9% across cells with a 2.5% worst cell; state-SD error averaged 1.0-1.7% with a 2.4% worst cell. Full-output emcee ESS/sec was 221.6-244.3 by parameter, 3.4-4.7x CmdStan's rate. This longer run strengthens the eight-cell CPU result, but there is still no emcee independent-chain R-hat; test multiple independent ensembles/per-cell diagnostics before a larger grid or production proposal.

The 32-cell pilot used the fixed IDs `0, 13, 27, 34, 41, 48, 54, 61, 68, 75, 82, 89, 96, 103, 109, 116, 123, 130, 137, 144, 151, 158, 164, 171, 178, 185, 192, 199, 206, 213, 219, 226`, four MPI workers, and two concurrent compute slots per worker. Two emcee seeds used 32 walkers, 1,000 warmup, and 10,000 retained steps per walker. Sampler wall times were 647.8 s and 638.6 s; JAX CPU smoothing added 131-140 s, for about 778-787 s end to end. Both acceptance rates were 0.551; minimum estimated ESS across cell/parameter combinations ranged from 4,214 to 5,369. The matched CmdStan run used four chains with two concurrent chains per worker, 1,000 warmup and 2,000 retained draws; it took 2,633.9 s. All cells completed; max hyperparameter R-hat was 1.0024 and min bulk ESS was 3,106.

For 32 cells, ArviZ CmdStan bulk ESS/sec summed over cells was 57.2-81.7 by parameter. Emcee's integrated-autocorrelation ESS/sec including smoothing was 213.7-247.6 across both seeds, an apparent 3.0-4.0x advantage. Parameter mean/SD differences were at most 2.4%/2.1% of CmdStan posterior SD; the largest parameter quantile error was 6.6%. State mean errors were at most 3.9% and state-SD errors at most 3.7% of CmdStan posterior SD in the worst cell. CmdStan emitted the known non-fatal NaN-scale warnings. The independent-seed replication supports the 32-cell CPU result, but emcee ESS remains a finite-ensemble autocorrelation estimate, not independent-chain R-hat. Consider a larger representative pilot only after reviewing per-cell tail diagnostics; keep CmdStanPy as production/reference.

### Numba emcee: Diverse 64-Cell Grid

The 64-cell set combines 48 evenly spread valid cells with the 16 cells having the highest observation-noise-to-series-SD ratios. The source grid has 312 valid nonconstant cells and no missing observations. Generate the fixed reference and run both emcee seeds on exactly those IDs:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/tfp_real_grid_reference.py \
	--cell-indices 0 12 24 28 29 30 31 32 36 40 44 49 53 57 65 69 73 81 85 89 \
	98 102 106 114 118 122 130 131 134 139 147 151 155 163 167 171 179 183 \
	188 196 200 204 212 216 220 228 233 237 245 249 253 256 261 265 268 273 \
	280 304 306 316 317 318 319 320 --output /tmp/dlmmc-numba-grid64-diverse-cmdstan.npz
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-numba-grid64-diverse-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid64-seed20261107.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 10000 --seed 20261107
conda run -n dlmmc-local-optimizations env FI_PROVIDER=tcp NUMBA_NUM_THREADS=2 \
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 5 \
	python experiments/backend-evaluation/numba_emcee_grid_mpi.py \
	--reference /tmp/dlmmc-numba-grid64-diverse-cmdstan.npz \
	--output /tmp/dlmmc-numba-emcee-grid64-seed20261108.npz \
	--walkers 32 --threads-per-worker 2 --warmup 1000 --samples 10000 --seed 20261108
```

The matched CmdStan run used four workers and two concurrent chains per worker, with 1,000 warmup and 2,000 retained draws. It took 5,166.7 s (86m07s); all cells completed, max hyperparameter R-hat was 1.0026, and min bulk ESS was 3,033. CmdStan emitted the known non-fatal NaN-scale warnings.

Both emcee seeds used 32 walkers and 10,000 retained steps, four MPI workers, and two Numba threads per worker. Sampling took 1,347.9 s and 1,327.4 s; JAX CPU state smoothing added 278.9 s and 267.4 s, for 1,626.8 s and 1,594.8 s end to end. Mean acceptance was 0.5604 for both; minimum ESS across cells/parameters ranged from 4,285 to 5,283. CmdStan aggregate bulk ESS/sec was 62.8-83.4 by parameter; emcee ESS/sec including smoothing was 215.4-251.0, an apparent 2.9-3.7x difference. The largest parameter mean/SD/quantile discrepancy was 1.9%/1.9%/5.9% of CmdStan posterior SD. State means were within 4.2% and state SDs within 4.7% in the worst cell. This supports a larger experimental pilot, not production adoption: emcee's walkers are dependent and its autocorrelation ESS is not independent-chain R-hat. Keep CmdStanPy as the production backend.

### Numba emcee: 96-Cell Follow-Up

The 96-cell follow-up reused the same eight-slot allocation and two emcee seeds. CmdStan completed all cells in 7,713.2 s (128m33s), with max hyperparameter R-hat 1.0038 and min bulk ESS 2,596. The emcee runs took 2,297.7-2,415.4 s including smoothing; their minimum estimated ESS ranged from 3,879 to 5,247. Aggregate means looked close, but per-cell review found a clear mixing failure: for cell 160, seed 1's `sigma_seas` mean was about 0.00242 with SD 0.00527 versus CmdStan 0.00160 with SD 0.00074; seed 2 matched CmdStan. Several walkers remained in the high-`sigma_seas` mode throughout both halves of seed 1, so its aggregate autocorrelation ESS overstated convergence. That cell's seasonal-state mean/SD discrepancy reached 14%/48% of CmdStan SD. Seed 2 had other state-SD outliers, up to 23.5% for trend and 18.5% for AR in cell 125. Do not treat this pilot as a clean scale-up pass. Investigate cell-level multimodality and independent-ensemble diagnostics, starting with cells 160 and 125; keep CmdStanPy as production/reference.

A focused rerun on cells 125 and 160 used 128 walkers, 2,000 warmup, and 10,000 retained steps for two independent seeds. Both targeted runs matched CmdStan's hyperparameter means/spreads at both cells; cell 160's `sigma_seas` walker-group split did not recur. Re-smoothing about 80,000 parameter draws per cell reduced state mean errors to at most 1.4% and state-SD errors to at most 3.6% of CmdStan SD; at cell 125, the worst trend-state SD error was 3.6%, down from 23.5% in the lower-draw 96-cell check. This supports a 128-walker mitigation for these cells, but not a full-grid claim: validate difficult cells under the same production settings and retain per-cell diagnostics.

Four independent 128-walker ensembles were also used to estimate rank-normalized split-R-hat, taking walker 0 from each ensemble as one chain and never treating within-ensemble walkers as independent. Across cells 125 and 160, max R-hat was 1.00993 (`rhoAR1`, cell 160); other parameter maxima were 1.0061 or lower. Minimum bulk ESS of these four selected trajectories ranged from 593 to 767 across parameters/cells. This is a useful independent-ensemble check but has only four chain representatives and is just below the common 1.01 screening threshold at its worst cell; it does not certify every walker or the full grid. The diagnostic script is `numba_emcee_independent_rhat.py`.

A full 96-cell rerun with 128 walkers, 2,000 warmup, and 10,000 retained steps took 8,941.5 s for sampling/JIT and 1,601.3 s for smoothing (10,542.8 s end to end). All cells had finite ESS; minimum estimated ESS ranged from 17,600 to 22,500. Against CmdStan, parameter mean/SD errors were at most 1.5%/1.35% of posterior SD, and state mean/SD errors at most 2.68%/2.59%. At cell 160 the seasonal-state mean/SD errors were 1.07%/1.00%; at cell 125 the trend-state SD error was 1.05%. This 128-walker configuration removes the earlier cell-160 discrepancy in a single full-grid seed. It still needs an independent 96-cell seed and per-cell convergence review before the full-grid or production gate is cleared.

To reproduce the independent-ensemble calculation from saved targeted runs:

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/numba_emcee_independent_rhat.py \
	--runs /tmp/dlmmc-numba-target-emcee-seed20261113.npz \
	/tmp/dlmmc-numba-target-emcee-seed20261114.npz \
	/tmp/dlmmc-numba-target-emcee-seed20261115.npz \
	/tmp/dlmmc-numba-target-emcee-seed20261116.npz \
	--output /tmp/dlmmc-numba-target-independent-rhat.json
```

### PyTorch CPU And MPS Kernel

The separate `dlmmc-torch-mps` environment uses PyTorch 2.9.1's MPS build. Basic matmul/autograd works on `mps:0`. The single-cell Torch implementation matched the fixed CmdStan likelihood and gradient in float64 on CPU, but an eager evaluation took about `46 ms`. MPS requires float32: the same single-cell likelihood took about `265 ms`, with absolute log-likelihood error `0.089` and maximum relative gradient error `6.1e-4`.

Batching 32 independent cells in Torch took 153-158 ms on CPU float64 at 1, 4, or 10 threads, with no meaningful thread scaling. Mean likelihood absolute error was `7.2e-10` and maximum gradient relative error `4.4e-11`. The earlier eight-cell batch took `82 ms` on CPU and `268 ms` on MPS float32; MPS was about 3.3x slower and less accurate. These are fixed likelihood/gradient results; no Torch sampler or complete-fit speedup is demonstrated.

### Affine Sampler Smoke

The public `justinalsing/affine` v0.1 source was installed into a temporary directory only. Its two-half stretch-move sampler completed 101 states for 20 walkers on a toy Gaussian target on CPU in `1.064 s`; about 73% of walker positions changed. The same test on Metal failed at `tf.math.is_nan` with soft placement disabled, reporting that only CPU supported the operation. This is a package/API smoke, not a DLM posterior test; it exposes no built-in seeding or convergence diagnostics and its source is from 2021.

### NumPyro CPU Real-Cell Fit

On the same pressure-22/latitude-8 real cell, four chains, 500 warmup iterations, 500 retained draws, float64, priors, and CmdStanPy reference, NumPyro 0.19.0 matched the fixed-parameter likelihood to `2.3e-10` and the maximum relative gradient error to `1.35e-12`. Both NumPyro runs had zero divergences; R-hat ranged from 0.9993 to 1.0044 and mean acceptance probability was about 0.90.

The CmdStanPy fit took 47.78 s with four parallel chains. NumPyro's two four-chain parallel CPU runs took 22.05 s and 22.86 s total, including JIT/warmup (about 1.1 s in those runs); retained sampling took 20.95 s and 21.78 s. NumPyro was faster for this one-cell test at the same nominal draws and comparable diagnostics. This is encouraging CPU evidence, not a grid-scale result or a universal speed claim; repeat on representative cells and compare full fit/output workflows before drawing a backend conclusion.
### NumPyro Versus CmdStan: Eight-Cell Grid

The fixed cell indices were `0, 14, 28, 42, 56, 70, 84, 98` from the 27×12 grid, each with 396 observations and five regressors. Both backends used four chains, 500 warmup, and 500 retained draws per cell; the fixed likelihood/gradient parity gate passed (maximum likelihood error `9.5e-10`, maximum gradient relative error `9.1e-12`).

| Backend/configuration | End-to-end time | Max R-hat | Minimum bulk ESS | Divergences / status |
| --- | ---: | ---: | ---: | --- |
| CmdStan MPI: 3 cell workers × 4 parallel chains | 4m22.5s | 1.0103 | 691 | 8/8 cells completed; non-fatal Stan NaN-scale warnings |
| NumPyro: 4 parallel CPU chains, 8 cell targets vectorized | 4m34.4s | 1.0011 | 1,592 | 0 divergences |

NumPyro was about 4.5% slower end to end on this eight-cell test, despite higher ESS and tighter R-hat; it is effectively at parity in wall time for this workload. The NumPyro total includes JIT/warmup, sampling, simulation smoothing, and compressed output. State-posterior mean RMSEs were 2.2–3.8% of the CmdStan posterior SD, and the compressed NPZ was 184 MB versus 163 MB for CmdStan's NetCDF. This is a promising CPU result, not a GPU result or full-grid benchmark.

The CmdStan runner assigns three workers and each uses four chain processes (12 chain processes total). NumPyro uses four logical JAX CPU devices; these are mapped devices, not extra physical cores. The 8-cell target’s cell-specific parameters are conditionally independent, but the prototype samples their product posterior in one batched model. Do not extrapolate to different grid sizes, per-cell allocation patterns, or a full 324-cell grid without another benchmark.

The NumPyro means closely tracked CmdStanPy: `sigma_trend` 0.0001540-0.0001545 vs 0.0001532, `sigma_seas` 0.002068-0.002070 vs 0.002070, `sigma_AR` 0.09129-0.09135 vs 0.09125, and `rhoAR1` 0.3299-0.3314 vs 0.3326. The JAX simulation smoother now generates the same public state outputs as CmdStan. Across 2,000 draws, posterior-mean RMSEs were 2.2% of CmdStan posterior SD for trend, 2.8% for slope, 3.1% for seasonal, 3.0% for AR, and 2.2% for beta. The smoother took 1.84 s after sampling. CmdStan reported zero divergences but also emitted the previously observed non-fatal NaN-scale warnings after sampling.

The parallel run used `XLA_FLAGS=--xla_force_host_platform_device_count=4` to map the four chains over four logical CPU devices. This does not add physical cores. The `jax_dlm` Metal device still fails a basic operation; this is CPU-only NumPyro evidence.

### TFP Likelihood And Gradient

TFP's faithful float32 sequential likelihood and gradient were explicitly placed on GPU:0 with TensorFlow soft device placement disabled. On the synthetic case, float32 TFP log likelihood differed from an independent double-precision Kalman recursion by about `4.3e-6` at fixed parameters. On the 32 real cells, mean absolute per-cell likelihood error was `0.216` on CPU and `0.213` on Metal (maximum errors `1.41` and `0.98`, respectively). Maximum aggregate-gradient relative errors were `0.021%` on CPU and `0.009%` on Metal.

For the real 32-cell batch, the median TFP likelihood-plus-gradient call took about `0.085` s on CPU and `70.9` s on Metal. The sequential Metal path was roughly 830 times slower for this workload. The CmdStanPy reference took about `0.97` s for all 32 cells, but `log_prob` launches a diagnostic subprocess per cell and is explicitly not an efficient API; this is not a fair end-to-end sampler benchmark.

### Sampling And Other GPU Experiments

- A prior-matched PyMC synthetic four-chain run showed close posterior summaries to CmdStanPy. The real-cell PyMC/nutpie runner also used JAX on CPU only (`jax.devices()` returned `CpuDevice(0)`).
- A TFP NUTS smoke test required a local float-backed workaround for TensorFlow Metal's integer `stateless_uniform` kernel. A short run completed, but its chain count and draw count were too small for convergence claims.
- A corrected 4-chain TFP Metal NUTS run did not return within a 10-minute limit. A parallel-filter TFP sampler run aborted in Apple's MPSGraph compiler (`MLIR pass manager failed`).
- A separate TFP STS smoke test ran on Metal, but it did not include the full model or establish a speedup.
- The JAX Metal prototype failed on a basic operation with `UNIMPLEMENTED: default_memory_space is not supported`.
- In the separate `jax_dlm` environment, JAX 0.8.2 and NumPyro 0.19.0 detect the M4 Pro `METAL:0` device, but a basic matmul fails with the same `default_memory_space` error. Forcing `JAX_PLATFORMS=cpu` runs the NumPyro DLM, and the simulation-smoothed outputs now agree with CmdStan in posterior-mean checks. This remains CPU-only; there is no local JAX GPU sampling result.
- Current probes find PyTorch absent from `jax_dlm` and `dlmmc-tfp-eval`; the separate `dlmmc-torch-mps` has PyTorch 2.9.1, and `dlmmc-local-optimizations` has Numba 0.68.1. TensorFlow 2.18.1 in `dlmmc-tfp-eval` sees `/device:GPU:0`. The Torch MPS and affine results above are kernel/API tests, not complete-fit speedups.
- Current probes find PyTorch absent from `jax_dlm` and `dlmmc-tfp-eval`; the separate `dlmmc-torch-mps` has PyTorch 2.9.1, and `dlmmc-local-optimizations` has Numba 0.68.1. TensorFlow 2.18.1 in `dlmmc-tfp-eval` sees `/device:GPU:0`. The Torch MPS and affine results above are kernel/API tests, not complete-fit speedups.

## Interpretation

The real-cell PyMC fit is encouraging as a CPU comparison, and TFP's batched likelihood/gradient is numerically close to CmdStanPy. There is no demonstrated GPU speedup: the tested TFP sequential Metal batch was much slower, and the parallel sampler path failed. Package behavior and dependencies are unchanged; these scripts and environment files are isolated under `experiments/backend-evaluation/`. Keep CmdStanPy as the stable package backend until a larger grid fit demonstrates both parity and an end-to-end benefit.

## Local MPI Scaling Smoke Sweep

Run the repeatable sweep from a CmdStanPy/MPI-enabled environment as described in [MPI-README.md](../../MPI-README.md#choosing-mpi-ranks-and-chains). For example:

```bash
python experiments/backend-evaluation/mpi_rank_sweep.py \
	--ranks 1 2 4 8 14 --cells 16 --chains 1 --parallel-chains 1 \
	--warmup 10 --samples 20 --repeats 2
```

On the local Apple M4 Pro (14 physical/logical CPU cores, 24 GiB RAM), this tested 16 evenly spaced valid, nonconstant cells, one chain per cell, 10 warmup iterations, 20 retained draws, and two repeats per rank count. The model was precompiled outside the trials. All cells completed in every trial.

| Total MPI ranks | Fitting workers | Median elapsed time |
| ---: | ---: | ---: |
| 1 | 1 serial fitter | 29.04 s |
| 2 | 1 | 30.80 s |
| 4 | 3 | 13.98 s |
| 8 | 7 | 8.76 s |
| 14 | 13 | 9.07 s |

Eight total ranks were fastest in this small sweep; fourteen ranks were slightly slower. These short chains and selected cells only establish a scaling smoke result, not a scientific or workload-independent optimum. Re-run with representative cells and scientifically adequate sampling settings before choosing production resources. The generated logs, NetCDF outputs, and JSON report are ignored under `results/`; the local summary is `results/mpi-rank-sweep-mac-20261101/summary.json`.
