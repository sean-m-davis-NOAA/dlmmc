# Experimental Backend Evaluation

This directory contains scratch-to-share validation code for comparing CmdStanPy, PyMC, and TensorFlow Probability (TFP). It is exploratory only; the production package remains CmdStanPy-based.

## Environments

The tests were run in separate Conda environments to avoid dependency conflicts:

- `dlmmc-cmdstanpy2`: CmdStanPy 1.3.0, ArviZ 1.3.0
- `dlmmc-pymc-eval`: PyMC 6.3.2, pymc-extras 0.15.1, nutpie 0.16.11, JAX 0.12.0.1, NumPy 2.5.4
- `dlmmc-tfp-eval`: TensorFlow 2.18.1, TFP 0.25.0, TensorFlow Metal on an Apple M4 Pro

Run commands from the repository root. CmdStan requires a working C++ toolchain.
To recreate the exploratory environments, use:

```bash
conda env create -f experiments/backend-evaluation/environment-pymc-eval.yml
conda env create -f experiments/backend-evaluation/environment-tfp-metal.yml
```

## Reproduce The Real-Cell Fit Comparison

The comparison uses pressure index 22, latitude index 8, 396 monthly observations, the five regressors used by `dlm_benchmark.py`, and known time-varying `o3_sigma`. Both runs use four chains, 500 warmup iterations, 500 retained draws, and the same model priors.

```bash
conda run -n dlmmc-cmdstanpy2 python experiments/backend-evaluation/cmdstan_real_cell.py
conda run -n dlmmc-pymc-eval python experiments/backend-evaluation/pymc_real_cell.py
```

The CmdStan script writes its inputs and draws to `/tmp/dlmmc-backend-eval-real-cell-cmdstan.npz`; the PyMC script reads that file and writes `/tmp/dlmmc-backend-eval-real-cell-pymc.npz`.

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

### TFP Likelihood And Gradient

TFP's faithful float32 sequential likelihood and gradient were explicitly placed on GPU:0 with TensorFlow soft device placement disabled. On the synthetic case, float32 TFP log likelihood differed from an independent double-precision Kalman recursion by about `4.3e-6` at fixed parameters. On the 32 real cells, mean absolute per-cell likelihood error was `0.216` on CPU and `0.213` on Metal (maximum errors `1.41` and `0.98`, respectively). Maximum aggregate-gradient relative errors were `0.021%` on CPU and `0.009%` on Metal.

For the real 32-cell batch, the median TFP likelihood-plus-gradient call took about `0.085` s on CPU and `70.9` s on Metal. The sequential Metal path was roughly 830 times slower for this workload. The CmdStanPy reference took about `0.97` s for all 32 cells, but `log_prob` launches a diagnostic subprocess per cell and is explicitly not an efficient API; this is not a fair end-to-end sampler benchmark.

### Sampling And Other GPU Experiments

- A prior-matched PyMC synthetic four-chain run showed close posterior summaries to CmdStanPy. The real-cell PyMC/nutpie runner also used JAX on CPU only (`jax.devices()` returned `CpuDevice(0)`).
- A TFP NUTS smoke test required a local float-backed workaround for TensorFlow Metal's integer `stateless_uniform` kernel. A short run completed, but its chain count and draw count were too small for convergence claims.
- A corrected 4-chain TFP Metal NUTS run did not return within a 10-minute limit. A parallel-filter TFP sampler run aborted in Apple's MPSGraph compiler (`MLIR pass manager failed`).
- A separate TFP STS smoke test ran on Metal, but it did not include the full model or establish a speedup.
- The JAX Metal prototype failed on a basic operation with `UNIMPLEMENTED: default_memory_space is not supported`.

## Interpretation

The real-cell PyMC fit is encouraging as a CPU comparison, and TFP's batched likelihood/gradient is numerically close to CmdStanPy. There is no demonstrated GPU speedup: the tested TFP sequential Metal batch was much slower, and the parallel sampler path failed. Package behavior and dependencies are unchanged; these scripts and environment files are isolated under `experiments/backend-evaluation/`. Keep CmdStanPy as the stable package backend until a larger grid fit demonstrates both parity and an end-to-end benefit.
