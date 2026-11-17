# Choosing a sampler and compute backend

Use `fit(..., sampler="affine" | "cmdstan", backend=..., settings=...)`. `"stan"` is an alias for `"cmdstan"`. Sampler counts have separate typed settings; passing settings for the wrong sampler fails before fitting.

```python
from dlmmc import (fit, Trend, Harmonics, Regression, AR,
                   AffineSettings, StanSettings, summarize_trends)

model = Trend() + Harmonics((12, 6)) + Regression() + AR(1)
affine = fit(data, regressors, uncertainty=uncertainty, model=model,
             sampler="affine", backend="mps", settings=AffineSettings(),
             output_dir="results/affine")
trend = summarize_trends(affine, output_dir="results/affine-trends")

stan = fit(data, regressors, uncertainty=uncertainty, model=model,
           sampler="cmdstan", backend="cpu", settings=StanSettings(),
           seed=42, output_dir="results/stan")
stan_trend = summarize_trends(stan)
```

These are two separate fits; choosing a sampler never runs both automatically. CmdStan and its compiler must be installed separately. See INSTALL.md for task-local dependency setup, including the reviewed Cosmo-Pop affine adapter rather than the unrelated PyPI affine package.

| Choice | Defaults | Computation |
|---|---|---|
| `AffineSettings()` | 128 total walkers,128 burn-in,256 retained steps,one ensemble; checkpoint every16 steps;8 output selections/step | Numba/CPU (`numba` or `cpu`),Torch CPU (`torch`),MPS (`mps`),CUDA (`cuda`) |
| `StanSettings()` | One chain,1,000 warmup,2,000 retained draws | CPU NUTS through CmdStanPy |

The Stan defaults reproduce the original demo's `iter=3000,warmup=1000,chains=1`: `iter_sampling=2000`, not3,000. They do not imply convergence certification. Affine steps are transitions per walker and are not independent samples or Stan iterations. Explicit options such as `StanSettings(chains=4,warmup=1000,samples=2000)` or `AffineSettings(walkers=128,burnin=128,steps=512)` change the corresponding counts. Settings dictionaries are also accepted. Stan options like `adapt_delta`, `inits`, `parallel_chains`, `cache_dir` and `show_progress` can be passed directly; defaults for those remain CmdStan's existing behavior. Original initialization guesses are not imposed on new data.

Production Affine automatically uses float64 on CPU and float32 on GPUs; no unavailable-device fallback. CUDA is implemented but native testing remains pending. Backend availability is independent of qualification; retained float32 failures and convergence flags remain. CmdStan cannot select the MPS/CUDA Kalman engines.

## Inputs and results

Data can be `DataArray(time)` or `DataArray(time,panel)`. Uncertainty is its observation standard deviation. Affine adds a singleton panel axis for a one-dimensional input. Shared named xarray or aligned NumPy predictors are accepted; panel-specific predictors use a panel dimension. See COMPONENTS.md for custom regressors, units and interpolation behavior.

Affine returns the existing checkpointed `FitResult`. `result.posterior()` contains unconstrained hyperparameters and log probabilities. CmdStan returns its existing xarray posterior with constrained parameters and generated state draws. `summarize_trends` accepts either result and returns `trend` and `trend_sd` in original data units. Bands are pointwise posterior standard deviations, including conditional state uncertainty. It never subtracts or conditions on1998; the Affine internal smoother may additionally calculate differences, but those are not returned as absolute trends.

`summary = summarize_components(result, parameter_draws=64, chunk_draws=8)` supplies slope, seasonal, AR and coefficient means/SDs for tutorial plots. For Affine this is an explicit bounded CPU64 conditional-moment calculation on at most five panels, using stratified retained hyperparameter selections and the law of total variance. It generates no latent state trajectories. The64-selection default is illustrative, not an MC-error guarantee. For large grids use the faster backend-specific `summarize_trends`. CmdStan component summaries use its generated draws. Production estimates keep their diagnostic flags; no short-run certificate is added.

## Model and compatibility scope

Affine handles every implemented component layout. Exact `Model` equivalents of the six historical Stan models map to the corresponding existing source; other compositions fail with a clear unsupported-model error rather than silently changing the model. Generic Stan code generation and coupled spatial/lag Stan sampling are not part of this dispatch layer.

The public `fit` call without `sampler=` retains the prior Stan-only API/defaults for compatibility. New code should explicitly choose its sampler. The lower-level `fit_production`, `fit_grid` qualification workflow, `resume_production` and existing diagnostic/reconstruction APIs remain available. `fit_grid` is the affine qualification/production convenience API; use the unified `fit` for sampler selection.

Independent CmdStan panel dispatch is limited to five cells and retains separate panel divergences. Larger Affine production jobs require matching explicit selection records. Notebook full-grid execution is disabled by default, and spatial/lag selections remain separate. No MPI+Stan full-grid reference benchmark is launched by the tutorial.

## Notebook

[dlm_tutorial.ipynb](dlm_tutorial.ipynb) loads the original BASIC file and predictor files, plots inputs, composes vanilla AR1, runs one fresh Affine panel at128/128/256, and plots absolute trends/components/parameters. Its opt-in312-panel call uses the same defaults. With that switch disabled it plots a checksum-verified historical128/128/512 grid, clearly labeled. The grid bands are ordinary mean ±1SD throughout.
