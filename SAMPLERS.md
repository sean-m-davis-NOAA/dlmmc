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
| `AffineSettings()` | 128 total walkers,128 burn-in,256 retained steps,one ensemble; checkpoint every16 steps;8 output selections/step | Numba/CPU (`numba` or `cpu`), MPS (`mps`), CUDA (`cuda`) |
| `StanSettings()` | One chain,1,000 warmup,2,000 retained draws | CPU NUTS through CmdStanPy |

The Stan defaults reproduce the original demo's `iter=3000,warmup=1000,chains=1`: `iter_sampling=2000`, not3,000. They do not imply convergence certification. Affine steps are transitions per walker and are not independent samples or Stan iterations. Explicit options such as `StanSettings(chains=4,warmup=1000,samples=2000)` or `AffineSettings(walkers=128,burnin=128,steps=512)` change the corresponding counts. Settings dictionaries are also accepted. Stan options like `adapt_delta`, `inits`, `parallel_chains`, `cache_dir` and `show_progress` can be passed directly; defaults for those remain CmdStan's existing behavior. Original initialization guesses are not imposed on new data.

Omit `backend` or set `backend="auto"` to discover available compute when sampling starts. Affine selects **CUDA → MPS → Numba CPU** and prints available backends, its selected device/dtype and how to override. Use `backend="numba"` to force CPU or `backend="mps"`/`"cuda"` to choose a GPU explicitly. An explicit `device="cpu"` selects Numba. With automatic backend, `dtype="float64"` selects CUDA if available, otherwise Numba CPU. Conflicting explicit backend/device/dtype choices fail instead of silently changing them. `available_compute()` returns detected availability; `resolve_compute(announce=True)` previews the choice without fitting. Pure Model/Settings objects stay independent of hardware; the announcement happens during sampler setup, not during each likelihood evaluation.

The same auto preference applies to `fit_production`, qualification `fit_grid`, and spatial/lag sampler startup. Explicit qualification dtype/device settings remain honored. Resuming a checkpoint retains its recorded backend; hardware is not silently switched.

Production defaults to float64 on CPU and float32 on GPUs. Explicit `dtype="float32"` or `"float64"` works on Numba CPU and CUDA. MPS only supports float32 and rejects float64; no unavailable-device fallback. `backend="torch"` is removed from public fitting; the Torch CPU numerical oracle remains internal. CUDA is implemented but native testing remains pending. Backend availability is independent of qualification; retained float32 failures and convergence flags remain. CmdStan cannot select the MPS/CUDA Kalman engines. It prints that CPU is selected and shows how to choose Affine for GPU execution, including for legacy calls without an explicit sampler.

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


## Precision, priors and normalization

```python
fit(data, regressors, uncertainty=uncertainty, model=model,
    sampler="affine", backend="cpu", dtype="float32",  # or numba/float64, cuda/float32|64, mps/float32
    settings=AffineSettings(), output_dir="results/cpu32",
    priors={"initial_state_prior_width": 3., "sigma_trend_prior_width": 1e-4,
            "sigma_seas_prior_width": .01, "sigma_AR_prior_width": .5})
```

Pass response data and observation **standard deviations** in original units. Each panel is centered by the mean of its finite response observations and divided by their range (maximum minus minimum). The observation SD is divided by that range before squaring. A nonfinite response or uncertainty masks that observation; finite response values still define normalization even if their uncertainty is missing. Constant or all-missing response panels are rejected. Returned trend/component means and SDs are restored to response units, and coefficient units are response units per supplied predictor unit. Hyperparameter samples and their prior scales remain in normalized-model units.

Predictors are linearly aligned to observation times but are **not automatically normalized**. Supply finite predictors in deliberate units, optionally scaling them yourself. Scaling a predictor changes the meaning of its coefficient prior; the library does not silently compensate or whiten predictors.

For the vanilla AR1 model, innovation scales have independent half-normal priors: `sigma_trend` scale1e-4, `sigma_seas` scale0.01 and `sigma_AR` scale0.5. The mapping keys above set these **standard deviations**, not variances. `rhoAR1` has a fixed Uniform(0,1) prior. `initial_state_prior_width=3` sets initial-state Gaussian **standard deviation** (internal variance `S=9`), including the zero-mean Gaussian coefficient state; AR initial covariance instead follows its stationary variance. Static coefficients preserve the historical tiny fixed innovation variance. Dynamic coefficients additionally have half-normal innovation priors (`sigma_reg_prior_width=1e-4`, scalar or per-predictor list). Composed named harmonic blocks use their corresponding named `sigma_<block>_prior_width` keys. Unknown/unused prior keys are rejected; arbitrary distribution families are not accepted through this mapping.

For AR2, the existing priors are rho1~Uniform(0,0.5), rho2|rho1~Uniform(0,1−rho1), including conditional normalization. Spatial `SpatialPrior(mean,std,correlation_lengths)` and `LagPrior(...)` are separate field contracts; they do not change independent priors unless explicitly selected. Spatial/lag scope gates remain unchanged.

`summary = summarize_components(fit, parameter_draws=64)` now also returns `regression`, `fitted`, and corresponding SDs. The fitted latent signal uses full covariance, including cross-component terms, and excludes a new measurement's observation noise. `beta_grid`/`beta_density` represent a Gaussian mixture posterior at `coefficient_time_index` (default midpoint); the chosen date is recorded in attributes. These additional outputs apply to Affine fits; CmdStan summaries retain its available generated component draws. Computation is explicitly CPU64, bounded to five panels and selected hyperparameters, with no latent-trajectory sampling. The tutorial plots component bands, coefficient bands and densities, and complete fitted signal alongside observations. The faster `summarize_trends` uses the selected backend/dtype.


At fit startup, DLMMC prints the selected compute device/precision, model components, predictor names and static/dynamic selection, resolved prior widths, AR priors, and normalization convention. Spatial summaries show replacing Gaussian fields rather than incorrectly printing independent sigma_trend priors.

The public mapping uses `initial_state_prior_width=3` and `sigma_*_prior_width`. All widths are Gaussian **standard deviations**; `initial_state_prior_width` is squared to supply internal initial covariance `S`. The default initial variance intentionally changes from10 to9, by user choice; innovation-prior defaults remain unchanged. Historical saved results keep their original priors. To reproduce the former initial prior use `initial_state_prior_width=np.sqrt(10)` or the exact compatibility alias `S=10`.

Old `sigma_*_prior` keys remain standard-deviation aliases. Old `S` remains a **variance** alias, never reinterpreted as a width. Supplying old and new names together is rejected, as is providing the same prior both in the mapping and as a direct keyword. Width aliases also work with named harmonic components, dynamic coefficient vectors, `prepare_data`, `build_state_space`, and CmdStan. Internal prepared/checkpoint dictionaries retain the original backend keys for compatibility.
