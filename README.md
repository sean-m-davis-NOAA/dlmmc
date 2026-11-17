# DLMMC modernization branch

Dynamic linear models for atmospheric time series, with tensorized affine sampling, Kalman filtering and posterior trend inference. This branch is ready for colleague testing; it is a development candidate, not a fully qualified release.

Start with [installation](INSTALL.md), [testing](TESTING.md), and the [support and evidence summary](VALIDATION_STATUS.md). The sampler is **Cosmo-Pop affine at a pinned revision plus the included checkpoint adapter**, not the unrelated PyPI package of the same name.

## Current API

The affine API now accepts composable models: `Trend() + Harmonics((12, 6)) + Regression(dynamic=("ENSO",)) + AR(2)`. One layout compiler constructs matrices and backend specializations. Optional Level/Trend, harmonic periods, mixed static/dynamic coefficients and optional AR terms are supported. The six historical preset strings remain compatibility shorthands. See [the component API](COMPONENTS.md) and [its validation/timings](COMPOSITION_VALIDATION.md). All implemented component models are selectable on Numba, Torch CPU, MPS and CUDA. Backend access is independent of numerical qualification; retained float32 failures remain documented.

```python
from dlmmc import fit_production, diagnose_production, summarize_production_trends
from dlmmc.examples import load_panels

data, uncertainty, regressors, provenance = load_panels("examples/modernization")
fit = fit_production(
    data.isel(panel=[0]), regressors.isel(panel=[0]),
    uncertainty=uncertainty.isel(panel=[0]), model="vanilla_ar1",
    backend="numba", output_dir="results/first-fit",
)
monitoring = diagnose_production(fit)
summary = summarize_production_trends(
    fit, output_dir="results/first-trends", anchor_month="1998-01")
```

The production/testing default is one ensemble, **128 walkers, 128 burn-in steps and 256 retained steps**, with 2,048 value-independent draws for trend integration. It reports diagnostic flags rather than certifying convergence. `fit_grid` defaults to the separate four-ensemble qualification workflow. See [execution profiles](EXECUTION_PROFILES.md) for checkpointing, output uncertainty, scope limits and explicit configuration.

| Backend | Production precision | Local evidence |
|---|---|---|
| `numba` | CPU float64 | All six likelihood and smoother kernels tested; scoped posterior checks |
| `torch` | CPU float64 | All six likelihood and smoother kernels tested |
| `mps` | Native Apple GPU float32 | All six kernels tested; measured production sampling and smoothing |
| `cuda` | GPU float32 | Implemented; native hardware validation pending |

GPU backends select float32 automatically and never silently fall back. CPU float64 remains the oracle. Historical float32 failures and posterior/convergence flags remain documented. See [MPS implementation](STRUCTURED_MPS.md) and [validation status](VALIDATION_STATUS.md).

[SpatialPrior, LagPrior and fit_spatial](SPATIAL_API.md) expose Gaussian means, standard deviations and named-axis correlation lengths. The current coupled model correlates sigma_trend, with an optional prior-independent ENSO lag field. Independent panels have independent acceptance; spatial fields use joint acceptance. Arbitrary or multiple lagged regressors are not yet supported.

## Tutorials

Run these from the repository root in the installed environment:

- [Single-panel tutorial](dlm_tutorial.ipynb)
- [Tensorized parallel tutorial](dlm_tensor_grid_tutorial.ipynb), replacing the primary MPI example workflow
- [Spatial priors and ENSO lags](dlm_spatial_priors_tutorial.ipynb)
- [Validation notebook](dlm_validation_tests.ipynb)

Defaults run bounded smoke checks on at most five independent panels or four coupled panels. Scientific figures load checksum-verified historical results, explicitly separate from new smoke fits. Spatial and lag extensions retain separate pilot, runtime-estimate and panel-selection gates; notebook long-fit switches stay off.

## Compatibility and scientific policy

`dlmmc.fit` remains the optional CmdStan interface; `fit_grid_mpi` remains a legacy workflow. No-seasonal models require declared deseasonalized inputs; the supplied examples use fitted annual and semiannual harmonics. Priors, Jacobians, observation masking and conditional state uncertainty are preserved. [Scientific policy](MODERNIZATION.md) describes precision, practical consistency checks and continuation restrictions.

The original [README](docs/LEGACY_README.md) and [installation guide](docs/LEGACY_INSTALL.md) are retained as historical documentation. Their released-package, PyStan and MPI instructions do not install this modernization branch. Original data, attribution and licenses remain in the repository.

Choose `sampler="affine"` or `sampler="cmdstan"` through the unified `fit` API with `AffineSettings`/`StanSettings`; see [SAMPLERS.md](SAMPLERS.md). The [vanilla AR1 tutorial](dlm_tutorial.ipynb) follows the original data/regressor walkthrough, runs one default Affine panel and includes an explicitly enabled312-panel run with unanchored trend bands.

Affine now automatically selects available compute in order CUDA → MPS → Numba CPU, announcing the selection and override instructions at sampler startup. Omit `backend` or use `backend="auto"`; use `backend="numba"` to force CPU. Explicit backend/device choices are preserved. CmdStan remains CPU-only and announces that distinction.
