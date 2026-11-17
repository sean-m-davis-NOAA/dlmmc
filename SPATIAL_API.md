# Spatial priors and selected regressor lags

Select the four pilot panels before calling the fitting API. Use named coordinates and underlying Gaussian mean/standard deviation, before truncation:

```python
from dlmmc import fit_spatial, SpatialPrior, LagPrior, MonthlyENSO

sigma = SpatialPrior(
    mean=0.0,
    std=1e-4,
    correlation_lengths={"log_pressure": 0.163, "latitude": 8.49},
    units="normalized",
)
fit = fit_spatial(
    ozone, regressors, uncertainty=uncertainty,
    sigma_trend=sigma, backend="mps", output_dir="new-spatial-fit",
    walkers=128, burnin=128, steps=256, ensembles=4,
)
```

Lengths are scientific inputs, not automatically inferred from grid spacing. The numbers above illustrate the project's existing four-panel test, not a universal ozone prior. The kernel is exp(-0.5 sum((axis distance / length)^2)). Latitude is in degrees north; log-pressure is the natural logarithm of a pressure ratio, with dimensionless differences. Altitude instead requires `correlation_lengths={"altitude": ..., "latitude": ...}` and `coordinate_units={"altitude": "km"}` matching the supplied coordinates.

`mean` and `std` may be scalars or arrays in prepared panel order. They describe the Gaussian before its strict positive truncation; post-truncation moments differ. `units="normalized"` uses the internal normalized ozone trend innovation scale. `units="data/month"` specifies the original data-unit trend-slope innovation, explicitly converting by each panel's normalization scale. The Gaussian is directly on sigma_trend, not its logarithm. Other parameter priors remain independent between panels.

The lag mapping explicitly selects which regressor changes:

```python
predictor = MonthlyENSO.from_dates(
    extended_enso.time.values, extended_enso.values, ozone.time.values,
    upper_lag=12,
)
lag_fit = fit_spatial(
    ozone, regressors, uncertainty=uncertainty,
    sigma_trend=sigma,
    lags={"ENSO": LagPrior(mean=0, std=2, upper=12)},
    predictors={"ENSO": predictor},
    backend="mps", output_dir="new-lag-fit",
    walkers=128, burnin=128, steps=256, ensembles=4,
)
```

The current researched/validated model supports ENSO only: `lags={"ENSO": ...}`; an empty mapping means no inferred lags. Other or multiple names fail clearly rather than silently applying ENSO assumptions. Adding arbitrary simultaneous lag fields is a later model/research extension. Lag mean/std are in months and support is strictly 0 < lag < upper; the convention is ENSO(t-lag). The lag field inherits sigma_trend's spatial lengths (an explicit identical mapping is also accepted), with a prior independent of the sigma field. The complete predictor record must cover the fixed observation window at every allowed lag; no extrapolation or proposal-dependent data removal occurs. This contract does not shorten the observation window automatically.

`sample_spatial_prior(prepared, ...)` offers the same prior/lag interface for already prepared, coordinate-checked inputs. Preparation is capped at five panels by default. Larger selected studies require a separate prepared input record and the existing matching stage-specific selection, after that stage's pilot/runtime estimate. Four independent ensembles support between-run checks; a completed 256-step job is not a convergence certificate. The sampler saves complete resumable hyperparameter/RNG states; no latent trajectory cube is saved during sampling. `load_spatial`, `resume_spatial`, `load_enso_lag` and `resume_enso_lag` retain their existing source/runtime and actual-wall-cap protections.

Independent fits batch local log densities with independent acceptance per panel. Coupled fits batch those same local likelihoods but accept a spatial field proposal jointly; nuisance parameters factorize only conditional on the field. A whole-field stretch move requires a complementary half-ensemble spanning the field dimension. Consequently 128 total walkers do not support a single 312-dimensional field move. Conditional subfield updates are needed to keep that walker count at full-grid scale; there is no silent independent-panel approximation.

## Composed state models

`fit_spatial(..., model=Model(...))` uses the same component compiler as independent fitting. A Trend component is required for the sigma_trend field. Regressor labels must include ENSO for the existing lag extension. Component composition does not change the prior-independence of the two fields, the shared length contract, joint acceptance or either panel-selection gate.
