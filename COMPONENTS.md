# Composing models

The affine API accepts `Model` objects instead of requiring a preset name. A single layout compiler determines the parameter order, state slices, transition sparsity, observation vectors, innovation covariance and initial covariance. Numba, Torch and the generated Metal kernels consume that layout. No process-local model registration is needed.

```python
from dlmmc import Model, Trend, Level, Harmonics, Regression, AR
from dlmmc import fit_production, summarize_production_trends

model = (
    Trend()
    + Harmonics(periods=(12, 6))
    + Regression(dynamic=("ENSO",))  # other provided predictors remain static
    + AR(order=2)
)
fit = fit_production(data, regressors, uncertainty=uncertainty,
                     model=model, backend="mps", output_dir="results/composed-fit")
summary = summarize_production_trends(fit, output_dir="results/composed-trends")
```

The independent default remains 128 walkers /128 burn-in /256 retained with one ensemble. These are estimates with diagnostic flags. `fit_grid` keeps the qualification workflow, and `fit_spatial(..., model=model, ...)` keeps joint acceptance and its separate spatial/lag scope gates.

## Building blocks and model semantics

| Component | State and prior |
|---|---|
| `Trend(level_variance=1e-10)` | Level and slope; fixed level innovation variance and inferred positive half-normal `sigma_trend` |
| `Level()` | One random-walk level; inferred positive half-normal `sigma_level` |
| `Harmonics(periods=(12, 6), name="seas")` | A rotation pair per period, periods in months; shared positive half-normal `sigma_seas` within this block |
| `Regression(dynamic=False)` | All supplied predictors, with static coefficients and the legacy fixed innovation variance |
| `Regression(dynamic=True)` | All supplied coefficients evolve, with independent positive half-normal `sigma_reg[i]` scales |
| `Regression(dynamic=("ENSO", "SOLAR"))` | Only named coefficients evolve; names are checked and their input ordering is checkpointed |
| `AR(1)` / `AR(2)` | Existing positive AR coefficients, stationary initial covariance and normalized conditional-uniform priors/Jacobians |

Components can be omitted. Choose at most one Trend or Level, one Regression and one AR. Separate harmonic blocks with distinct names/periods allow different innovation scales, for example `Harmonics((12,), name="annual") + Harmonics((6,), name="semi")`. Repeated periods or names are rejected. Select meaningful periods for the sampling interval: the compiler does not infer scientific identifiability or choose a seasonal model for you.

State order is canonical: trend/level, harmonic blocks sorted by name, coefficients in predictor order, AR. Inferred scale defaults remain 1e-4 for trend/level/coefficient innovations, .01 for harmonic innovations and .5 for AR innovations, in normalized data units. Override using the existing `priors` mapping (`sigma_annual_prior_width`, `sigma_reg_prior_width`, etc.). `initial_state_prior_width=3` is the default Gaussian standard deviation of non-AR initial states; the backend receives its square, `S=9`. This intentionally changes the former default variance of10; all innovation-prior defaults are unchanged. These are statistical assumptions, not numerical stabilization parameters.

Static coefficient innovation variance defaults to 1e-20 with AR2 and 1e-10 otherwise, preserving the historical models. `Regression(static_variance=...)`, `Trend(level_variance=...)` and `AR(auxiliary_variance=...)` make changes explicit; choosing different values changes the model. No floor, added jitter or precision fallback is introduced.

Omitting harmonics still requires declared deseasonalized data (or declared synthetic data without seasonality). `deseasonalize_harmonics(data).deseasonalized` supplies the existing fitted-harmonic preprocessing. Omitting Regression requires no predictor columns. Models with no sampled parameters can construct matrices directly but do not need an affine sampler.

## Compatibility and inspection

`Model.from_preset("vanilla_ar1")` gives `Model(Trend(), Harmonics(), Regression(), AR(1))`. All six existing strings remain accepted. Their prior families, fixed variances, parameter ordering and initial-state timing are preserved (the user-selected initial variance default is now9); they compile through the same component layout. The optional CmdStan API continues to support its six original Stan models only; arbitrary Stan code generation is not part of this change.

```python
from dlmmc import prepare_data, build_state_space
prepared = prepare_data(data, regressors, uncertainty=uncertainty, model=model)
print(prepared.parameter_names)
# model.bind(regressor_names) exposes the immutable layout before fitting.
layout = model.bind(prepared.regressor_names)
matrices = build_state_space(
    model, unconstrained_parameters, prepared.regressors,
    regressor_names=prepared.regressor_names, priors=prepared.priors,
)
```

`build_state_space` returns `observation`, `transition`, `innovation`, `initial`, transformed `log_prior`, support validity, parameter names and component slices. Parameter and regressor inputs must be Torch tensors of matching dtype/device; leading draw/panel axes broadcast. The function performs no sampling or implicit casts. GPU fitting selects float32 automatically as before; CPU float64 remains the oracle.

`model.to_dict()` / `Model.from_dict(...)` provide a versioned JSON-safe specification. Prepared checkpoints store a canonical self-contained representation plus predictor-name binding. Reload works in a fresh interpreter without registering a model. Exact continuation still requires matching sources/runtime/input identity; old saved fits remain readable, but a source update is not an automatic exact-resume migration.

## Output and qualification limits

Trend summaries require a Trend or Level component. Other compositions can evaluate likelihoods and reconstruct their full states through the existing diagnosed reconstruction API; no harmonic or AR state is silently mislabeled as a trend. Component slices identify state coordinates. The deterministic centering offset belongs to the baseline/signal, not to harmonic or regression states.

All implemented component models, including no-AR compositions, are selectable on `numba` (or `cpu`), `mps` and `cuda`, per the user’s explicit backend-access decision. There is no model-specific qualification gate. Retained no-AR ozone stress fixtures still fail the unchanged float32 density gate; making the backend available does not change that result. A new level-model cancellation fixture prompted a stable covariance calculation for anchor-relative changes; no tolerance or model was changed.

CUDA remains native-untested. Passing representative compositions does not certify every possible model, input or posterior. Scientific consistency, short-run convergence and numerical qualification remain distinct. No new spatial/lag panel selection or full-grid benchmark is authorized by creating a model object.

## Original notebook model and your own predictors

The original notebook’s local linear trend, annual/semiannual seasonality, five static coefficients and AR1 residual are `Trend() + Harmonics((12, 6)) + Regression() + AR(1)`. The old `model="vanilla_ar1"` spelling is equivalent. Priors default to the original values; the production sampling defaults are128 walkers/128 burn-in/256 retained, not the old Stan iteration counts.

Given the notebook arrays `d`, `s`, `enso`, `solar`, `qbo30`, `qbo50`, `saod`, and a matching monthly datetime coordinate `time`:

```python
import xarray as xr
from dlmmc import Trend, Harmonics, Regression, AR
from dlmmc import fit_production, summarize_production_trends

y = xr.DataArray(d[:, None], dims=("time", "panel"),
                 coords={"time": time, "panel": ["example"]})
errors = xr.DataArray(s[:, None], dims=y.dims, coords=y.coords)
x = xr.Dataset({
    "ENSO": ("time", enso), "SOLAR": ("time", solar),
    "QBO30": ("time", qbo30), "QBO50": ("time", qbo50),
    "SAOD": ("time", saod),
}, coords={"time": time})
model = Trend() + Harmonics((12, 6)) + Regression() + AR(1)
fit = fit_production(y, x, uncertainty=errors, model=model,
                     backend="mps", output_dir="results/example")
summary = summarize_production_trends(
    fit, output_dir="results/example-trends", anchor_month="1998-01")
```

Choose an anchor within your record and fresh output directories. `s` is observation **standard deviation**, not variance. The API handles model dimensions, data normalization, parameter transforms and initialization. Production data and uncertainty have dimensions `(time, panel)` even for one panel. All supplied predictor columns are used; no fixed regressor names/count are required.

For your own regressors, pass an `xarray.Dataset` with one named `time` series per predictor, an `xarray.DataArray(time, regressor)`, or an already aligned NumPy matrix `(Ntime, Nregressors)`. NumPy columns receive names `regressor_1`, etc. A pandas time-indexed DataFrame can be converted with `frame.rename_axis("time").to_xarray()`. Named xarray inputs are linearly interpolated onto observation times; they must cover the window and be finite after interpolation, with no automatic extrapolation or imputation. Use the original pre-interpolated arrays when reproducing the old notebook’s spline inputs. Predictors are not silently standardized; their scaling affects coefficient priors.

Shared predictors broadcast across panels. Panel-specific predictors use `DataArray(time, panel, regressor)` with exactly matching panel coordinates. `Regression(dynamic=True)` evolves all coefficients; `Regression(dynamic=("ENSO", "MY_PROXY"))` evolves only those names. Omit Regression and pass `None` for a model with no predictors. For lagged predictors, the separately documented spatial API currently supports ENSO only; arbitrary predictor ingestion does not imply arbitrary lag-field support.
