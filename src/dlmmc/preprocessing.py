"""Explicit, deterministic seasonal preprocessing, outside the likelihood."""
import hashlib
import json

import numpy as np
import xarray as xr


def deseasonalize_harmonics(data):
    """Remove fitted annual and semiannual harmonics from monthly series.

    Fit each non-time column separately by unweighted least squares, jointly
    with an intercept and linear trend. Subtract only the four harmonic terms,
    centered over the observed timestamps, retaining the overall observed mean.
    The trend is a nuisance in estimating seasonality and is NOT subtracted.
    NaNs are excluded from the fit and preserved in the result. No measurement
    uncertainty is changed or inferred: subsequent inference is conditional on
    this fitted deterministic cycle. Apply separately to predictors if desired.

    Returns a Dataset with corrected data, removed cycle and fit coefficients.
    Requires datetime monthly timestamps and a full-rank six-column design.
    The returned data are float64 preprocessing products; callers explicitly
    select float32/float64 for subsequent model evaluation.
    """
    if not isinstance(data, xr.DataArray) or 'time' not in data.dims:
        raise ValueError('Expected a DataArray with a time dimension')
    if data.attrs.get('seasonality') == 'deseasonalized':
        raise ValueError('Data already marked deseasonalized; refuse double removal')
    time = np.asarray(data.time.values)
    if not np.issubdtype(time.dtype, np.datetime64) or np.isnat(time).any():
        raise ValueError('Finite datetime monthly coordinates required')
    month = time.astype('datetime64[M]').astype(np.int64)
    if len(month) < 6 or np.any(np.diff(month) <= 0):
        raise ValueError('At least six unique increasing monthly timestamps required')
    t = (month - month[0]).astype(float)
    phase = 2 * np.pi * (t % 12) / 12
    design = np.column_stack([np.ones(len(t)), (t - t.mean()) / 12,
                              np.sin(phase), np.cos(phase),
                              np.sin(2 * phase), np.cos(2 * phase)])
    ordered = data.transpose('time', ...)
    raw = np.asarray(ordered.values, dtype=np.float64)
    if np.isinf(raw).any():
        raise ValueError('Infinite inputs are not supported; use explicit NaNs for missing values')
    flat = raw.reshape(len(t), -1)
    corrected = np.full_like(flat, np.nan)
    seasonal = np.empty_like(flat)
    coefficients = np.empty((6, flat.shape[1]))
    counts = np.empty(flat.shape[1], dtype=np.int64)
    for cell in range(flat.shape[1]):
        valid = np.isfinite(flat[:, cell])
        counts[cell] = valid.sum()
        if counts[cell] < 6:
            raise ValueError('Insufficient observed months for harmonic regression')
        coef, _, rank, _ = np.linalg.lstsq(design[valid], flat[valid, cell], rcond=None)
        if rank != 6:
            raise ValueError('Rank-deficient harmonic regression; no regularization or floor')
        cycle = design[:, 2:] @ coef[2:]
        cycle -= cycle[valid].mean()
        seasonal[:, cell] = cycle
        corrected[valid, cell] = flat[valid, cell] - cycle[valid]
        coefficients[:, cell] = coef
    policy = dict(method='annual_semiannual_ols_v1', period_months=12,
                  harmonics=[1, 2], nuisance=['intercept', 'linear_trend'],
                  subtract='harmonics_only_centered_over_observed_months',
                  weights='uniform', missing='finite_observations_only',
                  window_start=str(time[0]), window_end=str(time[-1]),
                  phase_origin=str(time[0]), uncertainty='conditional_on_fixed_fitted_cycle')
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode())
    for array in (time, raw, coefficients):
        digest.update(np.ascontiguousarray(array).tobytes())
    attrs = dict(data.attrs, seasonality='deseasonalized',
                 seasonal_preprocessing=json.dumps(policy, sort_keys=True),
                 seasonal_preprocessing_sha256=digest.hexdigest())
    clean = ordered.copy(data=corrected.reshape(raw.shape)).transpose(*data.dims)
    clean.attrs = attrs
    cycle = ordered.copy(data=seasonal.reshape(raw.shape)).transpose(*data.dims)
    cycle.attrs = dict(units=data.attrs.get('units', ''), meaning='Removed fixed seasonal component')
    dims = ordered.dims[1:]
    coords = {key: value for key, value in ordered.coords.items() if 'time' not in value.dims}
    coef = xr.DataArray(coefficients.reshape((6,) + raw.shape[1:]), dims=('term',) + dims,
        coords=dict(coords, term=['intercept', 'trend_per_year', 'annual_sin', 'annual_cos',
                                 'semiannual_sin', 'semiannual_cos']))
    count = xr.DataArray(counts.reshape(raw.shape[1:]), dims=dims, coords=coords)
    return xr.Dataset(dict(deseasonalized=clean, seasonal_component=cycle,
                           coefficients=coef, observed_count=count), attrs=attrs)
