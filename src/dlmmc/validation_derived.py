"""User-approved finite-moment scale for the stationary AR diagnostic.

Raw stationary variance remains a reported diagnostic. Its posterior SD is
infinite under the unchanged uniform persistence prior and stationary initial
state; an SD-based consistency gate therefore uses log variance explicitly.
"""
import numpy as np

PROFILE = 'practical_ess2000_v3_log_ar_variance'


def finite_moment_summaries(names, values):
    """Copy and transform only stationary_AR_variance; never alter draws."""
    names = list(names)
    values = np.array(values, copy=True)
    if values.shape[-1] != len(names):
        raise ValueError('Summary names and final axis must agree')
    if 'stationary_AR_variance' in names:
        j = names.index('stationary_AR_variance')
        if np.any(~np.isfinite(values[..., j]) | (values[..., j] <= 0)):
            raise ValueError('Finite positive stationary variance required; no floor')
        values[..., j] = np.log(values[..., j])
        names[j] = 'log_stationary_AR_variance'
    return names, values


def raw_variance_diagnostics(values):
    """Finite-window descriptive statistics, explicitly not population SD."""
    x = np.asarray(values)
    return dict(status='DESCRIPTIVE_RAW_VALUES_NOT_SD_QUALIFICATION',
        population_sd='INFINITE under the unchanged stationary AR model',
        empirical_mean=float(x.mean()), empirical_sd=float(x.std(ddof=1)),
        quantiles=np.quantile(x, [.05, .5, .95]).tolist(), maximum=float(x.max()),
        qualification_scale='log_stationary_AR_variance')
