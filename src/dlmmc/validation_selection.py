"""Value-independent, time-stratified selection for deferred reconstruction."""
import numpy as np


def stratified_indices(steps, walkers, count, seed):
    if min(steps, walkers, count) < 1 or count > steps:
        raise ValueError('Require 1 <= selected count <= retained steps')
    rng = np.random.default_rng(seed)
    edges = np.arange(count + 1, dtype=np.int64) * steps // count
    time = rng.integers(edges[:-1], edges[1:])
    walker = rng.integers(walkers, size=count)
    return np.column_stack([time, walker])


def selected_diagnostics(values, target_ess=2000, rhat_limit=1.05):
    """Ordered selected draws [independent run, draw, quantity].

    Diagnose means, squared deviations and quantile indicators separately.
    Selection does not make correlated draws independent. This supplements,
    rather than replaces, the original full parameter-window certificate.
    """
    from .diagnostics import ensemble_diagnostics, rank_split_rhat, split_rhat
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or x.shape[0] < 4 or not np.isfinite(x).all():
        raise ValueError('Four finite ordered independent runs required')
    center = x.mean((0, 1))
    functions = {'mean': x, 'squared_deviation': (x-center)**2}
    for q in (.05, .5, .95):
        functions[f'cdf_{q}'] = (x <= np.quantile(x, q, axis=(0, 1))).astype(float)
    reports = {}
    for name, val in functions.items():
        d = ensemble_diagnostics(val[:, :, None, :])
        # For a balanced binary indicator folding around median .5 is exactly
        # constant and contains no scale information. Ordinary split Rhat is
        # rank-equivalent for binary values; zero-event original indicators
        # still produce an invalid ESS/Rhat and are never certified.
        d['rank_rhat'] = (split_rhat(val) if name.startswith('cdf_')
                          else rank_split_rhat(val)).tolist()
        d['eligible'] = bool(np.all(np.asarray(d['mean_ess_estimate']) >= target_ess)
                             and all(d['iat_length_reliable'])
                             and np.all(np.asarray(d['rank_rhat']) < rhat_limit))
        reports[name] = d
    return {'eligible': all(d['eligible'] for d in reports.values()),
            'functions': reports,
            'minimum_ess': min(min(d['mean_ess_estimate']) for d in reports.values()),
            'maximum_iat': max(np.max(d['iat_by_run']) for d in reports.values()),
            'maximum_rhat': max(max(d['rank_rhat']) for d in reports.values())}
