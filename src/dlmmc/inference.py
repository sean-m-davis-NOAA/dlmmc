"""Public, dependence-aware eligibility checks for deferred inference.

Four independently randomized ensembles are replication; walkers are not.
Numerical eligibility and an explicit trace review are both required. This
gate is not a posterior-equivalence claim or a guarantee of convergence.
"""
import hashlib
import numpy as np
from scipy.special import expit
from .components import component_spec
from .transforms import parameter_names
from .diagnostics import ensemble_diagnostics, rank_split_rhat, split_rhat
from .validation_derived import PROFILE, finite_moment_summaries, raw_variance_diagnostics


def physical_parameters(u, model, nreg):
    """Vectorized physical parameters and stationary AR variance."""
    u = np.asarray(u)
    names = parameter_names(model, nreg)
    if u.shape[-1] != len(names) or not np.isfinite(u).all():
        raise ValueError('Finite parameters with the exact model dimension required')
    values = []
    for j, name in enumerate(names):
        if name.startswith('sigma'):
            value = np.exp(u[..., j])
        elif name == 'rhoAR1':
            value = expit(u[..., j]) * (.5 if component_spec(model)['ar2'] else 1.)
        else:
            value = expit(u[..., j]) * (1 - values[names.index('rhoAR1')])
        values.append(value)
    x = np.stack(values, -1)
    if 'rhoAR1' not in names:return names,x
    a = x[..., names.index('rhoAR1')]
    denominator = 1 - a*a
    if component_spec(model)['ar2']:
        b = x[..., names.index('rhoAR2')]
        denominator -= b*b + 2*b*a*a/(1-b)
    if np.any(denominator <= 0) or not np.isfinite(x).all():
        raise ValueError('Nonfinite or nonstationary parameters')
    arvar = x[..., names.index('sigma_AR')]**2 / denominator
    return names + ['stationary_AR_variance'], np.concatenate((x, arvar[..., None]), -1)


def window_diagnostics(values):
    """Check means, widths and tails of [run,time,walker,quantity].

    Practical profile: ESS2000, Rhat1.05 and MC-aware block stability.
    Fifty-IAT and point-width checks are retained as strict flags. The half
    and full windows are two diagnostic checkpoints, not independent data.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 4 or x.shape[0] < 4 or not np.isfinite(x).all():
        raise ValueError('Four finite independently randomized ensembles required')
    if x.shape[1] < 16:
        return dict(eligible=False, status='INSUFFICIENT_RETAINED_STEPS', retained_steps=x.shape[1])
    checkpoints = []
    for stop in (x.shape[1]//2, x.shape[1]):
        z = x[:, :stop]
        center = z.mean((0, 1, 2))
        functions = {'mean': z, 'squared_deviation': (z-center)**2}
        for q in (.05, .5, .95):
            functions[f'cdf_{q}'] = (z <= np.quantile(z, q, axis=(0, 1, 2))).astype(float)
        reports = {}
        for name, val in functions.items():
            d = ensemble_diagnostics(val)
            # Binary indicators can have constant folded ranks; their ensemble
            # averages still have meaningful ordinary split-Rhat diagnostics.
            rhat = split_rhat(val.mean(2)) if name.startswith('cdf_') else rank_split_rhat(val.mean(2))
            d['rank_rhat'] = rhat.tolist()
            d['mixing_eligible'] = bool(all(d['iat_length_reliable'])
                                       and np.all(np.isfinite(rhat) & (rhat < 1.05)))
            d['eligible'] = d['mixing_eligible'] and bool(np.all(np.asarray(d['mean_ess_estimate']) >= 2000))
            reports[name] = d
        # Single-walker independent HMC chains have no within-step ensemble
        # width. Their squared deviations supply the scale time series.
        width_series = z.std(2) if z.shape[2] > 1 else ((z-center)**2).mean(2)
        width_rhat = rank_split_rhat(width_series)
        checkpoints.append(dict(steps=stop, functions=reports,
                                width_rhat=width_rhat.tolist(), run_width=z.std((1, 2)).tolist(),
                                eligible=all(r['mixing_eligible'] for r in reports.values())
                                and bool(np.all(np.isfinite(width_rhat) & (width_rhat < 1.05)))))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.array(checkpoints[1]['run_width'])/np.array(checkpoints[0]['run_width'])
    width_stable = bool(np.all(np.isfinite(ratio) & (ratio > .95) & (ratio < 1.05)))
    strict_eligible = (width_stable and all(c['eligible'] for c in checkpoints)
                and all(r['eligible'] for r in checkpoints[-1]['functions'].values()))
    from .stability import mc_stability
    mc = mc_stability(x,checkpoints[-1]['functions'])
    rank_screen = all(np.all(np.asarray(c['width_rhat'])<1.05)
                      and all(np.all(np.asarray(v['rank_rhat'])<1.05) for v in c['functions'].values())
                      for c in checkpoints)
    eligible = rank_screen and mc['eligible']
    return dict(eligible=eligible, status='NUMERICALLY_ELIGIBLE' if eligible else 'INCONCLUSIVE',
                checkpoints=checkpoints, width_ratio=ratio.tolist(), width_stable=width_stable,
                strict_eligible=strict_eligible,mc_stability=mc,rank_screen=bool(rank_screen),
                minimum_ess=mc.get('minimum_ess',min(min(v['mean_ess_estimate']) for v in checkpoints[-1]['functions'].values())),
                maximum_iat=max(np.max(v['iat_by_run']) for v in checkpoints[-1]['functions'].values()))


def posterior_fingerprint(fit, posterior):
    h = hashlib.sha256(fit.config['provenance'].encode())
    for name in ('unconstrained', 'draw', 'cell'):
        a = np.ascontiguousarray(posterior[name].values)
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def diagnose_fit(fit, *, trace_review=None):
    """Diagnose a completed common window, preserving every cell's result.

    ``trace_review`` is a recorded human/agent observation of actual mean and
    width traces. Omit it for a numeric report awaiting visual review. Short
    notebook smoke runs remain ineligible and cannot reconstruct trajectories.
    """
    posterior = fit.posterior()
    p = fit.prepared
    names, physical = physical_parameters(posterior.unconstrained.values, p.model, p.regressors.shape[-1])
    raw_variance = physical[..., names.index("stationary_AR_variance")] if "stationary_AR_variance" in names else None
    names, physical = finite_moment_summaries(names, physical)
    cells = []
    for cell in range(p.cells):
        report = window_diagnostics(physical[..., cell, :])
        if raw_variance is not None:report['raw_stationary_AR_variance'] = raw_variance_diagnostics(raw_variance[..., cell])
        report['cell'] = cell
        report['coordinates'] = {k: str(v[cell]) for k, v in p.cell_coordinates.items()}
        cells.append(report)
    reviewed = isinstance(trace_review, str) and bool(trace_review.strip())
    eligible = reviewed and all(c['eligible'] for c in cells)
    return dict(schema=1, profile=PROFILE, eligible=eligible,
                status='PARAMETER_WINDOW_ELIGIBLE' if eligible else 'INCONCLUSIVE',
                scope='Parameter-window eligibility only; diagnose reconstructed scientific outputs separately',
                posterior_sha256=posterior_fingerprint(fit, posterior),
                trace_review=trace_review, quantities=names, cells=cells,
                walkers_are_independent_chains=False)
