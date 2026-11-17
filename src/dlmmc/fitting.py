"""Sampler selection, with explicit sampler-specific settings and no device fallback."""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class AffineSettings:
    """Total walkers; burn-in and retained transitions per walker, not ESS."""
    walkers: int = 128
    burnin: int = 128
    steps: int = 256
    ensembles: int = 1
    chunk_size: int = 16
    output_draws_per_step: int = 8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == 'burnin' else 1):
                raise ValueError(f'{name} must be a valid integer count')
        if self.walkers % 2 or self.output_draws_per_step > self.walkers:
            raise ValueError('Even walker count and output selection no larger than walkers required')

    def profile(self):
        from .profiles import ExecutionProfile
        return ExecutionProfile('production', self.ensembles, self.walkers // 2,
                                self.burnin, self.steps, self.chunk_size,
                                output_draws_per_sweep=self.output_draws_per_step)


@dataclass(frozen=True)
class StanSettings:
    """Original demo: iter=3000, warmup=1000, chains=1 => 2000 retained/chain."""
    chains: int = 1
    warmup: int = 1000
    samples: int = 2000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == 'warmup' else 1):
                raise ValueError(f'{name} must be a valid integer count')


def _settings(value, cls):
    if value is None:
        return cls()
    if isinstance(value, dict):
        return cls(**value)
    if not isinstance(value, cls):
        raise TypeError(f'This sampler requires {cls.__name__} or a matching dictionary')
    return value


def _stan_model(model):
    from .api import _resolve_model, _MODEL_SPECS
    from .components import as_model, Model
    key, spec = _resolve_model(model)
    if 'source' in spec:
        return key
    composed = as_model(key)
    for name in _MODEL_SPECS:
        if composed == Model.from_preset(name):
            return name
    raise ValueError('CmdStan currently supports the six historical model compositions only; '
                     'this composition requires sampler="affine". No model is substituted.')


def fit(data, regressors=None, model='vanilla_ar1', *, sampler=None,
        backend=None, settings=None, **options):
    """Fit using explicit ``sampler='affine'`` or ``'cmdstan'`` (alias ``'stan'``).

    Affine returns its checkpointed FitResult and requires output_dir. Backend
    selects numba/cpu, torch CPU, mps or cuda; GPU precision is float32. CmdStan
    runs CPU NUTS and returns an xarray posterior. It accepts StanSettings plus
    CmdStan options such as adapt_delta/inits. Preset-equivalent Model objects
    map exactly to existing Stan sources; other compositions are not translated.

    Omitting sampler preserves the original Stan-only fit API/defaults. Explicit
    CmdStan selection uses the original demo defaults in StanSettings instead.
    Use summarize_trends for either result type. Independent CmdStan panel jobs
    are bounded to five cells; larger production selection is affine-only.
    """
    from . import api
    if sampler is None:
        if backend is not None or settings is not None:
            raise ValueError('Specify sampler when choosing backend or settings')
        return api.fit(data, regressors, model=model, **options)
    if sampler not in ('affine', 'cmdstan', 'stan'):
        raise ValueError('Choose sampler="affine" or "cmdstan" (alias "stan")')
    if not isinstance(data, xr.DataArray) or set(data.dims) not in ({'time'}, {'time', 'panel'}):
        raise ValueError('Use DataArray(time) or DataArray(time, panel)')
    if sampler == 'affine':
        config = _settings(settings, AffineSettings)
        invalid = set(options) & {'chains', 'warmup', 'samples', 'iter', 'iter_warmup',
                                 'iter_sampling', 'parallel_chains', 'adapt_delta', 'profile'}
        if invalid:
            raise ValueError(f'Use AffineSettings for affine counts; incompatible options: {sorted(invalid)}')
        from .production import fit_production
        uncertainty = options.pop('uncertainty', None)
        original = data
        if data.dims == ('time',):
            label = str(data.coords['panel'].item()) if 'panel' in data.coords else 'panel-0'
            data = data.expand_dims(panel=[label])
        data = data.transpose('time', 'panel')
        if uncertainty is None:
            uncertainty = xr.full_like(data, 1e-20)
        elif not isinstance(uncertainty, xr.DataArray):
            uncertainty = xr.DataArray(uncertainty, dims=original.dims, coords=original.coords)
        data, uncertainty = xr.align(data, uncertainty, join='exact')
        uncertainty = uncertainty.broadcast_like(data).transpose('time', 'panel')
        return fit_production(data, regressors, uncertainty=uncertainty, model=model,
                              backend=backend, profile=config.profile(), **options)
    if backend not in (None, 'auto', 'cpu', 'cmdstan'):
        raise ValueError('CmdStan NUTS runs on CPU; use backend="cpu". GPU kernels belong to affine.')
    config = _settings(settings, StanSettings)
    invalid = set(options) & (set(AffineSettings.__dataclass_fields__) | {'selection', 'profile'})
    if invalid:
        raise ValueError(f'Affine-only options cannot be used with CmdStan: {sorted(invalid)}')
    defaults = asdict(config)
    if settings is not None and set(options) & set(defaults):
        raise ValueError('Specify Stan sampling counts in settings or direct keywords, not both')
    defaults.update(options)
    StanSettings(**{k: defaults[k] for k in asdict(config)})
    name = _stan_model(model)
    if data.dims == ('time',):
        return api.fit(data, regressors, model=name, **defaults)
    if data.sizes['panel'] > 5:
        raise ValueError('CmdStan reference jobs are limited to five panels; use affine for selected production grids')
    if data.sizes['panel'] < 1 or len(set(map(str, data.panel.values))) != data.sizes['panel']:
        raise ValueError('Unique nonempty panel coordinates required')
    results = []
    uncertainty = defaults.pop('uncertainty', None)
    if uncertainty is not None and not isinstance(uncertainty, xr.DataArray):
        uncertainty = xr.DataArray(uncertainty, dims=data.dims, coords=data.coords)
    output = defaults.pop('output_dir', None)
    for i in range(data.sizes['panel']):
        local = dict(defaults)
        if output is not None:
            local['output_dir'] = str(Path(output) / f'panel-{i:04d}')
        error = uncertainty.isel(panel=i, drop=True) if uncertainty is not None and 'panel' in uncertainty.dims else uncertainty
        predictors = regressors.isel(panel=i, drop=True) if 'panel' in getattr(regressors, 'dims', ()) else regressors
        result = api.fit(data.isel(panel=i, drop=True), predictors, model=name, uncertainty=error, **local)
        result = result.expand_dims(panel=[data.panel.values[i]])
        result['divergent_transitions'] = ('panel', [result.attrs.get('divergent_transitions', 0)])
        results.append(result)
    result = xr.concat(results, dim='panel', combine_attrs='drop_conflicts')
    result.attrs.pop('divergent_transitions', None)
    return result


def summarize_trends(result, *, output_dir=None, **options):
    """Unanchored trend mean and pointwise posterior SD in original data units.

    Affine integrates conditional Gaussian uncertainty and hyperparameter draws;
    Stan summarizes generated trend draws. Neither subtraction at 1998 nor
    baseline conditioning is applied. Affine's provisional diagnostics persist.
    """
    if isinstance(result, xr.Dataset):
        if options:
            raise ValueError('Affine smoother options do not apply to Stan draws')
        mean = result.trend.mean(('chain', 'draw'))
        sd = result.trend.std(('chain', 'draw'), ddof=0)
        summary = xr.Dataset({'trend': mean, 'trend_sd': sd}, attrs=dict(result.attrs))
        summary.attrs.update(sampler='cmdstan', uncertainty='pointwise posterior SD; unanchored')
        if output_dir is not None:
            folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=False)
            summary.to_netcdf(folder/'trends.nc', engine='scipy')
        return summary
    if output_dir is None:
        raise ValueError('Affine summaries require output_dir for provenance and diagnostic records')
    from .production_trends import summarize_production_trends
    from .examples import load_trend_summary
    # The existing smoother also computes differences. Only its absolute means
    # and SDs are returned here; choosing an internal anchor cannot pinch them.
    options.setdefault('anchor_month', str(result.prepared.time[0].astype('datetime64[M]')))
    summarize_production_trends(result, output_dir=output_dir, **options)
    return load_trend_summary(output_dir)


def summarize_components(result, *, parameter_draws=64, chunk_draws=8):
    """Small-panel deterministic state summaries for tutorial component plots.

    Explicit CPU64 conditional moments, no sampled state trajectories or GPU
    fallback. All retained parameters are eligible for deterministic stratified
    selection. This bounded production output is not a convergence certificate;
    trend-only production summaries use the faster backend-specific smoother.
    """
    if isinstance(result, xr.Dataset):
        names = [n for n in ('trend', 'slope', 'seasonal', 'ar', 'beta') if n in result]
        return xr.Dataset({key: value for n in names for key,value in
                           ((n, result[n].mean(('chain','draw'))),
                            (n+'_sd', result[n].std(('chain','draw'),ddof=0)))},
                          attrs={'sampler':'cmdstan','uncertainty':'pointwise posterior SD'})
    import torch
    from dataclasses import replace
    from .components import compiled_layout
    from .production import production_parameter_cache
    from .tensor_target import model_components
    from .precision_prototype import filter_tensor
    from .tensor_smoothing import conditional_moments
    p = result.prepared
    if p.cells > 5:
        raise ValueError('Full component output is bounded to five panels; use summarize_trends for large grids')
    if type(parameter_draws) is not int or type(chunk_draws) is not int or min(parameter_draws,chunk_draws) < 1:
        raise ValueError('Positive integer parameter selection and chunk size required')
    p = replace(p, **{n:getattr(p,n).cpu().to(dtype=torch.bool if n=='observed' else torch.float64)
                       for n in ('y','variance','observed','regressors')})
    values = production_parameter_cache(result).reshape(-1, p.cells, len(p.parameter_names))
    count = min(parameter_draws, len(values))
    indices = np.floor((np.arange(count)+.5)*len(values)/count).astype(int)
    layout = compiled_layout(p.model, len(p.regressor_names))
    labels, weights = [], []
    def add(name, coordinates):
        w = np.zeros(layout.dimension); w[list(coordinates)] = 1
        labels.append(name); weights.append(w)
    if layout.level_kind:
        add('trend',[0])
        if layout.level_kind == 'trend':add('slope',[1])
    if layout.rotations:add('seasonal',[i for i,_ in layout.rotations])
    if layout.ar_order:add('ar',[layout.ar_start])
    for j,n in enumerate(p.regressor_names):add('beta:'+n,[layout.regressor_start+j])
    weights = torch.tensor(np.asarray(weights),dtype=torch.float64)
    mean_sum = second_sum = None
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            for start in range(0,count,chunk_draws):
                u = torch.tensor(np.asarray(values[indices[start:start+chunk_draws]]),dtype=torch.float64)
                F,G,Q,C,_,valid = model_components(u,*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
                if not bool(valid.all()):raise ValueError('Retained parameters outside support')
                m,c,_ = conditional_moments(filter_tensor(p.y,p.variance,p.observed,F,G,Q,C),G,Q)
                means = torch.einsum('...ti,ki->...tk',m,weights).numpy()*p.scale[None,:,None,None]
                var = torch.einsum('ki,...tij,kj->...tk',weights,c,weights).numpy()*p.scale[None,:,None,None]**2
                if 'trend' in labels:means[...,labels.index('trend')] += p.offset[None,:,None]
                if not np.isfinite(means).all() or not np.isfinite(var).all() or np.any(var<0):
                    raise ValueError('Invalid conditional component moments; no variance floor')
                if mean_sum is None:
                    mean_sum=np.zeros(means.shape[1:]); second_sum=np.zeros_like(mean_sum)
                    # A fixed center avoids cancellation when integrating narrow distributions.
                    center=means[0].copy()
                mean_sum += (means-center).sum(0)
                second_sum += (var+(means-center)**2).sum(0)
    finally:
        torch.set_num_threads(old_threads)
    delta=mean_sum/count; variance=second_sum/count-delta**2
    if np.any(variance<0):raise ValueError('Invalid mixture covariance; no floor')
    means=center+delta;sd=np.sqrt(variance)
    variables={}
    for j,n in enumerate(labels):
        if n.startswith('beta:'):continue
        variables[n]=( ('panel','time'),means[:,:,j])
        variables[n+'_sd']=( ('panel','time'),sd[:,:,j])
    beta=[j for j,n in enumerate(labels) if n.startswith('beta:')]
    if beta:
        variables['beta']=(('panel','time','regressor'),means[:,:,beta])
        variables['beta_sd']=(('panel','time','regressor'),sd[:,:,beta])
    return xr.Dataset(variables,coords={'panel':p.cell_coordinates['panel'],'time':p.time,
                                       'regressor':p.regressor_names},
                      attrs={'sampler':'affine','compute':'explicit CPU64 conditional moments',
                             'selected_parameter_draws':count,'qualification_status':'PRODUCTION_ESTIMATES_WITH_FLAGS',
                             'uncertainty':'conditional variance plus hyperparameter mixture variance; pointwise SD'})
