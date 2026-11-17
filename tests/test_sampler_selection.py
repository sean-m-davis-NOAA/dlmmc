from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dlmmc
from dlmmc import AffineSettings, StanSettings, Trend, Harmonics, Regression, AR


def inputs(n=24):
    t = pd.date_range('1997-01-01', periods=n, freq='MS')
    y = xr.DataArray(2 + .03*np.arange(n) + .2*np.sin(np.arange(n)), dims='time', coords={'time':t})
    return y, xr.full_like(y, .1), xr.Dataset({'proxy':('time', np.cos(np.arange(n)))}, coords={'time':t})


def test_settings_and_original_demo_defaults():
    assert asdict(StanSettings()) == {'chains':1, 'warmup':1000, 'samples':2000}
    assert StanSettings().warmup + StanSettings().samples == 3000
    from dlmmc import PRODUCTION_PROFILE
    assert AffineSettings().profile() == PRODUCTION_PROFILE
    for values in ({'walkers':127}, {'steps':0}, {'burnin':-1}, {'walkers':True}):
        with pytest.raises(ValueError): AffineSettings(**values)
    with pytest.raises(ValueError): StanSettings(chains=0)


@pytest.mark.parametrize('sampler', ['stan', 'cmdstan'])
def test_stan_dispatch_and_exact_composed_mapping(monkeypatch, sampler):
    from dlmmc import api
    calls = []
    monkeypatch.setattr(api, 'fit', lambda *a, **k: calls.append((a,k)) or 'posterior')
    y,s,x = inputs()
    assert dlmmc.fit(y,x,uncertainty=s,model=Trend()+Harmonics()+Regression()+AR(1),sampler=sampler,backend='cpu') == 'posterior'
    kwargs = calls[0][1]
    assert kwargs['model'] == 'vanilla_ar1'
    assert {k:kwargs[k] for k in asdict(StanSettings())} == asdict(StanSettings())
    assert kwargs['uncertainty'] is s
    dlmmc.fit(y,x,sampler=sampler,settings={'chains':2,'warmup':3,'samples':4},adapt_delta=.95)
    assert calls[-1][1]['adapt_delta'] == .95 and calls[-1][1]['samples'] == 4
    with pytest.raises(ValueError,match='CPU'):dlmmc.fit(y,x,sampler=sampler,backend='mps')
    with pytest.raises(ValueError,match='float64 only'):dlmmc.fit(y,x,sampler=sampler,dtype='float32')
    dlmmc.fit(y,x,sampler=sampler,dtype='float64')
    assert 'dtype' not in calls[-1][1]
    with pytest.raises(ValueError,match='requires sampler'):dlmmc.fit(y,x,sampler=sampler,model=Trend()+Regression())
    with pytest.raises(TypeError):dlmmc.fit(y,x,sampler=sampler,settings=AffineSettings())
    with pytest.raises(ValueError,match='not both'):dlmmc.fit(y,x,sampler=sampler,settings=StanSettings(),samples=2)


def test_legacy_fit_does_not_change_defaults(monkeypatch):
    from dlmmc import api
    calls=[]
    monkeypatch.setattr(api,'fit',lambda *a,**k:calls.append(k))
    y,s,x=inputs();dlmmc.fit(y,x,uncertainty=s)
    assert 'samples' not in calls[0] and 'chains' not in calls[0]


def test_stan_grid_scope_and_summary(monkeypatch, tmp_path):
    from dlmmc import api
    calls=[]
    def fake(y,x,**kw):
        calls.append(kw)
        a=np.stack([y.values, y.values+2])[None]
        return xr.Dataset({'trend':(('chain','draw','time'),a)},coords={'time':y.time},attrs={'divergent_transitions':len(calls)})
    monkeypatch.setattr(api,'fit',fake)
    y,s,x=inputs();grid=y.expand_dims(panel=['a','b']).transpose('time','panel')
    p=dlmmc.fit(grid,x,uncertainty=s,sampler='cmdstan',output_dir=tmp_path/'stan')
    assert p.divergent_transitions.values.tolist()==[1,2]
    assert calls[1]['output_dir'].endswith('panel-0001')
    summary=dlmmc.summarize_trends(p,output_dir=tmp_path/'output')
    np.testing.assert_allclose(summary.trend.sel(panel='a'), y+1)
    np.testing.assert_allclose(summary.trend_sd,1)
    np.testing.assert_allclose(summary.trend_sd.sel(time='1998-01-01'), 1, rtol=1e-14)
    with pytest.raises(ValueError,match='five'):
        dlmmc.fit(y.expand_dims(panel=range(6)),x,sampler='cmdstan')


@pytest.mark.parametrize('backend',['numba','mps'])
def test_affine_public_dispatch_and_unanchored_output(tmp_path,backend):
    import torch
    if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
    y,s,x=inputs()
    settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2,chunk_size=4)
    fit=dlmmc.fit(y,x,uncertainty=s,sampler='affine',backend=backend,settings=settings,
                 output_dir=tmp_path/'fit',wall_seconds=120)
    assert fit.posterior().sizes['draw']==8
    summary=dlmmc.summarize_trends(fit,output_dir=tmp_path/'trends',wall_seconds=60)
    with np.load(tmp_path/'trends/cell-0000.npz') as z:
        np.testing.assert_array_equal(summary.trend.values[0],z['mean'])
        np.testing.assert_array_equal(summary.trend_sd.values[0],z['sd'])
        assert z['change_sd'][0]==0 and z['sd'][0]>0
    assert np.isfinite(summary.trend).all() and bool((summary.trend_sd>0).all())
    components = dlmmc.summarize_components(fit, parameter_draws=4, chunk_draws=2)
    assert np.isfinite(components.trend).all() and bool((components.trend_sd > 0).all())
    with pytest.raises(ValueError,match='incompatible'):
        dlmmc.fit(y,x,sampler='affine',chains=1,output_dir=tmp_path/'bad')
    with pytest.raises(TypeError):dlmmc.fit(y,x,sampler='affine',settings=StanSettings())
    from dlmmc.examples import load_trend_summary
    record=json.loads((tmp_path/'trends/execution.json').read_text());record['cells'][0]['sha256']='bad'
    (tmp_path/'trends/execution.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='checksum'):load_trend_summary(tmp_path/'trends')


def test_grid_plot_uses_absolute_mean_and_sd():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dlmmc.examples import plot_trend_grid
    dates=pd.date_range('1997-01-01',periods=24,freq='MS')
    mean=np.full((1,24),5.);sd=np.full_like(mean,.5)
    summary=xr.Dataset({'trend':(('panel','time'),mean),'trend_sd':(('panel','time'),sd)},
                       coords={'panel':['a'],'time':dates,'pressure':('panel',[10.]),'latitude':('panel',[5.])})
    figures=plot_trend_grid(summary)
    ax=figures[0].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),mean[0])
    vertices=ax.collections[0].get_paths()[0].vertices[:,1]
    assert vertices.min()==4.5 and vertices.max()==5.5
    plt.close('all')


def test_paper_domain_is_312():
    from pathlib import Path
    from dlmmc.examples import stratospheric_panels
    with xr.open_dataset(Path(__file__).resolve().parents[1]/'data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc') as ds:
        y,s=stratospheric_panels(ds)
        assert y.sizes['panel']==312 and y.dims==('time','panel')
        assert s.shape==y.shape
        assert bool((y.pressure.where(abs(y.latitude)<30,0)<=100).all())


def test_component_mixture_matches_independent_rts(tmp_path):
    import torch
    from dlmmc.reference import kalman,rts
    from dlmmc.tensor_target import model_components
    y,s,x=inputs(12)
    fit=dlmmc.fit(y,x,uncertainty=s,sampler='affine',backend='numba',
                 settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2),
                 output_dir=tmp_path/'fit',wall_seconds=120)
    output=dlmmc.summarize_components(fit,parameter_draws=4,chunk_draws=2)
    raw=fit.posterior().unconstrained.values.reshape(-1,1,4)
    ix=np.floor((np.arange(4)+.5)*len(raw)/4).astype(int)
    p=fit.prepared
    F,G,Q,C,_,_=model_components(torch.tensor(raw[ix]),*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
    means=[];vars=[]
    for i in range(4):
        m,c=rts(kalman(p.y[0].numpy(),p.variance[0].numpy(),p.observed[0].numpy(),
                      F[i,0].numpy(),G[i,0].numpy(),Q[i,0].numpy(),C[i,0].numpy()),G[i,0].numpy())
        means.append(m[:,0]*p.scale[0]+p.offset[0]);vars.append(c[:,0,0]*p.scale[0]**2)
    mean=np.mean(means,axis=0);variance=np.mean(vars,axis=0)+np.var(means,axis=0)
    np.testing.assert_allclose(output.trend[0],mean,atol=1e-8,rtol=0)
    np.testing.assert_allclose(output.trend_sd[0],np.sqrt(variance),atol=1e-8,rtol=0)
    assert set(('slope','seasonal','ar','beta')).issubset(output)
    assert np.isfinite(output.beta_sd).all()


def test_composite_covariance_and_coefficient_density(tmp_path):
    import torch
    from dlmmc.reference import kalman,rts
    from dlmmc.tensor_target import model_components
    y,s,x=inputs(24)
    fit=dlmmc.fit(y,x,uncertainty=s,sampler='affine',backend='numba',
        settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2),output_dir=tmp_path/'fit')
    out=dlmmc.summarize_components(fit,parameter_draws=4,chunk_draws=2)
    raw=fit.posterior().unconstrained.values.reshape(-1,1,4)
    ix=np.floor((np.arange(4)+.5)*len(raw)/4).astype(int);p=fit.prepared
    F,G,Q,C,_,_=model_components(torch.tensor(raw[ix]),*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
    means=[];variances=[]
    for i in range(4):
        f=F[i,0].numpy()
        m,c=rts(kalman(p.y[0].numpy(),p.variance[0].numpy(),p.observed[0].numpy(),
                      f,G[i,0].numpy(),Q[i,0].numpy(),C[i,0].numpy()),G[i,0].numpy())
        means.append(np.einsum('ti,ti->t',f,m)*p.scale[0]+p.offset[0])
        variances.append(np.einsum('ti,tij,tj->t',f,c,f)*p.scale[0]**2)
    np.testing.assert_allclose(out.fitted[0],np.mean(means,0),atol=1e-8,rtol=0)
    np.testing.assert_allclose(out.fitted_sd[0],np.sqrt(np.mean(variances,0)+np.var(means,0)),atol=1e-8,rtol=0)
    np.testing.assert_allclose(out.fitted,out.trend+out.seasonal+out.ar+out.regression,atol=1e-12,rtol=0)
    grid=out.beta_grid.values;pdf=out.beta_density.values
    np.testing.assert_allclose(np.trapezoid(pdf,grid,axis=-1),1,atol=1e-5,rtol=0)
    np.testing.assert_allclose(np.trapezoid(pdf*grid,grid,axis=-1),out.beta.isel(time=12),atol=1e-5,rtol=0)
