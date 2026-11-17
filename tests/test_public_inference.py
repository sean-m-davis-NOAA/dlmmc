import numpy as np
import pytest
import torch
import xarray as xr
from dlmmc.api import _MODEL_SPECS
from dlmmc.prepare import prepare_data
from dlmmc.inference import window_diagnostics,physical_parameters,diagnose_fit
from dlmmc.reconstruction import conditional_outputs,reconstruct_fit
from dlmmc.model_reference import matrices
from dlmmc.reference import dense_condition
from dlmmc.transforms import unconstrain


def test_independent_ensemble_diagnostics_detect_separation_and_duplicate_walkers():
    x=np.random.default_rng(729).normal(size=(4,2400,16,2))
    good=window_diagnostics(x)
    assert good['eligible']
    duplicate=window_diagnostics(np.repeat(x,2,axis=2))
    # Duplicating an ensemble's members creates no new information.
    np.testing.assert_allclose(good['minimum_ess'],duplicate['minimum_ess'],rtol=2e-5)
    separated=x.copy();separated[0,:,:,0]+=2
    assert not window_diagnostics(separated)['eligible']
    collapsed=x*.001;collapsed[:,1200:]*=1000
    assert not window_diagnostics(collapsed)['eligible']
    assert not window_diagnostics(x[:,:8])['eligible']
    independent_chains=np.random.default_rng(726).normal(size=(4,4000,1,2))
    assert window_diagnostics(independent_chains)['eligible']


def test_mc_stability_requires_blocks_and_retains_strict_flags():
    from dlmmc.stability import mc_stability
    from dlmmc.diagnostics import ensemble_diagnostics
    rng=np.random.default_rng(127)
    x=rng.normal(size=(4,4000,16,2))
    for t in range(1,x.shape[1]):x[:,t]=.8*x[:,t-1]+x[:,t]
    result=window_diagnostics(x)
    assert result['eligible']
    assert result['mc_stability']['blocks_per_run']>=6
    assert 'strict_eligible' in result
    impossible={'mean':{'iat_by_run':[[1000]],'mean_mcse':[1]}}
    assert mc_stability(x,impossible)['status']=='INSUFFICIENT_COMPLETE_BLOCKS'
    drifting=x.copy();drifting[:,2000:,:,0]+=3
    assert not window_diagnostics(drifting)['eligible']


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_public_conditional_outputs_match_independent_gaussian(model):
    n=5;k=0 if model=='noregs_ar1' else 1
    rng=np.random.default_rng(840);y=rng.normal(size=n);y[1]=np.nan
    data=xr.DataArray(y,dims='time',coords={'time':np.arange(n)})
    X=rng.normal(size=(n,k))
    p=prepare_data(data,X if k else None,uncertainty=xr.ones_like(data)*.3,model=model)
    theta=dict(sigma_trend=.01,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    u=unconstrain(model,theta,k);count=4096
    before=torch.get_rng_state().clone()
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:out=conditional_outputs(p,np.broadcast_to(u,(count,1,len(u))),seed=870)
    finally:torch.set_num_threads(old)
    assert torch.equal(before,torch.get_rng_state())
    F,G,Q,C=matrices(model,X,theta,S=p.priors["S"])
    _,mean,cov=dense_condition(p.y[0].numpy(),p.variance[0].numpy(),p.observed[0].numpy(),F,G,Q,C)
    dim=len(G);mean=mean.reshape(n,dim)*p.scale[0];mean[:,0]+=p.offset[0]
    cov=cov*p.scale[0]**2
    np.testing.assert_allclose(out['conditional_mean'][0,0],mean,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(out['conditional_variance'][0,0],np.diag(cov).reshape(n,dim),rtol=1e-8,atol=1e-9)
    draws=out['states'][:,0].reshape(count,-1);var=np.diag(cov)
    assert np.all(abs(draws.mean(0)-mean.ravel())<6*np.sqrt(var/count))
    se=np.sqrt((cov**2+var[:,None]*var[None,:])/(count-1))
    assert np.all(abs(np.cov(draws,rowvar=False)-cov)<6*se)
    assert np.isnan(out['posterior_predictive'][:,0,1]).all()
    noise=out['posterior_predictive'][:,0,0]-out['latent_signal'][:,0,0]
    assert abs(noise.var()-.09)<6*.09*np.sqrt(2/(count-1))
    names,physical=physical_parameters(u,model,k)
    assert physical[names.index('rhoAR1')]==pytest.approx(.3)


def test_reconstruction_rejects_short_smoke_before_writing(tmp_path):
    from dlmmc.sampling import sample_parameters
    from test_public_affine_api import setup
    p,h=setup()
    fit=sample_parameters(p,output_dir=tmp_path/'fit',backend='numba',retained_sweeps=8,
                          half_walkers=8,initial_halves=h,warmup=0)
    report=diagnose_fit(fit,trace_review='Short smoke traces inspected; inadequate length')
    assert not report['eligible']
    with pytest.raises(ValueError,match='diagnosed'):
        reconstruct_fit(fit,report,output_dir=tmp_path/'states',selected_per_run=4)
    assert not (tmp_path/'states').exists()


def test_diagnosed_reconstruction_is_bounded_resumable_and_includes_conditional_variance(tmp_path):
    from types import SimpleNamespace
    data=xr.DataArray(np.sin(np.arange(5)),dims='time',coords={'time':np.arange(5)})
    p=prepare_data(data,model='noregs_ar1',uncertainty=xr.ones_like(data)*.1)
    u=np.random.default_rng(819).normal(size=(4,2400,16,1,4))*.2+np.array([-9.,-4.6,-2.3,0.])
    posterior=xr.Dataset({'unconstrained':(('run','draw','walker','cell','parameter'),u)},
        coords={'draw':np.arange(2400),'cell':[0]})
    fit=SimpleNamespace(prepared=p,config={'provenance':'independent-synthetic-test-fixture','seed':123,'runs':4},posterior=lambda:posterior)
    report=diagnose_fit(fit,trace_review='Known independent synthetic fixture; no sampling qualification claim')
    assert report['eligible']
    args=dict(output_dir=tmp_path/'states',selected_per_run=8,chunk_draws=4,seed=2026111900)
    result=reconstruct_fit(fit,report,**args)
    chunks=list(result.iter_chunks());assert len(chunks)==8 and all(c.sizes['selected']==4 for c in chunks)
    before=(result.path/'index.json').read_bytes()
    resumed=reconstruct_fit(fit,report,**args)
    for a,b in zip(chunks,resumed.iter_chunks()):xr.testing.assert_identical(a,b)
    moments=resumed.moments()
    means=np.concatenate([c.signal_conditional_mean.values for c in chunks])
    conditional=np.concatenate([c.signal_conditional_variance.values for c in chunks])
    np.testing.assert_allclose(moments.variance,conditional.mean(0)+means.var(0),rtol=1e-10,atol=1e-12)
    assert np.all(moments.variance.values>means.var(0))
    output_report=resumed.diagnose_outputs(time_indices=[0,2,4],contrasts=[(0,4)],amplitude_times=[0,4])
    assert not output_report['eligible'] #32 selected draws cannot certify ESS2000
    assert 'trend[4]' in output_report['quantities']
    with pytest.raises(ValueError,match='provenance'):
        reconstruct_fit(fit,report,**{**args,'seed':2026111901})
    posterior.unconstrained.values[0,0,0,0,0]+=.001
    with pytest.raises(ValueError,match='diagnosed'):
        reconstruct_fit(fit,report,**args)
