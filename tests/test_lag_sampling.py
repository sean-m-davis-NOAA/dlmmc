import json
import numpy as np
import pytest
import torch
from test_enso_lag import fixture
from dlmmc.lag_sampling import sample_enso_lag,resume_enso_lag,load_enso_lag
from dlmmc.lag_blocks import initialize_lag_blocks,advance_lag_blocks
from dlmmc.enso_lag import ENSOLagTarget


def clouds():
    rng=np.random.default_rng(2026113101);center=np.tile([-9.,-4.,-2.,0.,-1.],4)
    return [tuple(torch.tensor(center+.01*rng.normal(size=(20,20))) for _ in range(2)) for _ in range(4)]


def test_two_field_checkpoint_exact_resume_cache_and_deadline(tmp_path,monkeypatch):
    import dlmmc.lag_blocks as blocks
    p,sigma,lag,source=fixture();target=ENSOLagTarget(p,sigma,lag,source)
    state=initialize_lag_blocks(target,clouds()[0],seed=77,warmup=2,thin=2,provenance='twofield')
    full,x,lp,extra,ix=advance_lag_blocks(target,state,9)
    clock=iter([0.,0.,0.,2.]);monkeypatch.setattr(blocks.time,'time',lambda:next(clock))
    first,a,al,ae,ai=advance_lag_blocks(target,state,9,deadline_unix=1.)
    assert first.iteration==3
    end,b,bl,be,bi=advance_lag_blocks(target,first,6)
    assert torch.equal(torch.cat([a,b]),x) and torch.equal(torch.cat([al,bl]),lp)
    assert torch.equal(torch.cat([ae,be]),extra) and torch.equal(end.rng,full.rng)
    np.testing.assert_array_equal(np.r_[ai,bi],ix)
    assert int(full.proposed.sum())==7*2*20*(2+4)
    for h in (0,1):
        torch.testing.assert_close(target(full.halves[h]),full.logp[h],atol=1e-10,rtol=0)
        torch.testing.assert_close(target.local(full.halves[h].reshape(20,4,5)),full.extras[h],atol=0,rtol=0)


def test_public_lag_replay_and_separate_scope_guard(tmp_path):
    p,sigma,lag,source=fixture();kw=dict(half_walkers=20,warmup=2,initial_halves=clouds(),seed=791,chunk_sweeps=2,reference=True)
    full=sample_enso_lag(p,sigma,lag,source,output_dir=tmp_path/'full',retained_sweeps=7,**kw)
    part=sample_enso_lag(p,sigma,lag,source,output_dir=tmp_path/'part',retained_sweeps=3,**kw)
    resumed=resume_enso_lag(part.path,retained_sweeps=7)
    np.testing.assert_array_equal(full.posterior().unconstrained,resumed.posterior().unconstrained)
    np.testing.assert_array_equal(full.posterior().joint_log_probability,resumed.posterior().joint_log_probability)
    assert resumed.posterior().unconstrained.shape==(4,7,40,4,5)
    assert load_enso_lag(part.path).predictor.origin_month==source.origin_month
    with pytest.raises(ValueError,match='selection'):
        sample_enso_lag(p,sigma,lag,source,output_dir=tmp_path/'bad',phase='selected')
    with pytest.raises(ValueError,match='selection'):
        sample_enso_lag(p,sigma,lag,source,output_dir=tmp_path/'bad',phase='selected',selection={'stage':'05','explicit_user_response':'4'})
    assert not (tmp_path/'bad').exists()


def test_two_fields_prior_independence_and_conditional_nuisance():
    from types import SimpleNamespace
    from dataclasses import replace
    from dlmmc.spatial import GaussianSpatialPrior
    import math
    class PriorTarget:
        prepared=SimpleNamespace(cells=2)
        parameters=3;field_indices=(0,2)
        def __init__(self):
            ell=1/math.sqrt(-2*math.log(.5))
            sigma=GaussianSpatialPrior((0,1),(0,0),ell,1,'altitude','km','degrees',(0,0),(1,1),'test')
            self.fields=[sigma.bind(),replace(sigma,scale=(2,2),field_unit='months',upper=12).bind()]
        def local(self,u):
            sigma=u[...,0].exp();lag=12*torch.sigmoid(u[...,2])
            jac=math.log(12)-torch.nn.functional.softplus(u[...,2])-torch.nn.functional.softplus(-u[...,2])
            return u[...,0]+jac-.5*((u[...,1]-.4*sigma-.1*lag)/.7).square()-math.log(.7)-.5*math.log(2*math.pi)
        def joint(self,flat,local):
            u=flat.reshape(-1,2,3)
            return local.sum(-1)+self.fields[0].log_kernel(u[...,0].exp())+self.fields[1].log_kernel(12*torch.sigmoid(u[...,2]))
    target=PriorTarget();samples=[]
    for run in range(4):
        rng=np.random.default_rng(564+run);center=np.tile([-.3,.5,-1.5],2)
        half=tuple(torch.tensor(center+.01*rng.normal(size=(24,6))) for _ in range(2))
        state=initialize_lag_blocks(target,half,seed=894+run,warmup=500,thin=1,provenance='prior')
        _,draws,*_=advance_lag_blocks(target,state,2500);samples.append(draws.numpy().reshape(-1,2,3))
    x=np.concatenate(samples);sigma=np.exp(x[...,0]);lag=12/(1+np.exp(-x[...,2]));residual=(x[...,1]-.4*sigma-.1*lag)/.7
    rng=np.random.default_rng(895);z=rng.multivariate_normal([0,0],[[1,.5],[.5,1]],size=250000);z=z[np.all(z>0,axis=1)]
    np.testing.assert_allclose(sigma.mean(0),z.mean(0),atol=.025,rtol=0)
    lag_oracle=2*z[np.all(z<6,axis=1)]
    np.testing.assert_allclose(lag.mean(0),lag_oracle.mean(0),atol=.05,rtol=0)
    np.testing.assert_allclose(np.corrcoef(sigma,rowvar=False),np.corrcoef(z,rowvar=False),atol=.03,rtol=0)
    np.testing.assert_allclose(np.corrcoef(lag,rowvar=False),np.corrcoef(lag_oracle,rowvar=False),atol=.03,rtol=0)
    assert np.max(np.abs(np.corrcoef(np.c_[sigma,lag],rowvar=False)[:2,2:]))<.025
    np.testing.assert_allclose(residual.mean(0),0,atol=.025,rtol=0)
    np.testing.assert_allclose(residual.var(0),1,atol=.04,rtol=0)
