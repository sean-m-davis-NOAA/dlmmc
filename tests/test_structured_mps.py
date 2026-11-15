"""Native-only experimental Metal tests; unavailable hardware is a skip."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import torch
from dlmmc import prepare_data
from dlmmc.examples import load_panels
from dlmmc.production import _cell
from dlmmc.structured_mps import StructuredMPSTarget,filter_mps,compile_kernel
from dlmmc.tensor_target import model_components
from dlmmc.numba_prototype import _batch_structured

native=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native MPS unavailable in this execution context')


def data():
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    return prepare_data(y,x,uncertainty=s,device='mps',dtype='float32')


def oracle(p,u):
    target=StructuredMPSTarget(p)
    _,G,Q,C,prior,valid=model_components(u,*target.empty,p.model,S=p.priors['S'],sampling=p.sampling)
    w,c=u.shape[:2];n=p.y.shape[1]
    def a(v):return v.cpu().numpy().astype(np.float64)
    F=np.ascontiguousarray(np.broadcast_to(a(target.F),(w,c,n,12)).reshape(w*c,n,12))
    ll,bad=_batch_structured(a(p.y),a(p.variance),p.observed.cpu().numpy(),F,
        a(G).reshape(-1,12,12),a(Q).reshape(-1,12,12),a(C).reshape(-1,12,12))
    assert np.all(bad==-1)
    return ll.reshape(w,c)+a(prior+target.prior_correction(u,valid))


def test_explicit_device_guard():
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    with pytest.raises(ValueError,match='MPS float32'):StructuredMPSTarget(prepare_data(y,x,uncertainty=s))


@native
def test_fixed_reference_and_layouts():
    torch.set_num_threads(1);p=data();compile_kernel()
    u=torch.tensor(np.array([-9.2,-4.6,-2.3,0.])+.1*np.random.default_rng(1401).normal(size=(4,5,4)),device='mps',dtype=torch.float32)
    with torch.no_grad():
        target=StructuredMPSTarget(p);full=target(u);ref=oracle(p,u)
        with pytest.raises(ValueError,match='autograd'):target(u.clone().requires_grad_())
        np.testing.assert_allclose(full.cpu().numpy(),ref,rtol=0,atol=.01)
        assert torch.equal(full,target(u))
        individual=torch.cat([StructuredMPSTarget(_cell(p,[g]))(u[:,g:g+1].contiguous()) for g in range(5)],1)
        assert torch.equal(full,individual)
        order=[4,2,0,3,1]
        perm=StructuredMPSTarget(_cell(p,order))(u[:,order].contiguous())
        assert torch.equal(perm[:,np.argsort(order)],full)
        chunk=torch.cat([StructuredMPSTarget(_cell(p,index))(u[:,index].contiguous()) for index in ([0,1],[2,3,4])],1)
        assert torch.equal(chunk,full)
        changed=replace(p,y=p.y.clone());changed.y[0]+=1
        other=StructuredMPSTarget(changed)(u)
        assert torch.equal(other[:,1:],full[:,1:])


@native
def test_masked_nan_and_prior_correction_and_support():
    p=data();p=replace(p,y=p.y[:,:24].contiguous(),variance=p.variance[:,:24].contiguous(),
        observed=p.observed[:,:24].contiguous(),regressors=p.regressors[:,:24].contiguous(),time=p.time[:24])
    p=replace(p,y=p.y.clone(),variance=p.variance.clone(),observed=p.observed.clone(),
        priors=dict(p.priors,sigma_trend_prior=.0002))
    p.observed[:,[0,5,23]]=False;p.y[:,[0,5,23]]=torch.nan;p.variance[:,[0,5,23]]=torch.nan
    u=torch.tensor([-9.2,-4.6,-2.3,0.],device='mps').expand(2,5,4).contiguous()
    with torch.no_grad():
        value=StructuredMPSTarget(p)(u);ref=oracle(p,u)
        np.testing.assert_allclose(value.cpu().numpy(),ref,rtol=0,atol=.01)
        bad=u.clone();bad[0,0,0]=100.
        result=StructuredMPSTarget(p)(bad)
        assert torch.isneginf(result[0,0]) and torch.equal(result[1],value[1])


@native
def test_invalid_innovation_raises_without_floor():
    p=_cell(data(),[0]);p=replace(p,variance=torch.full_like(p.variance,-1e10))
    u=torch.tensor([[[-9.2,-4.6,-2.3,0.]]],device='mps')
    with torch.no_grad(),pytest.raises(FloatingPointError,match='innovation'):
        StructuredMPSTarget(p)(u)
