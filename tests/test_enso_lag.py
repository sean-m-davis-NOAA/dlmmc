from dataclasses import replace
import math
import numpy as np
import pytest
import torch
import xarray as xr
from scipy.stats import multivariate_normal
from dlmmc.enso_lag import MonthlyENSO,ENSOLagTarget
from dlmmc.spatial import SpatialTarget
from dlmmc.prepare import prepare_data
from test_spatial_prior import prior


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_monthly_sign_fractional_integer_endpoints_and_no_clamping(dtype):
    spec=MonthlyENSO(np.arange(-12,20),np.arange(-12,20,dtype=float)**2,np.arange(8),12)
    f=spec.bind(cells=2,dtype=dtype,device='cpu')
    lag=torch.tensor([[0.,1.],[2.5,12.]],dtype=dtype);actual=f(lag).numpy()
    expected=np.array([[np.interp(np.arange(8)-l,spec.source_month+spec.origin_month,spec.values) for l in row] for row in lag.numpy()])
    np.testing.assert_array_equal(actual,expected)
    assert actual[0,1,1]==0 # Earlier ENSO(1-1), not later ENSO(1+1).
    with pytest.raises(ValueError,match='support'):f(torch.tensor([[-.1,0]],dtype=dtype))
    end=MonthlyENSO([0,1,2],[1,3,5],[1,2],1).bind(cells=1,dtype=dtype,device='cpu')
    np.testing.assert_array_equal(end(torch.zeros((1,1),dtype=dtype)),[[[3,5]]])


def test_coverage_gaps_and_monthly_anchors():
    with pytest.raises(ValueError,match='cover'):MonthlyENSO([0,1,2],[1,2,3],[0,1],1)
    with pytest.raises(ValueError,match='gaps'):MonthlyENSO([0,1,3],[1,2,3],[2,3],1)
    with pytest.raises(ValueError,match='Finite'):MonthlyENSO([0,1,2],[1,np.nan,3],[1,2],1)
    dates=np.arange('2000-01','2002-01',dtype='datetime64[M]').astype('datetime64[D]')+np.timedelta64(14,'D')
    spec=MonthlyENSO.from_dates(dates,np.arange(len(dates)),dates[12:])
    assert spec.origin_month==np.datetime64('2000-01').astype('datetime64[M]').astype(int)
    with pytest.raises(ValueError,match='anchor'):MonthlyENSO.from_dates(dates,np.arange(len(dates)),dates[12:]+np.timedelta64(1,'D'))


def fixture():
    t=np.arange(18);data=xr.DataArray(2+np.sin(t[:,None]/2)+np.arange(4)[None,:]*.3,dims=('time','panel'),coords={'time':t,'panel':['a','b','c','d']})
    x=xr.DataArray(np.stack([np.cos(t/3),t/10],-1),dims=('time','regressor'),coords={'time':t,'regressor':['ENSO','other']})
    p=prepare_data(data,x,uncertainty=xr.ones_like(data)*.1)
    source=np.arange(-12,18);predictor=MonthlyENSO(source,np.cos(source/3),t)
    sigma=prior();lag=replace(sigma,mean=(0,)*4,scale=(2,)*4,field_unit='months',upper=12)
    return p,sigma,lag,predictor


def test_joint_lag_prior_jacobian_and_unaffected_regressors():
    p,sigma,lag,predictor=fixture();target=ENSOLagTarget(p,sigma,lag,predictor,backend='torch')
    rng=np.random.default_rng(664);u=torch.tensor(np.array([-9.,-4.,-2.,0.,-1.])+rng.normal(0,.2,(5,4,5)))
    x=target.regressors(u)
    np.testing.assert_array_equal(x[...,1],p.regressors[None,...,1].expand(5,-1,-1))
    value=target(u.reshape(5,-1)).numpy();expected=[]
    lag_values=12*torch.sigmoid(u[...,-1]);jac=math.log(12)-torch.nn.functional.softplus(u[...,-1])-torch.nn.functional.softplus(-u[...,-1])
    for i in range(5):
        # Independent construction: one static-regressor spatial target for
        # each lag vector; then the full Gaussian lag prior plus Jacobian.
        q=replace(p,regressors=x[i]);base=SpatialTarget(q,sigma,backend='torch')(u[i:i+1,:,:4].reshape(1,-1))[0]
        expected.append(float(base+jac[i].sum())+multivariate_normal.logpdf(lag_values[i].numpy(),mean=lag.mean,cov=lag.covariance()))
    np.testing.assert_allclose(value,expected,atol=1e-10,rtol=0)
    fast=ENSOLagTarget(p,sigma,lag,predictor,backend='numba')
    np.testing.assert_allclose(fast(u.reshape(5,-1)),value,atol=1e-8,rtol=0)
    invalid=u.clone();invalid[0,1,-1]=1000
    assert torch.isneginf(target(invalid.reshape(5,-1))[0])
    bad=replace(lag,length_vertical=lag.length_vertical*2)
    with pytest.raises(ValueError,match='share'):ENSOLagTarget(p,sigma,bad,predictor)
    wrong=MonthlyENSO(np.arange(-12,19),np.cos(np.arange(-12,19)),np.arange(1,19))
    with pytest.raises(ValueError,match='timestamps'):ENSOLagTarget(p,sigma,lag,wrong)


def test_lag_fixed_permutation_chunks_and_uncoupled_cross_talk():
    p,sigma,lag,source=fixture()
    rng=np.random.default_rng(665)
    u=torch.tensor(np.array([-9.,-4.,-2.,0.,-1.])+rng.normal(0,.2,(5,4,5)))
    def subset(indices,prior):
        return replace(prior,**{k:tuple(np.asarray(getattr(prior,k))[indices]) for k in ('vertical','latitude','mean','scale')})
    def prepared(indices):
        return replace(p,y=p.y[indices],variance=p.variance[indices],observed=p.observed[indices],regressors=p.regressors[indices],scale=p.scale[indices],offset=p.offset[indices],cell_coordinates={k:v[indices] for k,v in p.cell_coordinates.items()})
    full=ENSOLagTarget(p,sigma,lag,source)
    ix=[2,0,3,1];permuted=ENSOLagTarget(prepared(ix),subset(ix,sigma),subset(ix,lag),source)
    torch.testing.assert_close(full(u.reshape(5,-1)),permuted(u[:,ix].reshape(5,-1)),atol=1e-9,rtol=0)
    # Matching Gaussian marginals in the exact uncoupled kernel limit.
    sigma=replace(sigma,length_vertical=1e-4,length_latitude=1e-4)
    lag=replace(lag,length_vertical=1e-4,length_latitude=1e-4)
    full=ENSOLagTarget(p,sigma,lag,source)
    for groups in ([[0],[1],[2],[3]],[[0,1],[2,3]]):
        total=sum(ENSOLagTarget(prepared(ix),subset(ix,sigma),subset(ix,lag),source)(u[:,ix].reshape(5,-1)) for ix in groups)
        torch.testing.assert_close(full(u.reshape(5,-1)),total,atol=1e-9,rtol=0)
    changed=u.clone();changed[:,1,-1]+=.5
    torch.testing.assert_close(full.local(u)[:,[0,2,3]],full.local(changed)[:,[0,2,3]],atol=0,rtol=0)
