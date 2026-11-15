from dataclasses import replace
import math
import numpy as np
import pytest
import torch
import xarray as xr
from scipy.stats import multivariate_normal
from dlmmc.spatial import GaussianSpatialPrior, SpatialTarget, test_lengths_from_correlation as lengths
from dlmmc.prepare import prepare_data
from dlmmc.sampling import make_target


def prior():
    ez,ep=lengths(.2,10.)
    return GaussianSpatialPrior((0,0,.2,.2),(0,10,0,10),ez,ep,'log_pressure',
        'dimensionless_log_pressure_ratio','degrees_north',(0,)*4,(1e-4,)*4,'normalized_ozone_per_month')


def test_kernel_lengths_permutation_units_and_two_steps():
    p=prior();r=p.correlation()
    np.testing.assert_allclose(r,[[1,.5,.5,.25],[.5,1,.25,.5],[.5,.25,1,.5],[.25,.5,.5,1]],atol=1e-15)
    q=replace(p,vertical=(0,0,.4,.4));assert q.correlation()[0,2]==pytest.approx(.0625)
    ix=[3,0,2,1];permuted=replace(p,**{k:tuple(np.asarray(getattr(p,k))[ix]) for k in ('vertical','latitude','mean','scale')})
    np.testing.assert_array_equal(permuted.correlation(),r[np.ix_(ix,ix)])
    rad=replace(p,latitude=tuple(np.deg2rad(p.latitude)),length_latitude=np.deg2rad(p.length_latitude),latitude_unit='radians')
    np.testing.assert_allclose(rad.correlation(),r,atol=1e-15)
    assert np.linalg.eigvalsh(r).min()>0


def test_no_implicit_nugget_or_boundary_clipping():
    p=prior()
    with pytest.raises(ValueError,match='Duplicate'):replace(p,vertical=(0,)*4,latitude=(0,)*4)
    with pytest.raises(ValueError):replace(p,length_vertical=0)
    with pytest.raises(ValueError,match='Log-pressure'):replace(p,vertical_unit='km')
    f=p.bind()
    assert torch.isneginf(f.log_kernel(torch.zeros((1,4),dtype=torch.float64))).all()
    bounded=replace(p,upper=12).bind()
    assert torch.isneginf(bounded.log_kernel(torch.full((1,4),12.,dtype=torch.float64))).all()


def test_direct_gaussian_cross_terms_and_float32_precision():
    p=replace(prior(),mean=(.2,.3,.4,.5),scale=(1.,2.,1.5,.7))
    x=np.random.default_rng(142).uniform(.01,2,(128,4))
    oracle=multivariate_normal.logpdf(x,mean=p.mean,cov=p.covariance())
    values=p.bind().log_kernel(torch.tensor(x)).numpy()
    np.testing.assert_allclose(values,oracle,atol=2e-14,rtol=1e-14)
    low=p.bind(dtype=torch.float32).log_kernel(torch.tensor(x,dtype=torch.float32)).numpy()
    # Same .01 acceptance-difference ceiling; a small fixture, not dtype qualification.
    assert np.max(np.abs((low-low[0])-(values-values[0])))<.01


def prepared():
    data=xr.DataArray(np.sin(np.arange(18)[:,None]/2)+np.arange(4)[None,:]*.3,
        dims=('time','panel'),coords={'time':np.arange(18),'panel':['a','b','c','d']})
    return prepare_data(data,model='noregs_ar1',uncertainty=xr.ones_like(data)*.1)


def test_joint_target_replaces_only_sigma_prior_and_keeps_jacobian():
    p=prepared();p.priors['sigma_trend_prior']=2e-4
    field=replace(prior(),scale=(2e-4,)*4)
    target=SpatialTarget(p,field,backend='numba')
    rng=np.random.default_rng(147);u=torch.tensor(np.array([-9.,-4.,-2.,0.])+rng.normal(0,.1,(6,4,4)))
    theta=np.exp(u.numpy()[...,0]);scale=p.priors['sigma_trend_prior']
    old=(.5*math.log(2/math.pi)-math.log(scale)-.5*(theta/scale)**2).sum(-1)
    oracle=make_target(p,'torch')(u).numpy().sum(-1)-old+multivariate_normal.logpdf(theta,mean=field.mean,cov=field.covariance())
    np.testing.assert_allclose(target(u.reshape(6,-1)),oracle,rtol=0,atol=1e-8)
    independent=SpatialTarget(p,replace(field,length_vertical=.001,length_latitude=.001),backend='numba')
    # Omitted orthant normalizer is exactly 4 log(2) in the independent limit.
    np.testing.assert_allclose(independent(u.reshape(6,-1))+4*math.log(2),make_target(p,'numba')(u).sum(-1),atol=1e-10,rtol=0)
    changed=u.clone();changed[:,0,0]+=.2
    assert not torch.allclose((target(changed.reshape(6,-1))-target(u.reshape(6,-1))),
        (make_target(p,'numba')(changed)-make_target(p,'numba')(u)).sum(-1))
