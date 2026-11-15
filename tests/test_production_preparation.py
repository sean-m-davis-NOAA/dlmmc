import numpy as np
import pytest
import torch
import xarray as xr
from dlmmc.prepare import prepare_data
from dlmmc.torch_filter import TorchTarget
from dlmmc.tensor_target import target
from dlmmc.api import _MODEL_SPECS


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_bounded_target_matches_reference_per_cell(model,dtype):
    rng=np.random.default_rng(53);n=24;g=5
    raw=rng.normal(size=(n,g));raw[4:7,2]=np.nan
    data=xr.DataArray(raw,dims=('time','panel'),coords={'time':np.arange(n),'panel':list('abcde')})
    uncertainty=xr.ones_like(data)*.2
    X=None if model=='noregs_ar1' else rng.normal(size=(n,2))
    p=prepare_data(data,X,uncertainty=uncertainty,model=model,dtype=dtype)
    u=p.y.new_tensor([-7 if name.startswith('sigma') else -.5 for name in p.parameter_names]).expand(6,g,-1).clone()
    value=TorchTarget(p)(u);expected=target(u,*p.arguments,model)
    torch.testing.assert_close(value,expected,atol=1e-8 if dtype=='float64' else 2e-4,rtol=0)
    assert p.cell_coordinates['panel'].tolist()==list('abcde')
    assert bool(torch.isfinite(p.y).all()) and not bool(p.observed[2,4:7].any())
    assert TorchTarget(p).F.shape[0]==g # no walker/time replication


def test_all_missing_constant_irregular_and_oversize_fail_before_fitting():
    data=xr.DataArray(np.arange(12.),dims='time',coords={'time':np.arange(12)})
    for bad in (data*0,data*np.nan,data.expand_dims(panel=np.arange(6))):
        with pytest.raises(ValueError):prepare_data(bad,model='noregs_ar1')
    with pytest.raises(ValueError):prepare_data(data.assign_coords(time=[0,1,2,4,5,6,7,8,9,10,11,12]),model='noregs_ar1')


def test_custom_prior_correction_and_support_rejection():
    raw=xr.DataArray(np.sin(np.arange(12)),dims='time',coords={'time':np.arange(12)})
    a=prepare_data(raw,model='noregs_ar1');b=prepare_data(raw,model='noregs_ar1',priors={'sigma_trend_prior':2e-4})
    u=torch.tensor([[[-9.,-5.,-2.,0.]]],dtype=torch.float64)
    actual=TorchTarget(b)(u)-TorchTarget(a)(u)
    sigma=torch.exp(u[...,0]);expected=np.log(.5)+.5*(sigma/1e-4)**2-.5*(sigma/2e-4)**2
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=0)
    u[0,0,2]=-1000;u[0,0,3]=1000
    assert bool(torch.isneginf(TorchTarget(a)(u)).all())
