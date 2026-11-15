import numpy as np
import pytest
import torch
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices
from dlmmc.tensor_smoothing import conditional_draw
from dlmmc.simulation_smoothing_mean import conditional_draw_mean


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
@pytest.mark.parametrize('zero',[False,True])
def test_mean_only_preserves_every_draw_and_rng(model,zero):
    n=12;k=0 if model=='noregs_ar1' else 2
    rng=np.random.default_rng(822)
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta,fixed_variance=0 if zero else None)
    y=rng.normal(size=n);v=np.full(n,0. if zero else .1)
    mask=np.ones(n,bool);mask[[0,3,4,11]]=False;y[~mask]=np.nan;v[~mask]=np.inf
    args=[torch.from_numpy(x) for x in [y,v,mask,F,G,Q,C]]
    args[3]=args[3].expand(16,*F.shape)
    a=torch.Generator().manual_seed(734);b=torch.Generator().manual_seed(734)
    before=torch.get_rng_state().clone()
    expected=conditional_draw(*args,generator=a)
    actual=conditional_draw_mean(*args,generator=b)
    assert torch.equal(actual,expected)
    assert torch.equal(a.get_state(),b.get_state())
    assert torch.equal(before,torch.get_rng_state())
