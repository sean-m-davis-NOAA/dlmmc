import numpy as np
import pytest
import torch
import xarray as xr
from scipy.special import ndtr
from dlmmc.prepare import prepare_data
from dlmmc.probabilities import conditional_trend_probabilities
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices
from dlmmc.reference import dense_condition
from dlmmc.transforms import unconstrain


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_public_probability_dense_cross_time_oracle(model):
    rng=np.random.default_rng(723);n=8;k=0 if model=='noregs_ar1' else 1
    y=xr.DataArray(rng.normal(size=n),dims='time',coords={'time':np.arange(n)})
    y[2]=np.nan;X=rng.normal(size=(n,k))
    p=prepare_data(y,X if k else None,uncertainty=xr.ones_like(y)*.1,model=model)
    theta=dict(sigma_trend=.01,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    u=unconstrain(model,theta,k)[None,None,:];pairs=[(0,3),(3,7),(0,7)]
    before=torch.get_rng_state().clone()
    actual=conditional_trend_probabilities(p,u,pairs)[0,0]
    assert torch.equal(before,torch.get_rng_state())
    F,G,Q,C=matrices(model,X,theta)
    _,m,v=dense_condition(p.y[0].numpy(),p.variance[0].numpy(),p.observed[0].numpy(),F,G,Q,C)
    expected=[]
    for a,b in pairs:
        weight=np.zeros(n*len(G));weight[a*len(G)]=-1;weight[b*len(G)]=1
        expected.append(ndtr((weight@m.ravel())/np.sqrt(weight@v@weight)))
    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)
    with pytest.raises(ValueError,match='float64'):conditional_trend_probabilities(p,u.astype(np.float32),pairs)
