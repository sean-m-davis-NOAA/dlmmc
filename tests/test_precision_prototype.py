import numpy as np
import torch
import pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices
from dlmmc.reference import kalman
from dlmmc.precision_prototype import filter_tensor

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_tensor_reference_isolation(model):
    rng=np.random.default_rng(20261104); n=12;k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta)
    y=rng.normal(size=(5,n));var=np.full_like(y,.1);mask=rng.random(y.shape)>.2
    y[~mask]=np.nan;var[~mask]=np.inf
    tensors=[torch.from_numpy(a) for a in (y,var,mask,F,G,Q,C)]
    actual=filter_tensor(*tensors)
    for i in range(5):
        expected=kalman(y[i],var[i],mask[i],F,G,Q,C)
        for a,e in zip(actual,expected):np.testing.assert_allclose(a[i],e,rtol=1e-8,atol=1e-10)
        single=filter_tensor(*[x[i:i+1] if j<3 else x for j,x in enumerate(tensors)])
        for a,b in zip(single,actual):torch.testing.assert_close(a[0],b[i],rtol=1e-8,atol=1e-10)
    for order in [[4,2,0,3,1],[0,1],[2,3,4]]:
        result=filter_tensor(*[x[order] if j<3 else x for j,x in enumerate(tensors)])
        for a,b in zip(result,actual):torch.testing.assert_close(a,b[order],rtol=1e-8,atol=1e-10)
    tensors[0][0]=100
    changed=filter_tensor(*tensors)
    for a,b in zip(changed,actual):torch.testing.assert_close(a[1:],b[1:],rtol=0,atol=0)
