import numpy as np
import pytest
import torch
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices
from dlmmc.reference import kalman
from dlmmc.square_root_prototype import filter_square_root


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
@pytest.mark.parametrize('zero',[False,True])
def test_square_root_independent_reference_and_five_cell_isolation(model,zero):
    rng=np.random.default_rng(20261104);n=12;k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta,fixed_variance=0 if zero else None)
    y=rng.normal(size=(5,n));v=np.full_like(y,0. if zero else .1);mask=rng.random(y.shape)>.2
    y[~mask]=np.nan;v[~mask]=np.inf
    tensors=[torch.from_numpy(a) for a in (y,v,mask,F,G,Q,C)]
    actual=filter_square_root(*tensors)
    for i in range(5):
        expected=kalman(y[i],v[i],mask[i],F,G,Q,C)
        for a,e in zip(actual,expected):np.testing.assert_allclose(a[i],e,rtol=1e-8,atol=1e-10)
    for order in [[0],[4,2,0,3,1],[0,1],[2,3,4]]:
        result=filter_square_root(*[x[order] if j<3 else x for j,x in enumerate(tensors)])
        for a,b in zip(result,actual):torch.testing.assert_close(a,b[order],rtol=1e-8,atol=1e-10)
    tensors[0][0]=100
    changed=filter_square_root(*tensors)
    for a,b in zip(changed,actual):torch.testing.assert_close(a[1:],b[1:],rtol=0,atol=0)


def test_preserved_scalar_float32_representation_obstruction():
    # Saved valid full-posterior stretch proposal, not a relaxed scientific gate.
    delta=-688372.9150953775
    nearest=np.float32(delta)
    candidates=[np.nextafter(nearest,np.float32(-np.inf)),nearest,np.nextafter(nearest,np.float32(np.inf))]
    assert min(abs(float(x)-delta) for x in candidates)>.01


def test_preserved_full_posterior_filter_arithmetic_failure():
    from pathlib import Path
    from dlmmc.tensor_target import model_components
    f=np.load(Path(__file__).parent/'fixtures/full_posterior_filter_failure.npz')
    args=[torch.as_tensor(f[k],dtype=torch.bool if k=='observed' else torch.float32) for k in ['y','variance','observed','X']]
    F,G,Q,C,prior,_=model_components(torch.as_tensor(f['u'],dtype=torch.float32),*args,'vanilla_ar1')
    result=filter_square_root(*args[:3],F,G,Q,C)
    lp=result[0].sum(-1)+prior
    assert lp.dtype==torch.float32
    assert abs(float(lp[1]-lp[0])-float(f['reference_delta']))<=.01
