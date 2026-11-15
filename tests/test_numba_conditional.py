import numpy as np
import pytest
import torch
from dlmmc.model_reference import matrices
from dlmmc.precision_prototype import filter_tensor
from dlmmc.tensor_smoothing import contrast_moments
from dlmmc.numba_conditional import contrast_statistics, positive_trend_probabilities
from dlmmc.conditional_statistics_stable import stable_positive_trend_probabilities


@pytest.mark.parametrize('model',['vanilla_ar1','vanilla_ar2','noregs_ar1','dynregs_ar1','vanilla_ar1_noseasonal','vanilla_ar2_noseasonal'])
def test_full_joseph_contrast_matches_qualified_torch(model):
    rng=np.random.default_rng(23);n=12;k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta)
    y=rng.normal(size=n);v=np.full(n,.1)
    mask=np.array([False,True,True,False,True,True,False,True,True,True,False,True])
    y[~mask]=np.nan;v[~mask]=np.inf
    arrays=(y,v,mask,F,G,Q,C);tensors=[torch.from_numpy(x) for x in arrays]
    filtered=filter_tensor(*tensors);pairs=[(0,5),(5,11),(0,11)]
    left=torch.zeros(len(G),dtype=torch.float64);left[0]=-1
    expected=np.stack([np.array([x.item() for x in contrast_moments(filtered,tensors[4],tensors[5],a,b,left,-left)]) for a,b in pairs])
    np.testing.assert_allclose(contrast_statistics(*arrays,pairs),expected,rtol=1e-8,atol=1e-10)
    probability=stable_positive_trend_probabilities(filtered,tensors[4],tensors[5],pairs).numpy()
    np.testing.assert_allclose(positive_trend_probabilities(*arrays,pairs),probability,rtol=1e-8,atol=1e-10)
    batched=(np.stack([y,y]),np.stack([v,v]),np.stack([mask,mask]),*[np.stack([x,x]) for x in (F,G,Q,C)])
    values=contrast_statistics(*batched,pairs)
    np.testing.assert_array_equal(values[0],values[1])
    altered=list(batched);altered[0]=altered[0].copy();altered[0][0,mask]+=100
    np.testing.assert_array_equal(contrast_statistics(*altered,pairs)[1],values[1])


def test_no_floors_and_explicit_input_contract():
    n=5;y=np.arange(n,dtype=float);v=np.ones(n);mask=np.ones(n,bool)
    F=np.ones((n,1));G=np.ones((1,1));Q=np.ones((1,1));C=np.ones((1,1))
    args=[y,v,mask,F,G,Q,C,[(0,4)]]
    for bad in [[(-1,4)],[(2,2)],[(0,5)],[(0.5,4.)]]:
        with pytest.raises(ValueError):contrast_statistics(*args[:-1],bad)
    args[1]=-v
    with pytest.raises(ValueError):contrast_statistics(*args)
    args[1]=v.astype(np.float32)
    with pytest.raises(ValueError):contrast_statistics(*args)


def test_all_missing_exact_zero_process_contrast():
    n=5
    args=(np.full(n,np.nan),np.full(n,np.inf),np.zeros(n,bool),np.ones((n,1)),np.ones((1,1)),np.zeros((1,1)),np.ones((1,1)),[(0,4)])
    np.testing.assert_array_equal(contrast_statistics(*args),[[0.,0.]])
    np.testing.assert_array_equal(positive_trend_probabilities(*args),[0.])
