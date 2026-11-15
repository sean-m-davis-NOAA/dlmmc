import numpy as np
import pytest
import torch
from scipy.special import ndtr
from dlmmc.conditional_statistics_stable import normal_positive_probability


def test_gaussian_rare_tail_against_independent_scipy_cdf():
    z=np.array([-30.,-14.,-12.,-9.,-6.,-1.,0.,1.,6.,9.,14.])
    actual=normal_positive_probability(torch.from_numpy(z),torch.ones(len(z),dtype=torch.float64)).numpy()
    np.testing.assert_allclose(actual,ndtr(z),rtol=3e-13,atol=0)
    assert np.all(actual[:4]>0)
    # Preserve the actual arithmetic failure instead of masking it with epsilon.
    assert torch.special.ndtr(torch.tensor(-12.,dtype=torch.float64))==0


def test_deterministic_gaussian_uses_strict_positive_event():
    np.testing.assert_array_equal(normal_positive_probability(torch.tensor([-1.,0.,1.],dtype=torch.float64),torch.zeros(3,dtype=torch.float64)).numpy(),[0.,0.,1.])


@pytest.mark.parametrize('variance',[-1.,float('nan')])
def test_no_variance_floor_or_nonfinite_substitution(variance):
    with pytest.raises(ValueError):normal_positive_probability(torch.tensor(0.,dtype=torch.float64),torch.tensor(variance,dtype=torch.float64))


@pytest.mark.parametrize('model',['vanilla_ar1','vanilla_ar2','noregs_ar1','dynregs_ar1','vanilla_ar1_noseasonal','vanilla_ar2_noseasonal'])
def test_cached_contrasts_exactly_match_uncached_joseph_operations(model):
    from dlmmc.model_reference import matrices
    from dlmmc.precision_prototype import filter_tensor
    from dlmmc.conditional_statistics_stable import stable_positive_trend_probabilities
    from dlmmc.conditional_statistics_cached import cached_positive_trend_probabilities
    rng=np.random.default_rng(23);n=12;k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.array([False,True,True,False,True,True,False,True,True,True,False,True])
    tensors=[torch.from_numpy(a) for a in (y,v,mask,F,G,Q,C)];filtered=filter_tensor(*tensors);contrasts=[(0,5),(5,11),(0,11)]
    expected=stable_positive_trend_probabilities(filtered,tensors[4],tensors[5],contrasts)
    actual=cached_positive_trend_probabilities(filtered,tensors[4],tensors[5],contrasts)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
