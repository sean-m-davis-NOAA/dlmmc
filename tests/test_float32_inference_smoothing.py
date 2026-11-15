"""Native float32 reconstruction prerequisite; not posterior qualification."""
import numpy as np
import pytest
import torch
from dlmmc.model_reference import matrices
from dlmmc.reference import dense_condition
from dlmmc.tensor_smoothing import conditional_draw


@pytest.mark.parametrize('fixed_variance',[None,0.])
def test_float32_simulation_smoother_against_dense_gaussian(fixed_variance):
    rng=np.random.default_rng(20261112);n=5
    theta=dict(sigma_trend=.01,sigma_AR=.1,sigma_seas=.01,rhoAR1=.3)
    F,G,Q,C=matrices('vanilla_ar1',rng.normal(size=(n,1)),theta,fixed_variance=fixed_variance)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.array([False,True,True,False,True])
    # Oracle conditions on the same represented float32 coefficients, with
    # independent dense float64 algebra; no ingestion/model-policy confound.
    arrays=[z if z.dtype==bool else z.astype(np.float32) for z in (y,v,mask,F,G,Q,C)]
    _,mean,cov=dense_condition(*[z if z.dtype==bool else z.astype(float) for z in arrays])
    tensors=[torch.from_numpy(z) for z in arrays];count=8192
    tensors[3]=tensors[3].expand(count,*F.shape)
    before=torch.get_rng_state().clone()
    output=conditional_draw(*tensors,generator=torch.Generator().manual_seed(20261112))
    assert output.dtype==torch.float32
    assert torch.equal(before,torch.get_rng_state())
    x=output.numpy().reshape(count,-1).astype(float);var=np.diag(cov)
    assert np.all(abs(x.mean(0)-mean)<6*np.sqrt(var/count))
    expected_se=np.sqrt((cov**2+var[:,None]*var[None,:])/(count-1))
    assert np.all(abs(np.cov(x,rowvar=False)-cov)<6*expected_se)
