import numpy as np
import pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.transforms import parameter_names,constrain
from dlmmc.model_reference import matrices
from dlmmc.validation_parameters import physical_parameter_draws,stationary_ar_variance


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_all_parameter_axes_and_ar_covariance_match_scalar_reference(model):
    k=0 if model=='noregs_ar1' else 3;names=parameter_names(model,k)
    u=np.random.default_rng(17).normal(size=(4,3,2,len(names)));actual=physical_parameter_draws(model,u,k)
    arvar=stationary_ar_variance(model,actual,k)
    for index in np.ndindex(u.shape[:-1]):
        theta,_=constrain(model,u[index],k)
        expected=[theta['sigma_reg'][int(n.split('[')[1][:-1])] if n.startswith('sigma_reg') else theta[n] for n in names]
        np.testing.assert_array_equal(actual[index],expected)
        _,G,_,C=matrices(model,np.zeros((2,k)),theta)
        ar=(6 if _MODEL_SPECS[model]['seasonal'] else 2)+k
        np.testing.assert_allclose(arvar[index],C[ar,ar],rtol=2e-15,atol=0)


def test_diagnostic_conversion_does_not_clip_invalid_represented_support():
    with pytest.raises(ValueError):physical_parameter_draws('vanilla_ar2',np.array([0.,0.,0.,1000.,0.]),2)
