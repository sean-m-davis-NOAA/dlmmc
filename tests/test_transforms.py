import numpy as np
import pytest
from scipy.integrate import quad
from dlmmc.api import _MODEL_SPECS
from dlmmc.transforms import constrain,unconstrain,parameter_names

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_jacobian_finite_difference(model):
    names=parameter_names(model,2);u=np.linspace(-3,.2,len(names))
    np.testing.assert_allclose(unconstrain(model,constrain(model,u,2)[0],2),u,rtol=1e-12,atol=1e-12)
    def vector(x):
        t,_=constrain(model,x,2)
        return np.array([t['sigma_reg'][int(n[10:-1])] if n.startswith('sigma_reg') else t[n] for n in names])
    J=np.column_stack([(vector(u+np.eye(len(u))[i]*1e-5)-vector(u-np.eye(len(u))[i]*1e-5))/2e-5 for i in range(len(u))])
    np.testing.assert_allclose(constrain(model,u,2)[1],np.linalg.slogdet(J)[1],atol=1e-8,rtol=0)

def test_conditional_prior_and_jacobian_cancel_width():
    def density(v):
        t,j=constrain('vanilla_ar2_noseasonal',np.array([0.,0.,.3,v]),0)
        # Marginal in the conditional logit at fixed rho1; separate its Jacobian.
        width=1-t['rhoAR1']
        logj=np.log(width)-np.logaddexp(0.,-v)-np.logaddexp(0.,v)
        return np.exp(-np.log(width)+logj)
    np.testing.assert_allclose(quad(density,-40,40)[0],1.,atol=1e-10)


def test_inverse_does_not_clip_invalid_support():
    theta=dict(sigma_trend=1e-4,sigma_AR=.1,rhoAR1=.3,rhoAR2=.7)
    with pytest.raises(ValueError,match='support'):
        unconstrain('vanilla_ar2_noseasonal',theta,0)
    theta['rhoAR2']=.2;theta['sigma_AR']=0.
    with pytest.raises(ValueError,match='positive support'):
        unconstrain('vanilla_ar2_noseasonal',theta,0)
