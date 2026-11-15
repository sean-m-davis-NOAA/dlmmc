"""Regression for the diagnosed infinite raw-variance-SD boundary mechanism."""
import numpy as np
import pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices,physical_logprior
from dlmmc.reference import dense_condition


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_boundary_density_has_square_root_decay(model):
    rng=np.random.default_rng(847);n=10;k=0 if model=='noregs_ar1' else 2
    X=rng.normal(size=(n,k));y=rng.normal(size=n);noise=np.full(n,.2)
    theta=dict(sigma_trend=1e-4,sigma_AR=.1,sigma_seas=.005,sigma_reg=[1e-4]*k)
    priors=dict(sigma_trend_prior=1e-4,sigma_AR_prior=.5,sigma_seas_prior=.01,sigma_reg_prior=1e-4)
    rows=[]
    for eps in (1e-6,1e-7):
        if _MODEL_SPECS[model]['ar2']:theta.update(rhoAR1=.25,rhoAR2=.75-eps)
        else:theta.update(rhoAR1=1-eps)
        F,G,Q,C=matrices(model,X,theta)
        ll=dense_condition(y,noise,np.ones(n,bool),F,G,Q,C)[0]
        rows.append(ll+physical_logprior(model,theta,priors))
    slope=(rows[1]-rows[0])/np.log(.1)
    assert abs(slope-.5)<.002
