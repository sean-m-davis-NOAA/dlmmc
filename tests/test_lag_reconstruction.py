from dataclasses import replace
import numpy as np
import pytest
import torch
from test_enso_lag import fixture
from test_lag_sampling import clouds
from dlmmc.enso_lag import ENSOLagTarget
from dlmmc.reconstruction import conditional_outputs
from dlmmc.lag_reconstruction import conditional_lag_outputs,reconstruct_enso_lag
from dlmmc.lag_sampling import sample_enso_lag
from dlmmc.lag_diagnostics import diagnose_enso_lag


def test_fixed_lag_conditional_moments_match_static_inputs():
    p,sigma,lag,source=fixture();target=ENSOLagTarget(p,sigma,lag,source,backend='torch')
    u=torch.tensor(np.tile([-9.,-4.,-2.,0.,-1.],(3,4,1)))
    u[:,:,-1]+=torch.arange(3,dtype=torch.float64)[:,None]*.2
    actual=conditional_lag_outputs(p,source,sigma,lag,u,seed=741)
    x=target.regressors(u)
    for i in range(3):
        expected=conditional_outputs(replace(p,regressors=x[i]),u[i:i+1,:,:-1].numpy(),seed=741)
        for key in ('conditional_mean','conditional_variance','signal_conditional_mean','signal_conditional_variance'):
            np.testing.assert_allclose(actual[key][i],expected[key][0],atol=1e-10,rtol=0)
    np.testing.assert_array_equal(actual['enso_lag'],12*torch.sigmoid(u[...,-1]))
    invalid=u.clone();invalid[0,0,-1]=1000
    with pytest.raises(ValueError,match='strict supported'):
        conditional_lag_outputs(p,source,sigma,lag,invalid,seed=741)


def test_short_lag_window_does_not_authorize_reconstruction(tmp_path):
    p,sigma,lag,source=fixture()
    fit=sample_enso_lag(p,sigma,lag,source,output_dir=tmp_path/'fit',half_walkers=20,warmup=2,initial_halves=clouds(),seed=791,chunk_sweeps=5,reference=True,retained_sweeps=20)
    diagnostics=diagnose_enso_lag(fit,trace_review='Unit-test short window; no scientific qualification')
    assert not diagnostics['eligible']
    assert 'lag_ENSO_months' in diagnostics['quantities']
    assert 'sigma_lag_product:0' in diagnostics['joint_quantities']
    with pytest.raises(ValueError,match='Diagnosed lag parameter'):
        reconstruct_enso_lag(fit,diagnostics,output_dir=tmp_path/'outputs')
    assert not (tmp_path/'outputs').exists()


def test_lag_analytic_probabilities_match_fixed_static_predictors():
    from dlmmc.lag_probabilities import conditional_lag_trend_probabilities
    from dlmmc.probabilities import conditional_trend_probabilities
    p,sigma,lag,source=fixture();target=ENSOLagTarget(p,sigma,lag,source,backend='torch')
    u=torch.tensor(np.tile([-9.,-4.,-2.,0.,-1.],(2,4,1)))
    u[1,:,-1]+=.4;contrasts=[(0,9),(9,17)]
    actual=conditional_lag_trend_probabilities(p,source,sigma,lag,u.numpy(),contrasts)
    for i in range(2):
        shifted=replace(p,regressors=target.regressors(u)[i])
        expected=conditional_trend_probabilities(shifted,u[i:i+1,:,:-1].numpy(),contrasts)
        np.testing.assert_allclose(actual[i],expected[0],atol=1e-12,rtol=0)
