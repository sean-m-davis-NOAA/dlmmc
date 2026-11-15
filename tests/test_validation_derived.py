import numpy as np
import pytest
from dlmmc.validation_derived import finite_moment_summaries,raw_variance_diagnostics


def test_log_variance_policy_preserves_other_quantities_and_raw_input():
    names=['rhoAR1','stationary_AR_variance','trend[0]'];raw=np.array([[.8,.1,3.],[.9,1e100,4.]])
    original=raw.copy();new,x=finite_moment_summaries(names,raw)
    assert new==['rhoAR1','log_stationary_AR_variance','trend[0]']
    np.testing.assert_array_equal(raw,original)
    np.testing.assert_array_equal(x[:,[0,2]],raw[:,[0,2]])
    np.testing.assert_allclose(x[:,1],np.log(raw[:,1]))
    assert 'INFINITE' in raw_variance_diagnostics(raw[:,1])['population_sd']
    for invalid in [0.,-1.,np.inf,np.nan]:
        bad=raw.copy();bad[0,1]=invalid
        with pytest.raises(ValueError,match='no floor'):finite_moment_summaries(names,bad)
