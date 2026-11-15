import numpy as np
from dlmmc.diagnostics import ensemble_diagnostics,split_rhat,autocorrelation_time

def test_duplicated_walkers_do_not_inflate_ess():
    rng=np.random.default_rng(42);x=rng.normal(size=(4,4096,1,2))
    a=ensemble_diagnostics(x);b=ensemble_diagnostics(np.repeat(x,32,axis=2))
    np.testing.assert_allclose(a['mean_ess_estimate'],b['mean_ess_estimate'],rtol=.05)
    assert max(b['mean_ess_estimate'])<22000

def test_ar1_iat_and_short_run_reliability():
    rng=np.random.default_rng(13);x=np.zeros(100000)
    for i in range(1,len(x)):x[i]=.8*x[i-1]+rng.normal()
    assert 7<autocorrelation_time(x)<11
    y=np.stack([x[i*100:i*100+100] for i in range(4)])[:,:,None,None]
    assert not all(ensemble_diagnostics(y)['iat_length_reliable'])

def test_stuck_separate_ensembles_not_convergence():
    rng=np.random.default_rng(9);x=rng.normal(size=(4,1000,2))+np.arange(4)[:,None,None]*5
    assert np.min(split_rhat(x))>3

def test_folded_rank_rhat_detects_width_disagreement():
    from dlmmc.diagnostics import rank_split_rhat
    rng=np.random.default_rng(42);x=rng.normal(size=(4,10000,1))
    assert rank_split_rhat(x)[0]<1.01
    x[0]*=10
    assert rank_split_rhat(x)[0]>1.1
