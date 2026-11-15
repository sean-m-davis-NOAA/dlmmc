import numpy as np
import pytest
from dlmmc.validation_bootstrap import block_counts,resampled_quantiles,bootstrap_metrics,bootstrap_probabilities,discrepancy_intervals


def test_weighted_quantiles_match_explicit_resampling_with_ties():
    rng=np.random.default_rng(1);blocks=rng.integers(-4,8,size=(20,7)).astype(float)
    counts=block_counts(4,5,100,rng)
    got=resampled_quantiles(blocks,counts,(0,.05,.5,.95,1))
    for i in range(len(counts)):
        actual=np.repeat(blocks,counts[i],axis=0).ravel()
        np.testing.assert_allclose(got[i],np.quantile(actual,[0,.05,.5,.95,1]),atol=1e-14)


def test_all_statistics_match_raw_block_bootstrap():
    rng=np.random.default_rng(2);x=rng.normal(size=(4,20,3,2))+100
    counts=block_counts(4,5,100,rng);point,boot=bootstrap_metrics(x,counts,4)
    for i in range(5):
        actual=np.repeat(x.reshape(20,12,2),counts[i],axis=0).reshape(-1,2)
        expected=np.vstack([actual.mean(0),actual.std(0,ddof=1),np.quantile(actual,[.05,.5,.95],axis=0)])
        np.testing.assert_allclose(boot[i],expected,atol=1e-12)
    np.testing.assert_allclose(point[0],x.mean((0,1,2)))


def test_walkers_are_never_independent_bootstrap_units():
    rng=np.random.default_rng(3);x=rng.normal(size=(4,100,1,1));counts=block_counts(4,10,500,rng)
    _,single=bootstrap_metrics(x,counts,10);_,duplicated=bootstrap_metrics(np.repeat(x,16,axis=2),counts,10)
    np.testing.assert_allclose(single[:,0],duplicated[:,0],atol=1e-14)


def test_serial_dependence_widens_bootstrap_mean_uncertainty():
    rng=np.random.default_rng(4);innov=rng.normal(size=(4,2000,1,1));x=innov.copy()
    for t in range(1,2000):x[:,t]=.95*x[:,t-1]+innov[:,t]
    _,b=bootstrap_metrics(x,block_counts(4,20,500,rng),100)
    naive=x.std()/np.sqrt(x.size)
    assert b[:,0,0].std()>3*naive


def test_sign_probabilities_match_explicit_bootstrap():
    rng=np.random.default_rng(5);x=rng.integers(0,2,size=(4,20,3,2));counts=block_counts(4,5,100,rng)
    _,boot=bootstrap_probabilities(x,counts,4)
    for i in range(5):np.testing.assert_allclose(boot[i],np.repeat(x.reshape(20,12,2),counts[i],axis=0).mean((0,1)))


def test_no_suffix_dropped_and_run_weights_preserved():
    x=np.ones((4,21,2,1));counts=block_counts(4,5,100,np.random.default_rng(6))
    with pytest.raises(ValueError,match='suffix'):bootstrap_metrics(x,counts,4)
    counts[:,0]+=1;counts[:,-1]-=1
    with pytest.raises(ValueError,match='weight'):bootstrap_metrics(x[:,:20],counts,4)


def test_detects_known_shift_and_identical_estimates_do_not_fail():
    rng=np.random.default_rng(7);a=rng.normal(0,.01,(4999,3));b=rng.normal(0,.01,(4999,3))
    result=discrepancy_intervals(np.zeros(3),np.zeros(3),a,b)
    assert not any(result['simultaneous_detected'])
    result=discrepancy_intervals(np.ones(3),np.zeros(3),a+1,b)
    assert all(result['simultaneous_detected'])


def test_degenerate_bootstrap_cannot_claim_no_bias():
    with pytest.raises(ValueError,match='Degenerate'):discrepancy_intervals([0],[0],np.zeros((100,1)),np.zeros((100,1)))


def test_paired_blocks_preserve_cross_dtype_covariance():
    rng=np.random.default_rng(8);shared=rng.normal(size=(4,200,1,1));other=shared+.1*rng.normal(size=shared.shape)
    counts=block_counts(4,10,500,rng);independent=block_counts(4,10,500,rng)
    _,a=bootstrap_metrics(shared,counts,20);_,paired=bootstrap_metrics(other,counts,20);_,unpaired=bootstrap_metrics(other,independent,20)
    assert (paired[:,0]-a[:,0]).std()<.2*(unpaired[:,0]-a[:,0]).std()
