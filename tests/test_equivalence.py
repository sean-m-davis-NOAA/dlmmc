import numpy as np
import pytest
from dlmmc.equivalence import compare_scalar_draws

def test_known_equal_and_shifted_gaussians():
    rng=np.random.default_rng(20261104);a=rng.normal(size=(4,4096,16,2));b=rng.normal(size=a.shape)
    kw=dict(candidate_block_steps=512,reference_block_steps=512,family_size=10,diagnostics_validated=True)
    assert compare_scalar_draws(a,b,**kw)['status']=='PASS'
    assert compare_scalar_draws(a+.2,b,**kw)['status']=='FAIL'
    assert compare_scalar_draws(a,b,**{**kw,'diagnostics_validated':False})['status']=='INCONCLUSIVE'

def test_duplicate_walkers_cannot_inflate_mean_precision():
    rng=np.random.default_rng(13);a=rng.normal(size=(4,4096,1,1));b=rng.normal(size=a.shape)
    kw=dict(candidate_block_steps=512,reference_block_steps=512,family_size=5)
    one=compare_scalar_draws(a,b,**kw);many=compare_scalar_draws(np.repeat(a,32,2),np.repeat(b,32,2),**kw)
    np.testing.assert_allclose(one['mcse'][0],many['mcse'][0],atol=1e-12,rtol=0)

def test_nonfinite_or_remainder_is_not_discarded():
    x=np.ones((4,100,1,1));x[0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):compare_scalar_draws(x,x,candidate_block_steps=20,reference_block_steps=20,family_size=5)


def test_sign_probability_precision_and_constant_tail_guard():
    from dlmmc.equivalence import compare_positive_probabilities
    rng=np.random.default_rng(20261112);a=rng.normal(size=(4,4096,32,1));b=rng.normal(size=a.shape)
    kw=dict(candidate_block_steps=512,reference_block_steps=512,family_size=1,diagnostics_validated=True)
    assert compare_positive_probabilities(a,b,**kw)['status']=='PASS'
    assert compare_positive_probabilities(a+.1,b,**kw)['status']=='FAIL'
    constant=compare_positive_probabilities(np.ones_like(a),np.ones_like(b),**kw)
    assert constant['status']=='INCONCLUSIVE'
    assert constant['interval_lower']==[-1.] and constant['interval_upper']==[1.]
    small=compare_positive_probabilities(a[:,:,:1],b[:,:,:1],**kw)
    copied=compare_positive_probabilities(np.repeat(a[:,:,:1],32,axis=2),np.repeat(b[:,:,:1],32,axis=2),**kw)
    np.testing.assert_allclose(small['mcse'],copied['mcse'],atol=1e-12,rtol=0)


def test_rao_blackwell_values_and_sampled_indicators_target_same_probability():
    from dlmmc.equivalence import compare_positive_probabilities
    rng=np.random.default_rng(20261114);shape=(4,4096,32,1)
    indicators=rng.random(shape)<.5;conditional=rng.random(shape)
    kw=dict(candidate_block_steps=512,reference_block_steps=512,family_size=1,diagnostics_validated=True,conditional_probabilities=True)
    assert compare_positive_probabilities(indicators,conditional,**kw)['status']=='PASS'
    assert compare_positive_probabilities(indicators,.2+.8*conditional,**kw)['status']=='FAIL'
    with pytest.raises(ValueError,match='lie in'):
        compare_positive_probabilities(indicators,conditional+1,**kw)
