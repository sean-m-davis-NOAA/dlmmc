import numpy as np
import pytest
from dlmmc.legacy_summaries import vanilla_generated_summaries
from dlmmc.validation_summaries import scientific_summaries


def fixture():
    rng=np.random.default_rng(837);x=rng.normal(size=(31,12,9));p=np.tile([.001,.002,.1,.3],(31,1));scale=4.;offset=2.
    opts=dict(time_indices=[0,4,11],contrasts=[(0,4),(4,11),(0,11)],amplitude_times=[0,11])
    args=dict(trend=x[...,0]*scale+offset,slope=x[...,1]*scale,seasonal_components=x[...,2:6]*scale,
              ar=x[...,-1]*scale,beta=x[:,0,6:8]*scale,scale=scale,offset=offset,**opts)
    return p,x,args,opts


def test_original_projection_preserves_all_queried_summaries():
    p,x,args,opts=fixture()
    expected=scientific_summaries('vanilla_ar1',p,x,nreg=2,scale=4.,offset=2.,**opts)
    actual=vanilla_generated_summaries(p,**args)
    assert actual[0]==expected[0] and actual[2]==expected[2]
    np.testing.assert_allclose(actual[1],expected[1],rtol=0,atol=2e-14)
    np.testing.assert_array_equal(actual[3],expected[3])
    # Later coefficient trajectories are not exported/queried in static GQ.
    altered=x.copy();altered[:,1:,6:8]+=1e6
    other=scientific_summaries('vanilla_ar1',p,altered,nreg=2,scale=4.,offset=2.,**opts)
    np.testing.assert_array_equal(expected[1],other[1])


def test_original_projection_rejects_missing_or_nonfinite_exports():
    p,x,args,opts=fixture();args['beta']=args['beta'][:-1]
    with pytest.raises(ValueError,match='dimensions'):vanilla_generated_summaries(p,**args)
    p,x,args,opts=fixture();args['trend'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):vanilla_generated_summaries(p,**args)
