import numpy as np
import pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.transforms import parameter_names
from dlmmc.validation_summaries import scientific_summaries


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_physical_units_first_state_and_temporal_contrasts(model):
    spec=_MODEL_SPECS[model];k=0 if spec['regressors']=='none' else 1
    base=6 if spec['seasonal'] else 2;ar=base+k;d=ar+(2 if spec['ar2'] else 1)
    x=np.zeros((2,6,d));x[...,0]=3+2*np.arange(6);x[...,1]=2;x[...,ar]=.1
    if spec['seasonal']:x[...,2]=3;x[...,3]=4;x[...,4]=5;x[...,5]=12
    if k:x[...,base]=7+np.arange(6)
    names=parameter_names(model,k);theta={name:.01 for name in names};theta.update(rhoAR1=.3,rhoAR2=.2,sigma_AR=.1)
    p=np.tile([theta[name] for name in names],(2,1))
    labels,values,plabels,positive=scientific_summaries(model,p,x,nreg=k,scale=10.,offset=100.,time_indices=[0,5],contrasts=[[0,5]],amplitude_times=[0,5])
    r=dict(zip(labels,values.T))
    np.testing.assert_array_equal(r['trend[0]'],[130.,130.])
    np.testing.assert_array_equal(r['trend_change[0,5]'],[100.,100.])
    np.testing.assert_array_equal(r['average_slope'],[20.,20.])
    np.testing.assert_allclose(r['sigma_AR'],1.)
    assert positive.all() and plabels==['trend_positive[0,5]']
    if k:np.testing.assert_array_equal(r['beta[0,0]'],[70.,70.])
    if spec['seasonal']:
        np.testing.assert_array_equal(r['annual_amplitude[0]'],[50.,50.])
        np.testing.assert_array_equal(r['semiannual_amplitude[0]'],[130.,130.])
