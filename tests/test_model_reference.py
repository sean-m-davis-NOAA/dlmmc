import numpy as np
import pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices,physical_logprior
from dlmmc.reference import kalman,dense_condition,rts

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_all_models_dense(model):
    rng=np.random.default_rng(20261104)
    n=12; k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta)
    y=rng.normal(size=n); var=np.full(n,.1); obs=np.ones(n,dtype=bool);obs[[0,4,5]]=False
    f=kalman(y,var,obs,F,G,Q,C); lp,m,c=dense_condition(y,var,obs,F,G,Q,C)
    np.testing.assert_allclose(f[0].sum(),lp,atol=1e-8,rtol=0)
    sm,sc=rts(f,G)
    np.testing.assert_allclose(sm,m.reshape(n,-1),rtol=1e-8,atol=1e-10)
    for t in range(n):
        d=len(G)
        np.testing.assert_allclose(sc[t],c[t*d:(t+1)*d,t*d:(t+1)*d],rtol=1e-8,atol=1e-10)

def test_ar2_conditional_normalization():
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,rhoAR1=.1,rhoAR2=.2)
    scales=dict(sigma_trend_prior=.001,sigma_AR_prior=.1,sigma_seas_prior=.01)
    x=physical_logprior('vanilla_ar2',theta,scales)
    theta['rhoAR1']=.4
    y=physical_logprior('vanilla_ar2',theta,scales)
    np.testing.assert_allclose(y-x,np.log(.9/.6),rtol=1e-12)
