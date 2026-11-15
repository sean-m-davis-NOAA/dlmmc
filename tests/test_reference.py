import numpy as np
import pytest
from dlmmc.reference import kalman, dense_condition, rts

@pytest.mark.parametrize('mask', [[True]*8, [False,True,True,False,False,True,True,False], [False]*8])
@pytest.mark.parametrize('q', [0., 1e-20, .01])
def test_dense_reference_and_timing(mask, q):
    rng = np.random.default_rng(20261104)
    G = np.array([[1.,1.],[0.,1.]])
    Q = np.diag([q, .003]); C = np.diag([.1,.02])
    F = np.tile([1.,0.], (8,1)); y = rng.normal(size=8)
    obs = np.array(mask); var = np.full(8,.1)
    y[~obs] = np.nan; var[~obs] = np.inf
    filt = kalman(y,var,obs,F,G,Q,C)
    lp, m, cov = dense_condition(y,var,obs,F,G,Q,C)
    np.testing.assert_allclose(filt[0].sum(),lp,rtol=0,atol=1e-8)
    sm, sc = rts(filt,G)
    np.testing.assert_allclose(sm,m.reshape(8,2),rtol=1e-8,atol=1e-10)
    for t in range(8):
        np.testing.assert_allclose(sc[t],cov[t*2:t*2+2,t*2:t*2+2],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(filt[4][0],G@C@G.T+Q)

def test_exact_measurement_not_missing():
    a=np.ones((1,1)); z=np.zeros((1,1))
    result=kalman(np.array([2.]),np.array([0.]),[True],a,a,z,a)
    assert result[1][0,0] == 2 and result[2][0,0,0] == 0
    with pytest.raises(ValueError,match='innovation'):
        kalman(np.array([2.]),np.array([0.]),[True],a,a,z,z)

def test_simulation_cross_time_covariance():
    from dlmmc.reference import simulate,simulation_smoother
    G=np.array([[.7]]);Q=np.array([[.2]]);C=np.array([[.3]])
    F=np.ones((4,1));var=np.full(4,.1);mask=np.array([True,False,True,True])
    y,_=simulate(F,G,Q,C,var,seed=20261104)
    f=kalman(y,var,mask,F,G,Q,C);_,m,c=dense_condition(y,var,mask,F,G,Q,C)
    x=simulation_smoother(f,G,seed=20261105,n_draws=30000).reshape(30000,4)
    # Prespecified 6-sigma simultaneous Monte Carlo checks, all 16 covariance entries.
    np.testing.assert_array_less(np.abs(x.mean(0)-m),6*np.sqrt(np.diag(c)/len(x)))
    mc=np.sqrt((c*c+np.outer(np.diag(c),np.diag(c)))/(len(x)-1))
    np.testing.assert_array_less(np.abs(np.cov(x,rowvar=False)-c),6*mc)
