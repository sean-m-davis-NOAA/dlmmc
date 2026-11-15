import numpy as np,torch,pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices
from dlmmc.reference import dense_condition
from dlmmc.precision_prototype import filter_tensor
from dlmmc.tensor_smoothing import conditional_moments,full_conditional_covariance

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
@pytest.mark.parametrize('zero',[False,True])
def test_complete_joint_conditional_reference(model,zero):
    rng=np.random.default_rng(20261104);n=8;k=0 if model=='noregs_ar1' else 2
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta,fixed_variance=0 if zero else None)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.array([False,True,True,False,False,True,True,True])
    _,expected_m,expected_C=dense_condition(y,v,mask,F,G,Q,C)
    f=filter_tensor(*[torch.from_numpy(a) for a in (y,v,mask,F,G,Q,C)])
    m,c,j=conditional_moments(f,torch.from_numpy(G),torch.from_numpy(Q))
    np.testing.assert_allclose(m.numpy().reshape(-1),expected_m,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(full_conditional_covariance(c,j),expected_C,rtol=1e-8,atol=1e-10)

def test_preserved_float32_covariance_cancellation():
    from pathlib import Path
    a=np.load(Path(__file__).parent/'fixtures/smoother_cancellation_noregs.npz')
    x=[torch.from_numpy(a[k]) for k in ['y','variance','observed','F','G','Q','C']]
    f=filter_tensor(*x);_,c,_=conditional_moments(f,x[4],x[5])
    cc=c.numpy().astype(float);sym=(cc+cc.swapaxes(-1,-2))/2
    tolerance=64*np.finfo(np.float32).eps*cc.shape[-1]*np.linalg.norm(cc,ord=2,axis=(-2,-1)).max()
    assert np.linalg.eigvalsh(sym).min()>=-tolerance
    assert c.dtype==torch.float32

def test_contrast_against_dense_cross_time_conditioning():
    from dlmmc.tensor_smoothing import contrast_moments
    rng=np.random.default_rng(20261104);n=8
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,rhoAR1=.3)
    F,G,Q,C=matrices('noregs_ar1',np.empty((n,0)),theta)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.ones(n,bool)
    _,mean,cov=dense_condition(y,v,mask,F,G,Q,C)
    f=filter_tensor(*[torch.from_numpy(a) for a in (y,v,mask,F,G,Q,C)])
    a=rng.normal(size=len(G));b=rng.normal(size=len(G))
    projection=np.zeros(n*len(G));projection[len(G):2*len(G)]=a;projection[6*len(G):7*len(G)]=b
    m,c=contrast_moments(f,torch.from_numpy(G),torch.from_numpy(Q),1,6,torch.from_numpy(a),torch.from_numpy(b))
    np.testing.assert_allclose(m,projection@mean,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(c,projection@cov@projection,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('stable',[False,True])
def test_conditional_sign_probability_against_dense_oracle(stable):
    from scipy.special import ndtr
    from dlmmc.conditional_statistics import positive_trend_probabilities
    if stable:
        from dlmmc.conditional_statistics_stable import stable_positive_trend_probabilities as positive_trend_probabilities
    rng=np.random.default_rng(23);n=8
    theta=dict(sigma_trend=.001,sigma_AR=.1,sigma_seas=.01,rhoAR1=.3)
    F,G,Q,C=matrices('noregs_ar1',np.empty((n,0)),theta)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.ones(n,bool)
    _,mean,cov=dense_condition(y,v,mask,F,G,Q,C)
    f=filter_tensor(*[torch.from_numpy(a) for a in (y,v,mask,F,G,Q,C)])
    actual=positive_trend_probabilities(f,torch.from_numpy(G),torch.from_numpy(Q),[(0,7),(1,6)])
    expected=[]
    for a,b in [(0,7),(1,6)]:
        weights=np.zeros(n*len(G));weights[a*len(G)]=-1;weights[b*len(G)]=1
        expected.append(ndtr((weights@mean)/np.sqrt(weights@cov@weights)))
    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_simulation_draws_against_independent_joint_gaussian(model):
    from dlmmc.tensor_smoothing import conditional_draw
    rng=np.random.default_rng(20261112);n=5;k=0 if model=='noregs_ar1' else 1
    theta=dict(sigma_trend=.01,sigma_AR=.1,sigma_seas=.01,sigma_reg=[.001]*k,rhoAR1=.3,rhoAR2=.2)
    F,G,Q,C=matrices(model,rng.normal(size=(n,k)),theta,fixed_variance=0.)
    y=rng.normal(size=n);v=np.full(n,.1);mask=np.array([False,True,True,False,True])
    _,mean,cov=dense_condition(y,v,mask,F,G,Q,C)
    count=8192
    tensors=[torch.from_numpy(x) for x in (y,v,mask,F,G,Q,C)]
    tensors[3]=tensors[3].expand(count,*F.shape)
    before=torch.get_rng_state().clone()
    draws=conditional_draw(*tensors,generator=torch.Generator().manual_seed(20261112)).numpy().reshape(count,-1)
    assert torch.equal(before,torch.get_rng_state())
    # Six standard errors, using exact Gaussian moments of the oracle; no
    # arbitrary data-dependent floors and no posterior qualification claim.
    var=np.diag(cov)
    assert np.all(abs(draws.mean(0)-mean)<6*np.sqrt(var/count))
    expected_se=np.sqrt((cov**2+var[:,None]*var[None,:])/(count-1))
    assert np.all(abs(np.cov(draws,rowvar=False)-cov)<6*expected_se)
