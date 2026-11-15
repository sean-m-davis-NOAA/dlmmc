import numpy as np,torch,pytest
from dlmmc.api import _MODEL_SPECS
from dlmmc.model_reference import matrices,physical_logprior
from dlmmc.reference import kalman
from dlmmc.transforms import constrain,parameter_names
from dlmmc.tensor_target import target

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_transformed_density(model):
    k=0 if model=='noregs_ar1' else 2;n=12;rng=np.random.default_rng(20261104)
    names=parameter_names(model,k);u=np.array([-7. if name.startswith('sigma') else -.5 for name in names]);t,j=constrain(model,u,k)
    X=rng.normal(size=(n,k));y=rng.normal(size=n);v=np.full(n,.1);mask=np.ones(n,bool)
    F,G,Q,C=matrices(model,X,t)
    scales=dict(sigma_trend_prior=1e-4,sigma_seas_prior=.01,sigma_AR_prior=.5,sigma_reg_prior=[1e-4]*k)
    expected=kalman(y,v,mask,F,G,Q,C)[0].sum()+physical_logprior(model,t,scales)+j
    actual=target(*[torch.from_numpy(x) for x in (u,y,v,mask,X)],model)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_underflow_and_rounded_ar_endpoint_are_support_rejections(dtype):
    # Preserved mechanism from the instrumented pinned Stan warmup rejection:
    # sigma_AR underflowed to zero and rho rounded to one, creating 0/0.
    u=torch.tensor([-4.67899,-406.,-1000.,1000.],dtype=dtype)
    y=torch.zeros(3,dtype=dtype);v=torch.ones(3,dtype=dtype);mask=torch.ones(3,dtype=torch.bool);X=torch.empty((3,0),dtype=dtype)
    result=target(u,y,v,mask,X,'noregs_ar1')
    assert torch.isneginf(result)

def test_mixed_precision_is_not_silently_cast():
    u=torch.zeros(4,dtype=torch.float32);y=torch.zeros(3,dtype=torch.float32)
    with pytest.raises(ValueError,match='common'):
        target(u,y,torch.ones(3,dtype=torch.float64),torch.ones(3,dtype=torch.bool),torch.empty(3,0),'noregs_ar1')

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
def test_full_target_batch_permutation_chunking_isolation(model):
    rng=np.random.default_rng(20261104);n=12;k=0 if model=='noregs_ar1' else 2
    names=parameter_names(model,k)
    u=torch.tensor(np.broadcast_to([-7. if name.startswith('sigma') else -.5 for name in names],(7,5,len(names))).copy())
    y=torch.tensor(rng.normal(size=(5,n)));v=torch.tensor(rng.uniform(.05,.2,size=(5,n)))
    mask=torch.tensor(rng.random((5,n))>.2);y[~mask]=torch.nan;v[~mask]=torch.inf
    X=torch.tensor(rng.normal(size=(5,n,k)))
    baseline=target(u,y,v,mask,X,model)
    assert baseline.shape==(7,5)
    for g in range(5):
        scalar=target(u[:,g],y[g],v[g],mask[g],X[g],model)
        singleton=target(u[:,g:g+1],y[g:g+1],v[g:g+1],mask[g:g+1],X[g:g+1],model)
        torch.testing.assert_close(scalar,baseline[:,g],atol=1e-8,rtol=0)
        torch.testing.assert_close(singleton[:,0],scalar,atol=1e-8,rtol=0)
    for ids in [[4,1,3,0,2],[0,1],[2,3,4]]:
        result=target(u[:,ids],y[ids],v[ids],mask[ids],X[ids],model)
        torch.testing.assert_close(result,baseline[:,ids],atol=1e-8,rtol=0)
    y[0]=10
    changed=target(u,y,v,mask,X,model)
    torch.testing.assert_close(changed[:,1:],baseline[:,1:],rtol=0,atol=0)
