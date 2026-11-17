import numpy as np
import pytest
import torch
import numba
from dlmmc.api import _MODEL_SPECS
from dlmmc.numba_prototype import target_numba
from dlmmc.tensor_target import target
from dlmmc.transforms import parameter_names

numba.set_num_threads(2)

@pytest.mark.parametrize('model',list(_MODEL_SPECS))
@pytest.mark.parametrize('algorithm',['dense','structured'])
def test_compiled_callback_density_and_five_cell_isolation(model,algorithm):
    rng=np.random.default_rng(20261115);n=12;k=0 if model=='noregs_ar1' else 2
    names=parameter_names(model,k)
    u=np.array([np.log(1e-4) if name.startswith('sigma_trend') or name.startswith('sigma_reg') else np.log(.01) if name=='sigma_seas' else np.log(.1) if name=='sigma_AR' else 0. for name in names])
    u=np.broadcast_to(u,(4,5,len(names))).copy();u+=rng.normal(size=u.shape)*.03
    y=rng.normal(size=(5,n));v=np.full_like(y,.1);mask=rng.random(y.shape)>.2;X=rng.normal(size=(5,n,k))
    mask[0]=False;y[~mask]=np.nan;v[~mask]=np.inf
    args=[torch.from_numpy(x) for x in [u,y,v,mask,X]]
    expected=target(*args,model);actual=target_numba(*args,model,algorithm=algorithm)
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-8)
    for indices in [[0],[4,2,0,3,1],[0,1],[2,3,4]]:
        subset=[args[0][:,indices],*[a[indices] for a in args[1:]]]
        torch.testing.assert_close(target_numba(*subset,model,algorithm=algorithm),actual[:,indices],rtol=0,atol=1e-8)
    args[1][0]=10.;args[2][0]=20.;args[3][0]=True
    changed=target_numba(*args,model,algorithm=algorithm)
    torch.testing.assert_close(changed[:,1:],actual[:,1:],rtol=0,atol=0)


def test_rejects_implicit_float32_candidate():
    x=torch.zeros(2,dtype=torch.float32)
    with pytest.raises(ValueError,match='explicit CPU float64'):
        target_numba(x,x,x,x>0,x[:,None],'noregs_ar1')


def test_research_float32_structured_dtype_and_layouts():
    """Explicit opt-in; one/5/permuted/chunked cells, masks and cross-talk."""
    rng=np.random.default_rng(2026111701)
    u=np.tile(np.array([-9.,-5.,-2.,0.]),(4,5,1))
    u+=rng.normal(0,.03,u.shape)
    y=rng.normal(size=(5,24));v=np.full_like(y,.1)
    mask=rng.random(y.shape)>.2;X=rng.normal(size=(5,24,2))
    y[~mask]=np.nan;v[~mask]=np.inf
    args64=[torch.from_numpy(x) for x in (u,y,v,mask,X)]
    args32=[a if a.dtype==torch.bool else a.float() for a in args64]
    def evaluate(args):
        return target_numba(*args,'vanilla_ar1',algorithm='structured',research_float32=True)
    actual=evaluate(args32)
    assert actual.dtype==torch.float32
    expected=evaluate(args64)
    torch.testing.assert_close(actual.double(),expected,rtol=0,atol=.01)
    for indices in [[0],[4,2,0,3,1],[0,1],[2,3,4]]:
        subset=[args32[0][:,indices],*[a[indices] for a in args32[1:]]]
        torch.testing.assert_close(evaluate(subset),actual[:,indices],rtol=0,atol=0)
    args32[1][0]=10.;args32[2][0]=20.;args32[3][0]=True
    torch.testing.assert_close(evaluate(args32)[:,1:],actual[:,1:],rtol=0,atol=0)
    from dlmmc.numba_prototype import _one_structured
    from dlmmc.tensor_target import model_components
    F,G,Q,C,_,_=model_components(*args32,'vanilla_ar1')
    scalar_args=[args32[1][0],args32[2][0],args32[3][0],F[0,0],G[0,0],Q[0,0],C[0,0]]
    signature=tuple(numba.typeof(a.numpy()) for a in scalar_args)
    # Loading a cached batch need not populate the callee dispatcher's registry.
    _one_structured.compile(signature)
    result_type=_one_structured.overloads[signature].signature.return_type
    assert str(result_type[0])=='float32'  # including scalar log-density accumulation
