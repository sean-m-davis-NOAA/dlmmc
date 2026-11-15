from types import SimpleNamespace
import math
import numpy as np
import torch
from affine.torch_stream import save_state,load_state
from dlmmc.spatial import GaussianSpatialPrior
from dlmmc.spatial_blocks import initialize_blocks,advance_blocks,_joint

class ConditionalPrior:
    parameters=2;index=0;old_scale=1.
    prepared=SimpleNamespace(cells=2)
    def __init__(self):
        ell=1/math.sqrt(-2*math.log(.5))
        self.field=GaussianSpatialPrior((0,1),(0,0),ell,1,'altitude','km','degrees',(0,0),(1,1),'test').bind()
    def local(self,u):
        sigma=u[...,0].exp()
        return (.5*math.log(2/math.pi)-.5*sigma.square()+u[...,0]
                -.5*((u[...,1]-.4*sigma)/.7).square()-math.log(.7)-.5*math.log(2*math.pi))
    def __call__(self,u):return _joint(self,u,self.local(u.reshape(-1,2,2)))

def halves(w=24,seed=442):
    rng=np.random.default_rng(seed);center=np.tile([-.3,.3],2)
    return tuple(torch.tensor(center+.01*rng.normal(size=(w,4))) for _ in range(2))

def test_block_resume_cache_counters_and_rng_identity(tmp_path):
    t=ConditionalPrior();s=initialize_blocks(t,halves(),seed=448,warmup=3,thin=2,provenance='test')
    rng=torch.get_rng_state().clone();full,x,lp,ex,ix=advance_blocks(t,s,17);part,a,al,ae,ai=advance_blocks(t,s,8)
    save_state(tmp_path/'part.npz',part);rest=load_state(tmp_path/'part.npz',provenance='test');end,b,bl,be,bi=advance_blocks(t,rest,9)
    torch.testing.assert_close(torch.cat([a,b]),x,atol=0,rtol=0);torch.testing.assert_close(torch.cat([al,bl]),lp,atol=0,rtol=0)
    torch.testing.assert_close(torch.cat([ae,be]),ex,atol=0,rtol=0);np.testing.assert_array_equal(np.r_[ai,bi],ix)
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(end.rng,full.rng)
    for h in range(2):
        torch.testing.assert_close(t(full.halves[h]),full.logp[h],atol=1e-12,rtol=0)
        torch.testing.assert_close(t.local(full.halves[h].reshape(-1,2,2)),full.extras[h],atol=0,rtol=0)
    assert int(full.proposed.sum())==14*2*24*3
    assert int(full.warmup_proposed.sum())==3*2*24*3
    assert bool((full.accepted<=full.proposed).all())

def test_correlated_field_and_conditionally_local_nuisance_distribution():
    t=ConditionalPrior();samples=[]
    for run in range(4):
        s=initialize_blocks(t,halves(seed=772+run),seed=998+run,warmup=500,thin=1,provenance='prior-test')
        s,x,*_=advance_blocks(t,s,2000);samples.append(x.numpy().reshape(-1,2,2))
    x=np.concatenate(samples);sigma=np.exp(x[...,0]);residual=(x[...,1]-.4*sigma)/.7
    rng=np.random.default_rng(889);z=rng.multivariate_normal([0,0],[[1,.5],[.5,1]],size=200000);oracle=z[np.all(z>0,axis=1)]
    np.testing.assert_allclose(sigma.mean(0),oracle.mean(0),atol=.025,rtol=0)
    np.testing.assert_allclose(np.cov(sigma,rowvar=False),np.cov(oracle,rowvar=False),atol=.025,rtol=0)
    np.testing.assert_allclose(residual.mean(0),0,atol=.025,rtol=0)
    np.testing.assert_allclose(residual.var(0),1,atol=.04,rtol=0)
    assert abs(np.corrcoef(residual,rowvar=False)[0,1])<.03


def test_deadline_stops_before_transition_preserving_rng_and_resume(monkeypatch):
    import dlmmc.spatial_blocks as blocks
    t=ConditionalPrior();s=initialize_blocks(t,halves(),seed=448,warmup=3,thin=2,provenance='deadline')
    clock=iter([0.,0.,2.])
    monkeypatch.setattr(blocks.time,'time',lambda:next(clock))
    part,a,al,ae,ai=advance_blocks(t,s,10,deadline_unix=1.)
    assert part.iteration==2
    expected,*_=advance_blocks(t,s,2)
    assert torch.equal(part.rng,expected.rng)
    for x,y in zip(part.halves,expected.halves):assert torch.equal(x,y)
    monkeypatch.setattr(blocks.time,'time',lambda:2.)
    stopped,x,*_=advance_blocks(t,part,10,deadline_unix=1.)
    assert stopped.iteration==2 and len(x)==0 and torch.equal(stopped.rng,part.rng)
    resumed,b,bl,be,bi=advance_blocks(t,stopped,8)
    full,c,cl,ce,ci=advance_blocks(t,s,10)
    assert torch.equal(resumed.rng,full.rng)
    assert torch.equal(torch.cat([a,b]),c)
    assert torch.equal(torch.cat([al,bl]),cl)
    np.testing.assert_array_equal(np.r_[ai,bi],ci)
