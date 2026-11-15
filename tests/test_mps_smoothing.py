from pathlib import Path
from dataclasses import replace
import numpy as np,pytest,torch
from dlmmc import prepare_data
from dlmmc.examples import load_panels
from dlmmc.production import _cell
from dlmmc.structured_mps_sampling import gpu_prepared
from dlmmc.mps_smoothing import fixed_mps_trend_moments
from dlmmc.production_smoothing import fixed_grid_moments
from dlmmc.sampling import _threads

native=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native MPS unavailable')
def data():
 y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization');return prepare_data(y,x,uncertainty=s)

def close(actual,ref):
 a=actual.cpu().numpy();assert np.isfinite(a).all()
 assert np.max(abs(a[...,0]-ref[...,0])/np.sqrt(ref[...,1]))<.05
 for i in (1,3):
  keep=ref[...,i]>0
  assert np.max(abs(np.sqrt(a[...,i][keep]/ref[...,i][keep])-1))<.05
 np.testing.assert_array_equal(a[:,:,156,2:],0)

@native
def test_full_time_reference_layout_and_workspace():
 torch.set_num_threads(1);p=data();pg=gpu_prepared(p)
 u=np.broadcast_to(np.array([[-9.2,-4.6,-2.3,0.],[-11.,-6.,-3.,6.9],[-7.,-3.,-1.,-4.]]),(5,3,4)).transpose(1,0,2).copy()
 z=torch.tensor(u,dtype=torch.float32,device='mps')
 with _threads('numba',1,8):ref=fixed_grid_moments(p,u,156)
 full=fixed_mps_trend_moments(pg,z,156);close(full,ref)
 assert torch.equal(full,fixed_mps_trend_moments(pg,z,156,workspace_bytes=5*396*304*4))
 order=[4,2,0,3,1]
 perm=fixed_mps_trend_moments(_cell(pg,order),z[:,order],156)
 assert torch.equal(full,perm[:,np.argsort(order)])
 for ix in ([0],[0,1],[2,3,4]):
  assert torch.equal(full[:,ix],fixed_mps_trend_moments(_cell(pg,ix),z[:,ix],156))
 changed=replace(pg,y=pg.y.clone());changed.y[0]+=1
 assert torch.equal(full[:,1:],fixed_mps_trend_moments(changed,z,156)[:,1:])
 with pytest.raises(ValueError,match='Workspace'):fixed_mps_trend_moments(pg,z,156,workspace_bytes=1)

@native
def test_mask_and_dense_oracle():
 from dlmmc.reference import dense_condition
 from dlmmc.tensor_target import model_components
 torch.set_num_threads(1);p=data();p=_cell(p,[0]);n=5
 p=replace(p,y=p.y[:,:n].clone(),variance=p.variance[:,:n].clone(),observed=p.observed[:,:n].clone(),regressors=p.regressors[:,:n].clone(),time=p.time[:n])
 p.observed[:,[0,3]]=False;p.y[:,[0,3]]=torch.nan;p.variance[:,[0,3]]=torch.nan
 pg=gpu_prepared(p);u=torch.tensor([[[-6.,-4.,-2.,0.]]],dtype=torch.float32,device='mps')
 out=fixed_mps_trend_moments(pg,u,2).cpu().numpy()[0,0]
 F,G,Q,C,_,_=model_components(u,*pg.arguments,pg.model,S=pg.priors['S'],sampling=pg.sampling)
 def a(t):return t.cpu().numpy().astype(float)
 _,m,c=dense_condition(a(pg.y[0]),a(pg.variance[0]),pg.observed[0].cpu().numpy(),a(F[0,0]),a(G[0,0]),a(Q[0,0]),a(C[0,0]))
 m=m.reshape(n,12)[:,0]*p.scale[0]+p.offset[0];cv=c[::12,::12]*p.scale[0]**2
 np.testing.assert_allclose(out[:,0],m,rtol=5e-4,atol=5e-6)
 np.testing.assert_allclose(out[:,1],np.diag(cv),rtol=5e-4,atol=5e-6)
 ref=np.diag(cv)+cv[2,2]-2*cv[:,2]
 np.testing.assert_allclose(out[:,3],ref,rtol=5e-4,atol=5e-6)
 pg=replace(pg,variance=torch.full_like(pg.variance,-1e10))
 with pytest.raises(FloatingPointError,match='no floor'):fixed_mps_trend_moments(pg,u,2)
