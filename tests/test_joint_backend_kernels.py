"""Four-cell joint density and lagged conditional moments; no chains."""
from dataclasses import replace
import numpy as np,pytest,torch
from test_enso_lag import fixture
from dlmmc.gpu_production import gpu_prepared
from dlmmc.production import initial_center
from dlmmc.spatial import SpatialTarget
from dlmmc.enso_lag import ENSOLagTarget
from dlmmc import conditional_trend_moments
from dlmmc.sampling import _threads

@pytest.mark.parametrize('backend',['torch','mps','cuda'])
@pytest.mark.parametrize('model',['vanilla_ar1','vanilla_ar2','dynregs_ar1','vanilla_ar1_noseasonal','vanilla_ar2_noseasonal'])
def test_coupled_density_and_lag_smoother(backend,model):
 if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
 if backend=='cuda' and not torch.cuda.is_available():pytest.skip('Native CUDA unavailable; no pass claim')
 torch.set_num_threads(1);p,sigma,lag,predictor=fixture();p=replace(p,model=model,priors={**p.priors,'sigma_reg_prior':[1e-4]*2})
 pg=p if backend=='torch' else gpu_prepared(p,backend)
 u=np.broadcast_to(initial_center(p),(3,4,len(p.parameter_names))).copy()+np.random.default_rng(1702).normal(0,.1,(3,4,len(p.parameter_names)))
 z=torch.tensor(u,dtype=pg.y.dtype,device=pg.y.device);cpu=torch.tensor(u)
 with _threads('numba',1,8):
  ref=SpatialTarget(p,sigma,backend='numba')(cpu.reshape(3,-1));got=SpatialTarget(pg,sigma,backend=backend)(z.reshape(3,-1)).cpu()
  np.testing.assert_allclose(got,ref,atol=.01 if backend!='torch' else 1e-8,rtol=0)
  ref=ENSOLagTarget(p,sigma,lag,predictor,backend='numba');target=ENSOLagTarget(pg,sigma,lag,predictor,backend=backend)
  lag_u=torch.tensor(np.random.default_rng(1703).normal(-1,.3,(3,4,1)))
  full=torch.cat((cpu,lag_u),-1);gpu=full.to(device=pg.y.device,dtype=pg.y.dtype)
  np.testing.assert_allclose(target(gpu.reshape(3,-1)).cpu(),ref(full.reshape(3,-1)),atol=.01 if backend!='torch' else 1e-8,rtol=0)
  before=target.local(gpu);changed=gpu.clone();changed[:,0,-1]+=.3
  torch.testing.assert_close(target.local(changed)[:,1:],before[:,1:],rtol=0,atol=0)
  xm=target.regressors(gpu);xr=ref.regressors(full)
  np.testing.assert_allclose(xm.cpu(),xr,atol=2e-6,rtol=2e-6)
  actual=conditional_trend_moments(pg,z,9,backend=backend,regressors=xm).cpu().numpy()
  expected=conditional_trend_moments(p,cpu,9,backend='numba',regressors=xr).numpy()
  assert np.max(abs(actual[...,0]-expected[...,0])/np.sqrt(expected[...,1]))<.05
  for j in (1,3):
   keep=expected[...,j]>0;assert np.max(abs(np.sqrt(actual[...,j][keep]/expected[...,j][keep])-1))<.05


def test_no_seasonal_prepared_policy_roundtrip_and_joint_guard(tmp_path):
 from test_backend_model_coverage import prepared
 from dlmmc.sampling import _save_prepared,_load_prepared
 from dlmmc.input_contract import validate_prepared_seasonality
 from dlmmc.spatial_sampling import sample_spatial
 from dlmmc.lag_sampling import sample_enso_lag
 p=prepared('vanilla_ar1_noseasonal',2)
 _save_prepared(tmp_path,p)
 restored=_load_prepared(tmp_path)
 assert restored.input_policy==p.input_policy
 validate_prepared_seasonality(restored)
 p=replace(p,input_policy=None)
 _,sigma,lag,predictor=fixture()
 with pytest.raises(ValueError,match='deseasonalised'):sample_spatial(p,sigma,output_dir=tmp_path/'bad')
 with pytest.raises(ValueError,match='deseasonalised'):sample_enso_lag(p,sigma,lag,predictor,output_dir=tmp_path/'bad')
 assert not (tmp_path/'bad').exists()

@pytest.mark.parametrize('backend',['torch','mps','cuda'])
def test_no_regressor_spatial_density(backend):
 if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
 if backend=='cuda' and not torch.cuda.is_available():pytest.skip('Native CUDA unavailable; no pass claim')
 p,sigma,_,_=fixture();p=replace(p,model='noregs_ar1',regressors=p.regressors[...,:0],regressor_names=[])
 pg=p if backend=='torch' else gpu_prepared(p,backend)
 u=np.broadcast_to(initial_center(p),(3,4,len(p.parameter_names))).copy()+np.random.default_rng(1705).normal(0,.1,(3,4,len(p.parameter_names)))
 with _threads('numba',1,8):
  ref=SpatialTarget(p,sigma,backend='numba')(torch.tensor(u).reshape(3,-1))
  got=SpatialTarget(pg,sigma,backend=backend)(torch.tensor(u,dtype=pg.y.dtype,device=pg.y.device).reshape(3,-1))
  np.testing.assert_allclose(got.cpu(),ref,atol=.01 if backend!='torch' else 1e-8,rtol=0)
