"""Fixed-parameter numerical coverage; no posterior qualification or long chains."""
from pathlib import Path
from dataclasses import replace
import json,os
import numpy as np
import pytest,torch
from dlmmc import prepare_data,deseasonalize_harmonics,conditional_trend_moments
from dlmmc.examples import load_panels
from dlmmc.production import _cell,initial_center
from dlmmc.gpu_production import gpu_prepared
from dlmmc.sampling import make_target,_threads
from dlmmc.api import _MODEL_SPECS

CASES=[('vanilla_ar1',1),('vanilla_ar1',3),('vanilla_ar1',5),('vanilla_ar2',1),('vanilla_ar2',5),('noregs_ar1',0),('dynregs_ar1',1),('dynregs_ar1',5),('vanilla_ar1_noseasonal',2),('vanilla_ar2_noseasonal',2)]

def inputs(model,k,n=None):
 y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
 if not _MODEL_SPECS[model]['seasonal']:y=deseasonalize_harmonics(y).deseasonalized
 if n:y=y.isel(time=slice(0,n));s=s.isel(time=slice(0,n));x=x.isel(time=slice(0,n))
 return y,s,x.isel(regressor=slice(0,k)) if k else None


def prepared(model,k):
 y,s,x=inputs(model,k);p=prepare_data(y,x,uncertainty=s,model=model,priors={'sigma_trend_prior':.00015})
 p=replace(p,y=p.y.clone(),variance=p.variance.clone(),observed=p.observed.clone())
 p.observed[:,[0,17,-1]]=False;p.y[:,[0,17,-1]]=torch.nan;p.variance[:,[0,17,-1]]=torch.nan
 return p


def check_case(model,k,backend):
 torch.set_num_threads(1);p=prepared(model,k);anchor=156
 u=np.broadcast_to(initial_center(p),(3,5,len(p.parameter_names))).copy()+np.random.default_rng(1701).normal(0,.15,(3,5,len(p.parameter_names)))
 if backend=='mps':pg=gpu_prepared(p,'mps')
 elif backend=='cuda':pg=gpu_prepared(p,'cuda')
 elif backend=='torch32':pg=replace(p,**{n:getattr(p,n).to(dtype=torch.bool if n=='observed' else torch.float32) for n in ['y','variance','observed','regressors']})
 else:pg=p
 device_backend='torch' if backend in ('torch','torch32') else backend
 z=torch.tensor(u,dtype=pg.y.dtype,device=pg.y.device)
 directory=os.environ.get('DLMMC_NUMERICAL_EVIDENCE')
 if directory:
  folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
  np.savez(folder/f'{model}-{k}-{backend}-fixture.npz',u=u,prepared_checksum=p.checksum,panels=p.cell_coordinates['panel'].astype(str))
 with _threads('numba',1,8):
  reference=make_target(p,'numba')(torch.tensor(u));moments=conditional_trend_moments(p,torch.tensor(u),anchor)
  target=make_target(pg,device_backend);value=target(z)
  actual=conditional_trend_moments(pg,z,anchor,backend=device_backend)
  a=actual.cpu().numpy();r=moments.numpy();valid=r[...,3]>0
  metrics=dict(model=model,nreg=k,backend=backend,cells=5,times=len(p.time),dtype=str(z.dtype),device=str(z.device),likelihood_max_abs=float(abs(value.cpu()-reference).max()),mean_max_reference_sd=float(np.max(abs(a[...,0]-r[...,0])/np.sqrt(r[...,1]))),sd_max_relative=float(np.max(abs(np.sqrt(a[...,1]/r[...,1])-1))),change_mean_max_reference_sd=float(np.max(abs(a[...,2][valid]-r[...,2][valid])/np.sqrt(r[...,3][valid]))),change_sd_max_relative=float(np.max(abs(np.sqrt(a[...,3][valid]/r[...,3][valid])-1))))
  metrics['per_panel']=[]
  for cell,panel in enumerate(p.cell_coordinates['panel']):
   rv=r[:,cell];av=a[:,cell];keep=rv[...,3]>0
   metrics['per_panel'].append(dict(panel=str(panel),likelihood_max_abs=float(abs(value.cpu()[:,cell]-reference[:,cell]).max()),mean_max_reference_sd=float(np.max(abs(av[...,0]-rv[...,0])/np.sqrt(rv[...,1]))),sd_max_relative=float(np.max(abs(np.sqrt(av[...,1]/rv[...,1])-1))),change_mean_max_reference_sd=float(np.max(abs(av[...,2][keep]-rv[...,2][keep])/np.sqrt(rv[...,3][keep]))),change_sd_max_relative=float(np.max(abs(np.sqrt(av[...,3][keep]/rv[...,3][keep])-1)))))
  if z.dtype==torch.float32:
   rounded=replace(p,**{name:getattr(pg,name).cpu().to(dtype=torch.bool if name=='observed' else torch.float64) for name in ('y','variance','observed','regressors')})
   rounded_ll=make_target(rounded,'numba')(z.cpu().double())
   metrics['input_parameter_rounding_max_log_density']=float(abs(rounded_ll-reference).max())
   metrics['arithmetic_max_log_density_at_rounded_inputs']=float(abs(value.cpu()-rounded_ll).max())
  if directory:(folder/f'{model}-{k}-{backend}-numerics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  assert np.isfinite(a).all();assert metrics['likelihood_max_abs']<(.01 if z.dtype==torch.float32 else 1e-7)
  if z.dtype==torch.float64:np.testing.assert_allclose(a,r,rtol=2e-8,atol=2e-10)
  else:
   assert max(metrics[q] for q in ['mean_max_reference_sd','sd_max_relative','change_mean_max_reference_sd','change_sd_max_relative'])<.05
  # Every individual panel, sizeone, full, permutation and chunking.
  for ix in ([0],[1],[2],[3],[4],[4,2,0,3,1],[0,1],[2,3,4]):
   pp=_cell(pg,ix);uu=z[:,ix];local=make_target(pp,device_backend)(uu)
   torch.testing.assert_close(local,value[:,ix],rtol=0,atol=1e-9 if z.dtype==torch.float64 else 0)
   small=conditional_trend_moments(pp,uu,anchor,backend=device_backend)
   torch.testing.assert_close(small,actual[:,ix],rtol=1e-9 if z.dtype==torch.float64 else 0,atol=1e-10 if z.dtype==torch.float64 else 0)
  changed=replace(pg,y=pg.y.clone());changed.y[0,1:]+=1
  torch.testing.assert_close(make_target(changed,device_backend)(z)[:,1:],value[:,1:],rtol=0,atol=0)
  torch.testing.assert_close(conditional_trend_moments(changed,z,anchor,backend=device_backend)[:,1:],actual[:,1:],rtol=0,atol=0)
  torch.testing.assert_close(actual[:,:,anchor,2:],torch.zeros_like(actual[:,:,anchor,2:]),rtol=0,atol=0)
 return metrics

@pytest.mark.parametrize('model,k',CASES)
@pytest.mark.parametrize('backend',['numba','torch','torch32','mps','cuda'])
def test_all_model_fixed_likelihood_and_smoother(model,k,backend,record_property):
 if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
 if backend=='cuda' and not torch.cuda.is_available():pytest.skip('Native CUDA unavailable; not a pass')
 record_property('numerics',json.dumps(check_case(model,k,backend)))


def test_short_profile_and_no_seasonal_guards(tmp_path):
 from dlmmc import PRODUCTION_PROFILE,QUALIFICATION_PROFILE,fit_production
 assert (PRODUCTION_PROFILE.half_walkers*2,PRODUCTION_PROFILE.warmup,PRODUCTION_PROFILE.retained_sweeps,PRODUCTION_PROFILE.output_draws_per_sweep)==(128,128,256,8)
 assert QUALIFICATION_PROFILE.runs==4
 y,s,x=inputs('vanilla_ar1',2,24)
 with pytest.raises(ValueError,match='deseasonalised'):fit_production(y,x,uncertainty=s,model='vanilla_ar1_noseasonal',output_dir=tmp_path/'invalid')
 assert not (tmp_path/'invalid').exists()

@pytest.mark.parametrize('backend',['numba','cpu','mps'])
@pytest.mark.parametrize('model,k',[('dynregs_ar1',2),('vanilla_ar2_noseasonal',2),('noregs_ar1',0)])
def test_production_dimensions_and_output(tmp_path,model,k,backend):
 if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
 from dlmmc import PRODUCTION_PROFILE,fit_production,summarize_production_trends
 from dlmmc.production import production_parameter_cache
 y,s,x=inputs(model,k,24);profile=replace(PRODUCTION_PROFILE,half_walkers=8,warmup=2,retained_sweeps=8,chunk_sweeps=2)
 fit=fit_production(y.isel(panel=[0]),None if x is None else x.isel(panel=[0]),uncertainty=s.isel(panel=[0]),model=model,profile=profile,backend=backend,output_dir=tmp_path/'fit',wall_seconds=120)
 assert production_parameter_cache(fit).shape[-1]==len(fit.prepared.parameter_names)
 out=summarize_production_trends(fit,output_dir=tmp_path/'out',anchor_month='1986-01')
 assert out['full_batch_shape']==[64,1]
