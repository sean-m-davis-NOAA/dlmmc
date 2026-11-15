from pathlib import Path
from dataclasses import replace
import numpy as np,pytest,torch
from dlmmc import fit_production,load_fit,summarize_production_trends,PRODUCTION_PROFILE,prepare_data
from dlmmc.examples import load_panels
from dlmmc.gpu_production import resume_gpu_production,cpu64_reconstruction_view
from dlmmc.gpu_smoothing import select_draws,fixed_cuda_trend_moments
from dlmmc.production_smoothing import fixed_grid_moments
from dlmmc.sampling import _threads

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native MPS unavailable')
def test_public_gpu_sampling_summary_and_resume(tmp_path):
 torch.set_num_threads(1)
 y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
 y=y.isel(time=slice(0,24));s=s.isel(time=slice(0,24));x=x.isel(time=slice(0,24))
 profile=replace(PRODUCTION_PROFILE,half_walkers=8,warmup=2,retained_sweeps=8,chunk_sweeps=2)
 fit=fit_production(y,x,uncertainty=s,backend='mps',profile=profile,output_dir=tmp_path/'fit',wall_seconds=120)
 assert fit.prepared.y.dtype==torch.float32 and fit.prepared.y.device.type=='mps'
 assert not fit.config['qualification_eligible']
 same=resume_gpu_production(fit.path,wall_seconds=120)
 assert torch.equal(same.prepared.y,fit.prepared.y)
 out=summarize_production_trends(fit,output_dir=tmp_path/'out',anchor_month='1986-01',seed=1601,selected_walkers_per_sweep=4)
 assert torch.device(out['device']).type=='mps' and out['dtype']=='float32' and not out['panel_chunking']
 selected,_,_,_=select_draws(fit,4,1601)
 with _threads('numba',1,8):ref=fixed_grid_moments(cpu64_reconstruction_view(fit).prepared,selected.astype(float),12)
 for g in range(5):
  with np.load(tmp_path/'out'/f'cell-{g:04d}.npz') as z:
   m=ref[:,g,:,0].mean(0);sd=np.sqrt((ref[:,g,:,1]+(ref[:,g,:,0]-m)**2).mean(0))
   assert np.max(abs(z['mean']-m)/sd)<.05
   assert np.max(abs(z['sd']/sd-1))<.05
   np.testing.assert_array_equal(z['change_mean'][12],0)
   np.testing.assert_array_equal(z['change_sd'][12],0)
 # Eight selections/sweep must also work when only one draw fits in workspace.
 small=summarize_production_trends(fit,output_dir=tmp_path/'small8',anchor_month='1986-01',seed=1602,workspace_bytes=5*24*304*4)
 large=summarize_production_trends(fit,output_dir=tmp_path/'large8',anchor_month='1986-01',seed=1602)
 assert small['draw_tile']==1 and small['selected_walkers_per_sweep']==8
 for g in range(5):
  with np.load(tmp_path/'small8'/f'cell-{g:04d}.npz') as a,np.load(tmp_path/'large8'/f'cell-{g:04d}.npz') as b:
   for key in ('mean','sd','change_mean','change_sd','selected_time_ensemble_mean'):
    np.testing.assert_allclose(a[key],b[key],rtol=2e-5,atol=2e-6)



def test_cuda_algorithm_on_cpu_oracle_not_native_cuda():
 y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
 p=prepare_data(y.isel(time=slice(0,12)),x.isel(time=slice(0,12)),uncertainty=s.isel(time=slice(0,12)))
 u=np.array([-9.2,-4.6,-2.3,0.])+np.random.default_rng(1602).normal(size=(2,5,4))*.1
 with _threads('numba',1,8):ref=fixed_grid_moments(p,u,6)
 out=fixed_cuda_trend_moments(p,torch.tensor(u),6).numpy()
 np.testing.assert_allclose(out,ref,rtol=1e-8,atol=1e-10)
