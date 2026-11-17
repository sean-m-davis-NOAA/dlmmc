"""Public device/dtype contracts and CPU32 end-to-end output, without long chains."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from dlmmc import fit, AffineSettings, summarize_trends
from dlmmc.compute import resolve_compute,available_compute
from dlmmc.production import _cell
from dlmmc.production_smoothing import fixed_grid_moments,compact_arguments,_reduce_grid
from dlmmc.sampling import make_target
from test_sampler_selection import inputs


@pytest.mark.parametrize('backend,dtype',[('cpu','float32'),('numba','float64'),('cuda','float32'),('cuda','float64'),('mps','float32')])
def test_dtype_policy(backend,dtype):
    assert resolve_compute(backend,dtype=dtype,check_available=False).dtype==dtype
    assert 'torch' not in available_compute()


def test_unsupported_precision_and_removed_backend():
    for kwargs in [dict(backend='mps',dtype='float64'),dict(backend='numba',dtype='float16'),dict(backend='torch')]:
        with pytest.raises(ValueError):resolve_compute(**kwargs,check_available=False)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_numba_precision_fit_smooth_and_resume(tmp_path,dtype):
    from dlmmc.production import resume_production, production_parameter_cache
    y,s,x=inputs(24)
    f=fit(y,x,uncertainty=s,sampler='affine',backend='cpu',dtype=dtype,
          settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2),output_dir=tmp_path/'fit')
    assert str(f.prepared.y.dtype)=='torch.'+dtype
    assert production_parameter_cache(f).dtype==np.dtype(dtype)
    before=f.posterior().unconstrained.values.copy()
    resumed=resume_production(f.path,retained_sweeps=16,wall_seconds=120)
    np.testing.assert_array_equal(resumed.posterior().unconstrained.values[:,:8],before)
    result=summarize_trends(resumed,output_dir=tmp_path/'out',anchor_month='1998-01')
    assert np.isfinite(result.trend).all() and bool((result.trend_sd>0).all())


def test_five_panel_float32_filter_smoother_and_layouts():
    from pathlib import Path
    from dlmmc.examples import load_panels
    from dlmmc import prepare_data
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_data(y,x,uncertainty=s)
    q=replace(p,**{n:getattr(p,n).float() for n in ('y','variance','regressors')})
    u=np.broadcast_to(np.array([-9.9,-7.,-2.9,1.1]),(2,5,4)).copy()
    u+=np.random.default_rng(526).normal(0,.05,u.shape)
    expected=make_target(p,'numba')(torch.tensor(u))
    actual=make_target(q,'numba')(torch.tensor(u,dtype=torch.float32))
    assert torch.isfinite(actual).all()
    reference=fixed_grid_moments(p,u,156);moments=fixed_grid_moments(q,u,156)
    assert moments.dtype==np.float32
    # Existing pointwise scientific margins, not relaxed arithmetic tolerances.
    assert np.max(abs(moments[...,0]-reference[...,0])/np.sqrt(reference[...,1]))<.1
    assert np.max(abs(np.sqrt(moments[...,1]/reference[...,1])-1))<.05
    for ix in [[0],[4,2,0,3,1],[0,1],[2,3,4]]:
        np.testing.assert_array_equal(fixed_grid_moments(_cell(q,ix),u[:,ix],156),moments[:,ix])
    other=_cell(q,list(range(5)));other.y=other.y.clone();other.y[0]+=.1
    np.testing.assert_array_equal(fixed_grid_moments(other,u,156)[:,1:],moments[:,1:])
    result,_,_=_reduce_grid(*compact_arguments(q,u,156),1)
    assert result.dtype==np.float32


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA unavailable')
def test_native_cuda_precision_fit_and_smoother(tmp_path,dtype):
    y,s,x=inputs(24)
    f=fit(y,x,uncertainty=s,sampler='affine',backend='cuda',dtype=dtype,
          settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2),output_dir=tmp_path/'fit')
    assert f.prepared.y.device.type=='cuda' and str(f.prepared.y.dtype)=='torch.'+dtype
    result=summarize_trends(f,output_dir=tmp_path/'out',anchor_month='1998-01')
    assert np.isfinite(result.trend).all()


@pytest.mark.xfail(strict=True,reason='Retained CPU32 p22-lat10 density fixture exceeds 0.01; availability is not precision qualification')
def test_numba32_retained_full_window_density_fixture():
    from pathlib import Path
    from dlmmc.examples import load_panels
    from dlmmc import prepare_data
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_data(y,x,uncertainty=s,priors={"S":10.})
    q=replace(p,**{n:getattr(p,n).float() for n in ('y','variance','regressors')})
    u=np.broadcast_to(np.array([-9.9,-7.,-2.9,1.1]),(2,5,4)).copy()
    u+=np.random.default_rng(526).normal(0,.05,u.shape)
    expected=make_target(p,'numba')(torch.tensor(u))
    actual=make_target(q,'numba')(torch.tensor(u,dtype=torch.float32))
    torch.testing.assert_close(actual.double(),expected,atol=.01,rtol=0)


def test_gpu_output_selection_does_not_round_float64(monkeypatch):
    from dlmmc import gpu_smoothing
    values=np.full((1,8,16,1,4),1.+1e-10,dtype=np.float64)
    monkeypatch.setattr(gpu_smoothing,'production_parameter_cache',lambda _:values)
    selected,_,_,_=gpu_smoothing.select_draws(None,2,17)
    assert selected.dtype==np.float64
    np.testing.assert_array_equal(selected,1.+1e-10)


@pytest.mark.parametrize('model',['vanilla_ar1','vanilla_ar2','dynregs_ar1','noregs_ar1','vanilla_ar1_noseasonal','vanilla_ar2_noseasonal'])
def test_cpu32_all_model_kernels(model):
    from dlmmc import prepare_data,conditional_trend_moments,deseasonalize_harmonics
    from dlmmc.production import initial_center
    y,s,x=inputs(24)
    if 'noseasonal' in model:y=deseasonalize_harmonics(y).deseasonalized
    if model=='noregs_ar1':x=None
    p=prepare_data(y,x,uncertainty=s,model=model,dtype='float32')
    u=p.y.new_tensor(initial_center(p)).reshape(1,1,-1)
    assert make_target(p,'numba')(u).dtype==torch.float32
    result=conditional_trend_moments(p,u,12,backend='cpu')
    assert result.dtype==torch.float32 and torch.isfinite(result).all()


@pytest.mark.parametrize('with_level',[True,False])
def test_cpu32_composed_no_ar_kernels(with_level):
    from dlmmc import prepare_data,conditional_trend_moments,Model,Level,Trend,Harmonics,Regression
    from dlmmc.production import initial_center
    y,s,x=inputs(24)
    model=Model(Level()) if with_level else Trend()+Harmonics()+Regression()
    if with_level:
        from dlmmc import deseasonalize_harmonics
        y=deseasonalize_harmonics(y).deseasonalized;x=None
    p=prepare_data(y,x,uncertainty=s,model=model,dtype='float32')
    u=p.y.new_tensor(initial_center(p)).reshape(1,1,-1)
    assert torch.isfinite(make_target(p,'numba')(u)).all()
    moments=conditional_trend_moments(p,u,12)
    assert moments.dtype==torch.float32 and torch.isfinite(moments).all()
