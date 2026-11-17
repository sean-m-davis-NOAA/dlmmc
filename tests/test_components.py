from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from dlmmc import Model,Trend,Level,Harmonics,Regression,AR,build_state_space,prepare_data
from dlmmc.components import compiled_layout,initial_coordinates,bound_model
from dlmmc.api import _MODEL_SPECS
from dlmmc.examples import load_panels
from dlmmc.sampling import make_target,_save_prepared,_load_prepared
from dlmmc.tensor_target import model_components
from dlmmc.production import _cell
from dlmmc import conditional_trend_moments


def inputs(model,k=3,n=36):
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    y=y.isel(time=slice(0,n));s=s.isel(time=slice(0,n));x=x.isel(time=slice(0,n),regressor=slice(0,k)) if k else None
    from dlmmc.components import component_spec
    if not component_spec(model)['seasonal']:
        from dlmmc import deseasonalize_harmonics
        y=deseasonalize_harmonics(y).deseasonalized
        if x is not None:x=deseasonalize_harmonics(x).deseasonalized
    return y,s,x


@pytest.mark.parametrize('name',list(_MODEL_SPECS))
def test_preset_compilation_matches_independent_numpy_reference(name):
    from dlmmc.model_reference import matrices,physical_logprior
    from dlmmc.transforms import constrain
    k=0 if name=='noregs_ar1' else 3
    y,s,x=inputs(name,k);m=Model.from_preset(name)
    p=prepare_data(y,x,uncertainty=s,model=m)
    u=initial_coordinates(p.model,k)+np.random.default_rng(2001).normal(0,.1,len(p.parameter_names))
    F,G,Q,C,lp,valid=model_components(torch.tensor(u),*p.arguments,p.model)
    theta,logj=constrain(name,u,k)
    for g in range(5):
        expected=matrices(name,p.regressors[g],theta)
        for a,b in zip((F[g],G[g],Q[g],C[g]),expected):np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-15)
    assert bool(valid)
    np.testing.assert_allclose(lp,physical_logprior(name,theta,p.priors)+logj,rtol=1e-13)
    old=prepare_data(y,x,uncertainty=s,model=name)
    assert p.parameter_names==old.parameter_names
    z=torch.tensor(u).expand(3,5,-1).clone()
    torch.testing.assert_close(make_target(p,'numba')(z),make_target(old,'numba')(z),rtol=0,atol=0)


def test_composition_binding_roundtrip_and_priors(tmp_path):
    y,s,x=inputs('vanilla_ar1',3)
    dynamic=str(x.regressor.values[1])
    m=Trend()+Harmonics((12,),name='annual')+Harmonics((6,),name='semi')+Regression(dynamic=(dynamic,))+AR(2)
    p=prepare_data(y,x,uncertainty=s,model=m,priors={'sigma_annual_prior':.03,'sigma_reg_prior':[.1,.2,.3]})
    l=compiled_layout(p.model,3)
    assert l.dynamic_indices==(1,)
    assert p.parameter_names==['sigma_trend','sigma_annual','sigma_semi','sigma_AR','sigma_reg[1]','rhoAR1','rhoAR2']
    _save_prepared(tmp_path,p);q=_load_prepared(tmp_path)
    assert q.model==p.model and q.parameter_names==p.parameter_names
    z=torch.tensor(initial_coordinates(p.model,3)).expand(2,5,-1).clone()
    torch.testing.assert_close(make_target(p,'numba')(z),make_target(q,'numba')(z),rtol=0,atol=0)
    built=build_state_space(p.model,z,p.regressors,priors=p.priors)
    Q=built['innovation'];base=l.regressor_start
    assert torch.all(Q[...,base,base]==1e-20)
    assert torch.all(Q[...,base+1,base+1]>1e-12)
    assert set(built['component_slices'])=={'Trend','annual','semi','Regression','AR'}
    # Fresh-interpreter decoding does not depend on a registered Python object.
    import subprocess,sys
    subprocess.run([sys.executable,'-c','from dlmmc.components import compiled_layout;import sys;assert compiled_layout(sys.argv[1],3).dynamic_indices==(1,)',p.model],check=True)


COMPOSITIONS=[
    (Model(Trend(),Harmonics((12,6,4)),Regression(dynamic=True),AR(2)),3),
    (Model(Level(),Harmonics((12,)),Regression(),AR()),2),
    (Model(Trend(),Harmonics((12,)),Regression(dynamic=('SOLAR',))),3),
    (Model(Level()),0),
    (Model(Harmonics((12,6)),AR(2)),0),
]


@pytest.mark.parametrize('well_conditioned',[False,True])
@pytest.mark.parametrize('model,k',COMPOSITIONS)
@pytest.mark.parametrize('backend',['numba','torch','mps','cuda'])
def test_new_composition_fixed_kernels_and_layouts(model,k,backend,well_conditioned,request):
    if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
    if backend=='cuda' and not torch.cuda.is_available():pytest.skip('Native CUDA unavailable')
    y,s,x=inputs(model,k)
    if isinstance(model.components[-1],Regression) and model.components[-1].dynamic==('SOLAR',):
        model=Model(Trend(),Harmonics((12,)),Regression(dynamic=(str(x.regressor.values[1]),)))
    p=prepare_data(y,x,uncertainty=s,model=model,priors={"S":10.})
    u=torch.tensor(initial_coordinates(p.model,k))+torch.tensor(np.random.default_rng(2002).normal(0,.08,(3,5,len(p.parameter_names))))
    if well_conditioned:
        p=replace(p,variance=torch.full_like(p.variance,.1))  # Separate numerical fixture, not an alteration to real inputs.
    if backend=='mps' and compiled_layout(p.model,k).ar_order==0 and not well_conditioned:
        request.node.add_marker(pytest.mark.xfail(strict=True,reason='Retained no-AR ozone float32 density failures; backend access is user-authorized; see composition precision fixtures'))
    ref=make_target(p,'numba')(u)
    device=backend if backend in ('mps','cuda') else 'cpu';dtype=torch.float32 if device!='cpu' else torch.float64
    q=replace(p,**{name:getattr(p,name).to(device=device,dtype=torch.bool if name=='observed' else dtype).contiguous() for name in ('y','variance','observed','regressors')});z=u.to(device=device,dtype=dtype)
    f=make_target(q,backend);actual=f(z)
    np.testing.assert_allclose(actual.cpu(),ref,rtol=0,atol=.01 if dtype==torch.float32 else 1e-8)
    for g in range(5):torch.testing.assert_close(make_target(_cell(q,[g]),backend)(z[:,g:g+1]),actual[:,g:g+1],rtol=0,atol=0)
    order=[4,1,3,0,2];torch.testing.assert_close(make_target(_cell(q,order),backend)(z[:,order])[:,np.argsort(order)],actual,rtol=0,atol=0)
    chunk=torch.cat([make_target(_cell(q,ix),backend)(z[:,ix]) for ix in ([0,1],[2,3,4])],1)
    torch.testing.assert_close(chunk,actual,rtol=0,atol=0)
    changed=replace(q,y=q.y.clone());changed.y[0]+=.1
    torch.testing.assert_close(make_target(changed,backend)(z)[:,1:],actual[:,1:],rtol=0,atol=0)
    if compiled_layout(p.model,k).level_kind is not None:
        expected=conditional_trend_moments(p,u,12,backend='torch')
        result=conditional_trend_moments(q,z,12,backend=backend).cpu()
        scale=expected[...,1].sqrt().clamp_min(1e-30)
        assert float(((result[...,0]-expected[...,0])/scale).abs().max())<.05
        np.testing.assert_allclose(result[...,1],expected[...,1],rtol=.05,atol=1e-9)
        keep=expected[...,3]>0
        assert float(((result[...,2][keep]-expected[...,2][keep])/expected[...,3][keep].sqrt()).abs().max())<.05
        assert float((torch.sqrt(result[...,3][keep]/expected[...,3][keep])-1).abs().max())<.05
    else:
        with pytest.raises(ValueError,match='Trend or Level'):conditional_trend_moments(q,z,12,backend=backend)


def test_invalid_and_serialization_contracts():
    for fn in [lambda:Model(Trend(),Level()),lambda:Model(AR(3)),lambda:Model(Harmonics((12,12))),lambda:Model(Harmonics((12,),name='trend')),lambda:Model(Regression(dynamic='ENSO'))]:
        with pytest.raises(ValueError):fn()
    model=Model(Trend(),Harmonics(),AR(2));assert Model.from_dict(json.loads(json.dumps(model.to_dict())))==model
    assert AR(2)+Harmonics()+Trend()==model
    y,s,x=inputs('vanilla_ar1',3)
    with pytest.raises(ValueError,match='names'):prepare_data(y,x,uncertainty=s,model=Model(Trend(),Harmonics(),Regression(dynamic=('absent',))))
    with pytest.raises(ValueError,match='deseasonalised'):
        import dlmmc
        dlmmc.prepare_data(y,model=Model(Level()))


@pytest.mark.parametrize('backend',['numba','mps'])
def test_composed_production_checkpoint_output(tmp_path,backend):
    if backend=='mps' and not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
    from dlmmc import PRODUCTION_PROFILE,fit_production,resume_production,summarize_production_trends,diagnose_production
    y,s,x=inputs('vanilla_ar1',3,24);y=y.isel(panel=[0]);s=s.isel(panel=[0]);x=x.isel(panel=[0])
    model=Model(Trend(),Harmonics((12,6,4)),Regression(dynamic=(str(x.regressor.values[1]),)),AR(2))
    profile=replace(PRODUCTION_PROFILE,half_walkers=16,warmup=4,retained_sweeps=8,output_draws_per_sweep=4,chunk_sweeps=4)
    fit=fit_production(y,x,uncertainty=s,model=model,profile=profile,backend=backend,output_dir=tmp_path/'fit',wall_seconds=120)
    assert fit.prepared.model_spec==model
    result=diagnose_production(fit);assert not result['qualification_eligible']
    fit=resume_production(fit.path,retained_sweeps=12,wall_seconds=60)
    assert fit.posterior().sizes['draw']==12
    output=summarize_production_trends(fit,output_dir=tmp_path/'summary',anchor_month='1986-01',wall_seconds=60)
    assert output['actual_wall_seconds']<60


def test_composed_joint_targets():
    from test_enso_lag import fixture
    from dlmmc.enso_lag import ENSOLagTarget
    from dlmmc.spatial_sampling import SpatialTarget
    p,s,l,x=fixture()
    model=Model.from_preset(p.model);q=replace(p,model=bound_model(model,p.regressor_names))
    u=torch.tensor(initial_coordinates(p.model,len(p.regressor_names))).expand(3,4,-1).clone()
    torch.testing.assert_close(SpatialTarget(q,s,backend='numba')(u.flatten(1)),SpatialTarget(p,s,backend='numba')(u.flatten(1)),rtol=0,atol=0)
    z=torch.cat((u,torch.full((3,4,1),-2.)),dim=-1)
    torch.testing.assert_close(ENSOLagTarget(q,s,l,x)(z.flatten(1)),ENSOLagTarget(p,s,l,x)(z.flatten(1)),rtol=0,atol=0)


@pytest.mark.parametrize('backend', ['numba', 'cpu', 'mps', 'cuda'])
@pytest.mark.parametrize('model', [Model(Level()), Model(Trend(), Harmonics(), Regression())])
def test_no_ar_backend_access(tmp_path, backend, model):
    """Bounded execution smoke, not numerical or posterior qualification."""
    if backend == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Native MPS unavailable')
    if backend == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Native CUDA unavailable; not a pass')
    from dlmmc import PRODUCTION_PROFILE, fit_production, summarize_production_trends
    k = 0 if isinstance(model.components[0], Level) else 3
    y, s, x = inputs(model, k, 24)
    y, s = y.isel(panel=[0]), s.isel(panel=[0])
    if x is not None:
        x = x.isel(panel=[0])
    # Well-conditioned output smoke; original ozone stress fixtures above remain unchanged.
    s = s.copy(data=np.full(s.shape, 0.2))
    profile = replace(PRODUCTION_PROFILE, half_walkers=8, warmup=4,
                      retained_sweeps=8, output_draws_per_sweep=2, chunk_sweeps=4)
    fit = fit_production(y, x, uncertainty=s, model=model, profile=profile,
                         backend=backend, output_dir=tmp_path/'fit', wall_seconds=120)
    assert fit.posterior().sizes['draw'] == 8
    assert fit.config['backend'] == ('numba' if backend == 'cpu' else backend)
    expected = torch.float32 if backend in ('mps', 'cuda') else torch.float64
    assert fit.prepared.y.dtype == expected
    output = summarize_production_trends(fit, output_dir=tmp_path/'summary',
                                        anchor_month='1986-01', wall_seconds=60)
    assert output['actual_wall_seconds'] < 60


def test_production_custom_regressor_inputs():
    from dlmmc.production import prepare_production
    import xarray as xr
    y, s, x = inputs('vanilla_ar1', 3, 24)
    y, s = y.isel(panel=[0]), s.isel(panel=[0])
    shared = x.isel(panel=0, drop=True)
    named = xr.Dataset({str(n): shared.sel(regressor=n, drop=True)
                        for n in shared.regressor.values})
    model = Model.from_preset('vanilla_ar1')
    a = prepare_production(y, named, uncertainty=s, model=model)
    b = prepare_production(y, shared.transpose('time', 'regressor').values,
                           uncertainty=s, model=model)
    torch.testing.assert_close(a.regressors, b.regressors, rtol=0, atol=0)
    assert a.regressor_names == list(named.data_vars)
    assert b.regressor_names == ['regressor_1', 'regressor_2', 'regressor_3']
