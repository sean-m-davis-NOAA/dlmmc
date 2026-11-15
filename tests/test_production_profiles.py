from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from dlmmc import (fit_grid, fit_production, resume_production, diagnose_production,
                   PRODUCTION_PROFILE, QUALIFICATION_PROFILE, prepare_data)
from dlmmc.examples import load_panels
from dlmmc.production import prepare_production, production_parameter_cache
from dlmmc.profiles import validate_production_scope


@pytest.fixture
def inputs():
    data,unc,reg,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    return data.isel(time=slice(0,24),panel=[0]),unc.isel(time=slice(0,24),panel=[0]),reg.isel(time=slice(0,24),panel=[0])


def test_production_scope_does_not_weaken_qualification(inputs,tmp_path):
    y,s,x=inputs
    assert PRODUCTION_PROFILE.runs==1 and QUALIFICATION_PROFILE.runs==4
    with pytest.raises(ValueError):fit_grid(y,x,uncertainty=s,output_dir=tmp_path/'bad',runs=1)
    with pytest.raises(ValueError):validate_production_scope([str(i) for i in range(6)],None,model='vanilla_ar1',backend='numba',dtype='float64',profile=PRODUCTION_PROFILE)
    with pytest.raises(ValueError):fit_grid(y,x,uncertainty=s,profile='typo',output_dir=tmp_path/'bad2')


def test_preparation_identity_and_named_scope(inputs):
    y,s,x=inputs
    p=prepare_production(y,x,uncertainty=s);q=prepare_data(y,x,uncertainty=s)
    for a,b in zip(p.arguments,q.arguments):np.testing.assert_array_equal(a,b)
    ids=[str(i) for i in range(6)]
    selection=dict(scope='independent_production',explicit_user_response='six selected test panels',panel_ids=ids,model='vanilla_ar1',backend='numba',dtype='float64',profile=PRODUCTION_PROFILE.to_dict())
    validate_production_scope(ids,selection,model='vanilla_ar1',backend='numba',dtype='float64',profile=PRODUCTION_PROFILE)
    with pytest.raises(ValueError):validate_production_scope(ids[::-1],selection,model='vanilla_ar1',backend='numba',dtype='float64',profile=PRODUCTION_PROFILE)


def test_single_ensemble_exact_resume_and_warning(inputs,tmp_path):
    y,s,x=inputs;profile=replace(PRODUCTION_PROFILE,half_walkers=8,warmup=16,retained_sweeps=16,chunk_sweeps=8)
    fit=fit_production(y,x,uncertainty=s,output_dir=tmp_path/'first',profile=profile,seed=901)
    assert fit.config['threads']=={'torch':1,'numba':8}
    u=production_parameter_cache(fit);assert u.shape==(1,16,16,1,4)
    fit=resume_production(fit.path,retained_sweeps=32)
    fresh=fit_production(y,x,uncertainty=s,output_dir=tmp_path/'fresh',profile=replace(profile,retained_sweeps=32),seed=901)
    np.testing.assert_array_equal(production_parameter_cache(fit),production_parameter_cache(fresh))
    d=diagnose_production(fit)
    assert not d['qualification_eligible'] and 'SINGLE_ENSEMBLE_NO_BETWEEN_RUN_RHAT' in d['cells'][0]['flags']
    assert d['cells'][0]['between_run_rhat'] is None
    with pytest.raises(ValueError):fit_production(y,x,uncertainty=s,output_dir=fit.path,profile=profile)
    from dlmmc import summarize_production_trends
    anchor=str(fresh.prepared.time[0].astype('datetime64[M]'))
    modern=summarize_production_trends(fresh,output_dir=tmp_path/'modern',anchor_month=anchor,seed=901,selected_walkers_per_sweep=4)
    legacy=summarize_production_trends(fresh,output_dir=tmp_path/'legacy',anchor_month=anchor,seed=901,engine='legacy',selected_walkers_per_sweep=4)
    assert modern['full_batch_shape']==[128,1] and modern['compiled_output_calls']==1
    assert not modern['draw_chunking'] and not modern['panel_chunking']
    with np.load(tmp_path/'modern/cell-0000.npz') as a,np.load(tmp_path/'legacy/cell-0000.npz') as b:
        for key in ('mean','sd','change_mean','change_sd','first_half_mean','second_half_mean','selected_time_ensemble_mean'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-10,atol=2e-12)
        np.testing.assert_array_equal(a['selected_walker'],b['selected_walker'])
