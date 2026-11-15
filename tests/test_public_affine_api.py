import json
import numpy as np
import pytest
import torch
import xarray as xr
from dlmmc.prepare import prepare_data
from dlmmc.sampling import sample_parameters,resume_fit,fit_grid,make_target


def setup():
    data=xr.DataArray(np.sin(np.arange(12)/2)+.2*np.arange(12),dims='time',coords={'time':np.arange(12)})
    p=prepare_data(data,model='noregs_ar1',uncertainty=xr.ones_like(data)*.1)
    gen=torch.Generator().manual_seed(112)
    center=torch.tensor([-9.2,-4.6,-2.3,0.],dtype=torch.float64)
    halves=[tuple(center+torch.randn((8,1,4),dtype=torch.float64,generator=gen)*.01 for _ in range(2)) for _ in range(4)]
    return p,halves


def test_public_resume_matches_uninterrupted_and_has_explicit_axes(tmp_path):
    p,h=setup()
    options=dict(backend='torch',warmup=3,half_walkers=8,thin=3,chunk_sweeps=4,seed=555,initial_halves=h)
    full=sample_parameters(p,output_dir=tmp_path/'full',retained_sweeps=17,**options).posterior()
    part=sample_parameters(p,output_dir=tmp_path/'part',retained_sweeps=8,**options)
    resumed=resume_fit(part.path,retained_sweeps=17).posterior()
    np.testing.assert_array_equal(full.unconstrained,resumed.unconstrained)
    np.testing.assert_array_equal(full.log_probability,resumed.log_probability)
    assert full.unconstrained.dims==('run','draw','walker','cell','parameter')
    assert full.sizes['run']==4 and full.sizes['walker']==16
    assert full.attrs['diagnostic_status']=='NOT_YET_DIAGNOSED'
    with pytest.raises(ValueError):sample_parameters(p,output_dir=part.path,retained_sweeps=1,**options)


def test_bounded_wall_limit_saves_initial_checkpoints(tmp_path):
    p,h=setup()
    result=sample_parameters(p,output_dir=tmp_path/'cap',backend='torch',retained_sweeps=10,
                            half_walkers=8,initial_halves=h,wall_seconds=.00001)
    status=json.loads((result.path/'latest.json').read_text())
    assert status['status']=='WALL_CAP_CHECKPOINTED'
    assert all((result.path/f'run-{i}'/'initial.npz').exists() for i in range(4))


def test_numba_and_torch_targets_have_matching_prior_override():
    p,_=setup();p.priors['sigma_trend_prior']=2e-4
    u=torch.tensor([[[-9.,-4.,-2.,0.]]],dtype=torch.float64)
    torch.testing.assert_close(make_target(p,'numba')(u),make_target(p,'torch')(u),atol=1e-8,rtol=0)


def test_grid_guard_precedes_sampler_allocation(tmp_path):
    raw=xr.DataArray(np.arange(72.).reshape(12,6),dims=('time','panel'),coords={'time':np.arange(12)})
    with pytest.raises(ValueError):fit_grid(raw,model='noregs_ar1',output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_modified_backend_configuration_cannot_resume(tmp_path):
    p,h=setup();folder=tmp_path/'mutated'
    sample_parameters(p,output_dir=folder,backend='torch',retained_sweeps=2,warmup=0,half_walkers=8,initial_halves=h)
    config=json.loads((folder/'config.json').read_text());config['backend']='numba'
    (folder/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration changed'):resume_fit(folder,retained_sweeps=4)


def test_threads_restored_and_rank_deficient_cloud_rejected(tmp_path):
    p,h=setup();old=torch.get_num_threads()
    sample_parameters(p,output_dir=tmp_path/'threaded',backend='torch',retained_sweeps=2,warmup=0,half_walkers=8,initial_halves=h,torch_threads=1)
    assert torch.get_num_threads()==old
    h[0]=(torch.zeros_like(h[0][0]),h[0][1])
    with pytest.raises(ValueError,match='rank deficient'):
        sample_parameters(p,output_dir=tmp_path/'bad-rank',backend='torch',retained_sweeps=2,half_walkers=8,initial_halves=h)
