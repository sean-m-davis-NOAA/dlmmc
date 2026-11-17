import numpy as np
import pytest
import torch
import xarray as xr
from dlmmc import fork_cells,resume_fit
from dlmmc.prepare import prepare_data
from dlmmc.sampling import sample_parameters


def test_selected_cell_continuation_preserves_prefix_coordinates_and_parent(tmp_path):
    raw=xr.DataArray(np.sin(np.arange(12)/2)[:,None]+np.arange(3)[None,:],dims=('time','panel'),
        coords={'time':np.arange(12),'panel':['a','b','c'],'pressure':('panel',[10.,20.,30.])})
    p=prepare_data(raw,model='noregs_ar1',uncertainty=xr.ones_like(raw)*.1)
    rng=torch.Generator().manual_seed(240)
    center=torch.tensor([-9.,-4.6,-2.3,0.],dtype=torch.float64)
    h=[tuple(center+torch.randn((8,3,4),dtype=torch.float64,generator=rng)*.01 for _ in range(2)) for _ in range(4)]
    parent=sample_parameters(p,output_dir=tmp_path/'parent',backend='numba',retained_sweeps=8,
        warmup=3,half_walkers=8,chunk_sweeps=4,initial_halves=h,thin=3)
    original=parent.posterior()
    fork=fork_cells(parent,[2,0],output_dir=tmp_path/'subset')
    before=fork.posterior()
    np.testing.assert_array_equal(before.unconstrained,original.unconstrained.values[:,:,:,[2,0]])
    assert before.panel.values.tolist()==['c','a']
    assert not fork.config['lineage']['independent_replicate']
    continued=resume_fit(fork.path,retained_sweeps=14).posterior()
    np.testing.assert_array_equal(continued.unconstrained.values[:,:3],before.unconstrained)
    xr.testing.assert_identical(parent.posterior(),original)
    assert continued.draw.values.tolist()==[4,7,10,13,16]
    with pytest.raises(ValueError):fork_cells(parent,[1,1],output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
