import json
import numpy as np
import pytest
import torch
from dlmmc.spatial_sampling import sample_spatial,resume_spatial,load_spatial
from test_spatial_prior import prepared,prior


def clouds():
    center=np.tile([-9.,-4.,-2.,0.],4)
    rng=np.random.default_rng(451)
    return [tuple(torch.tensor(center+.01*rng.normal(size=(20,16))) for _ in range(2)) for _ in range(4)]


@pytest.mark.parametrize('schedule',['global','field_local'])
def test_joint_resume_matches_uninterrupted_and_preserves_alignment(tmp_path,schedule):
    p=prepared();options=dict(backend='numba',half_walkers=20,warmup=3,runs=4,chunk_sweeps=4,seed=940,initial_halves=clouds(),reference=True,update_schedule=schedule)
    full=sample_spatial(p,prior(),output_dir=tmp_path/'full',retained_sweeps=13,**options).posterior()
    part=sample_spatial(p,prior(),output_dir=tmp_path/'part',retained_sweeps=5,**options)
    resumed=resume_spatial(part.path,retained_sweeps=13).posterior()
    np.testing.assert_array_equal(full.unconstrained,resumed.unconstrained)
    np.testing.assert_array_equal(full.joint_log_probability,resumed.joint_log_probability)
    assert full.unconstrained.shape==(4,13,40,4,4)
    assert full.joint_log_probability.shape==(4,13,40)
    assert not full.attrs['spatial_cells_are_independent']


def test_scope_and_dimension_guards_precede_allocation(tmp_path):
    p=prepared()
    with pytest.raises(ValueError,match='selection'):
        sample_spatial(p,prior(),output_dir=tmp_path/'bad',phase='selected')
    with pytest.raises(ValueError,match='Full-rank'):
        sample_spatial(p,prior(),output_dir=tmp_path/'bad',half_walkers=16)
    with pytest.raises(ValueError,match='pilot cap'):
        sample_spatial(p,prior(),output_dir=tmp_path/'bad',wall_seconds=601)
    assert not (tmp_path/'bad').exists()


def test_joint_source_or_config_mutation_cannot_resume(tmp_path):
    p=prepared();f=sample_spatial(p,prior(),output_dir=tmp_path/'fit',retained_sweeps=2,half_walkers=20,initial_halves=clouds(),warmup=0)
    c=json.loads((f.path/'config.json').read_text());c['prior']['length_latitude']*=2
    (f.path/'config.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='configuration changed'):resume_spatial(f.path,retained_sweeps=4)
