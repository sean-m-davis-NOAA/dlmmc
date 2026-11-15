import numpy as np
import pytest
import xarray as xr
from dlmmc import deseasonalize_harmonics


def fixture():
    t = np.arange(60.)
    cycle = 2*np.sin(2*np.pi*t/12) + .7*np.cos(4*np.pi*t/12)
    y = xr.DataArray((10 + .03*t + cycle)[:, None] * [1., 2., 3.],
        dims=('time', 'panel'), coords={'time': np.arange('2000-01', '2005-01', dtype='datetime64[M]'),
                                     'panel': ['a', 'b', 'c']}, attrs={'units': 'ppmv'})
    return t, cycle, y


def test_known_cycle_preserves_trend_units_and_reconstructs():
    t, cycle, y = fixture()
    original = y.copy(deep=True)
    result = deseasonalize_harmonics(y)
    np.testing.assert_allclose(result.deseasonalized, (10 + .03*t)[:, None]*[1,2,3], atol=2e-14)
    xr.testing.assert_allclose(result.deseasonalized + result.seasonal_component, y)
    xr.testing.assert_identical(y, original)
    assert result.deseasonalized.attrs['units'] == 'ppmv'
    assert result.deseasonalized.attrs['seasonality'] == 'deseasonalized'


def test_missing_mask_mean_and_layout_invariance():
    _, _, y = fixture()
    y[3:8, 1] = np.nan
    result = deseasonalize_harmonics(y)
    np.testing.assert_array_equal(np.isnan(result.deseasonalized), np.isnan(y))
    np.testing.assert_allclose(result.deseasonalized.mean('time'), y.mean('time'), atol=1e-14)
    for i in range(3):
        xr.testing.assert_allclose(result.deseasonalized.isel(panel=i),
            deseasonalize_harmonics(y.isel(panel=i)).deseasonalized)
    xr.testing.assert_allclose(result.deseasonalized.sel(panel=['c','a','b']),
        deseasonalize_harmonics(y.sel(panel=['c','a','b'])).deseasonalized)
    xr.testing.assert_allclose(result.deseasonalized,
        deseasonalize_harmonics(y.transpose('panel','time')).deseasonalized.transpose('time','panel'))
    changed = y.copy(); changed[:, 2] *= 100
    xr.testing.assert_allclose(result.deseasonalized.isel(panel=[0,1]),
        deseasonalize_harmonics(changed).deseasonalized.isel(panel=[0,1]))


def test_invalid_inputs_and_double_removal_rejected():
    _, _, y = fixture()
    for bad in (y.isel(time=slice(0,5)), y*float('inf'), y*np.nan,
                y.assign_coords(time=np.arange(60)), y.isel(time=[0,0,2,3,4,5])):
        with pytest.raises(ValueError): deseasonalize_harmonics(bad)
    with pytest.raises(ValueError):
        deseasonalize_harmonics(deseasonalize_harmonics(y).deseasonalized)
    with pytest.raises(ValueError, match='Rank-deficient'):
        deseasonalize_harmonics(y.isel(time=[0,12,24,36,48,49]))


def test_public_fit_boundaries_reject_raw_without_allocating(tmp_path):
    import dlmmc
    _, _, y = fixture()
    for model in ['vanilla_ar1_noseasonal', 'vanilla_ar2_noseasonal']:
        with pytest.raises(ValueError, match='explicitly deseasonalised'):
            dlmmc.fit_grid(y, model=model, output_dir=tmp_path/model)
        assert not (tmp_path/model).exists()
        with pytest.raises(ValueError, match='explicitly deseasonalised'):
            dlmmc.fit(y.isel(panel=0), model=model)
        with pytest.raises(ValueError, match='explicitly deseasonalised'):
            dlmmc.prepare_data(y, model=model)
    clean = deseasonalize_harmonics(y).deseasonalized
    p = dlmmc.prepare_data(clean, np.zeros((60,1)), model='vanilla_ar1_noseasonal')
    assert p.cells == 3


def test_checkpoint_declaration_bound_to_input(tmp_path):
    from types import SimpleNamespace
    from dlmmc.input_contract import record_input_policy, checkpoint_input_policy
    fit = SimpleNamespace(path=tmp_path, prepared=SimpleNamespace(model='vanilla_ar1_noseasonal', checksum='a'))
    with pytest.raises(ValueError, match='lacks an input declaration'): checkpoint_input_policy(fit)
    policy = dict(seasonality='deseasonalized', description='external harmonics')
    record_input_policy(fit, policy)
    assert checkpoint_input_policy(fit) == policy
    fit.prepared.checksum = 'b'
    with pytest.raises(ValueError, match='does not match'): checkpoint_input_policy(fit)
