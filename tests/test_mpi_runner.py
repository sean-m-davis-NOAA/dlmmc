import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dlm_lat_alt_mpi_run import _select_cells


def test_select_cells_uses_row_major_pressure_latitude_indices():
    data = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=('time', 'pressure', 'latitude'),
        coords={
            'time': pd.date_range('2000-01-01', periods=4),
            'pressure': [10, 20],
            'latitude': [-30, 0, 30],
        },
    )

    selected = _select_cells(data, [0, 4])

    np.testing.assert_array_equal(
        selected.isel(pressure=0, latitude=0),
        data.isel(pressure=0, latitude=0),
    )
    np.testing.assert_array_equal(
        selected.isel(pressure=1, latitude=1),
        data.isel(pressure=1, latitude=1),
    )
    assert np.isnan(selected.isel(pressure=0, latitude=1)).all()
    assert np.isnan(selected.isel(pressure=1, latitude=0)).all()


@pytest.mark.parametrize(
    ('cell_indices', 'message'),
    [([1, 1], 'duplicates'), ([-1], 'between 0 and 5'), ([6], 'between 0 and 5')],
)
def test_select_cells_rejects_duplicate_or_out_of_range_indices(cell_indices, message):
    data = xr.DataArray(
        np.ones((2, 2, 3)),
        dims=('time', 'pressure', 'latitude'),
        coords={'pressure': [10, 20], 'latitude': [-30, 0, 30]},
    )

    with pytest.raises(ValueError, match=message):
        _select_cells(data, cell_indices)


@pytest.mark.parametrize(("extra_args", "show_progress"), [([], True), (["--no-progress"], False)])
def test_main_forwards_progress_setting(monkeypatch, tmp_path, extra_args, show_progress):
    import sys
    from pathlib import Path

    import dlm_lat_alt_mpi_run

    captured = {}

    def fake_fit_grid_mpi(data, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(dlm_lat_alt_mpi_run.dlmmc, "fit_grid_mpi", fake_fit_grid_mpi)
    monkeypatch.setattr(
        sys, "argv", ["dlm_lat_alt_mpi_run.py", "--output", str(tmp_path / "out.nc"), "--cell-indices", "0", *extra_args]
    )

    dlm_lat_alt_mpi_run.main()

    assert captured["show_progress"] is show_progress
