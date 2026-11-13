import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "hpc" / "summarize_grid_output.py"
spec = importlib.util.spec_from_file_location("summarize_grid_output", SCRIPT)
summarize_grid_output = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_grid_output)


def _write(tmp_path, status, divergences=None):
    data = {"fit_status": (("pressure", "latitude"), np.asarray(status, dtype=np.int8))}
    if divergences is not None:
        data["divergent_transitions"] = (("pressure", "latitude"), np.asarray(divergences, dtype=np.int32))
    path = tmp_path / "grid.nc"
    xr.Dataset(data, coords={"chain": [0, 1], "draw": np.arange(5)}).to_netcdf(path)
    return path


def test_reports_status_counts_and_divergent_cells(tmp_path):
    path = _write(tmp_path, [[1, 0, 1], [1, 1, 0]], [[0, -1, 3], [7, 0, -1]])

    lines, failed = summarize_grid_output.summarize(path)

    assert not failed
    assert lines[0] == "Cells: 6 (2 skipped, 4 fitted)"
    assert lines[1] == "Divergent transitions: 10 total, 2 fitted cells with any (max 7 of 10 draws)"
    assert lines[2] == "Most divergences (flat ID: count): 3: 7, 2: 3"


def test_failed_cells_are_reported(tmp_path):
    path = _write(tmp_path, [[1, -1], [0, 1]])

    lines, failed = summarize_grid_output.summarize(path)

    assert failed
    assert "Failed flat cell IDs: [1]" in lines
    assert lines[-1] == "Divergent transitions: not stored in this file"
