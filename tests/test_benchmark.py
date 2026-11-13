import json
import sys

import numpy as np
import xarray as xr

import dlm_benchmark


def test_benchmark_forwards_parallel_chains_and_cache_dir(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    output_dir = tmp_path / 'output'
    calls = {}

    def compile_models(models, **kwargs):
        calls['compile'] = (models, kwargs)
        return {'vanilla_ar1': '/tmp/model'}

    def fit(data, **kwargs):
        calls['fit'] = (data, kwargs)
        return xr.Dataset({'draws': (('chain', 'draw'), np.ones((4, 2)))})

    monkeypatch.setattr(
        dlm_benchmark, 'load_example',
        lambda pressure, latitude: ('data', 'sigma', 'regressors'),
    )
    monkeypatch.setattr(dlm_benchmark.dlmmc, 'compile_models', compile_models)
    monkeypatch.setattr(dlm_benchmark.dlmmc, 'fit', fit)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dlm_benchmark.py', '--chains', '4', '--parallel-chains', '1',
            '--warmup', '1', '--samples', '2', '--cache-dir', str(cache_dir),
            '--output-dir', str(output_dir),
        ],
    )

    dlm_benchmark.main()

    assert calls['compile'] == (['vanilla_ar1'], {'cache_dir': cache_dir})
    assert calls['fit'][1]['parallel_chains'] == 1
    assert calls['fit'][1]['cache_dir'] == cache_dir
    result = json.loads(next(output_dir.glob('benchmark_*.json')).read_text())
    assert result['settings']['parallel_chains'] == 1
