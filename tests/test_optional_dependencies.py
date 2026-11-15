"""Base imports and source provenance must work without optional backends."""
import subprocess
import sys


def test_base_package_without_sampler_or_stan():
    code = '''
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numba', 'affine', 'cmdstanpy'}:
            raise ModuleNotFoundError('Optional dependency deliberately unavailable', name=fullname)
sys.meta_path.insert(0, BlockOptional())
import dlmmc
assert callable(dlmmc.fit)
assert callable(dlmmc.deseasonalize_harmonics)
assert not any(name in sys.modules for name in ['torch','numba','affine','cmdstanpy'])
try:
    dlmmc.compile_models(models=['vanilla_ar1'])
except ImportError as exc:
    assert 'dlmmc[stan]' in str(exc)
else:
    raise AssertionError('Missing Stan dependency was not reported')
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_optional_api_and_source_contract():
    from dlmmc import MonthlyENSO, sample_enso_lag
    from dlmmc.enso_lag import MonthlyENSO as implementation
    from dlmmc.sampling import _source_identity
    assert MonthlyENSO is implementation
    assert callable(sample_enso_lag)
    sources = _source_identity()
    assert {'api.py','input_contract.py','preprocessing.py','__init__.py'} <= sources.keys()
    assert all(len(value) == 64 for value in sources.values())
