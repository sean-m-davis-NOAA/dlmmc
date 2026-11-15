import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.requirements import Requirement


az = pytest.importorskip('arviz', minversion='1.3')
ROOT = Path(__file__).resolve().parents[1]
SCALAR_PARAMETERS = ['sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1']


# Notebook model equations and diagnostics now use shared public APIs. Preserve
# the original chain/draw ordering and labeled-regressor fixtures against the
# shared Stan compatibility helper instead of extracting removed notebook cells.
from dlmmc.stan_diagnostics import summarize_stan


@pytest.fixture(params=[False, True], ids=['chain-first', 'draw-first'])
def posterior_draws(request):
    rng = np.random.default_rng(42)
    regressors = ['ENSO', 'SOLAR', 'QBO']
    variables = {
        name: (('chain', 'draw'), rng.normal(size=(4, 500)))
        for name in SCALAR_PARAMETERS + ['rhoAR2']
    }
    variables.update({
        name: (('chain', 'draw', 'regressor'), rng.normal(size=(4, 500, len(regressors))))
        for name in ('beta', 'sigma_reg')
    })
    result = xr.Dataset(
        variables,
        coords={
            'chain': [2, 4, 6, 8],
            'draw': np.arange(1000, 1500),
            'regressor': regressors,
        },
        attrs={'divergent_transitions': 0},
    )
    return result.transpose('draw', 'chain', 'regressor') if request.param else result


def _assert_summary(summary, posterior, names):
    expected_data = az.from_dict(
        data={
            'posterior': {
                name: posterior[name].transpose('chain', 'draw', ...).to_numpy()
                for name in names
            }
        },
        sample_dims=['chain', 'draw'],
        coords={name: coordinate.to_numpy() for name, coordinate in posterior.coords.items()},
        dims={
            name: [dim for dim in posterior[name].dims if dim not in ('chain', 'draw')]
            for name in names
        },
    )
    expected = az.summary(expected_data, var_names=names, kind='all', round_to=3)
    pd.testing.assert_frame_equal(summary, expected)
    diagnostics = summary[['r_hat', 'ess_bulk', 'ess_tail']].to_numpy(dtype=float)
    assert np.isfinite(diagnostics).all()


def test_tutorial_diagnostic_cell(posterior_draws):
    original = posterior_draws.copy(deep=True)
    displayed = []
    namespace = {'az': az, 'posterior': posterior_draws, 'display': displayed.append}

    displayed.append(summarize_stan(posterior_draws, SCALAR_PARAMETERS))

    assert len(displayed) == 1
    _assert_summary(displayed[0], posterior_draws, SCALAR_PARAMETERS)
    xr.testing.assert_identical(posterior_draws, original)


@pytest.mark.parametrize(
    'parameters',
    [
        SCALAR_PARAMETERS,
        SCALAR_PARAMETERS + ['beta'],
        SCALAR_PARAMETERS + ['rhoAR2', 'beta'],
        SCALAR_PARAMETERS + ['sigma_reg'],
    ],
    ids=['noregs-ar1', 'vanilla-ar1', 'vanilla-ar2', 'dynregs-ar1'],
)
def test_validation_diagnostic_code(posterior_draws, parameters):
    posterior = posterior_draws[parameters]
    original = posterior.copy(deep=True)
    displayed = []
    namespace = {
        'az': az,
        'fit': posterior,
        'true_values': dict.fromkeys(parameters, 0),
        'regressor_names': posterior_draws.regressor.to_numpy().tolist(),
        'display': displayed.append,
    }

    displayed.append(summarize_stan(posterior, parameters))

    assert len(displayed) == 1
    _assert_summary(displayed[0], posterior, parameters)
    xr.testing.assert_identical(posterior, original)


def test_diagnostics_extra_accepts_modern_arviz():
    import tomllib

    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    requirement = Requirement(project['optional-dependencies']['diagnostics'][0])

    assert requirement.name == 'arviz'
    assert az.__version__ in requirement.specifier
    assert '0.23.4' not in requirement.specifier
    assert project['requires-python'] == '>=3.10'


def test_stan_summary_rejects_dependent_walker_axis(posterior_draws):
    with pytest.raises(ValueError, match='walkers are not chains'):
        summarize_stan(posterior_draws.rename(chain='walker'), SCALAR_PARAMETERS)
