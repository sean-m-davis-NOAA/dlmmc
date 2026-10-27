import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dlmmc import api
from dlmmc.api import _MODEL_SPECS, _prepare_regressors, _resolve_model, _resolve_priors


def test_model_names_accept_public_and_stan_forms():
    assert _resolve_model('vanilla-ar1')[0] == 'vanilla_ar1'
    assert _resolve_model('dlm_vanilla_ar1')[0] == 'vanilla_ar1'
    assert set(_MODEL_SPECS) == {
        'vanilla_ar1',
        'vanilla_ar2',
        'noregs_ar1',
        'dynregs_ar1',
        'vanilla_ar1_noseasonal',
        'vanilla_ar2_noseasonal',
    }


def test_prior_defaults_match_current_model_defaults():
    spec = _resolve_model('vanilla_ar1')[1]
    priors = _resolve_priors(spec, 3, None, {})
    assert priors == {
        'S': 10.0,
        'sigma_trend_prior': 1e-4,
        'sigma_seas_prior': 0.01,
        'sigma_AR_prior': 0.5,
    }


def test_prior_keyword_and_mapping_overrides():
    spec = _resolve_model('dynregs_ar1')[1]
    priors = _resolve_priors(
        spec,
        2,
        {'sigma_reg_prior': 0.02},
        {'sigma_trend_prior': 2e-4},
    )
    assert priors['sigma_trend_prior'] == 2e-4
    assert priors['sigma_reg_prior'] == [0.02, 0.02]


def test_duplicate_prior_specification_is_rejected():
    spec = _resolve_model('vanilla_ar1')[1]
    with pytest.raises(ValueError, match='either directly or in priors'):
        _resolve_priors(spec, 1, {'S': 5.0}, {'S': 10.0})


def test_model_inapplicable_prior_is_rejected():
    spec = _resolve_model('noregs_ar1')[1]
    with pytest.raises(ValueError, match='not used by this model'):
        _resolve_priors(spec, 0, {'sigma_reg_prior': 1e-4}, {})


def test_xarray_regressors_interpolate_to_data_time():
    target_time = xr.DataArray(
        pd.date_range('2000-01-01', periods=3, freq='2D'),
        dims=('time',),
        name='time',
    )
    source_time = pd.date_range('2000-01-01', periods=5, freq='D')
    regressors = xr.Dataset(
        {'index': ('time', np.arange(5.0))},
        coords={'time': source_time},
    )

    values, names = _prepare_regressors(regressors, target_time, 'static')

    np.testing.assert_allclose(values[:, 0], [0.0, 2.0, 4.0])
    assert names == ['index']


def test_fit_forwards_per_chain_initializers_and_returns_labeled_draws(monkeypatch):
    n_time = 5
    n_chains = 2
    n_draws = 3
    initial_values = [{'sigma_AR': 0.1}, {'sigma_AR': 0.2}]
    data = xr.DataArray(
        np.linspace(1.0, 2.0, n_time),
        dims=('time',),
        coords={'time': pd.date_range('2000-01-01', periods=n_time, freq='MS')},
    )

    class FakeFit:
        def draws_xr(self, vars):
            coords = {
                'chain': np.arange(1, n_chains + 1),
                'draw': np.arange(1, n_draws + 1),
                'trend_dim_0': np.arange(n_time),
            }
            values = np.ones((n_chains, n_draws))
            result = xr.Dataset(
                {
                    'trend': (('chain', 'draw', 'trend_dim_0'), np.ones((n_chains, n_draws, n_time))),
                    'slope': (('chain', 'draw', 'trend_dim_0'), np.ones((n_chains, n_draws, n_time))),
                    'ar': (('chain', 'draw', 'trend_dim_0'), np.ones((n_chains, n_draws, n_time))),
                    'seasonal': (('chain', 'draw', 'trend_dim_0'), np.ones((n_chains, n_draws, n_time))),
                    'sigma_trend': (('chain', 'draw'), values),
                    'sigma_seas': (('chain', 'draw'), values),
                    'sigma_AR': (('chain', 'draw'), values),
                    'rhoAR1': (('chain', 'draw'), values),
                },
                coords=coords,
            )
            return result[vars]

        def method_variables(self):
            return {'divergent__': np.zeros((n_draws, n_chains, 1))}

    class FakeModel:
        def sample(self, **kwargs):
            self.kwargs = kwargs
            return FakeFit()

    fake_model = FakeModel()
    monkeypatch.setattr(api, '_get_compiled_model', lambda model_name, spec, cache_dir=None: fake_model)

    posterior = api.fit(
        data,
        model='noregs_ar1',
        chains=n_chains,
        warmup=0,
        samples=n_draws,
        inits=initial_values,
        show_progress=False,
    )

    np.testing.assert_array_equal(fake_model.kwargs['data']['stddev'], np.full(n_time, 1e-20))
    assert fake_model.kwargs['inits'] == initial_values
    assert posterior['trend'].dims == ('chain', 'draw', 'time')
    assert np.array_equal(posterior['time'].values, data['time'].values)
