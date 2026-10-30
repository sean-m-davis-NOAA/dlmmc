import tempfile
from pathlib import Path

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


@pytest.fixture
def fake_sampler(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))

    class FakeFit:
        def draws_xr(self, vars):
            assert fake_model.output_dir.is_dir()
            if fake_model.failure_stage == 'draws':
                raise fake_model.error
            n_time = fake_model.kwargs['data']['N']
            n_chains = fake_model.kwargs['chains']
            n_draws = fake_model.kwargs['iter_sampling']
            coords = {
                'chain': np.arange(1, n_chains + 1),
                'draw': np.arange(1, n_draws + 1),
                'trend_dim_0': np.arange(n_time),
            }
            values = np.ones((n_chains, n_draws))
            result = xr.Dataset(
                {
                    name: (('chain', 'draw', 'trend_dim_0'), np.ones((n_chains, n_draws, n_time)))
                    for name in ('trend', 'slope', 'ar', 'seasonal')
                }
                | {
                    name: (('chain', 'draw'), values)
                    for name in ('sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1')
                },
                coords=coords,
            )
            return result[vars]

        def method_variables(self):
            assert fake_model.output_dir.is_dir()
            if fake_model.failure_stage == 'diagnostics':
                raise fake_model.error
            return {'divergent__': np.zeros((fake_model.kwargs['iter_sampling'], fake_model.kwargs['chains']))}

    class FakeModel:
        failure_stage = None
        error = RuntimeError('simulated Stan failure')

        def sample(self, **kwargs):
            self.kwargs = kwargs
            self.output_dir = Path(kwargs.get('output_dir') or tmp_path / 'cmdstan-default')
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / 'chain.csv').write_text('posterior draws\n')
            (self.output_dir / 'chain-stdout.txt').write_text('Stan console output\n')
            if self.failure_stage == 'sample':
                raise self.error
            return FakeFit()

    fake_model = FakeModel()
    monkeypatch.setattr(api, '_get_compiled_model', lambda model_name, spec, cache_dir=None: fake_model)
    return fake_model


@pytest.mark.parametrize('output_mode', ['default', 'none', 'explicit'])
def test_fit_forwards_per_chain_initializers_and_returns_labeled_draws(fake_sampler, tmp_path, output_mode):
    n_time = 5
    n_chains = 2
    n_draws = 3
    initial_values = [{'sigma_AR': 0.1}, {'sigma_AR': 0.2}]
    data = xr.DataArray(
        np.linspace(1.0, 2.0, n_time),
        dims=('time',),
        coords={'time': pd.date_range('2000-01-01', periods=n_time, freq='MS')},
    )

    options = {}
    if output_mode == 'none':
        options['output_dir'] = None
    elif output_mode == 'explicit':
        output_dir = tmp_path / 'user-output'
        output_dir.mkdir()
        (output_dir / 'unrelated.txt').write_text('keep this file\n')
        options['output_dir'] = output_dir
    posterior = api.fit(
        data,
        model='noregs_ar1',
        chains=n_chains,
        warmup=0,
        samples=n_draws,
        inits=initial_values,
        show_progress=False,
        **options,
    )

    np.testing.assert_array_equal(fake_sampler.kwargs['data']['stddev'], np.full(n_time, 1e-20))
    assert fake_sampler.kwargs['inits'] == initial_values
    assert posterior['trend'].dims == ('chain', 'draw', 'time')
    assert np.array_equal(posterior['time'].values, data['time'].values)
    np.testing.assert_array_equal(posterior['trend'].values, np.ones((n_chains, n_draws, n_time)))
    assert posterior.attrs['divergent_transitions'] == 0
    if output_mode == 'explicit':
        assert (output_dir / 'chain.csv').is_file()
        assert (output_dir / 'unrelated.txt').read_text() == 'keep this file\n'
    else:
        assert fake_sampler.output_dir.parent == tmp_path
        assert not fake_sampler.output_dir.exists()


@pytest.mark.parametrize('failure_stage', ['sample', 'draws', 'diagnostics'])
@pytest.mark.parametrize('explicit_output', [False, True])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_fit_preserves_failed_output_and_original_error(
    fake_sampler, tmp_path, caplog, failure_stage, explicit_output, error_type
):
    fake_sampler.failure_stage = failure_stage
    fake_sampler.error = error_type('simulated Stan failure')
    options = {'output_dir': tmp_path / 'user-output'} if explicit_output else {}

    with pytest.raises(error_type, match='simulated Stan failure') as caught:
        api.fit(
            xr.DataArray(np.arange(5.0), dims='time'),
            model='noregs_ar1',
            chains=1,
            warmup=0,
            samples=2,
            **options,
        )

    assert caught.value is fake_sampler.error
    assert (fake_sampler.output_dir / 'chain.csv').is_file()
    assert (fake_sampler.output_dir / 'chain-stdout.txt').is_file()
    if not explicit_output:
        assert str(fake_sampler.output_dir) in caplog.text


def test_repeated_fits_do_not_accumulate_temporary_output(fake_sampler, tmp_path):
    directories = []
    for _ in range(3):
        posterior = api.fit(
            xr.DataArray(np.arange(5.0), dims='time'),
            model='noregs_ar1',
            chains=1,
            warmup=0,
            samples=2,
        )
        directories.append(fake_sampler.output_dir)
        assert list(tmp_path.iterdir()) == []
        np.testing.assert_array_equal(posterior['trend'].values, np.ones((1, 2, 5)))
    assert len(set(directories)) == 3


@pytest.mark.parametrize(
    'source_name',
    [
        'dlm_vanilla_ar1',
        'dlm_vanilla_ar2',
        'dlm_dynregs_ar1',
        'dlm_noregs_ar1',
        'dlm_vanilla_ar1_noseasonal',
        'dlm_vanilla_ar2_noseasonal',
    ],
)
def test_kalman_filter_arrays_are_not_transformed_parameters(source_name):
    stan_source = getattr(api._stan_models, source_name)

    assert 'transformed parameters' not in stan_source
    assert stan_source.count('array[N+1] vector[nx] x_hat;') == 2
    assert '\n        array[N+1] vector[nx] x_hat;' in stan_source
    assert '\n    array[N+1] vector[nx] x_hat;' not in stan_source


def test_fit_csv_omits_local_kalman_scratch_arrays(tmp_path):
    from cmdstanpy import cmdstan_path

    try:
        cmdstan_path()
    except ValueError:
        pytest.skip('CmdStan is not installed in this environment')

    time = np.arange(12, dtype=float)
    series = xr.DataArray(
        0.3 + 0.01 * time + 0.1 * np.sin(2 * np.pi * time / 12),
        dims=('time',),
        coords={'time': time},
    )
    posterior = api.fit(
        series,
        model='noregs_ar1',
        chains=1,
        warmup=1,
        samples=1,
        seed=17,
        show_progress=False,
        adapt_engaged=False,
        output_dir=tmp_path,
    )

    csv_path = next(tmp_path.glob('*.csv'))
    header = next(line for line in csv_path.open() if line.startswith('lp__,'))
    columns = header.rstrip().split(',')
    scratch_prefixes = (
        'x_hat.',
        'x_bar.',
        'C_hat.',
        'C_y.',
        'K.',
        'C_bar.',
        'v.',
        'x.',
        'x_realization.',
        'x_ubar.',
        'x_realization_hat.',
        'y_realization.',
        'v_realization.',
        'r.',
        'r_realization.',
        'L.',
        'M.',
        'C_twidle.',
        'x_twidle.',
        'x_realization_twidle.',
        'G.',
        'W.',
        'C.',
    )

    assert not any(column.startswith(scratch_prefixes) for column in columns)
    assert {'trend.1', 'slope.1', 'seasonal.1', 'ar.1'} <= set(columns)
    assert {'trend', 'slope', 'seasonal', 'ar', 'sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1'} <= set(posterior.data_vars)
