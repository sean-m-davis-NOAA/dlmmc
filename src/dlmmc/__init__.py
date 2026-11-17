"""Dynamical linear models with optional affine tensor and Stan backends."""

from .api import compile_models
from .fitting import fit, AffineSettings, StanSettings, summarize_trends, summarize_components
from .mpi import fit_grid_mpi
from .preprocessing import deseasonalize_harmonics
from .components import Model, Trend, Level, Harmonics, Regression, AR, build_state_space

def fit_grid(*args, **kwargs):
    """Independent tensorized affine fit; explicit dtype/backend, bounded storage."""
    profile = kwargs.pop('profile', 'qualification')
    if profile == 'production':
        from .production import fit_production
        return fit_production(*args, **kwargs)
    if profile != 'qualification':
        raise ValueError('Choose production or qualification explicitly')
    from .sampling import fit_grid as implementation
    from .input_contract import validate_no_seasonal_input, record_input_policy
    data = args[0] if args else kwargs.get('data')
    policy = validate_no_seasonal_input(data, kwargs.get('model', 'vanilla_ar1'))
    fit = implementation(*args, **kwargs)
    record_input_policy(fit, policy)
    return fit


def prepare_data(*args, **kwargs):
    from .prepare import prepare_data as implementation
    from .input_contract import validate_no_seasonal_input
    validate_no_seasonal_input(args[0] if args else kwargs.get('data'), kwargs.get('model', 'vanilla_ar1'))
    return implementation(*args, **kwargs)


def resume_fit(*args, **kwargs):
    from .sampling import resume_fit as implementation
    from .sampling import load_fit
    from .input_contract import checkpoint_input_policy
    checkpoint_input_policy(load_fit(args[0] if args else kwargs['output_dir']))
    return implementation(*args, **kwargs)


def load_fit(*args, **kwargs):
    from .sampling import load_fit as implementation
    return implementation(*args, **kwargs)


def diagnose_fit(*args, **kwargs):
    from .inference import diagnose_fit as implementation
    return implementation(*args, **kwargs)


def reconstruct_fit(*args, **kwargs):
    from .reconstruction import reconstruct_fit as implementation
    return implementation(*args, **kwargs)


def fork_cells(*args, **kwargs):
    from .independent import fork_cells as implementation
    from .input_contract import checkpoint_input_policy, record_input_policy
    policy = checkpoint_input_policy(args[0] if args else kwargs['fit'])
    fit = implementation(*args, **kwargs)
    record_input_policy(fit, policy)
    return fit


def conditional_trend_probabilities(*args, **kwargs):
    """CPU float64 Gaussian sign probabilities at fixed parameters."""
    from .probabilities import conditional_trend_probabilities as implementation
    return implementation(*args, **kwargs)


__all__ = ["fit", "AffineSettings", "StanSettings", "summarize_trends", "summarize_components", "fit_grid", "fit_grid_mpi", "compile_models", "prepare_data", "resume_fit", "load_fit", "diagnose_fit", "reconstruct_fit", "fork_cells", "conditional_trend_probabilities", "deseasonalize_harmonics"]

# Importing the base package must not require Torch, Numba, or affine. Resolve
# explicit optional APIs only when requested; this changes no model equations.
_OPTIONAL_APIS = {
    'SpatialPrior': 'spatial_api', 'LagPrior': 'spatial_api', 'sample_spatial_prior': 'spatial_api', 'fit_spatial': 'spatial_api',
    'ExecutionProfile': 'profiles', 'TESTING_PROFILE': 'profiles', 'PRODUCTION_PROFILE': 'profiles', 'QUALIFICATION_PROFILE': 'profiles',
    'conditional_trend_moments': 'conditional_trends', 'fit_production': 'production', 'resume_production': 'production',
    'diagnose_production': 'production', 'summarize_production_trends': 'production_trends',
    'GaussianSpatialPrior': 'spatial', 'test_lengths_from_correlation': 'spatial',
    'sample_spatial': 'spatial_sampling', 'resume_spatial': 'spatial_sampling',
    'load_spatial': 'spatial_sampling', 'diagnose_spatial': 'spatial_diagnostics',
    'MonthlyENSO': 'enso_lag', 'sample_enso_lag': 'lag_sampling',
    'resume_enso_lag': 'lag_sampling', 'load_enso_lag': 'lag_sampling',
    'diagnose_enso_lag': 'lag_diagnostics', 'reconstruct_enso_lag': 'lag_reconstruction',
}
__all__ += list(_OPTIONAL_APIS) + ["Model","Trend","Level","Harmonics","Regression","AR","build_state_space"]


def __getattr__(name):
    if name not in _OPTIONAL_APIS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module
    value = getattr(import_module('.' + _OPTIONAL_APIS[name], __name__), name)
    globals()[name] = value
    return value
