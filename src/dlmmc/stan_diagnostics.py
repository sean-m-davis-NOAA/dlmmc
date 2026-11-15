"""Optional named-axis summaries for the historical Stan fit API.

Affine walkers are dependent and must use diagnose_fit/diagnose_enso_lag instead.
This helper accepts independent Stan chains with explicit chain/draw axes only.
"""


def summarize_stan(posterior, parameter_names):
    import arviz as az
    names = list(parameter_names)
    if not names or any(name not in posterior for name in names):
        raise ValueError('Explicit existing Stan parameter names required')
    if 'walker' in posterior.dims or not {'chain', 'draw'} <= set(posterior.dims):
        raise ValueError('Independent Stan chain/draw axes required; walkers are not chains')
    if posterior.sizes['chain'] < 4:
        raise ValueError('Four independent Stan chains required for this diagnostic summary')
    data = az.from_dict(
        data={'posterior': {name: posterior[name].transpose('chain', 'draw', ...).to_numpy()
                            for name in names}},
        sample_dims=['chain', 'draw'],
        coords={name: value.to_numpy() for name, value in posterior.coords.items()},
        dims={name: [dim for dim in posterior[name].dims if dim not in ('chain', 'draw')]
              for name in names},
    )
    return az.summary(data, var_names=names, kind='all', round_to=3)
