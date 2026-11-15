"""Conditional Gaussian sign probabilities at fixed, supported ENSO lags."""
from dataclasses import replace
import numpy as np
import torch
from .enso_lag import ENSOLagTarget
from .probabilities import compile_trend_probabilities,conditional_trend_probabilities


def _shift(prepared,predictor,sigma_prior,lag_prior,unconstrained):
    p=prepared
    if p.y.dtype!=torch.float64 or p.y.device.type!='cpu':
        raise ValueError('Lag probability integration requires explicit CPU float64')
    u=np.asarray(unconstrained)
    if u.dtype!=np.float64 or u.ndim!=3 or u.shape[1:]!=(p.cells,len(p.parameter_names)+1):
        raise ValueError('Expected float64 selected joint lag coordinates')
    target=ENSOLagTarget(p,sigma_prior,lag_prior,predictor,backend='torch')
    tensor=torch.from_numpy(u.copy());_,_,valid=target.lag_values(tensor)
    if not bool(valid.all()):raise ValueError('Strict supported lags required')
    return replace(p,regressors=target.regressors(tensor)),u[...,:-1]


def compile_lag_trend_probabilities(prepared,predictor,sigma_prior,lag_prior,unconstrained,contrasts):
    shifted,u=_shift(prepared,predictor,sigma_prior,lag_prior,unconstrained)
    return compile_trend_probabilities(shifted,u,contrasts)


def conditional_lag_trend_probabilities(prepared,predictor,sigma_prior,lag_prior,unconstrained,contrasts):
    """Integrate state uncertainty including cross-time covariance, without floors.

    This fixed-parameter calculation does not certify a posterior window.
    """
    shifted,u=_shift(prepared,predictor,sigma_prior,lag_prior,unconstrained)
    return conditional_trend_probabilities(shifted,u,contrasts)
