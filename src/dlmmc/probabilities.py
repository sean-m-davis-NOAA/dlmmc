"""Conditional Gaussian trend probabilities, without rare-event simulation.

This explicitly CPU/float64 path integrates out the states, preserving the
same posterior target. It does not certify parameter-window diagnostics.
"""
import numpy as np
import torch
from .tensor_target import model_components
from .numba_conditional import compile_contrasts, positive_trend_probabilities


def _arguments(prepared, unconstrained, contrasts):
    p = prepared
    if p.y.dtype != torch.float64 or p.y.device.type != 'cpu':
        raise ValueError('Conditional probability path requires explicit CPU float64')
    u = np.asarray(unconstrained)
    if u.dtype != np.float64 or u.ndim != 3 or u.shape[1:] != (p.cells, len(p.parameter_names)):
        raise ValueError('Expected float64 [selected draw, cell, parameter]')
    with torch.no_grad():
        F, G, Q, C, _, valid = model_components(torch.from_numpy(u.copy()), *p.arguments,
            p.model, S=p.priors['S'], sampling=p.sampling)
    if not bool(valid.all()):
        raise ValueError('All selected parameters must be in support')
    return (*[v.numpy() for v in (p.y, p.variance, p.observed, F, G, Q, C)], contrasts)


def compile_trend_probabilities(prepared, unconstrained, contrasts):
    """Explicitly precompile/load the exact specialization before timing."""
    return compile_contrasts(*_arguments(prepared, unconstrained, contrasts))


def conditional_trend_probabilities(prepared, unconstrained, contrasts):
    """Return P(trend[b] > trend[a] | parameters,y) for ordered index pairs.

    Covariance includes cross-time dependence. Positive observation scale and
    an additive offset cancel from the sign. No probability floor is applied.
    Use only diagnosed retained windows for posterior averaging; fixed-parameter
    calls are also useful for independent Gaussian validation.
    """
    return positive_trend_probabilities(*_arguments(prepared, unconstrained, contrasts))
