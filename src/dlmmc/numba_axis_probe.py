"""Experimental scheduling probes; not selected by production dispatch.

Only the independent iteration schedule changes. Every evaluation calls the
unchanged qualified structured Joseph filter in float64 without fastmath.
"""
import numpy as np
from numba import njit, prange
from .numba_prototype import _one_structured


@njit(cache=True, parallel=True, fastmath=False)
def walker_outer(y, variance, observed, F, G, Q, C):
    count = F.shape[0]
    cells = y.shape[0]
    walkers = count // cells
    values = np.empty(count)
    failures = np.empty(count, dtype=np.int64)
    for w in prange(walkers):
        for g in range(cells):
            i = w * cells + g
            values[i], failures[i] = _one_structured(
                y[g], variance[g], observed[g], F[i], G[i], Q[i], C[i])
    return values, failures


@njit(cache=True, parallel=True, fastmath=False)
def panel_outer(y, variance, observed, F, G, Q, C):
    count = F.shape[0]
    cells = y.shape[0]
    walkers = count // cells
    values = np.empty(count)
    failures = np.empty(count, dtype=np.int64)
    for g in prange(cells):
        for w in range(walkers):
            i = w * cells + g
            values[i], failures[i] = _one_structured(
                y[g], variance[g], observed[g], F[i], G[i], Q[i], C[i])
    return values, failures
