"""Offline CPU float64 Gaussian contrast prototype, with explicit compilation.

Uses the same full Joseph filter and backward conditional innovations as the
Torch reference. It neither samples parameters nor drops retained draws.
"""
import time
import numpy as np
from numba import njit, prange, typeof
from scipy.special import log_ndtr


@njit(cache=True, fastmath=False)
def _contrast_one(y, variance, observed, F, G, Q, C, contrasts):
    n, d = F.shape
    eye = np.eye(d)
    m = np.zeros(d)
    P = C.copy()
    means = np.empty((n, d))
    covs = np.empty((n, d, d))
    for t in range(n):
        m = G @ m
        P = G @ P @ G.T + Q
        if observed[t]:
            f = F[t]
            pf = P @ f
            innovation = f @ pf + variance[t]
            if not np.isfinite(innovation) or innovation <= 0:
                raise ValueError('Invalid innovation variance; no floor')
            gain = pf / innovation
            m = m + gain * (y[t] - f @ m)
            A = eye - np.outer(gain, f)
            P = A @ P @ A.T + np.outer(gain, gain) * variance[t]
        means[t] = m
        covs[t] = P
    smooth_m = means.copy()
    smooth_c = covs.copy()
    gains = np.empty((n-1, d, d))
    innovations = np.empty((n-1, d, d))
    for t in range(n-2, -1, -1):
        R = G @ covs[t] @ G.T + Q
        J = np.ascontiguousarray(np.linalg.solve(R, G @ covs[t]).T)
        smooth_m[t] = means[t] + J @ (smooth_m[t+1] - G @ means[t])
        A = eye - J @ G
        conditional = A @ covs[t] @ A.T + J @ Q @ J.T
        smooth_c[t] = conditional + J @ smooth_c[t+1] @ J.T
        gains[t] = J
        innovations[t] = conditional
    result = np.empty((len(contrasts), 2))
    for k in range(len(contrasts)):
        a, b = contrasts[k]
        weight = np.zeros(d)
        weight[0] = -1.
        value = 0.
        for t in range(a, b):
            value += np.sum(weight * (innovations[t] @ weight))
            weight = gains[t].T @ weight
        weight[0] += 1.
        value += np.sum(weight * (smooth_c[b] @ weight))
        result[k, 0] = smooth_m[b, 0] - smooth_m[a, 0]
        result[k, 1] = value
    return result


@njit(cache=True, parallel=True, fastmath=False)
def _contrast_batch(y, variance, observed, F, G, Q, C, contrasts):
    result = np.empty((len(F), len(contrasts), 2))
    for i in prange(len(F)):
        cell = i % len(y)
        result[i] = _contrast_one(y[cell], variance[cell], observed[cell], F[i], G[i], Q[i], C[i], contrasts)
    return result


def _arrays(y, variance, observed, F, G, Q, C, contrasts):
    arrays = [np.asarray(x) for x in (y, variance, observed, F, G, Q, C)]
    y, variance, observed, F, G, Q, C = arrays
    if any(x.dtype != np.float64 for j, x in enumerate(arrays) if j != 2):
        raise ValueError('Explicit float64 inputs required')
    if observed.dtype != bool or y.shape != variance.shape or y.shape != observed.shape:
        raise ValueError('Matching data, variance and Boolean mask required')
    n, d = F.shape[-2:]
    if y.ndim not in (1, 2) or y.shape[-1] != n or F.ndim < 2:
        raise ValueError('Invalid observation shape')
    shape = F.shape[:-2]
    if any(x.shape != (*shape, d, d) for x in (G, Q, C)):
        raise ValueError('Matrices must have identical explicit leading shapes')
    cells = 1 if y.ndim == 1 else len(y)
    if cells > 1 and (not shape or shape[-1] != cells):
        raise ValueError('Last leading matrix axis must match cells')
    if not (np.isfinite(y[observed]).all() and np.isfinite(variance[observed]).all() and (variance[observed] >= 0).all()):
        raise ValueError('Invalid observed input')
    if not all(np.isfinite(x).all() for x in (F, G, Q, C)):
        raise ValueError('Nonfinite model matrices')
    pairs = np.asarray(contrasts)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer) or not np.all((pairs[:,0] >= 0) & (pairs[:,0] < pairs[:,1]) & (pairs[:,1] < n)):
        raise ValueError('Ordered integer contrast indices required')
    # Sanitize masked observations before any compiled arithmetic.
    packed = [np.where(observed, y, 0.).reshape(-1,n), np.where(observed, variance, 0.).reshape(-1,n), observed.reshape(-1,n), F.reshape(-1,n,d), *[x.reshape(-1,d,d) for x in (G,Q,C)], pairs.astype(np.int64)]
    return shape, tuple(np.ascontiguousarray(x) for x in packed)


def compile_contrasts(*args):
    shape, arrays = _arrays(*args)
    signature = tuple(typeof(x) for x in arrays)
    start = time.perf_counter()
    _contrast_batch.compile(signature)
    return dict(compilation_or_cache_load_seconds=time.perf_counter()-start,
                signature=str(signature), leading_shape=shape, fastmath=False)


def contrast_statistics(*args):
    shape, arrays = _arrays(*args)
    result = _contrast_batch(*arrays).reshape(*shape, len(arrays[-1]), 2)
    if not np.isfinite(result).all() or np.any(result[...,1] < 0):
        raise FloatingPointError('Invalid conditional moments; no clipping')
    return result


def positive_trend_probabilities(*args):
    moments = contrast_statistics(*args)
    mean, variance = moments[...,0], moments[...,1]
    positive = variance > 0
    denominator = np.sqrt(np.where(positive, variance, 1.))
    return np.where(positive, np.exp(log_ndtr(mean/denominator)), (mean > 0).astype(float))
