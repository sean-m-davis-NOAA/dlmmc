"""Offline, low-power calibration checks; never establishes sampler convergence.

Keep independent datasets and independent ensemble endpoints distinct. In
particular, walkers within an ensemble are not independent SBC rank draws.
"""
from functools import lru_cache
from math import factorial, prod

import numpy as np
from scipy.stats import binomtest, t as student_t


@lru_cache(maxsize=16)
def _multinomial_multiplicities(n, bins):
    """Integer multiplicities avoid floating-point ambiguity in exact ties."""
    def compositions(total, width):
        if width == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, width - 1):
                    yield (first,) + rest
    numerator = factorial(n)
    return tuple(numerator // prod(factorial(v) for v in c)
                 for c in compositions(n, bins))


def exact_uniform_rank_test(counts):
    """Probability-ordering multinomial test, restricted to the small smoke."""
    raw = np.asarray(counts)
    if (raw.ndim != 1 or not 2 <= raw.size <= 5 or not np.isfinite(raw).all()
            or np.any(raw < 0) or np.any(raw != np.floor(raw))):
        raise ValueError('Nonnegative integer bin counts required')
    counts = [int(v) for v in raw]
    n = sum(counts)
    if not 1 <= n <= 20:
        raise ValueError('Only the bounded20-dataset smoke is supported')
    observed = factorial(n) // prod(factorial(v) for v in counts)
    mass = sum(v for v in _multinomial_multiplicities(n, len(counts)) if v <= observed)
    return mass / len(counts) ** n


def randomized_rank(truth, independent_draws, rng):
    """Rank against one draw from each independent ensemble, with exact ties."""
    draws = np.asarray(independent_draws, dtype=float)
    if draws.ndim != 1 or len(draws) != 4 or not np.isfinite(draws).all() or not np.isfinite(truth):
        raise ValueError('Exactly four finite independent ensemble endpoints required')
    ties = int(np.sum(draws == truth))
    return int(np.sum(draws < truth)) + int(rng.integers(ties + 1)), ties


def posterior_coverage_membership(draws, truth, *, block_steps, alpha):
    """Central95% interval membership with dependence-aware endpoint MC bands.

    Input is [independent_run, ordered_step, walker]. Every draw is retained,
    walkers stay together, and at least five complete blocks/run are required.
    Mixing and block-length adequacy must be checked by the caller first.
    """
    x = np.asarray(draws, dtype=float)
    if (x.ndim != 3 or x.shape[0] < 4 or x.shape[2] < 1
            or not np.isfinite(x).all() or not np.isfinite(truth) or not 0 < alpha < 1):
        raise ValueError('Finite draws, truth and valid alpha required')
    r, n, w = x.shape
    if not isinstance(block_steps, (int, np.integer)) or block_steps < 1 or n % block_steps or n // block_steps < 5:
        raise ValueError('At least five complete blocks per run; no discarded suffix')
    blocks = x.reshape(r * (n // block_steps), block_steps * w)
    endpoints = np.quantile(x, [.025, .975])
    block_endpoints = np.quantile(blocks, [.025, .975], axis=1)
    se = block_endpoints.std(axis=1, ddof=1) / np.sqrt(len(blocks))
    half = student_t.ppf(1 - alpha / 2, len(blocks) - 1) * se
    low, high = endpoints - half, endpoints + half
    # A degenerate sample does not establish zero MC error.
    if np.any(se == 0):
        membership = 'INCONCLUSIVE'
    elif high[0] <= truth <= low[1]:
        membership = 'COVERED'
    elif truth < low[0] or truth > high[1]:
        membership = 'NOT_COVERED'
    else:
        membership = 'INCONCLUSIVE'
    return dict(membership=membership, posterior_interval=endpoints.tolist(),
                endpoint_mcse=se.tolist(), endpoint_mc_lower=low.tolist(),
                endpoint_mc_upper=high.tolist(), endpoint_alpha=alpha,
                blocks=len(blocks), block_steps=int(block_steps))


def coverage_smoke_test(memberships, *, alpha):
    """Keep ambiguous memberships in the20-dataset family; never drop them."""
    if len(memberships) != 20 or not 0 < alpha < 1:
        raise ValueError('Exactly20 prespecified datasets and valid alpha required')
    if any(v not in {'COVERED', 'NOT_COVERED', 'INCONCLUSIVE'} for v in memberships):
        raise ValueError('Unknown coverage membership')
    covered = memberships.count('COVERED')
    ambiguous = memberships.count('INCONCLUSIVE')
    tests = [binomtest(k, 20, .95) for k in range(covered, covered + ambiguous + 1)]
    status = ('FAIL' if all(t.pvalue < alpha for t in tests) else
              'INCONCLUSIVE' if ambiguous else 'PASS')
    bounds = [t.proportion_ci(confidence_level=1-alpha, method='exact') for t in tests]
    return dict(status=status, covered=covered, ambiguous=ambiguous, datasets=20,
                possible_pvalue_range=[min(t.pvalue for t in tests), max(t.pvalue for t in tests)],
                coverage_interval=[min(b.low for b in bounds), max(b.high for b in bounds)],
                alpha=alpha, scope='Low-power20-dataset debugging, not release calibration')
