"""Performance-only sampler screening; never an inference certificate."""
import numpy as np

from .inference import window_diagnostics


def walker_candidates(moved_dimension):
    """Half-ensemble counts with a full-rank dimension guard."""
    if not isinstance(moved_dimension, int) or moved_dimension < 1:
        raise ValueError('Positive integer moved dimension required')
    base = 1 << (max(64, moved_dimension + 1) - 1).bit_length()
    return [base, 2 * base, 4 * base]


def tuning_window(values, *, target_ess=1000):
    """Screen four ensembles using the existing MC-aware stability estimator.

    All parameter/derived means, widths and tail functions must reach the
    tuning target. Rank, complete-block, two-block-scale variance and MC-aware
    half/full stability checks are unchanged. The ordinary ESS2000 scientific
    result is retained separately. Two successive eligible windows and fresh
    seed confirmation remain the caller's responsibility.
    """
    if target_ess < 1000:
        raise ValueError('The frozen tuning target is at least 1000 ESS')
    report = window_diagnostics(values)
    mc = report.get('mc_stability', {})
    precision = mc.get('precision', {})
    stable = mc.get('stability', [])
    eligible = bool(report.get('rank_screen') and precision and stable)
    eligible = eligible and all(item['eligible'] for item in stable)
    eligible = eligible and all(
        np.all(item['variance_intervals_overlap'])
        and np.all(np.isfinite(item['ess']))
        and np.all(np.asarray(item['ess']) >= target_ess)
        for item in precision.values())
    return {
        'eligible': bool(eligible),
        'status': 'TUNING_WINDOW_ELIGIBLE' if eligible else 'INCONCLUSIVE',
        'target_ess': target_ess,
        'minimum_ess': report.get('minimum_ess'),
        'scientific_window_eligible': report['eligible'],
        'diagnostics': report,
        'scope': 'Performance tuning only; no reconstructed-output qualification',
    }
