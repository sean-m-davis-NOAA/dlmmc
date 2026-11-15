"""Finite-Monte-Carlo compatibility screening, distinct from equivalence proof.

Inputs must use matching units. This function does not estimate ESS or diagnose
mixing. Missing eligibility prevents an operational practical result. A wide
interval alone is not evidence of accuracy; callers must report achieved ESS.
"""
import numpy as np


def practical_consistency(difference, lower, upper, margin_lower, margin_upper,
                          *, diagnostics_validated=False, ess_validated=False):
    """Compare discrepancy intervals with scientific bands without containment.

    Intersecting bands establish compatibility at the achieved MC precision,
    not equivalence. Point estimates outside the band remain explicit flags.
    """
    arrays = [np.asarray(x, dtype=float) for x in
              (difference, lower, upper, margin_lower, margin_upper)]
    d, lo, hi, ml, mh = np.broadcast_arrays(*arrays)
    if any(not np.isfinite(x).all() for x in (d, lo, hi, ml, mh)):
        raise ValueError("Nonfinite discrepancy or uncertainty cannot be dropped")
    if np.any(lo > hi) or np.any(ml >= mh):
        raise ValueError("Invalid uncertainty interval or scientific band")
    resolved = (hi < ml) | (lo > mh)
    point_outside = (d < ml) | (d > mh)
    eligible = bool(diagnostics_validated and ess_validated)
    metric = np.where(resolved, "INVESTIGATE", "MC_COMPATIBLE")
    status = ("INCONCLUSIVE" if not eligible else "INVESTIGATE" if resolved.any()
              else "PRACTICALLY_CONSISTENT_AT_ACHIEVED_MC_PRECISION")
    return {
        "status": status, "eligible": eligible,
        "metric_status": metric.tolist(),
        "point_estimate_outside_margin": point_outside.tolist(),
        "difference": d.tolist(), "interval_lower": lo.tolist(),
        "interval_upper": hi.tolist(), "margin_lower": ml.tolist(),
        "margin_upper": mh.tolist(),
        "strict_interval_containment": ((lo > ml) & (hi < mh)).tolist(),
        "claim": "Finite-MC compatibility only; not proof of equivalence or zero bias",
    }
