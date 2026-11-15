"""Scope the preserved original sign estimators without blocking new studies."""


def original_indicator_first_look(certificate):
    """Only the two already-started individual first looks retain indicators.

    New tensor studies of these same panels have no earlier sign comparison;
    they follow the frozen analytic-estimator revision from their first look.
    """
    protected = {
        'experiments/affine-pilot/float64',
        'experiments/affine-batch/individual-p05-lat01/float64',
        'experiments/affine-batch/individual-p10-lat04/float64',
        'experiments/affine-batch/numba-individual-p05-lat01/float64',
        'experiments/affine-batch/numba-individual-p10-lat04/float64',
    }
    return (certificate['panel'] in ('p05-lat01', 'p10-lat04')
            and certificate['formal_look'] < 2
            and certificate['sampling_directory'] in protected)
