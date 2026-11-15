"""Execution policies are separate from model and qualification thresholds."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    runs: int
    half_walkers: int
    warmup: int
    retained_sweeps: int
    chunk_sweeps: int = 16
    thin: int = 1
    output_draws_per_sweep: int = 4

    def to_dict(self):
        return asdict(self)


PRODUCTION_PROFILE = ExecutionProfile('production', 1, 64, 128, 256, output_draws_per_sweep=8)
TESTING_PROFILE = PRODUCTION_PROFILE
LEGACY_PRODUCTION_PROFILE = ExecutionProfile('production', 1, 64, 128, 512)
QUALIFICATION_PROFILE = ExecutionProfile('qualification', 4, 64, 100, 4000, 50)


def validate_production_scope(panel_ids, selection, *, model, backend, dtype, profile):
    ids = tuple(map(str, panel_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Unique named production panels required')
    if profile.name != 'production' or min(profile.runs, profile.retained_sweeps,
            profile.half_walkers, profile.chunk_sweeps, profile.thin) < 1 or profile.warmup < 0:
        raise ValueError('Invalid production profile')
    if profile.thin != 1:
        raise ValueError('Production monitoring currently requires no thinning')
    from .api import _MODEL_SPECS
    if model not in _MODEL_SPECS or (backend,dtype) not in (('numba','float64'),('torch','float64'),('mps','float32'),('cuda','float32')):
        raise ValueError('Production requires a supported DLM model and CPU64 or explicitly selected GPU32')
    if len(ids) <= 5 and selection is None:
        return
    if (not isinstance(selection, dict) or selection.get('scope') != 'independent_production'
            or not selection.get('explicit_user_response')
            or tuple(selection.get('panel_ids', ())) != ids
            or selection.get('model') != model or selection.get('backend') != backend
            or selection.get('dtype') != dtype or selection.get('profile') != profile.to_dict()):
        raise ValueError('Larger production grid requires matching explicit named selection and settings')
