"""Public input declarations; no automatic statistical/model changes."""
import json
from pathlib import Path
from .api import _resolve_model


def validate_no_seasonal_input(data, model):
    _, spec = _resolve_model(model)
    if spec['seasonal']:
        return None
    attrs = getattr(data, 'attrs', {})
    kind = attrs.get('seasonality')
    if kind not in ('deseasonalized', 'synthetic_no_seasonality'):
        raise ValueError('No-seasonal models require explicitly deseasonalised data. '
                         'Use deseasonalize_harmonics, or document external preprocessing '
                         'with seasonality and seasonal_preprocessing attributes. '
                         'Synthetic data generated without seasonality require an explicit declaration.')
    description = attrs.get('seasonal_preprocessing')
    if not isinstance(description, str) or not description.strip():
        raise ValueError('A seasonal_preprocessing description is required')
    return dict(seasonality=kind, description=description,
                preprocessing_sha256=attrs.get('seasonal_preprocessing_sha256'))


def record_input_policy(fit, policy):
    if policy is not None:
        path = Path(fit.path) / 'input-policy.json'
        content = dict(policy, prepared_checksum=fit.prepared.checksum)
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError('Existing input policy differs; do not overwrite')
        path.write_text(json.dumps(content, indent=2)+'\n')


def checkpoint_input_policy(fit):
    if _resolve_model(fit.prepared.model)[1]['seasonal']:
        return None
    path = Path(fit.path) / 'input-policy.json'
    if not path.exists():
        raise ValueError('No-seasonal checkpoint lacks an input declaration; original seasonal-data fits cannot continue')
    policy = json.loads(path.read_text())
    if (policy.pop('prepared_checksum') != fit.prepared.checksum
            or policy.get('seasonality') not in ('deseasonalized', 'synthetic_no_seasonality')):
        raise ValueError('Checkpoint seasonal input policy does not match prepared data')
    return policy


def validate_prepared_seasonality(prepared):
    """Require a durable declaration before sampling a no-seasonal joint fit."""
    _,spec=_resolve_model(prepared.model)
    if not spec['seasonal']:
        policy=getattr(prepared,'input_policy',None)
        if not isinstance(policy,dict) or policy.get('seasonality') not in ('deseasonalized','synthetic_no_seasonality') or not policy.get('description'):
            raise ValueError('Joint no-seasonal fits require declared deseasonalised preparation')
