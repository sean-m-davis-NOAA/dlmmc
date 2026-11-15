"""Integrity checks for saved offline validation artifacts."""
import hashlib
import json
from pathlib import Path


def verify_cache_provenance(root, folder, *, certificate=None, model=None, panel=None):
    """Check completion, immutable inputs and the requested scientific identity.

    This does not certify mixing or inferential equivalence, and does not hash
    output chunks for which the original writer recorded no digest.
    """
    root, folder = Path(root), Path(folder)
    config_path = folder/'config.json'
    config = json.loads(config_path.read_text())
    completion = json.loads((folder/'complete.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if completion['config_sha256'] != digest(config_path):
        raise ValueError('Cache completion/config mismatch')
    sources = config.get('source_hashes')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Cache source provenance is missing')
    for name, expected in sources.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Cache source must be a project-relative path')
        if digest(root/path) != expected:
            raise ValueError('Cache source changed: '+name)
    if certificate is not None:
        name = config.get('certificate')
        if name not in sources or json.loads((root/name).read_text()) != certificate:
            raise ValueError('Cache belongs to a different parameter window')
    for key, expected in [('model', model), ('panel', panel)]:
        if expected is not None and config.get(key) != expected:
            raise ValueError('Cache scientific identity mismatch: '+key)
    bridge = config.get('probability_prefix_bridge')
    if bridge is not None:
        path = Path(bridge['path'])
        if path.is_absolute() or '..' in path.parts or digest(root/path) != bridge['sha256']:
            raise ValueError('Probability-prefix qualification changed')
    return config


def verify_reference_probability_sources(state_config, probability_config):
    """Require the same reference chains and data for states and probabilities."""
    prefix = state_config['reference'].rstrip('/')+'/'
    sources = {p:h for p,h in state_config['source_hashes'].items() if p.startswith(prefix)}
    if not sources or any(probability_config['source_hashes'].get(p) != h for p,h in sources.items()):
        raise ValueError('State/probability reference inputs differ')
