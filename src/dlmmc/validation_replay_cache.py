"""Verify reconstruction reuse across an exact sampler-prefix replay.

This proves equality of the conditioning draws, not inference equivalence.
Log densities may differ within their separately qualified numerical tolerance;
the actual parameter bytes and warmup flags must match exactly.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def verify_replayed_prefix(root, old, new, receipt_path):
    root = Path(root)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt_path = root/receipt_path
    receipt = json.loads(receipt_path.read_text())
    source = root/new['sampling_directory']
    config_path = source/'config.json'
    config = json.loads(config_path.read_text())
    if receipt['status'] != 'PASS' or receipt['config_sha256'] != sha(config_path):
        raise ValueError('Intact successful sampler replay required')
    if config['replay_source'] != old['sampling_directory']:
        raise ValueError('Replay source differs from old window')
    if receipt['steps_per_run'] < old['retained_steps'] or receipt['runs'] != old['independent_runs'] or receipt['warmup'] != old['warmup_transitions']:
        raise ValueError('Replay does not cover old parameter window')
    pairs = {}
    for key, old_hash in old['chunk_sha256'].items():
        old_path = root/key
        new_path = source/old_path.relative_to(root/old['sampling_directory'])
        new_key = str(new_path.relative_to(root))
        if config['replay_manifest'].get(key) != old_hash or sha(old_path) != old_hash:
            raise ValueError('Old replay evidence changed')
        if new_key not in new['chunk_sha256'] or sha(new_path) != new['chunk_sha256'][new_key]:
            raise ValueError('New certified replay prefix changed')
        with np.load(old_path) as a, np.load(new_path) as b:
            x, y = a['draws'], b['draws']
            if old.get('cell_index') is not None:
                x = x[:,:,old['cell_index'],:]
            if new.get('cell_index') is not None:
                y = y[:,:,new['cell_index'],:]
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise ValueError('Conditioning parameter draws differ')
            if not np.array_equal(a['warmup'], b['warmup']):
                raise ValueError('Replay warmup accounting differs')
        pairs[key] = dict(old_sha256=old_hash, new_path=new_key, new_sha256=new['chunk_sha256'][new_key])
    if not pairs:
        raise ValueError('Empty parameter replay')
    return dict(replay_receipt=str(receipt_path.relative_to(root)), replay_receipt_sha256=sha(receipt_path),
                replay_config_sha256=sha(config_path), verified_parameter_chunks=pairs)
