"""Strict reuse of an unchanged, certified reconstruction prefix.

Validation support only: this does not certify sampling, mixing or inference.
"""
import hashlib
import json
import os
from pathlib import Path


def reuse_reconstruction_prefix(root, previous, destination, new_certificate, new_config, *, replay_receipt=None, probability_bridge=None):
    root=Path(root);previous=Path(previous);destination=Path(destination)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    old_config=json.loads((previous/'config.json').read_text())
    complete=json.loads((previous/'complete.json').read_text())
    if complete['config_sha256']!=sha(previous/'config.json'):
        raise ValueError('Old reconstruction configuration changed')
    old_cert_path=root/old_config['certificate']
    if sha(old_cert_path)!=old_config['source_hashes'][old_config['certificate']]:
        raise ValueError('Old certificate changed')
    old=json.loads(old_cert_path.read_text())
    if old['status']!=new_certificate['status'] or old['status']!='WINDOW_DIAGNOSTICS_PASS':
        raise ValueError('Both windows must be diagnosed')
    common=('model','panel','data_sha256','warmup_transitions','independent_runs','total_walkers')
    if replay_receipt is None:common+=('sampling_directory','cell_index')
    for key in common:
        if old.get(key)!=new_certificate.get(key):raise ValueError(f'Prefix differs: {key}')
    count=old['retained_steps'];chunk_size=old_config['steps_per_chunk']
    if count>=new_certificate['retained_steps'] or count%chunk_size or count!=complete['retained_steps']:
        raise ValueError('Requires a complete strict prefix of the new window')
    if complete['runs']!=old['independent_runs']:raise ValueError('Incomplete independent runs')
    for key in ('seed','steps_per_chunk','plot_selection'):
        if old_config[key]!=new_config[key]:raise ValueError(f'Reconstruction policy changed: {key}')
    old_hashes={k:v for k,v in old_config['source_hashes'].items() if k!=old_config['certificate']}
    new_hashes={k:v for k,v in new_config['source_hashes'].items() if k!=new_config['certificate']}
    bridge_proof=None
    if probability_bridge is not None:
        if replay_receipt is not None:raise ValueError('Do not combine replay and probability-engine transitions')
        from .validation_probability_bridge import validate_probability_bridge
        bridge_proof=validate_probability_bridge(root,previous,old_config,new_config,old,new_certificate,old_hashes,new_hashes,probability_bridge)
    elif old_hashes!=new_hashes:raise ValueError('Reconstruction equations/data/summary definitions changed')
    for key,value in old_hashes.items():
        if sha(root/key)!=value:raise ValueError('Reconstruction source no longer matches certificate')
    replay_proof=None
    if replay_receipt is not None:
        from .validation_replay_cache import verify_replayed_prefix
        replay_proof=verify_replayed_prefix(root,old,new_certificate,replay_receipt)
    else:
        for key,value in old['chunk_sha256'].items():
            if new_certificate['chunk_sha256'].get(key)!=value or sha(root/key)!=value:
                raise ValueError('Sampling prefix changed')
    paths=[previous/f'run-{run}-chunk-{start:06d}.npz' for run in range(complete['runs']) for start in range(0,count,chunk_size)]
    if len(list(previous.glob('run-*-chunk-*.npz')))!=len(paths):raise ValueError('Unexpected old reconstruction chunk set')
    paths.append(previous/'quantities.json')
    hashes={str(path.relative_to(root)):sha(path) for path in paths}
    # Validate all existing destinations before creating any links.
    for path in paths:
        dest=destination/path.name
        if dest.exists() and sha(dest)!=hashes[str(path.relative_to(root))]:
            raise ValueError('Destination already contains different reconstruction')
    destination.mkdir(parents=True,exist_ok=True)
    for path in paths:
        dest=destination/path.name
        if not dest.exists():os.link(path,dest)
    receipt=dict(status='VERIFIED_UNCHANGED_RECONSTRUCTION_PREFIX',previous=str(previous.relative_to(root)),retained_steps_reused=count,
                 source_sha256=hashes,old_config_sha256=sha(previous/'config.json'),new_certificate_sha256=sha(root/new_config['certificate']),
                 note='Same certified parameter draws, equations, per-chunk RNG seed, selected walkers, summaries and plots. Immutable hard links; no conditional draws regenerated.')
    if replay_proof is not None:receipt['exact_replay_proof']=replay_proof
    if bridge_proof is not None:
        receipt['status']='VERIFIED_NUMERICALLY_COMPATIBLE_PROBABILITY_PREFIX'
        receipt['probability_bridge']=bridge_proof
        receipt['note']='Original probability bytes retained by immutable links. Same conditioning parameters and estimator; cached-Torch prefix and new Numba arithmetic have explicit numerical qualification, not bitwise engine identity.'
    (destination/'prefix-reuse.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def reuse_reference_chains(root, previous, destination, new_config, *, probabilities=False):
    """Reuse leading complete independent chains when pooling reference runs.

    Chain order is verified by CSV bytes; conditional RNG policies are identical
    for the reused run indices. Appended runs receive their own original indices.
    """
    root=Path(root);previous=Path(previous);destination=Path(destination)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    old=json.loads((previous/'config.json').read_text());done=json.loads((previous/'complete.json').read_text())
    if done['config_sha256']!=sha(previous/'config.json'):raise ValueError('Old reference reconstruction changed')
    old_csv=sorted((k,v) for k,v in old['source_hashes'].items() if k.endswith('.csv'))
    new_csv=sorted((k,v) for k,v in new_config['source_hashes'].items() if k.endswith('.csv'))
    if len(old_csv)<4 or len(new_csv)<=len(old_csv) or [v for _,v in old_csv]!=[v for _,v in new_csv[:len(old_csv)]]:
        raise ValueError('New reference must append independent chains to the exact old CSV prefix')
    def other_sources(config):
        entries=[(Path(k).name,v) for k,v in config['source_hashes'].items() if not k.endswith('.csv')]
        if len({k for k,_ in entries})!=len(entries):raise ValueError('Ambiguous source basenames')
        return dict(entries)
    if other_sources(old)!=other_sources(new_config):raise ValueError('Reference data/equations changed')
    for config in [old,new_config]:
        for key,value in config['source_hashes'].items():
            if sha(root/key)!=value:raise ValueError('Source reference evidence changed')
    ignored={'source_hashes','reference'}
    if {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in new_config.items() if k not in ignored}:
        raise ValueError('Reference reconstruction/RNG/summary policy changed')
    if probabilities:
        # The analytic probability completion records an explicit shape.
        if done['shape'][0]!=len(old_csv):raise ValueError('Incomplete probability run count')
        lengths=[done['shape'][1]]*len(old_csv)
    else:
        if done['chains']!=len(old_csv):raise ValueError('Incomplete reference run count')
        lengths=done['draws_per_chain']
    paths=[previous/f'run-{run}-chunk-{start:06d}.npz' for run,n in enumerate(lengths) for start in range(0,n,old['chunk_size'])]
    if len(list(previous.glob('run-*-chunk-*.npz')))!=len(paths):raise ValueError('Unexpected reconstructed reference chunk set')
    if not probabilities:paths.append(previous/'quantities.json')
    hashes={str(path.relative_to(root)):sha(path) for path in paths}
    for path in paths:
        dest=destination/path.name
        if dest.exists() and sha(dest)!=hashes[str(path.relative_to(root))]:raise ValueError('Destination reference differs')
    destination.mkdir(parents=True,exist_ok=True)
    for path in paths:
        dest=destination/path.name
        if not dest.exists():os.link(path,dest)
    receipt=dict(status='VERIFIED_REFERENCE_CHAIN_REUSE',previous=str(previous.relative_to(root)),chains_reused=len(old_csv),source_sha256=hashes,old_config_sha256=sha(previous/'config.json'),probabilities=probabilities)
    (destination/'chain-reuse.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
