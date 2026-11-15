"""Offline evidence loaders for panel-specific independent ensemble comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .transforms import parameter_names
from .validation_parameters import physical_parameter_draws, stationary_ar_variance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pair(candidate, reference, candidate_seeds, reference_seeds):
    for certificate in (candidate, reference):
        if certificate['status'] != 'WINDOW_DIAGNOSTICS_PASS':
            raise ValueError('Both parameter windows must pass diagnostics')
    for key in ('model', 'panel', 'data_sha256'):
        if candidate[key] != reference[key]:
            raise ValueError('Panel/model/input mismatch: '+key)
    if candidate['sampling_directory'] == reference['sampling_directory']:
        raise ValueError('Cannot compare a study against itself')
    if len(set(candidate_seeds)) < 4 or len(set(reference_seeds)) < 4:
        raise ValueError('Four independent ensembles required on each side')
    if set(candidate_seeds) & set(reference_seeds):
        raise ValueError('Independent-run comparison has overlapping parameter seeds')


def require_summary_diagnostics(diagnostics, indices, block_steps):
    d = diagnostics
    if not all(d['iat_length_reliable'][j] and d['rank_rhat_ensemble_means'][j] < 1.01 for j in indices):
        raise ValueError('Derived mean/length diagnostics fail')
    if 'rank_rhat_ensemble_widths' in d and not all(d['rank_rhat_ensemble_widths'][j] < 1.01 for j in indices):
        raise ValueError('Derived width diagnostics fail')
    items = [d, *d['quantile_indicator_diagnostics'].values()]
    for item in items:
        if not all(item['iat_length_reliable'][j] and item['split_rhat_ensemble_means'][j] < 1.01 for j in indices):
            raise ValueError('Derived tail/length diagnostics fail')
        if block_steps < 5*np.max(np.asarray(item['iat_by_run'])[:, indices]):
            raise ValueError('MCSE blocks shorter than five controlling IATs')


def load_parameters(root, certificate_path):
    root = Path(root)
    cp = root/certificate_path
    cert = json.loads(cp.read_text())
    if cert['status'] != 'WINDOW_DIAGNOSTICS_PASS':
        raise ValueError('Uncertified parameter window')
    audit_path = root/cert['diagnostic_report']
    if digest(audit_path) != cert['diagnostic_report_sha256']:
        raise ValueError('Certified diagnostic evidence changed')
    if 'trace_review' in cert and digest(root/cert['trace_review']) != cert['trace_review_sha256']:
        raise ValueError('Certified trace review changed')
    source = root/cert['sampling_directory']
    config_path = source/'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        audit = json.loads(audit_path.read_text())
        if config != audit['config'] or config['panels'][cert['cell_index']] != cert['panel']:
            raise ValueError('Sampling configuration/panel map changed')
        if any(config.get(key) != expected for key, expected in [('dtype','float64'),('process_variances','legacy'),('observation_policy','explicit_mask'),('normalization','normalize64_then_cast')]):
            raise ValueError('Outside frozen backend/model-policy family')
        seeds = config['seeds']
    else:
        if cert['sampling_directory'] != 'experiments/affine-pilot/float64':
            raise ValueError('Unknown legacy cache')
        config_path = source/'initialization.json'
        seeds = json.loads(config_path.read_text())['seed_set']
    data_path = root/cert['data_file']
    if digest(data_path) != cert['data_sha256']:
        raise ValueError('Input provenance mismatch')
    data = json.loads(data_path.read_text())
    nreg = data.get('nreg',0)
    names = parameter_names(cert['model'],nreg)
    runs = []
    for run in range(cert['independent_runs']):
        parts = []
        for path in sorted((source/f'run-{run}').glob('chunk-*.npz')):
            key = str(path.relative_to(root))
            if key not in cert['chunk_sha256']:
                continue
            if digest(path) != cert['chunk_sha256'][key]:
                raise ValueError('Certified parameter chunk changed')
            with np.load(path) as z:
                u = z['draws'][~z['warmup']]
                if cert.get('cell_index') is not None:
                    u = u[:,:,cert['cell_index'],:]
                parts.append(u)
        u = np.concatenate(parts)
        if u.shape[:2] != (cert['retained_steps'],cert['total_walkers']):
            raise ValueError('Incomplete frozen retained window')
        x = np.concatenate([np.exp(u[...,:3]),expit(u[...,3:])],axis=-1) if cert['model']=='vanilla_ar1' else physical_parameter_draws(cert['model'],u,nreg)
        ar = x[...,2]**2/(1-x[...,3]**2) if cert['model']=='vanilla_ar1' else stationary_ar_variance(cert['model'],x,nreg)
        scale = np.ptp(data['time_series'])
        for j,name in enumerate(names):
            if name.startswith('sigma'): x[...,j] *= scale
        runs.append(np.concatenate([x,(ar*scale**2)[...,None]],axis=-1))
    provenance = {str(p.relative_to(root)):digest(p) for p in (cp,config_path,data_path)}
    return cert,seeds,np.stack(runs),names+['stationary_AR_variance'],provenance


def load_reconstruction(root, cert):
    root = Path(root)
    folder = root/'experiments/reconstruction'/cert['model']/cert['panel']/cert['window_id']
    completion = json.loads((folder/'complete.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    if json.loads((root/config['certificate']).read_text()) != cert:
        raise ValueError('Reconstruction belongs to a different parameter window')
    if completion['config_sha256'] != digest(folder/'config.json'):
        raise ValueError('Reconstruction completion/config mismatch')
    for name,h in config['source_hashes'].items():
        if digest(root/name) != h:
            raise ValueError('Reconstruction semantic source changed: '+name)
    names = json.loads((folder/'quantities.json').read_text())
    diagnostic = json.loads((folder/'diagnostics.json').read_text())
    runs = []
    for run in range(cert['independent_runs']):
        parts = []
        for path in sorted(folder.glob(f'run-{run}-chunk-*.npz')):
            with np.load(path) as z: parts.append(z['summaries'])
        runs.append(np.concatenate(parts))
    values = np.stack(runs)
    if values.shape[:3] != (cert['independent_runs'],cert['retained_steps'],len(completion['selected_walkers'])):
        raise ValueError('Incomplete reconstruction window')
    return folder,values,names,diagnostic,config


def load_analytic_signs(folder, cert):
    folder = Path(folder)
    candidates = [folder/name for name in ('conditional-sign-stable-numba','conditional-sign-stable-cached','conditional-sign-stable')]
    sign = next((p for p in candidates if (p/'complete.json').exists()),None)
    if sign is None: raise ValueError('Complete analytic sign cache required')
    completion = json.loads((sign/'complete.json').read_text())
    if completion['config_sha256'] != digest(sign/'config.json'):
        raise ValueError('Analytic completion/config mismatch')
    d = json.loads((sign/'diagnostics.json').read_text())
    if not all(d['iat_length_reliable']) or not np.all(np.asarray(d['rank_rhat']) < 1.01) or not np.all(np.asarray(d['rank_width_rhat']) < 1.01):
        raise ValueError('Analytic sign diagnostics fail')
    if cert['block_steps'] < 5*np.max(d['iat_by_run']):
        raise ValueError('Sign MCSE blocks too short')
    runs = []
    for run in range(cert['independent_runs']):
        parts = []
        for path in sorted(sign.glob(f'run-{run}-chunk-*.npz')):
            with np.load(path) as z: parts.append(z['probabilities'])
        runs.append(np.concatenate(parts))
    values = np.stack(runs)
    if list(values.shape) != completion['shape'] or values.shape[1] != cert['retained_steps']:
        raise ValueError('Incomplete analytic sign window')
    return sign,values
