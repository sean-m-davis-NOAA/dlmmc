"""Bounded deferred reconstruction with explicit selection and RNG provenance."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import fcntl
import numpy as np
import torch
import xarray as xr
from .api import _MODEL_SPECS
from .inference import diagnose_fit, posterior_fingerprint
from .validation_selection import stratified_indices
from .tensor_target import model_components
from .tensor_smoothing import conditional_draw, conditional_moments
from .precision_prototype import filter_tensor
from .simulation_smoothing_mean import conditional_draw_mean


def conditional_outputs(prepared, unconstrained, *, seed):
    """Fixed-parameter conditional inference, also used by Gaussian tests.

    Leading axis indexes selected parameters, followed by cell and parameter.
    Float64 CPU uses the validated mean-only simulation correction. Other
    explicit dtypes/devices use the native reference smoother, without fallback.
    Returned covariance is conditional marginal variance; full trajectories
    preserve cross-time covariance. No posterior eligibility is implied by a
    fixed-parameter calculation.
    """
    p = prepared
    u = torch.tensor(unconstrained, dtype=p.y.dtype, device=p.y.device)
    if u.ndim != 3 or u.shape[1] != p.cells:
        raise ValueError('Expected [selected draw, cell, parameter]')
    F, G, Q, C, _, valid = model_components(u, *p.arguments, p.model, S=p.priors['S'], sampling=p.sampling)
    if not bool(valid.all()):
        raise ValueError('Reconstruction parameters must all be in support')
    state_rng = torch.Generator(device=p.y.device).manual_seed(int(seed))
    predictive_rng = torch.Generator(device=p.y.device).manual_seed(int(seed) + 1)
    smoother = conditional_draw_mean if p.y.dtype == torch.float64 and p.y.device.type == 'cpu' else conditional_draw
    with torch.no_grad():
        states = smoother(p.y, p.variance, p.observed, F, G, Q, C, generator=state_rng)
        mean, covariance, _ = conditional_moments(filter_tensor(p.y, p.variance, p.observed, F, G, Q, C), G, Q)
        scale = torch.as_tensor(p.scale, dtype=u.dtype, device=u.device)[None, :, None]
        offset = torch.as_tensor(p.offset, dtype=u.dtype, device=u.device)[None, :, None]
        latent = (F * states).sum(-1) * scale + offset
        signal_mean = (F * mean).sum(-1) * scale + offset
        signal_variance = torch.einsum('...ti,...tij,...tj->...t', F, covariance, F) * scale**2
        predictive = latent + torch.sqrt(p.variance)*scale*torch.randn(latent.shape, dtype=u.dtype, device=u.device, generator=predictive_rng)
        # Missing uncertainty was sanitized for filtering; it is not a known
        # zero-noise predictive distribution. Retain an explicit missing value.
        predictive = torch.where(p.observed, predictive, torch.full_like(predictive, torch.nan))
        states = states * scale[..., None]
        mean = mean * scale[..., None]
        states[..., 0] += offset; mean[..., 0] += offset
        state_variance = covariance.diagonal(dim1=-2, dim2=-1) * scale[..., None]**2
    return {k: v.cpu().numpy() for k, v in dict(states=states, conditional_mean=mean,
            conditional_variance=state_variance, latent_signal=latent,
            signal_conditional_mean=signal_mean, signal_conditional_variance=signal_variance,
            posterior_predictive=predictive).items()}


@dataclass
class ReconstructionResult:
    path: Path

    def iter_chunks(self):
        """Load one bounded xarray chunk at a time, checking its checksum."""
        config = json.loads((self.path/'config.json').read_text())
        for row in json.loads((self.path/'index.json').read_text())['chunks']:
            path = self.path/row['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Reconstruction chunk changed')
            with np.load(path, allow_pickle=False) as z:
                coords = dict(selected=z['selected'], time=z['time'], cell=np.arange(config['cells']),
                              state=np.arange(z['states'].shape[-1]), parameter=config['parameter_names'])
                for k, v in config['cell_coordinates'].items(): coords[k] = ('cell', v)
                variables = {}
                for key in ('states', 'conditional_mean', 'conditional_variance'):
                    variables[key] = (('selected', 'cell', 'time', 'state'), z[key])
                for key in ('latent_signal', 'signal_conditional_mean', 'signal_conditional_variance', 'posterior_predictive'):
                    variables[key] = (('selected', 'cell', 'time'), z[key])
                variables['source_draw'] = ('selected', z['source_draw'])
                variables['source_walker'] = ('selected', z['source_walker'])
                variables['unconstrained'] = (('selected','cell','parameter'), z['unconstrained'])
                yield xr.Dataset(variables, coords=coords, attrs=dict(run=row['run'],
                    model=config['model'], units='original observation units',
                    diagnostic_status='OUTPUT_DIAGNOSTICS_PENDING', parameter_window=config['posterior_sha256']))

    def moments(self):
        """Mixture moments, including E[conditional variance] explicitly.

        These estimates still require scientific-output ESS/MCSE checks.
        """
        count=0; total=None; second=None; template=None
        for chunk in self.iter_chunks():
            m=chunk.signal_conditional_mean.values.astype(np.float64)
            v=chunk.signal_conditional_variance.values.astype(np.float64)
            if total is None: total=np.zeros(m.shape[1:]); second=total.copy()
            total+=m.sum(0); second+=(v+m*m).sum(0); count+=len(m); template=chunk
        if not count: raise ValueError('No completed reconstruction chunks')
        mean=total/count; variance=second/count-mean**2
        if np.any(variance<0): raise FloatingPointError('Negative mixture variance; no clipping')
        return xr.Dataset(dict(mean=(('cell','time'),mean),variance=(('cell','time'),variance)),
            coords={k:v for k,v in template.coords.items() if k not in ('selected','state','parameter')},
            attrs=dict(selected_draws=count, diagnostic_status='OUTPUT_DIAGNOSTICS_PENDING',aggregation_dtype='float64 on CPU',
                       variance_rule='E[Var(signal|parameters,y)] + Var(E[signal|parameters,y])'))

    def diagnose_outputs(self, *, time_indices, contrasts, amplitude_times):
        """Diagnose prespecified sampled scientific summaries separately.

        Returns every cell, including failures. Constant/unsampled sign events
        remain inconclusive; they require a separately validated conditional
        probability estimator or uncertainty bound, never an artificial floor.
        """
        from .inference import physical_parameters
        from .validation_summaries import scientific_summaries
        from .inference import window_diagnostics
        from .diagnostics import ensemble_diagnostics,split_rhat
        from .validation_derived import PROFILE, finite_moment_summaries, raw_variance_diagnostics
        config=json.loads((self.path/'config.json').read_text())
        index=json.loads((self.path/'index.json').read_text())
        if index['status']!='RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING':
            raise ValueError('Complete balanced output selection required')
        values={};signs={};labels=None;sign_labels=None
        for chunk in self.iter_chunks():
            run=chunk.attrs['run'];values.setdefault(run,[]);signs.setdefault(run,[])
            block=[];positive=[]
            for cell in range(config['cells']):
                _,theta=physical_parameters(chunk.unconstrained.values[:,cell],config['model'],config['nreg'])
                states=chunk.states.values[:,cell].copy()
                states[...,0]-=config['offset'][cell];states/=config['scale'][cell]
                labels,q,sign_labels,s=scientific_summaries(config['model'],theta[...,:-1],states,
                    nreg=config['nreg'],scale=config['scale'][cell],offset=config['offset'][cell],
                    time_indices=time_indices,contrasts=contrasts,amplitude_times=amplitude_times)
                block.append(q);positive.append(s)
            values[run].append(np.stack(block,1));signs[run].append(np.stack(positive,1))
        x=np.stack([np.concatenate(values[r]) for r in sorted(values)])
        positive=np.stack([np.concatenate(signs[r]) for r in sorted(signs)]).astype(float)
        raw_variance=x[...,labels.index('stationary_AR_variance')]
        labels,x=finite_moment_summaries(labels,x)
        reports=[]
        for cell in range(config['cells']):
            d=window_diagnostics(x[:,:,None,cell,:]);s=ensemble_diagnostics(positive[:,:,None,cell,:])
            rh=split_rhat(positive[:,:,cell]);s['rank_rhat']=rh.tolist()
            s['eligible']=bool(all(s['iat_length_reliable']) and np.all(np.array(s['mean_ess_estimate'])>=2000)
                               and np.all(np.isfinite(rh)&(rh<1.05)))
            reports.append(dict(cell=cell,scalar=d,positive=s,eligible=d['eligible'] and s['eligible'],
                                raw_stationary_AR_variance=raw_variance_diagnostics(raw_variance[:,:,cell])))
        return dict(profile=PROFILE,eligible=all(r['eligible'] for r in reports),cells=reports,quantities=labels,
                    positive_quantities=sign_labels,scope='Scientific-output precision; not cross-backend equivalence')


def _reconstruct_fit(fit, diagnostics, *, output_dir, selected_per_run=2048, chunk_draws=32,
                    seed=202611190000, wall_seconds=900, torch_threads=1):
    """Reconstruct a diagnosed window into resumable atomic CPU-storage chunks.

    Selection is stratified by time and independent of parameter values.
    A changed window, seed or chunk policy cannot silently reuse old outputs.
    Output diagnostics remain a separate requirement from parameter eligibility.
    """
    started=time.time()
    if selected_per_run<1 or chunk_draws<1 or wall_seconds<=0: raise ValueError('Invalid reconstruction budget')
    checked=diagnose_fit(fit, trace_review=diagnostics.get('trace_review'))
    posterior=fit.posterior()
    if (not checked['eligible'] or checked['posterior_sha256']!=diagnostics.get('posterior_sha256')
            or not diagnostics.get('eligible')):
        raise ValueError('A currently diagnosed, trace-reviewed parameter window is required')
    u=posterior.unconstrained.values; p=fit.prepared
    if selected_per_run>u.shape[1]: raise ValueError('Selection cannot exceed retained steps per run')
    if seed in range(fit.config['seed'],fit.config['seed']+10000+fit.config['runs']):
        raise ValueError('Use a separate reconstruction RNG seed range')
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    config=dict(schema=1,posterior_sha256=posterior_fingerprint(fit,posterior),model=p.model,cells=p.cells,
                cell_coordinates={k:v.tolist() for k,v in p.cell_coordinates.items()},
                parameter_names=p.parameter_names,nreg=p.regressors.shape[-1],offset=p.offset.tolist(),scale=p.scale.tolist(),
                selected_per_run=selected_per_run,chunk_draws=chunk_draws,seed=seed,
                dtype=str(p.y.dtype),device=str(p.y.device),trace_review=checked['trace_review'],
                source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('reconstruction.py','inference.py','stability.py','validation_derived.py','tensor_target.py','tensor_smoothing.py',
                                 'simulation_smoothing_mean.py','precision_prototype.py','validation_selection.py')})
    config_path=folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Reconstruction provenance changed; use a new output directory')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    index=json.loads((folder/'index.json').read_text()) if (folder/'index.json').exists() else {'chunks':[]}
    if not (folder/'index.json').exists():
        index['status']='RECONSTRUCTION_INCOMPLETE';(folder/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    saved={r['file']:r for r in index['chunks']}
    old_threads=torch.get_num_threads();torch.set_num_threads(torch_threads)
    try:
        for run in range(u.shape[0]):
            ix=stratified_indices(u.shape[1],u.shape[2],selected_per_run,seed+run)
            for start in range(0,selected_per_run,chunk_draws):
                filename=f'run-{run}-selected-{start:07d}.npz';path=folder/filename
                if filename in saved:
                    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=saved[filename]['sha256']:
                        raise ValueError('Saved reconstruction chunk changed')
                    continue
                if path.exists(): raise ValueError('Unindexed output needs explicit inspection')
                if time.time()-started>=wall_seconds:
                    index['status']='ACTUAL_WALL_CAP_RECONSTRUCTION_INCOMPLETE'
                    index.setdefault('executions',[]).append(dict(status=index['status'],actual_wall_seconds=time.time()-started))
                    temporary=folder/'index.tmp';temporary.write_text(json.dumps(index,indent=2)+'\n');temporary.replace(folder/'index.json')
                    return ReconstructionResult(folder)
                selection=ix[start:start+chunk_draws]
                chunk_seed=int(np.random.SeedSequence([seed,run,start,1]).generate_state(1,dtype=np.uint64)[0] % np.uint64(2**63-2))
                before=time.perf_counter()
                arrays=conditional_outputs(p,u[run,selection[:,0],selection[:,1]],seed=chunk_seed)
                elapsed=time.perf_counter()-before
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as f:
                    np.savez(f,**arrays,unconstrained=u[run,selection[:,0],selection[:,1]],selected=np.arange(start,start+len(selection)),time=p.time,
                        source_draw=posterior.draw.values[selection[:,0]],source_walker=selection[:,1])
                temp.replace(path)
                index['chunks'].append(dict(file=filename,run=run,start=start,count=len(selection),seed=chunk_seed,
                    kernel_seconds=elapsed,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                index['status']='OUTPUT_DIAGNOSTICS_PENDING'
                temporary=folder/'index.tmp';temporary.write_text(json.dumps(index,indent=2)+'\n');temporary.replace(folder/'index.json')
        index['status']='RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING'
        index.setdefault('executions',[]).append(dict(status=index['status'],actual_wall_seconds=time.time()-started))
        temporary=folder/'index.tmp';temporary.write_text(json.dumps(index,indent=2)+'\n');temporary.replace(folder/'index.json')
    finally:torch.set_num_threads(old_threads)
    return ReconstructionResult(folder)


def reconstruct_fit(fit, diagnostics, *, output_dir, **options):
    """Lock a diagnosed reconstruction directory and write bounded chunks.

    See ``_reconstruct_fit`` for selection, RNG and resume policy. The initial
    gate precedes directory creation; the complete gate is recomputed inside
    the writer to bind eligibility to the actual retained parameter window.
    """
    if not diagnostics.get('eligible'):
        raise ValueError('A diagnosed parameter window is required')
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _reconstruct_fit(fit,diagnostics,output_dir=folder,**options)
