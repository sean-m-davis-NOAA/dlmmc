"""Deferred trajectories conditional on retained, diagnosed ENSO lag draws."""
from dataclasses import replace
from pathlib import Path
import json,hashlib,time,fcntl
import numpy as np
import torch
from .enso_lag import ENSOLagTarget
from .spatial import GaussianSpatialPrior
from .reconstruction import ReconstructionResult,conditional_outputs
from .lag_diagnostics import diagnose_enso_lag
from .inference import posterior_fingerprint,window_diagnostics
from .validation_selection import stratified_indices
from .validation_derived import PROFILE


def conditional_lag_outputs(prepared,predictor,sigma_prior,lag_prior,unconstrained,*,seed):
    """Fixed-parameter conditional calculation; no eligibility implied by itself."""
    p=prepared;u=torch.as_tensor(unconstrained,dtype=p.y.dtype,device=p.y.device)
    target=ENSOLagTarget(p,sigma_prior,lag_prior,predictor,backend='torch')
    lag,_,valid=target.lag_values(u)
    if not bool(valid.all()):raise ValueError('Conditional reconstruction requires strict supported lags')
    shifted=replace(p,regressors=target.regressors(u))
    arrays=conditional_outputs(shifted,u[...,:target.base_parameters].cpu().numpy(),seed=seed)
    arrays.update(enso_lag=lag.cpu().numpy(),unconstrained_lag=u[...,-1].cpu().numpy())
    return arrays


class LagReconstructionResult(ReconstructionResult):
    def iter_chunks(self):
        index=json.loads((self.path/'index.json').read_text())
        for chunk,row in zip(super().iter_chunks(),index['chunks']):
            # The parent verifies the whole-file checksum before yielding.
            with np.load(self.path/row['file'],allow_pickle=False) as z:
                chunk['enso_lag']=(('selected','cell'),z['enso_lag'])
                chunk['unconstrained_lag']=(('selected','cell'),z['unconstrained_lag'])
            yield chunk

    def diagnose_outputs(self,*,time_indices,contrasts,amplitude_times):
        report=super().diagnose_outputs(time_indices=time_indices,contrasts=contrasts,amplitude_times=amplitude_times)
        runs={};names=['lag_ENSO_months']+[f'lag_times_trend_change:{a}-{b}' for a,b in contrasts]
        for c in self.iter_chunks():
            lag=c.enso_lag.values;arrays=[lag]
            arrays.extend(lag*(c.states.values[:,:,b,0]-c.states.values[:,:,a,0]) for a,b in contrasts)
            runs.setdefault(c.attrs['run'],[]).append(np.stack(arrays,axis=-1))
        x=np.stack([np.concatenate(runs[r]) for r in sorted(runs)])
        for g,cell in enumerate(report['cells']):
            d=window_diagnostics(x[:,:,None,g,:]);cell['lag_and_trend_products']=d;cell['eligible']=cell['eligible'] and d['eligible']
        report.update(lag_output_quantities=names,eligible=all(c['eligible'] for c in report['cells']))
        return report


def reconstruct_enso_lag(fit,diagnostics,*,output_dir,selected_per_run=2048,chunk_draws=16,seed=202611310010,wall_seconds=600):
    """Keep joint draw alignment and fixed data coverage across every output."""
    if not diagnostics.get('eligible') or diagnostics.get('profile')!=PROFILE:raise ValueError('Diagnosed lag parameter window required')
    if min(selected_per_run,chunk_draws)<1 or chunk_draws>64 or wall_seconds<=0:raise ValueError('Positive bounded reconstruction settings required')
    if seed in range(fit.config['seed'],fit.config['seed']+10000+fit.config['runs']):raise ValueError('Use a separate reconstruction RNG seed range')
    started=time.time();posterior=fit.posterior();fingerprint=posterior_fingerprint(fit,posterior)
    if fingerprint!=diagnostics['posterior_sha256']:raise ValueError('Posterior window changed')
    fresh=diagnose_enso_lag(fit,trace_review=diagnostics.get('trace_review'))
    if not fresh['eligible']:raise ValueError('Lag diagnostics no longer eligible')
    p=fit.prepared;u=posterior.unconstrained.values;folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    config=dict(schema=1,kind='enso_lag_reconstruction',model=p.model,nreg=p.regressors.shape[-1],cells=p.cells,runs=u.shape[0],parameter_names=p.parameter_names,posterior_sha256=fingerprint,prepared_checksum=p.checksum,seed=seed,selected_per_run=selected_per_run,chunk_draws=chunk_draws,scale=p.scale.tolist(),offset=p.offset.tolist(),cell_coordinates={k:np.asarray(v).tolist() for k,v in p.cell_coordinates.items()},dtype=str(p.y.dtype),device=str(p.y.device),sigma_prior=fit.config['sigma_prior'],lag_prior=fit.config['lag_prior'],fit_provenance=fit.config['provenance'],lag_convention='ENSO(t-lag)',selection='Value-independent stratified time/walker indices; allcellsaligned',source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('lag_reconstruction.py','enso_lag.py','lag_diagnostics.py','spatial.py','reconstruction.py','inference.py','stability.py','validation_derived.py','tensor_target.py','tensor_smoothing.py','simulation_smoothing_mean.py','precision_prototype.py','validation_selection.py')})
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=config:raise ValueError('Reconstruction configuration changed')
            index=json.loads((folder/'index.json').read_text())
        else:
            (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n');index=dict(status='RUNNING',chunks=[],executions=[])
        existing={r['file']:r for r in index['chunks']}
        def save(status):
            index['status']=status;tmp=folder/'index.tmp';tmp.write_text(json.dumps(index,indent=2)+'\n');tmp.replace(folder/'index.json')
        sigma=GaussianSpatialPrior(**fit.config['sigma_prior']);lag=GaussianSpatialPrior(**fit.config['lag_prior']);old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for run in range(u.shape[0]):
                indices=stratified_indices(u.shape[1],u.shape[2],selected_per_run,seed+run)
                for start in range(0,len(indices),chunk_draws):
                    name=f'run-{run}-chunk-{start:06d}.npz';file=folder/name
                    if name in existing:
                        if hashlib.sha256(file.read_bytes()).hexdigest()!=existing[name]['sha256']:raise ValueError('Saved lag output chunk changed')
                        continue
                    if time.time()-started>=wall_seconds:
                        index['executions'].append(dict(status='WALL_CAP_CHECKPOINTED',actual_wall_seconds=time.time()-started));save('WALL_CAP_CHECKPOINTED');return LagReconstructionResult(folder)
                    ix=indices[start:start+chunk_draws];values=u[run,ix[:,0],ix[:,1]]
                    chunk_seed=int(np.random.SeedSequence([seed,run,start,6]).generate_state(1,dtype=np.uint64)[0]%np.uint64(2**63-2))
                    before=time.perf_counter();arrays=conditional_lag_outputs(p,fit.predictor,sigma,lag,values,seed=chunk_seed);elapsed=time.perf_counter()-before
                    tmp=file.with_suffix('.tmp')
                    with tmp.open('wb') as stream:np.savez(stream,**arrays,unconstrained=values[...,:-1],selected=np.arange(start,start+len(ix)),time=p.time,source_draw=posterior.draw.values[ix[:,0]],source_walker=ix[:,1])
                    tmp.replace(file);index['chunks'].append(dict(file=name,run=run,start=start,count=len(ix),seed=chunk_seed,kernel_seconds=elapsed,sha256=hashlib.sha256(file.read_bytes()).hexdigest()));save('OUTPUT_DIAGNOSTICS_PENDING')
            index['executions'].append(dict(status='RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING',actual_wall_seconds=time.time()-started));save('RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING')
        finally:torch.set_num_threads(old_threads)
    return LagReconstructionResult(folder)
