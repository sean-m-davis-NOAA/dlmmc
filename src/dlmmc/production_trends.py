"""Stream deterministic trend moments; never save a grid of state trajectories."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from numba import njit,prange,typeof
from .tensor_target import model_components
from .production import production_parameter_cache,_cell
from .sampling import _threads
from .diagnostics import autocorrelation_time


def summarize_production_trends(fit, *, output_dir, anchor_month='1998-01',
        selected_walkers_per_sweep=None, seed=20261116, wall_seconds=5400,
        numba_threads=8, engine='auto', chunk_draws=None, workspace_bytes=1024**3):
    """Compute all panels/draws in one call; retain the original oracle explicitly.

    ``engine='legacy'`` uses the historical two-thread, draw-chunked routine.
    The default stores only online moments and per-worker temporary histories.
    """
    if selected_walkers_per_sweep is None:
        selected_walkers_per_sweep=fit.config['execution_profile'].get('output_draws_per_sweep',4)
    if engine=='auto':engine='gpu' if fit.prepared.y.device.type in ('mps','cuda') else 'full_batch'
    if engine=='gpu':
        if chunk_draws is not None:raise ValueError('GPU draw tiles are selected by workspace_bytes')
        from .gpu_smoothing import summarize_gpu_trends
        return summarize_gpu_trends(fit,output_dir=output_dir,anchor_month=anchor_month,
            selected_walkers_per_sweep=selected_walkers_per_sweep,seed=seed,wall_seconds=wall_seconds,workspace_bytes=workspace_bytes)
    if engine=='legacy':
        return summarize_production_trends_legacy(fit,output_dir=output_dir,
            anchor_month=anchor_month,selected_walkers_per_sweep=selected_walkers_per_sweep,
            seed=seed,wall_seconds=wall_seconds,chunk_draws=16 if chunk_draws is None else chunk_draws)
    if engine!='full_batch' or chunk_draws is not None:
        raise ValueError('Full-batch output does not accept draw chunks; choose engine="legacy" explicitly')
    from .production_smoothing import summarize_full_batch
    return summarize_full_batch(fit,output_dir=output_dir,anchor_month=anchor_month,
        selected_walkers_per_sweep=selected_walkers_per_sweep,seed=seed,
        wall_seconds=wall_seconds,numba_threads=numba_threads)


@njit(cache=True,fastmath=False)
def _trend_one(y,variance,observed,F,G,Q,C,anchor):
    n,d=F.shape;eye=np.eye(d);m=np.zeros(d);P=C.copy()
    means=np.empty((n,d));covs=np.empty((n,d,d))
    for t in range(n):
        m=G@m;P=G@P@G.T+Q
        if observed[t]:
            f=F[t];pf=P@f;s=f@pf+variance[t]
            if not np.isfinite(s) or s<=0:raise ValueError('Invalid innovation; no floor')
            gain=pf/s;m=m+gain*(y[t]-f@m);A=eye-np.outer(gain,f)
            P=A@P@A.T+np.outer(gain,gain)*variance[t]
        means[t]=m;covs[t]=P
    sm=means.copy();sc=covs.copy();gains=np.empty((n-1,d,d))
    for t in range(n-2,-1,-1):
        R=G@covs[t]@G.T+Q;J=np.ascontiguousarray(np.linalg.solve(R,G@covs[t]).T)
        sm[t]=means[t]+J@(sm[t+1]-G@means[t]);A=eye-J@G
        sc[t]=A@covs[t]@A.T+J@Q@J.T+J@sc[t+1]@J.T;gains[t]=J
    result=np.empty((n,4))
    for t in range(n):
        result[t,0]=sm[t,0];result[t,1]=sc[t,0,0];result[t,2]=sm[t,0]-sm[anchor,0]
    result[anchor,3]=0.
    cross=sc[anchor,:,0].copy()
    for t in range(anchor-1,-1,-1):
        cross=gains[t]@cross;result[t,3]=sc[t,0,0]+sc[anchor,0,0]-2*cross[0]
    weight=np.zeros(d);weight[0]=1.
    for t in range(anchor+1,n):
        weight=gains[t-1].T@weight;result[t,3]=sc[t,0,0]+sc[anchor,0,0]-2*(weight@sc[t,:,0])
    if not np.isfinite(result).all() or np.any(result[:,1]<0) or np.any(result[:,3]<0):
        raise ValueError('Invalid trend covariance; no clipping or floor')
    return result


@njit(cache=True,parallel=True,fastmath=False)
def _trend_batch(y,variance,observed,F,G,Q,C,anchor):
    result=np.empty((len(F),len(y),4))
    for i in prange(len(F)):result[i]=_trend_one(y,variance,observed,F[i],G[i],Q[i],C[i],anchor)
    return result


def _arguments(p,u,anchor):
    if p.cells!=1 or p.y.dtype!=torch.float64 or p.y.device.type!='cpu':raise ValueError('One CPU64 cell per bounded moment chunk')
    z=torch.tensor(u,dtype=torch.float64)
    with torch.no_grad():F,G,Q,C,_,valid=model_components(z,*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
    if not bool(valid.all()) or not 0<=anchor<len(p.time):raise ValueError('Valid parameters and anchor required')
    return tuple(np.ascontiguousarray(v.numpy()) for v in (p.y[0],p.variance[0],p.observed[0],F[:,0],G[:,0],Q[:,0],C[:,0]))+(int(anchor),)


def fixed_trend_moments(prepared,unconstrained,anchor):
    """Fixed-parameter Gaussian oracle operation; no inference eligibility claim."""
    p=prepared;a=_arguments(p,unconstrained,anchor);out=_trend_batch(*a)
    out[:,:,0]=out[:,:,0]*p.scale[0]+p.offset[0]
    out[:,:,1]*=p.scale[0]**2;out[:,:,2]*=p.scale[0];out[:,:,3]*=p.scale[0]**2
    return out


def summarize_production_trends_legacy(fit,*,output_dir,anchor_month='1998-01',selected_walkers_per_sweep=4,
        chunk_draws=16,seed=20261116,wall_seconds=5400):
    if fit.config.get('execution_profile',{}).get('name')!='production':raise ValueError('Explicit production fit required')
    started=time.time();folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'execution.json').exists():raise ValueError('Preserve completed trend summary')
    p=fit.prepared;matches=np.flatnonzero(p.time.astype('datetime64[M]')==np.datetime64(anchor_month,'M'))
    if len(matches)!=1:raise ValueError('One observation at requested anchor month required')
    anchor=int(matches[0]);u=production_parameter_cache(fit);runs,steps,walkers,cells,_=u.shape
    if not 1<=selected_walkers_per_sweep<=walkers:raise ValueError('Valid value-independent walker selection required')
    sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('production_trends.py','tensor_target.py','production.py')}
    config=dict(fit_provenance=fit.config['provenance'],source_identity=sources,anchor_month=anchor_month,anchor_index=anchor,
        selected_walkers_per_sweep=selected_walkers_per_sweep,chunk_draws=chunk_draws,seed=seed,
        parameter_iterations=[runs,steps,walkers,cells],method='analytic conditional Gaussian moments; law of total variance; no state draws',
        uncertainty='pointwise posterior SD,not Monte Carlo error; anchored contrasts include cross-time covariance')
    cfg=folder/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Output provenance mismatch')
    cfg.write_text(json.dumps(config,indent=2)+'\n')
    with _threads('numba',1,2):
        one=_cell(p,[0]);dummy=np.asarray(u[0,0,:chunk_draws,:1]);args=_arguments(one,dummy,anchor)
        before=time.perf_counter();_trend_batch.compile(tuple(typeof(a) for a in args));compilation=time.perf_counter()-before
        before=time.perf_counter();fixed_trend_moments(one,dummy,anchor);warm=time.perf_counter()-before
        measured=time.perf_counter();rows=[]
        for g in range(cells):
            path=folder/f'cell-{g:04d}.npz';receipt=folder/f'cell-{g:04d}.json'
            if path.exists():
                row=json.loads(receipt.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Saved output changed')
                rows.append(row);continue
            if time.time()-started>=wall_seconds:raise TimeoutError('Output cap; completed cells preserved')
            cell_start=time.perf_counter();local=_cell(p,[g]);run_moments=[];selections=[]
            for run in range(runs):
                rng=np.random.default_rng(np.random.SeedSequence([seed,g,run]))
                ww=np.stack([rng.choice(walkers,selected_walkers_per_sweep,replace=False) for _ in range(steps)])
                ti=np.repeat(np.arange(steps),selected_walkers_per_sweep);wi=ww.ravel()
                selected=np.asarray(u[run,ti,wi,g:g+1]);parts=[]
                for start in range(0,len(selected),chunk_draws):
                    if time.time()-started>=wall_seconds:raise TimeoutError('Output cap; prior cells preserved')
                    parts.append(fixed_trend_moments(local,selected[start:start+chunk_draws],anchor))
                run_moments.append(np.concatenate(parts).reshape(steps,selected_walkers_per_sweep,len(p.time),4));selections.append(ww)
            moments=np.stack(run_moments)
            center=moments[...,0].mean((0,1,2));change=moments[...,2].mean((0,1,2))
            # Center before squaring: includes conditional uncertainty without
            # cancellation of large ozone offsets.
            variance=(moments[...,1]+(moments[...,0]-center)**2).mean((0,1,2))
            change_variance=(moments[...,3]+(moments[...,2]-change)**2).mean((0,1,2))
            if np.any(variance<0) or np.any(change_variance<0):raise ValueError('Negative mixture variance')
            sd=np.sqrt(variance);change_sd=np.sqrt(change_variance)
            # Cheap MC-error sensitivity for posterior means at selected times.
            indices=[0,len(p.time)//2,len(p.time)-1];series=moments[...,0].mean(2)
            taus=np.array([[autocorrelation_time(series[r,:,i]) for i in indices] for r in range(runs)])
            mcse=np.sqrt((series[:,:,indices].var(1,ddof=1)*taus/steps).sum(0)/runs**2)
            flags=['SINGLE_ENSEMBLE_OUTPUT_MCSE_PROVISIONAL'] if runs==1 else []
            if np.any(steps<50*taus):flags.append('SHORT_OUTPUT_CORRELATION_ESTIMATE')
            if np.any(mcse/sd[indices]>.05):flags.append('OUTPUT_MEAN_MCSE_OVER_0.05_POSTERIOR_SD')
            np.savez(path,mean=center,sd=sd,change_mean=change,change_sd=change_sd,time=p.time,
                selected_walker=np.stack(selections),mcse_time_indices=np.array(indices),
                selected_time_ensemble_mean=series[:,:,indices],
                first_half_mean=moments[:,:steps//2,...,0].mean((0,1,2)),second_half_mean=moments[:,steps//2:,...,0].mean((0,1,2)))
            row=dict(cell=g,panel=str(p.cell_coordinates['panel'][g]),seconds=time.perf_counter()-cell_start,flags=flags,
                mcse_indices=indices,mean_mcse=mcse.tolist(),mcse_over_posterior_sd=(mcse/sd[indices]).tolist(),maximum_iat=float(taus.max()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            receipt.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(f'trend moments {g+1}/{cells}: {row["seconds"]:.2f}s',flush=True)
        record=dict(status='PRODUCTION_TREND_ESTIMATES_WITH_FLAGS',cells=rows,panels=cells,compilation_or_cache_load_seconds=compilation,
            disposable_exact_shape_warmup_seconds=warm,post_compile_output_seconds=time.perf_counter()-measured,
            actual_wall_seconds=time.time()-started,qualification_eligible=False,**config)
        (folder/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
