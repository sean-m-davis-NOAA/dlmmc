"""Full-panel production smoothing with compact matrix construction/reduction.

The scalar Gaussian smoother remains the same tested Joseph/RTS routine.
Parameter-independent F/G/Q/C entries are stored once per panel. No full-grid
latent histories and no sixteen-draw Python loop are materialized.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch
from numba import njit, prange, typeof
from .tensor_target import model_components
from .production_trends import _trend_one
from .production import production_parameter_cache
from .sampling import _threads
from .diagnostics import autocorrelation_time


def compact_arguments(p, unconstrained, anchor):
    """All selected draws × all panels; only five matrix entries vary by draw."""
    u=torch.as_tensor(np.asarray(unconstrained),dtype=torch.float64)
    if (p.y.device.type!='cpu' or p.y.dtype!=torch.float64
            or u.ndim!=3 or u.shape[1:]!=(p.cells,len(p.parameter_names)) or not len(u)):
        raise ValueError('CPU64 draws/panels/model parameter dimension required')
    if not 0<=anchor<len(p.time):raise ValueError('Valid anchor required')
    with torch.no_grad():
        F,G,Q,C,_,_=model_components(u[:1],*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
        if p.model=='vanilla_ar1':
            sig=torch.exp(u[...,:3]);rho=torch.sigmoid(u[...,3])
            if not bool(torch.isfinite(u).all() and torch.isfinite(sig).all() and (sig>0).all() and (rho>0).all() and (rho<1).all()):
                raise ValueError('All selected parameters must be valid; no draw removal')
            q=sig*sig;cv=q[...,2]/(1-rho*rho)
            varying=torch.stack((q[...,0],q[...,1],q[...,2],rho,cv),-1).permute(1,0,2).contiguous()
        else:
            from .api import _MODEL_SPECS
            spec=_MODEL_SPECS[p.model];names=p.parameter_names;theta={}
            valid=torch.isfinite(u).all(-1)
            for j,name in enumerate(names):
                if name.startswith('sigma'):value=u[...,j].exp();valid=valid&torch.isfinite(value)&(value>0)
                else:
                    prob=u[...,j].sigmoid();valid=valid&(prob>0)&(prob<1)
                    value=prob*((.5 if spec['ar2'] else 1.) if name=='rhoAR1' else 1-theta['rhoAR1'])
                theta[name]=value
            if not bool(valid.all()):raise ValueError('All selected parameters must be valid; no draw removal')
            d=G.shape[-1];base=6 if spec['seasonal'] else 2;ar=base+p.regressors.shape[-1]
            diag=Q[0].diagonal(dim1=-2,dim2=-1).expand(len(u),-1,-1).clone()
            diag[...,1]=theta['sigma_trend']**2
            if spec['seasonal']:diag[...,2:6]=theta['sigma_seas'][...,None]**2
            if spec['regressors']=='dynamic':
                for j in range(p.regressors.shape[-1]):diag[...,base+j]=theta[f'sigma_reg[{j}]']**2
            diag[...,ar]=theta['sigma_AR']**2;a=theta['rhoAR1'];b=theta.get('rhoAR2',torch.zeros_like(a))
            v=theta['sigma_AR']**2/(1-a*a-b*b-2*b*a*a/(1-b)) if spec['ar2'] else theta['sigma_AR']**2/(1-a*a)
            cross=v*a/(1-b) if spec['ar2'] else torch.zeros_like(v)
            extra=torch.stack((a,b,v,cross,torch.full_like(a,ar),torch.full_like(a,int(spec['ar2']))),-1)
            varying=torch.cat((diag,extra),-1).permute(1,0,2).contiguous()
    return tuple(np.ascontiguousarray(z.numpy()) for z in
        (p.y,p.variance,p.observed,F[0],G[0],Q[0],C[0],varying))+(int(anchor),)


@njit(cache=True,fastmath=False)
def _matrices(G,Q,C,values):
    g=G.copy();q=Q.copy();c=C.copy();ar=len(G)-1
    if len(values)==5:
        q[1,1]=values[0]
        for i in range(2,6):q[i,i]=values[1]
        q[ar,ar]=values[2];g[ar,ar]=values[3];c[ar,ar]=values[4]
    else:
        d=len(G);ar=int(values[-2]);ar2=bool(values[-1])
        for i in range(d):q[i,i]=values[i]
        g[ar,ar]=values[d];c[ar,ar]=values[d+2]
        if ar2:
            g[ar,ar+1]=values[d+1];c[ar+1,ar+1]=values[d+2]
            c[ar,ar+1]=values[d+3];c[ar+1,ar]=values[d+3]
    return g,q,c


@njit(cache=True,parallel=True,fastmath=False)
def _fixed_grid(y,variance,observed,F,G,Q,C,values,anchor):
    cells,draws=values.shape[:2];n=y.shape[1]
    out=np.empty((draws,cells,n,4))
    for i in prange(draws*cells):
        s=i//cells;g=i%cells
        gg,qq,cc=_matrices(G[g],Q[g],C[g],values[g,s])
        out[s,g]=_trend_one(y[g],variance[g],observed[g],F[g],gg,qq,cc,anchor)
    return out


def fixed_grid_moments(p,u,anchor):
    out=_fixed_grid(*compact_arguments(p,u,anchor))
    out[:,:,:,0]=out[:,:,:,0]*p.scale[None,:,None]+p.offset[None,:,None]
    out[:,:,:,1]*=p.scale[None,:,None]**2
    out[:,:,:,2]*=p.scale[None,:,None];out[:,:,:,3]*=p.scale[None,:,None]**2
    return out


@njit(cache=True,parallel=True,fastmath=False)
def _reduce_grid(y,variance,observed,F,G,Q,C,values,anchor,per_sweep):
    """One compiled call for every panel and every selected draw, online moments."""
    cells,draws=values.shape[:2];n=y.shape[1];steps=draws//per_sweep
    result=np.empty((cells,n,4));halves=np.zeros((cells,2,n))
    series=np.zeros((cells,steps,3));indices=np.array([0,n//2,n-1])
    for g in prange(cells):
        mean=np.zeros((n,2));m2=np.zeros((n,2));conditional=np.zeros((n,2))
        for s in range(draws):
            gg,qq,cc=_matrices(G[g],Q[g],C[g],values[g,s])
            moments=_trend_one(y[g],variance[g],observed[g],F[g],gg,qq,cc,anchor)
            for t in range(n):
                for k in range(2):
                    delta=moments[t,2*k]-mean[t,k]
                    mean[t,k]+=delta/(s+1)
                    m2[t,k]+=delta*(moments[t,2*k]-mean[t,k])
                    conditional[t,k]+=moments[t,2*k+1]
                halves[g,0 if s<draws//2 else 1,t]+=moments[t,0]/(draws//2)
            for k in range(3):series[g,s//per_sweep,k]+=moments[indices[k],0]/per_sweep
        for t in range(n):
            for k in range(2):
                result[g,t,2*k]=mean[t,k]
                v=(conditional[t,k]+m2[t,k])/draws
                result[g,t,2*k+1]=v
    return result,halves,series


def summarize_full_batch(fit,*,output_dir,anchor_month='1998-01',selected_walkers_per_sweep=4,
        seed=20261116,wall_seconds=5400,numba_threads=8):
    if fit.config.get('execution_profile',{}).get('name')!='production':raise ValueError('Production fit required')
    started=time.monotonic();folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'execution.json').exists():raise ValueError('Preserve completed output')
    p=fit.prepared;matches=np.flatnonzero(p.time.astype('datetime64[M]')==np.datetime64(anchor_month,'M'))
    if len(matches)!=1:raise ValueError('Unique anchor month required')
    anchor=int(matches[0]);u=production_parameter_cache(fit);runs,steps,walkers,cells,_=u.shape
    if runs!=1 or steps%2 or not 1<=selected_walkers_per_sweep<=walkers:
        raise ValueError('One ensemble, even retained window and valid selection count required')
    selections=[];selected=np.empty((steps*selected_walkers_per_sweep,cells,len(p.parameter_names)))
    for g in range(cells):
        rng=np.random.default_rng(np.random.SeedSequence([seed,g,0]))
        # A nested, value-independent family: one/two/four use the same base four.
        count=max(4,selected_walkers_per_sweep)
        if count>walkers:count=walkers
        ww=np.stack([rng.choice(walkers,count,replace=False) for _ in range(steps)])[:,:selected_walkers_per_sweep]
        selected[:,g]=u[0,np.repeat(np.arange(steps),selected_walkers_per_sweep),ww.ravel(),g]
        selections.append(ww)
    selection_seconds=time.monotonic()-started
    with _threads('numba',1,numba_threads):
        before=time.perf_counter();args=compact_arguments(p,selected,anchor)
        preparation=time.perf_counter()-before
        before=time.perf_counter();_reduce_grid.compile(tuple(typeof(x) for x in args)+(typeof(selected_walkers_per_sweep),))
        compilation=time.perf_counter()-before
        print(f'Full-batch smoother: disposable warmup {len(selected)} draws x {cells} panels, {numba_threads} threads',flush=True)
        before=time.perf_counter();_reduce_grid(*args,selected_walkers_per_sweep)
        warm=time.perf_counter()-before
        signatures=list(map(str,_reduce_grid.signatures))
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('Output preparation cap')
        print(f'Smoother warmup {warm:.3f}s; starting timed full batch',flush=True)
        before=time.perf_counter();result,halves,series=_reduce_grid(*args,selected_walkers_per_sweep)
        compute=time.perf_counter()-before
        print(f'Timed full-batch smoother {compute:.3f}s',flush=True)
        if signatures!=list(map(str,_reduce_grid.signatures)):raise RuntimeError('Compilation during timed reduction')
    before=time.perf_counter();rows=[];indices=[0,len(p.time)//2,len(p.time)-1]
    if not np.isfinite(result).all() or np.any(result[:,:,1::2]<0):
        raise ValueError('Invalid mixture variance; no floor')
    result[:,:,1::2]=np.sqrt(result[:,:,1::2])
    result[:,:,0]=result[:,:,0]*p.scale[:,None]+p.offset[:,None]
    result[:,:,1:]*=p.scale[:,None,None]
    halves=halves*p.scale[:,None,None]+p.offset[:,None,None]
    series=series*p.scale[:,None,None]+p.offset[:,None,None]
    for g in range(cells):
        mean,sd,change,change_sd=result[g].T
        tau=np.array([autocorrelation_time(series[g,:,k]) for k in range(3)])
        mcse=np.sqrt(series[g].var(0,ddof=1)*tau/steps)
        flags=['SINGLE_ENSEMBLE_OUTPUT_MCSE_PROVISIONAL']
        if np.any(steps<50*tau):flags.append('SHORT_OUTPUT_CORRELATION_ESTIMATE')
        if np.any(mcse/sd[indices]>.05):flags.append('OUTPUT_MEAN_MCSE_OVER_0.05_POSTERIOR_SD')
        path=folder/f'cell-{g:04d}.npz'
        np.savez(path,mean=mean,sd=sd,change_mean=change,change_sd=change_sd,time=p.time,
            selected_walker=np.asarray(selections[g])[None],mcse_time_indices=np.array(indices),
            selected_time_ensemble_mean=series[g][None],first_half_mean=halves[g,0],second_half_mean=halves[g,1])
        row=dict(cell=g,panel=str(p.cell_coordinates['panel'][g]),flags=flags,mcse_indices=indices,
            mean_mcse=mcse.tolist(),mcse_over_posterior_sd=(mcse/sd[indices]).tolist(),maximum_iat=float(tau.max()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (folder/f'cell-{g:04d}.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    io=time.perf_counter()-before
    record=dict(status='PRODUCTION_TREND_ESTIMATES_WITH_FLAGS',qualification_eligible=False,panels=cells,cells=rows,
        anchor_month=anchor_month,anchor_index=anchor,fit_provenance=fit.config['provenance'],seed=seed,
        selected_walkers_per_sweep=selected_walkers_per_sweep,parameter_iterations=[runs,steps,walkers,cells],
        threads={'torch':1,'numba':numba_threads},full_batch_shape=[len(selected),cells],compiled_output_calls=1,
        draw_chunking=False,panel_chunking=False,selection_seconds=selection_seconds,preparation_seconds=preparation,
        compilation_or_cache_load_seconds=compilation,disposable_exact_shape_warmup_seconds=warm,
        smoothing_and_reduction_seconds=compute,diagnostics_and_output_io_seconds=io,
        post_compile_output_seconds=time.monotonic()-started-compilation-warm,actual_wall_seconds=time.monotonic()-started,
        cap_seconds=wall_seconds,cap_exceeded=time.monotonic()-started>wall_seconds,
        source_identity={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in
            ('production_smoothing.py','production_trends.py','production.py','tensor_target.py')})
    (folder/'execution.json').write_text(json.dumps(record,indent=2)+'\n');return record
