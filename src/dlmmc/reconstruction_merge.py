"""Combine additional state selections without discarding any earlier draws."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .reconstruction import ReconstructionResult


def combine_reconstructions(results, *, output_dir, chunk_draws=16):
    """Merge selections from one posterior, preserving run and temporal order.

    Additional selections use distinct RNG seeds. They are not independent
    MCMC runs. Original draws, including repeated source parameters, are all
    retained; output precision must be diagnosed again. Inputs are immutable.
    Memory is bounded to two input chunks and one output chunk.
    """
    paths = [Path(r.path) for r in results]
    if len(paths) < 2 or chunk_draws < 1:
        raise ValueError('Require multiple selections and positive chunk size')
    configs = [json.loads((p/'config.json').read_text()) for p in paths]
    indexes = [json.loads((p/'index.json').read_text()) for p in paths]
    keys = ('posterior_sha256','model','cells','cell_coordinates','parameter_names',
            'nreg','offset','scale','dtype','device','source_sha256')
    if any(any(c[k] != configs[0][k] for k in keys) for c in configs[1:]):
        raise ValueError('Selections must have identical posterior and scientific provenance')
    if len({c['seed'] for c in configs}) != len(configs):
        raise ValueError('Selections must use distinct RNG seeds')
    if any(i['status'] != 'RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING' for i in indexes):
        raise ValueError('All input selections must be complete')
    parents = [{str(p.resolve() / f):hashlib.sha256((p/f).read_bytes()).hexdigest()
                for f in ('config.json','index.json')} for p in paths]
    records = {}; time = None; expected_runs = None
    for source,(p,idx,cfg) in enumerate(zip(paths,indexes,configs)):
        counts = {}
        for row in idx['chunks']:
            file = p/row['file']
            if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Input chunk changed')
            with np.load(file,allow_pickle=False) as z:
                if time is not None and not np.array_equal(time,z['time']):
                    raise ValueError('Time coordinates differ')
                time=z['time'].copy()
                for j,(draw,walker) in enumerate(zip(z['source_draw'],z['source_walker'])):
                    records.setdefault(row['run'],[]).append((int(draw),source,int(z['selected'][j]),str(file),j))
                counts[row['run']]=counts.get(row['run'],0)+len(z['selected'])
        if expected_runs is None: expected_runs = set(counts)
        if not counts or any(n != cfg['selected_per_run'] for n in counts.values()) or set(counts) != expected_runs:
            raise ValueError('Unbalanced input selections')
    out=Path(output_dir)
    if out.exists():raise ValueError('Preserve existing combined selection')
    out.mkdir(parents=True)
    config=dict(configs[0]);config.update(selected_per_run=sum(c['selected_per_run'] for c in configs),
        chunk_draws=chunk_draws,seed=None,selection_parents=parents,
        selection_policy='All input draws, stable time ordering within original run; no new MCMC',
        merge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    index={'status':'RECONSTRUCTION_INCOMPLETE','chunks':[]}
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    for run,rows in sorted(records.items()):
        rows.sort(key=lambda r:r[:3]);cache={}
        for start in range(0,len(rows),chunk_draws):
            arrays={}
            for draw,source,selected,file,j in rows[start:start+chunk_draws]:
                if file not in cache:
                    if len(cache)>=2:cache.pop(next(iter(cache)))
                    with np.load(file,allow_pickle=False) as z:cache[file]={k:z[k] for k in z.files}
                for k,v in cache[file].items():
                    if k not in ('time','selected'):arrays.setdefault(k,[]).append(v[j].copy())
            arrays={k:np.stack(v) for k,v in arrays.items()}
            arrays.update(time=time,selected=np.arange(start,start+len(rows[start:start+chunk_draws])))
            name=f'run-{run}-selected-{start:07d}.npz';file=out/name
            np.savez(file,**arrays)
            index['chunks'].append(dict(file=name,run=run,start=start,count=len(arrays['selected']),
                sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    index['status']='RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING'
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return ReconstructionResult(out)
