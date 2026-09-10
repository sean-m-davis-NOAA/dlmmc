#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:29:35 2026

@author: sdavis
"""

import time, json, os, pickle, traceback
import numpy as np
import netCDF4
from utils.utils import prepare_missing_data, sampling_rate
import stan_backend
from datetime import datetime

def main():
    results = {}
    results['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    # Load model pickle (may be PyStan object or a marker for cmdstanpy)
    model_pkl = 'models/dlm_vanilla_ar1.pkl'
    if not os.path.exists(model_pkl):
        raise SystemExit(f'model pickle not found: {model_pkl}')
    model_obj = pickle.load(open(model_pkl,'rb'))
    results['model_pickle_type'] = type(model_obj).__name__
    
    # Load example timeseries from tutorial data
    ds_path = 'data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc'
    if not os.path.exists(ds_path):
        raise SystemExit(f'data file not found: {ds_path}')
    ds = netCDF4.Dataset(ds_path)
    
    # Heuristic: find a variable that depends on time and select first pressure/lat index if present
    time_dim = None
    for name, var in ds.variables.items():
        if 'time' in var.dimensions:
            time_dim = 'time'
            break
    # Choose a data variable (not the time coordinate)
    data_var = None
    for name, var in ds.variables.items():
        if name.lower() in ('time','latitude','longitude','lat','lon','pressure','plev'):
            continue
        if 'time' in var.dimensions:
            data_var = var
            data_name = name
            break
    if data_var is None:
        raise SystemExit('Could not find a suitable data variable in the netCDF file')
    print('Using data variable:', data_name, 'with dimensions', data_var.dimensions)
    # Build a 1D time-series by indexing other dims at 0 where necessary
    idx = []
    for dim in data_var.dimensions:
        if dim == 'time':
            idx.append(slice(None))
        else:
            idx.append(0)
    arr = data_var[tuple(idx)]
    d = np.array(arr, dtype=float).squeeze()
    # Try to find a corresponding stddev variable (heuristic)
    s = None
    for cand in ('stddev','std','error','uncertainty','sigma'):
        if cand in ds.variables:
            v = ds.variables[cand]
            if 'time' in v.dimensions:
                s = np.array(v[tuple(idx)], dtype=float).squeeze(); break
    # Fallback: tiny constant stddev
    if s is None:
        s = np.ones_like(d) * 1e-3
    # Prepare missing data if any
    d, s = prepare_missing_data(d, s)
    # Build input_data as in tutorial
    input_data = {
        'time_series': d,
        'stddev': s,
        'N': int(len(d)),
        'nreg': 1,
        'regressors': [[0.0] for _ in range(len(d))],
        'sampling': sampling_rate('monthly'),
        'S': 10.0,
        'sigma_trend_prior':1e-4,
        'sigma_seas_prior':0.01,
        'sigma_AR_prior':0.5
    }
    # Sampling settings (full test as requested)
    iterations = 100 + 100  # total not used directly for cmdstanpy; we use warmup+sampling below
    warmup = 100
    sampling = 100
    chains = 4
    results['settings'] = dict(warmup=warmup, sampling=sampling, chains=chains)
    # Prepare results output dir
    os.makedirs('results', exist_ok=True)
    out_json = f'results/benchmark_{int(time.time())}.json'
    # Try PyStan (legacy) if available
    backend_results = {}
    try:
        import pystan
        HAVE_PYSTAN = True
    except Exception:
        HAVE_PYSTAN = False
    # If model_obj is a PyStan model or HAVE_PYSTAN, attempt PyStan sampling
    if HAVE_PYSTAN:
        try:
            # If model_obj is a PyStan StanModel pickled, use it; else compile from .stan if source available
            from time import perf_counter
            py_res = {}
            if hasattr(model_obj, 'model_code') or hasattr(model_obj, 'stan_json') or 'pystan' in str(type(model_obj)).lower():
                print('Detected PyStan model in pickle; will run PyStan sampling')
                stan_model = model_obj
                compile_time = 0.0
            else:
                # try to compile from source .stan if present
                stan_src = 'models/dlm_vanilla_ar1.stan'
                if os.path.exists(stan_src):
                    print('Compiling PyStan model from', stan_src)
                    code = open(stan_src).read()
                    t0 = perf_counter()
                    stan_model = pystan.StanModel(model_code=code)
                    compile_time = perf_counter() - t0
                else:
                    raise RuntimeError('No Stan source available to compile PyStan model')
            py_res['compile_time'] = compile_time
            # Run sampling
            t0 = perf_counter()
            fit = stan_model.sampling(data=input_data, iter=warmup+sampling, warmup=warmup, chains=chains, n_jobs=chains)
            sample_time = perf_counter() - t0
            py_res['sample_time'] = sample_time
            # Attempt to gather simple diagnostics
            try:
                summary = fit.summary(pars=['sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1'])
                py_res['pystan_summary_keys'] = list(summary.keys())
            except Exception as e:
                py_res['pystan_summary_error'] = str(e)
            backend_results['pystan'] = py_res
        except Exception as e:
            backend_results['pystan_error'] = traceback.format_exc()
    else:
        backend_results['pystan_error'] = 'pystan not available in this environment'
    # CmdStanPy benchmark
    try:
        from time import perf_counter
        cs_res = {}
        # If model_obj is a marker dict pointing to stan file, use that; else use models/dlm_vanilla_ar1.stan
        if isinstance(model_obj, dict) and model_obj.get('backend') == 'cmdstanpy':
            stan_file = model_obj['stan_file']
        else:
            stan_file = 'models/dlm_vanilla_ar1.stan'
        t0 = perf_counter()
        from cmdstanpy import CmdStanModel
        model = CmdStanModel(stan_file=stan_file)
        cs_res['compile_time'] = perf_counter() - t0
        # run sampling
        t0 = perf_counter()
        mcmc = model.sample(data=input_data, chains=chains, iter_sampling=sampling, iter_warmup=warmup, show_console=True)
        cs_res['sample_time'] = perf_counter() - t0
        # diagnostics: try to get divergent__ if present
        try:
            df = mcmc.draws_pd()
            if 'divergent__' in df.columns:
                cs_res['n_divergent'] = int(df['divergent__'].sum())
            else:
                cs_res['n_divergent'] = None
            # basic R_hat/ESS via summary if available
            try:
                summary_df = mcmc.summary()
                # List the specific parameters you want to keep
                pars_to_keep = ['sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1', 'lp__']
                
                # Filter the pandas dataframe to only include these rows
                trimmed_summary = summary_df.loc[summary_df.index.isin(pars_to_keep)]
                
                # Convert the trimmed dataframe to a dict for the JSON file
                cs_res['summary'] = trimmed_summary.to_dict()                
            except Exception as e:
                cs_res['summary_error'] = str(e)
        except Exception as e:
            cs_res['draws_error'] = str(e)
        backend_results['cmdstanpy'] = cs_res
    except Exception as e:
        backend_results['cmdstanpy_error'] = traceback.format_exc()
    results['backends'] = backend_results
    with open(out_json,'w') as f:
        json.dump(results, f, indent=2)
    print('Benchmark finished; results written to', out_json)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("fork")
    main()