import os
import pickle
import re

try:
    import pystan
    HAVE_PYSTAN = True
except Exception:
    HAVE_PYSTAN = False

try:
    from cmdstanpy import CmdStanModel
    HAVE_CMDSTANPY = True
except Exception:
    HAVE_CMDSTANPY = False


class FitShim:
    def __init__(self, extract_dict):
        self._extract = extract_dict

    def extract(self):
        return self._extract


def detect_backend():
    if HAVE_PYSTAN:
        return 'pystan'
    if HAVE_CMDSTANPY:
        return 'cmdstanpy'
    return None


def _extract_from_cmdstan(mcmc):
    try:
        df = mcmc.draws_pd()
        import numpy as np
        var_dict = {}
        for col in df.columns:
            match = re.match(r'(?P<name>[^\[]+)(\[(?P<index>.+)\])?', col)
            if match is None:
                continue
            name = match.group('name')
            var_dict.setdefault(name, []).append(df[col].values)
        extract_dict = {}
        for name, arrays in var_dict.items():
            arr = np.vstack(arrays).T
            if arr.shape[1] == 1:
                extract_dict[name] = arr[:, 0]
            else:
                extract_dict[name] = arr
        return extract_dict
    except Exception:
        extract_dict = {}
        for name in ('sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1', 'trend', 'slope', 'beta', 'seasonal'):
            try:
                extract_dict[name] = mcmc.stan_variable(name)
            except Exception:
                pass
        return extract_dict


def write_model_files(model_defs):
    os.makedirs('models', exist_ok=True)
    written = []
    for name, code in model_defs:
        stan_path = os.path.join('models', name + '.stan')
        with open(stan_path, 'w') as fh:
            fh.write(code)

        pkl_path = os.path.join('models', name + '.pkl')
        if HAVE_PYSTAN:
            model = pystan.StanModel(model_code=code)
            with open(pkl_path, 'wb') as fh:
                pickle.dump(model, fh)
        else:
            marker = {'stan_file': stan_path, 'backend': 'cmdstanpy'}
            with open(pkl_path, 'wb') as fh:
                pickle.dump(marker, fh)
        written.append((name, stan_path, pkl_path))
    return written


def load_model_object(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def sample_with_backend(model_obj, input_data, iterations, warmup, chains, init=None, params=None):
    if isinstance(model_obj, dict) and model_obj.get('backend') == 'cmdstanpy':
        if not HAVE_CMDSTANPY:
            raise ImportError('cmdstanpy is required for cmdstanpy backend. Install via pip: pip install cmdstanpy')
        model = CmdStanModel(stan_file=model_obj['stan_file'])
        iter_sampling = max(1, iterations - warmup)
        mcmc = model.sample(data=input_data, chains=chains, iter_sampling=iter_sampling, iter_warmup=warmup)
        return FitShim(_extract_from_cmdstan(mcmc))

    if not HAVE_PYSTAN:
        raise ImportError('PyStan is required for the legacy PyStan backend.')
    return model_obj.sampling(
        data=input_data,
        iter=iterations,
        warmup=warmup,
        chains=chains,
        init=init,
        verbose=False,
        pars=params,
    )
