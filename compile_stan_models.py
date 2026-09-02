try:
    import pystan
    HAVE_PYSTAN = True
except Exception:
    HAVE_PYSTAN = False
import pickle
import os
from models.stan_dlm_models import *

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Helper to write stan file and either compile via pystan or write a marker for cmdstanpy
models_to_write = [
    ('dlm_vanilla_ar1', dlm_vanilla_ar1),
    ('dlm_vanilla_ar2', dlm_vanilla_ar2),
    ('dlm_noregs_ar1', dlm_noregs_ar1),
    ('dlm_dynregs_ar1', dlm_dynregs_ar1),
    ('dlm_vanilla_ar1_noseasonal', dlm_vanilla_ar1_noseasonal),
    ('dlm_vanilla_ar2_noseasonal', dlm_vanilla_ar2_noseasonal),
]

for name, code in models_to_write:
    stan_path = os.path.join('models', name + '.stan')
    with open(stan_path, 'w') as fh:
        fh.write(code)

    pkl_path = os.path.join('models', name + '.pkl')

    if HAVE_PYSTAN:
        # Compile with PyStan and pickle the compiled model (original behaviour)
        model = pystan.StanModel(model_code=code)
        with open(pkl_path, 'wb') as fh:
            pickle.dump(model, fh)
    else:
        # PyStan not available: write a small marker pickle that points to the stan file and
        # indicates the cmdstanpy backend can be used to compile/run it at runtime.
        marker = {'stan_file': stan_path, 'backend': 'cmdstanpy'}
        with open(pkl_path, 'wb') as fh:
            pickle.dump(marker, fh)


