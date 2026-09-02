from models.stan_dlm_models import *
from stan_backend import write_model_files

# Helper to write stan file and either compile via pystan or write a marker for cmdstanpy
models_to_write = [
    ('dlm_vanilla_ar1', dlm_vanilla_ar1),
    ('dlm_vanilla_ar2', dlm_vanilla_ar2),
    ('dlm_noregs_ar1', dlm_noregs_ar1),
    ('dlm_dynregs_ar1', dlm_dynregs_ar1),
    ('dlm_vanilla_ar1_noseasonal', dlm_vanilla_ar1_noseasonal),
    ('dlm_vanilla_ar2_noseasonal', dlm_vanilla_ar2_noseasonal),
]

write_model_files(models_to_write)


