# Contributing

Here I provide a short summary of how to contribute extended DLM models to this package. 

Note that in order to contribute new models, you will need to be able to write code in [stan](https://mc-stan.org) (click on the link for documentation and access to the stan manual). This might sound like pain for the uninitiated, but stan is a very intuitive language and you will only need to make relatively minor changes to the existing models code in order to implement new models (see below).

## Developing extended models

The DLMs are characterized by the matrices **F**, **G**, **W**, and **C** (see the model description in `models/model_descriptions/`). For new development, use the CmdStanPy model source in `models/cmdstanpy_dlm_models.py`; `models/pystan2_dlm_models.py` is retained for legacy compatibility. Add each model as a module-level Stan source string. Generated `.stan` files, marker pickles, and compiled executables are build artifacts and should not be committed.

A workflow for implementing a model is:

1. Copy and rename a model string in `models/cmdstanpy_dlm_models.py`.
2. Update the `data` block for any additional input data or quantities.
3. Update the `transformed data` block to construct **F** and set the state-vector size `nx` correctly.
4. Update the `transformed parameters` block to construct **G**, **W**, and **C**.
5. Update the `model` block for new hyperparameters and their priors.
6. Update `generated quantities` to expose any additional state-vector components needed for downstream analysis.
7. Regenerate model source and marker files, then compile the model with CmdStanPy:

```bash
python compile_stan_models.py
python -c "from cmdstanpy import CmdStanModel; CmdStanModel(stan_file='models/my_model_cmdstanpy.stan')"
```

Run a small representative sample with the model's actual input data and verify the returned variables and dimensions. `dlm_benchmark.py` exercises the vanilla AR1 model; it is a useful installation check, but does not validate a newly added model. The existing `dlm_validation_tests.ipynb` uses the legacy PyStan API and is not currently a CmdStanPy test suite.

## When a model is ready

The model source is discovered automatically by `compile_stan_models.py`; no hand-written pickle or registration code is needed. For an end-to-end example over the gridded ozone data, see `dlm_lat_alt_mpi_run.py` and `MPI-README.md`. The MPI example runs fits on worker ranks and streams completed results to one rank-0 NetCDF writer; output files can be opened with xarray.

Add a concise description of the model to `models/model_descriptions/` by updating the relevant TeX and bibliography sources. Add or update validation coverage for the CmdStanPy workflow when contributing model behavior.

For help with model design or implementation, open an issue in this repository.
