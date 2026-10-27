# Contributing

Here I provide a short summary of how to contribute extended DLM models to this package. 

Note that in order to contribute new models, you will need to be able to write code in [stan](https://mc-stan.org) (click on the link for documentation and access to the stan manual). This might sound like pain for the uninitiated, but stan is a very intuitive language and you will only need to make relatively minor changes to the existing models code in order to implement new models (see below).

## Developing extended models

The DLMs are characterized by the matrices **F**, **G**, **W**, and **C** (see `models/model_descriptions/`). Add new model source strings to `src/dlmmc/_stan_models.py` and register their public names and model requirements in `src/dlmmc/api.py`. CmdStan source is packaged with dlmmc; compiled executables are cached in the user's cache directory and should not be committed.

A workflow for implementing a model is:

1. Copy and rename a model string in `src/dlmmc/_stan_models.py`.
2. Update the `data` block for any additional input data or quantities.
3. Update the `transformed data` block to construct **F** and set the state-vector size `nx` correctly.
4. Update the `transformed parameters` block to construct **G**, **W**, and **C**.
5. Update the `model` block for new hyperparameters and their priors.
6. Update `generated quantities` to expose any additional state-vector components needed for downstream analysis.
7. Register the model name and its data requirements in `src/dlmmc/api.py`, then precompile to catch Stan errors:

```bash
python compile_stan_models.py
```

Then run a representative `dlmmc.fit()` with model-appropriate xarray data, regressors, priors, and optional per-chain `inits`. Verify returned variables, dimensions, and diagnostics. Add unit tests under `tests/` for API validation and model-specific inputs. `dlm_benchmark.py` is an installation check; `dlm_validation_tests.ipynb` contains synthetic parameter-recovery examples for the packaged models.

## When a model is ready

The model source is compiled from the package automatically; no generated marker files are needed. For an end-to-end example over the gridded ozone data, see `dlm_lat_alt_mpi_run.py` and `MPI-README.md`. The MPI example runs fits on worker ranks and streams completed results to one rank-0 NetCDF writer; output files can be opened with xarray.

Add a concise description of the model to `models/model_descriptions/` by updating the relevant TeX and bibliography sources. Add or update validation coverage for the CmdStanPy workflow when contributing model behavior.

For help with model design or implementation, open an issue in this repository.
