# Installation and validation

The recommended workflow uses CmdStanPy. The commands below install the Python dependencies for the benchmark and MPI example, install CmdStan, and run a small validation.

## Create the environment

For a local workstation, create the environment with Conda:

```bash
conda create -n dlmmc-cmdstanpy -c conda-forge python=3.10 cmdstanpy netCDF4 numpy scipy xarray arviz tqdm mpi4py openmpi matplotlib jupyter ipykernel pandas -y
conda activate dlmmc-cmdstanpy
```

CmdStan needs a working C++ toolchain. On macOS, install the Xcode command-line tools (`xcode-select --install`). On Linux, use the system compiler or install a Conda compiler with `conda install -c conda-forge cxx-compiler make`.

Install CmdStan in the environment:

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

The environment includes Jupyter and `ipykernel`. In VS Code, open the notebook and select `Python (dlmmc-cmdstanpy)` as its kernel. If Jupyter runs in a separate environment, register this environment as a kernel:

```bash
python -m ipykernel install --user --name dlmmc-cmdstanpy --display-name "Python (dlmmc-cmdstanpy)"
```

On HPC systems, use the site's compiler and MPI modules. Ensure `mpi4py` is installed against the MPI implementation used by the scheduler; do not mix the site's MPI libraries with Conda Open MPI.

## Generate and test models

From the repository root, generate the CmdStanPy model sources and marker files:

```bash
python compile_stan_models.py
```

The generator writes versioned files with the `_cmdstanpy` suffix and canonical model files such as `models/dlm_vanilla_ar1.pkl`, which the benchmark and MPI example load. The marker points to the generated Stan source. CmdStanPy compiles the native executable when the model is first instantiated. Run the benchmark as an end-to-end check; it samples one example series with four chains, 100 warmup iterations, and 100 sampling iterations:

```bash
python dlm_benchmark.py
```

The benchmark writes a JSON result under `results/`. It may take several minutes on the first run while CmdStan compiles the model.
To verify success, open the newest `results/benchmark_*.json` file and confirm `backends.cmdstanpy` contains `compile_time` and `sample_time`, with no `backends.cmdstanpy_error` entry.

## Run the notebooks

After the benchmark succeeds, open `dlm_tutorial.ipynb` in JupyterLab or VS Code and run its cells in order. It demonstrates xarray-based time/data handling, CmdStanPy sampling, and posterior diagnostics. It uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

Next, run `dlm_validation_tests.ipynb`. It generates synthetic data with `utils/dlm.py` and tests parameter recovery for four model variants. Each test uses four chains with 250 warmup and 500 sampling iterations per chain and reports posterior intervals, R-hat/ESS, trend coverage, and divergences. These are validation demonstrations; warnings or missed recovery intervals indicate that a model/parameter needs further investigation, not necessarily an installation failure.

## Test the MPI example

The MPI example fits the pressure/latitude grid in parallel and streams completed fits to one rank-0 writer. The resulting NetCDF is readable with xarray. On a local machine with Open MPI, run:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The example defaults to one chain, one warmup iteration, and two sampling iterations per grid cell. These settings are only for a pipeline smoke test and do not provide converged posterior estimates. Increase them for scientific runs. The output is `results/BASIC_V1_2017.nc`; `fit_status` is 1 for completed cells, 0 for skipped cells, and -1 for failed fits.

On an HPC system, launch the script using the site's MPI launcher and scheduler allocation. See `MPI-README.md` for details. Rank 0 is reserved for writing when multiple ranks are used.

If CmdStan reports that its precompiled header needs rebuilding, run:

```bash
python -c "import cmdstanpy; cmdstanpy.rebuild_cmdstan()"
```

On macOS, long-running compilation and sampling can be interrupted by system sleep. Use `caffeinate -i <command>` or run on a remote system or in a `tmux` session.
