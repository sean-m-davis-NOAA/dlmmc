# Installation and validation

The recommended workflow installs CmdStanPy and its CmdStan dependency with
Conda, then the released dlmmc package from PyPI. A source checkout is not needed
for installation. Notebook/example dependencies and MPI support are optional
and installed separately.

## C++ compiler and Make

CmdStan compiles models into executables, so it needs a C++ compiler and Make.
On macOS or Linux, check in a terminal:

```bash
c++ --version
make --version
```

Both commands should print version information, not "command not found".
This checks tool availability; the benchmark below tests model compilation.

- **macOS:** if either tool is missing, run `xcode-select --install` and complete
  the Xcode Command Line Tools installer. You do not need the full Xcode app.
  Repeat the version checks after installation.
- **Ubuntu/Debian Linux:** install the compiler and Make with
  `sudo apt update` followed by `sudo apt install build-essential`, then repeat
  the checks. On other distributions, use the equivalent development-tool
  packages.
- **Linux without administrator access:** create and activate `dlmmc-env` as
  described below, then run `conda install -c conda-forge cxx-compiler make`.
  Reactivate the environment afterward. Check the activated compiler with
  `"$CXX" --version` and Make with `make --version`; the Conda compiler may use
  a prefixed name rather than `c++`.
- **HPC systems:** use the site's compiler module and scheduler instructions.
  Load the module before running the checks, and use the same environment when
  compiling and running models. Ask your system administrator if no compiler is
  available.

## Create the environment

For a local workstation, create the environment with Conda:

```bash
conda create -n dlmmc-env -c conda-forge python pip cmdstanpy -y
conda activate dlmmc-env
python -m pip install dlmmc
```

Conda installs CmdStan automatically through CmdStanPy. No specific Python
version needs to be selected; dlmmc requires Python 3.10 or newer. Pip installs
the released dlmmc package in non-editable mode. For the upcoming Conda-only
alternative, see [README.md](README.md#installation).

To obtain the example notebooks, scripts, and data, clone the repository or download its source
archive as described in [README.md](README.md#getting-started). Run the examples
below from that repository root after installing the optional example
dependencies below; there is no need to reinstall dlmmc.
Contributors can instead install a checkout with
`python -m pip install -e '.[diagnostics]'` so local source edits take effect
without reinstalling.

Conda installs CmdStan in the environment; no separate download/build is needed.
Verify that Python can locate it:

```bash
# Optional check: verify imports and CmdStan discovery.
# Expected output: CmdStan: <path to CmdStan in the active Conda environment>
python -c "import cmdstanpy, dlmmc; print('CmdStan:', cmdstanpy.cmdstan_path())"
```

## Optional notebook and example dependencies

To run the example notebooks, benchmark, and plots, install:

```bash
conda install -c conda-forge jupyterlab ipykernel matplotlib scipy netcdf4 "arviz>=1.3,<2" -y
```

The notebooks pass their labeled xarray posterior directly to modern ArviZ
summaries. The optional ArviZ 1.3 stack requires Python 3.12 or newer; Conda
selects a compatible version automatically. The core package still supports
Python 3.10 or newer without ArviZ. For a pip-only diagnostics installation,
use `python -m pip install 'dlmmc[diagnostics]'` in a Python 3.12+ environment.

JupyterLab is not needed when using VS Code; omit `jupyterlab` from the command
if you prefer. In VS Code, open the notebook and select `dlmmc-env` as its kernel.
If Jupyter runs in a separate environment, register this environment as a kernel:

```bash
python -m ipykernel install --user --name dlmmc-env --display-name "Python (dlmmc-env)"
```

## Optional MPI support

For MPI grid fitting on a local macOS/Linux workstation:

```bash
conda install -c conda-forge mpi4py openmpi netcdf4 scipy -y
```

This includes NetCDF support and SciPy for the grid example without requiring the
notebook packages. On HPC systems, use the site's compiler and MPI modules.
Ensure `mpi4py` is installed against the MPI implementation used by the scheduler;
do not mix the site's MPI libraries with Conda Open MPI.
See [MPI-README.md](MPI-README.md) for launch instructions.

## Temporary storage

Stan can write large CSV files, even though the public API returns only selected
posterior variables. Successful fits remove their per-fit temporary output after
the posterior and diagnostics have been loaded into memory, including fits made
by MPI workers. Failed or interrupted fits retain their output and log its
location; inspect those files and remove them manually when no longer needed.

Choose a writable scratch directory with sufficient space and quota before
launching Python or a notebook kernel:

```bash
mkdir -p /path/to/your/scratch/dlmmc-tmp
export TMPDIR=/path/to/your/scratch/dlmmc-tmp
```

Replace the example path with your actual scratch location. For VS Code, ensure
the kernel inherits this setting, or restart the kernel and run the following
cell before importing dlmmc:

```python
import os
import tempfile
from pathlib import Path

scratch = Path("/path/to/your/scratch/dlmmc-tmp")
scratch.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(scratch)
tempfile.tempdir = str(scratch)
print("Temporary directory:", tempfile.gettempdir())
```

To keep raw Stan output, pass `output_dir="/path/to/output"` to `dlmmc.fit()`.
Explicit output directories are never removed by dlmmc. `cache_dir` controls
compiled models, not sampling output.

## Precompile and test models

Model sources ship inside the Python package and compile lazily on the first fit. To precompile every model from the repository root:

```bash
python compile_stan_models.py
```

Run the benchmark as an end-to-end check; it uses `dlmmc.fit()` on one example series with four chains, 100 warmup iterations, and 100 sampling iterations:

```bash
python dlm_benchmark.py
```

The benchmark writes a JSON result under `results/`. It may take several minutes on the first run while CmdStan compiles the model. To verify success, open the newest `results/benchmark_*.json` and confirm it contains `compile_seconds`, `fit_seconds`, posterior dimensions, and variables.

## Run the notebooks

After the benchmark succeeds, open `dlm_tutorial.ipynb` in JupyterLab or VS Code and run its cells in order. It demonstrates xarray-based time/data handling, CmdStanPy sampling, and posterior diagnostics. It uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

Next, run `dlm_validation_tests.ipynb`. It generates synthetic data with `utils/dlm.py` and calls the public `dlmmc.fit()` API to test parameter recovery for four model variants. Each test uses four chains with 250 warmup and 500 sampling iterations per chain and reports posterior intervals, R-hat/ESS, trend coverage, and divergences. These are validation demonstrations; warnings or missed recovery intervals indicate that a model/parameter needs further investigation, not necessarily an installation failure.

## Test the MPI example

The MPI example calls `dlmmc.fit_grid_mpi()` to fit the pressure/latitude grid in parallel and stream completed fits to one rank-0 NetCDF writer. The resulting file is readable with xarray. On a local machine with Open MPI, run:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The example defaults to one chain, one warmup iteration, and two sampling iterations per grid cell. These settings are only for a pipeline smoke test and do not provide converged posterior estimates. Increase them for scientific runs. The output is `results/BASIC_V1_2017.nc`; `fit_status` is 1 for completed cells, 0 for skipped cells, and -1 for failed fits. Existing output is not overwritten unless `--overwrite` is specified.

On an HPC system, launch the script using the site's MPI launcher and scheduler allocation. See `MPI-README.md` for details. Rank 0 is reserved for writing when multiple ranks are used.

If CmdStan reports that its precompiled header needs rebuilding, run:

```bash
python -c "import cmdstanpy; cmdstanpy.rebuild_cmdstan()"
```

On macOS, long-running compilation and sampling can be interrupted by system sleep. Use `caffeinate -i <command>` or run on a remote system or in a `tmux` session.
