## Installation validation

Here we show what a successful installation looks like, following the installation instructions in the `README`


For the purposes of validation, create and activate a reproducible conda environment. Two supported workflows are shown below: (A) the legacy PyStan workflow and (B) the recommended modern CmdStan / cmdstanpy workflow.

A) Legacy PyStan (for compatibility testing)
```
# create a conda env with Python 3.7 (legacy)
conda create -n dlmmc python=3.7 anaconda
conda activate dlmmc
# install dependencies (legacy)
conda install netCDF4 pystan
```

B) Recommended: CmdStan + cmdstanpy (modern, faster, more robust)
```
# create a modern conda env using a recent Python
conda create -n dlmmc-cmdstanpy python=3.10 -y
conda activate dlmmc-cmdstanpy
# install runtime dependencies; cmdstanpy is installed via pip
conda install -y numpy scipy matplotlib jupyter netCDF4
pip install cmdstanpy
# Install CmdStan (downloads and compiles CmdStan; requires a C++ toolchain)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Notes:
- CmdStan compilation requires a working C++ toolchain (macOS: Xcode command-line tools, `xcode-select --install`). On CI or headless servers prefer a packaged compiler or Docker.
- On macOS, long-running compile/sampling steps may be interrupted by system sleep. Use `caffeinate -i <command>` to keep the machine awake, or run on a remote server / in a tmux session over SSH.
- After CmdStan is installed you can compile the models and create marker files: `python3 compile_stan_models.py`

Compile the DLM models following instructions in `README`:
```
justinalsing$ python3 compile_stan_models.py

INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_1769d29906593e8f6fa11e816b642cff NOW.
INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_323f0530039bc4ac2c22bb5250e1d6c1 NOW.
INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_c3ff00cf2253f51bed2b150f31119693 NOW.
INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_b9cb9e0eb2389c8a6e3078345a6a1dd4 NOW.
```

Note that you might get some additional deep-copy warnings from pystan during compilation - these are not a problem (so long as compliation completes without errors).

Finally let's execute the `dlm_tutorial.ipynb` notebook to check everything worked correctly:
```
justinalsing$ jupyter-nbconvert --to notebook --execute --ExecutePreprocessor.timeout=100000 dlm_tutorial.ipynb
[NbConvertApp] Converting notebook dlm_tutorial.ipynb to notebook
[NbConvertApp] Executing notebook with kernel: python3

Gradient evaluation took 0.026314 seconds
1000 transitions using 10 leapfrog steps per transition would take 263.14 seconds.
Adjust your expectations accordingly!


Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
Exception: multiply: B[12] is nan, but must not be nan!  (in 'unknown file name' at line 159)

If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
Exception: multiply: B[12] is nan, but must not be nan!  (in 'unknown file name' at line 159)

If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

Iteration:    1 / 3000 [  0%]  (Warmup)
Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
Exception: multiply: B[12] is nan, but must not be nan!  (in 'unknown file name' at line 159)

If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
Exception: multiply: B[12] is nan, but must not be nan!  (in 'unknown file name' at line 159)

If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.

Iteration:  300 / 3000 [ 10%]  (Warmup)
Iteration:  600 / 3000 [ 20%]  (Warmup)
Iteration:  900 / 3000 [ 30%]  (Warmup)
Iteration: 1001 / 3000 [ 33%]  (Sampling)
Iteration: 1300 / 3000 [ 43%]  (Sampling)
Iteration: 1600 / 3000 [ 53%]  (Sampling)
Iteration: 1900 / 3000 [ 63%]  (Sampling)
Iteration: 2200 / 3000 [ 73%]  (Sampling)
Iteration: 2500 / 3000 [ 83%]  (Sampling)
Iteration: 2800 / 3000 [ 93%]  (Sampling)
Iteration: 3000 / 3000 [100%]  (Sampling)

 Elapsed Time: 180.043 seconds (Warm-up)
               372.797 seconds (Sampling)
               552.839 seconds (Total)

[NbConvertApp] Writing 687553 bytes to dlm_tutorial.nbconvert.ipynb
```

The notebook completed without error! You're now ready to start using the code.
