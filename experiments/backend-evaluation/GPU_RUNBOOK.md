
# GPU DLM Evaluation Runbook

This document describes how to evaluate a GPU implementation of a dlmmc DLM on a workstation or HPC system, and how to compare it with the existing CmdStan CPU and MPI workflows. It is an experiment guide, not a claim that dlmmc currently has a production GPU mode.

## Current Backend Status

- `dlmmc.fit()` and `dlmmc.fit_grid_mpi()` use CmdStanPy. There is no `backend="gpu"` option or GPU-enabled production API. Keep CmdStanPy as the CPU backend and scientific reference.
- The TFP code in this directory is a separate prototype. `tfp_real_grid_reference.py` creates fixed-parameter CmdStan likelihood/gradient references, and `tfp_real_grid_benchmark.py` compares those values on TensorFlow CPU/GPU. This is not a full GPU posterior sampler or a replacement for `fit_grid_mpi()`.
- Numba 0.68.1 in `dlmmc-local-optimizations` matches a fixed CmdStan AR(1) likelihood to 1.1e-10 with a 0.413 ms median warm call after 2.46 s compile-plus-first-call; it has no autodiff sampler or smoother.
- PyTorch 2.9.1 MPS in `dlmmc-torch-mps` passes basic matmul/autograd, but the full float32 DLM recurrence is slower than CPU. An eight-cell likelihood/gradient batch took 268 ms on MPS versus 82 ms on CPU and had larger numerical error.
- Numba 0.68.1 with emcee 3.1.6 provides an experimental CPU sampler path through a vectorized Numba log-probability callback. A matched 10,000-step eight-cell run took 194.6 s including JAX CPU smoothing; its minimum estimated ESS (4,456) exceeded the matched CmdStan run's minimum bulk ESS (3,084), and state mean/SD summaries were within 2.5% of CmdStan posterior SD in the worst cell. Estimated aggregate ESS/sec was 3.4-4.7x higher, but walker autocorrelation ESS is not independent-chain R-hat. Validate multiple independent ensembles and per-cell diagnostics before scaling up.
- Two matched 32-cell, eight-slot Numba/emcee runs took 778-787 s including smoothing versus 2,633.9 s for CmdStan; estimated aggregate ESS/sec was 3.0-4.0x higher across both seeds. State mean and SD summaries were within 3.9% and 3.7% of CmdStan posterior SD at worst. Emcee's finite-ensemble autocorrelation ESS is not independent-chain R-hat; review per-cell tail diagnostics before full-grid work.
- A diverse 64-cell, eight-slot Numba/emcee test with two seeds took 1,595-1,627 s including smoothing versus 5,166.7 s for CmdStan; estimated aggregate ESS/sec was 2.9-3.7x higher. State mean and SD summaries were within 4.2% and 4.7% of CmdStan posterior SD at worst. This remains experimental CPU evidence, and walker-autocorrelation ESS is not independent-chain R-hat.
- A 96-cell follow-up exposed a per-cell mixing failure hidden by aggregate emcee diagnostics: seed 1's walkers split between modes for cell 160 `sigma_seas`, causing seasonal-state mean/SD errors of 14%/48% of CmdStan SD. Seed 2 matched cell 160 but had state-SD outliers in cell 125. CmdStan completed all cells with max R-hat 1.0038 and min bulk ESS 2,596. Do not infer full-grid readiness from aggregate ESS/sec; investigate independent ensembles and per-walker traces for these cells first.
- A targeted two-cell test with 128 walkers and two independent seeds removed the cell 160 `sigma_seas` split. With about 80,000 simulation-smoother draws per cell, state mean/SD errors were <=1.4%/3.6% of CmdStan SD. This indicates the prior low-walker and low-smoother-draw settings were insufficient for those cells, but the entire 96-cell workload has not been rerun with 128 walkers; do not treat it as full-grid clearance.
- A full 96-cell, 128-walker rerun completed in 10,542.8 s including smoothing; minimum estimated ESS was 17,600-22,500, and parameter/state mean and SD summaries matched CmdStan within 2.68% of CmdStan SD at worst. This resolves the prior cell-160 discrepancy for one seed only. Run an independent full-96 seed and inspect per-cell diagnostics before scaling further.
- The separate affine v0.1 TensorFlow sampler runs a toy CPU target, but its Metal path fails at `tf.math.is_nan`; its API lacks built-in seeding and convergence diagnostics.
- The `jax_dlm` environment has JAX 0.8.2 and NumPyro 0.19.0 and detects `METAL:0`, but a basic matmul fails with `UNIMPLEMENTED: default_memory_space is not supported`. On CPU, an eight-cell 4x500/500 NumPyro run passed fixed-likelihood/gradient parity, had zero divergences and max R-hat 1.0011, and took 4m34s including smoothing/output. The matched CmdStan MPI run took 4m22s with all eight cells complete and max R-hat 1.0103. Full-grid scaling remains open.
- Past local results are cautionary: the tested TFP sequential Metal path was much slower than CPU for the 32-cell likelihood/gradient workload; GPU sampler attempts were incomplete or failed. PyMC/nutpie used CPU in the tested setup. These results do not predict performance on NVIDIA or AMD hardware.
- A machine having a GPU, a visible device, or a successful matrix multiplication does not prove that the DLM model operations run on the GPU or that sampling is faster. The local PyTorch MPS kernel and affine Metal smoke illustrate that distinction.

The goal is portability of the model and evaluation procedure, not one universal binary environment. GPU frameworks, drivers, compiler runtimes, and site modules differ by hardware and machine. Keep host-specific environments separate from production dependencies.

## Choose The Comparison

| Workload | Implementation | Purpose |
| --- | --- | --- |
| One-cell CPU fit | `dlmmc.fit()` / CmdStanPy | Validate the installed package and establish a single-cell CPU reference. |
| Gridded CPU fit | `dlmmc.fit_grid_mpi()` / CmdStanPy | Measure CPU scaling and provide the practical grid baseline. |
| GPU likelihood/gradient | Experimental TFP script | Check numerical agreement and device placement; not a sampling-speed result. |
| GPU posterior sampling | Not implemented as a supported dlmmc backend | Future work after a model-faithful sampler path is selected and validated. |

Stan can use OpenCL for particular supported operations when built and configured accordingly, but the current dlmmc models have not been tested or certified for that route. Treat it as a separate experiment; do not assume enabling an OpenCL build automatically accelerates the sequential Kalman/state-space work.

## Local JAX And NumPyro Feasibility

The separate `jax_dlm` environment contains JAX 0.8.2 and NumPyro 0.19.0. JAX enumerates `METAL:0`, but a synchronized 128x128 matmul fails with `UNIMPLEMENTED: default_memory_space is not supported`. Forcing `JAX_PLATFORMS=cpu` allows the one-cell NumPyro DLM experiment. Metal visibility is therefore not usable GPU execution in this environment.

1. In a separate evaluation environment, install a JAX/NumPyro combination supported by the target platform. For local Apple GPU testing, verify current JAX Metal support and its version constraints from the package's official installation guidance before creating the environment. Keep the existing CmdStanPy environment separate as the reference.
2. Run the environment probe and a tiny JAX operation explicitly on the reported device, force completion, and confirm the result's device. If only CPU is available or the operation fails, stop GPU claims there; NumPyro CPU may still be evaluated separately.
3. If a local accelerator works, implement one-cell `vanilla_ar1` log likelihood and autodiff gradient in isolated experiment code. Compare fixed parameters and gradients with the CmdStan reference, first at float64 and then quantify float32 differences. Verify the full computation, not only a matrix operation, is placed on the accelerator.
4. After numerical parity, compare NumPyro NUTS with the existing CmdStan real-cell fit. The one-cell test now includes sampled smoother outputs; compare R-hat, bulk/tail ESS, divergences, acceptance, retained draws, state-posterior summaries, and ESS/sec. CPU NumPyro remains a CPU comparison, not a GPU result.

Do not add a host-specific Metal environment file until JAX Metal availability and compatibility are verified for the target macOS and Python versions. Keep any NumPyro dependencies in an isolated evaluation environment, not the package's production dependencies.
Do not add JAX Metal, PyTorch, Numba, or affine to production dependencies based on these experiments. Keep the candidate paths isolated until one demonstrates posterior parity, smoother parity, diagnostics, and end-to-end throughput.

## Procedure

### 1. Select A Host And Allocation

Use the target machine's current scheduler and accelerator documentation. Do not copy a partition name, QoS, module, GPU request, or launcher command from another machine.

- Ursa is the current documented GPU target: `u1-h100` is the first evaluation candidate; `u1-gh` and `u1-mi300x` are additional exploratory targets. See the [Ursa probe notes](README.md#preparing-an-ursa-gpu-probe) and verify access and current scheduler settings on the live system.
- HERA's public guide has not listed a GPU partition. The user reports GPU nodes, so verify live availability with the scheduler and site support. HERA remains a useful CPU/MPI comparison host either way.
- For a workstation or another center, record the exact GPU model and use that site's supported software stack.

Run the probe inside the allocation, not only on a login node:

```bash
python experiments/backend-evaluation/hpc_environment_probe.py \
  --output "$TMPDIR/dlmmc-environment-${SLURM_JOB_ID:-local}.json"
```

Retain the JSON with the run results. Record the repository commit separately. The probe reports installed framework versions, loaded modules when exposed through `LOADEDMODULES`, scheduler fields, visible devices, and tiny framework device operations. Use `--output` to keep framework initialization messages from contaminating the JSON artifact. It is an inventory check, not a DLM validation. On systems without Slurm, unset fields are expected.

### 2. Create A Host-Specific Environment

1. Follow the center's policy for Python distributions, package channels, compilers, GPU drivers, and environment storage. For RDHPCS, use Miniforge/conda-forge; do not treat either checked-in environment YAML as an Ursa specification.
2. Select a framework/runtime combination supported by both the exact GPU and the site driver/modules. NVIDIA/CUDA, AMD/ROCm, Apple/Metal, and other accelerators need separate compatibility checks. The checked-in `environment-tfp-metal.yml` is macOS/Apple Metal only.
3. Keep the CPU CmdStan environment and the accelerator environment isolated. Pin Python, framework, compiler/runtime, and relevant module versions in the experiment record. Do not add these packages to the production package dependencies.
4. Build/compile on the target architecture where required. For MPI, build `mpi4py` against the same MPI implementation used by the site launcher. Keep compiler caches and per-fit scratch on approved writable filesystems; check quota and purge policy first.

Use `ursa_h100_probe.sbatch.example` only as a starting template. Replace its account, environment activation, and GPU request after verifying them on the live site. The template is intentionally not submission-ready.

### 3. Establish The CPU Controls

Run both controls on the same host class, data, model variant, and intended grid where possible:

1. Run a one-cell CmdStan fit to check compilation and sampling on that host.
2. Run the representative grid with CmdStan MPI. Specify scientific warmup/sample counts explicitly: `dlm_lat_alt_mpi_run.py` defaults to one warmup and two samples, which is only a pipeline smoke test.
3. Measure a single-rank serial run where useful, then increase MPI ranks within the allocation. In `fit_grid_mpi()`, rank 0 initializes and writes the NetCDF file; with more than one rank, the other `N-1` ranks fit cells. Do not count the writer as a fitting worker.
4. Record CPU cores, threads, MPI implementation, rank count, node count, CmdStan compilation/cache state, elapsed time, memory, temporary storage, completed/failed cells, and posterior diagnostics.

Use the site's supported MPI launcher and binding flags on HPC. `mpiexec -n N` is suitable only when it matches the local/site MPI configuration; it is not a universal Slurm launch command.

### 4. Check GPU Placement And Numerical Fidelity

Use the existing fixed-parameter experiment as a first filter, not as evidence of sampler speed:

```bash
# Generate reference data and values with the CmdStanPy environment.
python experiments/backend-evaluation/tfp_real_grid_reference.py \
  --cells 32 --output "$TMPDIR/dlmmc-tfp-real-stan-reference.npz"

# In a compatible TFP environment, test CPU and GPU separately or together.
python experiments/backend-evaluation/tfp_real_grid_benchmark.py \
  --reference "$TMPDIR/dlmmc-tfp-real-stan-reference.npz" \
  --device both --repeats 3
```

If the host has no supported TensorFlow GPU, use `--device cpu`; do not interpret a CPU fallback as a GPU run. A GPU request should fail clearly when placement is unsupported. Confirm the printed likelihood and gradient devices are the intended accelerator. Save likelihood/gradient errors, warm-up/compilation time, steady-state timings, and actual device utilization.
For the local CPU NumPyro comparison, run `cmdstan_real_cell.py` first, then `numpyro_real_cell.py` from `jax_dlm`. The runner verifies fixed likelihood/gradient parity, samples the hyperparameters, and reproduces the simulation-smoothed outputs. Two 4x500/500 runs took 22-23 seconds; smoothing took another 1.84 seconds, versus 48.18 seconds for CmdStanPy on this cell. State posterior means agree within roughly 2-3% of CmdStan posterior SD; do not extrapolate to grid or GPU performance.
The single-cell and eight-cell CPU NumPyro runs both pass fixed likelihood/gradient parity and generate smoothed outputs. For eight cells, NumPyro took 4m34s including sampling, smoothing, and compressed output; the best-sized CmdStan MPI run took 4m22s. NumPyro had zero divergences and max R-hat 1.0011 versus CmdStan's max 1.0103. This is near-parity for eight cells, not a full-grid or GPU result.

Before implementing or timing full GPU sampling, check model fidelity in this order:

1. Fixed-parameter likelihoods and gradients for the selected DLM against CmdStan, including constrained/unconstrained parameter transforms.
2. Edge cases: missing observations and uncertainties, constant/invalid cells, time-varying uncertainty, regressor alignment, first scored observation, priors, and initialization.
3. Posterior results for representative cells: compare posterior summaries and smoothed-state outputs, and report divergences, R-hat, bulk/tail ESS, acceptance, and retained draws.
4. Representative multi-cell/grid behavior and output handling.

Do not call a path a DLM GPU run based only on framework-device discovery, likelihood timing, or a short sampler smoke test.

### 5. Benchmark End To End

Compare at least these configurations:

- CmdStanPy single-cell CPU and CmdStanPy grid MPI at several rank counts.
- The alternate backend on CPU, when available, to isolate framework/backend effects from accelerator effects.
- The same alternate backend on one GPU, with cells batched only where the implementation preserves independent-cell semantics.
- Multi-GPU or hybrid CPU/GPU work only after one-device correctness and performance are understood.

Use identical input cells, data preprocessing, DLM variant, priors, uncertainty, regressors, output requirements, and sampler quality targets. Record seeds and repeat runs; random-number streams need not match between backends. Report cold-start and warm timings separately, and include the full job wall time for the operational comparison. Split out compilation/JIT, data preparation/transfers, warmup, sampling, summaries, and file writing.

Report both throughput and inference quality:

- End-to-end wall time, cells completed per hour, and per-cell elapsed time.
- ESS/second for key parameters at comparable R-hat, divergences, acceptance, and effective sample counts. A faster run with materially worse mixing is not a speedup.
- CPU/GPU utilization, memory, allocation size, number of fitting workers, failed cells, output size, and temporary-storage use.
- Repeats and variability; do not compare a cold GPU compile with a warmed CPU run without reporting both.

The current evaluation plan has a provisional target of at least 2x end-to-end ESS/second at comparable diagnostics, repeated across runs. This is a proposed decision gate, not an approved requirement or a result. Decide with the project whether that gain justifies the extra backend and maintenance burden.

Avoid oversubscription. Do not launch many MPI ranks that each create GPU workers or target the same GPU unless that strategy has been explicitly measured. Start with one process bound to one GPU and a small batch. For CPU/MPI, sweep ranks and chain parallelism only within allocated cores. Preserve rank 0 as the sole NetCDF writer for the existing grid API.

## Interpretation And Next Decision

Keep CmdStanPy as the production CPU backend. A GPU result should lead to a separate proposal for an optional backend, not a silent change to `dlmmc.fit()`. The proposal should include supported hardware/runtime combinations, model parity, failure behavior, test coverage, installation burden, and a repeatable end-to-end benefit over the best CPU/MPI baseline.

If no candidate passes fidelity, reliability, and performance gates, document the result and continue using CmdStan CPU/MPI. A negative result on one GPU/runtime does not prove that all GPU hardware is slower; equally, a fast microbenchmark on one device does not establish a portable benefit.