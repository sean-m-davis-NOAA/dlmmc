# Production and qualification profiles

The execution profile changes the amount of monitoring and validation, not the statistical model, priors, Jacobians, masks or numerical precision policy.

| Policy | Production | Qualification |
|---|---|---|
| Intended use | Fixed-budget scientific estimates with visible warnings | Evidence that an implementation/backend preserves inference |
| Default ensembles | 1 | 4 |
| Walkers per ensemble | 128 (64 per half) | 128 (64 per half); bounded tuning separate |
| Burn-in / retained sweeps | 128 / 256 | Recommended initial comparison 100 / 4,000; diagnosed continuation as needed |
| Parameter retention | Every post-burn-in sweep, bounded checkpoints | Every post-burn-in sweep, bounded checkpoints |
| Monitoring | Finite support, acceptance, half/full means and widths, nominal mean ESS/MCSE | Independent-run rank, mean/width/tail ESS, MC-aware stability and actual trace review |
| Reference comparisons | Not repeated for every scientific fit | Stan/backend equivalence and per-panel layout tests |
| State output | Deterministic marginal trend moments, including conditional uncertainty | Diagnosed parameter window before conditional state trajectories; separate output gates |
| Grid scope | At most five panels by default; larger named selection must match coordinates/settings | Independent validation remains at most five panels; separate spatial/lag gates unchanged |
| Result label | Production estimates with diagnostic flags, not a qualification pass | Existing PASS / FAIL / INCONCLUSIVE rules |

Production and its `TESTING_PROFILE` alias now support all six independent models with `backend="numba"` (CPU64), `"torch"` (CPU64), `"mps"` (native fused GPU32), or `"cuda"` (portable Torch GPU32). `noregs_ar1` takes no regressors; the other models accept their supported nonzero regressor counts. Dynamic models retain one process scale per regressor. No-seasonal inputs require an explicit deseasonalization declaration; the project uses fitted harmonics. GPU dtype is automatic, with no fallback.

All six families have fixed likelihood and marginal/anchored trend-moment validation on CPU and native MPS, including singleton, permutation, chunking and cross-talk checks. This extends kernel/API support, not posterior qualification. CUDA code is implemented but native tests are unavailable. The independent default is one ensemble, 128 walkers, 128 burn-in and 256 retained sweeps; eight value-independent output selections per sweep retain 2,048 integration draws. Qualification thresholds and its four-run profile are unchanged. `LEGACY_PRODUCTION_PROFILE` retains 128/128/512 with four selections per sweep for historical reproduction.

Spatial sigma_trend targets support all six models. ENSO-lag targets support all five regressor-bearing models; a no-regressor model cannot contain an ENSO lag. These public joint samplers retain four independent runs, explicit scope gates and bounded checkpoints; their initial default budget is now 128 burn-in/256 retained, not an automatic convergence declaration. Four-cell CPU/MPS joint density and lag-interpolation/moment checks pass; bounded four-panel GPU pipeline/recovery runs have executed, with convergence still INCONCLUSIVE. Larger-grid conditional field blocks and streamed diagnostic work remain separate.

## Public APIs

For a named single panel or an ordinary group of at most five:

```python
from dlmmc import fit_production, diagnose_production, summarize_production_trends

fit = fit_production(data, regressors, uncertainty=uncertainty,
                     backend="mps", output_dir="results/new-production-fit")
monitoring = diagnose_production(fit)
summary = summarize_production_trends(
    fit, output_dir="results/new-production-trends", anchor_month="1998-01")
```

Data have `time` and `panel` dimensions; regressors have `panel`, `time`, `regressor`. Output directories are new, and completed evidence is not overwritten. For explicit configuration use `fit_production(..., profile=PRODUCTION_PROFILE)` with an `ExecutionProfile` value; `fit_grid(profile="production")` selects the preset. The profile is saved and hashed into checkpoint provenance. `resume_production` resumes complete state including both halves and RNG, requiring matching sources/runtime. It does not silently extend an exhausted job or change the ensemble size.

Larger independent production jobs require a selection dictionary containing `scope="independent_production"`, the actual `explicit_user_response`, the exact ordered `panel_ids`, `model`, `backend`, `dtype`, and `profile=PRODUCTION_PROFILE.to_dict()`. A spatial or lag selection is not accepted. The named 312-panel November2026 experiment has its own selection record in the harness. That selection does not enable a general benchmark/calibration campaign.

The existing `fit_grid` defaults and `profile="qualification"` retain the four-ensemble sampling guard. An explicit qualification run can use the settings in `QUALIFICATION_PROFILE`:

```python
from dlmmc import fit_grid, QUALIFICATION_PROFILE
q = QUALIFICATION_PROFILE
fit = fit_grid(data, regressors, uncertainty=uncertainty,
               profile="qualification", output_dir="results/new-qualification-fit",
               runs=q.runs, half_walkers=q.half_walkers, warmup=q.warmup,
               retained_sweeps=q.retained_sweeps, chunk_sweeps=q.chunk_sweeps)
```

Sampling completion alone is never a qualification certificate. The existing diagnostic and reconstruction requirements still apply. The qualification preset is a starting budget, not a promise of convergence.

## Output meaning and bounded memory

Production trend summaries evaluate the Gaussian conditional analytically. Their variance is
`E[conditional variance] + Var[conditional mean]`, retaining both state uncertainty and parameter uncertainty. For changes relative to an anchor month, cross-time covariance is included. The reported band is posterior mean ± one posterior SD, not a confidence interval for the estimated mean and not a simultaneous band. No conditional trajectories are sampled from a short uncertified window.

The output routine selects eight walkers independently of their values at every retained sweep (2,048 draws at the production preset). The CPU implementation passes all selected draws and all panels to one compiled call, parallelizes panels, and accumulates mixture moments online. Common model matrices are constructed once per panel; each worker holds only its current draw's filter/smoother history. Only final curves, selected indices and monitoring time series are saved. The CPU smoother does not chunk draws or panels. The native GPU smoother keeps all panels together and tiles draws to bound filter/smoother histories; a 1 GiB workspace allows four draws ×312 panels per tile. GPU reductions accumulate mixture moments on device, and only final summaries/monitoring series move to the CPU. Set `workspace_bytes=` to control this memory budget, not the panel count. The sampler's 16-sweep checkpoint interval is a separate time-axis setting; each likelihood half-step includes every panel.

Production fitting and smoothing default to eight Numba threads and one Torch thread; override with `numba_threads=` on either public API. Checkpoints save the thread setting and resume it exactly. Qualification settings remain as recorded in their own runs. `summarize_production_trends(..., engine="legacy", chunk_draws=16)` retains the historical two-thread output oracle explicitly. The full-batch reducer currently supports the single-ensemble production preset. All parameter checkpoints remain; single-run nominal ESS/MCSE is explicitly provisional and no between-run Rhat is invented.

Five-panel tests found that reducing output integration to 512 draws changed some anchor-relative widths by more than the existing 5% check. The default therefore retains 2,048 selected draws. No latent trajectories or full-grid all-pairs diagnostic products are stored.

Initialization optimizes each panel separately, then creates full-rank tight clouds in transformed coordinates. Sampling uses the same validated target and affine transitions as qualification. The large production grid does not require a global high-dimensional optimizer or treat independent panels as one coupled parameter block.

Timing reports separate preparation, optimization, compilation/cache loading, disposable warmup, sampling including burn-in, monitoring, output computation and plotting. Actual wall caps include setup. Scientific timing comparisons exclude compilation/disposable warmup and do not equate unequal achieved precision or different output contracts.

The earlier local release archive predates these new APIs and remains immutable. This branch has its own refreshed package and notebook checks; the earlier archive remains a historical snapshot.

## Backend and precision policy

`fit_production(..., backend="numba")` defaults to CPU float64. Selecting `backend="mps"` or `backend="cuda"` automatically selects native float32 for sampling and default smoothing. Conflicting explicit dtype/device requests fail; unavailable devices fail instead of falling back. `fit_grid(..., profile="production", backend=...)` uses the same dispatch. `summarize_production_trends(..., engine="auto")` follows the fit backend. Canonical input preparation/initial optimization and host diagnostics may use float64; GPU kernels do not silently mix precision. The CPU64 oracle remains available.

MPS requires a Torch build exposing `torch.mps.compile_shader`; the tested version is 2.15.0. Start with `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0`. CUDA uses portable eager Torch kernels with highest float32 matmul precision; native CUDA correctness, performance and memory qualification are pending. No MPS speedup is promised on CUDA.

The user approved production availability of the GPU path. Historical float32 log-density-difference failures, single-ensemble limitations and scientific discrepancy flags remain visible; availability is not a blanket numerical or convergence qualification. The original release archive predates this update.

## Fixed likelihood and smoother checks

`conditional_trend_moments(prepared, u, anchor, backend=...)` returns physical-unit conditional trend mean, variance, anchor-relative mean and variance, with shape `[draw, cell, time, 4]`. The anchor is a time index. Optional `regressors=` accepts `[draw, cell, time, regressor]` arrays, including per-draw ENSO interpolation; the corresponding parameters and regressors must be kept aligned. This API is deterministic and does not certify a posterior window. Prepare CPU64 inputs for the oracle or native GPU32 inputs for MPS/CUDA; sampling APIs perform their declared backend transfer explicitly.

The measured 312-panel fixed-batch comparison gives 8/10/16-thread likelihood medians 0.670/0.610/0.607 seconds and 32-draw smoothing medians 5.516/4.667/4.643 seconds. Ten threads is the useful override on the tested ten-core laptop. The existing portable default remains eight; set `numba_threads=10` for either production API. Testing 16 requires `NUMBA_NUM_THREADS=16` before Python starts. Oversubscription is not a general recommendation.

## Named spatial and lag specifications

See [SPATIAL_API.md](SPATIAL_API.md) for scalar/per-panel Gaussian means and standard deviations, named grid-axis correlation lengths and `lags={"ENSO": LagPrior(...)}` with fixed predictor coverage. These calls dispatch to joint field acceptance, not independent panel acceptance. The new short-budget experiments preserve their INCONCLUSIVE convergence flags; a useful pipeline/recovery check does not replace four-run scientific qualification. Full-grid spatial and lag tests remain outside the currently selected four-panel scope.

## Composed models

The same profiles accept `model=Model(...)`; see [COMPONENTS.md](COMPONENTS.md). Independent batching and joint field acceptance are unchanged. No-AR GPU sampling is available by explicit user decision; failed numerical fixtures remain recorded. Trend summaries require Trend or Level. For an explicitly selected larger production job, serialize a composed selection model with `model.to_dict()`; the supplied selection is checked against the canonical specification. This API update does not itself authorize a larger run.
