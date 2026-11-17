# API and scientific policy

The affine API now composes supported state-space components with one shared layout compiler. Six historical names remain compatible shorthands; the CmdStan API keeps its original six models. See [COMPONENTS.md](COMPONENTS.md) for construction, persistence and restrictions. [README.md](README.md) lists supported backends; [EXECUTION_PROFILES.md](EXECUTION_PROFILES.md) describes independent production versus qualification; [SPATIAL_API.md](SPATIAL_API.md) defines coupled targets.

## Precision and inference

CPU float64 remains the reference. MPS/CUDA production uses float32 automatically, without mixed-precision fallback. GPU production availability was explicitly approved while retaining numerical and convergence failures. Successful scoped inference comparisons do not erase the unchanged 0.01 log-density-difference failures at adverse saved proposals. CUDA hardware is untested.

Fixed innovation variances, priors, transformation Jacobians, AR2 conditional-prior normalization and initial-state timing are preserved. Observation masks avoid invalid observation arithmetic. No numerical floors or extra covariance jitter are introduced. Production summaries retain both conditional state uncertainty and parameter uncertainty; anchor-relative bands include cross-time covariance.

No-seasonal models require explicitly deseasonalized inputs. The example preprocessing jointly fits annual/semiannual harmonics with intercept and linear-trend nuisance terms, subtracting only harmonics from ozone and predictors. Uncertainties and masks are retained. Inference is conditional on the estimated cycle; its estimation uncertainty is not propagated. Earlier raw seasonal-data no-seasonal evidence remains superseded.

The user-selected practical validation profile targets about 2,000 effective samples per scientific summary and uses dependence-aware Monte Carlo stability checks, retaining strict flags. Effect-size bands are unchanged. Derived-quantity checks use log(stationary_AR_variance), whose relevant moments are finite; raw values remain diagnostics because their posterior SD can be infinite. Walkers within an ensemble are dependent. Production's single-ensemble settings do not provide a between-run convergence certificate.

## Spatial and lag model

The Gaussian prior is directly on positive-truncated sigma_trend, not its logarithm. The dense kernel uses named axis lengths; the testing helper targets underlying Gaussian neighbor correlation, not post-truncation correlation. The ozone example explicitly uses natural log-pressure distance and latitude degrees, not inferred altitude.

ENSO adds a prior-independent positive-truncated Gaussian lag field with the same lengths, mean zero and scale two months, strictly 0 < lag < 12 months. ENSO(t − lag) means earlier ENSO predicts ozone. Complete predictor coverage is required for one fixed observation window at every allowed lag; no extrapolation or proposal-dependent deletion occurs. Other parameter priors remain independent between panels. Only ENSO can currently be lagged.

Each coupled study retains its own four-panel pilot, full-grid runtime/memory estimate and explicit post-pilot selection. A spatial selection does not authorize the lag study. Coupled posterior comparisons use a four-cell joint reference. Whole-field stretch moves need enough walkers to span the field: 128 walkers cannot update a 312-dimensional field this way. Conditional subfield moves and streamed large-grid diagnostics remain future work.

## Checkpoints, memory and timing

Complete affine checkpoints retain both terminal ensemble halves, cached probabilities, RNG state, acceptance counters, global retention phase, inputs and source/runtime identities. Exact continuation requires matching sources, inputs, dtype, backend and runtime. Historical outputs remain readable after changes; source migrations require a separate audit. CmdStan replacement references are new chains, not exact continuation.

Sampling saves hyperparameters rather than latent trajectory cubes. GPU trend output tiles draws while retaining all selected panels. Qualification diagnostics can still materialize very large joint/all-pairs arrays; historical terabyte estimates described that diagnostic approach and long windows, not unavoidable sampler storage. Full-grid coupled diagnostics are not supported by this candidate.

Timing comparisons exclude compilation/cache loading and disposable warmup, report these separately, and synchronize devices. Time-to-ESS includes initialization and burn-in. Operational caps count actual wall time. Fixed-kernel speed ratios are distinct from end-to-end sampling speedups. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for measured scope and unresolved gates.
