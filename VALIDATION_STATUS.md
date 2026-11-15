# Validation status for colleague testing

This branch is implementation-complete for the supported local scope. Availability, fixed-kernel correctness and posterior qualification are separate claims. No new scientific sampling was required for packaging cleanup.

## Feature and evidence matrix

| Feature | Implemented | Evidence and remaining gate |
|---|---|---|
| Six preset models, Numba/Torch CPU64 likelihood and trend moments | Yes | Individual and layout numerical checks pass |
| Six presets, native MPS32 likelihood and trend moments | Yes | Native fixed checks pass; scoped production runs measured; historical adverse float32 failures remain |
| CUDA32 | Yes | Native hardware unavailable; correctness/performance pending |
| CPU64 posterior consistency | Scoped | Five-panel practical ladders for vanilla AR1, no-regressor AR1 and harmonic-preprocessed no-seasonal variants; seasonal AR2 and dynamic-regressor long comparisons remain incomplete under retained caps |
| Independent production batching | Yes | All panels batched in each likelihood call; full 312-panel AR1 run explicitly selected and measured |
| Spatial positive-Gaussian sigma_trend | Yes | Four-cell joint reference and pipeline checks; new short MPS convergence INCONCLUSIVE |
| Positive ENSO lag field | Yes | Four-cell joint checks and synthetic recovery in principle; short MPS convergence INCONCLUSIVE |
| Arbitrary/multiple lag fields; arbitrary model composition | No | Deferred API/research work |
| Large coupled field blocks/streamed qualification diagnostics | No | Needed for a practical 128-walker full-grid coupled run |
| Full hardware/large-grid qualification and 200-replication calibration | No | Pending, opt-in; no pass inferred from local evidence |

The independent default is 128 walkers /128 burn-in /256 retained, one ensemble, 2,048 trend-integration draws. Four independent ensembles remain the joint validation default. These are finite-budget settings, not convergence promises. Walkers are not independent chains. Saved adverse float32 and cancellation fixtures remain in `tests/fixtures` and regression tests.

## Per-panel consistency and precision

The [current evidence bundle](examples/modernization/current/manifest.json) contains checksummed copies of the numerical and posterior ledgers. The [fixed-kernel per-panel table](examples/modernization/current/17-per-panel-numerics.csv) covers all six families and five frozen panels: individual CPU comparisons followed by batch-one/all-five, permutation, chunking and cross-talk checks. Native MPS fixed likelihood errors reached 0.000799 in this set; conditional trend mean differences reached 0.00308 posterior SD and width differences 0.0768%. These scoped results do not overwrite the older 0.01 log-density-difference failures at saved adverse proposals.

The [four-panel joint comparison ledger](examples/modernization/current/18-per-panel-results.csv) retains each model/parameter comparison to the matching saved joint reference. At 256 retained steps, spatial sigma means differ by at most 0.0306 reference SD and widths by 1.29%; after continuation to 1,024 retained, all parameter means differ by at most 0.0142 SD and widths by 1.04%. Real ENSO lag means differ by at most 0.0335 SD and widths by 4.39%. Complete-block and between-run reliability flags remain INCONCLUSIVE despite minimum ESS estimates above 2,000. The pipeline is useful; convergence has not been silently certified.

In the single synthetic lag-recovery fixture, the three signal-bearing truth lags (1.2, 1.6, 2.4 months) fall within the estimated 95% intervals. The zero-signal panel stays broad. This demonstrates recovery in principle, not repeated-simulation coverage. Four-panel sampling times were 45.42 seconds spatial, 84.87 seconds lag and 36.33 seconds synthetic recovery, with cached optimized centers and separately recorded compilation/warmup. Spatial continuation added 90.82 sampling seconds. These are warm-start pipeline timings, not cold end-to-end estimates.

Float64 remains the offline oracle and CPU production default. Float32 is available for user-approved GPU production and further scientific testing. Earlier single-panel bootstrap results supported practical inference consistency within MC error; full-grid short-run comparisons retain width discrepancies. The float32 decision is availability with explicit scope/flags, not universal numerical equivalence.

## Measured performance

Fresh earlier 312-panel vanilla AR1 runs used one ensemble, 128 walkers, 128 burn-in, **512 retained** and 2,048 output draws. The laptop has 10 CPU cores, 32 GiB shared memory and native Apple MPS; CPU used eight threads. Compilation/cache loading and disposable warmup were excluded from these comparisons and retained separately in the [timing record](examples/modernization/current/16-gpu-comparison-results.json).

| Phase | Numba CPU64 | MPS32 | CPU time / GPU time |
|---|---:|---:|---:|
| Sampling including burn-in | 854.78 s | 380.96 s | 2.24× |
| Trend-summary pipeline | 350.57 s | 61.81 s | 5.67× |
| End to end, excluding compilation/warmup | 1,254.56 s | 486.50 s | 2.58× |
| Actual run including setup | 1,614.75 s | 489.03 s | Different setup costs; not a steady-state speedup |

Matched-draw smoother agreement is much tighter than independent short-chain differences: at all 312 panels, mean differences reached 0.00389 SD and width differences 0.1295%. Independent CPU/GPU sampling produced a maximum width difference of 12.46%, with seven anchored-width discrepancies above 5%. Those flags remain.

The newer 256-retained default was selected from frozen five-panel checks: at matched 2,048 output draws, halving steps changed means by at most 0.0321 SD and widths by 2.35%. A full 312-panel run at this new default has **not** been measured; about 3.8 minutes sampling /5.6 minutes end to end is an extrapolation. It is not the measured 512-step result.

The [fixed-call thread test](examples/modernization/current/17-numba-threads.json) measured likelihood medians of 0.670/0.610/0.607 s and 32-draw smoother medians of 5.516/4.667/4.643 s at 8/10/16 threads. Ten captures almost all the improvement on this laptop. Eight remains the portable default; 16 is not generally recommended. MPI+Stan was not rerun for the full grid; no measured full-grid MPI speedup is claimed here.

## Reproduction and provenance

[TESTING.md](TESTING.md) gives the clean-install, unit/kernel and fresh-kernel notebook commands. Tutorial figures load labeled historical science and preserve its original source/settings rather than relabeling those fits as current-source runs. The current bundle's SHA256 manifest is checked by refreshed notebooks. Source distribution and wheel installation checks, actual tests/skips and notebook review are recorded in [CLEANUP_CHECKS.md](CLEANUP_CHECKS.md) when completed.

The implementation base is DLMMC e1bde457e077f44e54d963010e3580fb6e37321d. Affine remains Cosmo-Pop 5fdd3843b099a2386f75d97b6c059f64560afdf9 with the shipped adapter patch. Detailed original experiments, failed attempts and full chain archives remain in the modernization harness; this checkout includes small reproducible fixtures and science bundles, not the large chain archives. Historical provenance can contain original local paths; these are metadata, not dependencies needed to run the tutorials.
