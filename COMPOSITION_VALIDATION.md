# Composable API validation — 2026-11-17

The component API compiles state-space layouts for the existing CPU and GPU engines. The six preset strings remain compatibility shorthands. This report separates exact compatibility evidence from qualification of new compositions. See [COMPONENTS.md](COMPONENTS.md) for the API.

## Results and scope

Against frozen commit `7478764e490a9272ffe16c8079af94c97a9ced04`, all six models produced bitwise-identical MPS likelihoods and conditional trend moments on the five frozen panels. A fresh vanilla AR1 production fit, using independently repeated optimization and the same seed, reproduced every retained parameter draw and every trend/anchor-relative mean and standard deviation exactly. The plotted curves and bands overlap. This is a paired implementation regression, not independent-chain convergence certification. Existing short-run and precision flags remain.

| Panel | Parameter draws | Trend means and standard deviations |
|---|---|---|
| p05-lat01 | Bitwise identical | Bitwise identical |
| p10-lat04 | Bitwise identical | Bitwise identical |
| p16-lat07 | Bitwise identical | Bitwise identical |
| p22-lat10 | Bitwise identical | Bitwise identical |
| p00-lat00 | Bitwise identical | Bitwise identical |

All panel comparisons preserve the same input, masking, preprocessing, priors, Jacobians and initial-state timing. New fixed-parameter tests cover optional AR, arbitrary harmonic periods, multiple named harmonic blocks, Level instead of Trend, mixed dynamic/static coefficients, serialization and fresh-process checkpoint reload. Backend comparisons include every individual panel, size-one/all-five batches, permutation, chunking and cross-talk. Four-cell spatial/ENSO target compatibility is checked without new long joint chains.

Two new no-AR ozone fixtures fail the existing MPS float32 absolute density tolerance 0.01: maxima 1.11642 and 0.0359595. Input/parameter rounding, matrix rounding and filter arithmetic were measured separately. The first is chiefly filter arithmetic; the second also reaches scalar precision limits. These fixtures remain strict expected failures. A subsequent explicit user decision removed the no-AR GPU sampling gate: all compute backends are selectable, without treating these failures as scientific passes.

A Level-model anchor-relative covariance cancellation was repaired using backward conditional innovations and positive covariance sums, without added floors, jitter, changed model or relaxed tolerance. The failing fixture is preserved. This adds one covariance history to Level smoothing only; legacy Trend kernels are unchanged.

## Measured native MPS timing

Apple Silicon laptop, 10 CPU cores, 32 GiB unified memory; Torch 2.15.0, Python 3.13.2. MPS float32 with CPU fallback and fast math disabled. Fixed callbacks are synchronized and exact specializations are precompiled and warmed. Medians of 25 likelihood/10 smoother repetitions: five panels ×396 months,64 likelihood walkers and8 smoother draws.

| Preset-equivalent composition | Likelihood before → after (ms) | Smoother before → after (ms) |
|---|---:|---:|
| vanilla_ar1 | 6.559 → 6.553 | 90.117 → 90.105 |
| vanilla_ar2 | 7.872 → 7.955 | 154.585 → 154.783 |
| noregs_ar1 | 3.278 → 3.249 | 32.734 → 32.582 |
| dynregs_ar1 | 6.730 → 6.758 | 90.424 → 90.407 |
| vanilla_ar1_noseasonal | 2.900 → 2.901 | 33.460 → 33.584 |
| vanilla_ar2_noseasonal | 3.715 → 3.705 | 63.360 → 63.395 |

Observed likelihood ratios are 0.991–1.010 and smoother ratios 0.995–1.004 (after/before). No material timing regression was observed. These are local repeated callback measurements, not a universal performance guarantee.

Five-panel production comparison: one ensemble,128 total walkers,128 burn-in,256 retained sweeps,2,048 conditional-output selections; seed2026111701. Optimization and tight full-rank initialization are repeated in each run. All five panels stay in the same batch; output memory is bounded by draw tiling.

| Operation | Before (s) | Composed API (s) |
|---|---:|---:|
| Sampling including burn-in | 5.150 | 5.162 |
| Fit + trend summaries, excluding compilation/disposable warmup | 6.879 | 6.812 |
| Actual measured driver time after imports | 9.342 | 7.412 |
| Fit compilation/cache loading, reported separately | 1.911 | 0.100 |
| Fit disposable warmup | 0.066 | 0.065 |
| Smoother compilation/cache loading | 0.000302 | 0.000299 |
| Smoother disposable warmup | 0.425 | 0.374 |

The actual-time difference mostly reflects compilation/cache loading; it is not claimed as an API speedup. Comparable end-to-end time includes optimization, initialization, MCMC burn-in, retained sampling, fit persistence and trend summarization; compilation and disposable warmup are subtracted. Driver imports are outside this clock. No 312-panel timing or new Stan sampling was performed for this update.

## Support matrix

| Capability | Local status |
|---|---|
| Six historical models through component compiler | CPU64/native MPS32 numerical PASS; five-panel vanilla MPS replay PASS |
| New AR-bearing component combinations | Representative CPU64/MPS32 likelihood and smoother PASS; bounded checkpoint/output smoke PASS |
| New compositions without AR | CPU64 PASS in tested scope; GPU available by explicit user decision, numerical failures retained |
| Level anchor-relative summaries | Stable covariance repair tested on CPU/MPS; failure fixture retained |
| Model dict/object/checkpoint predictor binding | PASS, including fresh-process reload |
| Spatial sigma_trend and ENSO lag contracts | Same joint-target semantics and four-panel gates retained; fixed compatibility checks PASS |
| CUDA | Implemented through shared layout; native tests UNAVAILABLE |
| CmdStan | Original six models only; arbitrary Stan generation not implemented |
| Arbitrary custom components / AR orders above 2 / multiple lag fields | Not implemented; rejected explicitly |

Regression suite: **511 passed,27 CUDA skips,1 optional Stan deselected,2 retained expected precision failures**. The non-editable installed wheel additionally passed39 component tests (10 CUDA skips,2 expected precision failures). Four notebooks executed in fresh kernels;33 prior plots are byte-identical and their review is reused, and one new overlay was inspected. Completion is recorded in the harness report `reports/20-composable-api.md`; portable comparison arrays, checksums and timings are in `examples/modernization/composition/`. The tutorials distinguish fresh bounded smoke fits from cached scientific evidence. They preserve both spatial/lag selection gates.

## Reproduction

From this checkout with its pinned affine adapter installed (see INSTALL.md), native MPS hardware and test/notebook extras:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python -m pytest -q -m "not stan"
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python scripts/benchmark_components.py --repo . --output /tmp/dlmmc-components-fixed --form composed --kind fixed
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python scripts/benchmark_components.py --repo . --output /tmp/dlmmc-components-fit --form composed --kind fit
python scripts/check_notebooks.py --output-dir /tmp/dlmmc-components-notebooks
```

Use new output paths. The benchmark refuses existing destinations; fixed work has a300-second actual-time cap, fit/output600/300 seconds. To reproduce the before/after comparison, archive commit7478764 into a separate directory and run the same script with that directory on PYTHONPATH, `--repo` pointing there and `--form preset`. The harness report records the exact original commands and source hashes. The affine source pin remains `5fdd3843b099a2386f75d97b6c059f64560afdf9`; no dependency update is implied.
