"""Public CmdStanPy API for fitting dynamical linear models."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from cmdstanpy import CmdStanModel
from platformdirs import user_cache_dir

from . import _stan_models


_MODEL_SPECS = {
    "vanilla_ar1": {"source": "dlm_vanilla_ar1", "regressors": "static", "seasonal": True, "ar2": False},
    "vanilla_ar2": {"source": "dlm_vanilla_ar2", "regressors": "static", "seasonal": True, "ar2": True},
    "noregs_ar1": {"source": "dlm_noregs_ar1", "regressors": "none", "seasonal": True, "ar2": False},
    "dynregs_ar1": {"source": "dlm_dynregs_ar1", "regressors": "dynamic", "seasonal": True, "ar2": False},
    "vanilla_ar1_noseasonal": {"source": "dlm_vanilla_ar1_noseasonal", "regressors": "static", "seasonal": False, "ar2": False},
    "vanilla_ar2_noseasonal": {"source": "dlm_vanilla_ar2_noseasonal", "regressors": "static", "seasonal": False, "ar2": True},
}
_DEFAULT_PRIORS = {
    "S": 10.0,
    "sigma_trend_prior": 1e-4,
    "sigma_seas_prior": 0.01,
    "sigma_AR_prior": 0.5,
    "sigma_reg_prior": 1e-4,
}


def _resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    name = model.lower().replace("-", "_")
    if name.startswith("dlm_"):
        name = name[4:]
    try:
        return name, _MODEL_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(_MODEL_SPECS)
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}.") from exc


def _resolve_priors(
    spec: Mapping[str, Any],
    nreg: int,
    priors: Mapping[str, Any] | None,
    direct_priors: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {"S", "sigma_trend_prior", "sigma_AR_prior"}
    if spec["seasonal"]:
        allowed.add("sigma_seas_prior")
    if spec["regressors"] == "dynamic":
        allowed.add("sigma_reg_prior")

    supplied = dict(priors or {})
    if not isinstance(priors, (Mapping, type(None))):
        raise TypeError("priors must be a mapping of Stan prior names to values.")
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Priors not used by this model: {', '.join(sorted(unknown))}.")

    for name, value in direct_priors.items():
        if value is None:
            continue
        if name in supplied:
            raise ValueError(f"Set {name!r} either directly or in priors, not both.")
        if name not in allowed:
            raise ValueError(f"Prior {name!r} is not used by this model.")
        supplied[name] = value

    resolved = {name: _DEFAULT_PRIORS[name] for name in allowed}
    resolved.update(supplied)
    for name, value in resolved.items():
        values = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Prior {name!r} must contain finite positive values.")
        if name == "sigma_reg_prior":
            if values.ndim == 0:
                values = np.full(nreg, float(values))
            if values.shape != (nreg,):
                raise ValueError(f"sigma_reg_prior must be a scalar or have length {nreg}.")
            resolved[name] = values.tolist()
        else:
            if values.ndim != 0:
                raise ValueError(f"Prior {name!r} must be a scalar.")
            resolved[name] = float(values)
    return resolved


def _time_coordinate(data: xr.DataArray) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be a one-dimensional xarray.DataArray with a 'time' dimension.")
    if data.dims != ("time",):
        raise ValueError("data must have exactly one dimension named 'time'.")
    if "time" in data.coords:
        return data.coords["time"]
    return xr.DataArray(np.arange(data.sizes["time"]), dims=("time",), name="time")


def _prepare_regressors(
    regressors: xr.DataArray | xr.Dataset | np.ndarray | None,
    time: xr.DataArray,
    mode: str,
) -> tuple[np.ndarray | None, list[str]]:
    if mode == "none":
        if regressors is not None:
            raise ValueError("This model does not accept regressors.")
        return None, []
    if regressors is None:
        raise ValueError("This model requires regressors.")

    if isinstance(regressors, xr.Dataset):
        names = list(regressors.data_vars)
        if not names or any(regressors[name].dims != ("time",) for name in names):
            raise ValueError("Each Dataset regressor must be one-dimensional with a 'time' dimension.")
        aligned = [regressors[name].interp(time=time).to_numpy() for name in names]
        values = np.column_stack(aligned)
    elif isinstance(regressors, xr.DataArray):
        if "time" not in regressors.dims:
            raise ValueError("regressors must have a 'time' dimension.")
        aligned = regressors.interp(time=time)
        if aligned.dims == ("time",):
            values = aligned.to_numpy()[:, None]
            names = [str(regressors.name or "regressor_1")]
        elif set(aligned.dims) == {"time", "regressor"}:
            aligned = aligned.transpose("time", "regressor")
            values = aligned.to_numpy()
            names = [str(value) for value in aligned.coords["regressor"].values]
        else:
            raise ValueError("DataArray regressors must have dimensions 'time' or ('time', 'regressor').")
    else:
        values = np.asarray(regressors, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != time.size:
            raise ValueError("NumPy regressors must have shape (time, regressor).")
        names = [f"regressor_{index + 1}" for index in range(values.shape[1])]

    if values.shape[0] != time.size or values.shape[1] == 0:
        raise ValueError("Regressors must align with time and contain at least one column.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Regressors must be finite after time alignment/interpolation.")
    return np.asarray(values, dtype=float), names


def _sampling_interval(sampling: str | float) -> float:
    rates = {"monthly": 1.0, "daily": 12.0 / 365.25, "annual": 12.0}
    if isinstance(sampling, str):
        try:
            return rates[sampling.lower()]
        except KeyError as exc:
            raise ValueError("sampling must be 'daily', 'monthly', 'annual', or a positive number.") from exc
    value = float(sampling)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sampling must be positive and finite.")
    return value

@lru_cache(maxsize=None)
def _compiled_model(source_name: str, digest: str, cache_root: str) -> CmdStanModel:
    source = getattr(_stan_models, source_name)
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stan_file = cache_dir / f"{source_name}-{digest}.stan"
    if not stan_file.exists():
        temporary = stan_file.with_name(f"{stan_file.name}.{os.getpid()}.tmp")
        temporary.write_text(source)
        os.replace(temporary, stan_file)
    return CmdStanModel(stan_file=str(stan_file))


def _get_compiled_model(
    model_name: str,
    spec: Mapping[str, Any],
    cache_dir: str | os.PathLike[str] | None = None,
) -> CmdStanModel:
    source = getattr(_stan_models, spec["source"])
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else Path(user_cache_dir("dlmmc")) / "stan"
    return _compiled_model(spec["source"], digest, str(cache_root.resolve()))


def compile_models(
    models: Sequence[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Compile selected packaged Stan models and return their executable paths."""
    selected = list(_MODEL_SPECS) if models is None else list(models)
    if not selected:
        raise ValueError("models must contain at least one model name.")
    compiled: dict[str, str] = {}
    for model in selected:
        model_name, spec = _resolve_model(model)
        compiled[model_name] = _get_compiled_model(model_name, spec, cache_dir).exe_file
    return compiled


@contextmanager
def _sampling_output_directory(
    output_dir: str | os.PathLike[str] | None,
) -> Iterator[str | os.PathLike[str]]:
    if output_dir is not None:
        yield output_dir
        return

    directory = tempfile.mkdtemp(prefix="dlmmc-")
    try:
        yield directory
    except BaseException:
        logging.getLogger(__name__).warning(
            "Fit did not complete; Stan output retained at %s", directory
        )
        raise
    else:
        shutil.rmtree(directory)


def fit(
    data: xr.DataArray,
    regressors: xr.DataArray | xr.Dataset | np.ndarray | None = None,
    model: str = "vanilla_ar1",
    *,
    uncertainty: xr.DataArray | np.ndarray | None = None,
    priors: Mapping[str, Any] | None = None,
    S: float | None = None,
    sigma_trend_prior: float | None = None,
    sigma_seas_prior: float | None = None,
    sigma_AR_prior: float | None = None,
    sigma_reg_prior: float | Sequence[float] | None = None,
    sampling: str | float = "monthly",
    chains: int = 4,
    warmup: int = 1000,
    samples: int = 1000,
    seed: int | None = None,
    inits: Any = None,
    parallel_chains: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    show_progress: bool = True,
    **sampler_options: Any,
) -> xr.Dataset:
    """Fit one time series and return posterior draws as a labeled xarray Dataset.

    ``data`` must be a one-dimensional DataArray with a ``time`` dimension.
    Regressors may be an xarray Dataset/DataArray or a NumPy array. Regressor
    xarray objects are interpolated onto the data time coordinate. If
    ``uncertainty`` is omitted, a tiny positive measurement uncertainty (1e-20)
    is used and the model's AR component accounts for residual variability.

    Prior scales default to the values used by the current examples and are in
    the normalized units used internally by the Stan models. Priors may be
    overridden individually or through ``priors``; duplicate specifications
    are rejected. ``inits`` supplies sampler starting values and may vary by
    chain; it does not change the prior or target posterior. ``cache_dir`` can
    point to shared storage when fitting across multiple MPI nodes.

    By default, Stan output is written to a per-fit temporary directory and
    deleted after the posterior and diagnostics are loaded into memory. Failed
    or interrupted fits retain their output and log its location. Pass
    ``output_dir`` through sampler options to keep output in a directory you
    manage; that directory is never removed by dlmmc. Set ``TMPDIR`` before
    starting Python to choose temporary storage.
    """
    model_name, spec = _resolve_model(model)
    time = _time_coordinate(data)
    if data.sizes["time"] < 2:
        raise ValueError("data must contain at least two time points.")

    raw_data = np.asarray(data.to_numpy(), dtype=float)
    if uncertainty is None:
        raw_uncertainty = np.full_like(raw_data, 1e-20)
    elif isinstance(uncertainty, xr.DataArray):
        if uncertainty.dims != ("time",):
            raise ValueError("uncertainty must be one-dimensional with a 'time' dimension.")
        try:
            data, uncertainty = xr.align(data, uncertainty, join="exact")
        except ValueError as exc:
            raise ValueError("uncertainty time coordinates must exactly match data.") from exc
        time = _time_coordinate(data)
        raw_data = np.asarray(data.to_numpy(), dtype=float)
        raw_uncertainty = np.asarray(uncertainty.to_numpy(), dtype=float)
    else:
        raw_uncertainty = np.asarray(uncertainty, dtype=float)
        if raw_uncertainty.shape != raw_data.shape:
            raise ValueError("uncertainty must have the same shape as data.")

    valid_data = np.isfinite(raw_data)
    if not valid_data.any():
        raise ValueError("data must contain at least one finite observation.")
    data_mean = float(np.mean(raw_data[valid_data]))
    filled_data = np.where(valid_data, raw_data, data_mean)
    missing = ~valid_data | ~np.isfinite(raw_uncertainty)
    if np.any(raw_uncertainty[~missing] < 0):
        raise ValueError("uncertainty values must be non-negative.")
    stddev = np.where(missing, 1e20, raw_uncertainty)

    regressor_values, regressor_names = _prepare_regressors(
        regressors, time, spec["regressors"]
    )
    nreg = 0 if regressor_values is None else regressor_values.shape[1]
    direct_priors = {
        "S": S,
        "sigma_trend_prior": sigma_trend_prior,
        "sigma_seas_prior": sigma_seas_prior,
        "sigma_AR_prior": sigma_AR_prior,
        "sigma_reg_prior": sigma_reg_prior,
    }
    prior_values = _resolve_priors(spec, nreg, priors, direct_priors)

    if chains < 1 or warmup < 0 or samples < 1:
        raise ValueError("chains and samples must be positive; warmup must be non-negative.")
    if parallel_chains is not None and not 1 <= parallel_chains <= chains:
        raise ValueError("parallel_chains must be between 1 and chains.")
    if isinstance(inits, Sequence) and not isinstance(inits, (str, bytes, Mapping)):
        if len(inits) != chains:
            raise ValueError("When inits is a sequence, provide one initialization per chain.")

    model_data = {
        "time_series": filled_data,
        "stddev": stddev,
        "N": data.sizes["time"],
        "sampling": _sampling_interval(sampling),
        "S": prior_values["S"],
        "sigma_trend_prior": prior_values["sigma_trend_prior"],
        "sigma_AR_prior": prior_values["sigma_AR_prior"],
    }
    if spec["seasonal"]:
        model_data["sigma_seas_prior"] = prior_values["sigma_seas_prior"]
    if regressor_values is not None:
        model_data.update(nreg=nreg, regressors=regressor_values)
    if spec["regressors"] == "dynamic":
        model_data["sigma_reg_prior"] = prior_values["sigma_reg_prior"]

    stan_model = _get_compiled_model(model_name, spec, cache_dir)
    sample_options = {
        "data": model_data,
        "chains": chains,
        "iter_warmup": warmup,
        "iter_sampling": samples,
        "seed": seed,
        "inits": inits,
        "show_progress": show_progress,
        **sampler_options,
    }
    if parallel_chains is not None:
        sample_options["parallel_chains"] = parallel_chains
    with _sampling_output_directory(sample_options.get("output_dir")) as output_dir:
        sample_options["output_dir"] = output_dir
        fit_result = stan_model.sample(**sample_options)

        output_names = ["trend", "slope", "ar", "sigma_trend", "sigma_AR", "rhoAR1"]
        if spec["seasonal"]:
            output_names.extend(["seasonal", "sigma_seas"])
        if spec["ar2"]:
            output_names.append("rhoAR2")
        if regressor_values is not None:
            output_names.append("beta")
        if spec["regressors"] == "dynamic":
            output_names.append("sigma_reg")
        draws = fit_result.draws_xr(vars=output_names)

        posterior_vars: dict[str, Any] = {}
        time_vars = {"trend", "slope", "ar", "seasonal"} & set(output_names)
        for name in time_vars:
            posterior_vars[name] = (("chain", "draw", "time"), draws[name].to_numpy())
        scalar_vars = set(output_names) - time_vars - {"beta", "sigma_reg"}
        for name in scalar_vars:
            posterior_vars[name] = (("chain", "draw"), draws[name].to_numpy())
        if "beta" in output_names:
            beta_values = draws["beta"].to_numpy()
            if spec["regressors"] == "dynamic":
                posterior_vars["beta"] = (("chain", "draw", "time", "regressor"), beta_values)
            else:
                posterior_vars["beta"] = (("chain", "draw", "regressor"), beta_values)
        if "sigma_reg" in output_names:
            posterior_vars["sigma_reg"] = (("chain", "draw", "regressor"), draws["sigma_reg"].to_numpy())

        coords: dict[str, Any] = {
            "chain": draws["chain"].to_numpy(),
            "draw": draws["draw"].to_numpy(),
            "time": time.to_numpy(),
        }
        if regressor_values is not None:
            coords["regressor"] = regressor_names
        posterior = xr.Dataset(posterior_vars, coords=coords)
        posterior.attrs.update(
            model=model_name,
            backend="cmdstanpy",
            chains=chains,
            warmup=warmup,
            samples_per_chain=samples,
            sampling_interval=_sampling_interval(sampling),
        )
        divergences = fit_result.method_variables().get("divergent__")
        if divergences is not None:
            posterior.attrs["divergent_transitions"] = int(np.asarray(divergences).sum())
    return posterior