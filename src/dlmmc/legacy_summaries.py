"""Project the original vanilla Stan generated quantities onto shared summaries."""
import numpy as np
from .validation_summaries import scientific_summaries


def vanilla_generated_summaries(parameters, *, trend, slope, seasonal_components,
        ar, beta, scale, offset, time_indices, contrasts, amplitude_times):
    """Summarize actual saved GQ outputs, without inventing a full state history.

    Original Stan exports static coefficients only at the first observation.
    The shared prespecified summary map also queries these coefficients only
    there. Unqueried coefficient slots are temporary zero placeholders; the
    projection is never returned as a reconstructed trajectory or used by a
    likelihood. Every queried state coordinate comes from the original GQ.
    Parameters use normalized model units; GQ outputs use observation units.
    """
    trend=np.asarray(trend);slope=np.asarray(slope);seasonal_components=np.asarray(seasonal_components)
    ar=np.asarray(ar);beta=np.asarray(beta);parameters=np.asarray(parameters)
    if trend.ndim!=2 or slope.shape!=trend.shape or ar.shape!=trend.shape:
        raise ValueError('Matching draw/time arrays required')
    draws,n=trend.shape
    if seasonal_components.shape!=(draws,n,4) or beta.ndim!=2 or beta.shape[0]!=draws:
        raise ValueError('Original seasonal and first-time coefficient dimensions required')
    if parameters.shape!=(draws,4) or not np.isfinite(scale) or scale<=0:
        raise ValueError('Four vanilla parameters and positive normalization scale required')
    if not all(np.isfinite(a).all() for a in (parameters,trend,slope,seasonal_components,ar,beta)):
        raise ValueError('Nonfinite original draws cannot be discarded')
    k=beta.shape[-1]
    states=np.zeros((draws,n,7+k),dtype=np.float64)
    states[...,0]=(trend-offset)/scale;states[...,1]=slope/scale
    states[...,2:6]=seasonal_components/scale;states[...,-1]=ar/scale
    states[:,0,6:6+k]=beta/scale
    return scientific_summaries('vanilla_ar1',parameters,states,nreg=k,scale=scale,offset=offset,
        time_indices=time_indices,contrasts=contrasts,amplitude_times=amplitude_times)
