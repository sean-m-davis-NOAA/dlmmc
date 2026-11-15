"""Named Gaussian prior specifications and explicit coupled sampling dispatch."""
from dataclasses import dataclass,field,asdict
import json,math
import numpy as np
from .spatial import GaussianSpatialPrior


@dataclass(frozen=True)
class SpatialPrior:
    """Underlying Gaussian mean/std before positive truncation.

    Lengths use {'log_pressure': ..., 'latitude': ...}, or altitude instead of
    log_pressure with an explicit altitude unit. Sigma units are 'normalized'
    or 'data/month'; the latter are divided by each panel's normalization scale.
    Scalars broadcast; arrays must have exactly one value per prepared panel.
    """
    mean: object
    std: object
    correlation_lengths: dict
    units: str = 'normalized'
    coordinate_units: dict = field(default_factory=dict)

    def bind(self,prepared,*,upper=math.inf):
        lengths=dict(self.correlation_lengths);axes=set(lengths)
        if axes not in ({'log_pressure','latitude'},{'altitude','latitude'}):
            raise ValueError('Supply correlation_lengths for log_pressure (or altitude) and latitude')
        vertical=next(a for a in axes if a!='latitude');coords=prepared.cell_coordinates
        if 'latitude' not in coords:raise ValueError('Prepared panels need named latitude coordinates')
        if vertical=='log_pressure':
            if vertical in coords:z=np.asarray(coords[vertical],float)
            elif 'pressure' in coords:
                pressure=np.asarray(coords['pressure'],float)
                if np.any(pressure<=0):raise ValueError('Positive pressure required for log-pressure distance')
                z=np.log(pressure)
            else:raise ValueError('Prepared panels need pressure or log_pressure coordinates')
            unit='dimensionless_log_pressure_ratio'
            if self.coordinate_units.get('log_pressure',unit)!=unit:raise ValueError('Log-pressure lengths are dimensionless natural-log pressure ratios')
        else:
            if vertical not in coords:raise ValueError('Prepared panels need altitude coordinates')
            z=np.asarray(coords[vertical],float);unit=self.coordinate_units.get('altitude')
            if not unit:raise ValueError('Declare altitude coordinate_units, e.g. km')
        if self.coordinate_units.get('latitude','degrees_north')!='degrees_north':raise ValueError('Latitude lengths use degrees_north')
        def vector(value):
            a=np.asarray(value,dtype=float)
            if a.ndim==0:return np.full(prepared.cells,float(a))
            if a.shape!=(prepared.cells,):raise ValueError('Prior mean/std must be scalar or one value per prepared panel')
            return a
        mean,std=vector(self.mean),vector(self.std)
        if self.units=='data/month':mean=mean/prepared.scale;std=std/prepared.scale;unit_field='normalized_ozone_per_month'
        elif self.units=='normalized':unit_field='normalized_ozone_per_month'
        elif self.units=='months':unit_field='months'
        else:raise ValueError("units must be 'normalized', 'data/month', or 'months'")
        return GaussianSpatialPrior(tuple(z),tuple(coords['latitude']),lengths[vertical],lengths['latitude'],vertical,unit,'degrees_north',tuple(mean),tuple(std),unit_field,0.,upper)


@dataclass(frozen=True)
class LagPrior:
    """Gaussian lag mean/std in months, strictly between zero and upper.

    None correlation_lengths inherits sigma_trend's lengths. A supplied mapping
    must match them for the current ENSO-only model. Other regressors stay fixed.
    """
    mean: float = 0.
    std: float = 2.
    upper: float = 12.
    correlation_lengths: dict | None = None


def sample_spatial_prior(prepared,*,sigma_trend,lags=None,predictors=None,walkers=128,
                         burnin=128,steps=256,ensembles=4,backend='numba',output_dir,**options):
    """Joint spatial fit; lag mapping explicitly selects the regressor to shift.

    Current scientific contract supports lags={'ENSO': LagPrior(...)} only.
    Unknown/multiple lag names fail rather than being silently ignored. Supply
    predictors={'ENSO': MonthlyENSO(...)} with complete pre-observation coverage.
    Separate scope/selection gates and exact prior definitions remain in force.
    """
    if not isinstance(sigma_trend,SpatialPrior):raise TypeError('sigma_trend must be a SpatialPrior')
    if sigma_trend.units=='months':raise ValueError('sigma_trend uses normalized or data/month units')
    if not isinstance(walkers,int) or walkers<2 or walkers%2:raise ValueError('walkers is an even total count, not a half-count')
    prior=sigma_trend.bind(prepared);lags={} if lags is None else dict(lags);predictors={} if predictors is None else dict(predictors)
    settings=dict(output_dir=output_dir,backend=backend,half_walkers=walkers//2,warmup=burnin,retained_sweeps=steps,runs=ensembles,**options)
    if not lags:
        if predictors:raise ValueError('Predictors supplied without an explicitly selected lag')
        from .spatial_sampling import sample_spatial
        settings.setdefault('update_schedule','field_local');fit=sample_spatial(prepared,prior,**settings)
    else:
        if set(lags)!={'ENSO'}:raise ValueError("Current validated lag contract supports {'ENSO': LagPrior(...)} only; other/multiple lag fields require a research extension")
        if set(predictors)!=set(lags):raise ValueError('Provide one fixed-coverage predictor for each selected lag name')
        lag=lags['ENSO']
        if not isinstance(lag,LagPrior):raise TypeError('Each selected lag needs a LagPrior')
        lengths=sigma_trend.correlation_lengths if lag.correlation_lengths is None else lag.correlation_lengths
        if lengths!=sigma_trend.correlation_lengths:raise ValueError('ENSO lag and sigma_trend must share correlation lengths')
        lp=SpatialPrior(lag.mean,lag.std,lengths,'months',sigma_trend.coordinate_units).bind(prepared,upper=lag.upper)
        from .lag_sampling import sample_enso_lag
        fit=sample_enso_lag(prepared,prior,lp,predictors['ENSO'],**settings)
    record=dict(sigma_trend=asdict(sigma_trend),lags={k:asdict(v) for k,v in lags.items()},walkers_are_total=walkers,independent_ensembles=ensembles,lag_convention='regressor(t-lag)',batching='Joint correlated field acceptance; conditionally independent nuisance blocks; all local likelihoods tensorized')
    (fit.path/'public-prior-specification.json').write_text(json.dumps(record,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    return fit


def fit_spatial(data,regressors=None,*,uncertainty=None,model='vanilla_ar1',priors=None,sampling='monthly',**options):
    """Prepare labeled inputs and dispatch a bounded spatial/ENSO fit.

    Ordinary preparation is capped at five cells. Larger explicitly selected
    grids use sample_spatial_prior with their separately prepared input record.
    """
    from .prepare import prepare_data
    from .input_contract import validate_no_seasonal_input
    validate_no_seasonal_input(data,model)
    p=prepare_data(data,regressors,uncertainty=uncertainty,model=model,priors=priors,sampling=sampling)
    return sample_spatial_prior(p,**options)
