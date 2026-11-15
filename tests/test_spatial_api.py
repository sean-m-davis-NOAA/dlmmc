from dataclasses import replace
import numpy as np,pytest
from dlmmc import SpatialPrior,LagPrior,sample_spatial_prior
from test_enso_lag import fixture


def prepared():
 p,s,l,x=fixture();p=replace(p,cell_coordinates={**p.cell_coordinates,'log_pressure':np.array(s.vertical),'latitude':np.array(s.latitude)})
 return p,s,l,x


def test_named_lengths_mean_std_and_physical_units():
 p,s,_,_=prepared();spec=SpatialPrior(0,1e-4,{'log_pressure':s.length_vertical,'latitude':s.length_latitude})
 assert spec.bind(p)==s
 physical=SpatialPrior(np.asarray(s.mean)*p.scale,np.asarray(s.scale)*p.scale,spec.correlation_lengths,'data/month').bind(p)
 np.testing.assert_allclose(physical.covariance(),s.covariance(),rtol=1e-14)
 with pytest.raises(ValueError,match='correlation_lengths'):SpatialPrior(0,1,{'pressure':1,'latitude':10}).bind(p)
 with pytest.raises(ValueError,match='scales'):SpatialPrior(0,-1,spec.correlation_lengths).bind(p)


def test_explicit_lag_selection_and_joint_dispatch(tmp_path,monkeypatch):
 from types import SimpleNamespace
 import dlmmc.spatial_sampling as spatial,dlmmc.lag_sampling as lag
 p,s,l,x=prepared();spec=SpatialPrior(0,1e-4,{'log_pressure':s.length_vertical,'latitude':s.length_latitude})
 calls=[]
 def fake(*args,**kw):calls.append((args,kw));return SimpleNamespace(path=tmp_path)
 monkeypatch.setattr(spatial,'sample_spatial',fake);monkeypatch.setattr(lag,'sample_enso_lag',fake)
 sample_spatial_prior(p,sigma_trend=spec,output_dir=tmp_path)
 assert calls[-1][1]['update_schedule']=='field_local'
 sample_spatial_prior(p,sigma_trend=spec,lags={'ENSO':LagPrior()},predictors={'ENSO':x},output_dir=tmp_path)
 assert calls[-1][0][2]==l
 assert calls[-1][1]['half_walkers']==64 and calls[-1][1]['warmup']==128 and calls[-1][1]['retained_sweeps']==256
 for lags,predictors in [({'typo':LagPrior()},{'typo':x}),({'ENSO':LagPrior()},{}),({}, {'ENSO':x})]:
  with pytest.raises(ValueError):sample_spatial_prior(p,sigma_trend=spec,lags=lags,predictors=predictors,output_dir=tmp_path)
 with pytest.raises(ValueError,match='share'):sample_spatial_prior(p,sigma_trend=spec,lags={'ENSO':LagPrior(correlation_lengths={'log_pressure':1,'latitude':2})},predictors={'ENSO':x},output_dir=tmp_path)


def test_declared_axis_units_and_high_level_scope(tmp_path):
 from dlmmc import fit_spatial
 import xarray as xr
 p,s,_,_=prepared();lengths={'log_pressure':s.length_vertical,'latitude':s.length_latitude}
 with pytest.raises(ValueError,match='dimensionless'):SpatialPrior(0,1e-4,lengths,coordinate_units={'log_pressure':'km'}).bind(p)
 data=xr.DataArray(np.arange(36).reshape(6,6),dims=('time','panel'),coords={'time':np.arange(6),'panel':np.arange(6)})
 with pytest.raises(ValueError):fit_spatial(data,model='noregs_ar1',sigma_trend=SpatialPrior(0,1e-4,lengths),output_dir=tmp_path/'bad')
 assert not (tmp_path/'bad').exists()
