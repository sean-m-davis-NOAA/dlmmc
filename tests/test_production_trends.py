from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from dlmmc import prepare_data
from dlmmc.examples import load_panels
from dlmmc.production_trends import fixed_trend_moments
from dlmmc.tensor_target import model_components
from dlmmc.precision_prototype import filter_tensor
from dlmmc.tensor_smoothing import conditional_moments, full_conditional_covariance


def test_trend_moments_and_anchored_uncertainty_match_gaussian_oracle():
    torch.set_num_threads(1)
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_data(y.isel(time=slice(0,24),panel=[0]),x.isel(time=slice(0,24),panel=[0]),uncertainty=s.isel(time=slice(0,24),panel=[0]))
    p.observed[:,5]=False
    u=np.array([[[-9.2,-4.6,-2.3,0.]], [[-11.,-6.,-3.,6.9]], [[-7.,-3.,-1.,-4.]]])
    actual=fixed_trend_moments(p,u,12)
    F,G,Q,C,_,_=model_components(torch.tensor(u),*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
    m,c,j=conditional_moments(filter_tensor(*p.arguments[:3],F,G,Q,C),G,Q)
    dense=full_conditional_covariance(c,j).numpy()[:,0];dim=G.shape[-1]
    expected=m[:,0,:,0].numpy()*p.scale[0]+p.offset[0]
    np.testing.assert_allclose(actual[:,:,0],expected,rtol=2e-11,atol=1e-12)
    np.testing.assert_allclose(actual[:,:,1],c[:,0,:,0,0].numpy()*p.scale[0]**2,rtol=2e-10,atol=1e-12)
    np.testing.assert_allclose(actual[:,:,2],expected-expected[:,12,None],rtol=2e-10,atol=1e-12)
    for t in range(24):
        v=(dense[:,t*dim,t*dim]+dense[:,12*dim,12*dim]-2*dense[:,t*dim,12*dim])*p.scale[0]**2
        np.testing.assert_allclose(actual[:,t,3],v,rtol=2e-9,atol=1e-12)
    np.testing.assert_array_equal(actual[:,12,2:],0.)


def test_five_panel_preparation_and_moments_match_reference():
    from dlmmc.production import prepare_production,_cell
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_production(y,x,uncertainty=s);q=prepare_data(y,x,uncertainty=s)
    for a,b in zip(p.arguments,q.arguments):np.testing.assert_array_equal(a,b)
    u=np.array([[[-9.2,-4.6,-2.3,0.]]])
    for g in range(5):
        one=_cell(p,[g]);actual=fixed_trend_moments(one,u,156)
        F,G,Q,C,_,_=model_components(torch.tensor(u),*one.arguments,one.model,S=one.priors['S'],sampling=one.sampling)
        m,c,_=conditional_moments(filter_tensor(*one.arguments[:3],F,G,Q,C),G,Q)
        np.testing.assert_allclose(actual[0,:,0],m[0,0,:,0].numpy()*one.scale[0]+one.offset[0],rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(actual[0,:,1],c[0,0,:,0,0].numpy()*one.scale[0]**2,rtol=2e-9,atol=2e-12)
