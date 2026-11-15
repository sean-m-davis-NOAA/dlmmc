from pathlib import Path
import numpy as np
import torch
from dlmmc.examples import load_panels
from dlmmc import prepare_data
from dlmmc.production import _cell
from dlmmc.production_trends import fixed_trend_moments
from dlmmc.production_smoothing import compact_arguments, _matrices, fixed_grid_moments, _reduce_grid
from dlmmc.tensor_target import model_components
from dlmmc.sampling import _threads


def test_compact_matrices_and_full_grid_moments():
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_data(y,x,uncertainty=s)
    u=np.broadcast_to(np.array([[-9.2,-4.6,-2.3,0.],[-11.,-6.,-3.,6.9],[-7.,-3.,-1.,-4.]]), (5,3,4)).transpose(1,0,2).copy()
    u+=np.arange(5)[None,:,None]*.001
    with _threads('numba',1,8):
        args=compact_arguments(p,u,156)
        F,G,Q,C,_,_=model_components(torch.tensor(u),*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)
        for j in range(3):
            for g in range(5):
                gg,qq,cc=_matrices(args[4][g],args[5][g],args[6][g],args[7][g,j])
                for a,b in [(args[3][g],F[j,g].numpy()),(gg,G[j,g].numpy()),(qq,Q[j,g].numpy()),(cc,C[j,g].numpy())]:
                    np.testing.assert_array_equal(a,b)
        full=fixed_grid_moments(p,u,156)
        for g in range(5):
            reference=fixed_trend_moments(_cell(p,[g]),u[:,g:g+1],156)
            np.testing.assert_array_equal(full[:,g],reference)
        order=[4,2,0,3,1]
        perm=fixed_grid_moments(_cell(p,order),u[:,order],156)
        np.testing.assert_array_equal(perm[:,np.argsort(order)],full)
        for subset in ([0,1],[2,3,4],[0]):
            np.testing.assert_array_equal(fixed_grid_moments(_cell(p,subset),u[:,subset],156),full[:,subset])
        changed=_cell(p,list(range(5)));changed.y=changed.y.clone();changed.y[0]+=1
        altered=fixed_grid_moments(changed,u,156)
        np.testing.assert_array_equal(altered[:,1:],full[:,1:])


def test_fused_mixture_reduction_matches_explicit_moments():
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    p=prepare_data(y.isel(time=slice(0,24)),x.isel(time=slice(0,24)),uncertainty=s.isel(time=slice(0,24)))
    p.observed[:,5]=False
    u=np.array([-9.2,-4.6,-2.3,0.])+np.random.default_rng(901).normal(size=(8,5,4))*.2
    with _threads('numba',1,8):
        args=compact_arguments(p,u,12);actual,halves,series=_reduce_grid(*args,2)
        # Same standardized fixed moments; center before variance accumulation.
        from dlmmc.production_smoothing import _fixed_grid
        fixed=_fixed_grid(*args)
    for k in (0,2):
        center=fixed[:,:,:,k].mean(0)
        variance=(fixed[:,:,:,k+1]+(fixed[:,:,:,k]-center)**2).mean(0)
        np.testing.assert_allclose(actual[:,:,k],center,rtol=2e-12,atol=1e-13)
        np.testing.assert_allclose(actual[:,:,k+1],variance,rtol=2e-12,atol=1e-13)
    np.testing.assert_allclose(halves[:,0],fixed[:4,:,:,0].mean(0),rtol=2e-12,atol=1e-13)
    np.testing.assert_allclose(series,fixed[:,:,[0,12,23],0].reshape(4,2,5,3).mean(1).transpose(1,0,2),rtol=2e-12,atol=1e-13)
    np.testing.assert_array_equal(actual[:,12,2:],0)
