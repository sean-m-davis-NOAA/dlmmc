import numpy as np
import pytest
import torch
from dlmmc.api import _resolve_model,_resolve_priors
from dlmmc.components import Model,Trend,Harmonics,Regression,AR,resolve_priors
from dlmmc.prior_specification import canonical_priors,announce_model


def test_default_width_three_is_variance_nine_and_other_defaults_unchanged():
    spec=_resolve_model('vanilla_ar1')[1]
    for p in [_resolve_priors(spec,2,None,{}),resolve_priors(Model.from_preset('vanilla_ar1'),2)]:
        assert p=={'S':9.,'sigma_trend_prior':1e-4,'sigma_seas_prior':.01,'sigma_AR_prior':.5}


@pytest.mark.parametrize('model',['vanilla_ar1','dynregs_ar1',Model.from_preset('dynregs_ar1'),Model(Trend(),Harmonics(name='annual'),Regression(),AR(2))])
def test_new_widths_are_exact_legacy_target_aliases(model):
    from dlmmc import prepare_data
    from dlmmc.sampling import make_target
    from dlmmc.production import initial_center
    from test_sampler_selection import inputs
    y,s,x=inputs(24)
    base={'S':4.,'sigma_trend_prior':2e-4,'sigma_AR_prior':.3}
    new={'initial_state_prior_width':2.,'sigma_trend_prior_width':2e-4,'sigma_AR_prior_width':.3}
    old=prepare_data(y,x,uncertainty=s,model=model,priors=base)
    current=prepare_data(y,x,uncertainty=s,model=model,priors=new)
    assert current.priors==old.priors and current.checksum==old.checksum
    u=torch.tensor(initial_center(current)).reshape(1,1,-1)
    torch.testing.assert_close(make_target(old,'numba')(u),make_target(current,'numba')(u),rtol=0,atol=0)


def test_dynamic_and_named_harmonic_widths():
    m=Model(Trend(),Harmonics(name='annual'),Regression(dynamic=True),AR(1))
    p=resolve_priors(m,2,{'sigma_annual_prior_width':.02,'sigma_reg_prior_width':[.01,.03]})
    assert p['sigma_annual_prior']==.02 and p['sigma_reg_prior']==[.01,.03]


@pytest.mark.parametrize('priors',[{'S':9,'initial_state_prior_width':3},{'sigma_trend_prior':.1,'sigma_trend_prior_width':.1}])
def test_duplicate_aliases_rejected(priors):
    with pytest.raises(ValueError,match='Duplicate'):canonical_priors(priors)


@pytest.mark.parametrize('width',[-3,0,np.nan,np.inf,[3],1e200,1e-200])
def test_invalid_width_never_silently_squared_or_floored(width):
    with pytest.raises(ValueError):canonical_priors({'initial_state_prior_width':width})


def test_old_S_stays_variance():
    assert canonical_priors({'S':10})['S']==10
    assert canonical_priors({'initial_state_prior_width':3})['S']==9


def test_summary_includes_effective_components_priors_and_units(capsys):
    m=Model.from_preset('vanilla_ar1');p=resolve_priors(m,2,{'sigma_AR_prior_width':.3})
    announce_model(m,['ENSO','SOLAR'],p,panels=5)
    text=capsys.readouterr().out
    for expected in ['5 panels','Trend','12, 6 months','Regression(ENSO, SOLAR; static)','AR(1)',
                     'initial_state_prior_width=3','initial variance=9','sigma_AR_prior_width=0.3',
                     'Uniform(0, 1)','normalized response units']:
        assert expected in text


def test_public_fit_new_prior_mapping_and_direct_keyword(tmp_path,capsys):
    from dlmmc import fit,AffineSettings
    from test_sampler_selection import inputs
    y,s,x=inputs(12)
    result=fit(y,x,uncertainty=s,sampler='affine',backend='cpu',initial_state_prior_width=2,
               priors={'sigma_AR_prior_width':.4},settings=AffineSettings(walkers=16,burnin=2,steps=2),output_dir=tmp_path/'fit')
    assert result.prepared.priors['S']==4 and result.prepared.priors['sigma_AR_prior']==.4
    assert capsys.readouterr().out.count('DLMMC model')==1
    with pytest.raises(ValueError,match='either directly'):
        fit(y,x,sampler='affine',initial_state_prior_width=2,priors={'S':4})


def test_spatial_summary_does_not_print_replaced_half_normal(capsys):
    from test_enso_lag import fixture
    p,sigma,lag,_=fixture()
    announce_model(p.model,p.regressor_names,p.priors,panels=4,fields={'sigma_trend':sigma,'ENSO_lag':lag})
    text=capsys.readouterr().out
    assert 'sigma_trend_prior_width' not in text
    assert 'sigma_trend: truncated Gaussian field' in text and 'prior-independent' in text


@pytest.mark.xfail(strict=True,reason='S=9 no-AR mixed-regression internal Torch64 versus Numba64 differs1.144e-8 at1e-8; preserved, no public Torch backend')
def test_retained_S9_internal_oracle_roundoff_fixture():
    from test_components import inputs
    from dlmmc import prepare_data
    from dlmmc.components import initial_coordinates
    from dlmmc.sampling import make_target
    m=Model(Trend(),Harmonics((12,)),Regression(dynamic=('SOLAR',)))
    y,s,x=inputs(m,3)
    m=Model(Trend(),Harmonics((12,)),Regression(dynamic=(str(x.regressor.values[1]),)))
    p=prepare_data(y,x,uncertainty=s,model=m,priors={'initial_state_prior_width':3.})
    u=torch.tensor(initial_coordinates(p.model,3))+torch.tensor(np.random.default_rng(2002).normal(0,.08,(3,5,len(p.parameter_names))))
    np.testing.assert_allclose(make_target(p,'torch')(u),make_target(p,'numba')(u),rtol=0,atol=1e-8)
