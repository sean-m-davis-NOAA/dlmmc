import numpy as np
import pytest
import torch
from dlmmc.compute import available_compute, resolve_compute, select_backend, announce_cmdstan


def devices(monkeypatch, cuda=False, mps=False):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:cuda)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:mps)


@pytest.mark.parametrize('cuda,mps,chosen',[(True,True,'cuda'),(True,False,'cuda'),(False,True,'mps'),(False,False,'numba')])
@pytest.mark.parametrize('argument',[None,'auto'])
def test_priority_and_message(monkeypatch,capsys,cuda,mps,chosen,argument):
    devices(monkeypatch,cuda,mps)
    p=resolve_compute(argument,announce=True)
    assert p.backend==chosen
    assert p.dtype==('float64' if chosen=='numba' else 'float32')
    msg=capsys.readouterr().out
    assert 'available compute:' in msg and f"backend='{chosen}'" in msg
    assert "Change with backend=" in msg and 'CUDA > MPS > Numba CPU' in msg
    assert msg.count('DLMMC Affine:')==1


@pytest.mark.parametrize('backend',['numba','cpu','mps','cuda'])
def test_explicit_overrides(monkeypatch,backend):
    devices(monkeypatch,True,True)
    assert resolve_compute(backend).backend==('numba' if backend=='cpu' else backend)


def test_explicit_device_precision_and_missing_hardware(monkeypatch):
    devices(monkeypatch,True,True)
    assert resolve_compute(device='cpu').backend=='numba'
    assert resolve_compute(dtype='float64').backend=='cuda'
    with pytest.raises(ValueError,match='no longer'):resolve_compute('torch')
    assert resolve_compute(device='cuda:1').device=='cuda:1'
    with pytest.raises(ValueError,match='disagree'):resolve_compute('mps',device='cpu')
    with pytest.raises(ValueError,match='requires'):resolve_compute('mps',dtype='float64')
    devices(monkeypatch,False,False)
    with pytest.raises(RuntimeError,match='no CPU fallback'):resolve_compute('cuda')
    assert resolve_compute().backend=='numba'


def test_cmdstan_explains_cpu_and_gpu_switch(monkeypatch,capsys):
    devices(monkeypatch,True,True)
    announce_cmdstan()
    text=capsys.readouterr().out
    assert 'CUDA, MPS' in text and 'Selected CPU (float64)' in text
    assert "sampler='affine', backend='auto'" in text


def test_default_native_production_announces_once(tmp_path,capsys):
    from dlmmc import fit,AffineSettings,summarize_trends
    from test_sampler_selection import inputs
    y,s,x=inputs(24)
    selected=resolve_compute().backend
    result=fit(y,x,uncertainty=s,sampler='affine',
        settings=AffineSettings(walkers=16,burnin=4,steps=8,output_draws_per_step=2),
        output_dir=tmp_path/'fit',wall_seconds=120)
    assert result.config['backend']==selected
    msg=capsys.readouterr().out
    assert msg.count('DLMMC Affine:')==1 and f"backend='{selected}'" in msg
    summary=summarize_trends(result,output_dir=tmp_path/'trends',wall_seconds=60)
    assert np.isfinite(summary.trend).all()


def test_qualification_explicit_float64_keeps_cpu(tmp_path,capsys):
    from dlmmc import fit_grid
    from test_sampler_selection import inputs
    y,s,x=inputs(12)
    result=fit_grid(y,x,uncertainty=s,dtype='float64',output_dir=tmp_path/'fit',
        half_walkers=8,runs=4,warmup=2,retained_sweeps=2,chunk_sweeps=2,wall_seconds=120)
    assert result.config['backend']=='numba'
    assert result.prepared.y.dtype==torch.float64
    assert capsys.readouterr().out.count('DLMMC Affine:')==1


def test_qualification_default_uses_available_device(tmp_path):
    from dlmmc import fit_grid
    from test_sampler_selection import inputs
    y,s,x=inputs(12)
    expected=resolve_compute()
    fit=fit_grid(y,x,uncertainty=s,output_dir=tmp_path/'fit',
                 half_walkers=8,runs=4,warmup=2,retained_sweeps=2,chunk_sweeps=2,wall_seconds=120)
    assert fit.config['backend']==expected.backend
    assert str(fit.prepared.y.dtype)=='torch.'+expected.dtype
