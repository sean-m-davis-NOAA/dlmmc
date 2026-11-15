import pytest
from dlmmc.compute import resolve_compute

@pytest.mark.parametrize('backend,device,dtype',[('torch','cpu','float64'),('numba','cpu','float64'),('cpu','cpu','float64'),('mps','mps','float32'),('cuda','cuda','float32'),('cuda','cuda:1','float32')])
def test_automatic_dtype(backend,device,dtype):
 p=resolve_compute(backend,device=device,check_available=False)
 assert p.dtype==dtype and p.device==device
 if backend=='cuda':assert p.validation=='CUDA_UNTESTED'

@pytest.mark.parametrize('backend,device,dtype',[('mps',None,'float64'),('cuda',None,'float64'),('numba',None,'float32'),('mps','cpu',None),('numba','cuda',None)])
def test_no_silent_fallback(backend,device,dtype):
 with pytest.raises(ValueError):resolve_compute(backend,device=device,dtype=dtype,check_available=False)


def test_missing_device_fails(monkeypatch):
 import torch
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 with pytest.raises(RuntimeError,match='no CPU fallback'):resolve_compute('cuda')


def test_prepared_backend_transfer():
 import torch
 from dlmmc.compute import prepared_for_backend
 from test_spatial_prior import prepared
 p=prepared()
 assert prepared_for_backend(p,'numba') is p
 if not torch.backends.mps.is_available():pytest.skip('Native MPS unavailable')
 gpu=prepared_for_backend(p,'mps')
 assert gpu.y.device.type=='mps' and gpu.y.dtype==torch.float32
 assert gpu.observed.dtype==torch.bool
 assert gpu.checksum==p.checksum
