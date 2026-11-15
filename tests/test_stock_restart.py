"""Stage-01 thin=1 restart accounting only; not general sampler extensions."""
import torch
from affine.torch import sample

def test_thin_one_chunked_terminal_state():
    def lp(x):return -.5*(x*x).sum(-1)
    torch.manual_seed(20261104)
    halves=tuple(torch.randn(8,2) for _ in range(2));rng=torch.get_rng_state()
    full=sample(lp,24,halves,progress=False)
    torch.set_rng_state(rng)
    first=sample(lp,8,halves,progress=False)
    terminal=first[-1]
    second=sample(lp,17,(terminal[:8],terminal[8:]),progress=False)
    assert torch.equal(full,torch.cat((first,second[1:]),0))

def test_batch_restart_preserves_cells_and_rng():
    from affine.torch import sample_batch
    for dtype in [torch.float32,torch.float64]:
        def lp(x):return -.5*(x*x).sum(-1)
        torch.manual_seed(20261104)
        halves=tuple(torch.randn(8,5,2,dtype=dtype) for _ in range(2));rng=torch.get_rng_state()
        full=sample_batch(lp,24,halves,progress=False)
        torch.set_rng_state(rng)
        first=sample_batch(lp,8,halves,progress=False);terminal=first[-1]
        second=sample_batch(lp,17,(terminal[:8],terminal[8:]),progress=False)
        assert torch.equal(full,torch.cat((first,second[1:]),0))
