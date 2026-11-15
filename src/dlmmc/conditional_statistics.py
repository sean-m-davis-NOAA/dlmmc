"""Exact Gaussian conditional sign probabilities for retained parameters."""
import torch
from .tensor_smoothing import contrast_moments


def positive_trend_probabilities(filtered,G,Q,contrasts):
    """Integrate the Gaussian state conditional analytically, without floors.

    The remaining uncertainty is parameter MCMC uncertainty. A genuine zero
    conditional variance is treated as a point mass; negative variance raises
    in contrast_moments instead of being clipped.
    """
    left=G.new_zeros(G.shape[-1]);left[0]=-1;right=-left
    result=[]
    for a,b in contrasts:
        mean,variance=contrast_moments(filtered,G,Q,a,b,left,right)
        positive=variance>0
        denominator=torch.sqrt(torch.where(positive,variance,torch.ones_like(variance)))
        probability=torch.special.ndtr(mean/denominator)
        result.append(torch.where(positive,probability,(mean>0).to(mean.dtype)))
    return torch.stack(result,-1)
