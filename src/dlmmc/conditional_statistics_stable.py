"""Float64 analytic Gaussian sign probabilities, including rare positive tails.

Separate from the earlier ndtr implementation so its cached evidence remains
reproducible. No probability floor, sampled-event substitution or model change.
"""
import torch
from .tensor_smoothing import contrast_moments


def normal_positive_probability(mean,variance):
    if mean.dtype!=torch.float64 or variance.dtype!=torch.float64:raise ValueError('Explicit float64 probability reference required')
    if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()) or bool((variance<0).any()):raise ValueError('Invalid Gaussian moments')
    positive=variance>0
    denominator=torch.sqrt(torch.where(positive,variance,torch.ones_like(variance)))
    # log_ndtr avoids the cancellation in (1+erf(z/sqrt(2)))/2 for negative z.
    probability=torch.exp(torch.special.log_ndtr(mean/denominator))
    return torch.where(positive,probability,(mean>0).to(mean.dtype))


def stable_positive_trend_probabilities(filtered,G,Q,contrasts):
    left=G.new_zeros(G.shape[-1]);left[0]=-1;right=-left;result=[]
    for a,b in contrasts:
        mean,variance=contrast_moments(filtered,G,Q,a,b,left,right)
        result.append(normal_positive_probability(mean,variance))
    return torch.stack(result,-1)
