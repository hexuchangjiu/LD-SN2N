"""Unpaired feature-distribution guidance for DriftingSN2N.

The attraction--repulsion construction follows Deng et al., Drifting Models.
Each spatial bin supplies one group of student queries and independently
sampled teacher positives. Student queries also supply the negative bank.
"""

import math

import torch
from torch import Tensor


def drifting_loss(queries: Tensor, positives: Tensor, *, temperatures=(0.02, 0.05, 0.2)):
    """Return the loss and a detached feature scale, using uniform sample weights.

    Arrays have shape [spatial bin, batch, channel]. Only the query argument
    receives gradients. The target is displaced in feature space at training
    time; it introduces no inference iterations.
    """
    if (queries.ndim != 3 or positives.ndim != 3 or queries.shape[1] < 2
            or min(positives.shape) < 1 or queries.shape[::2] != positives.shape[::2]
            or queries.device != positives.device):
        raise ValueError("banks must share groups, channels and device; need at least two queries")
    if not (queries.is_floating_point() and positives.is_floating_point()
            and queries.isfinite().all() and positives.isfinite().all()):
        raise ValueError("feature banks must be finite floating point")
    radii = tuple(temperatures)
    if not radii or any(not math.isfinite(t) or t <= 0 for t in radii):
        raise ValueError("temperatures must be positive and finite")
    students = queries.float()
    with torch.no_grad():
        x, p = students.detach(), positives.float().detach()
        bank = torch.cat([x, p], dim=1)
        square = x.square().sum(-1).unsqueeze(2) + bank.square().sum(-1).unsqueeze(1)
        distances = (square - 2 * (x @ bank.transpose(1, 2))).clamp_min(1e-8).sqrt()
        distance_scale = distances.mean()
        unit = (distance_scale / math.sqrt(x.size(2))).clamp_min(1e-3)
        cost = distances / distance_scale.clamp_min(1e-3)
        cost[:, :, :x.size(1)].diagonal(dim1=1, dim2=2).add_(100)
        x0, p0 = x / unit, p / unit
        displacement = torch.zeros_like(x0)
        for t in radii:
            score = -cost / t
            affinity = (score.softmax(1) * score.softmax(2)).clamp_min(1e-6).sqrt()
            neg, pos = affinity.split([x.size(1), p.size(1)], dim=2)
            # Product of the two masses times their barycentre difference.
            pull = (pos @ p0) * neg.sum(2, keepdim=True)
            push = (neg @ x0) * pos.sum(2, keepdim=True)
            field = pull - push
            displacement += field / field.square().mean().clamp_min(1e-8).sqrt()
        target = x0 + displacement
    loss = (students / unit - target).square().mean()
    return loss, {"scale_inputs": unit, "normalized_force_rms": displacement.square().mean().sqrt()}
