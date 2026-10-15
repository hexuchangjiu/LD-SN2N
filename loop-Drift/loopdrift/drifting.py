"""Compact equation-level Drifting example, not the E57 port.

Method reference: Deng et al., Generative Modeling via Drifting,
https://arxiv.org/html/2602.04770v1, equations 6, 8-12.
This implementation uses separately row-normalized exponential-distance kernels.
It does not reproduce the reference repository's additional normalizations.
See docs/ALGORITHMS.md for precise scope and provenance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _bank(value: Tensor, name: str, like: Tensor | None = None) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape [groups,samples,channels]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    if like is not None and (value.shape[0] != like.shape[0]
                             or value.shape[2] != like.shape[2]
                             or value.device != like.device):
        raise ValueError(f"{name} must match the generated groups, channels and device")


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@torch.no_grad()
def drifting_field(generated: Tensor, positive: Tensor, negative: Tensor | None = None,
                   *, temperature: float = 0.2) -> Tensor:
    """Return detached attraction-minus-repulsion vectors [G,N,D].

    For each generated x, V(x) = sum softmax(-|x-p|/tau)*p
                               - sum softmax(-|x-n|/tau)*n.
    Default negatives are the other generated examples (exact self exclusion).
    Explicit negatives are used as supplied, with no inferred correspondence.
    Positives must be independently sampled by the caller. Coordinates, including
    their units and intensity normalization, are the caller's responsibility.
    """
    _bank(generated, "generated")
    _bank(positive, "positive", generated)
    temperature = _positive(temperature, "temperature")
    self_bank = negative is None
    if self_bank:
        if generated.shape[1] < 2:
            raise ValueError("default negatives require at least two generated samples")
        negative = generated
    _bank(negative, "negative", generated)
    dtype = torch.float64 if generated.dtype == torch.float64 else torch.float32
    if temperature < torch.finfo(dtype).tiny:
        raise ValueError("temperature is too small for the calculation dtype")
    query, pos, neg = (value.detach().to(dtype=dtype)
                       for value in (generated, positive, negative))

    def center(bank, exclude_self=False):
        distances = torch.cdist(query, bank, p=2)
        if not torch.isfinite(distances).all():
            raise FloatingPointError("pairwise distances overflowed; rescale feature coordinates")
        # Subtract each row's minimum first to protect very small temperatures.
        if exclude_self:
            diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool)
            distances = distances.masked_fill(diagonal.unsqueeze(0), float("inf"))
        logits = -(distances - distances.amin(dim=-1, keepdim=True)) / temperature
        return torch.softmax(logits, dim=-1) @ bank

    field = center(pos) - center(neg, exclude_self=self_bank)
    if not torch.isfinite(field).all():
        raise FloatingPointError("nonfinite drifting field; rescale feature coordinates")
    return field


def drifting_loss(generated: Tensor, positive: Tensor, negative: Tensor | None = None,
                  *, temperature: float = 0.2, step_size: float = 1.0):
    """Per-coordinate MSE to stopgrad(x + step_size * V(x)).

    Only the generated prediction receives gradients. Positive/negative banks,
    the field and target are detached. No adaptive feature/force scaling occurs.
    Returns (scalar loss, detached diagnostics); no parameter update is performed.
    """
    step_size = _positive(step_size, "step_size")
    field = drifting_field(generated, positive, negative, temperature=temperature)
    prediction = generated.to(dtype=field.dtype)
    target = prediction.detach() + step_size * field
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("drifting loss overflowed; rescale features or step_size")
    amplitude = field.abs().amax()
    field_rms = (field / amplitude).square().mean().sqrt() * amplitude if amplitude > 0 else amplitude
    return loss, {"field_rms": field_rms, "target": target,
                  "loss_per_group": (prediction.detach() - target).square().mean((1, 2))}

