"""Paired distillation, unpaired Drifting and spatial SN2N objectives.

No pair generation, architecture, optimizer, training data or teacher is included.
The caller owns sampling independence and a common intensity/feature coordinate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .drifting import drifting_loss
from .features import _image, image_feature_groups


def _matching(a: Tensor, b: Tensor):
    _image(a, "prediction")
    _image(b, "target")
    if a.shape != b.shape or a.device != b.device:
        raise ValueError("paired tensors must have identical shapes and devices")


def _coefficient(value, name, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _finite_loss(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("objective overflowed; rescale the input coordinate or coefficients")
    return loss


def pixel_objective(output: Tensor, paired_teacher: Tensor) -> Tensor:
    """Paired per-pixel MSE; the teacher image is always detached."""
    _matching(output, paired_teacher)
    return _finite_loss((output - paired_teacher.detach()).square().mean())


def sn2n_constraint(output_a: Tensor, output_b: Tensor | None,
                    noisy_a: Tensor, noisy_b: Tensor, *, reg: float = 0.5):
    """Spatial L1 SN2N loss with zero sparsity coefficient.

    For reg > 0: (MAE(D(a),b)+MAE(D(b),a)+reg*MAE(D(a),D(b)))/(2+reg).
    Both predictions retain gradients in the consistency term. For reg == 0,
    preserve the historical branch MAE(D(a),b), NOT a bidirectional average.
    Noisy pair targets are detached; the caller supplies valid self-supervised
    pairs and model outputs. This does not implement wavelet or sparse variants.
    """
    reg = _coefficient(reg, "reg")
    _matching(output_a, noisy_b)
    _matching(noisy_a, noisy_b)
    forward = _finite_loss((output_a - noisy_b.detach()).abs().mean())
    if reg == 0:
        return forward, {"forward": forward.detach()}
    if output_b is None:
        raise ValueError("output_b is required when reg > 0")
    _matching(output_b, noisy_a)
    reverse = (output_b - noisy_a.detach()).abs().mean()
    agreement = (output_a - output_b).abs().mean()
    loss = _finite_loss((forward + reverse + reg * agreement) / (2.0 + reg))
    return loss, {"forward": forward.detach(), "reverse": reverse.detach(),
                  "agreement": agreement.detach()}


def feature_objectives(output: Tensor, paired_teacher: Tensor | None,
                       positive: Tensor, encoder: nn.Module,
                       *, pool_size: int = 4, need_feature: bool = True,
                       need_drift: bool = True):
    """Equal-weight objectives across the same named feature groups.

    The frozen encoder exposes .features(image) -> named feature maps.
    Both objectives use the same detached scale from student queries and
    independent teacher positives, even when only the paired loss is used.
    The paired target retains sample correspondence; the positive bank does not.
    """
    if not need_feature and not need_drift:
        return {}, {}
    _image(output)
    if not isinstance(encoder, nn.Module) or not callable(getattr(encoder, "features", None)):
        raise ValueError("encoder must be an nn.Module with a features method")
    if any(module.training for module in encoder.modules()) or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("encoder must be in eval mode with all parameters frozen")

    def extract(value):
        learned = encoder.features(value)
        if not isinstance(learned, Mapping):
            raise ValueError("encoder.features must return a named mapping")
        return image_feature_groups(value, learned, pool_size)

    generated = extract(output)
    with torch.no_grad():
        if need_feature:
            if paired_teacher is None:
                raise ValueError("paired_teacher is required for feature objective")
            _matching(output, paired_teacher)
            paired = extract(paired_teacher.detach())
        _image(positive, "independent positives")
        if output.shape[1:] != positive.shape[1:] or output.device != positive.device:
            raise ValueError("positive must match output channels, spatial shape and device")
        positives = extract(positive.detach())
    losses, diagnostics = {}, {}
    feature_terms, drift_terms = [], []
    for name, values in generated.items():
        if name not in positives:
            raise ValueError("positive encoder feature names changed")
        drift, detail = drifting_loss(values, positives[name])
        scale = detail["scale_inputs"]
        diagnostics[name] = detail
        if need_feature:
            if name not in paired or paired[name].shape != values.shape:
                raise ValueError("paired encoder feature layout changed")
            feature_terms.append(((values - paired[name]) / scale).square().mean())
        if need_drift:
            drift_terms.append(drift)
    if need_feature:
        losses["feature"] = _finite_loss(torch.stack(feature_terms).mean())
    if need_drift:
        losses["drift"] = _finite_loss(torch.stack(drift_terms).mean())
    return losses, diagnostics


def combine_objectives(losses: Mapping[str, Tensor], *, family: str,
                        family_scale: float = 1.0, sn2n_weight: float = 0.0):
    """family_scale*L_family + sn2n_weight*L_sn2n (or SN2N-only).

    Fixed caller-supplied coefficients; no calibration, tuning or update occurs.
    For family='sn2n', coefficients must be family_scale=1, sn2n_weight=0;
    the function returns the supplied SN2N loss exactly once.
    """
    if family not in {"pixel", "feature", "drift", "sn2n"}:
        raise ValueError("family must be pixel, feature, drift or sn2n")
    scale = _coefficient(family_scale, "family_scale", positive=True)
    weight = _coefficient(sn2n_weight, "sn2n_weight")
    required = {family} | ({"sn2n"} if weight else set())
    for name in required:
        value = losses.get(name)
        if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 0 or not torch.isfinite(value):
            raise ValueError(f"{name} must be a finite floating-point scalar loss")
    if family == "sn2n":
        if scale != 1 or weight != 0:
            raise ValueError("SN2N-only uses family_scale=1 and sn2n_weight=0")
        return losses["sn2n"]
    result = scale * losses[family]
    if weight:
        if losses[family].device != losses["sn2n"].device:
            raise ValueError("objective losses must share a device")
        result = result + weight * losses["sn2n"]
    return _finite_loss(result)
