"""Explicit spatial feature layout with no supplied pretrained encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _image(value: Tensor, name="image"):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 4 or min(value.shape) < 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite nonempty BCHW")


def spatial_feature_groups(features: Mapping[str, Tensor], pool_size: int = 4):
    """Equal-bin spatial means: BCHW -> [pool_size**2,B,C].

    B denotes independent images/crops, not pixels or biological replicates.
    Nondivisible spatial dimensions are rejected rather than changing the bins.
    """
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    grouped = {}
    for name, values in features.items():
        _image(values, str(name))
        b, c, h, w = values.shape
        if h % pool_size or w % pool_size or min(h, w) < pool_size:
            raise ValueError(f"{name}: spatial dimensions must be divisible by pool_size")
        kernel = (h // pool_size, w // pool_size)
        means = torch.nn.functional.avg_pool2d(values, kernel, stride=kernel)
        grouped[name] = means.permute(2, 3, 0, 1).reshape(pool_size ** 2, b, c)
    return grouped


def image_feature_groups(image: Tensor, learned_features: Mapping[str, Tensor] | None = None,
                         pool_size: int = 4):
    """Optional learned groups plus raw mean/RMS, in unchanged intensity units.

    No MAE weights or architecture are bundled. Learned features must already
    come from a frozen encoder, while retaining the student image's gradient.
    """
    _image(image)
    if image.shape[1] != 1:
        raise ValueError("image must have one grayscale channel")
    features = dict(learned_features or {})
    if {"raw_mean", "raw_rms"} & features.keys():
        raise ValueError("learned feature names raw_mean/raw_rms are reserved")
    for name, feature in features.items():
        _image(feature, str(name))
        if feature.shape[0] != image.shape[0] or feature.device != image.device:
            raise ValueError("learned features must match image batch and device")
    groups = spatial_feature_groups(features, pool_size)
    raw = spatial_feature_groups({"raw_mean": image, "raw_squared": image.square()}, pool_size)
    groups["raw_mean"] = raw["raw_mean"]
    groups["raw_rms"] = (raw["raw_squared"] + 1e-8).sqrt()
    if any(not torch.isfinite(value).all() for value in groups.values()):
        raise FloatingPointError("feature moments overflowed; rescale image coordinates")
    return groups

