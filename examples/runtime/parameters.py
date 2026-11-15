"""Inference for the two example checkpoints. NPZ files contain numeric arrays only."""
import numpy as np
import torch
from torch.nn import functional as F


class _FrozenDenoiser(torch.nn.Module):
    def __init__(self, arrays):
        super().__init__()
        for key, value in arrays.items():
            self.register_buffer(key.replace(".", "_"), torch.from_numpy(value))

    def _apply_pair(self, x, prefix):
        for c, b in ((0, 1), (3, 4)):
            conv = f"{prefix}_double_conv_{c}_"
            norm = f"{prefix}_double_conv_{b}_"
            x = F.conv2d(x, getattr(self, conv+"weight"), getattr(self, conv+"bias"), padding=1)
            x = F.batch_norm(x, getattr(self, norm+"running_mean"), getattr(self, norm+"running_var"),
                             getattr(self, norm+"weight"), getattr(self, norm+"bias"), training=False, eps=1e-5)
            x = F.leaky_relu(x, negative_slope=0.02)
        return x

    def forward(self, x):
        levels = [self._apply_pair(x, "inc")]
        for depth in range(1, 5):
            levels.append(self._apply_pair(F.max_pool2d(levels[-1], 2), f"down{depth}_maxpool_conv_1"))
        x = levels.pop()
        for stage, side in enumerate(reversed(levels), 1):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
            dy, dx = side.shape[-2]-x.shape[-2], side.shape[-1]-x.shape[-1]
            if dx or dy:
                x = F.pad(x, (dx//2, dx-dx//2, dy//2, dy-dy//2))
            x = self._apply_pair(torch.cat((side, x), dim=1), f"up{stage}_conv")
        return torch.sigmoid(F.conv2d(x, self.outc_conv_weight, self.outc_conv_bias))


def load_model(path, map_location="cpu"):
    """Read the registered numeric weights and construct an inference-only module."""
    blocks = ["inc"] + [f"down{i}.maxpool_conv.1" for i in range(1,5)] + [f"up{i}.conv" for i in range(1,5)]
    expected = {"outc.conv.weight", "outc.conv.bias"}
    for block in blocks:
        for layer in (0, 1, 3, 4):
            fields = ("weight","bias") if layer in (0,3) else ("weight","bias","running_mean","running_var","num_batches_tracked")
            expected.update(f"{block}.double_conv.{layer}.{field}" for field in fields)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("Checkpoint tensor names do not match the example network.")
        buffers = {key: archive[key] for key in archive.files}
    for key, a in buffers.items():
        dtype = np.int64 if key.endswith("num_batches_tracked") else np.float32
        if a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError(f"Invalid numeric tensor: {key}")
    return _FrozenDenoiser(buffers).eval().to(map_location)
