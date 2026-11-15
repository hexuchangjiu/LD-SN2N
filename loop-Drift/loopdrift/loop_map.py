"""Spatial stability, gradient gain and their bivariate Loop Response Map."""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb

from .energy import _grad_field


def _validate_Y(Y: np.ndarray) -> np.ndarray:
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Y must have shape (K,T,H,W); got ndim={arr.ndim}, shape={arr.shape}")
    K, T, H, W = arr.shape
    if K < 1 or T < 2 or H < 2 or W < 2:
        raise ValueError(f"Y requires K>=1, T>=2 and H,W>=2; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Y contains NaN or Inf")
    return arr


def _positive_scalar(value, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a finite positive scalar; got {value!r}")
    return out


def _pair(value, name: str, *, percentile: bool = False) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lo, hi = float(value[0]), float(value[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{name} must satisfy finite lo < hi; got {value!r}")
    if percentile and (lo < 0 or hi > 100):
        raise ValueError(f"{name} percentiles must lie in [0,100]; got {value!r}")
    return lo, hi


def signature_map_from_Y(Y, *, eps, horizon=3, delta=0.20, grad_sigma=2.0,
                         ratio_eps=1e-6, structure_floor=1e-6):
    """Spatial readout of a saved [stream,state,y,x] trajectory.

    The states and reference scale eps must use the same intensity units.
    C_stab measures dispersion and first-step displacement; gamma is the gain
    ratio at its largest departure (one is neutral). A
    flat reference gradient has no interpretable direction: see structure_valid.
    """
    y = _validate_Y(Y)
    if isinstance(horizon, bool) or int(horizon) != horizon or not 2 <= horizon <= y.shape[1]:
        raise ValueError("horizon must be an available integer number of states >= 2")
    y = y[:, :int(horizon)]
    eps = _positive_scalar(eps, "eps")
    delta = _positive_scalar(delta, "delta")
    grad_sigma = _positive_scalar(grad_sigma, "grad_sigma")
    ratio_eps = _positive_scalar(ratio_eps, "ratio_eps")
    if not np.isfinite(structure_floor) or structure_floor < 0:
        raise ValueError("structure_floor must be finite and nonnegative")
    spread = y.var(0).max(0)
    motion = np.mean(np.square(y[:, 1] - y[:, 0]), axis=0)
    stability = np.exp(-(spread + motion) / (2 * eps**2 + 1e-12))
    gradient = np.stack([gaussian_filter(_grad_field(im), grad_sigma) for im in y.mean(0)])
    gain = gradient / (gradient[0] + ratio_eps)
    peak = abs(gain - 1).argmax(0)
    gamma = np.take_along_axis(gain, peak[None], axis=0)[0]
    structure = np.exp(-np.square(np.log(np.clip(gamma, 1e-12, None))) / (2 * delta**2))
    return {"C_stab": stability, "gamma": gamma, "sigma2_K": spread, "sigma2_T": motion,
            "tau2": spread + motion, "g_dev": abs(gamma-1), "kmax": peak.astype(np.int16),
            "structure_valid": gradient[0] > structure_floor, "S": structure,
            "R": stability * structure, "eps": eps}


def render_2d_colormap(
    gamma,
    C_stab,
    fg_mask=None,
    *,
    structure_valid=None,
    delta: float = 0.20,
    C_max: float = 55.0,
    L_min: float = 35.0,
    L_max: float = 92.0,
    H_blue: float = 265.0,
    H_orange: float = 58.0,
    invert_luma: bool = False,
    gamma_clip=(1.0, 99.0),
    cstab_stretch=(2.0, 98.0),
    gamma_log_limits=None,
    cstab_limits=None,
):
    """Render structure as LCh hue/chroma and stability as luminance.

    ``gamma_log_limits`` and ``cstab_limits`` provide shared absolute ranges
    for cross-checkpoint comparison.  If omitted, per-image
    percentile enhancement is retained for visualisation only.
    """
    delta = _positive_scalar(delta, "delta")
    gamma = np.asarray(gamma, dtype=np.float64).copy()
    cs = np.asarray(C_stab, dtype=np.float64).copy()
    if gamma.ndim != 2 or cs.shape != gamma.shape:
        raise ValueError(f"gamma and C_stab must be same-shape 2-D maps; got {gamma.shape} and {cs.shape}")
    gamma[~np.isfinite(gamma)] = 1.0
    cs[~np.isfinite(cs)] = 0.0

    if structure_valid is not None:
        structure_valid = np.asarray(structure_valid, dtype=bool)
        if structure_valid.shape != gamma.shape:
            raise ValueError("structure_valid shape must match gamma")
        gamma[~structure_valid] = 1.0

    lgam = np.log(np.clip(gamma, 1e-12, None))
    if gamma_log_limits is None:
        q0, q1 = _pair(gamma_clip, "gamma_clip", percentile=True)
        lo, hi = np.percentile(lgam, (q0, q1))
        lo, hi = min(float(lo), -1e-9), max(float(hi), 1e-9)
    else:
        lo, hi = _pair(gamma_log_limits, "gamma_log_limits")
    lgam = np.clip(lgam, lo, hi)
    hue = np.where(lgam < 0, H_blue, H_orange)
    chroma = float(C_max) * np.tanh(np.abs(lgam) / delta)

    if cstab_limits is None:
        q0, q1 = _pair(cstab_stretch, "cstab_stretch", percentile=True)
        p0, p1 = np.percentile(cs, (q0, q1))
        if float(p1 - p0) <= 1e-12:
            cs_scaled = np.clip(cs, 0.0, 1.0)
        else:
            cs_scaled = np.clip((cs - p0) / (p1 - p0), 0.0, 1.0)
    else:
        p0, p1 = _pair(cstab_limits, "cstab_limits")
        cs_scaled = np.clip((cs - p0) / (p1 - p0), 0.0, 1.0)
    if invert_luma:
        cs_scaled = 1.0 - cs_scaled
    lightness = float(L_min) + (float(L_max) - float(L_min)) * cs_scaled

    radians = np.deg2rad(hue)
    lab = np.stack(
        [lightness, chroma * np.cos(radians), chroma * np.sin(radians)],
        axis=-1,
    )
    rgb = np.clip(lab2rgb(lab), 0.0, 1.0)
    if fg_mask is not None:
        fg = np.asarray(fg_mask, dtype=bool)
        if fg.shape != gamma.shape:
            raise ValueError("fg_mask shape must match gamma")
        bg = ~fg
        yy, xx = np.mgrid[0 : gamma.shape[0], 0 : gamma.shape[1]]
        hatch = (((yy + xx) // 4) % 2).astype(np.float64)
        grey = 0.60 + 0.07 * hatch
        for channel in range(3):
            rgb[..., channel][bg] = grey[bg]
    return rgb
