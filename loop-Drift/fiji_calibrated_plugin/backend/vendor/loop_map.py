"""Spatial LoopSN2N re-measurement maps and their bivariate renderer.

The reducers in this module reuse an existing ``Y`` trajectory with shape
``(K, T, H, W)``.  They do not run another loop and do not require a clean
reference.  The map is a spatial view of finite-horizon re-measurement
dispersion; it is not, by itself, a calibrated prediction-error map.

Versioning is explicit because three historically different reducers were
previously called "legacy":

* :func:`signature_map_e13_v0_from_Y` reproduces the E13 scratch reducer
  (terminal seed variance, all-step residual sum, exponential score).
* :func:`signature_map_2073_v1_from_Y` reproduces commit ``2073ea0``
  (terminal seed mixture interval).
* :func:`signature_map_from_Y` defaults to the current horizon reducer
  (``map_version="horizon_v1"``).  The unvalidated block/interval proposal is
  available only through the explicit ``map_version="candidate_v2"`` preset.

``C_stab`` is a bounded stability score.  ``cstab_mode="interval"`` uses a
zero-mean Gaussian proxy for the heuristic scale ``tau2``; it is not a
posterior probability or a calibrated seed-mixture confidence.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.special import ndtr
from skimage.color import lab2rgb

from .energy import _grad_field


_MAP_VERSIONS = {"horizon_v1", "candidate_v2", "custom"}
_GAIN_MODES = {"perpixel", "block"}
_CSTAB_MODES = {"exp", "interval"}
_SEED_REDUCERS = {"mean_image", "per_seed_ratio"}
_RATIO_STYLES = {"legacy_denominator", "masked_ratio"}


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


def _struct_field(
    image: np.ndarray,
    *,
    mode: str,
    block: int,
    grad_sigma: float,
    numerical_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a local structure field and the scale used for validity.

    ``perpixel`` is the historical smoothed gradient magnitude.  ``block`` is
    the candidate local roughness

        q_B(y) = local_mean(|grad y|) / (local_std(y) + numerical_floor).

    The latter is contrast invariant, but it is not a universal structure
    truth: edge displacement and texture frequency can still change its
    direction.  It therefore remains an experimental map channel.
    """
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2 or not np.all(np.isfinite(im)):
        raise ValueError("_struct_field expects one finite 2-D image")
    if mode not in _GAIN_MODES:
        raise ValueError(f"unknown gain_mode={mode!r}; expected one of {sorted(_GAIN_MODES)}")
    if mode == "perpixel":
        sigma = _positive_scalar(grad_sigma, "grad_sigma")
        field = gaussian_filter(_grad_field(im), sigma)
        return field, field

    if isinstance(block, bool) or int(block) != block or int(block) < 2:
        raise ValueError(f"block must be an integer >=2; got {block!r}")
    B = int(block)
    grad = uniform_filter(_grad_field(im), size=B, mode="reflect")
    mu = uniform_filter(im, size=B, mode="reflect")
    var = uniform_filter(im * im, size=B, mode="reflect") - mu * mu
    local_std = np.sqrt(np.maximum(var, 0.0))
    field = grad / (local_std + float(numerical_floor))
    return field, local_std


def _gain_trajectory(
    Y: np.ndarray,
    h: int,
    *,
    gain_mode: str,
    block: int,
    grad_sigma: float,
    seed_reduce: str,
    ratio_style: str,
    ratio_eps: float,
    structure_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gain after choosing the nonlinear seed aggregation order."""
    if seed_reduce not in _SEED_REDUCERS:
        raise ValueError(f"unknown seed_reduce={seed_reduce!r}; expected one of {sorted(_SEED_REDUCERS)}")
    if ratio_style not in _RATIO_STYLES:
        raise ValueError(f"unknown ratio_style={ratio_style!r}; expected one of {sorted(_RATIO_STYLES)}")
    ratio_eps = _positive_scalar(ratio_eps, "ratio_eps")
    if not np.isfinite(structure_floor) or structure_floor < 0:
        raise ValueError("structure_floor must be finite and non-negative")

    images = Y[:, :h] if seed_reduce == "per_seed_ratio" else Y[:, :h].mean(axis=0, keepdims=True)
    n_series, _, H, W = images.shape
    fields = np.empty((n_series, h, H, W), dtype=np.float64)
    scales0 = np.empty((n_series, H, W), dtype=np.float64)
    for s in range(n_series):
        for t in range(h):
            fields[s, t], scale = _struct_field(
                images[s, t], mode=gain_mode, block=block, grad_sigma=grad_sigma
            )
            if t == 0:
                scales0[s] = scale

    ref = fields[:, 0]
    valid = (scales0 > structure_floor) & (ref > structure_floor)
    if ratio_style == "legacy_denominator":
        # Exact current horizon-v1 behaviour: gain[0] is slightly below one
        # and a flat field maps to zero.  Kept only for numerical regression.
        ratios = fields / (ref[:, None] + ratio_eps)
        gain = ratios.mean(axis=0)
        return gain, valid.any(axis=0)

    ratios = np.ones_like(fields)
    np.divide(fields, ref[:, None], out=ratios, where=valid[:, None])
    ratios[:, 0] = 1.0
    valid_count = valid.sum(axis=0)
    gain = np.ones((h, H, W), dtype=np.float64)
    for t in range(h):
        numerator = np.where(valid, ratios[:, t], 0.0).sum(axis=0)
        np.divide(numerator, valid_count, out=gain[t], where=valid_count > 0)
    gain[0] = 1.0
    return gain, valid_count > 0


def _stability_score(tau2: np.ndarray, eps: float, mode: str) -> np.ndarray:
    if mode not in _CSTAB_MODES:
        raise ValueError(f"unknown cstab_mode={mode!r}; expected one of {sorted(_CSTAB_MODES)}")
    eps = _positive_scalar(eps, "eps")
    tau = np.maximum(np.asarray(tau2, dtype=np.float64), 0.0)
    if mode == "exp":
        return np.exp(-tau / (2.0 * eps**2 + 1e-12))

    # Gaussian-interval-shaped score.  The exact tau=0 limit is one; using a
    # positive variance floor here would incorrectly make an invariant flat
    # trajectory look unstable.
    std = np.sqrt(tau)
    z = np.divide(eps, std, out=np.full_like(std, np.inf), where=std > 0)
    return np.clip(2.0 * ndtr(z) - 1.0, 0.0, 1.0)


def signature_map_from_Y(
    Y,
    *,
    horizon: int = 3,
    delta: float = 0.20,
    eps: float | None = None,
    grad_sigma: float = 2.0,
    map_version: str = "horizon_v1",
    gain_mode: str | None = None,
    cstab_mode: str | None = None,
    seed_reduce: str | None = None,
    ratio_style: str | None = None,
    ratio_eps: float | None = None,
    block: int = 32,
    structure_floor: float = 1e-6,
):
    """Return a horizon-localised spatial signature from ``Y``.

    ``horizon_v1`` reproduces the current baseline by default:
    ``perpixel + exp + mean_image + legacy_denominator``.

    ``candidate_v2`` is opt-in and uses
    ``block + interval + per_seed_ratio + masked_ratio``.  Invalid/flat
    structure is neutralised to ``gamma=1, g_dev=0, S=1``.

    The returned local components do *not* pool exactly to
    :func:`loop_energy.verdict`: P2 is absent, P3 uses
    ``mean_pixel(max_round(.))`` rather than ``max_round(mean_pixel(.))``, and
    nonlinear seed aggregation can differ.  The map is a diagnostic spatial
    view, not a second implementation of the scalar verdict.
    """
    arr = _validate_Y(Y)
    if map_version not in _MAP_VERSIONS:
        raise ValueError(f"unknown map_version={map_version!r}; expected one of {sorted(_MAP_VERSIONS)}")
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 2:
        raise ValueError(f"horizon must be an integer >=2; got {horizon!r}")
    delta = _positive_scalar(delta, "delta")
    h = min(int(horizon), arr.shape[1])

    if map_version == "candidate_v2":
        defaults = dict(
            gain_mode="block",
            cstab_mode="interval",
            seed_reduce="per_seed_ratio",
            ratio_style="masked_ratio",
            ratio_eps=1e-8,
        )
    else:
        defaults = dict(
            gain_mode="perpixel",
            cstab_mode="exp",
            seed_reduce="mean_image",
            ratio_style="legacy_denominator",
            ratio_eps=1e-6,
        )
    gain_mode = defaults["gain_mode"] if gain_mode is None else str(gain_mode)
    cstab_mode = defaults["cstab_mode"] if cstab_mode is None else str(cstab_mode)
    seed_reduce = defaults["seed_reduce"] if seed_reduce is None else str(seed_reduce)
    ratio_style = defaults["ratio_style"] if ratio_style is None else str(ratio_style)
    ratio_eps = defaults["ratio_eps"] if ratio_eps is None else ratio_eps

    var_per_round = arr[:, :h].var(axis=0, ddof=0)
    sigma2_K = var_per_round.max(axis=0)
    sigma2_T = ((arr[:, 1] - arr[:, 0]) ** 2).mean(axis=0)
    tau2 = sigma2_K + sigma2_T
    eps_source = "adaptive_per_map"
    if eps is None:
        eps = float(np.median(np.sqrt(np.maximum(tau2, 0.0))) + 1e-8)
    else:
        eps_source = "external_fixed"
    eps = _positive_scalar(eps, "eps")
    C_stab = _stability_score(tau2, eps, cstab_mode)

    gain, structure_valid = _gain_trajectory(
        arr,
        h,
        gain_mode=gain_mode,
        block=block,
        grad_sigma=grad_sigma,
        seed_reduce=seed_reduce,
        ratio_style=ratio_style,
        ratio_eps=float(ratio_eps),
        structure_floor=float(structure_floor),
    )
    gdev_t = np.abs(gain - 1.0)
    kmax = gdev_t.argmax(axis=0)
    g_dev = np.take_along_axis(gdev_t, kmax[None], axis=0)[0]
    gamma = np.take_along_axis(gain, kmax[None], axis=0)[0]
    if ratio_style == "masked_ratio":
        gamma = np.where(structure_valid, gamma, 1.0)
        g_dev = np.where(structure_valid, g_dev, 0.0)

    S = np.exp(-(np.log(np.clip(gamma, 1e-12, None)) ** 2) / (2.0 * delta**2))
    if ratio_style == "masked_ratio":
        S = np.where(structure_valid, S, 1.0)
    R = C_stab * S
    return {
        "sigma2_K": sigma2_K,
        "sigma2_T": sigma2_T,
        "tau2": tau2,
        "gamma": gamma,
        "g_dev": g_dev,
        "kmax": kmax.astype(np.int16),
        "structure_valid": structure_valid,
        "C_stab": C_stab,
        "S": S,
        "R": R,
        "eps": eps,
        "protocol": {
            "map_version": map_version,
            "horizon": h,
            "gain_mode": gain_mode,
            "cstab_mode": cstab_mode,
            "seed_reduce": seed_reduce,
            "ratio_style": ratio_style,
            "block": int(block),
            "eps_source": eps_source,
        },
    }


def signature_map_e13_v0_from_Y(Y, *, delta: float = 0.20, eps: float | None = None, grad_sigma: float = 2.0):
    """Exactly reproduce the E13 scratch reducer used to create ``E1_*.tif``."""
    arr = _validate_Y(Y)
    delta = _positive_scalar(delta, "delta")
    sigma = _positive_scalar(grad_sigma, "grad_sigma")
    sigma2_K = arr.var(axis=0, ddof=0)[-1]
    sigma2_T = ((arr[:, 1:] - arr[:, :-1]) ** 2).mean(axis=0).sum(axis=0)
    tau2 = sigma2_K + sigma2_T
    eps_source = "adaptive_terminal_seed"
    if eps is None:
        eps = float(np.median(np.sqrt(np.maximum(sigma2_K, 0.0))) + 1e-8)
    else:
        eps_source = "external_fixed"
    eps = _positive_scalar(eps, "eps")
    C_stab = np.exp(-tau2 / (2.0 * eps**2))
    first = gaussian_filter(_grad_field(arr[:, 0].mean(axis=0)), sigma)
    last = gaussian_filter(_grad_field(arr[:, -1].mean(axis=0)), sigma)
    gamma = last / (first + 1e-6)
    g_dev = np.abs(gamma - 1.0)
    S = np.exp(-(np.log(np.clip(gamma, 1e-12, None)) ** 2) / (2.0 * delta**2))
    R = C_stab * S
    return {
        "sigma2_K": sigma2_K,
        "sigma2_T": sigma2_T,
        "tau2": tau2,
        "gamma": gamma,
        "g_dev": g_dev,
        "kmax": np.full(gamma.shape, arr.shape[1] - 1, dtype=np.int16),
        "structure_valid": first > 1e-6,
        "C_stab": C_stab,
        "S": S,
        "R": R,
        "eps": eps,
        "protocol": {"map_version": "e13_v0", "eps_source": eps_source},
    }


def signature_map_2073_v1_from_Y(Y, *, delta: float = 0.20, eps: float | None = None, grad_sigma: float = 2.0):
    """Exactly reproduce the terminal seed-mixture reducer from commit ``2073ea0``."""
    arr = _validate_Y(Y)
    delta = _positive_scalar(delta, "delta")
    sigma = _positive_scalar(grad_sigma, "grad_sigma")
    mu_s = arr[:, -1]
    sigma2_K = mu_s.var(axis=0, ddof=0)
    sigma2_T = ((arr[:, 1:] - arr[:, :-1]) ** 2).mean(axis=0).sum(axis=0)
    tau2 = sigma2_K + sigma2_T
    eps_source = "adaptive_terminal_seed"
    if eps is None:
        eps = float(np.median(np.sqrt(np.maximum(sigma2_K, 0.0))) + 1e-8)
    else:
        eps_source = "external_fixed"
    eps = _positive_scalar(eps, "eps")
    mu_bar = mu_s.mean(axis=0)
    sigT_s = np.sqrt(arr.var(axis=1, ddof=0) + 1e-12)
    lo = (mu_bar[None] - eps - mu_s) / sigT_s
    hi = (mu_bar[None] + eps - mu_s) / sigT_s
    C_stab = (ndtr(hi) - ndtr(lo)).mean(axis=0)
    first = gaussian_filter(_grad_field(arr[:, 0].mean(axis=0)), sigma)
    last = gaussian_filter(_grad_field(mu_bar), sigma)
    gamma = last / (first + 1e-6)
    g_dev = np.abs(gamma - 1.0)
    S = np.exp(-(np.log(np.clip(gamma, 1e-12, None)) ** 2) / (2.0 * delta**2))
    R = C_stab * S
    return {
        "sigma2_K": sigma2_K,
        "sigma2_T": sigma2_T,
        "tau2": tau2,
        "gamma": gamma,
        "g_dev": g_dev,
        "kmax": np.full(gamma.shape, arr.shape[1] - 1, dtype=np.int16),
        "structure_valid": first > 1e-6,
        "C_stab": C_stab,
        "S": S,
        "R": R,
        "eps": eps,
        "protocol": {"map_version": "2073_v1", "eps_source": eps_source},
    }


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
    for cross-checkpoint comparison.  If omitted, the historical per-image
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


def render_simple(gamma, R, fg_mask=None, *, s_clip: float = 95.0):
    """Render signed diagnostic fusion on a blue-pale-orange bar."""
    gamma = np.asarray(gamma, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if gamma.ndim != 2 or R.shape != gamma.shape:
        raise ValueError("gamma and R must be same-shape 2-D maps")
    finite = np.isfinite(gamma) & np.isfinite(R)
    if not finite.any():
        raise ValueError("gamma and R contain no finite pixels")
    s = np.sign(np.where(finite, gamma, 1.0) - 1.0) * (1.0 - np.clip(np.where(finite, R, 0.0), 0.0, 1.0))
    if not 0 < float(s_clip) <= 100:
        raise ValueError("s_clip must lie in (0,100]")
    scale = float(np.percentile(np.abs(s[finite]), float(s_clip)))
    t = np.zeros_like(s) if scale <= 1e-12 else np.clip(s / scale, -1.0, 1.0)
    blue = np.array([0.37, 0.53, 0.70])
    pale = np.array([0.93, 0.93, 0.90])
    orange = np.array([0.85, 0.51, 0.25])
    rgb = np.where(
        t[..., None] < 0,
        (1.0 + t[..., None]) * pale + (-t[..., None]) * blue,
        (1.0 - t[..., None]) * pale + t[..., None] * orange,
    )
    if fg_mask is not None:
        fg = np.asarray(fg_mask, dtype=bool)
        if fg.shape != gamma.shape:
            raise ValueError("fg_mask shape must match gamma")
        rgb[~fg] = 0.62
    return np.clip(rgb, 0.0, 1.0)
