"""Versioned, structure-aware *display* of an unchanged bivariate Map.

The noise-reference-normalized contrast is not a calibrated significance test,
especially for correlated empirical noise or denoiser outputs. The display
support must never be fed into scientific Map reduction or checkpoint gates.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d, label

PROFILE = 'structure_support_display_v1_1px'
CONFIG = {
    'structure_scales_px': [1., 2., 4.],
    'background_scale_px': 16.,
    'gaussian_truncate': 4.,
    'hysteresis_low': 2.,
    'hysteresis_high': 4.5,
    'dilation_radius_px': 1,
    'minimum_component_area': 0,
}


def kernel_norm(s, background=16., truncate=4.):
    radius = int(truncate * background + .5)
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1
    narrow = gaussian_filter1d(impulse, s, truncate=truncate, mode='constant')
    broad = gaussian_filter1d(impulse, background, truncate=truncate, mode='constant')
    return float(np.linalg.norm(np.outer(narrow, narrow) - np.outer(broad, broad)))


def detect_support(image, sigma_native):
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2 or not np.isfinite(im).all() or not np.isfinite(sigma_native) or sigma_native <= 0:
        raise ValueError('Display support requires a finite 2D image and positive native noise reference')
    broad = gaussian_filter(im, CONFIG['background_scale_px'],
                            truncate=CONFIG['gaussian_truncate'], mode='reflect')
    score = np.full(im.shape, -np.inf)
    for s in CONFIG['structure_scales_px']:
        smooth = gaussian_filter(im, s, truncate=CONFIG['gaussian_truncate'], mode='reflect')
        contrast = (smooth - broad) / (sigma_native * kernel_norm(s))
        np.maximum(score, contrast, out=score)
    regions, n = label(score > CONFIG['hysteresis_low'], structure=np.ones((3, 3), bool))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(regions[score > CONFIG['hysteresis_high']])] = True
    keep[0] = False
    return keep[regions]


def render_support_display(query, trajectories, sigma_native, chart, original_rgb, check_stop=None):
    """Return display arrays/provenance; never modify input or infer a new image.

    query and trajectories use the same native intensity units. D1 gray uses
    the frozen inference chart, not a refitted per-state percentile window.
    """
    q, y = np.asarray(query), np.asarray(trajectories)
    rgb = np.asarray(original_rgb)
    if q.ndim != 2 or y.ndim != 4 or y.shape[2:] != q.shape or min(y.shape[:2]) < 1:
        raise ValueError('Expected full-plane query YX and matching KTYX trajectory')
    if rgb.dtype != np.uint8 or rgb.shape != q.shape + (3,):
        raise ValueError('Require byte-exact original Map RGB')
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite Loop values cannot be hidden by display masking')
    if not all(np.array_equal(y[k, 0], y[0, 0]) for k in range(1, y.shape[0])):
        raise ValueError('This display protocol requires the same D1 across streams')
    offset, scale = float(chart['offset']), float(chart['scale'])
    if not np.isfinite([offset, scale]).all() or scale <= 0:
        raise ValueError('Require a finite fixed model chart')
    if check_stop is not None:
        check_stop()
    raw_core = detect_support(q, sigma_native)
    loop_core = np.zeros(q.shape, bool)
    indices = [(0, 0)] + [(k, t) for k in range(y.shape[0]) for t in range(1, y.shape[1])]
    counts = []
    for k, t in indices:
        if check_stop is not None:
            check_stop()
        state_core = detect_support(y[k, t], sigma_native)
        loop_core |= state_core
        counts.append({'k': k, 't_zero_based': t, 'core_pixels': int(state_core.sum())})
    core = raw_core | loop_core
    # Euclidean disk of radius 1 on the square pixel lattice, not a 3x3 square.
    disk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    support = binary_dilation(core, structure=disk)
    gray = np.rint(255 * np.clip((y[0, 0].astype(float) - offset) / scale, 0, 1)).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=-1)
    rgba = np.dstack((rgb, support.astype(np.uint8) * 255))
    overlay = np.where(support[..., None], rgb, base)
    assert not np.any(core & ~support)
    return dict(rgb=overlay, rgba=rgba, d1_gray_rgb=base, support=support,
                core=core, raw_core=raw_core, loop_core=loop_core,
                metadata=dict(profile=PROFILE, parameters=dict(CONFIG),
                    display_only=True, numerical_Map_and_gates_unchanged=True,
                    noise_reference_native=float(sigma_native),
                    noise_reference_source='existing noise calibration; never estimated from Loop output',
                    contrast_interpretation='raw noise L2 reference, not calibrated p-values',
                    core_pixels=int(core.sum()), coloured_pixels=int(support.sum()),
                    coloured_fraction=float(support.mean()),
                    loop_only_core_pixels=int((loop_core & ~raw_core).sum()),
                    retained_detected_core_fraction=1.0,
                    evaluated_full_plane_before_any_display_crop=True,
                    state_support_counts=counts,
                    grayscale_chart=dict(offset=offset, scale=scale),
                    original_bivariate_RGB_exact_where_visible=True,
                    limitations=[
                        'Weak, broad or isolated structures may be undetected.',
                        'Grayscale pixels retain D1 but do not encode zero Map response or certified background.',
                        'Reduced padding is not a resolution or denoising improvement.',
                        'A display mask must not be used for scientific statistics or gates.']))
