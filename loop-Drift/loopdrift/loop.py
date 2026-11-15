"""Feedback trajectories and scalar LoopSN2N measurements."""

from __future__ import annotations

import numpy as np

from . import energy
from .loop_map import (
    render_2d_colormap,
    signature_map_from_Y,
)

def feedback(observation, denoise, perturb, *, streams=5, states=3, anchored=False):
    """Return [stream,state,y,x] in a fixed image coordinate.

    denoise(image) returns a single full image; perturb(state, stream, step)
    returns a *complete* perturbed input in the denoiser's coordinate. Supply
    the acquisition or simulation protocol for the images at hand. To compare
    probe modes, key the noise by (stream, step) in both calls. D1 is shared;
    anchored probes always start from D1, recursive probes from the last state.
    """
    def plane(image, shape=None):
        a = np.asarray(image)
        if (a.ndim != 2 or min(a.shape) < 2 or a.dtype.kind not in "uif"
                or not np.isfinite(a).all() or (shape is not None and a.shape != shape)):
            raise ValueError("expected a finite 2-D image in the original coordinate")
        return a.copy()

    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (streams, states)):
        raise ValueError("streams and states must be positive integers")
    image = plane(observation)
    d1 = plane(denoise(image), image.shape)
    result = []
    for k in range(streams):
        row = [d1]
        for step in range(1, states):
            source = d1 if anchored else row[-1]
            noisy = plane(perturb(source.copy(), k, step), image.shape)
            row.append(plane(denoise(noisy), image.shape))
        result.append(row)
    return np.asarray(result)


def loop_signature(trajectory, *, gain_mode="global_mean", gain_contrast_norm="none",
                   fg_quantile=0.7):
    """Reduce finite denoised iterates [seeds, rounds, height, width].

    The returned T/K metadata makes the signature usable by the shared verdict
    engine. These are response diagnostics, not a fidelity certificate. A single
    seed is allowed for exploration, but cannot estimate cross-seed dispersion.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 4 or min(values.shape) < 1:
        raise ValueError("trajectory must be a nonempty [K,T,H,W] array")
    if values.shape[1] < 2 or min(values.shape[2:]) < 2:
        raise ValueError("trajectory requires T >= 2 and H,W >= 2")
    if not np.isfinite(values).all():
        raise ValueError("trajectory must contain only finite values")
    if gain_mode not in {"global_mean", "foreground_median"}:
        raise ValueError("unknown gain_mode")
    if gain_contrast_norm not in {"none", "std", "mad"}:
        raise ValueError("unknown gain_contrast_norm")
    if not np.isfinite(fg_quantile) or not 0 <= fg_quantile <= 1:
        raise ValueError("fg_quantile must lie in [0,1]")
    result = energy.signature_from_Y(values, gain_mode=gain_mode,
                                     gain_contrast_norm=gain_contrast_norm,
                                     fg_quantile=fg_quantile)
    return dict(result, K=values.shape[0], T=values.shape[1])


def evaluate_loop(trajectory, *, gain_mode="global_mean", gain_contrast_norm="none",
                  fg_quantile=0.7, horizon=3, gain_band_delta=0.10,
                  var_cap=5e-5, near_inv_cap=2e-3, diss_tol=1.10,
                  gain_defer_q=0.05):
    """Return one-trajectory signature, P1-P4 checks and a diagnostic decision.

    Evaluate precomputed denoised states in the coordinate used to set the
    thresholds. For two-state or single-stream probes, use loop_signature:
    a complete P2 or P3 decision, respectively, is unavailable.
    Defaults are the research thresholds, not the Fiji example decision profile.
    """
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 3:
        raise ValueError("the four checks require a horizon of at least three")
    for name, value in {"gain_band_delta": gain_band_delta, "var_cap": var_cap,
                        "near_inv_cap": near_inv_cap, "diss_tol": diss_tol}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(gain_defer_q) or not 0 <= gain_defer_q <= gain_band_delta:
        raise ValueError("gain_defer_q must lie in [0,gain_band_delta]")
    values = np.asarray(trajectory)
    if values.ndim != 4 or values.shape[0] < 2 or values.shape[1] < horizon:
        raise ValueError("a decision requires two streams and all requested states")
    signature = loop_signature(values[:, :int(horizon)], gain_mode=gain_mode,
                               gain_contrast_norm=gain_contrast_norm,
                               fg_quantile=fg_quantile)
    checks = energy.verdict(signature, horizon=horizon, gain_band_delta=gain_band_delta,
                            var_cap=var_cap, near_inv_cap=near_inv_cap, diss_tol=diss_tol)
    decision, reason, _ = energy.decide_from_energy(checks, gain_defer_q=gain_defer_q)
    return {"signature": signature, "checks": checks, "decision": decision,
            "reason": reason, "scope": "single_trajectory_diagnostic_only"}

