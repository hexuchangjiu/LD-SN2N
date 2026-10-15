"""E69 empirical, same-scene additive perturbations from static repeat frames.

This is not a camera calibration, a Poisson-law identification, or permission
to deploy a reliability classifier. Pair differences identify second-order
noise only conditionally on a static scene and suitable repeat dependence.
The finite bank retains observed spatial dependence, not arbitrary signal-
dependent noise or the full observation distribution (notably its skewness).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ALGORITHM_VERSION = "e69_same_scene_pair_bank_v1"
QUALIFIED = "EMPIRICAL_SAME_SCENE_QUALIFIED"
NOT_QUALIFIED = "NOT_QUALIFIED"
# Frozen engineering screens for this protocol, not confidence bounds or
# theoretical noise-admission constants. Do not tune them to admit a data set.
THRESHOLDS = {
    "minimum_pairs_per_split": 8,
    "heldout_variance_ratio_min": 0.5,
    "heldout_variance_ratio_max": 2.0,
    "neighbor_correlation_absolute_difference_max": 0.20,
    "frame_mean_p95_p05_over_sigma_max": 2.0,
    "first_last_quarter_mean_change_over_sigma_max": 0.5,
    "first_last_quarter_mean_image_change_ratio_max": 4.0,
    "pair_mean_rms_over_expected_sigma_max": 3.0,
    "first_last_pair_variance_ratio_min": 0.5,
    "first_last_pair_variance_ratio_max": 2.0,
    "pixel_tail_excess_kurtosis_max": 20.0,
    "lag_variogram_ratio_min": 0.75,
    "lag_variogram_ratio_max": 1.3333333333333333,
    "signal_bound_sigma_margin": 8.0,
}


class NoiseCalibrationError(ValueError):
    """Invalid repeat contract, malformed data, or inadmissible bank use."""


def _hash_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(list(a.shape)).encode("ascii"))
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _array(value: Any, ndim: int, label: str) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "fiu" or original.ndim != ndim:
        raise NoiseCalibrationError(f"{label} must be a real numeric {ndim}-D array")
    a = np.asarray(original, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise NoiseCalibrationError(f"{label} must be nonempty and finite")
    return a


def _neighbor_correlation(d: np.ndarray, axis: int) -> float | None:
    # Per-pixel centering across repeats preserves a common spatial mode in
    # each draw; per-frame spatial centering would incorrectly remove it.
    z = d - d.mean(axis=0, keepdims=True)
    a, b = (z[:, :, 1:], z[:, :, :-1]) if axis == 2 else (z[:, 1:, :], z[:, :-1, :])
    denominator = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
    return float(np.mean(a * b) / denominator) if denominator > 0 else None


def _binned_variance(mean_image: np.ndarray, fit_var: np.ndarray,
                     heldout_var: np.ndarray) -> dict[str, Any]:
    # Bins are fixed from fitting data only; validation intensities do not
    # select support. A constant mean field does not require a PTC slope.
    x = mean_image.ravel()
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 13)))
    rows = []
    if len(edges) > 1:
        labels = np.searchsorted(edges[1:-1], x, side="right")
        for i in range(len(edges) - 1):
            mask = labels == i
            if int(mask.sum()) < 16:
                continue
            vf = float(fit_var.ravel()[mask].mean())
            vh = float(heldout_var.ravel()[mask].mean())
            rows.append({"pixels": int(mask.sum()), "mean_native": float(x[mask].mean()),
                         "fit_variance_native2": vf, "heldout_variance_native2": vh,
                         "heldout_over_fit": vh / vf if vf > 0 else None})
    line = None
    if len(rows) >= 3:
        means = np.array([r["mean_native"] for r in rows])
        variances = np.array([r["fit_variance_native2"] for r in rows])
        if np.ptp(means) > np.finfo(float).eps * max(1.0, float(np.max(np.abs(means)))):
            design = np.column_stack((means - means.mean(), np.ones(len(means))))
            slope, centered_intercept = np.linalg.lstsq(design, variances, rcond=None)[0]
            intercept = centered_intercept - slope * means.mean()
            pred = slope * means + intercept
            total = float(np.sum((variances - variances.mean()) ** 2))
            line = {"slope_native": float(slope), "intercept_native2": float(intercept),
                    "r_squared": float(1 - np.sum((variances - pred) ** 2) / total) if total > 0 else None,
                    "interpretation": "descriptive variance-intensity fit only; no gain/photon or conditional scaling inference"}
    return {"bins": rows, "linear_fit": line, "selection": "fit-pair mean-image quantiles",
            "conditional_scaling_enabled": False}


def _temporal_variograms(a: np.ndarray, used: list[int]) -> list[dict[str, Any]]:
    """Native-index lag differences, without treating the query gap as lag 1.

    Overlapping lag differences are descriptive, not independent samples.
    Mean-change energy is retained rather than silently detrended away.
    Accumulate framewise to avoid allocating another full FYX stack.
    """
    included = set(used)
    rows = []
    for lag in (1, 2, 4, 8):
        pairs = [(i, i + lag) for i in used if i + lag in included]
        sum_delta = np.zeros(a.shape[1:], dtype=np.float64)
        square = 0.0
        for i, j in pairs:
            delta = a[j] - a[i]
            sum_delta += delta
            square += float(np.mean(delta * delta)) / 2.0
        mean_square = square / len(pairs)
        mean_term = float(np.mean((sum_delta / len(pairs)) ** 2) / 2.0)
        rows.append({"lag_original_frames": lag, "number_of_overlapping_differences": len(pairs),
                     "half_mean_square_difference_native2": mean_square,
                     "half_mean_delta_square_native2": mean_term,
                     "half_centered_difference_variance_ddof0_native2": max(0.0, mean_square - mean_term)})
    baseline = rows[0]["half_mean_square_difference_native2"]
    for row in rows:
        row["ratio_to_lag1"] = row["half_mean_square_difference_native2"] / baseline if baseline > 0 else None
    return rows


def calibrate_stack(stack: Any, query_index: int = 0, domain: str = "static_repeats",
                    output_dir: str | Path | None = None) -> tuple[dict[str, Any], np.ndarray]:
    """Return a JSON-native report and fit-only float32[M,H,W] native-unit bank.

    `static_repeats` is a caller declaration, not an inference from FYX shape.
    Z, phase, live dynamics and multichannel series must not use this label.
    Original pairs (0,1),(2,3),... are fixed before exclusion. The entire pair
    containing query_index is removed; an unpaired last frame is never used.
    Alternating remaining pairs form fit and heldout sets, without shared
    frames. A NOT_QUALIFIED report may be inspected but cannot be sampled.
    """
    if domain != "static_repeats":
        raise NoiseCalibrationError("Only explicitly declared static_repeats FYX is supported; not Z/phase/live/multichannel")
    a = _array(stack, 3, "stack FYX")
    t, h, w = a.shape
    if min(h, w) < 4:
        raise NoiseCalibrationError("Spatial shape must be at least 4 by 4")
    if isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer)) or not 0 <= query_index < t:
        raise NoiseCalibrationError("query_index must be an in-bounds integer")
    pairs = [(i, i + 1) for i in range(0, t - 1, 2) if query_index not in (i, i + 1)]
    fit_pairs, heldout_pairs = pairs[::2], pairs[1::2]
    if min(len(fit_pairs), len(heldout_pairs)) < THRESHOLDS["minimum_pairs_per_split"]:
        raise NoiseCalibrationError("Need at least 8 disjoint fit and 8 heldout pairs after query-pair exclusion (normally at least 34 frames)")
    fit_indices = [i for pair in fit_pairs for i in pair]
    heldout_indices = [i for pair in heldout_pairs for i in pair]
    used = sorted(fit_indices + heldout_indices)
    with np.errstate(over="raise", invalid="raise"):
        try:
            fit = np.stack([(a[i] - a[j]) / np.sqrt(2.0) for i, j in fit_pairs])
            heldout = np.stack([(a[i] - a[j]) / np.sqrt(2.0) for i, j in heldout_pairs])
            fit_var, heldout_var = fit.var(axis=0, ddof=1), heldout.var(axis=0, ddof=1)
            variance = float(fit_var.mean())
        except FloatingPointError as exc:
            raise NoiseCalibrationError("Native values overflow supported statistics") from exc
    if not np.isfinite(variance) or variance <= 0:
        raise NoiseCalibrationError("No positive finite repeat-noise variance; constant/identical repeats cannot form a bank")
    sigma = float(np.sqrt(variance))
    nfit, nheld = len(fit), len(heldout)
    centered = (fit - fit.mean(axis=0, keepdims=True)) * np.sqrt(nfit / (nfit - 1))
    if np.max(np.abs(centered)) > np.finfo(np.float32).max:
        raise NoiseCalibrationError("Native perturbations exceed float32 bank range")
    bank = centered.astype(np.float32)
    bank_var = float(np.mean(bank.astype(np.float64) ** 2))
    if not np.isfinite(bank_var) or bank_var <= 0:
        raise NoiseCalibrationError("Native perturbations cannot be represented by the float32 bank")
    mean_image = a[fit_indices].mean(axis=0)
    intensity = _binned_variance(mean_image, fit_var, heldout_var)
    ratio = float(heldout_var.mean() / variance)
    neighbor = {}
    reasons: list[str] = []

    def check(ok: bool, reason: str) -> None:
        if not ok:
            reasons.append(reason)

    check(THRESHOLDS["heldout_variance_ratio_min"] <= ratio <= THRESHOLDS["heldout_variance_ratio_max"],
          "heldout_variance_mismatch")
    for axis, number in (("x", 2), ("y", 1)):
        cf, ch = _neighbor_correlation(fit, number), _neighbor_correlation(heldout, number)
        delta = abs(cf - ch) if cf is not None and ch is not None else None
        neighbor[axis] = {"fit": cf, "heldout": ch, "absolute_difference": delta}
        check(delta is not None and delta <= THRESHOLDS["neighbor_correlation_absolute_difference_max"],
              f"heldout_spatial_correlation_mismatch_{axis}")
    for i, row in enumerate(intensity["bins"]):
        r = row["heldout_over_fit"]
        check(r is not None and THRESHOLDS["heldout_variance_ratio_min"] <= r <= THRESHOLDS["heldout_variance_ratio_max"],
              f"heldout_intensity_bin_variance_mismatch_{i}")

    means = np.array([a[i].mean() for i in used])
    quarter = max(2, len(used) // 4)
    early, late = a[used[:quarter]].mean(axis=0), a[used[-quarter:]].mean(axis=0)
    mean_span = float((np.quantile(means, 0.95) - np.quantile(means, 0.05)) / sigma)
    mean_change = float((late.mean() - early.mean()) / sigma)
    image_change = float(np.mean((late - early) ** 2) / (variance * 2.0 / quarter))
    check(mean_span <= THRESHOLDS["frame_mean_p95_p05_over_sigma_max"], "frame_mean_nonstationarity")
    check(abs(mean_change) <= THRESHOLDS["first_last_quarter_mean_change_over_sigma_max"], "brightness_drift_or_bleaching")
    check(image_change <= THRESHOLDS["first_last_quarter_mean_image_change_ratio_max"], "spatial_drift_dynamics_or_temporal_dependence")
    # Test difference mean fields, retaining their spatial pattern. Under iid
    # zero-mean repeats, RMS has scale sigma/sqrt(number of pairs), not zero.
    fit_bias = float(np.sqrt(np.mean(fit.mean(axis=0) ** 2)) / (sigma / np.sqrt(nfit)))
    heldout_bias = float(np.sqrt(np.mean(heldout.mean(axis=0) ** 2)) / (sigma / np.sqrt(nheld)))
    check(max(fit_bias, heldout_bias) <= THRESHOLDS["pair_mean_rms_over_expected_sigma_max"], "pair_mean_field_bias")
    chronological = np.stack([(a[i] - a[j]) / np.sqrt(2.0) for i, j in pairs])
    half = len(chronological) // 2
    first_variance = float(chronological[:half].var(axis=0, ddof=1).mean())
    last_variance = float(chronological[half:].var(axis=0, ddof=1).mean())
    time_ratio = last_variance / first_variance if first_variance > 0 else None
    check(time_ratio is not None and THRESHOLDS["first_last_pair_variance_ratio_min"] <= time_ratio <= THRESHOLDS["first_last_pair_variance_ratio_max"],
          "noise_variance_nonstationarity")
    variograms = _temporal_variograms(a, used)
    for row in variograms[1:]:
        lag_ratio = row["ratio_to_lag1"]
        check(lag_ratio is not None and THRESHOLDS["lag_variogram_ratio_min"] <= lag_ratio <= THRESHOLDS["lag_variogram_ratio_max"],
              f"lag_{row['lag_original_frames']}_dependence_or_mean_change")
    # Standardize each location so ordinary signal-dependent variance is not
    # mistaken for contamination. This is an engineering tail screen, not a
    # Gaussian-law requirement or a kurtosis hypothesis test.
    pooled_residual = chronological - chronological.mean(axis=0, keepdims=True)
    local_var = np.mean(pooled_residual ** 2, axis=0)
    valid = local_var > variance * 1e-8
    standardized = pooled_residual[:, valid] / np.sqrt(local_var[valid])
    excess = float(np.mean(standardized ** 4) - 3.0)
    check(excess <= THRESHOLDS["pixel_tail_excess_kurtosis_max"], "extreme_temporal_contamination")
    low, high = float(mean_image.min()), float(mean_image.max())
    margin = THRESHOLDS["signal_bound_sigma_margin"] * sigma
    status = QUALIFIED if not reasons else NOT_QUALIFIED
    report: dict[str, Any] = {
        "schema": "loopsn2n_empirical_noise_calibration_v1",
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "domain": domain, "array_axes": "FYX", "native_units": "input stored intensity units; no photon/electron conversion",
        "shape": [int(v) for v in a.shape], "query_index": int(query_index),
        "source": {"array_sha256": _hash_array(a), "fit_pairs": [list(p) for p in fit_pairs],
                   "heldout_pairs": [list(p) for p in heldout_pairs],
                   "fit_frame_indices": fit_indices, "heldout_frame_indices": heldout_indices,
                   "excluded_frame_indices": [i for i in range(t) if i not in used],
                   "pair_formula": "(X[2j] - X[2j+1]) / sqrt(2)",
                   "split": "alternating retained original pairs; no shared frames; query entire pair excluded"},
        "qualification": {"status": status, "reasons": reasons, "qualified_for_empirical_same_scene_sampling": not reasons,
                          "deployment_qualified": False, "physical_noise_law_qualified": False,
                          "threshold_basis": "frozen conservative engineering screens, not theoretical constants or calibrated error rates",
                          "thresholds": dict(THRESHOLDS)},
        "measured_noise_sigma_native": sigma,
        "measured_noise_variance_native2": variance,
        "heldout_validation": {"variance_ratio": ratio, "heldout_sigma_native": float(np.sqrt(heldout_var.mean())),
                               "neighbor_correlation": neighbor, "fit_pair_mean_field_rms_over_expected_sigma": fit_bias,
                               "heldout_pair_mean_field_rms_over_expected_sigma": heldout_bias,
                               "variance_intensity": intensity, "heldout_used_for_parameter_fit": False},
        "stationarity": {"frame_means_native": means.tolist(), "frame_indices": used,
                         "frame_mean_p95_p05_over_sigma": mean_span,
                         "first_last_quarter_mean_change_over_sigma": mean_change,
                         "first_last_quarter_mean_image_change_ratio": image_change,
                         "first_last_pair_variance_ratio": time_ratio,
                         "temporal_variograms": variograms,
                         "temporal_interpretation": "Flat lag statistics are a bounded screen, not proof of conditional independence; overlapping differences are not independent observations.",
                         "pixel_standardized_excess_kurtosis": excess},
        "bank": {"sha256": _hash_array(bank), "shape": list(bank.shape), "dtype": "float32",
                 "mean_centering": "per pixel across fit pairs only; each spatial draw is not spatially centered",
                 "finite_bank_variance_factor": float(np.sqrt(nfit / (nfit - 1))),
                 "second_moment_native2": bank_var, "sampling": "uniform with replacement; finite measured bank",
                 "conditional_scaling_enabled": False, "random_max_normalization": False, "clipping": False},
        "applicability": {"same_scene_same_FOV_same_acquisition_settings_only": True,
                          "observed_mean_intensity_native": [low, high],
                          "signal_sampling_bounds_native": [low - margin, high + margin],
                          "observed_perturbation_native": [float(bank.min()), float(bank.max())],
                          "cross_scene_or_intensity_extrapolation_validated": False},
        "limitations": ["FYX and static_repeats are explicit caller declarations; shape cannot distinguish time, Z, phase or channel.",
                        "Disjoint pairs and heldout frames are not a certificate of temporal independence or independent acquisitions.",
                        "Pair differencing requires a common static conditional mean and appropriate temporal covariance.",
                        "The finite additive bank preserves measured same-scene spatial structure, not a full physical law, Poisson skewness, arbitrary signal dependence or unseen tails.",
                        "Heldout checks evaluate this acquisition only; no deployment, gate-error, biological or camera-transfer qualification.",
                        "No conditional variance scaling, whitening, clipping, per-frame normalization, gain or photon inference is performed."],
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bank_path, report_path = out / "noise_bank.npy", out / "noise_calibration_report.json"
        if bank_path.exists() or report_path.exists():
            raise NoiseCalibrationError("Refusing to overwrite existing calibration artifacts")
        report["artifacts"] = {"bank": str(bank_path.resolve()), "report": str(report_path.resolve())}
        with bank_path.open("xb") as f:
            np.save(f, bank, allow_pickle=False)
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    # Also enforce JSON safety when the caller has not asked for files.
    json.dumps(report, allow_nan=False)
    return report, bank


def sample_noise(signal_native: Any, rng: np.random.Generator, report: dict[str, Any],
                 bank: np.ndarray) -> np.ndarray:
    """Add one measured native-unit perturbation, with no hidden processing.

    Reusing a finite bank produces independent *indices* conditional on the
    bank, not new measured observations or proof of physical independence.
    Caller must use a fixed recorded affine map to/from a network coordinate.
    """
    if not isinstance(rng, np.random.Generator):
        raise NoiseCalibrationError("rng must be an explicit numpy.random.Generator")
    q = report.get("qualification", {})
    if report.get("algorithm_version") != ALGORITHM_VERSION or q.get("status") != QUALIFIED or q.get("qualified_for_empirical_same_scene_sampling") is not True:
        raise NoiseCalibrationError("A qualified same-scene empirical calibration report is required; inspect qualification reasons")
    signal = _array(signal_native, 2, "signal_native")
    if not isinstance(bank, np.ndarray) or bank.dtype != np.float32 or bank.ndim != 3 or tuple(bank.shape[1:]) != signal.shape or not np.isfinite(bank).all():
        raise NoiseCalibrationError("Expected the finite float32[M,H,W] bank in the same native FOV")
    if _hash_array(bank) != report.get("bank", {}).get("sha256"):
        raise NoiseCalibrationError("Empirical bank identity mismatch")
    lo, hi = report["applicability"]["signal_sampling_bounds_native"]
    if float(signal.min()) < lo or float(signal.max()) > hi:
        raise NoiseCalibrationError("Signal exceeds the declared same-scene native intensity range; conditional extrapolation is not qualified")
    return signal + bank[int(rng.integers(bank.shape[0]))].astype(np.float64)
