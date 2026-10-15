"""Verbatim pure functions; source identities in source_identity.json."""
import numpy as np
ENERGY_GAIN_DEFER_Q = 0.05

def _grad_mag(y: np.ndarray) -> float:
    gy, gx = np.gradient(y.astype(np.float64))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))

def _grad_field(y: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(y.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)

def _contrast_scale(y: np.ndarray, mode: str = "std", eps: float = 1e-8) -> float:
    """Per-frame contrast scale for a contrast-INVARIANT gain (energy.md §8 amendment).

    Dividing the mean gradient by this makes gain_k UNCHANGED under an affine photometric transform
    y -> a*y + b (a benign contrast/brightness amplification, structure intact) while staying sensitive
    to a real HF spectral change (over-sharpen raises grad more than the contrast scale; over-smooth
    lowers it). 'std' = RMS contrast (recommended). 'mad' is DEGENERATE when >50% of pixels are a flat
    background (MAD->0), so it is NOT recommended for low-SNR microscopy. 'none' -> 1.0 (legacy raw gain)."""
    if mode == "std":
        return max(float(np.std(y)), eps)
    if mode == "mad":
        med = float(np.median(y))
        return max(float(1.4826 * np.median(np.abs(y - med))), eps)
    return 1.0

def _gain_foreground_median(Y: np.ndarray, fg_quantile: float) -> np.ndarray:
    """energy.md §8: HF-gain as the MEDIAN over a FOREGROUND/edge region Ω_e of the per-pixel
    ratio |∇y_t|/|∇y_1|, robust to the flat background that dilutes a global mean in low-SNR
    fields (rFRC-style background suppression, Zhao 2023). Foreground = pixels of y_1 above the
    `fg_quantile` intensity quantile; falls back to the whole frame if too few foreground pixels.
    Y: (K,T,H,W) -> gain (T,) (mean over seeds of the per-seed foreground-median ratio)."""
    K, T = Y.shape[0], Y.shape[1]
    out = np.ones(T, dtype=np.float64)
    for t in range(T):
        ratios = []
        for k in range(K):
            y1 = Y[k, 0]
            thr = float(np.quantile(y1, fg_quantile))
            m = y1 > thr
            if int(m.sum()) < 16:
                m = np.ones_like(m, dtype=bool)
            r = (_grad_field(Y[k, t])[m] + 1e-12) / (_grad_field(y1)[m] + 1e-12)
            ratios.append(float(np.median(r)))
        out[t] = float(np.mean(ratios))
    return out

def signature_from_Y(Y: np.ndarray, *, gain_mode: str = "global_mean", fg_quantile: float = 0.7,
                     gain_contrast_norm: str = "none") -> dict:
    """Pure energy-signature reducer (factored so extract_seq_features / a cache reader can reuse it).
    Y: (K,T,H,W) denoised iterates y_t over K seeds. Returns the per-round signature arrays:
      V_res (T-1,) = E_seed‖y_{k+1}−y_k‖²   (env-internal residual = Φ-residual along the loop, §3)
      reg   (T-1,) = E_seed‖y_{k+1}−y_k‖    (KM iterate-change, §2/§4)
      var   (T,)   = mean_pixel Var_seed(y_t)(cross-seed stochastic margin, §10)
      gain  (T,)   = HF-gain ratio to round 1 (global-mean OR foreground-median, §8)."""
    Y = np.asarray(Y, dtype=np.float64)
    K, T = Y.shape[0], Y.shape[1]
    diff2 = ((Y[:, 1:] - Y[:, :-1]) ** 2).mean(axis=(2, 3))   # (K, T-1)
    Vres = diff2.mean(axis=0)
    reg = np.sqrt(diff2).mean(axis=0)
    var = Y.var(axis=0, ddof=0).mean(axis=(1, 2))             # (T,)
    if gain_mode == "foreground_median":
        gain = _gain_foreground_median(Y, fg_quantile)        # (T,)
    else:
        def _gm(y):
            gm = _grad_mag(y)
            return gm / _contrast_scale(y, gain_contrast_norm) if gain_contrast_norm != "none" else gm
        g = np.array([[_gm(Y[k, t]) for t in range(T)] for k in range(K)])  # (K,T)
        gain = (g / np.clip(g[:, :1], 1e-12, None)).mean(axis=0)
    return {"V_res": Vres, "reg": reg, "var": var, "gain": gain}

def verdict(sig, *, horizon=3, gain_band_delta=0.10, var_cap=5e-5, near_inv_cap=2e-3, diss_tol=1.10):
    """Theory P1–P4 verdict read at the finite DEPLOYMENT horizon (A1: a reliable denoiser is near-invariant
    after ONE pass and stays so over the short loop — do NOT over-iterate; the terminal of a long loop is
    contaminated by cumulative smoothing even for good models). The discriminators are the **first-step
    residual** ‖y₂−y₁‖² and the **early-horizon HF-gain stability** max_{k≤h}|gain_k−1| (catches over-sharpen
    spikes AND over-smooth collapse); the terminal gain is kept only as a diagnostic."""
    V = np.asarray(sig["V_res"], float); var = np.asarray(sig["var"], float); gain = np.asarray(sig["gain"], float)
    T = int(sig["T"]); h = max(2, min(int(horizon), T))
    first_residual = float(V[0])                                       # ‖y₂−y₁‖²: near-invariance after one pass (A1)
    var_margin = float(np.max(var[:h]))                                # stochastic-stability margin over the horizon
    gdev = np.abs(gain[:h] - 1.0); gain_instab = float(np.max(gdev)); kmax = int(np.argmax(gdev))
    struct = "preserve" if gain_instab <= gain_band_delta else ("over-sharpen" if gain[:h][kmax] > 1.0 else "over-smooth")
    p1 = bool(first_residual < near_inv_cap)                           # stationary after one pass
    p2 = bool(np.all(V[1:] <= V[:-1] * diss_tol)) if V.size >= 2 else True   # dissipative (no residual rebound)
    p3 = bool(var_margin < var_cap)                                    # stochastically stable over the horizon
    p4 = bool(gain_instab <= gain_band_delta)                          # structure-preserving over the horizon
    rel = -(first_residual + var_margin + gain_instab)                 # RAW theory-centred sum (kept for back-compat)
    # COMMENSURABILITY FIX (energy.md §11): the raw sum is gain-dominated (the three terms differ by 2–3 orders
    # of magnitude, so rel ≈ −gain_instability, Spearman(rel,gain)=−1). Normalise each term by its own theory cap
    # so all three contribute on a common O(1) scale before summing; this is the deployable scalar to use for
    # ranking. (Per-term PASS/FAIL P1–P4 below is the magnitude-robust primary read-out either way.)
    rel_norm = -(first_residual / max(near_inv_cap, 1e-12)
                 + var_margin / max(var_cap, 1e-12)
                 + gain_instab / max(gain_band_delta, 1e-12))
    return {"horizon": h, "P1_stationary": p1, "P2_dissipative": p2, "P3_stable": p3, "P4_structure": p4,
            "structure_mode": struct, "first_step_residual": first_residual, "var_margin": var_margin,
            "gain_instability": gain_instab, "gain_T_terminal": float(gain[-1]),
            "energy_reliability": rel, "energy_reliability_norm": rel_norm}
ENERGY_GAIN_DEFER_Q = 0.05

def decide_from_energy(vd, *, gain_defer_q=ENERGY_GAIN_DEFER_Q):
    """PRIMARY deployed decision from the FULL finite-horizon P1–P4 energy signature
    (`loop_energy.verdict`). The STRUCTURE axis (P4) is the decisive failure detector — the
    keystone that residual stability ≠ structure preservation: a checkpoint outside the HF-gain
    band is unreliable in a named direction (over-smooth / over-sharpen) even if its loop is
    perfectly stable. Within the structure band, a confident `reliable` call additionally requires
    the pure stability conditions P1–P3 and a strict inner gain band; otherwise the loop abstains.
    `vd`: a loop_energy.verdict dict. Returns (decision in {reliable, uncertain, unreliable},
    reason, gain_instability)."""
    gi = float(vd.get("gain_instability", float("nan")))
    mode = str(vd.get("structure_mode", "?"))
    if not bool(vd.get("P4_structure", False)):                 # |gain−1| beyond the structure band
        return "unreliable", f"{mode} (HF-structure gain instability {gi:.2f} outside the band)", gi
    reasons = []
    if not bool(vd.get("P1_stationary", True)):
        reasons.append(f"not near-invariant after one pass (first-step residual {float(vd.get('first_step_residual', float('nan'))):.2g})")
    if not bool(vd.get("P2_dissipative", True)):
        reasons.append("residual energy rebounds over the horizon")
    if not bool(vd.get("P3_stable", True)):
        reasons.append(f"high cross-seed variance ({float(vd.get('var_margin', float('nan'))):.2g})")
    if np.isfinite(gi) and gi > gain_defer_q:
        reasons.append(f"near the structure edge (|gain−1|={gi:.2f} > {gain_defer_q:.2f})")
    if not reasons:
        return "reliable", "near-invariant, dissipative, stochastically stable, structure-preserving (P1–P4)", gi
    return "uncertain", "; ".join(reasons), gi
