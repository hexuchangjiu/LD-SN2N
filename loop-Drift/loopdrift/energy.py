"""Scalar finite-horizon LoopSN2N measurements."""
import numpy as np
ENERGY_GAIN_DEFER_Q = 0.05

def _grad_mag(y: np.ndarray) -> float:
    gy, gx = np.gradient(y.astype(np.float64))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))

def _grad_field(y: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(y.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)

def _contrast_scale(y: np.ndarray, mode: str = "std", eps: float = 1e-8) -> float:
    """Contrast normalization for the optional global gradient-gain readout."""
    if mode == "std":
        return max(float(np.std(y)), eps)
    if mode == "mad":
        med = float(np.median(y))
        return max(float(1.4826 * np.median(np.abs(y - med))), eps)
    return 1.0

def _gain_foreground_median(Y: np.ndarray, fg_quantile: float) -> np.ndarray:
    """Median pixelwise gradient ratio on foreground, averaged over streams."""
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
    """Mean squared displacement, displacement norm, between-stream variance and gradient gain."""
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
    """Compute P1–P4 from first-step displacement, residual evolution, dispersion and gradient gain."""
    V = np.asarray(sig["V_res"], float); var = np.asarray(sig["var"], float); gain = np.asarray(sig["gain"], float)
    T = int(sig["T"]); h = max(2, min(int(horizon), T))
    first_residual = float(V[0])                                       # ‖y₂−y₁‖²: near-invariance after one pass
    var_margin = float(np.max(var[:h]))                                # stochastic-stability margin over the horizon
    gdev = np.abs(gain[:h] - 1.0); gain_instab = float(np.max(gdev)); kmax = int(np.argmax(gdev))
    struct = "preserve" if gain_instab <= gain_band_delta else ("over-sharpen" if gain[:h][kmax] > 1.0 else "over-smooth")
    p1 = bool(first_residual < near_inv_cap)                           # stationary after one pass
    p2 = bool(np.all(V[1:] <= V[:-1] * diss_tol)) if V.size >= 2 else True   # dissipative (no residual rebound)
    p3 = bool(var_margin < var_cap)                                    # stochastically stable over the horizon
    p4 = bool(gain_instab <= gain_band_delta)                          # structure-preserving over the horizon
    rel = -(first_residual + var_margin + gain_instab)

    # Normalize each term by its own cap before combining their different units.
    rel_norm = -(first_residual / max(near_inv_cap, 1e-12)
                 + var_margin / max(var_cap, 1e-12)
                 + gain_instab / max(gain_band_delta, 1e-12))
    return {"horizon": h, "P1_stationary": p1, "P2_dissipative": p2, "P3_stable": p3, "P4_structure": p4,
            "structure_mode": struct, "first_step_residual": first_residual, "var_margin": var_margin,
            "gain_instability": gain_instab, "gain_T_terminal": float(gain[-1]),
            "energy_reliability": rel, "energy_reliability_norm": rel_norm}

def decide_from_energy(vd, *, gain_defer_q=ENERGY_GAIN_DEFER_Q):
    """Read the structure gate first; abstain in the gain margin or when a stability check fails."""
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
