# Fiji plugin scope

This document separates implemented software from scientific qualification and untested deployment settings. Source paths below are relative to `loop-Drift`.

## Implemented

| Function | Implementation |
|---|---|
| Fiji menu, input controls and results | `LoopSN2NCalibratedCommand.java`: SciJava command, background worker, progress and cancellation |
| Input preservation | Full-plane uint16/float32 export; static-repeat declaration; source pixels and active ROI preserved |
| Calibration | `backend/noise_calibration.py`: query pair exclusion, disjoint fit/held-out pairs, variance, correlation, intensity and temporal checks |
| Frozen inference | `backend/model_adapter.py`: supported AUnet, FP32/evaluation mode, fixed chart, reflection halo; no silent tiling or resizing |
| Loop | `backend/bridge.py`: fixed K=5/T=3, shared D1, eleven forwards and ten re-noising operations |
| P1–P4 diagnostics | `backend/vendor/energy.py`: residual, dispersion and structure checks with historical three-way diagnostic |
| Loop Response Map | `vendor/loop_map.py` and `support_display.py`: full selected-plane numerical fields, blue/orange direction, stability, one-pixel support boundary and transparent layer |
| Outputs and errors | TIFF, JSON, CSV, NPY/NPZ, PNG and HTML; new output directories, failure/cancellation reports and provenance |
| Single-file distribution | JAR with embedded backend and source hashes; versioned extraction and validation of existing Python environments |

Java sources are in `fiji_calibrated_plugin/java/src/main/java/org/loopsn2n/fiji`; backend sources are in `fiji_calibrated_plugin/backend`. Build and execution evidence is recorded in [verification](VERIFICATION.md).

## Partial support

- **Three-way assessment:** historical P1–P4 diagnostics are available. The calibrated empirical-noise entry still returns formal `DEFER`; it is not a universal checkpoint certification.
- **Preprocessing:** stored training coordinates take priority. Without them, a one-time query range is an exploratory fallback, not proof of calibration.
- **Map coverage:** the full selected plane is mapped, not every frame. Display support is not a validated biological segmentation.
- **Python discovery:** checks configured and discoverable environments, not every directory on a computer. A manual executable override remains available.

## Not implemented

- Arbitrary denoiser architectures or arbitrary N2V/N2N weights.
- Calibration for moving movies, multichannel stacks, Z-stacks or raw SIM phase sequences.
- Drifting training or checkpoint-bank selection in the Fiji interface.
- A Fiji update site, signed release, or bundled Python, PyTorch and CUDA drivers.

## Requires separate verification

Interactive GUI acceptance, macOS/Linux installations, and a current-release CUDA assessment have not been established by the headless tests. The physical applicability of the noise challenge at feedback states, independent deployment thresholds and cross-modality biological efficacy are scientific questions, not packaging tests. Public redistribution also requires resolution of the [license status](../LICENSE_STATUS.md).

The single-JAR update changes distribution and environment discovery, not the numerical definitions of noise calibration, P1–P4 or the Map.
