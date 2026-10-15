# Fiji installation and use

## Install

1. Close Fiji.
2. Download [LoopSN2N-0.5.0.jar](../../plugin/LoopSN2N-0.5.0.jar?raw=true) and place it in Fiji's `plugins` folder.
3. If an older calibrated LoopSN2N JAR is installed, move that old JAR outside Fiji's `plugins` and `jars` folders to keep a backup and avoid duplicate menu entries. Do not remove unrelated plugins.
4. Restart Fiji. Open **Plugins → LoopSN2N → Calibrate noise and assess (research)**.

There is no separate backend or installer to download. The JAR unpacks its bundled Python source into a versioned cache and verifies the source hashes before using it. It does not modify the input images or the selected Python environment.

## Python environment

Python 3.10 or later, PyTorch 2.6 or later and the dependencies in [requirements.txt](../requirements.txt) are required. Python and PyTorch are not inside the JAR. Existing environments are checked automatically; an unusable candidate does not prevent checking the next candidate. A manually selected executable must pass the checks and is not silently replaced.

If no suitable environment is found, the error report lists the checks that failed. Install the missing dependencies in an environment you control, or select a suitable Python executable in the advanced options. For example, from the `loop-Drift` directory:

```console
python -m pip install -r requirements.txt
python scripts/check_runtime.py --json
```

The plugin does not run pip, download environments or change CUDA drivers. CPU execution is supported. Selecting `cuda` requires a successful CUDA tensor check; an unavailable GPU is not silently treated as CUDA execution. Python paths may need to be selected again after moving an environment.

## Inputs

- A single-channel uint16 or float32 stack containing at least 34 static repeated observations. Do not convert it to 8-bit.
- A checkpoint compatible with SN2N `AUnet(1,1,bilinear=True)`, base width 64.
- A writable output directory and enough memory for full-frame inference.

Select the frame to assess before launching the plugin. The selected frame is processed at full size; an active ROI does not crop the analysis. Z-stacks, SIM phase stacks, moving time series and multichannel stacks are not static repeats and are rejected by this entry.

Use checkpoints from a trusted source. Pure state dictionaries do not need the legacy trust option. Enabling legacy pickle compatibility is an explicit trust decision, not a way to repair an incompatible or unknown checkpoint.

## Outputs

| File | Contents |
|---|---|
| `report.html`, `response.json` | Execution status, calibration and formal decision |
| `calibration.json`, `noise_bank.npy` | Calibration checks and fit-only noise bank |
| `D1.tif` | Single-pass denoised image, float32 |
| `D_trajectory.tif`, `R_trajectory.tif` | Denoised and re-noised trajectories, seed-major order |
| `map_rgb.tif`, `map_rgba.tif` | Structure-supported Loop Response Map and transparent layer |
| `map_rgb_unmasked.tif` | Unmasked bivariate map |
| `C_stab.tif`, `gamma.tif`, `map_components.npz` | Numerical response coordinates |
| `map_legend.png`, `diagnostics.csv` | Bivariate legend and P1–P4 diagnostics |
| `manifest.json`, `worker.log` | Parameters, source hashes and execution log |

Blue and orange indicate structural attenuation and enhancement; they are not labels for a bad or good checkpoint. The other colour dimension describes stability. Uncoloured regions retain D1 rather than certify zero error. A failed noise check produces a report, not a fabricated Map.

The formal calibrated decision remains `DEFER`. The historical three-way P1–P4 result is a separate diagnostic, not an independently qualified deployment decision.

## Troubleshooting

- **No Python found:** select an executable in the advanced options. Python may exist outside the locations searched automatically.
- **Dependency check failed:** read the runtime report, then use or prepare a compatible environment. The plugin will not alter it automatically.
- **Out of memory:** keep the failed-run log and try CPU in a new run. Inference does not silently switch to tiles or resize the image.
- **Noise calibration failed:** check that the stack really contains static repeated observations. Do not loosen the checks simply to obtain a Map.
- **Cancel:** cancellation is cooperative during assessment. Wait for the worker to exit before starting another run; a GPU operation may finish before the stop is processed.

Windows is the locally tested platform. Interactive GUI acceptance, macOS/Linux installations and a current-release CUDA assessment require separate verification. No minimum RAM or VRAM capacity has been established.
