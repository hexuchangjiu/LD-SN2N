# Loop–Drift code

This directory contains the shared LoopSN2N numerical engine, selected DriftingSN2N objectives, and the source for the Fiji plugin. Fiji users only need the [JAR](../plugin/LoopSN2N-0.5.0.jar?raw=true); the files here are for development and algorithm use.

## Layout

| Directory | Contents |
|---|---|
| `loopdrift/` | Loop signatures, response maps and selected learning objectives |
| `fiji_calibrated_plugin/backend/` | Noise calibration, AUnet inference, Loop and image export |
| `fiji_calibrated_plugin/java/` | Fiji interface, embedded runtime and Python discovery |
| `scripts/` | Runtime checks, build tools and command-line assessment |
| `tests/` | Numerical and packaging tests |
| `examples/` | Small algorithm examples and a synthetic installation test |
| `docs/` | Algorithm notes, usage and verification |
| `third_party/` | Upstream notices |
| `releases/` | Archived releases; not inputs to new packages |

## Development

Run these commands from this directory in a Python environment with the required dependencies:

```console
python -m pip install -r requirements.txt
python scripts/check_runtime.py --json
python -m unittest discover -s tests -v
python -m examples.algorithm_demo
```

To rebuild the single JAR, use an existing Fiji installation and JDK 11 or later:

```console
python scripts/build_fiji.py --fiji "/path/to/Fiji" --output "../plugin/LoopSN2N-0.5.0.jar"
```

The build embeds the backend, dependency checker, requirements and notices together with a file-hash manifest. It does not embed Python, PyTorch, Fiji libraries, data or trained checkpoints. The numerical code is shared between the Python interface and plugin, rather than maintained as separate implementations.

See [algorithm interfaces](docs/ALGORITHMS.md) for definitions and scope, [installation](docs/INSTALL.md) for Fiji use, and [licensing](LICENSE_STATUS.md) for redistribution conditions. The synthetic fixture uses an untrained constant-output network; it tests installation, not denoising performance.
