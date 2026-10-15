# Licensing and attribution

The project license has not yet been selected. This private repository and its build artifacts should not be treated as a fully cleared public open-source release.

## Source attribution

- `fiji_calibrated_plugin/backend/vendor/aunet.py` follows the SN2N/AUnet source lineage. It is retained without numerical changes for checkpoint compatibility. The inspected [upstream license](https://github.com/WeisongZhao/SN2N/blob/main/LICENSE.txt) is an ODbL database-license document, which is not being treated as confirmed permission to redistribute the software. The notice is retained in `third_party/SN2N_upstream_LICENSE.txt`; applicable software permission must be confirmed before public release.
- The Loop energy, Map, noise-calibration and display modules are selected implementations from the LoopSN2N project. They remain subject to the project's eventual license.
- `loopdrift/drifting.py` implements a compact form of the published Drifting equations. It is not a copy of the unlicensed reference repository or the historical training port. See [algorithm notes](docs/ALGORITHMS.md) for attribution and differences. No clean-room claim is made.

The single JAR includes this project's Java classes, Python runtime source and attribution files. It does not include the Python interpreter, PyTorch, Fiji, ImageJ, SciJava, Gson or other dependency libraries. Those packages retain their own licenses and are obtained separately.

## Public release requirements

An open-source release requires a license and copyright attribution for the original project code, resolution of upstream software permissions, and retention of applicable notices. Data and model permissions are separate; this repository contains no research images or trained checkpoints. A successful build or test does not establish redistribution rights.
