# Loop–Drift code

This directory contains the shared LoopSN2N numerical engine, selected DriftingSN2N objectives, and the source for the Fiji plugin. Fiji users only need the [JAR](../plugin/LoopSN2N-0.5.0.jar?raw=true); the files here are for development and algorithm use.

## Layout

| Directory | Role | Retention |
|---|---|---|
| `loopdrift/` | Public Loop interfaces and Drifting, pixel-distillation, feature-distillation and SN2N objectives | Keep: the algorithm module provided by this repository |
| `fiji_calibrated_plugin/backend/` | Noise calibration, AUnet inference, shared Loop/Map calculations and image export | Keep: embedded in the JAR and imported by `loopdrift/`; this is not a redundant copy |
| `fiji_calibrated_plugin/java/` | Fiji interface, input export, runtime extraction, Python discovery and Java integration tests | Keep: needed to rebuild and maintain the plugin |
| `scripts/` | Dependency checks, build/package tools and command-line assessment | Keep: several build tools share helpers defined in `install.py` |
| `tests/` | Numerical, input-safety and packaging regression tests | Keep for maintenance; not read by an installed JAR |
| `examples/` | Algorithm usage example and a synthetic negative-control generator | Optional for end users; keep the algorithm example if retaining the current verification workflow |
| `docs/` | Installation, algorithm interfaces, supported features and test evidence | Keep installation instructions; other reports can be consolidated with their links updated |
| `third_party/` | Upstream attribution and license text | Keep: included in builds and needed for attribution |
| `releases/` | Original 0.4.0 artifacts and their checksums | Optional archive: not used by current builds; can be stored outside the working repository if archive links are updated |

An installed JAR does not read this source directory. Keeping it is what makes the plugin maintainable and rebuildable. For a smaller repository, the first candidate for external archiving is `releases/`, not the backend or tests. No directories have been removed.

已安装的 JAR 不依赖旁边这份源码目录；保留源码是为了后续修改、构建与测试。若需精简，优先将 `releases/` 移至仓库外归档，不应直接删除共享后端或测试。本次未删除任何目录。

### Files with less obvious dependencies

- `scripts/install.py` is not required for a normal single-JAR installation, but `build_fiji.py` and `package_release.py` import its validation and packaging helpers. Remove the installer only after separating those shared functions and updating the tests.
- `scripts/check_runtime.py`, `requirements.txt`, `LICENSE_STATUS.md` and `third_party/SN2N_upstream_LICENSE.txt` are inputs to the embedded runtime. Removing them breaks the current build.
- `pyproject.toml` defines the Python package and dependencies. Keep it if the algorithm module is to remain installable.
- `scripts/verify_release.py` runs `examples/algorithm_demo.py`; deleting that example requires a corresponding verification change.

其中 `install.py` 虽不是用户安装时的必需品，仍被构建脚本调用；`requirements.txt`、运行环境检查与许可文件也参与 JAR 构建，不能只按文件名判断是否可删。

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
