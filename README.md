# LoopSN2N / DriftingSN2N

LoopSN2N measures how a denoiser responds to repeated, controlled noise challenges. It combines stability and structure diagnostics with a spatial Loop Response Map. DriftingSN2N explores distribution-based learning with a paired self-supervised constraint.

## Fiji plugin

Download [LoopSN2N-0.5.0.jar](plugin/LoopSN2N-0.5.0.jar?raw=true), place it in Fiji's `plugins` folder, and restart Fiji. Open **Plugins → LoopSN2N → Calibrate noise and assess (research)**.

The JAR contains the interface, Python backend, noise calibration and Map code. It unpacks its backend on first use and checks existing Python environments. No separate backend folder or installer is needed. **Python and PyTorch are not bundled.** A compatible environment needs Python 3.10 or later, PyTorch 2.6 or later, and the packages in [requirements.txt](loop-Drift/requirements.txt). The plugin reports missing dependencies; it does not install packages or change your environments.

Open a single-channel uint16 or float32 stack with at least 34 static repeated frames, then choose a compatible SN2N AUnet checkpoint and an output folder. The selected plane is analysed at full size. CPU execution is supported; a working CUDA environment is optional. If automatic discovery cannot find your environment, select its Python executable in the advanced options.

[Installation and use](loop-Drift/docs/INSTALL.md) · [Supported functions](loop-Drift/docs/PLUGIN_AUDIT.md) · [Verification](loop-Drift/docs/VERIFICATION.md)

A small image stack, a compatible trained checkpoint and a step-by-step walkthrough are provided in [plugin/example](plugin/example/). These example assets stay outside Fiji's `plugins` folder; only the JAR needs to be installed there. See the example README for its provenance and tested scope.

最小使用示例见 [plugin/example](plugin/example/)：包含图像堆栈、训练权重与操作说明。仅将 JAR 放入 Fiji 的 `plugins`，示例文件可保存在任意可读目录。

## Code

```text
plugin/          One JAR for Fiji and a minimal usage example
loop-Drift/      Algorithm modules, plugin source, tests and documentation
```

The [code module](loop-Drift/README.md) provides Loop diagnostics and selected Drifting, pixel-distillation, feature-distillation and SN2N objectives. It is not the full experiment or training repository. Only the small usage example is included, not the research dataset or checkpoint bank. The module README explains which directories are required for maintenance and which are optional archives.

The calibrated plugin is a research tool: its formal decision remains `DEFER`, with P1–P4 reported separately as diagnostic checks. A Loop Response Map describes operator response, not pixelwise error probability. Dynamic sequences, Z-stacks, multichannel stacks and arbitrary network architectures are not supported by this entry.

## License

This repository remains private while the project license and upstream software permissions are being resolved. See [licensing and attribution](loop-Drift/LICENSE_STATUS.md).
