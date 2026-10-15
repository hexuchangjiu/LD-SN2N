# LoopSN2N / DriftingSN2N

ImageJ/Fiji plugin for controlled denoise–renoise response assessment, plus selected algorithm interfaces. **Release candidate 0.4.0. Research use.**

用于受控加噪–去噪响应评估的 ImageJ/Fiji 插件，以及选定的算法接口。**0.4.0 候选交付版，研究用途。**

## Start here / 从这里开始

- [Download the verified 0.4.0 bundle / 下载已验证的完整安装包](releases/0.4.0/README.md)
- [Install and use / 安装与使用](docs/INSTALL.md)
- [Plugin capability audit / 插件完成度](docs/PLUGIN_AUDIT.md)
- [Selected algorithms / 代表性算法](docs/ALGORITHMS.md)
- [Build and test evidence / 构建测试](docs/VERIFICATION.md)
- [Licensing and attribution / 许可与来源](LICENSE_STATUS.md)

## What works / 可以做什么

Open a **uint16 or float32, single-channel stack of at least 34 static repeated frames** in Fiji. Select the query plane and a compatible SN2N AUnet checkpoint. The plugin measures same-scene noise, checks held-out repeats, runs a fixed K=5 / T=3 Loop, reports P1–P4 diagnostics, and saves the full selected-plane **Loop Response Map (LRM)**, numerical fields, floating-point trajectories and a local report.

在 Fiji 打开至少34帧的单通道静态重复图像堆栈，选当前平面和兼容的 SN2N AUnet 权重。插件自动检查经验噪声、运行固定 K=5/T=3 Loop，并导出 P1–P4、整幅选定平面的 LRM、数值场、浮点轨迹和本地报告。

**Important:** a measured empirical noise bank is not a physical camera-noise certificate. The calibrated research entry always returns formal **DEFER**; historical three-way P1–P4 readings remain explicitly diagnostic. A completed run is not a guarantee of denoising quality. LRM is not an error probability. Z-stacks, SIM phase stacks, moving movies and multi-channel analysis are not supported by this entry.

**重要：**经验噪声通过不等于物理噪声模型或部署判决已验证。该研究入口正式结论始终为 **DEFER**；历史三态读出单独作为诊断。LRM不是误差概率。此入口不支持Z-stack、SIM相位、运动电影或多通道分析。

## Minimal setup / 最小安装

Use an existing Fiji installation, Python 3.10+ and a current PyTorch 2.6+ runtime. CPU works; CUDA is optional. From this directory:

需要已有 Fiji、Python 3.10+ 和 PyTorch 2.6+；CPU可运行，CUDA可选。从本目录执行：

```console
python -m pip install -r requirements.txt
python scripts/check_runtime.py --json
python scripts/install.py --fiji "/path/to/Fiji.app" --python "/path/to/python"
```

The installation bundle includes `dist/loopsn2n-calibrated-fiji-0.4.0.jar`. A source checkout must build it first, using Fiji's libraries and JDK 11+:

安装包附带JAR；源码仓库先使用已有Fiji依赖和JDK 11+构建：

```console
python scripts/build_fiji.py --fiji "/path/to/Fiji.app"
```

Restart Fiji. Menu: **Plugins → LoopSN2N → Calibrate noise and assess (research)**. Runtime/project paths are resolved from the installation, not the developer's computer. The installer records the chosen Python interpreter; rerun it after moving Python.

重启Fiji后使用上述菜单。项目路径自动读取安装位置；Python解释器移动后需重新运行安装器。

## Repository / 目录

```text
fiji_calibrated_plugin/
  java/                 Fiji GUI, stack export, worker, result display
  backend/              calibration, fixed inference, Loop, LRM, output contract
loopdrift/              selected Loop and Drifting/objective APIs
scripts/               preflight, build, install, CLI and release packaging
examples/              tiny algorithm demo and synthetic negative control
tests/                 numerical, safety, inference and installation contracts
docs/                  short installation, audit and algorithm notes
third_party/           upstream notices; rights review remains explicit
```

No datasets, trained weights, private manuscripts or historical experiment queues are included. Drifting is an equation-level example, not the full paper training implementation. The optional synthetic fixture deliberately uses a **non-trained constant-output network** to test installation; it is not an efficacy example.

不包含实验数据库、训练权重、私有论文或历史队列。Drifting仅开放公式级示例，不提供完整论文训练实现。可选合成示例采用**未训练、恒定输出的负对照网络**，仅检查安装，不代表效果。

## Development / 开发

```console
python -m unittest discover -s tests -v
python -m examples.algorithm_demo
```

Keep numerical changes separate from GUI/packaging edits and rerun contracts. Never tune noise admission or gates to obtain an attractive result. See the audit for supported and unfinished capabilities. No telemetry or remote inference is used; data stay in the selected local output directory.

算法变更与界面/打包分开验证；不为改善展示而放宽标定或门控。当前无遥测或远程推理，图像留在本地。
