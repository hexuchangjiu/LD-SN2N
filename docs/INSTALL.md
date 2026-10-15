# 安装与使用 / Install and use

## 1. 准备 / Prerequisites

✅ 已安装 Fiji（包含 `jars/` 和 `plugins/`）及 Python 3.10+。建议在独立虚拟环境安装 `requirements.txt`；选择与电脑匹配的 CPU/CUDA PyTorch，不需要重新安装整套实验环境。 / Existing Fiji and Python are required. Use a dedicated environment; choose an appropriate PyTorch build. Full paper-training environment reconstruction is unnecessary.

```console
python -m venv .venv
# Activate this environment, then:
python -m pip install -r requirements.txt
python scripts/check_runtime.py --json
```

⚠️ 使用自己控制的checkpoint；即使weights-only也不是资源隔离沙箱。不要启用legacy pickle信任以绕过陌生文件错误。 / Use controlled checkpoints; even weights-only is not a resource sandbox. Never enable legacy trust to bypass an unknown file's error.

## 2. 构建与安装 / Build and install

✅ 安装包附带 `dist/loopsn2n-calibrated-fiji-0.4.0.jar`。仅源码用户先运行（需JDK 11+，可用Fiji自带JDK）： / The bundle includes a JAR. Source users first run with JDK 11+:

```console
python scripts/build_fiji.py --fiji "C:/Program Files/Fiji"
# Optional: --jdk "/path/to/jdk" --work-dir "/writable/build"
python scripts/install.py --fiji "C:/Program Files/Fiji" --python "D:/myenv/python.exe"
```

以上是路径示例，不是硬编码要求；macOS传Fiji.app，Linux传Fiji目录及bin/python。 / Paths are examples only; macOS uses Fiji.app, Linux its Fiji root and Python binary.

✅ 先关闭Fiji。安装器先检查依赖，支持 `--dry-run`；仅更新本插件，并将旧版本保存到 Fiji/LoopSN2N_backups。不会修改严格历史插件。程序目录无写权限时，用用户可写的Fiji副本；不要绕过系统安全设置。 / Close Fiji first. The installer preflights Python, supports dry-run and backs up prior plugin versions. Use a writable Fiji installation when needed.

✅ 安装在 `plugins/LoopSN2N/`，含JAR、独立runtime及配置；无需原研究仓库。安装器记录Python绝对路径，搬家后重跑安装器即可。回退时关闭Fiji，按备份的 `upgrade_manifest.json` 恢复旧目录与退役JAR，不同时启用两版相同菜单插件。 / Installation is standalone. Reinstall after relocating Python. Backups include a manifest; avoid duplicate menu versions during rollback.

## 3. 输入与运行 / Input and execution

1. 在Fiji打开单通道 uint16/float32 静态重复堆栈，至少34帧；禁止先转8-bit。 / Open ≥34 same-scene static repeats, not an 8-bit conversion.
2. 选择要评估的平面；原图及ROI不被修改，分析使用整个平面。 / Select the query plane; input and ROI are preserved and the full plane is used.
3. 菜单 **Plugins → LoopSN2N → Calibrate noise and assess (research)**。明确确认不是Z、相位或运动序列。 / Open the menu and confirm the acquisition type.
4. 选择兼容 AUnet(1,1,bilinear=True), base64 checkpoint 和可写输出目录；device可选auto/cpu/cuda，advanced paths通常留空。 / Choose a compatible checkpoint and writable output root.
5. 默认禁止legacy pickle；纯state_dict无需勾选。旧`.pth`仅在来源受控时手动允许。当前不支持任意N2V/N2N网络架构。 / State dictionaries do not need legacy trust; architecture support is deliberately narrow.
6. 等待标定及11次网络前向。Cancel写停止标记，在安全边界退出，不中途杀死CUDA调用。 / Cancellation is cooperative, not instantaneous kernel termination.

✅ checkpoint含固定offset/scale时优先使用；无记录才冻结一次query min/max并明确标记未资格化。不得把无训练域身份的预处理称为已校准。 / Stored training coordinates take priority. One-time fallback is explicitly exploratory.

## 4. 查看结果 / Read outputs

| 文件 / File | 用途 / Meaning |
|---|---|
| `report.html`, `response.json` | 完成/失败/取消、标定及正式DEFER / terminal state and formal abstention |
| `calibration.json`, `noise_bank.npy` | held-out检查、fit-only经验噪声 / measured challenge and checks |
| `D1.tif` | 单次恢复结果，float32 / single-pass restoration |
| `D_trajectory.tif`, `R_trajectory.tif` | 15个D与10个R，先k后t；D1共享 / K-major trajectories |
| `map_rgb.tif`, `map_rgba.tif` | 一像素结构边界着色及透明层 / support-aware overlay and alpha |
| `map_rgb_unmasked.tif` | 未掩膜的原双变量颜色 / original unmasked display |
| `C_stab.tif`, `gamma.tif`, `map_components.npz` | 数值响应坐标 / numerical response coordinates |
| `map_legend.png`, `diagnostics.csv` | 双变量图例与P1–P4 / legend and diagnostics |
| `manifest.json`, `worker.log` | 参数、来源hash与运行日志 / provenance and logs |

⚠️ 蓝色=结构衰减方向，橙色=增强方向；另一维编码稳定性。灰色区域保留D1，不代表零响应或已证明背景。噪声检查不通过时只输出报告，不编造Map。 / Blue/orange encode structure direction, not bad/good labels. Gray is D1, not certified zero response. Failed calibration does not fabricate a map.

## 5. 常见问题 / Troubleshooting

- `DEFER`且执行完成：该研究域尚未验证正式门阈值，不是程序故障。 / Completed-with-abstention is intentional.
- 不支持checkpoint：核对架构；不自动替换网络、缩放或tile。 / Unsupported weights fail closed.
- 噪声失败：先查是否真正静态重复及曝光变化；不要放宽筛查以强行通过。 / Inspect acquisition assumptions.
- OOM：保存日志，可显式改CPU重新运行到新目录；不静默改变推理算子。 / Use CPU explicitly if needed; no hidden tile fallback.
- 卡住/取消：查看worker.log，确认旧进程退出后使用新输出目录；不要删除活动锁。 / Do not remove an active run lock.

可选安装自测 / Optional installation fixture:

```console
python examples/make_engineering_demo.py --output demo_assets
python scripts/run_assessment.py --stack demo_assets/synthetic_static_150frames.tif --checkpoint demo_assets/NOT_TRAINED_constant_control.pt --output outputs/smoke --static-repeats --device cpu
```

⚠️ 这是负对照，不是训练成果。可直接在Fiji打开相同文件检查安装。 / This is a deliberately poor non-trained negative control, not a denoising result.
