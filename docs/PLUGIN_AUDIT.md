# Fiji插件完成度审查 / Fiji plugin capability audit

审查对象 / Scope: release candidate 0.4.0, extracted from the existing calibrated plugin, not a new scientific method. 四类分别为：已实现、部分实现、尚未实现、需要验证。 / Categories distinguish implemented, partial, absent and not yet verified.

## 已实现 / Implemented

| 功能 / Function | 实际实现与依据 / Source evidence |
|---|---|
| Fiji菜单与GUI参数 | `java/src/main/java/org/loopsn2n/fiji/LoopSN2NCalibratedCommand.java`: SciJava command, input/checkpoint/device/output, Swing worker/progress/cancel/result display |
| 原始像素输入 | 同文件 `validateRepeatStack` / `exportAndWriteRequest`: uint16/float32、单通道、≥34帧、显式静态声明、锁定复制全平面、不按ROI裁切 / raw full-frame copy, preserved source |
| 独立运行时 | `RuntimeResolver.java`, `scripts/install.py`, `scripts/check_runtime.py`: 安装相对路径、明确Python、预检、无旧工作树依赖 / relocatable backend, explicit interpreter |
| 噪声检查与采样 | `backend/noise_calibration.py`: query所在pair排除；fit/heldout不同帧对；方差、邻域相关、强度分层、时变/漂移/lag检查；fit-only有限噪声bank / held-out empirical qualification |
| 固定推理 | `backend/model_adapter.py`: AUnet架构/张量/BN检查，eval、FP32、反射halo32、固定chart、无裁剪或tile降级，checkpoint及状态hash / frozen full-frame operator |
| Loop | `backend/bridge.py:run_loop`: K5/T3，共用D1、11次网络调用、10次R，显式seed / bounded deterministic protocol |
| P1–P4核及三态诊断 | `backend/vendor/energy.py:signature_from_Y/verdict/decide_from_energy`，由`bridge.map_and_checks`调用；保留原公式/阈值 / unchanged diagnostic engine |
| LRM与可视化 | `vendor/loop_map.py` + `support_display.py`: 完整选定平面数值Map、蓝橙结构方向/稳定性双变量图例、一像素支持边界、透明层、原Map留存 / numerical and display outputs separated |
| 输出与异常 | JSON/CSV/NPY/NPZ/TIFF/PNG/HTML；排他锁、新目录、失败/取消报告、标定失败不推理、不伪造Map / provenance and fail-closed behavior |
| 构建与安装 | `scripts/build_fiji.py`, `scripts/install.py`: JDK+Fiji依赖构建、SciJava发现smoke、备份升级与回退 / independent package, no experimental repository needed |

所有源码路径在 `fiji_calibrated_plugin/` 下（scripts除外）。文件存在不是完成依据；构建、单测及真实Java/Python链证据见 [VERIFICATION](VERIFICATION.md)。 / Paths are relative to the plugin directory except scripts. Test evidence is separate from file presence.

## 部分实现 / Partially implemented

- **自动三判：**历史P1–P4核可给出reliable/borderline/unreliable兼容诊断；但该新经验噪声域未有独立门控资格，所以正式部署结果始终DEFER。不能称为全自动可信checkpoint认证。 / Diagnostic three-way rules exist; formal qualification is deliberately withheld.
- **预处理与参数：**固定训练chart优先；缺失时query一次min/max明确为探索性。K/T和科学阈值固定，不开放任意调参。 / Chart provenance is limited when absent; no universal preprocessing claim.
- **Map覆盖：**当前选中平面整幅覆盖；不是堆栈每一帧都有Map。支持掩膜不是生物学真值或自动细胞分割。 / Full selected plane only, not full-stack batch maps or validated segmentation.
- **平台便携：**已将构建/安装脚本写为跨平台；本轮只在Windows完成实际验收。 / Cross-platform code is provided; only Windows is locally exercised.

## 尚未实现 / Not implemented in this release

- 任意网络架构/任意N2V、N2N权重加载；目前仅固定AUnet。 / General architecture adapters.
- 活细胞运动、多色、Z-stack和原始SIM相位自动标定路由。 / Dynamic/multichannel/Z/phase calibration routes.
- 插件内Drifting训练、批量教师银行选择、完整论文训练复现。 / GUI training, bank selection and paper training reproduction.
- Fiji update-site自动更新与签名发布、打包Python或CUDA驱动。 / Update site, signed publishing, bundled Python/drivers.

这些没有伪装成已开放功能；可选算法接口不等于GUI完成。 / Algorithm examples do not imply those GUI features exist.

## 需要验证 / Still requires verification

- 实际桌面点击、窗口交互与取消时的用户体验。本轮验证是真实ImageJ对象和共享Java启动链的headless测试，不能替代人工GUI验收。 / Interactive desktop acceptance.
- macOS/Linux真实Fiji安装；新版依赖组合与CUDA完整链。 / Other platforms and current-release CUDA runtime.
- 经验源噪声在反馈态中的物理适用性、独立部署门、跨模态生物学效力。 / Scientific external validation, not engineering patches.
- 公开许可；见根目录授权状态。 / Public redistribution clearance.

## 本次最小必要修复 / Release hardening

1. 去掉内部实验路径依赖的测试与旧打包脚本；冻结核心AST指纹，保留行为测试。 / Portable tests, unchanged numerical engine.
2. 允许weights-only且默认不启用legacy；Python/依赖明确预检。 / Safer checkpoint policy and dependency checks.
3. 拒绝复用有结果的目录，包括错误请求；保留用户已有文件。 / Overwrite protection on successful and failing requests.
4. 不再让query零极差提前阻断有效checkpoint chart；仅真正缺chart才尝试fallback。 / Correct fixed-chart precedence.
5. CLI补TIFF轴/精度/单通道检查；Fiji已声明静态重复的导出明确为TYX，不放宽真正Z输入。 / Matched Java/backend axis contract.
6. 补数值字段/重新加噪轨迹/CSV链接、标准构建和备份安装器。 / Complete export and independent installation.

✅ 没改噪声门槛、P1–P4、Map公式、结构支持参数或原研究资产。 / No scientific threshold/formula changes or edits to original research assets.
