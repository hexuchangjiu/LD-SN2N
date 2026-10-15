# 构建与测试 / Build and test verification

✅ **2026-10-15，候选版0.4.0：工程检查通过；不是新增效果实验。** / Engineering checks pass; no new efficacy claim.

| 检查 / Check | 本轮结果 / Result |
|---|---|
| 干净Python环境 / Fresh runtime | Python3.11.13，PyTorch2.6.0+cpu，requirements全新安装成功 / fresh dependency installation |
| Python contracts | **86/86 PASS**，包括22算法、50后端/推理/标定/显示、14安装/构建检查 / no skipped test |
| 实际Java构建 / Real build | JDK21.0.7，现有Fiji484个依赖JAR，目标Java8 / compiled plugin discovery index present |
| Fiji headless smoke | frontend、runtime resolver、menu discovery **3/3 PASS** |
| 中文/空格路径 / Path portability | 实际中文路径构建成功，与普通路径JAR逐字节相同 / identical binary |
| 安装/升级 / Install and upgrade | 独立模拟Fiji目录安装及升级、原文件保留、备份与失败回退通过；未修改用户实际Fiji / isolated installation only |
| Java→已安装Python / Installed end-to-end | 150×64×80 float32，经Java完整导出，**像素完全相同、TYX**，weights-only加载、11次前向、完整17个产物 / real subprocess chain |
| 输出预期 / Expected output | COMPLETE + EMPIRICAL_SAME_SCENE_QUALIFIED + formal DEFER；恒定负对照P4=false，所有D=125 / deliberately poor untrained model |
| Python wheel | 构建及隔离目录安装/import成功；包含backend与算法，无研究目录依赖 / independent package import |
| 数学源码保护 / Scientific source preservation | noise_calibration/support_display/energy/loop_map/aunet五个文件与原资产SHA256完全相同 / no mathematical rewrite |

最终 JAR / Final JAR:

```text
loopsn2n-calibrated-fiji-0.4.0.jar
SHA256 7161464ff77022576341bfbe54bef3ef33487a952a8e5d602d9f0eac3a2bfc6b
```

## 如何重复工程检查 / Repeat engineering checks

```console
python scripts/verify_release.py --output build/verification-new
python scripts/build_fiji.py --fiji "/path/to/Fiji" --work-dir build/fiji-new
```

第一个输出运行时、单测、算法示例日志及来源hash；第二个保留编译和smoke日志。输出目录必须新建，不能覆盖已有研究资产。 / These commands retain diagnostics; use new output paths.

## 已修复的实际失败 / Observed failures corrected

- 最初43项旧测试中2项依赖内部实验路径失败，现改成可移植冻结指纹并保留数值行为检查。 / Two private-path test failures removed.
- 旧PyTorch2.5.1环境被新增安全模式依赖检查拒绝；未降要求，新建2.6CPU环境完成验证。 / Old runtime rejected, not silently accepted.
- Java导出原为ZYX，与严格repeat检查冲突；改为明确TYX、像素不变。 / Axis contract repaired.
- Windows Java参数文件在中文路径按本地编码读取；构建器改用classpath manifest与直接参数。 / Actual Unicode launch fixed.
- 已有输出文件和畸形请求保护、常量query/有效chart优先级补充了回归测试。 / Regression guards added.

## 未验证 / Not verified

⚪ 实际GUI点击、真人交互取消体验；macOS/Linux实机；0.4.0新依赖环境下的CUDA完整链；GitHub Actions远端运行（提供workflow不等于已运行）。 / Desktop acceptance, other platforms, current CUDA and remote CI remain unverified.

⚠️ 演示权重未训练。该测试不会证明细胞结构恢复、正式门控资格、跨模态效力或Drifting提升。AUnet等许可事项另见 `LICENSE_STATUS.md`。 / Synthetic engineering success is not scientific or legal qualification.
