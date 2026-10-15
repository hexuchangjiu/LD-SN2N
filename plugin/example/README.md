# Small Fiji example / Fiji 小示例

A real 150-frame LAMP crop plus an existing trained SN2N AUnet checkpoint; about 70.7 MiB in total. This is a runnable workflow example, not a denoising-quality benchmark or an accepted checkpoint.

真实 LAMP 150 帧裁剪与已有训练完成的 SN2N AUnet 权重，总计约 70.7 MiB。本例用于走通流程，不是去噪质量基准，也不是“已通过可靠性认证”的权重。

## Run in Fiji / 在 Fiji 运行

1. Close Fiji, copy [LoopSN2N-0.5.0.jar](../LoopSN2N-0.5.0.jar) into its `plugins` folder, then restart Fiji. Move any older calibrated LoopSN2N JAR outside Fiji's `plugins` and `jars` folders first to avoid duplicate menus; preserve unrelated plugins. Python 3.10+ with PyTorch 2.6+ and the dependencies below must already be available. / 关闭 Fiji，将 JAR 复制到其 `plugins` 目录再重启；旧版 calibrated LoopSN2N JAR 先移出 `plugins`/`jars` 目录备份，不动其他插件。另需准备下述 Python 环境。
2. Open `LAMP_C1_first150_center128_uint16.tif`. Keep all 150 frames and the original 16-bit pixels. Select **frame 3** (zero-based query index 2). Do not convert to 8-bit, resize, or select an ROI. / 打开 TIFF，保留全部 150 帧与 16-bit 像素，选择**第 3 帧**，不转换、不缩放、不另选 ROI。
3. Open **Plugins → LoopSN2N → Calibrate noise and assess (research)**. Confirm static repeats and select `SN2N_LAMP_seed400101_update24000.pth`. / 进入此菜单，确认静态重复观测，并选本目录中的权重。
4. Leave **Allow trusted legacy checkpoint compatibility** unchecked. Select **cpu**, leave the developer backend override blank, and choose a writable output folder **outside this repository**. Python discovery may be automatic or explicitly pointed at your compatible environment. / 旧 pickle 兼容选项保持关闭；设备选 cpu，后端覆盖路径留空，输出选仓库外的可写目录。可自动查找或手动指定 Python。
5. Run, then open the new run's `report.html`. The plugin processes one selected 128×128 plane; the other frames supply noise calibration. / 运行后看新目录中的报告；仅对选中的 128×128 平面生成 Map，其余帧用于噪声标定。

Python dependencies: `torch>=2.6,<3`, `numpy>=1.24,<3`, `scipy>=1.10,<2`, `tifffile>=2023.7.10`, `Pillow>=10`, `matplotlib>=3.7,<4`, `scikit-image>=0.21,<1`. The JAR bundles the backend but does not install Python or these packages. / JAR 自带后端源码，不自动安装 Python 或以上依赖。

## Expected result / 预期结果

Tested on 2026-10-15 with the actual 0.5.0 JAR backend, Python 3.11 / PyTorch 2.6.0+cpu, 4 CPU threads. Real ImagePlus reading/export preserved every uint16 pixel; Java→Python execution and direct backend execution produced identical numerical arrays. The backend took about 5 seconds on the test machine; this is not a runtime guarantee. Interactive GUI clicking and other machines remain unverified.

已用实际 0.5.0 JAR 后端验证；ImagePlus 读写逐像素无损，Java 调用与后端直跑的数值数组完全一致。本机约 5 秒，不保证其他机器速度；交互界面点击尚未验证。

| Item / 项目 | Observed / 实测 |
|---|---|
| Execution / 执行 | `COMPLETE` / `COMPLETED_WITH_ABSTENTION` |
| Noise screen / 噪声检查 | `EMPIRICAL_SAME_SCENE_QUALIFIED` |
| Formal decision / 正式决策 | **`DEFER`** |
| Legacy diagnostic / 历史诊断 | P1/P2/P3 true; P4 false, `over-smooth` |
| Loop / 循环 | K=5, T=3; 11 D calls, 10 R calls / 11 次去噪、10 次重加噪 |

Look for `D1.tif` (128×128 float32), `D_trajectory.tif` (15 planes), `R_trajectory.tif` (10 planes), `map_rgb.tif`, `map_rgba.tif`, `map_rgb_unmasked.tif`, `C_stab.tif`, `gamma.tif`, the two-axis `map_legend.png`, and reports/manifests. Outputs are generated locally, not pre-bundled. / 应生成这些图像、数值分量与报告；本目录不预装运行结果。

The checkpoint has no recorded training intensity chart. Current backend behavior freezes this query's min/max once (`offset=77`, `scale=70`) for exploratory inference and reports `manual_unqualified`; it does not renormalize each loop round. This is not verified training-domain matching. The noise screen is same-scene empirical qualification, not physical camera calibration. DEFER and the P4 over-smoothing warning are expected outcomes. No ground truth, PSNR improvement or deployment safety is claimed.

权重没有记录训练强度坐标；当前后端仅将该查询帧的最小/最大值固定一次，报告 `manual_unqualified`，不会逐轮重归一化。这不代表训练域匹配已经验证。噪声检查仅适用于同视野经验扰动，不是物理相机标定。DEFER 和 P4 过平滑警告是本例的预期结果；本例没有 GT、PSNR 改善或部署安全性结论。Map 颜色是响应诊断，不是像素误差或概率。

## Provenance and preservation / 来源与保留

The crop was fixed before testing: original 500×1249×1183 uint16 `LAMP_C1_low_full500.tif`, frames `[0:150]`, rows `[560:688]`, columns `[527:655]` (zero-based, end-exclusive). Frame 3 was retained from the Complete package's example settings before either run. Values and frame order are unchanged; there was no crop or query-frame sweep. The trained seed-400101 / update-24000 checkpoint is copied byte-for-byte from the 2026-09-23 Complete package and matches its training record. It is a 128-tensor `model_state_dict` container, safely loaded with `weights_only=True`; no retraining or model modification occurred. File hashes and detailed QA are in [manifest.json](manifest.json).

测试前固定取原始 500 帧数据的前 150 帧及中央 128×128 区域，并沿用 Complete 包预设的第 3 帧作为查询。像素和帧序未变，没有按结果挑选裁剪或查询帧。权重来自已有 2026-09-23 Complete 包，与训练记录哈希一致，仅逐字节复制；没有新训练或模型修改。详情见 manifest。

These user-controlled assets are included only for the currently authorized private repository. This folder does **not** grant a public redistribution license for data or weights. Confirm rights before publishing them elsewhere. Source TIFFs, weights, prior demo packages and scientific code were not changed. / 仅用于当前已授权私有仓库；此目录不授予数据或权重的公开再分发许可。公开发布前须另核权利。原始数据、权重、旧演示包与科学代码均保留。
