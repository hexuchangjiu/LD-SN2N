# 授权状态 / Licensing status

⚠️ **本包目前是私有候选交付版，不声明全部内容已获开源许可。** 自有代码的MIT/Apache等许可等待作者选择；在选择前不默认为任何第三方授予再分发权。仓库可见性不在本次自动修改范围。

**This is a private release candidate, not a claim of fully cleared open-source licensing.** An owner-selected license for original work remains pending. Repository visibility is unchanged; no blanket third-party redistribution permission is asserted.

## 复用代码 / Reused code

- `fiji_calibrated_plugin/backend/vendor/aunet.py`: retained byte-for-byte from the existing project snapshot to preserve checkpoint/operator identity. It belongs to the SN2N/AUnet source lineage. The inspected [SN2N upstream license](https://github.com/WeisongZhao/SN2N/blob/main/LICENSE.txt) is an ODbL database-license document; this is **not treated as confirmed software redistribution permission**. A copy is retained under `third_party/SN2N_upstream_LICENSE.txt`. Confirm applicable software authorization before public release, or replace it with a properly authorized compatible implementation and validate numerical parity.
- `vendor/energy.py`, `vendor/loop_map.py`, `noise_calibration.py`, `support_display.py`: selected existing LoopSN2N implementations. Their numerical code is unchanged. Original project ownership must be included in the author's license choice.
- `loopdrift/drifting.py`: a newly written compact implementation of published equations, **not a copy of the unlicensed reference repository/production port**. See `docs/ALGORITHMS.md` for the omitted normalizations and paper attribution. No clean-room claim is made.
- Fiji, ImageJ, SciJava, Gson, PyTorch, NumPy and other dependencies are not vendored into this source archive; installation/build uses separately obtained packages under their own terms. The built plugin JAR contains only this project's compiled Java classes and discovery metadata, not Fiji's libraries.

中文说明：AUnet保留原样是为了保证原权重兼容，但当前看到的上游数据库许可不能直接被当作软件许可。公开前需确认代码授权。选定Loop模块仍待作者统一指定自有代码许可。Drifting未复制缺少LICENSE的上游端口。依赖库不被重新许可或打进插件JAR。

## 公开发布前 / Before making public

1. Confirm the license and copyright-holder names for original work. / 确认自有代码的许可与权利人。
2. Resolve AUnet software permission and retain required notices. / 确认AUnet软件授权并保留声明。
3. Review publication/data/model permissions separately; no private image or trained checkpoint is shipped here. / 数据及模型另行授权，本包不含私有图像或训练权重。

⚠️ 工程构建测试通过不等于上述许可事项完成。本文件是发布核对记录，不提供法律结论。 / Passing engineering checks does not resolve licensing. This is a release checklist, not legal advice.
