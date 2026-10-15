# 0.4.0 候选交付下载 / Candidate downloads

✅ 2026-10-15：应作者要求将已验证的本地交付原件上传到本私有仓库。 / Author-authorized upload of the verified local artifacts to this private repository.

- [完整安装包 / Full installation bundle](LD-SN2N-0.4.0-fiji-bundle.zip?raw=true)
- [源码快照 / Source snapshot](LD-SN2N-0.4.0-source.zip?raw=true)
- [Fiji JAR](loopsn2n-calibrated-fiji-0.4.0.jar?raw=true)
- [SHA-256 校验清单 / Checksums and file manifest](release_manifest.json)

请优先下载并解压完整安装包，按包内 `docs/INSTALL.md` 安装。已有 Fiji 和 Python 依赖仍是必要条件；单独复制 JAR 不能替代后端安装。 / Prefer the full bundle and follow its installation guide; Fiji and Python dependencies are still required.

这些文件逐字节保留此前已测试交付，源码身份为 `c508543418044c280954c0b74826bba5c54276bd`。后续仅增加下载入口与发布记录；压缩包不会假称包含新的发布提交。 / Archives retain the tested source identity above; later publication metadata is not retroactively included.

✅ 原版工程验证：86项 Python 检查、3项 Fiji headless 检查、安装后的 Java-to-Python 完整链通过；见 [验证报告](../../docs/VERIFICATION.md)。上传前重新核对全部归档成员、字节数及 SHA-256。 / Engineering checks and artifact integrity verified.

⚠️ 内含合成150帧堆栈及明确未训练的恒定输出负对照，不是实际实验数据或合格教师。正式标定入口保持 DEFER；没有新增科学效力结论。 / The synthetic fixture is an installation check, not an efficacy example.

⚠️ 仓库保持私有。自有代码许可及上游软件再分发许可仍待确认；详见 [许可状态](../../LICENSE_STATUS.md)。这不是许可证已闭合的公开发布。 / Private candidate only; licensing clearance remains pending.
