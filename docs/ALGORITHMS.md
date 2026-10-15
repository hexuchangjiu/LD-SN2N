# 选定算法 / Selected algorithms

✅ 本目录提供可运行的小型算法接口，不包含数据、权重、完整训练或历史实验复现。Loop 数值引擎只有一份；本层导入 `fiji_calibrated_plugin.backend.vendor`。  
These are small runnable interfaces, not complete training or historical experiment reproduction. Loop imports the single shared vendor engine.

## Loop 诊断 / Loop diagnostics

`loopdrift.loop_signature(Y)` 接受有限 `Y[K,T,H,W]`（K≥1、T≥2、H/W≥2），输出逐轮 `V_res/reg/var/gain` 和 K/T 元数据。`evaluate_loop(Y)` 复用原 P1–P4 与三态规则。`signature_map_from_Y`、`render_2d_colormap` 为 vendor 的直接导出，不重写公式。  
`loop_signature` returns trajectory metrics; `evaluate_loop` reuses the vendor checks and decision rule. Map and renderer functions are direct re-exports.

⚠️ 便捷接口只分析一条轨迹，不执行插件的标定、完整探针组或噪声准入。默认阈值不是跨模态保证。K=1 的 seed 方差为零，不构成随机稳定证据。常量图的原始 gain 也不应解释为结构保真。vendor 的 P2 检查所有提供的相邻 residual，而 P1/P3/P4 按其原有 horizon 语义执行；这里不改变历史定义。  
This is single-trajectory diagnostics, not the calibrated Fiji protocol. Defaults are not universal thresholds. One seed cannot estimate stochastic dispersion; flat-image gain does not certify fidelity. Vendor P2 checks all supplied residual transitions; other checks retain their original horizon semantics.

⚠️ Map 是闭环响应离散度/结构方向图，不是误差图、正确概率、后验样本或 P1–P4 的逐像素复制。跨样本比较须显式固定共同 `eps` 与显示范围。  
The map is a response diagnostic, not an error probability or pixelwise copy of the gate. Comparisons require shared scales.

## Drifting 公式示例 / Equation-level Drifting example

来源 / Reference: Deng et al., [Generative Modeling via Drifting, v1](https://arxiv.org/html/2602.04770v1), Eqs. 6, 8–12.

对于每个组的查询 x、正样本 p 与负样本 n，定义 / For each feature group:

```text
a_j(x) = softmax_j(-||x-p_j||_2 / temperature)
b_j(x) = softmax_j(-||x-n_j||_2 / temperature)
V(x)   = sum_j a_j(x)*p_j - sum_j b_j(x)*n_j
target = stopgrad(x + step_size*V(x))
loss   = mean_over_groups_samples_channels((x-target)^2)
```

✅ `loopdrift.drifting.drifting_field` 返回分离梯度的 V；`drifting_loss` 返回 `(loss, diagnostics)`。输入为浮点 `[G,N,D]`。默认负样本为同批生成样本，严格去除自身；此时 N≥2。显式负样本不猜测对应关系。正样本由调用者独立抽样；所有 bank/field/target 分离梯度，只有 prediction 接收梯度。  
Inputs are `[groups,samples,channels]`. Default negatives exclude self; explicit negatives are used as supplied. Callers independently sample positives. Only predictions receive gradients.

⚠️ 这是新写的精简公式实现，不是 E57 端口、论文完整训练实现或数值逐位复现。不使用双向 affinity、affinity floor、batch 自适应尺度、force-RMS 归一、多温度累加、样本权重、queue、CFG 或预训练 MAE；每坐标均方为向量平方范数再除 D。temperature/step_size 的单位由调用者特征坐标决定。有限 bank 相同但留一法掩码不同不保证零场；显式完全相同正负 bank 的场为零。  
This new compact implementation is not the E57 port or a full paper reproduction. It omits the listed production normalizations and machinery. Per-coordinate MSE differs from squared vector norm by 1/D. Leave-one-out banks need not yield zero empirical drift; identical explicit banks do.

## 对照损失 / Comparison objectives

- ✅ `pixel_objective(output, paired_teacher)`：配对像素 MSE，teacher 分离梯度。 / Paired image MSE with detached teacher.
- ✅ `image_feature_groups`：等分空间池化 `[B,C,H,W]→[P²,B,C]`，附加原强度坐标下的 raw mean 与 `sqrt(mean(x²)+1e-8)`。可传入冻结 encoder 特征；不附带网络或权重。 / Equal spatial bins retain the sample axis; optional frozen features and raw photometric moments.
- ✅ `feature_objectives`：相同组的等权 feature-MSE 与 Drifting；`encoder.features` 可选，须 eval 且所有参数冻结，student 图像路径仍可微。`feature_scale` 是调用者指定的固定正数，两种目标同除该尺度，不声称 E59 的自适应匹配尺度复现。 / Matched groups, explicit shared fixed scale, detached target paths; no E59 adaptive scale reproduction.
- ✅ `sn2n_constraint(D(a),D(b),a,b,reg=0.5)`：空间 L1、稀疏系数固定为零。reg>0 时为 `[MAE(D(a),b)+MAE(D(b),a)+reg*MAE(D(a),D(b))]/(2+reg)`；**reg=0 时只用 MAE(D(a),b)**。一致性项两侧均保留梯度，a/b 目标分离。无 wavelet、sparse 或 pair-generator。 / Spatial L1 only; preserve the historical forward-only zero-reg branch and both prediction gradients.
- ✅ `combine_objectives(losses,family=...,family_scale=1,sn2n_weight=0)`：固定 `scale*L_family + weight*L_sn2n`。SN2N-only 模式只返回一次原损失，系数须为 1/0。 / Explicit fixed coefficients, no automatic calibration.

⚠️ 分布匹配不保证每张图像保结构；低损失不等于可靠性。训练/教师/编码器不得读取测试数据，抽样独立性不能由张量接口自动证明。  
Distribution matching does not ensure per-image structure preservation. Low loss is not reliability; independence and train/test separation remain caller responsibilities.

## 来源及权限边界 / Provenance and rights boundary

✅ 接口设计核对了项目 `experiments/e57_driftingsn2n/{drift_core,encoder}.py`、`experiments/e59_drifting_attribution/objectives.py` 及原空间 SN2N `_calculate_loss`。只提取选定数学操作，没有搬入旧训练器、checkpoint、私有路径或原 Drifting 端口源码/注释。  
The selected operations were checked against those project sources and the original spatial SN2N loss; training assets and the old Drifting port were not copied.

⚠️ 2026-10-15 核对 `lambertae/drifting@accd0cf09c33b70892d33941d2a287ca86cb92e1` 的 Git tree（68 项）没有 LICENSE/COPYING/NOTICE。故不提供虚构的第三方许可证，不声称该源码获授权；阅读过参考实现，亦不声称 clean-room。论文来源与软件授权是不同问题，整个发布包的授权以根目录通知为准。  
The pinned reference tree has no license file; no upstream redistribution permission or clean-room claim is made. Paper attribution is not a software license. Consult the release's root notices.

## 运行 / Run

```console
python -m examples.algorithm_demo
python -m unittest discover -s tests -p test_algorithms.py -v
```

✅ 测试覆盖手算 field/gradient、bank 分离、反对称性、非法输入、pool 轴、冻结 encoder、SN2N 两个分支，以及 Loop 对 vendor 的精确一致性。这里只验证工程/公式行为，不新增去噪效果结论。  
Tests check formulas, gradients, guards and shared-engine identity, not denoising efficacy.
