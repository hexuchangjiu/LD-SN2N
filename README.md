# LoopSN2N

## Install · 安装

Close [Fiji](https://fiji.sc/), download and extract this repository, and open a terminal in the extracted folder. With Python 3.10 or 3.11, run the command below. Replace the path with the Fiji directory that contains `plugins` and `jars`.

For the BioSR example, use Fiji with Java 21. 关闭 Fiji，解压仓库，在仓库目录运行以下命令。BioSR 需要 Fiji 使用 Java 21；第一次安装需要联网。

```sh
python examples/prepare.py --fiji "C:/Fiji.app"
```

To install just one example, add `--example simulation` or `--example biosr`. 只需一个示例时，在命令末尾加上对应选项。

## Run · 运行

Restart Fiji and open **Plugins → LoopSN2N**. Select an image and its matching model, then click **一键分析** (Analyze). Leave at least 21 GiB free on the drive containing your home folder; results are saved in `LoopSN2N_results` there.

| Image · 图像 | Model · 模型 | Feedback · 循环 |
| --- | --- | --- |
| `Simulation.tif` | `Simulation.npz` | Full 2048 × 2048 simulated field, with the paired simulation noise setting. 全幅仿真图，使用配套仿真噪声。 |
| `BioSR_MT.mrc` | `BioSR_MT.npz` | Nine measured microtubule phases. Each feedback step models a new raw exposure, reconstructs it with fairSIM, then denoises. 从原始九相位模拟再曝光，再重建、去噪。 |

## Read D1 and LRM · 看结果

**D1**, on the left, is the first denoised image. Use **对照输入** to switch between it and the input. 左图 D1 是保留的去噪结果；后续循环不替代 D1。

**LRM**, on the right, shows the local response to repeated perturbations. Blue indicates weaker gradients; orange indicates stronger gradients. Within the same hue, brighter means more stable across streams. Gray alone does not establish that a structure was preserved. 右图中蓝色表示梯度减弱，橙色表示增强；同一色相下越亮，跨流响应越稳定。颜色不是重建正确率。

Choose **查看循环** to inspect the first stream, or **导出结果** to save `D1.tif` and `LRM.tif`. **重开导出** reopens the saved result. A pass describes this image–model pair under the example settings, not ground-truth accuracy.
