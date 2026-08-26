# Li 阈值与三种经典方案的联合结果分析

## 分析范围

本文件比较同一个 `unet_c1` checkpoint 在 calibration 100 个 PDB 与 validation
200 个 PDB 上的 Stage1 basic 结果。完整概率图相同，区别只来自语义截断方法和
basic 候选选择参数。

服务器结果根：

- 经典方案：
  `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950`
- Li 方案：
  `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950--Li`

用户要求比较“一种 Li 阈值与三种经典方案”。Li 是一种逐 PDB 语义截断方法，
但后处理分别按 beta=1 和 beta=2 拟合，因此表中把它列为两个参数配置；这不把
Li 误写成两种语义阈值方法。

| 参数配置 | 语义截断 | basic 候选选择 | 调参 beta |
| --- | --- | --- | ---: |
| Li-F1 | 每个 PDB 独立计算 Li 阈值 | 保留前 0.3391224863 比例，`min_voxels=37` | 1 |
| Li-F2 | 每个 PDB 独立计算 Li 阈值 | 保留前 0.7596153846 比例，`min_voxels=37` | 2 |
| 经典 F1 | calibration macro F1 阈值 0.6070861816 | `score_threshold=0.8210563660`，`min_voxels=16` | 1 |
| 经典 F2 | calibration macro F2 阈值 0.2307128906 | `score_threshold=0.3140856624`，`min_voxels=21` | 2 |
| 经典 F2-beta1 | 与经典 F2 共用相同 F2 blobs | `score_threshold=0.5382110476`，`min_voxels=24` | 1 |

三项和均为 semantic、coverage@0.3 与 one-to-one@0.3 的 PDB 等权 macro
F-beta 之和。`macro F1 三项和` 与 `macro F2 三项和` 对所有配置都重新计算，
不受该配置当初使用哪个 beta 调参的限制。

## Calibration 结果

| 参数配置 | macro F1 三项和 | macro F2 三项和 | semantic macro F1 | semantic macro F2 | top-3@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Li-F1 | 1.031947 | 1.080991 | 0.335450 | 0.388268 | 0.740 |
| Li-F2 | 0.993166 | 1.185250 | 0.324514 | 0.422316 | 0.740 |
| 经典 F1 | 1.270706 | 1.261752 | 0.393582 | 0.423096 | 0.690 |
| 经典 F2 | 1.270434 | **1.372621** | 0.396806 | **0.459373** | **0.750** |
| 经典 F2-beta1 | **1.311421** | 1.354916 | **0.397546** | 0.449525 | 0.730 |

在拟合清单上，经典 F2-beta1 的 macro F1 三项和最高，比经典 F1 高
0.040716，比 Li-F1 高 0.279474。经典 F2 的 macro F2 三项和最高，比
Li-F2 高 0.187371。两项结果都符合各自调参目标。

## Validation 结果

| 参数配置 | macro F1 三项和 | macro F2 三项和 | semantic macro F1 | semantic macro F2 | top-3@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Li-F1 | 1.166699 | 1.232384 | 0.373288 | 0.430503 | 0.800 |
| Li-F2 | 1.087750 | 1.303038 | 0.360381 | 0.458813 | 0.800 |
| 经典 F1 | **1.467956** | 1.463794 | **0.455557** | 0.477842 | 0.785 |
| 经典 F2 | 1.356688 | 1.484019 | 0.438296 | 0.505509 | **0.810** |
| 经典 F2-beta1 | 1.464170 | **1.520253** | 0.455294 | **0.509688** | 0.805 |

经典 F1 的 validation macro F1 三项和最高，但只比经典 F2-beta1 高
0.003786，差异约为 0.26%。逐 PDB 配对结果中，经典 F1 在 121 个 PDB 上高于
经典 F2-beta1，62 个 PDB 上较低，17 个 PDB 相同。经典 F2-beta1 并未牺牲
validation 的 F2 表现：其 macro F2 三项和为 1.520253，反而高于按 beta=2
拟合的经典 F2 结果 1.484019。

Li-F1 相对经典 F1 的 macro F1 三项和低 0.301258；经典 F1 在 144 个 PDB
上更高，Li-F1 在 42 个 PDB 上更高，14 个 PDB 相同。Li-F2 相对经典 F2 的
macro F2 三项和低 0.180981；经典 F2 在 155 个 PDB 上更高，Li-F2 在 35 个
PDB 上更高，10 个 PDB 相同。因此，Li 的差距不是少数大体积 PDB 主导的
micro 现象，在 PDB 等权比较中同样存在。

## 候选规模与 PRAUC

| 数据划分 | 参数配置 | 来源候选总数 | 最终入选总数 | 每 PDB 入选中位数 | 零入选 PDB |
| --- | --- | ---: | ---: | ---: | ---: |
| calibration | Li-F1 | 5,575 | 1,635 | 6.0 | 0 |
| calibration | Li-F2 | 5,575 | 3,248 | 12.5 | 0 |
| calibration | 经典 F1 | 2,998 | 2,198 | 4.0 | 16 |
| calibration | 经典 F2 | 3,703 | 2,877 | 8.5 | 6 |
| calibration | 经典 F2-beta1 | 3,703 | 2,467 | 5.0 | 8 |
| validation | Li-F1 | 14,134 | 4,181 | 7.0 | 0 |
| validation | Li-F2 | 14,134 | 8,161 | 16.0 | 0 |
| validation | 经典 F1 | 7,617 | 5,542 | 6.0 | 12 |
| validation | 经典 F2 | 9,285 | 7,275 | 10.0 | 7 |
| validation | 经典 F2-beta1 | 9,285 | 6,234 | 7.0 | 10 |

Li 实际阈值在 calibration 的中位数为 0.039642，validation 的中位数为
0.042236，明显低于两个经典语义阈值。它因此比经典 F2 多产生约一半来源
候选，再由候选比例和 `min_voxels` 收缩。Li 的 top-3@0.3 仍有竞争力，例如
validation 为 0.800；但该局部优势没有转化为更高的 semantic、coverage 与
one-to-one macro 三项目标。

所有配置在同一数据划分读取相同完整概率图，所以 PRAUC 完全一致：

| 数据划分 | semantic macro PRAUC | semantic micro PRAUC |
| --- | ---: | ---: |
| calibration | 0.4096417393 | 0.5433163278 |
| validation | 0.4572529597 | 0.5080070029 |

PRAUC 在这里验证概率图来源一致，不用于区分候选截断与后处理方案。

## 结论

1. 当前 `unet_c1` 结果不支持用逐 PDB Li 阈值替代正式经典语义阈值。Li-F1
   与 Li-F2 在 calibration 和 validation 的对应 macro 三项目标上都明显较低。
2. 若需要一套兼顾 F1 与 F2 的 basic 参数，经典 F2-beta1 是本次结果中最均衡
   的方案：validation F1 与经典 F1 近乎相同，同时取得最高 validation F2
   三项和；其 top-3@0.3 也只比经典 F2 低 0.005。
3. 若唯一目标是 calibration 拟合口径下的 F2 或 validation top-3@0.3，经典
   F2-beta2 仍应保留；若只看 validation macro F1，经典 F1 以极小优势领先。
4. calibration 与 validation 都是既定开发数据划分。本分析可以作为后续选择
   默认候选方案的实证依据，但不会自动修改正式推理契约，也不把两者的差异
   外推为独立测试集结论。
