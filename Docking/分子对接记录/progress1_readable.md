# progress1_readable：分子对接阶段报告

本文档更加详细地解释了当前分子对接工作已经完成了什么、结果能说明什么、不能说明什么，以及后续如何扩展。

## 1. 本阶段要解决的问题

Pocket Plus 当前主要预测 ligand 的大致空间位置。下游分子对接希望回答一个更具体的问题：

> 给定一个样本的 cryo-EM map、受体结构、网络预测的若干 ligand 位点，以及该样本的候选 mol2，能否把每个候选 ligand 放到每个预测位点附近，运行 Rosetta / EMERALD-ID 风格的 density-guided docking，并把输出转化为可比较的 site-ligand 分数？

本阶段不是最终 benchmark。它主要验证流程是否可运行、可复用、可审计。

## 2. “成功率”的定义

之前提到的成功率，必须明确：当前统计的是“流程成功率”，不是 docking 几何精度。

对一个 Rosetta docking job，当前定义流程成功为：

1. Rosetta 进程正常结束，即 `returncode == 0`。
2. 输出 PDB 文件存在。
3. scorefile 存在。
4. scorefile 或 stdout 日志中可以解析出 `dG`、`dH`、`lig_dens` 等字段。

因此：

```text
流程成功率 = 成功完成的 Rosetta job 数 / 提交的 Rosetta job 数
```

当前没有计算：

```text
RMSD < 2 Å
RMSD < 3 Å
top-1 ligand identity accuracy
top-k ligand identity accuracy
site assignment accuracy
```

这些需要把 docking 输出和真实 ligand 坐标做系统匹配后才能定义。当前结果不能被解释为“对接精度已经达到某个百分比”。

## 3. 已完成的结果总览

目前完成了 4 组实验。

### 3.1 7zdf round1：单位点闭环

服务器目录：

```text
/home/penghongen/分子对接尝试/7zdf_round1_site001
```

实验矩阵：

```text
1 个预测 site
2 个候选 ligand: ANP, ISW
2 套 receptor: true_receptor, cryoatom_receptor
总 job 数 = 1 x 2 x 2 = 4
```

结果：

```text
4/4 流程成功
```

解释：

* 说明 Rosetta 可以读取当前生成的 receptor-ligand complex。
* 说明 `molfile_to_params.py` 可以为 ANP 和 ISW 生成可用 params。
* 说明真实 EMDB map 可以被 `GALigandDock` 用于 `reference_pool="map"` 的流程。
* 不说明 docking pose 已经与真实 ligand 坐标接近。

### 3.2 7zdf round2：所有预测位点

服务器目录：

```text
/home/penghongen/分子对接尝试/7zdf_round2_all_sites
```

实验矩阵：

```text
2 个预测 site
2 个候选 ligand: ANP, ISW
2 套 receptor
总 job 数 = 2 x 2 x 2 = 8
```

结果：

```text
8/8 流程成功
```

进一步做了 site-ligand 匹配：

* 对每个 site-ligand pair，取 Rosetta `dG` 作为一项 docking cost。
* 另取网络预测 instance 的体素形状和 ligand mol2 原子云形状的径向差异，作为初版 shape cost。
* 组合成本为：

$$
\mathrm{cost}\=0.7 \cdot \mathrm{minmax}(dG)+0.3 \cdot \mathrm{minmax}(\mathrm{shape})
$$

其中越低越好。

当前匹配结果：

```text
true_receptor: site001 -> ANP, site002 -> ISW
mean_receptor: site001 -> ANP, site002 -> ISW
cryoatom_receptor: site001 -> ISW, site002 -> ANP
```

解释：

* true receptor 和两套 receptor 平均结果一致。
* cryoatom receptor 单独结果不同，说明 receptor 来源会显著影响 Rosetta 分数。
* 这个分歧必须在后续多样本实验中持续记录，而不能只汇总平均分。

### 3.3 多样本 smoke：8dd7 和 8x9s

服务器目录：

```text
/home/penghongen/分子对接尝试/multi_sample_smoke_8dd7_8x9s
```

样本：

```text
8dd7: 1 个预测 site, 1 个候选 ligand FAD
8x9s: 1 个预测 site, 1 个候选 ligand DHT
```

每个样本运行：

```text
1 site x 1 ligand x 2 receptor = 2 jobs
```

总结果：

```text
4/4 流程成功
```

解释：

* 说明流程可以跨样本自动读取 map、resolution、receptor、mol2 和推理位点。
* 因为每个样本只有 1 个候选 ligand，不能测试 ligand identity ranking。
* 它们更像跨样本 smoke test。

### 3.4 8pmd：矩形匹配测试

服务器目录：

```text
/home/penghongen/分子对接尝试/multi_sample_8pmd_rectangular
```

样本条件：

```text
3 个预测 site
2 个 ATP mol2 条目
2 套 receptor
```

实验矩阵：

```text
3 x 2 x 2 = 12 jobs
```

结果：

```text
12/12 流程成功
```

这个样本的意义在于：预测 instance 数和 ligand mol2 数不同。因此它不是普通的方阵匹配，而是矩形匹配。

## 4. 普通匈牙利匹配

假设有：

```text
X = 预测 instance 集合
Y = ligand mol2 集合
```

如果 `|X| == |Y|`，可以构造一个方阵成本矩阵：

$$
C_{ij} \= \mathrm{cost}(x_i, y_j)
$$

匈牙利匹配要找一个一一分配，使总成本最小：

$$
\min_{\pi} \sum_i C_{i,\pi(i)}
$$

在 `7zdf` 中：

```text
X = {site001, site002}
Y = {ANP, ISW}
```

所以可以做普通的 2 x 2 匹配。

## 5. 矩形匈牙利匹配

如果 `|X| != |Y|`，比如 `8pmd`：

```text
X = 3 个预测 site
Y = 2 个 ATP mol2
```

当前实现是“矩形匹配”的一个直接版本：

1. 从 3 个 site 中选择 2 个。
2. 把 2 个 ligand 一一分配给这 2 个 site。
3. 剩下 1 个 site 标记为 unmatched。
4. 穷举所有选择与排列，取总成本最低方案。

它等价于小规模矩形 assignment。因为当前数量小，穷举比引入额外依赖更稳妥。

需要强调：当前矩形匹配还没有对 unmatched site 施加显式惩罚。也就是说，“忽略一个 site”目前是免费的。这适合流程验证，但不适合作为最终评价。

## 6. 虚拟实例的下一版设计

用户提出的虚拟实例设计是后续必须加入的方向。

设：

```text
X = 真实预测 instance
Y = 真实 ligand mol2
```

加入虚拟节点后：

```text
X' = X + 虚拟预测 instance
Y' = Y + 虚拟 ligand
```

使得：

```text
|X'| == |Y'|
```

不同边的含义：

1. 真实预测 instance 匹配真实 ligand
   表示接受这个 site-ligand 对，成本来自 docking 分数、shape 分数、位置分数等。
2. 真实预测 instance 匹配虚拟 ligand
   表示忽略该预测 instance，成本应由该预测 instance 的置信度、体积、形状、网络概率等决定。高置信预测被忽略应付出更高成本。
3. 虚拟预测 instance 匹配真实 ligand
   表示候选 ligand 没有找到合适预测 site，成本应由 ligand 本身、真实标签先验或任务设定决定。
4. 虚拟预测 instance 匹配虚拟 ligand
   表示空对空匹配，成本为 0。

这样可以把“多预测”“漏预测”“忽略噪声 instance”都放进同一个 assignment 框架。

## 7. 当前 shape score 的定义与局限

当前 shape score 是一个初版 proxy：

1. 对网络预测 instance，取其体素点坐标。
2. 对 ligand mol2，取其重原子坐标。
3. 分别计算到各自中心的半径分布。
4. 比较两个径向直方图，并加入 90% 半径差异。

形式上：

$$
\mathrm{shape}\=0.7 \cdot L1(\mathrm{radial\ hist})+0.3 \cdot \Delta r_{90}
$$

优点：

* 不依赖 ligand 朝向。
* 计算便宜。
* 可以快速接入匹配矩阵。

缺点：

* 没有比较 docking 后 ligand 的真实空间位置。
* 不考虑拓扑结构。
* 不考虑局部密度重叠。
* 对形状相似但空间错位的情况不敏感。

下一版可以额外加入以下信息：

```text
网络预测 instance voxel
vs
Rosetta docking 后 ligand pose 体素化形状
```

并可以加入：

* 体素 IoU。
* 距离变换损失。
* ligand 原子云到预测 mask 的平均距离。
* 拓扑或连通性描述符。
* ligand 大小、半径、主轴比例等特征。

## 8. 后续机器学习调参

当前 cost 权重是手设：

```text
0.7 * dG + 0.3 * shape
```

这只适合第一阶段探索。

后续可以构造验证集，例如 300 个 density map 对应样本。对每个 site-ligand pair 提取特征：

```text
Rosetta dG
Rosetta dH
lig_dens
ligscore
网络 instance score_mean
网络 instance score_max
预测体素数
预测 mask 与 docking pose 的 shape overlap
ligand heavy atom 数
ligand 半径
receptor 来源
map 分辨率
这个全样本的总体特征，如"样本内全部真实ligand的统计信息"，"样本内全部预测的instance的统计信息"等
... (其他的额外特征)
```

然后训练传统模型：

* 网格搜索线性权重。
* XGBoost 或 LightGBM 或 Catboost 等先进的 tree ensemble(如果实现不复杂且训练成本可接受)。
* Logistic regression / ranking SVM。
* Random forest。

目标可以是：

* 正确 site-ligand pair 的二分类。
* 正确 ligand 在同一 site 中排名靠前。
* 正确 site assignment 的整体匹配成本最小、排名靠前、正确率最高后者它们的混合。

## 9. 当前风险

1. `nstruct=1` 太低，分数波动可能很大。
2. `cryoatom_receptor` 与 `true_receptor` 结果可能不一致。
3. 含金属 ligand 需要单独标注。
4. 当前 shape score 只是 proxy。
5. 当前矩形匹配忽略 unmatched site 的成本。
6. 还没有计算 RMSD 或 identity accuracy。

## 10. 下一步建议

1. 先用 `Docking/docking_pipeline` 复现已有结果。
2. 把所有脚本产物都写进 `/home/penghongen/分子对接尝试`。
3. 把虚拟实例加入 matching。
4. 在 shape score 的基础上增加 docking 后 ligand pose。
5. 将 `nstruct`、权重、虚拟节点惩罚、shape 参数统一配置化。
6. 在更多样本上生成特征表，准备验证集调参。