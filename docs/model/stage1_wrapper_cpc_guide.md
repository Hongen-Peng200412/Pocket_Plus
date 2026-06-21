# Stage1 Wrapper CPC 与验证指标指南

本文按实际运行顺序解释 Stage1 validation 中 dense -> C -> P -> refined 的指标含义。

## 1. Dense voxel 先给整张空间打概率

模型先在整个 voxel 网格上输出 ligand 概率。这个阶段还没有稀疏候选，也不受 C/P 数量限制。

常看指标：

```text
val_score/global/voxel_ligand_PRAUC
```

它回答的是 dense ligand head 整体排序能力是否还可以。

## 2. `val_uncapped/best` 看 dense 理论上限

```text
val_uncapped/best/global/best_F1
val_uncapped/best/global/p_best
val_uncapped/best/global/numC_p_best_cutoff
```

这一步在 dense 全空间里扫描阈值，找到理论 best-F1。它不代表实际进入 C 的候选，只说明 dense 分支如果直接切阈值，最多能做到什么程度。

如果这里很低，问题优先在 dense ligand 分支。

## 3. `val_uncapped/sampling` 看采样策略打算怎样切 C

```text
val_uncapped/sampling/global/sampling_F1
val_uncapped/sampling/global/numC_sampling_target
```

这一步使用 candidate builder 真实记录的 per-BOX/per-class sampling boundary 和最终候选行，而不是用全局分位数反推。它回答当前采样策略实际交给后续 C/refine 的候选覆盖情况。

sampling boundary 的完整分布写入 `validation_diagnostics/epoch_xxxxxx/histograms/uncapped_sampling_boundary_hist.csv`，分位数和均值从 CSV 推导。

## 4. `val_capped` 看真正进入 C 的候选是否覆盖正例

```text
val_capped/global/recall
val_capped/global/num_C
val_capped/global/num_P
```

C 会经过 cap、去重和类别路由，实际进入 refine 的候选可能少于采样阶段的目标数。

- `val_capped/global/recall` 高：实际 C 覆盖了大部分 dense GT 正例。
- `val_capped/global/recall` 低：后续 refine 再强也救不回没进入 C 的正例。

## 5. C 进入 P/point/refine 流程

进入 C 后，模型会围绕候选构造 P/anchor 并运行 sparse refine。`num_P` 可以帮助判断 anchor 规模是否异常。

## 6. `val_unrefined` 看 C 内原始 dense logit 判别力

```text
val_unrefined/global/F1
```

它只在实际 C 内计算，用的是进入 C 时保存的 dense candidate logits。这里的 FN 只统计 C 内漏判，不包含 C 外没有进入候选集的正例。

## 7. `val_refined` 看 C 内 refined logit 判别力

```text
val_refined/global/F1
```

它也只在实际 C 内计算，但用 refined logits。它回答 sparse refine 分支在已经给定 C 的情况下能不能把 C 内候选分好。

## 8. `val_score` 看端到端最终结果

```text
val_score/global/refined_F1
val_score/global/unrefined_F1
```

`val_score/global/refined_F1` 是最终端到端 refined 分数。它的 FN 使用 dense 全空间 GT，所以没有进入 C 的正例会被计入漏检。

关键区别：

```text
val_score/global/refined_F1        端到端最终分数
val_refined/global/F1              C 内局部判别分数
val_capped/global/recall           C 是否覆盖正例
val_uncapped/best/global/best_F1   dense 理论上限
```

## 9. global 与 task class suffix

`global` 是整个验证集聚合。当前 wrapper diagnostics 不再按原始数据文件夹生成分组指标。

多分类时，前景类别指标会带 task class 后缀，例如：

```text
val_refined/global/F1_metal_ion
val_refined/global/F1_small_molecule
```

## 10. 二分类为什么没有 macro

二分类只有一个前景类别，macro 与前景指标重复。新版日志省略二分类 macro，避免 W&B 上出现多个数值相同但名字不同的指标。

## 11. 如何读异常组合

- `val_uncapped/best` 高、`val_capped/recall` 低：dense 能分，但 C 采样或 cap 没覆盖正例。
- `val_capped/recall` 高、`val_score/refined_F1` 低：C 覆盖够，但 refined 判别差或 FP 多。
- `val_refined/F1` 高、`val_score/refined_F1` 低：C 内分得好，但 C 外漏召回多。
- 某 task class 指标是 `nan`：该类可能没有 GT 正例，应结合其他类别指标和 warnings 判断。
