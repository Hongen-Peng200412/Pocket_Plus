# Stage1 Wrapper 指标与产物参考

本文面向后续 AI agent，用于快速定位 Stage1 validation 的时间顺序、指标语义和本地产物。

## Validation 时间顺序

1. 同步 sparse candidate runtime 与阈值 cache 到 backbone。
2. 前向得到 dense logits、candidate C、anchor P 与 refined C logits。
3. 计算 `val_loss/global/*`。
4. 更新常规 AP/PRAUC：`val_score/global/atom_PRAUC`、`receptor_PRAUC`、`voxel_ligand_PRAUC`。
5. 用 dense ligand logits 更新 `val_uncapped/best`。
6. 用 candidate builder 真实 per-BOX/per-class sampling boundary 更新 `val_uncapped/sampling`。
7. 用实际唯一化/路由后的 C 更新 `val_capped`。
8. 用 C 内 dense candidate logits 更新 `val_unrefined`。
9. 用 C 内 refined logits 更新 `val_refined`。
10. 由 C 内 TP/FP 和 dense 全空间 GT 计算 `val_score/*_F1`。
11. epoch end 写 scalar、local artifact 与 W&B curve；本地 artifact 和 W&B curve 只在 `trainer.is_global_zero` 执行。

## CPC 字段

- C：candidate voxel set，由 sparse candidate builder 从 dense ligand logits 生成。
- P：pseudo-anchor / anchor points，由 backbone 的 sparse refine 路径生成。
- `candidate_logits`：C 位置的原始 dense ligand logits，用于 `val_unrefined`。
- `ligand_refine_logits_C`：C 位置 refined logits，用于 `val_refined`。
- `candidate_prob`：实际进入唯一 C 后的候选概率，用于 C 质量摘要。
- `candidate_counts`：每个 BOX 的实际唯一 C 数。
- `candidate_p_sampling_by_class`：每 BOX/候选类真实 sampling 边界概率。
- `candidate_target_counts_by_class`：每 BOX/候选类在 cap/实际有效体素约束前的目标候选数。

## 指标命名

路径结构：

```text
<panel>/<optional_subpanel>/global/<metric_leaf>
```

- `global`：全验证集聚合；当前 wrapper diagnostics 不再按原始数据文件夹生成分组指标。
- 二分类不追加 task class suffix。
- 多分类对 task-class 指标追加 `_<class_name>`。
- 对外使用 `receptor`，内部旧字段仍可叫 `voxel_aux_*`。

## 关键指标定义

- `val_uncapped/best/global/p_best`：dense 全空间 best-F1 阈值。
- `val_uncapped/best/global/best_F1`：dense 分支在全空间理论最佳 F1。
- `val_uncapped/best/global/numC_p_best_cutoff`：按 `p_best` 切 dense 空间会得到的候选数。
- `val_uncapped/sampling/global/sampling_F1`：按真实 sampling boundary 在 dense 空间切割得到的 F1。
- `val_uncapped/sampling/global/numC_sampling_target`：采样策略 cap 前目标候选数。
- `val_capped/global/recall`：实际唯一 C 覆盖 dense GT 正例的比例。
- `val_capped/global/num_C`：每 BOX 平均实际 C 数。
- `val_capped/global/num_P`：每 BOX 平均 P/anchor 数。
- `val_unrefined/global/F1`：只在 C 内，用 dense candidate logits 计算的局部 F1。
- `val_refined/global/F1`：只在 C 内，用 refined logits 计算的局部 F1。
- `val_score/global/refined_F1`：端到端 refined F1；FN 使用 dense 全空间 GT，因此会惩罚没进入 C 的正例。
- `val_score/global/unrefined_F1`：端到端 unrefined F1；同样使用 dense 全空间 GT。

## 运行产物

```text
<run_dir>/checkpoints/
<run_dir>/validation_diagnostics/
<run_dir>/validation_diagnostics/epoch_000001/summary.json
<run_dir>/validation_diagnostics/epoch_000001/warnings.json
<run_dir>/validation_diagnostics/epoch_000001/curves/*.csv
<run_dir>/validation_diagnostics/epoch_000001/histograms/*.csv
```

- `summary.json`：当前 epoch 的 scalar 快照。
- `warnings.json`：无正例、空 sampling 边界等 diagnostics warning。
- `curves/*.csv`：PR/threshold 曲线表，若 payload 提供则写出。
- `histograms/*.csv`：固定 bin 统计表，例如 `uncapped_sampling_boundary_hist.csv` 与 `capped_routed_prob_hist.csv`；sampling boundary 分位数和均值从 CSV 推导。

## 排查流程

1. 找最新 run dir，看 `validation_diagnostics/epoch_xxxxxx/summary.json`。
2. 先看 `val_uncapped/best`：判断 dense 分支理论上限。
3. 再看 `val_uncapped/sampling`：判断当前采样策略是否打算切出足够候选。
4. 看 `val_capped/recall` 和 `num_C/num_P`：判断 cap、unique+routed 后是否覆盖正例。
5. 对比 `val_refined/global/F1` 与 `val_score/global/refined_F1`：区分 C 内判别问题和 C 外漏召回问题。
6. 对比 W&B scalar 与本地 `summary.json`；若 W&B curve 缺失，先确认是否为非 global-zero rank 或 logger 不是 W&B。

## 常见异常

- `val_uncapped/best` 高但 `val_capped/recall` 低：dense 可分，但 sampling/cap/unique C 没覆盖正例。
- `val_capped/recall` 高但 `val_score/refined_F1` 低：C 覆盖够，但 refined 判别或阈值选择差。
- `val_refined/F1` 高但 `val_score/refined_F1` 低：C 内局部好，但 C 外仍漏掉大量 dense GT。
- 某 task class 无正例：该类相关 F1 可为 `nan`，其他有正例的 task class 仍可正常统计。
- `uncapped_sampling_boundary_hist.csv` 为空：candidate builder 没产生 finite sampling boundary，检查 selection mode、warmup topc、valid mask 和阈值 cache。
