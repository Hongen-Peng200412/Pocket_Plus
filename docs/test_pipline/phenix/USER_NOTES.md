# Phenix Baseline 诊断说明

这份文档面向用户，解释 Phenix baseline 诊断要看什么、为什么分阶段检查、结果差时如何归因。正式运行命令见上级目录的 `运行指导.md`；Phenix 预生成契约见上级目录的 `phenix.md`。

## 1. 为什么 Phenix 特殊

其它 density-channel baseline 可以直接从已经重采样到模型网格的实验图和模拟图构造 score map。Phenix 需要先由外部程序生成 real-space difference map，再由项目代码读取对齐后的差图。

因此 Phenix 的风险点包括：

1. 外部二进制是否可用。
2. model 结构是否适合喂给 phenix。
3. resolution 是否正确。
4. phenix 输出 map 是否能定位。
5. 输出 map 是否对齐到 cache 网格。
6. 差图转 baseline cache 后，score 分布和 hardmask 是否合理。

## 2. 诊断阶段

### 2.1 生成层

检查 `generate_phenix_diff_maps.py`：

- 从 `protein_40.json` / `protein_110.json` 读取样本。
- 按 system 选择结构和实验图。
- `stardard` 用 receptor-only mmCIF 作为 phenix model。
- `strict` 用 `cif_path` 作为 phenix model。
- 调用 `phenix.real_space_diff_map model.cif map resolution=...`。
- 将输出差图对齐并写到：

```text
EVAL_OUT/phenix_diff_maps/<system>/<sample_name>/phenix_diff_aligned.mrc
```

### 2.2 cache 层

检查 `build_baseline_cache.py`：

- 按 A2 约定派生差图路径。
- 读取对齐差图作为 raw score。
- 用 `hardmask == 0` 的有效区做 rank equalization。
- 写出 DL-compatible `.npz`。
- hardmask 内的 `ligand_pred` 应为 0。

### 2.3 评估层

检查 `run_baseline_two_stage.py`：

- `protein_40 Stage1` 使用高分位阈值粗扫。
- `protein_40 Stage2` 使用高分位局部窗口选择参数。
- `protein_110 fixed test` 固定 Stage2 best params。
- 输出 `cov03/cov06` 的 global instance、loose instance、top-K 与 PR-AUC。

## 3. 指标怎么看

- `avg_voxel_f1`：体素级前景预测和 GT ligand mask 的平均 F1。
- `global_instance_f1_cov03/cov06`：所有样本的预测 instance 与 GT instance 做一对一匹配，再用双向覆盖率阈值 0.3 或 0.6 统计全局 F1。
- `top3/4/5_success_ratio_cov03/cov06`：每个样本按 `score_mean` 取前 K 个预测 blob，只要任意一个 blob 与任意 GT instance 满足覆盖率阈值，就记为该样本命中。
- `pr_auc_macro`：test 阶段 voxel 语义级 Average Precision，在有 GT 正类的样本上做 macro 平均。

这些指标能区分两类问题：voxel F1 差说明整体 score map 或阈值可能不对；global/top-K 差说明候选 instance 是否能落到真实 ligand 上、排序是否把有效 blob 放到前 K。

## 4. 关键检查点

Phenix baseline 结果偏低时，按下面顺序排查：

1. Phenix 命令是否成功，stdout/stderr 是否有异常。
2. `resolution_source` 是否可靠。
3. `phenix_diff_aligned.mrc` 是否存在、非 NaN、非全 0、非常数。
4. 对齐差图的 shape/origin/voxel_size 是否与 reference grid 一致。
5. cache 中 `ligand_pred/hardmask/resampled_emdb/gt_ligand_mask/gt_instance_label` 形状是否一致。
6. rank equalization 后有效区是否覆盖 `[0,1]`，hardmask 区是否为 0。
7. threshold sweep 是否有有效变化，不是全空或全满。
8. top-K 失败时，查看 GT overlap 最大的 blob 的 `score_mean` 排名。
9. instance 失败时，查看 merge、min_component、threshold 对 blob 切分的影响。

## 5. 推荐 smoke 梯度

1. `stardard` 少量样本：确认 receptor-only model、resolution 和对齐链路。
2. `strict` 少量样本：确认 `cif_path` 与 mmCIF 输入链路。
3. 扩到更多样本：确认少量样本结论不由单个样本主导。
4. 完整 `protein_40/protein_110`：进入正式 baseline 对比。

## 6. 结果解释原则

Phenix baseline 先按“非退化 + 可归因”验收：

- 差图非退化。
- cache 字段完整。
- 阈值曲线有效。
- summary 和 per-sample 指标完整。
- 可视化能同时显示实验 map、GT、预测 mask/instance 和 raw diff map。

这些条件满足后，再比较 voxel、instance、top-K 和 PR-AUC 指标。若 voxel 有信号但 top-K 弱，优先分析候选排序与 instance 切分，而不是直接否定差图生成链路。
