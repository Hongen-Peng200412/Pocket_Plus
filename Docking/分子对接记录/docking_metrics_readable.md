# 分子对接指标口径说明

本文档面向用户，解释当前 Pocket Plus Docking 总任务中容易混淆的指标。它不替代 `docking_master_exec_plan.md`，而是把每个指标的分母、分子、阈值和 caveat 单独写清楚，避免把“流程跑通”“前置预测命中”和“真实对接成功”混成一个数。

## 本轮结论

2026-05-21 本地核对 `src/inference/main/two_stage_basic.py`、`src/inference/voxel_tuning.py` 和 `src/inference/voxel_evaluator.py` 后确认：

- `/stage2_threshold_component_policy/best_summary.json` 的 `avg_instance_f1` 属于体素 mask / 连通域覆盖评估口径，不是 ligand center hit 口径。
- 如果此前把 `best_summary.json` 中的 `avg_instance_*` 当成“前置 center hit”，这个解释是错误的。
- `Docking/evaluation/run_evaluation.py` 第一版里的 `site_hit_summary` 才是 docking 评估中定义的前置 center hit，但它当前只看进入 docking 的 selected pred site，仍需补充 Hungarian precision/recall 和分层说明。
- 2026-05-21 00:55+08:00 已在本地补充 `site_hit_summary` 的 Hungarian precision/recall 输出；但由于本轮服务器 SSH 被本地 socket 权限拦截，远端 full evaluation 尚未重跑。
- `rank_summary` 第一版仍不可作为 assignment 成功率，因为它只按 ligand label 是否属于 GT label 集合判断 truth，未绑定 GT occurrence 与 pred site。
- 本轮先因 SSH key 权限未能核对服务器；用户补充 `.skill` 密码后，已只读确认服务器实际值：`best_summary.json` 的 `avg_instance_f1=0.36896760343806495`，仍然不是 center hit；Slurm job `268268` 仍在运行。

## 推理后处理指标

### `avg_voxel_precision`

- 分母：所有预测为 ligand 正类的体素数。
- 分子：预测正类中同时属于 GT ligand mask 的体素数。
- 阈值：由后处理 `threshold` 把概率图二值化；GT ligand mask 由训练/评估数据构建。
- Caveat：这是体素级 mask 质量，不知道 ligand occurrence，不知道 docking pose，也不要求 center 接近。

### `avg_voxel_recall`

- 分母：所有 GT ligand 正类体素数。
- 分子：GT 正类中被预测正类覆盖的体素数。
- 阈值：同 `avg_voxel_precision`。
- Caveat：高 recall 可能来自一个很大的预测区域，不能直接说明有可用 docking site。

### `avg_voxel_f1`

- 分母/分子：由 `avg_voxel_precision` 与 `avg_voxel_recall` 组合得到，公式为 `2PR/(P+R)`。
- 阈值：同体素二值化阈值。
- Caveat：F1 是 mask overlap 的综合分，不是 ligand center 命中率。

### `avg_instance_precision`

- 分母：每个样本的预测 instance 数，再对样本取平均。
- 分子：满足 `pred_cover_ratio >= alpha` 的预测 instance 数，再对样本取平均。`pred_cover_ratio` 是某个预测 instance 被单个 GT instance 覆盖的最大比例。
- 阈值：`alpha`，来自 `eval_params`。
- Caveat：这是连通域覆盖精确率；一个预测 instance 的中心可以离 ligand center 较远，但只要体素覆盖比例过阈值仍可能算 TP。

### `avg_instance_recall`

- 分母：每个样本的 GT instance 数，再对样本取平均。
- 分子：满足 `gt_cover_ratio >= beta` 的 GT instance 数，再对样本取平均。`gt_cover_ratio` 是某个 GT instance 被单个预测 instance 覆盖的最大比例。
- 阈值：`beta`，来自 `eval_params`。
- Caveat：这是 GT 连通域被覆盖的召回，不是“预测中心落在 ligand center 几 Å 内”。

### `avg_instance_f1`

- 分母/分子：由 `avg_instance_precision` 与 `avg_instance_recall` 组合得到，公式为 `2PR/(P+R)`。
- 阈值：`alpha` 与 `beta`。
- Caveat：可用于选择推理后处理参数，但不能直接汇报为 docking 前置 center hit。

### `avg_num_candidates`

- 分母：参与参数搜索的样本数。
- 分子：每个样本后处理保留下来的候选 instance 数求和。
- 阈值：受 `threshold`、`min_component_voxels`、`connectivity_policy` 等后处理参数影响。
- Caveat：候选数少不等于质量高；候选数多会放大 Rosetta job 成本。

## Docking 流程指标

### 样本级流程完成数

- 分母：run 的样本目录数或样本清单数。
- 分子：存在 `samples/{sample_id}/audit/summary.json` 的样本数。
- 阈值：无几何阈值。
- Caveat：样本级 `summary.json` 存在只说明流程结束并写出审计，不等于所有 Rosetta job 都跑通。

### 样本级跳过数

- 分母：run 的样本目录数或样本清单数。
- 分子：没有 `summary.json`，且被 audit 标记为 pending、sample_error、无 ligand、无 selected site 等的样本数。
- 阈值：无几何阈值。
- Caveat：不同跳过原因的科学含义不同；无 dockable ligand 与运行失败必须分开看。

### Rosetta job 跑通率

- 分母：计划生成的 Rosetta job 数。
- 分子：Rosetta 进程返回可解析 scorefile/PDB，且 `result.success == true` 的 job 数。
- 阈值：无 RMSD 阈值。
- Caveat：这只是工程流程跑通率；即使 100% 跑通，也可能 RMSD 很差。

### Rosetta job 失败列表

- 分母：所有失败 job。
- 分子：按失败原因归类的 job，例如 residue 命名冲突、zero-length vector、signal 11、PDB residue 不支持等。
- 阈值：无几何阈值。
- Caveat：失败列表用于修 pipeline，不用于计算真实对接成功率。

### Assignment solver 字段

- 分母：每个 `assignments.json` 中的 assignment 记录数。
- 分子：记录为 `dp_rectangular`、`dp_virtual` 或 `scipy_hungarian` 的求解器名称。
- 阈值：矩阵规模 `n <= 16` 时虚拟节点走 DP，大矩阵应走 `scipy_hungarian`。
- Caveat：solver 只说明 assignment 如何求解，不说明求解出的 ligand/site 是否正确。

## 真实 evaluation 指标

### 前置 center loose hit

- 分母：GT ligand instance 数。
- 分子：每个 GT ligand instance 最近的 selected pred site center 距离小于等于阈值的 GT 数。
- 阈值：当前报告 3 Å、4 Å、8 Å。
- Caveat：当前第一版允许多对一，一个预测 site 可让多个 GT 被记为 loose hit；它不检查 ligand identity，也不检查 pose RMSD。

### 前置 center Hungarian precision

- 分母：selected pred site 数。
- 分子：Hungarian 一一匹配后距离小于等于阈值的 pred site 数。
- 阈值：建议与 loose hit 同报 3 Å、4 Å、8 Å。
- Caveat：本地代码已实现，远端 full evaluation 尚待重跑；它会惩罚多余预测点，因此通常不应高于 loose recall。

### 前置 center Hungarian recall

- 分母：GT ligand instance 数。
- 分子：Hungarian 一一匹配后距离小于等于阈值的 GT instance 数。
- 阈值：建议与 loose hit 同报 3 Å、4 Å、8 Å。
- Caveat：本地代码已实现，远端 full evaluation 尚待重跑；它比 loose hit 更严格，因为一个预测 site 不能同时覆盖多个 GT。

### Pose RMSD 成功率

- 分母：可可靠计算 RMSD 的 pose 行数。
- 分子：RMSD 小于等于阈值的 pose 行数。
- 阈值：主阈值 2 Å，同时报告 3 Å、5 Å。
- Caveat：第一版为 direct atom-order RMSD；atom correspondence 不可靠的行必须单独标记 warning，不应静默混入严格成功率。

### Rank / top-k 接近程度

- 分母：可定义真实 pair 或真实 assignment 的候选组数。
- 分子：真实 pair 或真实 assignment 的 rank 进入 top 1、top 3、top 10% 的组数。
- 阈值：top 1、top 3、top 10%。
- Caveat：当前第一版 `rank_summary` 只按 ligand label 判 truth，且没有绑定 occurrence 与 pred site，因此会高估；修正前只能作为 bug 信号，不可作为结果。

### Assignment-level accuracy

- 分母：有 GT occurrence、selected pred site 和完整候选 cost matrix 的样本数。
- 分子：Hungarian 选出的 site-ligand occurrence 与 GT 一致的样本或 pair 数，具体要同时报告样本级和 pair 级。
- 阈值：identity/occurrence 完全一致；也可以附带 center 和 RMSD 阈值版本。
- Caveat：当前 evaluation 计划中定义了目标，但第一版尚未完成严格实现。

## 当前应采用的汇报顺序

1. 先报流程状态：样本完成/跳过、Rosetta job 跑通率、失败 job 列表、solver 字段。
2. 再报前置预测：center loose hit，后续补 Hungarian precision/recall。
3. 再报最终几何：RMSD 2/3/5 Å，并标明 atom correspondence 方法与 warning 数。
4. 最后报 ranking/assignment：在 occurrence 修正前只说“当前第一版不可用，需要修正”。
