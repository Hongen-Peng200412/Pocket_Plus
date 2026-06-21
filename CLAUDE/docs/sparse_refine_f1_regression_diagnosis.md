# MINI sparse refine: loss 下降但 refined F1 低于 unrefined F1 的诊断报告

本文记录 2026-06-05 对 8 个 `MINI` sparse refine 任务的排查过程、证据、结论和后续建议。目标读者是后续 AI agent 和第一次接触这条链路的开发者；只读本文应能理解问题背景、已经排除的方向、当前最可能原因，以及下一步该如何验证。

## 1. 问题背景

用户在 `/home/penghongen/My_Project/feedback_plus/logs/MINI` 中观察到 8 个 `MINI` 任务有同一种现象：

- 训练前期，`val_score/global/refined_F1` 与 `val_score/global/unrefined_F1` 恒等。
- sparse refine head 解冻并开始训练后，`train_loss/global/ligand_sparse_refine_step` 下降。
- 但 `val_score/global/refined_F1` 普遍低于 `val_score/global/unrefined_F1`。

这令人困惑，因为直觉上 refined logits 的监督 loss 下降应当带来 refined F1 上升，至少不应系统性差于 unrefined。

这里的两个 logits 语义如下：

- `candidate_logits`: C 候选体素位置上采样出来的原始 dense ligand logits，用于 `val_unrefined/*`。
- `ligand_refine_logits_C`: sparse refine head 输出的 C 级 refined logits，用于 `val_refined/*`。

需要特别注意：C 空间概率分布不等于 dense 全空间概率分布。C 是 candidate builder 从 dense 空间中按阈值、cap、unique 和路由规则筛出的候选集合，通常集中在 dense 概率较高或 sampling 认为重要的位置。C 内 loss 和 F1 是在被选择后的分布上计算的，不能直接等同于 dense 全空间 loss/F1。

## 2. 操作约束和资源使用

本次排查遵守了以下约束：

- 远端只允许在 `/home/penghongen/My_Project/tmp` 内新增或修改诊断文件。
- 只使用服务器资源 `293324`，即 `/home/penghongen/pre_lock_293324`、`/home/penghongen/try_lock_293324`、`/home/penghongen/run_cmd_293324.sh` 对应的 allocation。
- 不修改远端项目源码。
- 本地只新增本文档，不改训练代码。

关于 `293324` 的关键事实：

- `sbatch/_train_core.sh` 支持动态 `run_cmd` 机制。任务进入 `try_lock` 后，可以编辑 `/home/penghongen/run_cmd_293324.sh`，删除 `/home/penghongen/try_lock_293324` 后复用同一个 allocation 执行新的诊断脚本。
- 本次确实通过这个机制跑了 GPU 诊断和补充分项诊断。
- 诊断结束后，`/home/penghongen/run_cmd_293324.sh` 已恢复为原始训练入口：

```bash
#!/bin/bash
python /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=MINI_sparse_refine001"
```

- 截止本报告写入时，`/home/penghongen/after_lock_293324` 和 `/home/penghongen/try_lock_293324` 均存在，allocation 处于暂停状态，不会自动继续运行。

## 3. 诊断产物

远端诊断目录：

```text
/home/penghongen/My_Project/tmp/refine_anomaly_check_293324
```

主要文件：

```text
summary_metrics.csv
wandb_relevant_history_scan.csv
wandb_val_rows_scan.csv
wandb_train_epoch_rows_scan.csv
wandb_train_step_sample_scan.csv
diagnose_refine_logits.py
run_diagnose_refine_logits.sh
MINI_both_front_epoch3_5_subset_logits_diag.json
MINI_both_front_epoch3_5_subset_logits_diag.txt
diagnose_loss_components.py
MINI_both_front_epoch3_5_loss_components.json
MINI_both_front_epoch3_5_loss_components.txt
```

其中：

- `summary_metrics.csv`: 8 个 MINI run 的 epoch 级 refined/unrefined 指标汇总。
- `wandb_*_scan.csv`: W&B history 扫描后的相关行，避免依赖 W&B UI 截图。
- `diagnose_refine_logits.py`: 复用 checkpoint 和 val dataset，比较 base/refined logits、loss、F1、概率扰动。
- `diagnose_loss_components.py`: 在同一批样本上拆分 sparse refine loss 的 focal/Tversky 分项，并计算 C 内 AP。

## 4. 代码链路核对

### 4.1 训练执行入口

`sbatch/_train_core.sh` 的动态执行机制是本次复用 GPU allocation 的基础：

- 先生成 `/home/penghongen/run_cmd_${SLURM_JOB_ID}.sh`。
- 每次执行前打印并运行该文件。
- 成功或失败后，如果 `TRY_AFTER_END_ENABLED=1` 且 `after_lock` 仍存在，则创建 `try_lock` 并暂停。
- 删除 `try_lock` 会进入下一轮执行。

因此，`run_cmd_293324.sh` 不只是一个固定训练命令，也可以临时指向诊断脚本。这个点在本次排查中已实际使用。

### 4.2 sparse refine loss 调度

相关实现位置：

```text
src/wrappers/voxel_point_stage1.py
```

关键逻辑：

- `_resolve_sparse_refine_loss_start_on_steps()`: 解析 `start_on_steps/start_on_ratio`。
- `_resolve_sparse_refine_loss_warmup_steps()`: 解析 `warmup_steps/warmup_ratio`。
- `_compute_sparse_refine_loss_effective_weight()`: 若 `global_step < start_on_steps`，返回硬 0；若超过 warmup，返回最终权重；中间线性插值。
- `_compute_total_loss()`: 始终计算 sparse refine loss term，但总 loss 中乘以 schedule 后的 effective weight。

8 个 MINI 实验的配置覆盖了：

```yaml
model:
  ligand_sparse_refine_loss_schedule:
    warmup_ratio: 0.3
    start_on_ratio: 0.2
```

每个 epoch 约 696 optimizer steps，`max_epochs=20`，估计总步数约：

```text
696 * 20 = 13920
```

因此：

```text
start_on_steps = round(13920 * 0.2) = 2784
warmup_steps   = round(13920 * 0.3) = 4176
```

这正好解释了现象的时间点：

- epoch 0-3 validation 前后，sparse refine effective weight 为 0，head 不参与梯度。
- epoch 4 validation 时，weight 约为 `0.599967`。
- epoch 5 validation 时，weight 约为 `1.199935`，接近/达到最终权重。

### 4.3 sparse refine head 初始 identity

相关实现位置：

```text
src/model/sparse_refine/sparse_refine_head.py
```

当前默认 head 配置：

```yaml
mode: residual
zero_init_residual: true
inputs:
  use_voxel_logits: true
```

含义：

- head 输出 residual delta。
- refined logits = base logits + delta。
- residual 输出 MLP 的最后一层零初始化。
- 初始状态下 delta 为 0，因此 refined logits 应该精确等于 base candidate logits。

GPU 诊断也验证了这一点：`MINI_both_front` epoch3 checkpoint 的 `output_final_l2=0.0`，base/refined loss、F1、概率扰动全部相等。

### 4.4 metric 计算路径

相关实现位置：

```text
src/wrappers/voxel_point_stage1.py
src/wrappers/voxel_point_stage1_diagnostics.py
src/wrappers/metric.md
```

validation 中的关键顺序：

1. forward 得到 dense logits、C、P、refined C logits。
2. `_sample_ligand_refine_supervision()` 从 dense ligand distance map 采样 `target_C/valid_C`。
3. `update_unrefined()` 用 `candidate_logits` 更新 C 内 unrefined histogram。
4. `update_refined()` 用 `ligand_refine_logits_C` 更新 C 内 refined histogram。
5. `_C_panel_scalars_from_hist()` 分别从 histogram 计算：
   - `val_unrefined/global/F1` 与 `val_refined/global/F1`: C 内 local F1，分母是 C 内正例数。
   - `val_score/global/unrefined_F1` 与 `val_score/global/refined_F1`: 端到端 F1，分母是 dense 全空间 GT 正例数。

核对结果：

- unrefined/refined 使用同一份 `target_C`、`valid_C` 和 `dense_num_gt`。
- 二者走同一个 histogram 和 threshold-search 实现。
- `val_score/*_F1` 中 unrefined/refined 各自独立选择最优阈值，不共享阈值。
- local F1 也下降，因此异常不只是 `val_score` 使用 dense GT 分母造成的。

### 4.5 loss 实现路径

相关实现位置：

```text
src/modules/losses.py
src/wrappers/voxel_point_stage1_losses.py
configs/loss/sparse_refine.yaml
```

当前 sparse refine loss 配置：

```yaml
ligand_sparse_refine_loss:
  _target_: src.modules.losses.AdaptiveClassificationCompositeLoss
  num_classes: 2
  hard_label_threshold: 1.7
  focal_gamma: 2.0
  focal_alpha: [0.5, 0.5]
  focal_alpha_neg: 0.5
  focal_alpha_pos: 0.5
  tversky_alpha: 0.5
  tversky_beta: 0.5
  tversky_smooth: 1.0
  w_focal: 0.7
  w_tversky: 0.3
  w_mse: 0.0
```

重要细节：

- `AdaptiveClassificationCompositeLoss.forward()` 在 `logits.shape[1] == 1` 时会转发到 `UnifiedCompositeLoss`。
- 二分类路径实际使用 `focal_alpha_neg` 和 `focal_alpha_pos`，不是多分类路径里的 `focal_alpha` buffer。
- 当前二者都是 `0.5/0.5`，所以数值上没有冲突；但后续调参时不要误以为只改 `focal_alpha` 就一定影响二分类路径。

二分类 composite loss 为：

```text
loss = 0.7 * focal_loss + 0.3 * tversky_loss
```

其中：

- focal 是逐 C 元素的 hard-label focal BCE。
- Tversky 是在当前 batch 的有效 C 上聚合 soft TP/FP/FN 后计算的 soft overlap loss。
- 这个 loss 不直接优化 best-F1、AP 或 ranking。

## 5. 8 个 MINI run 的 W&B/CSV 证据

所有 8 个 run 在 epoch 0-3 refined 与 unrefined 完全相等，epoch 4 之后 refined 低于 unrefined。

下表中：

- `min_score_delta`: epoch >= 4 后 `val_score/global/refined_F1 - val_score/global/unrefined_F1` 的最小值。
- `last_score_delta`: 该 run 最后一条记录的同一差值。
- `min_local_delta`: epoch >= 4 后 `val_refined/global/F1 - val_unrefined/global/F1` 的最小值。
- `last_local_delta`: 该 run 最后一条记录的同一差值。

| run | 记录 epoch | min_score_delta | last_score_delta | min_local_delta | last_local_delta |
|---|---:|---:|---:|---:|---:|
| `MINI_both_both` | 0-5 | -0.007159 | -0.004251 | -0.011790 | -0.006150 |
| `MINI_both_front` | 0-5 | -0.008265 | -0.008265 | -0.010300 | -0.010300 |
| `MINI_none_back` | 0-5 | -0.005121 | -0.003904 | -0.006968 | -0.002877 |
| `MINI_none_both` | 0-4 | -0.005255 | -0.005255 | -0.008843 | -0.008843 |
| `MINI_none_front` | 0-5 | -0.006695 | -0.004481 | -0.009671 | -0.004560 |
| `MINI_real_back` | 0-5 | -0.004746 | -0.003557 | -0.006034 | -0.004299 |
| `MINI_real_both` | 0-5 | -0.005801 | -0.003576 | -0.008257 | -0.005795 |
| `MINI_real_front` | 0-5 | -0.005980 | -0.005253 | -0.008881 | -0.007289 |

代表性 run `MINI_both_front` 的 W&B val row：

| epoch | refine weight(val) | val sparse refine loss | unrefined_F1 | refined_F1 | delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.272181 | 0.565657 | 0.565657 | 0.000000 |
| 1 | 0.000000 | 0.352489 | 0.588947 | 0.588947 | 0.000000 |
| 2 | 0.000000 | 0.353074 | 0.604825 | 0.604825 | 0.000000 |
| 3 | 0.000000 | 0.319593 | 0.614535 | 0.614535 | 0.000000 |
| 4 | 0.599967 | 0.210656 | 0.623791 | 0.617636 | -0.006155 |
| 5 | 1.199935 | 0.215503 | 0.612489 | 0.604224 | -0.008265 |

这说明：

- 等值阶段与 schedule 硬 0 阶段完全对齐。
- 一旦 sparse refine loss 开始进入总 loss，refined F1 就系统性低于 unrefined。
- 此现象跨 8 个 ablation 条件复现。

## 6. GPU 子集 logits 诊断

代表性 checkpoint：

```text
/home/penghongen/My_Project/feedback_plus/logs/MINI/MINI_both_front____job292662/checkpoints/TOP_epoch_03_score_0.6145.ckpt
/home/penghongen/My_Project/feedback_plus/logs/MINI/MINI_both_front____job292662/checkpoints/TOP_epoch_04_score_0.6176.ckpt
/home/penghongen/My_Project/feedback_plus/logs/MINI/MINI_both_front____job292662/checkpoints/TOP_epoch_05_score_0.6042.ckpt
```

诊断设置：

- 使用 293324 allocation。
- 使用前 12 个 validation batches。
- 为了兼容历史 checkpoint 的 `src_snapshot`，诊断脚本将 `src.model` 指向 run dir 中的 `src_snapshot/src/model`，其余 `src.wrappers/src.datasets/src.modules` 使用当前项目代码。
- 诊断中将 candidate warmup runtime 置为 0，使其按 checkpoint cache 走正式 candidate 逻辑。

结果摘要：

| checkpoint | base loss | refined loss | loss delta | local base F1 | local refined F1 | e2e base F1 | e2e refined F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch3 | 0.265956 | 0.265956 | 0.000000 | 0.594485 | 0.594485 | 0.581001 | 0.581001 |
| epoch4 | 0.282999 | 0.209378 | -0.073621 | 0.593014 | 0.588739 | 0.577227 | 0.573292 |
| epoch5 | 0.347958 | 0.231370 | -0.116588 | 0.566419 | 0.556313 | 0.553585 | 0.544151 |

概率扰动：

| checkpoint | 正例平均 `refined_prob - base_prob` | 负例平均 `refined_prob - base_prob` | base 阈值附近平均 delta |
|---|---:|---:|---:|
| epoch3 | 0.000000 | 0.000000 | 0.000000 |
| epoch4 | -0.078392 | +0.042556 | -0.274671 |
| epoch5 | -0.093022 | +0.035724 | -0.163731 |

同一 base local threshold 下的迁移：

| checkpoint | base p | pos_lost | pos_gained | neg_removed | neg_added |
|---|---:|---:|---:|---:|---:|
| epoch3 | 0.965820 | 0 | 0 | 0 | 0 |
| epoch4 | 0.975586 | 16298 | 2 | 13768 | 0 |
| epoch5 | 0.987305 | 17093 | 0 | 17346 | 0 |

解释：

- epoch3 证明初始 refined 是 identity。
- epoch4/5 证明 sparse refine head 训练后确实改变 logits，且原始 sparse refine loss 大幅下降。
- 但这些改变没有改善 C 内 ranking/F1，反而使 local F1、e2e F1 同时下降。
- base 阈值附近大量正例被压到阈值以下；同时一些高分负例也被压低。focal loss 可以因此下降，但 F1/AP 会因为正例排序和最佳阈值结构被破坏而下降。

## 7. loss 分项诊断

补充脚本 `diagnose_loss_components.py` 在同一批样本上拆分了：

- focal loss
- Tversky loss
- soft TP/FP/FN
- C 内 AP
- base threshold 附近的概率和 focal 变化

结果：

| checkpoint | total delta | focal delta | Tversky delta | soft TP delta | soft FP delta | soft FN delta |
|---|---:|---:|---:|---:|---:|---:|
| epoch3 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 |
| epoch4 | -0.073970 | -0.150656 | +0.104964 | -185.335 | +897.508 | +185.335 |
| epoch5 | -0.116676 | -0.204036 | +0.087165 | -220.816 | +798.470 | +220.816 |

C 内 AP 和 F1：

| checkpoint | AP base | AP refined | local F1 base | local F1 refined | e2e F1 base | e2e F1 refined |
|---|---:|---:|---:|---:|---:|---:|
| epoch3 | 0.657619 | 0.657619 | 0.594659 | 0.594659 | 0.581194 | 0.581194 |
| epoch4 | 0.646270 | 0.629610 | 0.593138 | 0.589901 | 0.577297 | 0.574486 |
| epoch5 | 0.603788 | 0.574819 | 0.566377 | 0.556889 | 0.553592 | 0.544495 |

这一组结果是本次排查最关键的证据：

- 总 loss 下降主要由 focal 分项下降贡献。
- Tversky 分项反而变差。
- soft TP 下降，soft FN 上升，soft FP 上升，说明 soft overlap 口径也在变差。
- C 内 AP 下降，说明不是简单阈值选择问题，而是 ranking 也变差。
- 因为总 loss 里 focal 权重为 0.7、Tversky 权重为 0.3，focal 的大幅下降足以掩盖 Tversky、AP、F1 的退化。

## 8. 当前结论

### 8.1 已排除或基本排除

1. 不是 W&B UI 或 plotting artifact。CSV/history 中的标量已经复现。
2. 不是 refined head 在冻结期意外训练。epoch3 `output_final_l2=0.0`，base/refined 完全相等。
3. 不是 `val_score` dense GT 分母单独造成的。C 内 local F1 和 AP 也下降。
4. 不是 refined/unrefined metric 使用了不同 target/mask。代码核对显示二者共享同一 C 级监督和同一 histogram/F1 计算函数。
5. loss 梯度方向不是简单“反着教”。对 hard positive，二分类 focal/Tversky 局部方向仍是推高概率；对 hard negative，局部方向仍是推低概率。问题不是标签正负号写反。

### 8.2 最可能原因

当前最可能原因是：

```text
sparse refine 的训练目标与实际关心的 refined F1/AP/ranking 不匹配。
```

更具体地说：

1. C 是 dense 空间筛出的候选集合，概率分布高度偏向高分区域，不等同于 dense 全空间。
2. 当前 sparse refine loss 在 C 分布上用 `0.7 * focal + 0.3 * Tversky`。
3. `focal_gamma=2.0` 的 focal 对高置信错误非常敏感，会强烈奖励降低高分 false positive 的概率。
4. refined head 学到一种重校准/压缩高分 logits 的 residual 变换：它能显著降低 focal loss，但会同时压低许多 threshold 附近的正例。
5. 由于没有 identity preservation、ranking preservation 或 F1/AP 对齐约束，head 不需要保留 base dense logits 原本较好的排序结构。
6. 结果是 total sparse refine loss 下降，但 Tversky、soft TP/FN、local AP、local F1 和 e2e F1 同时变差。

因此，“损失不匹配”的核心不是一句简单的“focal 权重过高”就结束，而是：

```text
在当前 C 分布、hard-label focal gamma=2、w_focal=0.7、w_tversky=0.3 的组合下，
focal 分项主导了优化方向，使模型更倾向于修正/压缩高置信概率，
而不是保序地提升 C 内 ranking 或 best-F1。
```

## 9. 关于 C 概率分布与 dense 概率分布

用户特别指出 C 概率分布和原始 dense 空间概率分布不是同一个。这是正确且重要的。

区别如下：

- dense 空间包含所有 valid voxel，负例极多，概率分布主体可能在低概率区域。
- C 空间是 candidate builder 按 dense logits、sampling boundary、cap 和 unique 后得到的候选集合，通常富集高概率 voxel 和接近决策边界的位置。
- sparse refine loss 只在 C 上计算，不在 dense 全空间计算。
- `val_unrefined/global/F1` 和 `val_refined/global/F1` 只看 C 内 local 区分能力。
- `val_score/global/*_F1` 虽然用 dense GT 作 recall 分母，但预测仍只能来自 C 内被判正的位置。

这带来两个后果：

1. 在 C 上优化 focal loss，不等价于在 dense 空间优化 ligand 分支。
2. C 内的高分负例和阈值附近样本对 focal loss 的贡献可能极大，导致 head 学到的概率重校准对 F1/ranking 不友好。

后续分析不应把 C loss 曲线直接解读为 dense ligand F1 会改善。

## 10. 改进建议

建议按风险从低到高做实验。

### 10.1 增加常态化日志

先不要只看 `train_loss/global/ligand_sparse_refine_step`。建议新增或临时诊断以下量：

- `ligand_sparse_refine_focal`
- `ligand_sparse_refine_tversky`
- `ligand_sparse_refine_soft_tp`
- `ligand_sparse_refine_soft_fp`
- `ligand_sparse_refine_soft_fn`
- `val_refined/global/AP` 或 C 内 AP
- `delta_prob_pos/neg` 的 histogram 或分位数
- base threshold 附近的 `pos_lost/neg_removed/pos_gained/neg_added`

原因：当前 total loss 下降会掩盖 Tversky、AP 和 F1 退化。

### 10.2 调整 loss 权重和 focal 强度

优先做小矩阵实验：

1. 降低 `w_focal`，提高 `w_tversky`。
   - 例如从 `w_focal=0.7, w_tversky=0.3` 改成 `0.3/0.7` 或 `0.5/0.5`。
2. 降低 `focal_gamma`。
   - 例如 `gamma=1.0` 或暂时 `gamma=0.0` 对照 BCE。
3. 检查二分类路径中实际生效的是 `focal_alpha_neg/pos`。
   - 若要调正负权重，明确改 `focal_alpha_neg` 和 `focal_alpha_pos`，不要只改 `focal_alpha`。

预期观察：

- 如果 refined F1/AP 立刻改善，说明 focal 主导确实是主要问题。
- 如果 Tversky 改善但 F1 仍下降，说明还需要 ranking/identity 约束。

### 10.3 加 identity 或 residual 约束

当前 head 是 residual 结构，但训练目标没有约束 residual 必须小或保序。可以尝试：

- `L2/Huber(refined_logits_C - candidate_logits)`，只作为弱正则。
- 对高置信正例或 base threshold 附近样本加更强的 identity penalty。
- residual scale 参数：`refined = base + scale * delta`，`scale` 初始化很小并限制最大值。
- 对 final residual bias/temperature 施加单独约束，避免整体概率平移或压缩过强。

目的不是让 head 不动，而是防止它在早期破坏 dense logits 已经学到的排序。

### 10.4 加 ranking/AP 对齐目标

如果目标指标是 F1/AP，建议给 sparse refine 加直接面向排序的辅助目标：

- pairwise ranking loss：正例分数应高于负例分数。
- sampled AUC/AP surrogate。
- 只在 C 内 hard positive 与 hard negative 上做 pairwise loss，控制计算量。
- 对 threshold 附近样本加 margin，避免压低大量正例。

这比单纯 focal/Tversky 更贴近 `val_refined/global/F1` 和 `val_score/global/refined_F1`。

### 10.5 调整 sparse refine loss schedule 和权重

当前 epoch4 进入后 refined F1 立刻低于 unrefined。可以尝试：

- 降低 `ligand_sparse_refine_loss_weight`，例如从 `1.2` 降到 `0.3` 或 `0.5`。
- 延长 warmup，避免 residual 在少量 step 内快速改变高分结构。
- 更晚 start_on，只用于验证是否“过早介入”会破坏 dense logits。

这不是根治，但可以判断问题是否与早期 residual 更新幅度相关。

### 10.6 重新审视 C 构造与 loss 采样

由于 C 分布与 dense 不同，可以考虑：

- sparse refine loss 中按 C 内正负比例重采样或重加权。
- 分别统计 routed class、GT positive、GT negative 的概率分布。
- 对进入 C 的负例按 base probability 分桶，避免高分负例完全主导 focal 梯度。
- 对 C 外 dense GT 的影响单独建模，因为 refine 不能恢复没有进入 C 的正例。

## 11. 后续 agent 的建议排查流程

如果后续继续这个问题，建议按以下顺序：

1. 先读本文，再读 `CLAUDE/docs/wrapper_metrics_reference_ai.md`。
2. 检查 293324 是否仍暂停：

```bash
ls -l /home/penghongen/*_lock_293324 /home/penghongen/run_cmd_293324.sh
```

3. 确认 `run_cmd_293324.sh` 是否仍为原始训练命令，不要误跑诊断脚本。
4. 查看已有诊断结果：

```bash
cd /home/penghongen/My_Project/tmp/refine_anomaly_check_293324
sed -n '1,220p' MINI_both_front_epoch3_5_subset_logits_diag.txt
sed -n '1,220p' MINI_both_front_epoch3_5_loss_components.txt
```

5. 若要继续跑 GPU 诊断，使用 `try_lock` 机制：

```bash
# 1. 编辑 /home/penghongen/run_cmd_293324.sh 指向 tmp 下的诊断脚本
# 2. 删除 try_lock，触发下一轮
rm /home/penghongen/try_lock_293324
# 3. 等任务跑完重新生成 try_lock
# 4. 恢复 run_cmd_293324.sh 为原始训练入口
```

6. 不要直接在登录节点跑模型，也不要修改远端项目源码。新增脚本或结果只放在：

```text
/home/penghongen/My_Project/tmp
```

## 12. 一句话结论

这次异常的最可信解释是：sparse refine head 的 composite loss 在 C 分布上被 focal 分项主导，训练后确实降低了 sparse refine total loss，但这个下降来自概率重校准/高分压缩，而不是改善 C 内 ranking；Tversky、soft TP/FN、AP、local F1 和 e2e F1 的证据都指向 refined logits 对实际 F1 指标变差。因此下一步应优先让 sparse refine loss 更贴近 F1/AP/ranking，或给 residual 加保序/identity 约束，而不是只根据 `ligand_sparse_refine_step` 下降判断 refine 有效。
