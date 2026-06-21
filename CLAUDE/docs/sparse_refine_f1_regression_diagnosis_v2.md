# Sparse refine F1 regression diagnosis v2

生成时间：2026-06-15  
目标运行：`/home/penghongen/My_Project/feedback_plus/logs/MINI/MINI_real_front____job297981`  
本次任务边界：半只读诊断。服务器与项目源码只读取、统计、对照；未启动训练/推理，未修改服务器文件；本地仅写入 `tmp/sparse_refine_f1_regression_v2/` 中间材料和本文档。

## 1. 结论摘要

最明显的结论是：`job297981` 这次换 loss 后，确实基本修掉了“refined 在 C-local 层面明显低于 unrefined”的旧问题，但 global 层面仍然吃亏的主因已经不像是 sparse refine loss 本身，而更像是候选集 `C` 的召回瓶颈和 refine head 只能在 `C` 内做局部重排序。

`297981` 的 sparse refine 在 epoch 4 之后开始产生非零差异。C-local 的 refined F1 相比 unrefined F1 变成焦灼或微弱优势，例如 epoch 5 的 C-local F1 delta 约 `+3.60e-4`，epoch 12/latest 约 `+1.93e-4`。这说明新 loss 至少不再像旧 MINI runs 那样系统性破坏 C 内排序。

但 global score 仍然没有稳定优势。以 best refined-score checkpoint 所在的 epoch 9 看，`val_score/global/refined_F1=0.52386862`，`unrefined=0.52398998`，delta 约 `-1.21e-4`。latest epoch 12 也类似：C-local delta 约 `+1.93e-4`，global score delta 约 `-8.48e-5`。量级上这不是“大幅退化”，但需要正视一个可疑现象：global `refined_F1` 在 epoch 4、9、12 等多数 epoch 上**系统性、普遍地低于 unrefined**（只有 epoch 5/6 短暂转正）。“不升”本身可以接受，但“普遍低”说明这更像是 refine 引入的校准/排序漂移在 dense 全空间分母下被稳定地放大成净负，而不是随机噪声抖动。这一点单靠“调整量级极小”无法解释，需在后文（§3.1、§5.5）专门追因。

更关键的是，`297981` 相比旧 `MINI_real_front` run 的 candidate budget 收得很紧：`warmup_topc_per_class/max_candidate_voxels_per_class` 从 `25000` 降到 `10000`，`adaptive_expand_factor` 从 `7.0` 降到 `2.5`，`max_anchors_per_class` 从 `1024` 降到 `800`。对应现象非常直接：旧 run `292668` 的 `num_C≈12599.7`、`val_capped/global/recall≈0.7505`；新 run `297981` 的 `num_C≈4270.1`、`val_capped/global/recall≈0.4857`。也就是说，global F1 很大一部分已经被候选集召回上限锁住了。

补充回应一个关键追问（“num_C≈4270.1，但 C 上限是 10000，是不是少量样本最佳 F1 截断数超过 10000，先天卡死 refine head？”）：`num_C≈4270.1` 是 batch 平均值，而每类硬上限 `max_candidate_voxels_per_class=10000`。从 `candidate_set.py` 的 `adaptive_threshold` 逻辑看，`num_C≈ceil(n_best_box·expand_factor)` 再被 `min(·, 10000, N_valid)` 截断，其中 `n_best_box=count(prob>p_best)`、`p_best` 就是验证集记录的 best-F1 阈值。按 `expand_factor=2.5` 反推，平均 `n_best_box≈1708`，`2.5×1708≈4270<10000`——**平均意义上 10000 这个 cap 根本没被触发，真正卡住 C 规模的是 `expand_factor=2.5`**。但你的直觉在“少量样本”这一层是成立的：对 ligand 体积大、`n_best_box` 高的样本，一旦 `2.5·n_best_box>10000`（即 `n_best_box>4000`），甚至 `n_best_box` 本身就接近/超过 10000，cap 会把这些样本的真正正例直接挡在 C 外，refine head 对 C 外漏召确实“先天无能为力”——这与 §3.3 中 `val_uncapped/best_F1≈0.668` 远高于 `val_capped/F1≈0.477` 的大缺口一致。

但要修正一处推理链：cap 截断对 refined 和 unrefined 是**同等**作用的（两者都在同一个 C 内统计、用同一全空间分母），它解释的是 global **绝对值偏低**，并不能单独解释为什么 refined 普遍略低于 unrefined。后者是 C 内的校准/排序漂移问题（见 §5.5），与 cap 是两条独立的因果链，会叠加但不应混为一谈。

因此优先级最高的下一步不是继续微调 refine head，而是先把 candidate budget 上调到 `cap 25000 / expand_factor 5.0 / max_anchors 1024`（其余尽量不动）。这里回应两个问题。其一“expand_factor 是否仍保持 2.5？”——不：承接上一段的反推，平均 `num_C` 由 `expand_factor·n_best_box` 决定而非 cap，若只把 cap 抬回 25000 却保留 `expand_factor=2.5`，`num_C` 仍会停在约 4270、召回几乎不动；`expand_factor` 才是恢复 C recall 的真杠杆，cap（10000→25000）只在少数 `n_best_box` 很高的大 ligand 样本上二次生效。其二“是否一路回到旧值 7.0？”——不必：7.0 是把 C 撑到一个目前看较难达到的高上限，性价比和稳定性都存疑，折中取 `expand_factor=5.0` 更稳妥（约 `5.0×1708≈8540`，仍在 25000 cap 之下、又显著高于现在的 4270）。所以这组控制变量定为 `expand_factor 2.5→5.0`、`cap 10000→25000`、`max_anchors 800→1024`。如果 global score 随之恢复，而 refined-vs-unrefined delta 仍接近零或微正，就可以确认“旧 loss 造成 C-local regression，新 budget 造成 global absolute regression”这两个问题是叠加的。

## 2. 本次采集的运行事实

### 2.1 job 目录与状态

目标目录存在：

```text
/home/penghongen/My_Project/feedback_plus/logs/MINI/MINI_real_front____job297981
```

主要内容包括：

```text
checkpoints/
config.yaml
src_snapshot/
train.yaml
validation_diagnostics/
wandb/
```

训练诊断 epoch 目录从 `epoch_000000` 到 `epoch_000012`，checkpoint 包括：

```text
TOP_epoch_03_score_0.5078.ckpt
TOP_epoch_04_score_0.5205.ckpt
TOP_epoch_06_score_0.5220.ckpt
TOP_epoch_09_score_0.5239.ckpt
TOP_epoch_10_score_0.5227.ckpt
TOP_epoch_12_score_0.5220.ckpt
last.ckpt
```

W&B 主 run 为：

```text
run-20260612_010441-3c753wi5
```

Slurm 上 `sacct -j 297981` 显示 `RUNNING`，但这不能直接解释为 sparse refine training 仍在跑。`_temp_slurm/train_297981.out` 显示该训练段曾被 `kill_lock_297981` 终止，之后同一个 job slot/run command 被复用为不相关的 `src/inference/main/two_stage_basic.py`。因此判断 sparse refine 训练结果应以该 log 目录和 W&B/diagnostics 为准，而不是以当前 Slurm job id 的 `RUNNING` 状态为准。

### 2.2 关键配置

监控指标：

```yaml
monitor_metric: val_score/global/refined_F1
monitor_mode: max
```

训练主要设置：

```yaml
max_epochs: 20
global_batch_size: 40
batch_size: 5
precision: bf16-mixed
gradient_clip_val: 0.5
optimizer: AdamW
lr: 3e-5
weight_decay: 0.01
```

sparse refine / detach 相关：

```yaml
enable_atom_head_front: true
enable_atom_head_back: false
detach_real_point_feat: true
detach_pseudo_point_feat: false
detach_voxel_feat_into_real_point: true
detach_voxel_feat_into_pseudo_point: true
detach_voxel_into_refine: true
refine_receptor_from_voxel: true
pseudo_density_residual: false
enable_interface_norm: true
sparse_refine_residual_mode: residual
sparse_refine_use_voxel_logits: true
```

candidate/P 相关：

```yaml
selection_mode: adaptive_threshold
candidate_class_ids: [1]
warmup_topc_per_class: [10000]
adaptive_expand_factor: [2.5]
max_candidate_voxels_per_class: [10000]
max_anchors_per_class: [800]
anchor_sampler.mode: unweighted_fps
nms_radius_voxel: 2
anchor_to_candidate.num_neighbors: 4
anchor_to_candidate.same_class_only: true
```

sparse refine head 相关：

```yaml
mode: residual
zero_init_residual: true
distance_weight.mode: softmax_negative_squared_distance
distance_weight.temperature: 9.0
distance_weight.learnable: true
message_dim: 128
edge_hidden: 128
hidden_channels: 128
num_layers: 2
use_voxel_logits: true
use_candidate_voxel_feat: true
use_anchor_point_feat: true
use_anchor_atom_feat: true
use_anchor_voxel_feat: true
use_relative_coords: true
use_candidate_class_embedding: false
```

loss 相关：

```yaml
voxel_ligand_loss:
  hard_label_threshold: 1.7
  focal_gamma: 2.0
  w_focal: 0.7
  w_tversky: 0.3
  tversky_alpha: 0.5
  tversky_beta: 0.5

ligand_sparse_refine_loss:
  hard_label_threshold: 1.7
  focal_gamma: 0.0
  w_focal: 0.3
  w_tversky: 0.7
  tversky_alpha: 0.5
  tversky_beta: 0.5

ligand_sparse_refine_loss_schedule:
  mode: linear_warmup
  start_weight: 0.0
  final_weight: 1.0
  start_on_ratio: 0.2
  warmup_ratio: 0.3
```

按实际日志估计每 epoch 约 `678` optimizer steps，总步数约 `13560`。因此 sparse refine loss 在约 `2712` steps 前为硬 0，大致到 epoch 3/4 才开始；约 `4068` steps，即 epoch 5/6 左右，才到 full weight。这个时间点与 metrics 中 epoch 0-3 refined 完全等于 unrefined、epoch 4 起出现微小差异相吻合。

## 3. 指标证据

### 3.1 job297981 epoch 走势

`job297981` 的关键现象如下：

| epoch | C-local refined-unrefined F1 delta | global score refined-unrefined F1 delta | 备注 |
|---:|---:|---:|---|
| 0-3 | `0` | `0` | residual zero-init + refine loss 尚未真正生效，refined 等于 unrefined |
| 4 | `+8.77e-5` | `-9.67e-5` | 刚开始非零 |
| 5 | `+3.60e-4` | `+4.91e-5` | C-local 最明显微优 |
| 6 | `-1.56e-5` | `+1.72e-4` | global delta 最好，但仍极小 |
| 9 | `+7.998e-5` | `-1.21e-4` | best refined-score checkpoint |
| 12 | `+1.93e-4` | `-8.48e-5` | latest/last 附近 |

epoch 9 的代表值：

```text
val_score/global/refined_F1   = 0.52386862
val_score/global/unrefined_F1 = 0.52398998
val_refined/global/F1         = 0.77785158
val_unrefined/global/F1       = 0.77777159
val_capped/global/F1          = 0.47720796
val_uncapped/global/best_F1   = 0.66817969
num_C                         = 4270.09
num_P                         = 633.11
```

epoch 12/latest 的代表值：

```text
val_score/global/refined_F1   = 0.52195674
val_score/global/unrefined_F1 = 0.52204150
val_refined/global/F1         = 0.78146422
val_unrefined/global/F1       = 0.78127110
val_capped/global/F1          = 0.47366667
val_uncapped/global/best_F1   = 0.66594803
num_C                         = 4216.43
num_P                         = 639.93
```

这个表明：refine 的实际影响量级非常小。它在 C 内可以略微提高 precision，但也常伴随 recall 轻微下降；global score 对这种阈值/校准层面的漂移很敏感，所以 C-local 微优不必然转化为 global 微优。

### 3.2 与旧 MINI runs 的最关键差异

旧 `MINI_real_front` 的代表 run `292668`：

```text
warmup_topc_per_class          = [25000]
adaptive_expand_factor         = [7.0]
max_candidate_voxels_per_class = [25000]
max_anchors_per_class          = [1024]

best refined-score epoch       = 9
val_capped/global/F1           = 0.37013
val_capped/global/recall       = 0.75049
val_unrefined/global/F1        = 0.73817
val_refined/global/F1          = 0.73540
val_score/global/unrefined_F1  = 0.63699
val_score/global/refined_F1    = 0.63520
num_C                          = 12599.67
num_P                          = 891.02
```

新 `297981`：

```text
warmup_topc_per_class          = [10000]
adaptive_expand_factor         = [2.5]
max_candidate_voxels_per_class = [10000]
max_anchors_per_class          = [800]

best refined-score epoch       = 9
val_capped/global/F1           = 0.47721
val_capped/global/recall       = 0.48567
val_unrefined/global/F1        = 0.77777
val_refined/global/F1          = 0.77785
val_score/global/unrefined_F1  = 0.52399
val_score/global/refined_F1    = 0.52387
num_C                          = 4270.09
num_P                          = 633.11
```

解释：

- 旧 run 的 refine loss 让 C-local F1 明显下降，说明旧 loss 确实有问题。
- 新 run 的 refine loss 已经把 C-local regression 压到几乎消失，甚至略微正向。
- 但新 run 的 C 规模和 C 召回显著变低，global score 被候选集召回上限强烈限制。
- 因此“换 loss 后 global 仍吃亏”不能单独归因于 loss；更大的嫌疑是 candidate budget / adaptive threshold 策略变化。

### 3.3 dense oracle、capped、score 之间的缺口

`297981` epoch 9 有一个很刺眼的差距：

```text
val_uncapped/global/best_F1 ≈ 0.668
val_capped/global/F1       ≈ 0.477
val_score/global/F1        ≈ 0.524
```

这说明 dense logits 本身仍有相当高的 oracle threshold potential，但候选抽样/端到端路径没有把这个潜力完整带进来。换句话说，sparse refine head 现在是在一个已经被大幅截断的 C 上工作；C 外的真实正例不可能被 refine 恢复。

## 4. 代码机制证据

本次对照了 server `src_snapshot` 和本地关键文件哈希，以下关键 sparse refine 代码与本地一致。

### 4.1 Candidate set 是硬瓶颈

`src/model/sparse_refine/candidate_set.py` 中 `adaptive_threshold` 的核心逻辑是：

```text
n_best_box      = count(prob_valid > p_best)
target_before_cap = ceil(n_best_box * adaptive_expand_factor[class])
target_count    = min(target_before_cap, max_candidate_voxels_per_class[class], N_valid)
```

也就是说，`p_best` 决定 best-F1 阈值以上有多少候选，`adaptive_expand_factor` 决定围绕这些候选扩张多少倍，`max_candidate_voxels_per_class` 再做硬 cap。把 `adaptive_expand_factor` 从 `7.0` 降到 `2.5`，并把 max C 从 `25000` 降到 `10000`，理论上就会明显降低 C 的覆盖面；日志中的 `num_C` 和 `val_capped/global/recall` 正好证实了这一点。

### 4.2 refine loss 不会修 dense logits 或候选召回

`src/model/stage1_model.py` 中 `_run_sparse_refine_head` 受 `detach_voxel_into_refine` 控制。`297981` 配置为：

```yaml
detach_voxel_into_refine: true
```

对应效果是 refine 读到的 base logits、C/P voxel feature 都 detach。再加上 candidate logits 本身来自 candidate builder 输出的 detached logits，sparse refine loss 的梯度不会回传去改善 dense voxel logits，也不会改善下一轮 C 的召回。它只是在已选出的 C 内学习一个 refined logit。

这个设计对隔离训练很干净，但也意味着如果 global 问题来自 C 外漏召，refine head 天生无能为力。

需要补记一条实验事实：这里的 detach 不是可随意取消的设计开关。之前已经做过实验，去掉 detach、让 sparse refine 梯度回灌 dense voxel logits / voxel feature，会让**所有损失项都震荡**、训练不稳定。因此“放开 detach 改善 C 外召回”应被降级为高风险选项（与 §6.7 一致），而不是优先项。

更值得投入的关键决策其实是另一条：refine head 到底应该**输出“相对 base 的残差”（`mode=residual`，当前配置），还是直接学习“原始概率/绝对 logit”（`mode=direct`，代码已支持）**。这关系到 head 的优化地形与“是否敢改 base”，影响比要不要 detach 更重，值得专门搜索文献。文献上对“残差式迭代细化”基本是正面的：[Jastrzębski et al., 2017](https://arxiv.org/abs/1710.04773) 指出残差连接会自然诱导“逐步迭代式推断（iterative inference）”，每一步只学一个小修正量，优化更稳、更容易保住已学到的解；在光流/深度/位姿/时序等多步细化任务里，残差/增量预测通常比直接重预测更稳、误差累积更小（参见 [Loop-Residual NN, 2024](https://arxiv.org/abs/2409.14199)）。但残差也有代价：当 `zero_init_residual=true` 且 base logits 又作为强输入/强残差基线时，head 的“全局最优”很容易就是“delta≈0、原样复制 base”，这正是当前 `1e-4` 量级“几乎不动”的来源（见 §5.3）。`direct` 模式不把 base 当锚点，理论上更可能学出与 base 不同的概率，但也更容易破坏 base、更依赖 §6.3 的 positive-preservation 这类约束兜底。因此这一点不能拍脑袋，建议作为一组独立对照实验（residual vs direct，各自配 §6.3 的 preservation/ranking 约束）专门验证，而不是和 budget/anchor 实验混在一起。

### 4.3 refined/global 与 score/global 不是同一个分母

`src/wrappers/voxel_point_stage1_diagnostics.py` 中 C-local 和 score 的区别很重要：

```text
val_unrefined/global/F1, val_refined/global/F1:
  在 C 内统计，recall denominator 是 num_gt_in_C。

val_score/global/{unrefined,refined}_F1:
  端到端统计，recall denominator 是 dense 全空间 GT 正例数。
```

因此 `val_refined/global/F1 > val_unrefined/global/F1` 只说明 refine 在进入 C 的正例/负例上排序略好；它并不保证 dense 全空间的端到端 F1 变好。只要 C 本身漏掉大量 GT，global score 就可能仍然低。

### 4.4 residual zero-init 让 head 初始为 identity

`src/model/sparse_refine/sparse_refine_head.py` 中 residual mode 且 `zero_init_residual=true` 时，最后一层 residual delta 初始化为 0，因此初始 refined logits 等于 base logits。这解释了 epoch 0-3 refined 完全等于 unrefined。

该设计本身合理，能避免一开始破坏 base。但配合 `start_on_ratio=0.2`、`warmup_ratio=0.3` 和训练段只产出到 epoch 12，会让 refine head 的有效训练窗口偏短；如果 residual delta 又被 loss/feature detach/弱 message 压得很小，最终就会表现为“几乎不做事”。

### 4.5 P -> C message 的表达可能偏弱

当前 head 对每个 C 点只从同类 P anchors 中取 `num_neighbors=4`，距离权重为 negative squared distance softmax，再乘 learned sigmoid gate：

```text
message = sum(distance_softmax_weight * learned_gate * neighbor_content)
```

注意 learned gate 乘上去后没有再 normalize。这样 gate 除了改变相对邻居权重，也会整体缩小 message magnitude。若 residual zero-init、base logits 又作为强输入/强残差，head 很容易学成“保守地不改 base”。

anchor sampler 用的是 `unweighted_fps`，它更像几何覆盖而不是“错误/不确定区域覆盖”。单看 sparse refine 这一个目标，PointRend 式的“困难点/不确定点覆盖”通常更有价值（见 §5.4）。

但这里有一个被刻意保留的、跨阶段的关键约束，结论是**暂不改 anchor 采样**：P anchors 不只是 stage1 refine 的工具点，它们就是要被 stage2 复用的虚拟原子（pseudo atoms，参见 `src/model/pseudo_atoms.py` 的注入路径）。计划中的 stage2 要模仿 Emap2lig（AF3 式 diffusion：从原点附近高斯噪声起步、以 trunk 表征为条件迭代去噪，参见 [AlphaFold 3, Abramson et al., Nature 2024](https://www.nature.com/articles/s41586-024-07487-w) / [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)），并把这些虚拟原子直接纳入扩散。这就产生了**同一组 P anchors 的目标冲突**：

- 作为 refine 工具点，P 希望偏向“模型最容易错的地方”（高不确定、边界、疑似漏召），信息增益最大。
- 作为 diffusion 的先验/锚点，P 更希望是对 ligand 真实占据区域的**忠实、均匀覆盖**——一个偏向“模型不确定区”的点集，等于把 stage1 的不确定性当成空间先验喂给 diffusion，用“我不确定的地方”去定义“分子应该在哪”，会污染 diffusion 的空间先验，反而不利。

因此你倾向拒绝把 anchor 改成困难点覆盖，是有道理的：在 P 被 diffusion 复用的前提下，`unweighted_fps` 这种几何覆盖恰恰是更安全的先验。本文据此把这条从“应改的缺陷”降级为“受 stage2 约束的有意选择”。真正干净的解法是**解耦两种角色**：保留 `unweighted_fps`（或概率/几何混合）作为喂给 diffusion 的 P；若后续确认 sparse refine 确实需要困难点信号，再为 refine 单独引入一组**不进 diffusion**的辅助困难点，或在 P→C 消息里用 per-edge 的不确定度特征补偿，而不是把 diffusion 锚点整体改成困难点采样。这条“关键问题”应在 stage2 设计定稿前先定下来，§6.4 的 hybrid anchor 建议据此降级为受约束的可选项。

## 5. 失败原因排序

### 5.1 第一嫌疑：candidate budget/recall 改变，global 被 C 锁死

这是证据最强的原因。

两阶段检测/分割类系统的共同约束是：第二阶段只能处理第一阶段给出的候选。Faster R-CNN/RPN 的基本思想也是先生成高质量 proposals，再由后续检测头处理；如果 proposal 阶段没有覆盖目标，后续 head 无法凭空恢复。`297981` 的 refine head 也一样，只能在 `C` 内改 logits。

`297981` 的 C 比旧 `292668` 小约 3 倍，`val_capped/global/recall` 从约 `0.7505` 变为约 `0.4857`。这足以解释 global score 明显低于旧设置，也足以解释为何 C-local 微优不能变成 global 优势。

### 5.2 第二嫌疑：训练目标和监控目标不完全一致

sparse refine loss 是在 C 上做 hard-label threshold supervision，优化的是 C 内候选体素的分类损失；monitor 是 `val_score/global/refined_F1`，它用 dense 全空间 GT 分母、histogram threshold 和端到端 precision/recall。

这种 objective mismatch 会导致：

- loss 可以让 C 内分类更好，但不增加 C 外召回。
- loss 可以提高 precision，同时轻微降低 recall；C-local F1 仍可持平，但 global F1 可能变差。
- refined logits 的 calibration 改变后，histogram best threshold 会迁移；在极度不平衡任务中，PR/F1 对 score distribution 很敏感。

### 5.3 第三嫌疑：refine head 被设计成“很难破坏 base”，但也因此很难产生实际收益

`residual + zero_init + use_voxel_logits + detach_voxel_into_refine + delayed schedule` 是非常保守的组合。它适合防止 regression，但如果没有强监督信号或明确的 delta/ranking objective，就可能停留在接近 identity 的局部最优。

从 metrics 看，refined 与 unrefined 的差异量级多数是 `1e-4`，这更像“head 几乎没动”，而不是“head 学到了稳定新能力但 global 被某个小 bug 抵消”。

**这一条是真正的主决策点。** §5.1/§5.2 的 candidate recall 决定了 refine 的“天花板”在哪，但即使把 C recall 修好，只要 head 仍是这套“极保守组合”，它大概率还是停在 identity 附近、把天花板让出去。组合里最该被单独拎出来做对照的是 `mode=residual` 这一项：到底继续“学相对 base 的残差”，还是改成 `direct` 直接学原始概率（详见 §4.2 末尾的残差 vs 直接预测讨论与文献）。`zero_init` / `delayed schedule` / `detach` 都是围绕“残差”这个核心假设的保护措施；如果核心假设换成 `direct`，这些保护项的取舍也要随之重估。换句话说：调 budget/anchor 是“把天花板抬高”，而定下 residual-vs-direct ＋ 配套监督（§6.3）才是“让 head 真正够得着天花板”。

### 5.4 第四嫌疑：anchor/P 采样没有对准 sparse refine 的价值点

Point-wise refinement 的典型价值在于把计算集中到困难点、边界点、不确定点。PointRend 的经验就是自适应选择需要细化的位置，而不是平均地处理所有位置。当前 `unweighted_fps` 更强调几何覆盖；`num_neighbors=4` 和 `max_P=800` 在 C 已缩小的情况下可能进一步削弱对困难区域的信号。

若 P anchors 不覆盖 dense head 最容易犯错的位置，C 点收到的 message 就很难包含“为什么 base logit 该被修正”的信息。

### 5.5 第五嫌疑：precision/recall 校准漂移

`297981` 中 refined 经常表现为 precision 略升、recall 略降。Tversky/Focal 这类 loss 适合处理不平衡，但不同权重会改变 precision-recall tradeoff。当前 sparse refine loss 的 Tversky `alpha=0.5,beta=0.5` 等价于对 FP/FN 对称；既然 global 的痛点是漏召，**正式实现里把 sparse refine 的 Tversky 改成 `alpha=0.3,beta=0.7`，让 FN 更贵以加强 recall**，并与 §6.3 的 positive-preservation 约束一起托底召回。

现代神经网络概率校准本来就不可靠，refined head 加上 residual delta 后也可能改变 logit calibration。局部 F1 微优而 global threshold score 微负，符合这种“排序/校准改善很小且不稳定”的模式。

### 5.6 第六嫌疑：训练段没有完整跑满 max_epochs

配置是 `max_epochs=20`，但该训练目录只看到 `epoch_000000` 到 `epoch_000012`。同时 Slurm 输出显示训练段被 kill lock 终止，之后 job id 被复用。这不一定是主要失败原因，因为 epoch 9/10/12 已经有 top checkpoints，但它让“head 是否需要更长时间才起作用”这个问题无法从当前 run 排除。

## 6. 如何提升 sparse refine head 的实际作用

### 6.1 先恢复候选召回，再评估 refine head

第一组实验应只改 candidate budget，保持 `297981` 的新 loss 不变：

```yaml
warmup_topc_per_class: [25000]
adaptive_expand_factor: [7.0]
max_candidate_voxels_per_class: [25000]
max_anchors_per_class: [1024]
```

这是最重要的控制变量。它能回答：

- 新 loss 是否真的解决了旧 C-local regression。
- global absolute regression 是否主要来自 candidate budget。
- 当 C recall 恢复后，sparse refine 是否有机会贡献正向 delta。

可加一个中间预算版本，避免一次回到太重：

```yaml
warmup_topc_per_class: [15000] 或 [20000]
adaptive_expand_factor: [4.0] 或 [5.0]
max_candidate_voxels_per_class: [15000] 或 [20000]
max_anchors_per_class: [1024]
```

### 6.2 把 C recall 当作一等分析量（不新增日志）

明确：这一节不引入任何新的日志或记录字段，也不要为这一点改动任何代码——纯离线分析。所需信号当前 validation diagnostics 已经全部产出——`val_capped/global/{F1,recall,precision}`、`val_uncapped_best/global/best_F1`、`val_score/global/{unrefined,refined}_F1`、`val_{unrefined,refined}/global/F1`、`num_C`、`num_P` 等都已有，只是分析时要主动去读，而不是只盯 `monitor_metric`。

唯一的要点是认知层面的：只看 `val_score/global/refined_F1` 容易把三件事混在一起——dense oracle 能力、candidate recall、refine 在 C 内的重排序。分析时（离线表格里，不入训练日志）把这三层拆开看即可：用 `val_uncapped_best/global/best_F1` 看 dense 上限，用 `val_capped/global/recall` 看候选召回，用 `refined−unrefined` 的 C-local 差看 head 净效果。两个缺口（oracle−capped、capped−score）直接用现有字段相减得到，无需落盘任何新指标。

### 6.3 给 sparse refine 一个“相对 base 的改进目标”（重点）

这是本轮最关键的设计决策。当前 `ligand_sparse_refine_loss` 只是 `AdaptiveClassificationCompositeLoss`（focal+tversky）在 C 上的逐点分类，没有任何“相对 base 更好”的显式信号，这与 §5.3 “head 停在 identity”互为因果。拟定的目标损失为四项加权（设计取舍详见 `CLAUDE/docs/损失建议.md` 的两条建议）：

```text
L_refine = w_focal·Focal(默认 w_focal=0)
         + w_dice ·Dice/Tversky
         + w_pos  ·L_pos_keep      # positive preservation
         + w_rank ·L_rank          # pairwise ranking
```

focal 默认权重 0（保留接口即可；当前 `focal_gamma=0.0` 本就退化为加权 BCE，留 0 让 dice 主导），dice/Tversky 作为主分类项保留。下面逐条回答设计问题。

**(1) Positive preservation loss**

形式（logit 域 hinge）：

```text
L_pos_keep = mean_{i∈P_keep} max(0, base_logit_i − refined_logit_i + m_pos)
```

它只惩罚“把 base 已经判对的正例往下压”，直接对症 §3.1 里 refined 普遍略掉 recall 的现象，并与 tversky 的 FN 项互补：tversky 是绝对意义“别漏正例”，preservation 是相对意义“别把 base 已经找到的正例弄丢”，锚在 base 上、梯度方向更明确。

- **可信正例集合 P_keep 怎么选（已确认）**：复用验证集记录的 best-F1 阈值，而不是新调一个 `τ_base+`。理由是它工程上已经存在——wrapper 里 `_cached_voxel_ligand_p_best_by_class`（来自 `val_uncapped_best/global/p_best`）每个验证 epoch 自动更新并同步给 backbone，正是 `candidate_set.py` 生成 C 所用的同一个 `p_best`。用它定义 `P_keep = {i∈C : target_i=正 且 base_prob_i > p_best}`，既不引入新超参，又让“候选选择”和“损失监督”共用同一条操作阈值，语义自洽。代价是 `p_best` 每 epoch 才更新一次、滞后一个 epoch；但 refine loss 要到 epoch 3-5 才升权，那时 `p_best` 已稳定，滞后可接受。topk-ratio（每个 BOX 取 base_prob 前 50% 的正例）是备选：逐样本自适应、对校准漂移更稳，但多一个 ratio 超参且把损失耦合到 per-sample 统计。建议主用 `p_best`，topk-ratio 作为 `p_best` 不稳时的回退。

- **要不要加对称的 Negative preservation？（已确认先不加）**：positive preservation 保护的是 recall（别压低可信正例），对称的 negative preservation 会保护“别抬高可信负例”，即进一步把 head 钉向 base——但 §5.3 的核心病灶恰恰是 head 太像 identity，再加一个把它往 base 拉的对称约束，是在和目标对着干。而且“别造 FP”这件事 tversky 的 `α·FP` 项已经在管，negative preservation 与之部分冗余。这里的不对称是有意的：正例侧没有“相对 base 不退步”的现成约束才需要 preservation 补，负例侧已有 tversky 兜底。只有当后续诊断显示 refined 抬高了大量 FP、precision 崩了，再考虑补一个轻量 negative preservation。

**(2) Pairwise ranking loss**

采用 `损失建议.md` 推荐的版本 B（绝对排序，先稳）：

```text
L_rank = mean_{(i,j)∈Q_C} max(0, m_rank − refined_i + refined_j)
```

- **采样集 Q_C（已确认采用）**：取 `损失建议.md` 给出的 `Q_C = P_C × TopK_{j∈N_C}(base_prob_j)`，即“所有正例 × base 最容易误判成 ligand 的那批硬负例”。正例本就稀缺，不再下采样（全取）；负例只取 per-sample 的 TopK 硬负例。
- **hard 用阈值还是 topk-ratio**：负例侧推荐 **topk-ratio（per-sample TopK）**，不推荐用 best-F1 阈值。因为“硬负例”本质是逐样本相对概念（这个 BOX 里分最高的一批负例）；用全局阈值 `τ_neg` 会让每个样本命中的负例数剧烈波动（有的 BOX 一大片高分负例、有的几乎没有），pair 数和梯度尺度都不稳，TopK 则给每样本固定 pair 预算、更稳。这与 (1) 里 positive preservation 用 `p_best` 阈值并不矛盾：两项损失失效模式不同，preservation 关心“是否过操作阈值”，ranking 关心“相对排序谁更靠前”，各用最适合的 hardness 定义即可，不必强行统一。ranking 直接对齐 AP/PR-AUC 的优化思路，在极不平衡检测里通常优于纯逐点分类（参见 [Chen et al., AP-Loss, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf)）。

**(3) 关于 m_pos 与 m_rank：强制 0 / 各自可调 / 绑同值？**

**已确认采用**：各自独立可调、默认值不同，不绑成同一个值——两者活在不同的损失几何里：

- `m_pos`：preservation 是“锚在 base 上别下滑”。默认取 `m_pos=0` 最干净，含义是“refined 不低于 base 即可，不强求严格超过”。注意按上式 `m_pos>0` 其实是在要求 refined 至少比 base 高出 `m_pos`（变成“强制改进”而非“保持”），会重新引入激进改动和不稳定，所以 preservation 的 margin 不该取正大值，0 或极小为宜。
- `m_rank`：ranking 必须有正 margin，否则 `refined_i` 只要比 `refined_j` 高一点点 hinge 就饱和、没梯度，正负分不开。建议 `m_rank` 取 logit 域的小正值（如 0.5~1.0）并可调。
- 因此两者数值不同、语义不同，绑定同值没有道理；强制都为 0 则 ranking 失去作用。结论：`m_pos` 默认 0（保持项），`m_rank` 默认 0.5 左右（分离项），各自独立调。
- 进一步的“相对 base 改进 margin”（`损失建议.md` 的版本 A，要求 refine 的正负间隔比 base 拉得更开）更激进，建议先跑稳版本 B 与 preservation，确认不再 regression 后再视 PR-AUC 决定是否升级。

**(4) 实验方法：一个主实验 + 若干针对性消融**

不采用“逐项叠加”的渐进顺序，而是按你的方法论：**主实验一次性上齐我们猜测最有利的全部参数/损失**（dice + positive-preservation + pairwise-ranking，配 §6.7 的公共底座、§5.5 的 recall-biased Tversky `0.3/0.7`、§1 的 budget `5.0/25000/1024`、§6.6 的早 schedule）；**消融只针对主实验里我们最不确定的机制**，逐个关掉或替换，看它是否真的 work。权重起点 `w_pos≈0.05~0.1`、`w_rank≈0.01~0.05`，`m_pos=0`、`m_rank≈0.5`，focal 维持默认 0。具体主实验配置与消融清单见 §7；本节是“让 head 够得着天花板”的核心，必须与 §4.2/§5.3 的 residual-vs-direct 决策一并设计，而不是只调权重。

### 6.4 改 anchor/P 采样：从几何覆盖转向困难点覆盖（受 stage2 约束，降级为可选）

前置约束：按 §4.5 的决定，P anchors 会被 stage2 的 Emap2lig 式 diffusion 直接复用为虚拟原子，困难点采样可能污染 diffusion 的空间先验。因此本节整体降级为“受约束的可选项”——只有在确认 P 不复用、或为 refine 单独引入不进 diffusion 的辅助困难点时才适用，不要直接把喂给 diffusion 的 P 改成困难点采样。

据此，**sampler 不改**：继续用 `unweighted_fps`，不引入 `topk_nms`/hybrid，也不让 P 去包含 near-threshold、C 外临界点等“困难点/边界信号”（这些与 stage2 diffusion 复用相冲突，见 §4.5）。只调 P 的规模这两项：

- `max_anchors_per_class`: `800 -> 1024`
- `anchor_to_candidate.num_neighbors`: `4 -> 8`

即：保留几何覆盖式采样，只是让每个 BOX 的 P 更多、每个 C 连到的 P 邻居更多，给 refine 更充裕的消息带宽，而不改变 P 的语义。

### 6.5 让 message attention 更稳定

当前权重是：

```text
softmax(-dist^2 / T) * sigmoid(edge_gate)
```

这个形式可能让 gate 改变总体 message magnitude。可以试：

```text
attention_score = -dist^2 / T + edge_gate_logit
attention_weight = softmax(attention_score)
```

或者在乘 gate 后重新 normalize：

```text
w = softmax_distance * sigmoid_gate
w = w / sum(w)
```

这样 gate 更像“邻居间重分配”，不容易把 message 整体压小。

### 6.6 更早训练 head，但保留小步保护

当前 schedule 偏晚：

```yaml
start_on_ratio: 0.2
warmup_ratio: 0.3
```

候选有两组，**采用后一组**（更温和的提前）：

```yaml
start_on_ratio: 0.05
warmup_ratio: 0.15
```

```yaml
start_on_ratio: 0.1   # ← 采用这一组
warmup_ratio: 0.2
```

为避免早期破坏 base，可以同时增加 residual scale gate：

```text
refined = base + sigmoid(g) * delta
g 初始化为负值，使初始 scale 很小
```

这样比完全延后训练更可控：head 从早期就接触数据，但初始改动仍小。

### 6.7 主实验与消融的公共配置

下面是主实验和所有消融共享的固定底座（除非某条消融明确改动其中一项）：

1. **detach：全量 detach（固定）。** 上一版“逐步放开 detach 让 refine 影响候选生成”的建议作废——已有实验证明放开 detach 会让所有损失震荡（见 §4.2）。因此主实验和消融一律保持全量 detach，不把它当可调项。
2. **embed head：上 `configs/model/embed_head/point_clipbig.yaml`。** 但体素分支仍要投影原子特征，且需要改代码：把当前的三线性插值（有不良几何偏置）换成对 3×3×3 BOX 做距离加权投影，同时仍然附加 2 维“位置编码”。
3. **detach/residual 接线：以 `configs/model/detach_residual/max_detach.yaml` 为底座。** 但其中 `sparse_refine_residual_mode: residual` 不固定，要作为小消融项（residual vs direct，见 §4.2/§5.3）。
4. **（需另行讨论）P 自身是否引入监督。** 例如监督 P 所在体素是否落在 ligand 区域；若引入，可与受体原子监督合并、同一套处理。这是尚未定的关键改动，先标记，落盘计划前单独定。
5. **其余改动（如 CPC 本身的参数）见此前评论，不在此重复。**

## 7. v3 实验方案：主实验 + 消融

整体结构由 §6.3(4) 的方法论决定：一个主实验上齐全部“猜测最有利”的配置，消融只逐项打掉主实验里我们最不确定的机制。

### 7.1 主实验（上齐全部“猜测最有利”配置）

公共底座见 §6.7（全量 detach、`point_clipbig` embed head + 3×3×3 距离加权投影 + 2 维位置编码、`max_detach.yaml` 接线）。主实验在此之上一次性启用：

```text
# candidate budget（§1）
adaptive_expand_factor: 5.0
max_candidate_voxels_per_class: 25000
warmup_topc_per_class: 25000
max_anchors_per_class: 1024
anchor_to_candidate.num_neighbors: 8
anchor_sampler: unweighted_fps        # 不改（§4.5）

# sparse refine loss（§6.3）
w_focal: 0                             # 默认关
Tversky: alpha=0.3, beta=0.7           # recall-biased（§5.5）
+ positive-preservation: m_pos=0, P_keep 用验证集 p_best
+ pairwise-ranking:      m_rank≈0.5, Q_C=所有正例 × TopK 硬负例(per-sample)
w_pos≈0.05~0.1, w_rank≈0.01~0.05

# head / 接线
sparse_refine_residual_mode: residual  # 主实验用 residual；direct 留给消融
detach: 全量 detach（§6.7，不动）

# schedule（§6.6）
start_on_ratio: 0.1
warmup_ratio: 0.2
```

### 7.2 消融（只打我们最不确定的机制）

每条消融 = 主实验改一项、其余不动：

- A. **residual vs direct**：`sparse_refine_residual_mode: residual → direct`。验证 §4.2/§5.3 的主决策点。
- B. **pairwise-ranking 有无**：去掉 `w_rank`。验证 ranking 是否真把 C-local 排序变好。
- C. **positive-preservation 有无**：去掉 `w_pos`。验证它是否止住“普遍略掉 recall”。
- D. **recall-biased Tversky**：`0.3/0.7 → 0.5/0.5`。验证偏 recall 权重的净效果。
- E. **budget 敏感性**：`expand_factor 5.0 → 2.5`（回到 297981）。验证 C recall 对 global 的贡献量级。
- F.（可选，待定）**P 监督有无**：见 §6.7 第 4 点，定了再加。

判读口径见 §6.2：每条都看 `val_capped/global/recall`、`refined−unrefined` 的 C-local 差、global score delta 三层。

## 8. 当前最可能的判断

一句话版：

```text
297981 的新 loss 已经把 sparse refine 在 C 内的负作用压到几乎没有；
但 candidate set 被收得太小，global 召回上限大幅下降；
而 refine head 又被 detach/residual/zero-init/晚 warmup 设计成只能在 C 内做很小的保守修正，
所以它没有足够通道把 C-local 微优转成 global F1 优势。
```

最值得优先验证的不是再换一个复杂 head，而是：

1. 同新 loss 恢复旧 candidate budget。
2. 用 `val_capped/global/recall` 和 `num_gt_in_C/dense_gt` 确认 C recall。
3. 在 C recall 恢复后，再判断 refine head 是否仍只是 identity。

## 9. 本次本地中间材料

中间材料位于：

```text
D:\OneDrive\My_Project\Pocket_Plus\tmp\sparse_refine_f1_regression_v2\
```

主要文件：

```text
remote_job297981_dump.txt
remote_metrics_tables.txt
remote_deep_file_and_curve_scan.txt
remote_slurm_job297981.txt
job297981_config.yaml
job297981_train.yaml
job297981_config_facts.json
remote_mini_comparison_best_ref_score.txt
remote_config_compare_grep.txt
remote_sparse_refine_task_compare.txt
```

这些文件是本次诊断的原始摘录/表格来源。

## 10. 参考资料

- Ren et al., Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497. https://arxiv.org/abs/1506.01497
- Lin et al., Focal Loss for Dense Object Detection, arXiv:1708.02002. https://arxiv.org/abs/1708.02002
- Salehi et al., Tversky loss function for image segmentation using 3D fully convolutional deep networks, arXiv:1706.05721. https://arxiv.org/abs/1706.05721
- Sudre et al., Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations, arXiv:1707.03237. https://arxiv.org/abs/1707.03237
- Davis and Goadrich, The Relationship Between Precision-Recall and ROC Curves, ICML 2006. https://dl.acm.org/doi/10.1145/1143844.1143874
- Guo et al., On Calibration of Modern Neural Networks, arXiv:1706.04599. https://arxiv.org/abs/1706.04599
- Kirillov et al., PointRend: Image Segmentation as Rendering, arXiv:1912.08193. https://arxiv.org/abs/1912.08193
- Cai and Vasconcelos, Cascade R-CNN: Delving into High Quality Object Detection, arXiv:1712.00726. https://arxiv.org/abs/1712.00726
- Qi et al., PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space, arXiv:1706.02413. https://arxiv.org/abs/1706.02413
- Wang et al., Dynamic Graph CNN for Learning on Point Clouds, arXiv:1801.07829. https://arxiv.org/abs/1801.07829
- Chen et al., Towards Accurate One-Stage Object Detection with AP-Loss, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf
- Greff et al., Highway and Residual Networks learn Unrolled Iterative Estimation, arXiv:1612.07771（残差连接诱导 iterative inference 的代表论述；另见 Jastrzębski et al., arXiv:1710.04773）. https://arxiv.org/abs/1710.04773
- Ng, Loop-Residual Neural Networks for Iterative Refinement, arXiv:2409.14199. https://arxiv.org/abs/2409.14199
- Abramson et al., Accurate structure prediction of biomolecular interactions with AlphaFold 3, Nature 2024. https://www.nature.com/articles/s41586-024-07487-w
