# 最后重构代码 v3：轻量最终分类头、冻参再训与两轮消融计划

## 0. 总目标

本轮重构的目标是把 Stage1 的最终判别路径收束成一条更短、更容易解释、更适合从最优 checkpoint 冻参增量训练的路径：

```text
voxel backbone
  -> point backbone / voxel-point fusion
  -> final real/P 单向 cross-attention 信息交换
  -> real_atom_head / pseudo_atom_head
  -> sparse refine head
```

这里的“移除 atom head”只指移除额外的 `Stage1SerializedAttentionStack` 这类重 attention block，以及移除 `front/back` 双分类头体系；不是把所有与真实原子/P anchor 分类有关的机制都删掉。分类前的单向 cross-attention 信息交换、MLP 分类尾部、prior 初始化、零初始化增量等有益机制要保留并前移到最终分类头路径中。

新主线只保留一套 real 分类头和一套 P 分类头：

- `real_atom_head`: 真实原子最终分类头。
- `pseudo_atom_head`: P anchor ligand 区域归属最终分类头。

不再出现“前置/后置”分类头，不再使用 `atom_logits_front`、`pseudo_logits_front`、`atom_loss_front_weight`、`pseudo_loss_front_weight` 这套语义。

## 1. 结构重构原则

### 1.1 删除的东西

必须从主线代码、配置、日志与文档中彻底清理：

- `front/back` 对偶命名体系。
- `enable_atom_head_front` / `enable_atom_head_back`。
- `atom_logits_front` / `pseudo_logits_front`。
- `atom_loss_front_weight` / `atom_loss_back_weight` 的双头解释。
- `pseudo_loss_front_weight` / `pseudo_loss_back_weight` 的双头解释。
- `atom_tokens` / `atom_hidden` 作为训练主路径输出。
- `P_atom_head_feat` 作为 sparse refine head 的标准输入名。
- `Stage1SerializedAttentionStack` 作为 final classification 前的额外重 block。

不保留历史兼容字段。配置、wrapper、日志和注释要一次性切到新语义；如果旧实验需要复现，应从 git 历史或独立 legacy 文件恢复，而不是让新主线长期背兼容分支。

### 1.2 保留并前移的东西

必须保留：

- final recycle 才注入 P anchors 的 mixed layout。
- real/P typed point 分参能力。
- P anchor 的 ligand 区域归属监督。
- real atom 的分类监督。
- P/C candidate 与 sparse refine 框架。
- `detach_pseudo_point_feat_into_refine`、`detach_voxel_into_refine` 等最大 detach 思想。
- 分类前单向 cross-attention 信息交换。
- 所有新增 residual / interaction / refine 输出增量的零初始化。
- candidate threshold cache 与 CPC validation diagnostics。

### 1.3 文件归属

不新增 `src/model/final_point_heads.py`。直接重构 `src/model/stage1_atom_head.py`，让这个文件从“重 atom head”退化为“最终 real/P 分类头与 cross-attention 交互头”的实现文件。

建议保留的类/结构：

- `RealToPseudoCrossAttention`
- `PseudoToRealCrossAttention`
- `RealPseudoFinalClassificationHead`
- 可复用现有 `PseudoGeometricAggregationV2` 的实现思想：radius 建图、source detach、relative coord、可选 bind probability、scatter softmax、零初始化输出投影。

建议删除或不再由主线调用的结构：

- `Stage1SerializedAttentionStack`
- 依赖 PTV3 `Block` 的 final atom head attention stack。

## 2. 最终分类头设计

### 2.1 输入

最终分类头直接消费最后一轮 point backbone 的输出特征：

- real 槽位：`final_real_feat_before_interaction`
- P 槽位：`final_pseudo_feat_before_interaction`

这两个特征来自 `outputs["point_feat_raw"]` 按 `PseudoAtomLayout` 切分；如果需要接收 voxel aux base logit，仍可通过明确开关拼接或残差相加，但不要恢复 front/back 语义。

### 2.2 单向 cross-attention

分类前允许两个方向的信息交换：

1. `real -> pseudo`
   - P 为 query。
   - real 为 source/context。
   - source real feature 默认 detach。
   - 可吃 detached real bind probability。
   - 输出投影零初始化。

2. `pseudo -> real`
   - real 为 query。
   - P 为 source/context。
   - source P feature 默认 detach。
   - 输出投影零初始化。

由于 source 侧 detach，且本轮训练计划采用“旧主干冻结，只训新增小模块”的方式，两个方向同时打开理论上近似线性叠加，不需要拆成两个单独消融。消融时用一个配置同时关闭两个 interaction，检验分类前单向信息交换整体贡献即可。

推荐配置命名：

```yaml
enable_real_to_pseudo_interaction: true
enable_pseudo_to_real_interaction: true
interaction_radius: 4.0
interaction_max_neighbors: 48
interaction_num_heads: 4
interaction_detach_source_feat: true
```

零初始化不做可选开关，直接写进代码逻辑。

### 2.3 分类头

分类头命名固定为：

- `real_atom_head`
- `pseudo_atom_head`

结构：

```text
LayerNorm
  -> Linear
  -> activation
  -> Linear
  -> logits
```

要求：

- real head 输出 `atom_logits`。
- pseudo head 输出 `pseudo_logits`。
- 最后一层支持 prior bias 初始化。
- 若开启 receptor base logit 残差，末层零初始化，使初值等价于 base。
- 不再输出 `atom_logits_front` / `pseudo_logits_front`。

### 2.4 输出命名

最终 forward 输出要风格对称，方便 sparse refine、诊断和后续可视化读取。

建议输出：

```text
real_feat_before_interaction
pseudo_feat_before_interaction
real_feat_after_interaction
pseudo_feat_after_interaction
atom_logits
pseudo_logits
pseudo_voxel_zyx
pseudo_batch_index
```

如果 sparse refine 使用分类头前/后的 P 特征，字段名必须明确：

- `pseudo_feat_before_interaction`: 纯 final point feature。
- `pseudo_feat_after_interaction`: 经 real->P 信息交换后的 P feature。

不要再使用 `pseudo_feature` 作为默认字段名，避免暗示它来自旧 atom head。

## 3. Sparse refine head 重构

### 3.1 输入重命名

Sparse refine head 不再接收 `P_atom_head_feat`。新输入名：

- `P_final_point_feat`: P 的 final point backbone 特征。
- `P_interaction_feat`: P 经 real->P cross-attention 后的特征，可选。
- `P_voxel_backbone_feat`: P home voxel 的 `voxel_final` 特征。
- `C_voxel_backbone_feat`: C voxel 的 `voxel_final` 特征。
- `voxel_logits`: C 原始 ligand logits，可选。
- `candidate_class_embedding`: 可选。
- `relative_coords`: 可选。

配置 `inputs` 同步改名：

```yaml
inputs:
  use_C_voxel_backbone_feat: true
  use_P_final_point_feat: true
  use_P_interaction_feat: true
  use_P_voxel_backbone_feat: true
  use_relative_coords: true
  use_candidate_class_embedding: false
```

删除：

```yaml
use_P_atom_head_feat
```

构造注入参数同步改名：

- `P_atom_head_dim` -> `P_final_point_dim` 或 `P_interaction_dim`。

### 3.2 最大 detach 主线

主实验使用最大 detach：

- old voxel backbone 冻结。
- old point backbone 冻结。
- embed head 冻结。
- density cube encoder 按实验设定冻结或只训练新增 head 时冻结。
- refine 吃到的 old trunk 特征默认 detach。

因此不做“sparse refine 是否回灌 trunk”的消融。主实验的 sparse refine head 已经与 trunk 隔离，相关消融收益低且会扩大工程面。

### 3.3 零初始化

Sparse refine head 必须零初始化关键输出增量：

- residual 模式：输出 delta 的末层零初始化。
- direct 模式若采用 base + delta 的等价实现，也要让新增 delta 初值为 0。
- 若保留 direct logits head，则需要通过 prior bias 或 base-copy 逻辑保证初始行为稳定。

零初始化不是消融开关，直接作为实现契约。

## 4. 冻参数加载与训练策略

### 4.1 从最优 checkpoint 初始化

新增“初始化自 checkpoint，但不恢复训练状态”的入口：

```yaml
train:
  init_from_best_ckpt: null
  init_from_best_strict: false
```

语义：

- 只加载模型参数。
- 不恢复 optimizer。
- 不恢复 scheduler。
- 不恢复 global step。
- 允许旧 checkpoint 中的重 atom head 参数成为 unexpected / ignored。
- 允许新 `real_atom_head`、`pseudo_atom_head`、cross-attention、sparse refine 新字段为 missing 并重新初始化。

训练启动时必须打印：

- checkpoint 路径。
- loaded key 数量。
- missing key 分组。
- unexpected key 分组。
- trainable parameter count。
- frozen parameter count。
- trainable module name 摘要。

### 4.2 冻参配置

推荐配置：

```yaml
train:
  freeze:
    enabled: true
    trainable_name_patterns:
      - "sparse_refine_head"
      - "real_atom_head"
      - "pseudo_atom_head"
      - "real_to_pseudo"
      - "pseudo_to_real"
    frozen_name_patterns:
      - "voxel_backbone"
      - "point_backbone"
      - "embed_head"
      - "density_cube_encoder"
      - "real_density_cube_encoder"
```

默认只训练：

- `real_atom_head`
- `pseudo_atom_head`
- `real_to_pseudo` cross-attention
- `pseudo_to_real` cross-attention
- `sparse_refine_head`
- 必要的 LayerNorm / projection / bias

### 4.3 不做的训练消融

不做：

- P supervision 关闭消融。
- 零初始化开关消融。
- sparse refine 回灌 trunk 消融。
- sparse refine 完全关闭消融。

原因：

- P supervision 是新结构稳定训练的基础，不应取消。
- 零初始化属于实现安全契约，不应为了消融写额外开关。
- sparse refine 在主实验中已经最大 detach，回灌/关闭都不是本轮最关键不确定性。

## 5. 体素输入与 density block

### 5.1 3x3x3 + 5D 体素投影

体素分支的 raw atom feature 在线投影主线采用：

- `online_pdb_feature_scatter_kernel: gauss27`
- `online_pdb_feature_add_occupancy: true`
- `online_pdb_feature_add_centroid: true`

5D 编码固定为：

- 2 维 occupancy：`log(1+sum_w)`、`sum_w/max(sum_w)`。
- 3 维 centroid offset：原子相对体素中心的加权偏移 `(x,y,z)`。

这是主实验默认，不再写成“待确认”。

### 5.2 density block 消融

`configs/experiment/CPC/CPC_A7_density.yaml` 当前消融轴是：

```yaml
主实验: /model/sparse_refine/density_cube: both
消融:   /model/sparse_refine/density_cube: pseudo
```

也就是主实验同时使用：

- P anchor density cube encoder。
- 真实原子 density cube 调制。

A7 只保留 P density，关闭真实原子 density 调制。这一轴应保留到 v3 第一轮消融，因为它直接检验真实原子 density block 是否值得继续放在冻结主干后的新增路径中。

## 6. Loss 与新增日志

### 6.1 Sparse refine loss 主线

主线继续使用 sparse refine 的分类项 + ranking 辅助项，但需要更新注释与命名以适配新结构：

- `ligand_sparse_refine_loss`: C 级分类监督。
- `ligand_sparse_refine_delta_loss`: ranking-only 辅助项。
- `ligand_sparse_refine_w_rank`: ranking 权重。

当前 sparse refine loss 以 recall-biased Tversky 为主。第一轮不做预测模式消融；预测模式保持与主实验一致。

### 6.2 P 自身分类能力日志

增加一条或一组最小 wandb 日志，用来衡量 P anchor 本身的分类能力。

优先级：

1. `val_score/global/pseudo_PRAUC`
2. 可选 `val_score/global/pseudo_accuracy_at_0.5`

实现约束：

- 复用 wrapper 中已经存在的 `_sample_ligand_pseudo_supervision()`。
- 不重写 CPC diagnostics。
- 不改 candidate 中间产物的既有日志面板。
- 二分类时不额外加 `_foreground` 后缀。

### 6.3 日志清理范围

只做明确必要的局部修改：

- 从 `train_loss` 中移除非 loss 语义字段。
- `recycle_passes` 从当前零散位置收束到 runtime 语义位置。
- 删除 front/back 相关 loss 与 metric。

不要大改 wrapper 日志系统，尤其不要搬动 CPC 采样过程的中间产物：

- candidate 数量。
- anchor 数量。
- cap 前后统计。
- p_best / p_sampling。
- refined/unrefined CPC 诊断。

这些保持现有 diagnostics 面板和命名体系，除非实现 front/back 清理时必须同步改名。

## 7. 第一轮消融配置

第一轮消融仿照 `configs/experiment/CPC` 组织，生成 1 个预计最好的主实验 + 6 到 9 个单轴或强相关轴实验。每个实验只改动一个核心假设，便于对照。

建议新目录：

```text
configs/experiment/CPC_v3/
```

也可以继续放在 `configs/experiment/CPC/`，但文件名必须明确带 `v3`，避免与旧 CPC 混用。

### 7.1 主实验：CPC_v3_main

主实验配置：

- 去掉重 atom attention stack。
- 单套 `real_atom_head`。
- 单套 `pseudo_atom_head`。
- `real -> pseudo` 与 `pseudo -> real` cross-attention 同时开启。
- cross-attention source detach。
- 所有新增交互输出零初始化。
- sparse refine head 零初始化。
- sparse refine 最大 detach。
- density cube 使用 `both`。
- 体素输入使用 `gauss27 + 5D`。
- P supervision 开启。
- sparse refine loss 使用主线 recall-biased Tversky + ranking。

### 7.2 A1：关闭双向分类前 cross-attention

目的：检验分类前 real/P 单向信息交换整体贡献。

改动：

```yaml
enable_real_to_pseudo_interaction: false
enable_pseudo_to_real_interaction: false
```

说明：

- 不拆 real->P 与 P->real 两个实验。
- source detach 使两个方向近似线性叠加，合成一个消融即可。

### 7.3 A2：density block 消融

目的：检验真实原子 density cube 调制是否有价值。

改动：

```yaml
- override /model/sparse_refine/density_cube: pseudo
```

对照主实验：

```yaml
- override /model/sparse_refine/density_cube: both
```

### 7.4 A3：Sparse refine ranking loss 消融

目的：检验 ranking-only delta loss 对 refined F1 的贡献。

改动：

```yaml
model:
  ligand_sparse_refine_w_rank: 0.0
```

保留：

- C 级分类 loss。
- P supervision。
- 预测模式。
- 最大 detach。

### 7.5 A4：Sparse refine 分类 loss 7:3 正负/难例权重方案

目的：检验 sparse refine head 本身的 loss 权重是否需要更重视正负平衡。

本轮可设置一项最小对照：将 sparse refine 分类项改为寻常 `focal + Tversky`，正负权重统一写成 7:3。预测模式、candidate 预算、detach 策略保持主实验一致。

示例方向：

```yaml
model:
  ligand_sparse_refine_loss:
    focal_alpha_neg: 0.7
    focal_alpha_pos: 0.3
    w_focal: 0.7
    w_tversky: 0.3
```

具体字段以 `AdaptiveClassificationCompositeLoss` 当前参数为准；不要为了这个消融引入新的 loss 类。

### 7.6 A5：体素输入 hard scatter

目的：检验 `gauss27 + 5D` 相比最朴素 hard scatter 是否值得保留。

改动方向：

```yaml
online_pdb_feature_scatter_kernel: legacy
online_pdb_feature_use_soft_splatting: false
online_pdb_feature_add_occupancy: false
online_pdb_feature_add_centroid: false
```

注意：

- hard scatter 的通道数与 voxel backbone 输入通道要同步校验。
- 若 raw feature 通道数变化导致配置不闭合，先补配置组，不在代码里静默兜底。

### 7.7 A6：体素输入 point embedding

目的：检验可学习 point embedding -> voxel scatter 是否优于 raw gauss27 投影。

改动方向：

- 关闭 `online_pdb_feature`。
- 使用 `Stage1EmbedHead` 的 `voxel_pdb_embed_grid`。
- 保持旧主干冻结策略，除非单独配置写明 train embed head。

第一轮建议只做 frozen embed 对照；训练 embed head 放第二轮。

### 7.8 A7：预算轻量对照

目的：检验 C/P budget 是否过大或过小，但第一轮只做一个轻量预算对照，不展开网格。

改动方向：

- 减小 candidate expand factor 或每类 C 上限。
- P anchor 上限按比例减小。

这一项如果资源不足，可移到第二轮。

### 7.9 A8：备用名额

备用给实现后暴露出的最大不确定项。不得用于：

- 关闭 P supervision。
- 关闭零初始化。
- 关闭 sparse refine。
- 打开 refine 回灌 trunk。

## 8. 第二轮消融

第二轮在第一轮结果稳定后再做，不阻塞 v3 主线实现。

### 8.1 fusion 模式消融

目标：消融体素分支与点云分支的 fusion 模式。

候选轴：

- `/model/fusion: e234d4`
- `/model/fusion: d4321`
- `/model/fusion: none`
- 调整 `point_fusion_map`
- 调整 `sampler_modes`

重点观察：

- unrefined F1。
- refined F1。
- candidate recall。
- refined - unrefined。

### 8.2 P/C 预算数目消融

目标：系统检查 candidate C 与 P anchor 数量是否是瓶颈。

候选轴：

- `adaptive_expand_factor`
- `max_candidate_voxels_per_class`
- `max_anchors_per_class`
- `anchor_to_candidate.num_neighbors`

建议做小网格：

```text
budget_low
budget_main
budget_high
```

不要一次性扩太多组合，先看 cap 命中率、candidate recall 和显存。

### 8.3 Sparse refine 正负权重进一步消融

目标：围绕 sparse refine head 内部正负权重做更细扫描。

候选：

- 7:3
- 5:5
- 3:7
- recall-biased Tversky 当前主线

约束：

- 预测模式与主实验一致。
- detach 策略与主实验一致。
- candidate 预算与主实验一致。
- 不新增 loss 类，只改已有 loss 配置参数。

### 8.4 point embedding 是否训练

如果第一轮 A6 表现接近或优于 raw gauss27，可在第二轮追加：

- point embedding frozen。
- point embedding trainable。

这个实验只影响 embed head 是否加入 trainable patterns，不改变主干其他冻结策略。

## 9. 实现落点

### 9.1 `src/model/stage1_atom_head.py`

重构为最终分类头文件：

- 删除或停用 `Stage1SerializedAttentionStack` 主线调用。
- 保留/改造 cross-attention 聚合模块。
- 新增 `real_atom_head`。
- 新增 `pseudo_atom_head`。
- 新增 `real_to_pseudo` interaction。
- 新增 `pseudo_to_real` interaction。
- 输出 before/after interaction 的 real/P 对称特征。

### 9.2 `src/model/stage1_model.py`

主要改动：

- 构造轻量化后的 `Stage1AtomHead`。
- 删除 front/back 构造逻辑。
- 删除 `sparse_refine_head_cfg 启用时必须启用 atom head` 这类旧约束，改成“必须存在 final P feature”。
- `_run_atom_head()` 可以保留函数名，但函数语义改为运行最终 real/P 分类头；若改名，使用 `_run_final_real_pseudo_heads()`。
- `_run_sparse_refine_head()` 改为消费 `pseudo_feat_before_interaction` / `pseudo_feat_after_interaction`。

### 9.3 `src/model/sparse_refine/sparse_refine_head.py`

主要改动：

- 删除 `P_atom_head_feat` 输入名。
- 增加 `P_final_point_feat` / `P_interaction_feat`。
- 同步文档、错误信息和配置字段。
- 保持零初始化契约。

### 9.4 `src/wrappers/voxel_point_stage1.py`

主要改动：

- 删除 front/back loss 分支。
- real loss 只消费 `atom_logits`。
- P loss 只消费 `pseudo_logits`。
- 增加 P PRAUC 或 P accuracy 的最小验证日志。
- 只移动明确错误的 runtime 字段，不重排 CPC diagnostics。

### 9.5 `src/wrappers/voxel_point_stage1_losses.py`

主要改动：

- `compute_pseudo_loss_term()` 文档去掉“前/后置共用”表述。
- loss term 名称只保留 `pseudo`。
- atom loss term 名称只保留 `atom`。

### 9.6 `configs/`

需要新增或重写：

- `configs/experiment/CPC_v3/CPC_v3_main.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A1_no_interaction.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A2_density_pseudo.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A3_no_rank.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A4_sparse_loss_73.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A5_hard_scatter.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A6_point_embed.yaml`
- `configs/experiment/CPC_v3/CPC_v3_A7_budget_low.yaml`

第二轮可后置新增：

- fusion 模式组。
- P/C budget 组。
- sparse refine 正负权重组。

## 10. 验收

### 10.1 结构验收

- 主线不再执行额外 atom attention stack。
- 全项目无 `front/back` 分类头语义。
- `atom_logits_front` / `pseudo_logits_front` 不再出现。
- `atom_tokens` / `atom_hidden` 不再作为 wrapper-facing 输出。
- sparse refine 不再依赖 `P_atom_head_feat`。

### 10.2 训练验收

- 能从最优 checkpoint 初始化。
- unexpected/missing keys 报告清楚。
- trainable 参数只包含计划内模块。
- 零初始化模块启动时输出增量为 0 或近似 0。

### 10.3 日志验收

- `train_loss` 只包含 loss 与 loss 权重。
- recycle/runtime 字段不混入 loss。
- CPC candidate / cap / refined diagnostics 不被大搬家。
- 新增 `pseudo_PRAUC` 或等价 P 分类能力指标。

### 10.4 实验验收

第一轮至少完成：

- `CPC_v3_main`
- `A1_no_interaction`
- `A2_density_pseudo`
- `A3_no_rank`
- `A4_sparse_loss_73`

资源允许时补：

- `A5_hard_scatter`
- `A6_point_embed`
- `A7_budget_low`

每个 run 至少看：

- candidate recall。
- unrefined F1。
- refined F1。
- refined - unrefined。
- pseudo_PRAUC。
- 训练是否稳定。

## 11. 不做事项

本轮不做：

- 大规模重训全模型。
- 改数据集/label 体系。
- 改 candidate selection 算法本身。
- 关闭 P supervision 的消融。
- 零初始化开关消融。
- sparse refine 关闭消融。
- sparse refine 回灌 trunk 消融。

但必须重新写 v3 消融配置，组织方式仿照现有 `configs/experiment/CPC`，形成可直接 sbatch 的主实验与消融实验文件。
