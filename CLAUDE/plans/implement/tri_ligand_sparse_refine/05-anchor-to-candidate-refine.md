# 05：唯一候选体素 C 上的 P -> C 消息回写与 sparse refine head

## 背景与目标

本计划是三分类 ligand sparse refine 的第 5 个实施步骤，但实施前必须收拢 03/04 已落地代码中的一处契约变化：`C` 不再允许同一个 `(batch, voxel_zyx)` 因多个候选类别而保留多行，而是在各类别独立提名完成后，将冲突 voxel 合并为唯一一行，并随机选择一个提名类别作为该行的消息路由类别。

当前代码已经具备本阶段的主要前置条件：

| 能力                                       | 当前状态                                                 | 位置                                                                                                                                                                    |
| ---------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 从 `voxel_logits_ligand` 生成候选集合 C         | 已实现，但仍允许同 voxel 多类别重复                                | [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py)                                                                    |
| 从 C 采样 P anchors                         | 已实现，当前仍有按 `candidate_prob` 去重的可选路径                   | [src/model/sparse_refine/anchor_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py)                                                                  |
| 为 P 构造初始 `pseudo_feat` 并注入 final recycle | 已实现                                                  | [src/model/sparse_refine/density_cube.py](../../../../src/model/sparse_refine/density_cube.py), [src/model/stage1_model.py](../../../../src/model/stage1_model.py) |
| P 经过 point backbone 后的 `point_feat`      | final mixed 点布局中已保留                                  | [src/model/stage1_model.py](../../../../src/model/stage1_model.py)                                                                                                   |
| P 经过 atom head 后的特征                      | 实际字段名为 `pseudo_feature`，不是旧计划的 `anchor_refined_feat` | [src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py), [src/model/stage1_model.py](../../../../src/model/stage1_model.py)                     |
| voxel backbone 高分辨率最终特征                  | 已以 `voxel_final` 形式导出并被 ligand head 消费               | [src/model/stage1_voxel_backbone.py](../../../../src/model/stage1_voxel_backbone.py)                                                                                |

已执行的本地前置验证：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\model\test_sparse_candidate_set.py tests\model\test_anchor_sampler.py tests\model\test_density_cube.py tests\model\test_stage1_model.py tests\model\test_stage1_atom_head.py tests\test_voxel_ligand_thresholds.py -q
```

结果为 `59 passed, 8 warnings`；warnings 来自第三方包弃用提示，不构成本阶段阻塞。

本阶段目标：

1. 将 `C` 的运行时契约改为每个 `(batch, voxel_zyx)` 唯一一行，并令冲突类别随机路由。
2. 在同类路由约束下，从 P 向 C 构造轻量的一跳 KNN 消息，而不是旧计划中的固定 three-nn 纯距离插值。
3. 将 `voxel_logits`、P/C backbone feature、P atom head feature、相对坐标与可选类别 embedding 作为逐项可配置输入。
4. 在 C 上输出 `ligand_refine_logits_C`，支持 `direct` 与 `residual` 两种 head 模式，并支持 voxel logits 是否 detach。
5. 保持 `src/train.py` 所维护的“同样数据集和实质 batch size 时数据出现顺序相同”这一数据顺序意图；不为内部冲突随机路由增加额外的稳定种子、哈希路由或 dataloader 改动。

> [!IMPORTANT]
> 旧 05 中的 `anchor_refined_feat`、`candidate_interp_feat`、`three_nn` 与“允许 C 同 voxel 多类重复”的描述全部废弃。实现和后续 06 必须以本计划的字段与契约为准。

## 当前架构与已知约束

### final recycle 数据流

当前 `VolumePointStage1Model.forward()` 在 final recycle 中按以下顺序执行：

```text
voxel backbone
  -> _prepare_pseudo_batch(): candidate builder -> anchor sampler -> density cube -> inject P
  -> point backbone: mixed real/P point_feat
  -> _run_atom_head(): real atom logits + P pseudo_feature
  -> outputs
```

因此，P -> C 回写必须发生在 `_run_atom_head()` 之后，否则无法使用 `P_atom_head_feat = outputs["pseudo_feature"]`。

### 当前必须修正的旧契约

| 旧契约                                                      | 与新设计的冲突                                            | 本阶段处理                                     |
| -------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| C 主键是 `(batch, voxel_zyx, candidate_class)`，允许同 voxel 多行 | refine loss 和 C 输出对同一物理 voxel 出现多份预测，且用户要求 C 采样即互斥 | 将 C 主键改为 `(batch, voxel_zyx)`；类别仅表示随机路由来源 |
| P sampler 可在重复 C 上按最大 `candidate_prob` 去重                | 会把“随机选类别”改回偏向概率较大类别                                | 移除该业务路径；P 直接消费唯一 C                        |
| P -> C 使用固定 `three_nn` 逆距离插值                             | 无法表达 P/C voxel feature 与 learnable edge gate       | 改为可配置 KNN 一跳消息聚合                          |
| P refined feature 写作 `anchor_refined_feat`               | 当前代码实际输出 `pseudo_feature`                          | 使用现有字段并新增清晰的回写输出字段                        |

### 范围约束

1. `C` 与 `P` 仍只在 final recycle 中生成和消费。
2. `C` 不进入 point backbone；只有 P 进入 mixed point 布局。
3. 本阶段不实现 refine loss、metrics 或 dense scatter，这些留给 [06-loss-metrics-configs.md](06-loss-metrics-configs.md)。
4. `voxel_final` 是本阶段 P/C voxel backbone feature 的固定来源；不增加 `voxel_backbone_feature_name` 配置项，也不写运行时特征名查询/分支逻辑。
5. 当前本地 `tests/` 受仓库 `.gitignore` 规则影响；实现时仍必须维护并运行对应本地测试。

## 已有可复用代码

| 已有代码                                                 | 可复用能力                                                     | 本阶段使用方式                                                      |
| ---------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| `SparseCandidateSetBuilder.forward()`                | 按类从 ligand logits 筛选 C，已有二分类 sigmoid / 多分类 softmax 支持     | 保留每类提名逻辑，在输出物化前新增唯一 voxel 合并与随机路由                            |
| `SparseAnchorSampler.forward()`                      | 按 BOX/类别采样 P，生成 `anchor_class`、`anchor_batch_index`、坐标与计数 | 改为假定输入 C 已唯一，以随机路由后的 `candidate_class` 作为 P 类别               |
| `DensityCubeEncoder`                                 | 生成 P 初始 `pseudo_feat`                                     | 不修改其特征职责                                                     |
| `inject_pseudo_atoms()` 与 `PseudoAtomLayout`         | 构造 final mixed real/P 布局                                  | 用现有 layout 从 mixed `point_feat` 中提取 P 部分                     |
| `outputs["fused_point_feat"]`                        | 保存 final mixed `point_feat`                               | 提取 `P_point_backbone_feat`                                   |
| `outputs["pseudo_feature"]`                          | 保存 atom head 输出的 P 特征                                     | 直接作为 `P_atom_head_feat`                                      |
| `voxel_output_dict["voxel_features"]["voxel_final"]` | 最终高分辨率 voxel feature                                      | 固定采样 C/P 的 `C_voxel_backbone_feat` 与 `P_voxel_backbone_feat` |
| `_sample_voxel_feature_trilinear()`                      | 现有 voxel-to-point 特征采样逻辑                                  | 对连续 P 坐标复用；C 位于 voxel center 时可用同一坐标语义调用，避免新增不同采样约定          |

## 设计决策

### 1. C 为唯一 voxel，冲突类别随机路由

每个 BOX 中，builder 仍先按 `candidate_class_ids` 独立完成类别提名，保留不同类别各自阈值与预算语义；随后在输出 C 前，按 `(batch, voxel_zyx)` 合并重复提名：

1. 仅一个类别提名该 voxel 时，保留该类别。
2. 多个类别同时提名同一 voxel 时，在这些提名行中均匀随机选择一行。
3. 被选中的行提供 `candidate_class` 与该类别对应的 `candidate_prob`；`candidate_logits` 仍表示该 voxel 的完整原始 logits。
4. 不对失去冲突 voxel 的类别执行候补、补齐 quota 或二次筛选。

新的语义：

| 字段                                              | 新语义                                           |
| ----------------------------------------------- | --------------------------------------------- |
| `candidate_voxel_zyx` / `candidate_batch_index` | 共同确定唯一物理候选 voxel；同一 key 最多一行                  |
| `candidate_class`                               | 当前 C 的随机路由类别，用于 P 采样分组和 P -> C 同类消息，不表示最终监督标签 |
| `candidate_prob`                                | 被随机路由类别在该 voxel 的提名概率，仅用于采样/诊断，不作为类别公平性的裁决依据  |
| `candidate_counts_by_class`                     | 随机路由并合并后的实际 C 数量，不再等于各类原始提名数                  |
| `candidate_target_counts_by_class`              | 合并前每类计划提名数，继续用于诊断候选预算                         |

随机性约束：

1. 不新增 `generator`、固定随机种子、stable hash 或基于样本 id 的确定性路由接口。
2. 不修改 [src/train.py](../../../../src/train.py) 的数据加载、sampler 或 batch 顺序逻辑。
3. 训练与推理中的内部冲突路由允许沿用普通 PyTorch 随机状态。

### 2. P 继承路由类别，并只向同类 C 发送消息

`SparseAnchorSampler` 在唯一 C 上按已有 per-class 采样规则产生 P：

```text
candidate_class (random routed C class) -> anchor_class
```

P -> C 聚合默认并首版必须使用：

```yaml
same_class_only: true
```

即每个 `C_i` 只搜索同时满足以下条件的 `P_j`：

1. `candidate_batch_index[i] == anchor_batch_index[j]`。
2. `candidate_class[i] == anchor_class[j]`。

这样，冲突时随机得到的 `candidate_class` 才真正控制消息路径，而不是只作为无效 metadata 透传。

### 3. 使用可调 KNN 一跳消息，而非固定 three-nn

新增 P -> C 邻居搜索配置，模式名使用通用语义 `knn_message`。该模块只构造稀疏边；learnable gate、距离权重和消息聚合统一由 `sparse_refine_head_cfg` 控制：

```yaml
anchor_to_candidate_cfg:
  mode: knn_message
  num_neighbors: 3
  same_class_only: true
  chunk_size: 8192
```

行为要求：

1. `num_neighbors` 为显式可调的正整数，默认 `3`；实现中不保留名为 `three_nn` 的固定分支。
2. 若某个 C 所在同类 P 数少于 `num_neighbors`，只使用实际存在的 P。
3. 若某个 C 没有任何同类 P，则消息聚合自然得到零向量 `delta=0`；无需特殊回退网络。
4. 按 `batch/class/chunk` 计算邻居，不构造全局 `(sumC, sumP)` 距离矩阵。

### 4. 特征输入全部逐项可开关

`sparse_refine_head_cfg.inputs` 必须支持以下开关，并使用大写 `C` / `P` 标识候选与 anchor 角色：

```yaml
sparse_refine_head_cfg:
  mode: residual                # direct | residual
  detach_voxel_logits: true
  edge_weight_activation: sigmoid
  distance_weight:
    mode: softmax_negative_squared_distance
    temperature: 1.0

  inputs:
    use_voxel_logits: true
    use_C_voxel_backbone_feat: true
    use_P_point_backbone_feat: true
    use_P_atom_head_feat: true
    use_P_voxel_backbone_feat: true
    use_relative_coords: true
    use_candidate_class_embedding: false
```

特征来源和放置位置：

| 配置项                             | 张量语义                                                      | 消费位置                                       | 默认值     |
| ------------------------------- | --------------------------------------------------------- | ------------------------------------------ | ------- |
| `use_voxel_logits`              | C 所在 voxel 的 ligand logits；二分类为原始一维 logit，多分类为完整多维 logits | final C head 输入；residual 模式也作为 base logits | `true`  |
| `use_C_voxel_backbone_feat`     | 在 C 位置采样固定 `voxel_final` 得到的特征                            | edge MLP 输入，同时直接进入 final C head            | `true`  |
| `use_P_point_backbone_feat`     | final mixed `point_feat` 中属于 P 的部分                        | P content MLP 输入                           | `true`  |
| `use_P_atom_head_feat`          | `outputs["pseudo_feature"]`                               | P content MLP 输入                           | `true`  |
| `use_P_voxel_backbone_feat`     | 在 P 位置采样固定 `voxel_final` 得到的特征                            | edge MLP 输入                                | `true`  |
| `use_relative_coords`           | 每条边的 `C_coord - P_coord`，坐标系沿用 centered-world xyz         | edge MLP 输入                                | `true`  |
| `use_candidate_class_embedding` | P 的 `anchor_class` embedding；因当前 class 来自随机路由，默认关闭        | edge MLP 输入                                | `false` |

约束：

1. `voxel_final` 直接写入实现，不以 YAML 暴露 `voxel_backbone_feature_name`。
2. `P_content_mlp` 至少启用 `use_P_point_backbone_feat` 或 `use_P_atom_head_feat` 之一；配置边界处 fail-fast。
3. `use_C_voxel_backbone_feat=true` 时，其特征既影响边 gate，也直接提供无消息场景下的 C 本地信息。
4. `use_voxel_logits=false` 只适用于 `mode=direct`；`mode=residual` 必须启用 base logits。

### 5. learned gate 与距离权重组合

本阶段采用一跳 gated message aggregation，不引入多层图网络或全连接 attention。下面的 P content、edge gate、距离加权与 `delta` 聚合全部由 `SparseRefineHead` 消费其单一配置来源完成；邻居搜索模块仅提供 `nn_index`、距离与有效 mask。

```python
P_content = P_content_mlp(
    cat_optional(P_point_backbone_feat, P_atom_head_feat)
)  # (sumP, H_msg)

edge_input = cat_optional(
    C_voxel_backbone_feat[:, None, :],
    P_voxel_backbone_feat[nn_index],
    relative_coords,
    candidate_class_embedding[nn_index],
)  # (sumC, K, H_edge_in)

learned_gate = sigmoid(edge_mlp(edge_input))  # (sumC, K, 1)

distance_weight = masked_softmax(
    -squared_distance / temperature,
    valid_neighbor_mask,
    dim=1,
)  # (sumC, K, 1)

delta = sum_j(distance_weight * learned_gate * P_content[nn_index])  # (sumC, H_msg)
```

采用 `sigmoid` 而不是 `tanh` 或无界激活的理由：

1. gate 在此处表达消息置信强度，`sigmoid` 的 `[0, 1]` 范围与该职责一致。
2. P content 与输出 MLP 已能承载正负信息，不需要 edge gate 额外翻转消息符号。
3. 距离项已经进行归一化，正且有界的 learned gate 使初版训练尺度更稳定。

距离项采用 `masked_softmax(-squared_distance / temperature)`，而不是 `softmax(+distance)`；距离越近，权重越高。它仅对当前 C 的有效同类邻居归一化。

### 6. sparse refine head 与梯度策略

final C head 输入：

```python
final_C_feat = cat_optional(
    voxel_logits,
    C_voxel_backbone_feat,
    delta,
)
delta_logits = output_mlp(final_C_feat)
```

head 模式：

```python
if mode == "direct":
    ligand_refine_logits_C = delta_logits
elif mode == "residual":
    base_logits = voxel_logits.detach() if detach_voxel_logits else voxel_logits
    ligand_refine_logits_C = base_logits + delta_logits
```

输出通道约定：

| ligand 任务   | `voxel_logits` / `ligand_refine_logits_C` shape |
| ----------- | ----------------------------------------------- |
| 二分类 sigmoid | `(sumC, 1)`，保留原始一维 logit                        |
| 多分类 softmax | `(sumC, num_classes)`，保留包含背景的全部 logits          |

梯度策略：

1. C 筛选、冲突随机路由和 P 采样继续是非可导离散路径。
2. 当前 builder 输出的 `candidate_logits` 已 detach，可用于 `detach_voxel_logits=true` 的默认 residual base。
3. `detach_voxel_logits=false` 时，`VolumePointStage1Model` 必须依据唯一 C 索引从 final `voxel_logits_ligand` 重新 gather 带梯度的 logits，不得复用 builder 的 detached 字段。
4. residual 模式下将 `output_mlp` 最后一层零初始化，使初始 `ligand_refine_logits_C` 接近 base logits。

### 7. 无同类 P 的 C

当一个 C 在其随机路由类别下找不到 P：

1. `delta` 为 `(H_msg,)` 零向量，这是聚合公式的自然结果。
2. `candidate_message_valid_mask[i] = False`，用于诊断和供 06 决定是否做 loss 过滤/分组统计。
3. head 仍正常计算该 C 的 logits；默认 residual 模式可依赖 `voxel_logits` 与可选的 `C_voxel_backbone_feat`。
4. 05 不在该情况下删除 C，也不在 P 采样阶段补造 P。

## 输出契约

`VolumePointStage1Model.forward()` 新增或更新以下 sparse refine 输出。未列出的现有 real-atom supervised 输出契约保持不变。

| 输出字段                                    | shape                               | 语义                       |
| --------------------------------------- | ----------------------------------- | ------------------------ |
| `candidate_voxel_zyx`                   | `(sumC, 3)`                         | 唯一 C voxel 坐标，轴顺序 `z/y/x` |
| `candidate_batch_index`                 | `(sumC,)`                           | C 所属 BOX                 |
| `candidate_class`                       | `(sumC,)`                           | 冲突合并后随机路由类别              |
| `candidate_prob`                        | `(sumC,)`                           | 路由类别对应概率                 |
| `candidate_logits`                      | `(sumC, 1)` 或 `(sumC, num_classes)` | detached C voxel logits  |
| `anchor_voxel_zyx`                      | `(sumP, 3)`                         | P 来源 voxel               |
| `anchor_class`                          | `(sumP,)`                           | P 继承的路由类别                |
| `candidate_neighbor_index`              | `(sumC, K)`                         | 每个 C 的 P 邻居下标，无效位置由 mask 屏蔽 |
| `candidate_neighbor_squared_distance`   | `(sumC, K)`                         | C 到 P 邻居的平方距离            |
| `candidate_neighbor_relative_coords`    | `(sumC, K, 3)`                      | `C_coord - P_coord`       |
| `candidate_neighbor_valid_mask`         | `(sumC, K)`                         | 每个邻居槽位是否有效               |
| `candidate_message_valid_mask`          | `(sumC,)`                           | 是否存在至少一个有效同类 P 邻居        |
| `ligand_refine_logits_C`                | `(sumC, 1)` 或 `(sumC, num_classes)` | C 上的 refined logits      |

`candidate_message_delta`、P/C backbone feature 与 P atom head feature 仅是 head 内部中间量，不作为 `VolumePointStage1Model.forward()` 的长期公开输出。对 wrapper/06 必须稳定暴露 `candidate_*` 元数据、`candidate_neighbor_*`、`candidate_message_valid_mask` 与 `ligand_refine_logits_C`。

## Proposed Changes

### 1. 将 candidate set 改为唯一 C

#### [MODIFY] [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py)

修改 `SparseCandidateSetBuilder` 的输出物化逻辑：

1. 保留现有二分类 sigmoid、多分类 softmax、warmup/top-k、adaptive/recorded threshold 选择过程。
2. 每类生成 provisional rows 后，按 `(batch, voxel_zyx)` 分组。
3. 单行组直接保留；冲突组对该组 provisional row 做普通随机选择。
4. 按选中行生成唯一 `candidate_*` 输出，并在合并后重算 `candidate_counts` 与 `candidate_counts_by_class`。
5. `candidate_target_counts_by_class` 保持合并前提名预算含义；在 Docstring 中明确其和实际 count 可能不同。
6. 更新类级与 `forward()` Docstring：`candidate_class` 为路由类别，不再是“同 voxel 可重复的候选类别主键”。

不新增 routing 配置字段：随机路由是本阶段固定契约，不提供 max-prob / first-row 等可切换业务策略。

### 2. 收拢 P sampler 的重复 C 兼容逻辑

#### [MODIFY] [src/model/sparse_refine/anchor_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py)

1. 删除或停止暴露 `deduplicate_candidates` 配置分支及其“保留最大 `candidate_prob`”逻辑。
2. 将 `forward()` 契约改为输入 `candidate_outputs` 已满足 `(batch, voxel_zyx)` 唯一。
3. 继续按 `candidate_class` 进行 per-class P 采样，使随机路由类别控制 P quota 和后续同类消息。
4. 保持现有坐标构造、三种 P 采样 mode 和计数字段不变。

### 3. 新增 P -> C KNN 邻居搜索模块

#### [NEW] [src/model/sparse_refine/interpolation.py](../../../../src/model/sparse_refine/interpolation.py)

保留模块文件名 `interpolation.py` 以承接原计划位置，但该模块只负责在同类路由约束下找到稀疏邻居边，不持有 feature 开关或 learnable edge 参数：

```python
class AnchorToCandidateKnnSearch(nn.Module):
    def __init__(
        self,
        mode: str,
        num_neighbors: int,
        same_class_only: bool,
        chunk_size: int,
    ) -> None: ...

    def forward(
        self,
        candidate_coord_centered_world: torch.Tensor,
        candidate_batch_index: torch.Tensor,
        candidate_class: torch.Tensor,
        anchor_coord_centered_world: torch.Tensor,
        anchor_batch_index: torch.Tensor,
        anchor_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...
```

返回字段：

| 字段                                    | shape          | 语义                              |
| ------------------------------------- | -------------- | ------------------------------- |
| `candidate_neighbor_index`            | `(sumC, K)`    | 每个 C 的 P 邻居下标；无效位置配合 mask 忽略    |
| `candidate_neighbor_squared_distance` | `(sumC, K)`    | 邻居平方距离；供 head 计算距离权重            |
| `candidate_neighbor_relative_coords`  | `(sumC, K, 3)` | `C_coord - P_coord`，供可选 edge 输入 |
| `candidate_neighbor_valid_mask`       | `(sumC, K)`    | 有效邻居掩码；某 C 全 False 表示无同类 P      |

内部控制流：

1. 按 BOX 和路由类别筛选允许相连的 C/P。
2. 在每组中分块计算 C 到 P 的 squared distance，并取实际最多 `num_neighbors` 个邻居。
3. 输出 padded neighbor index、squared distance、relative coords 与有效 mask。
4. 不在此模块中引入 MLP、embedding 或 feature 开关，避免配置所有权分裂。

### 4. 新增 sparse refine head

#### [NEW] [src/model/sparse_refine/sparse_refine_head.py](../../../../src/model/sparse_refine/sparse_refine_head.py)

```python
class SparseRefineHead(nn.Module):
    def __init__(
        self,
        mode: str,
        detach_voxel_logits: bool,
        edge_weight_activation: str,
        distance_weight_mode: str,
        distance_temperature: float,
        message_dim: int,
        edge_hidden_dim: int,
        hidden_dim: int,
        num_layers: int,
        logit_dim: int,
        C_voxel_backbone_dim: int,
        P_point_backbone_dim: int,
        P_atom_head_dim: int,
        P_voxel_backbone_dim: int,
        inputs: dict[str, bool],
        candidate_class_ids: tuple[int, ...],
        candidate_class_embedding_dim: int,
        zero_init_residual: bool,
    ) -> None: ...

    def forward(
        self,
        voxel_logits: torch.Tensor | None,
        C_voxel_backbone_feat: torch.Tensor | None,
        P_point_backbone_feat: torch.Tensor | None,
        P_atom_head_feat: torch.Tensor | None,
        P_voxel_backbone_feat: torch.Tensor | None,
        anchor_class: torch.Tensor,
        candidate_neighbor_index: torch.Tensor,
        candidate_neighbor_squared_distance: torch.Tensor,
        candidate_neighbor_relative_coords: torch.Tensor,
        candidate_neighbor_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...
```

实现要求：

1. `mode` 显式接受 `"direct"` 或 `"residual"`。
2. head 内部根据 `inputs` 构造 `P_content_mlp`、edge gate、距离权重与局部消息向量，并只返回 `candidate_message_valid_mask = candidate_neighbor_valid_mask.any(dim=1)` 与 `ligand_refine_logits_C`。
3. 二分类/多分类不做 logits 语义重编码，只按配置提供的 `logit_dim` 输出自然维度。
4. `mode="residual"` 时要求 `voxel_logits` 非空；是否 detach 由 `detach_voxel_logits` 控制。
5. residual 默认对输出增量最后一层零初始化。
6. 输入可选项仅拼接启用的张量；不额外拼接 softmax probability。

### 5. 更新 sparse refine 包导出

#### [MODIFY] [src/model/sparse_refine/**init**.py](../../../../src/model/sparse_refine/__init__.py)

新增导出：

```python
from src.model.sparse_refine.interpolation import AnchorToCandidateKnnSearch
from src.model.sparse_refine.sparse_refine_head import SparseRefineHead
```

### 6. 集成到 Stage1 final recycle 输出路径

#### [MODIFY] [src/model/stage1_model.py](../../../../src/model/stage1_model.py)

构造参数新增：

```python
anchor_to_candidate_cfg: dict[str, Any] | nn.Module | None = None
sparse_refine_head_cfg: dict[str, Any] | nn.Module | None = None
```

初始化约束：

1. `anchor_to_candidate_cfg` 或 `sparse_refine_head_cfg` 启用时，`candidate_set_builder`、`anchor_sampler` 与 atom head 必须启用。
2. 两者必须同时开启或同时关闭，避免产生消息但没有 logits、或 head 没有 P 消息输入。
3. 开启消息聚合时，将 `"voxel_final"` 无条件纳入 `self.voxel_feature_names_to_return`；这是固定实现常量，不读取额外 feature-name 配置。
4. `sparse_refine_head_cfg` 中只保存实验可调策略与网络宽度；`VolumePointStage1Model.__init__()` 在 instantiate 时显式注入下列由已构造模块确定的维度，避免 YAML 重复维护通道数：

| 注入参数                                            | 来源                                                            |
| ----------------------------------------------- | ------------------------------------------------------------- |
| `logit_dim`                                     | `self.voxel_backbone.voxel_ligand_logit_dim`                  |
| `C_voxel_backbone_dim` / `P_voxel_backbone_dim` | `self.voxel_backbone.feature_channels_by_name["voxel_final"]` |
| `P_point_backbone_dim`                          | `self.point_backbone.feature_channels_by_name["point_feat"]`  |
| `P_atom_head_dim`                               | `self.atom_head.pseudo_feature_dim`                           |
| `candidate_class_ids`                           | `self.candidate_set_builder.candidate_class_ids`              |

新增 final refine 私有流程，例如：

```python
def _run_sparse_refine_head(
    self,
    outputs: dict[str, Any],
    voxel_output_dict: dict[str, Any],
    pseudo_layout: PseudoAtomLayout | None,
) -> None: ...
```

调用位置：

```python
self._run_atom_head(outputs, atom_head_batch=last_atom_head_batch, pseudo_layout=last_pseudo_layout)
self._run_sparse_refine_head(outputs, last_voxel_output_dict, last_pseudo_layout)
```

流程职责：

1. 从 final mixed `outputs["fused_point_feat"]` 与 `pseudo_layout` 提取 `P_point_backbone_feat`。
2. 读取 `outputs["pseudo_feature"]` 作为 `P_atom_head_feat`。
3. 从固定 `voxel_output_dict["voxel_features"]["voxel_final"]` 按 P/C 来源 voxel 下标直接 gather P 与 C feature。
4. 构造 C centered-world 坐标；其 voxel center 语义必须与 P 现有坐标转换一致。
5. 若 `detach_voxel_logits=false`，从 final `voxel_logits_ligand` 按唯一 C 索引 gather 带梯度 logits；否则使用 detached `candidate_logits`。
6. 先调用 `AnchorToCandidateKnnSearch` 产生稀疏边，再调用 `SparseRefineHead` 完成 edge gate、距离加权、消息聚合与 C logits。
7. 将 `candidate_neighbor_*`、`candidate_message_valid_mask` 与 `ligand_refine_logits_C` 写入最终 outputs；不暴露仅用于调试的消息向量或 P/C 中间特征。

### 7. Hydra 配置组

#### [MODIFY] [configs/base.yaml](../../../../configs/base.yaml)

在现有 sparse refine 子模块 defaults 旁增加默认关闭项：

```yaml
- model/sparse_refine/anchor_to_candidate: none  # P -> C KNN 邻居搜索(默认关闭)
- model/sparse_refine/sparse_refine_head: none   # C refined logits head(默认关闭)
```

#### [NEW] [configs/model/sparse_refine/anchor_to_candidate/none.yaml](../../../../configs/model/sparse_refine/anchor_to_candidate/none.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_to_candidate_cfg: null
```

#### [NEW] [configs/model/sparse_refine/anchor_to_candidate/knn_message.yaml](../../../../configs/model/sparse_refine/anchor_to_candidate/knn_message.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_to_candidate_cfg:
      _target_: src.model.sparse_refine.interpolation.AnchorToCandidateKnnSearch
      mode: knn_message
      num_neighbors: 3
      same_class_only: true
      chunk_size: 8192
```

#### [NEW] [configs/model/sparse_refine/sparse_refine_head/none.yaml](../../../../configs/model/sparse_refine/sparse_refine_head/none.yaml)

```yaml
# @package _global_
model:
  backbone:
    sparse_refine_head_cfg: null
```

#### [NEW] [configs/model/sparse_refine/sparse_refine_head/default.yaml](../../../../configs/model/sparse_refine/sparse_refine_head/default.yaml)

```yaml
# @package _global_
model:
  backbone:
    sparse_refine_head_cfg:
      _target_: src.model.sparse_refine.sparse_refine_head.SparseRefineHead
      mode: residual                  # str, direct | residual; residual 在 voxel logits 上学习增量
      detach_voxel_logits: true       # bool, residual base 是否阻断到 voxel ligand head 的梯度
      edge_weight_activation: sigmoid # str, P -> C edge gate 激活; [0, 1] 有界门控
      distance_weight:
        mode: softmax_negative_squared_distance  # str, 按负平方距离在有效邻居内归一化
        temperature: 1.0                         # float, 距离 softmax 温度, 必须 > 0
      message_dim: 128                # int, P -> C 聚合消息通道数
      edge_hidden_dim: 128            # int, edge MLP 隐藏通道数
      candidate_class_embedding_dim: 16 # int, 仅在类别 embedding 开启时使用的通道数
      hidden_dim: 128                 # int, C head 隐藏维度
      num_layers: 2                   # int, C head MLP 层数
      zero_init_residual: true        # bool, residual 增量末层是否零初始化

      inputs:
        use_voxel_logits: true                # bool, 二分类使用原始一维 logit; 多分类使用完整 logits
        use_C_voxel_backbone_feat: true       # bool, C 位置固定 voxel_final 特征; 同时供 edge 与 final head 使用
        use_P_point_backbone_feat: true       # bool, P 的 final point backbone point_feat
        use_P_atom_head_feat: true            # bool, P 的 atom head pseudo_feature
        use_P_voxel_backbone_feat: true       # bool, P 位置固定 voxel_final 特征
        use_relative_coords: true             # bool, edge 上的 C-P centered-world 相对坐标
        use_candidate_class_embedding: false  # bool, P 路由类别 embedding; 随机路由主方案默认关闭
```

`inputs`、`edge_weight_activation` 与 `distance_weight` 只由 `SparseRefineHead` 消费；`anchor_to_candidate_cfg` 只控制邻居边构造，不复制上述特征或 gate 配置。

#### [MODIFY] [configs/model/sparse_refine/anchor_sampler/topk_nms.yaml](../../../../configs/model/sparse_refine/anchor_sampler/topk_nms.yaml) 及其它 sampler mode 配置

删除 `deduplicate_candidates` 字段和“同 voxel 多类别分别生成 P”的旧注释，改为声明输入 C 已在 candidate builder 中唯一化，`candidate_class` 为路由类别。

### 8. 同步跨阶段计划契约

本阶段实现前或同一提交中，应同步修改以下计划文档，防止代码与计划互相冲突：

| 文档                                                                           | 必须同步的内容                                                                                   |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [00-master.md](00-master.md)                                                 | C 唯一 voxel + 随机路由、同类 KNN message、head 输入开关、输出字段                                           |
| [03-sparse-candidate-C.md](03-sparse-candidate-C.md)                         | 删除“C 允许同 voxel 多类别重复”的主键契约，增加冲突随机路由与 counts 新语义                                           |
| [04-anchor-P-sampling-and-features.md](04-anchor-P-sampling-and-features.md) | 删除按最高概率 P 层去重决策；P 继承已路由的唯一 C 类别                                                           |
| [06-loss-metrics-configs.md](06-loss-metrics-configs.md)                     | 将 `candidate_interp_valid_mask` 替换为 `candidate_message_valid_mask`；loss/metric 只在唯一 C 上定义 |

## 不修改的部分

1. 不实现完整 graph network、多轮 message passing 或全局 attention。
2. 不增加 `voxel_final` 之外的多尺度 P/C voxel feature 选项。
3. 不实现 C 的 dense scatter 回全体素网格。
4. 不实现 refined loss、metrics、漏检惩罚或 checkpoint monitor 逻辑；这些属于 06。
5. 不改变 density cube 的 P 初始化职责。
6. 不为冲突随机路由增加 quota 补齐、候补筛选、固定 hash 或独立可复现 RNG 管线。
7. 不改动 `src/train.py` 中与数据出现顺序相关的机制。

## 改动文件汇总

| 文件                                                                                                                              | 改动内容                                                         |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py)                              | C 合并为唯一 voxel，冲突提名随机路由，更新字段语义与计数                             |
| [src/model/sparse_refine/anchor_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py)                            | 移除概率去重业务分支，消费唯一 C 与随机路由类别                                    |
| [src/model/sparse_refine/interpolation.py](../../../../src/model/sparse_refine/interpolation.py)                               | 新增同类 KNN 邻居边搜索                                               |
| [src/model/sparse_refine/sparse_refine_head.py](../../../../src/model/sparse_refine/sparse_refine_head.py)                   | 新增 P content、edge gate、消息聚合与 direct/residual C logits head   |
| [src/model/sparse_refine/**init**.py](../../../../src/model/sparse_refine/__init__.py)                                         | 导出新增模块                                                       |
| [src/model/stage1_model.py](../../../../src/model/stage1_model.py)                                                             | final atom head 后接 P -> C 回写；固定采样 `voxel_final`；输出 refine 字段 |
| [configs/base.yaml](../../../../configs/base.yaml)                                                                              | 注册新的默认关闭配置组                                                  |
| `configs/model/sparse_refine/anchor_to_candidate/{none,knn_message}.yaml`                                                       | 新增消息聚合配置                                                     |
| `configs/model/sparse_refine/sparse_refine_head/{none,default}.yaml`                                                            | 新增 head 与输入开关配置                                              |
| `configs/model/sparse_refine/anchor_sampler/*.yaml`                                                                             | 删除 P 层概率去重配置与冲突注释                                            |
| [CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md](00-master.md)                                                 | 同步总控契约                                                       |
| [CLAUDE/plans/implement/tri_ligand_sparse_refine/03-sparse-candidate-C.md](03-sparse-candidate-C.md)                         | 同步 C 唯一化契约                                                   |
| [CLAUDE/plans/implement/tri_ligand_sparse_refine/04-anchor-P-sampling-and-features.md](04-anchor-P-sampling-and-features.md) | 同步 P 输入契约                                                    |
| [CLAUDE/plans/implement/tri_ligand_sparse_refine/06-loss-metrics-configs.md](06-loss-metrics-configs.md)                     | 同步 loss/metric mask 与唯一 C 语义                                 |
| [tests/model/test_sparse_candidate_set.py](../../../../tests/model/test_sparse_candidate_set.py)                             | 更新随机冲突路由与唯一 C 单测                                             |
| [tests/model/test_anchor_sampler.py](../../../../tests/model/test_anchor_sampler.py)                                          | 删除 max-prob 去重测试，增加唯一 C 输入/路由采样测试                            |
| `tests/model/test_anchor_to_candidate.py`                                                                                       | 新增 KNN message/gate/无邻居零消息测试                                 |
| `tests/model/test_sparse_refine_head.py`                                                                                        | 新增 direct/residual/detach/二分类多分类维度测试                         |
| [tests/model/test_stage1_model.py](../../../../tests/model/test_stage1_model.py)                                              | 新增 Stage1 final refine 集成输出测试                                |

## Verification Plan

### Automated Tests

使用项目本地 Windows 环境执行：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\model\test_sparse_candidate_set.py tests\model\test_anchor_sampler.py tests\model\test_density_cube.py tests\model\test_anchor_to_candidate.py tests\model\test_sparse_refine_head.py tests\model\test_stage1_model.py tests\model\test_stage1_atom_head.py tests\test_voxel_ligand_thresholds.py -q
```

关键测试覆盖：

| 测试模块                           | 必须覆盖的行为                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_sparse_candidate_set.py` | 二分类/多分类原路径仍可运行；多类同时提名同一 voxel 后只输出一行；其 `candidate_class` 属于提名类别集合；合并后 counts 正确；builder 输出 logits 保持 detached                                                                |
| `test_anchor_sampler.py`       | sampler 输入唯一 C；不再依赖按最大概率去重；P `anchor_class` 继承 C 随机路由类别；原采样 mode 与坐标测试保留                                                                                                     |
| `test_anchor_to_candidate.py`  | `num_neighbors=1/3/大于可用 P 数`；`same_class_only=true`；无同类 P 时邻居 mask 全 False；chunk 结果与小规模直接 KNN 一致                                                                             |
| `test_sparse_refine_head.py`   | 距离近邻 softmax 权重；`sigmoid` gate shape；无有效邻居时 `delta==0` 且 message mask\=False；binary `(sumC,1)` 与 tri `(sumC,3)`；direct/residual；`detach_voxel_logits=true/false` 梯度路径；输入开关组合 |
| `test_stage1_model.py`         | fixed `voxel_final` 被请求并采样；从 mixed point 输出提取 P feature；atom head 后产生 `ligand_refine_logits_C`；real-only atom 监督输出保持不变                                                       |

对随机路由单测的约束：测试只验证“唯一化”和“所选类别属于冲突提名集合”，或在测试内部临时控制普通 PyTorch RNG 以覆盖分支；生产接口不增加可复现随机参数。

### Config / Hydra Smoke Tests

在实现新的配置组后执行三分类 smoke：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m src.train +experiment=tri001_tunedloss model/sparse_refine/candidate_set=tri model/sparse_refine/anchor_sampler=topk_nms model/sparse_refine/density_cube=default model/sparse_refine/anchor_to_candidate=knn_message model/sparse_refine/sparse_refine_head=default trainer.fast_dev_run=true
```

并至少执行一次二分类实例化 smoke，验证 `voxel_logits` 的一维 logit 输出契约：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m src.train model/task=binary model/sparse_refine/candidate_set=binary model/sparse_refine/anchor_sampler=topk_nms model/sparse_refine/density_cube=default model/sparse_refine/anchor_to_candidate=knn_message model/sparse_refine/sparse_refine_head=default trainer.fast_dev_run=true
```

若训练入口实际 fast-dev-run override 名不同，实施时使用仓库现有等价入口，验证目标不变。

### Manual Checks

```powershell
rg "anchor_refined_feat|candidate_interp_feat|candidate_interp_valid_mask|three_nn|deduplicate_candidates" src configs tests CLAUDE\plans\implement\tri_ligand_sparse_refine
```

预期：执行代码和新配置不再依赖旧字段/旧 mode；历史计划文件若命中，必须随跨阶段同步更新。

```powershell
rg "voxel_backbone_feature_name|voxel_final|use_C_voxel_backbone_feat|use_P_voxel_backbone_feat|use_P_point_backbone_feat|use_P_atom_head_feat|use_relative_coords|use_candidate_class_embedding" src configs CLAUDE\plans\implement\tri_ligand_sparse_refine
```

预期：实现固定读取 `voxel_final`，不存在 feature-name 配置项；七类输入开关均可追踪到消费位置。

### Acceptance Criteria

1. C 输出按 `(batch, voxel_zyx)` 唯一；冲突 voxel 只保留一个随机路由的 `candidate_class`。
2. 冲突随机路由不使用 `candidate_prob` 做胜负裁决，不增加 quota 回填和独立确定性随机机制。
3. P 按路由类别采样，P -> C 默认只连接同 BOX 且同路由类别的邻居。
4. `num_neighbors` 可配置，默认 `3`；邻居不足时自然减少，无邻居时 `delta=0`。
5. 消息聚合使用 learnable `sigmoid` edge gate 与 `masked_softmax(-squared_distance / temperature)` 距离权重。
6. P/C voxel backbone feature 固定来自 `voxel_final`，不引入 `voxel_backbone_feature_name` 查询配置。
7. `sparse_refine_head_cfg.inputs` 包含并能独立控制用户确认的七项输入；默认仅 `use_candidate_class_embedding=false`。
8. 二分类 head 输出原始一维 logit，多分类 head 输出完整多维 logits。
9. head 支持 `direct` / `residual` 与 `detach_voxel_logits`；默认 residual + detach，且 residual 初始接近 base logits。
10. `candidate_message_valid_mask` 表示是否收到 P 消息；无消息的 C 仍产生 logits。
11. Stage1 的 real atom supervision 契约与 final recycle 注入时机不被改变。
12. 03/04/06 与 00-master 的冲突契约在实现时同步修订，测试通过后再进入 06 loss/metrics 实施。