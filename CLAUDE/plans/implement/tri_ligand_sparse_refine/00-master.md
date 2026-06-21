# 三分类 ligand sparse refine 总控计划

> [!IMPORTANT]
> 第 05 阶段实现后，运行时契约已更新：`C` 从允许同 voxel 多类别重复改为按 `(batch, voxel_zyx)` 唯一，并在冲突提名中随机选择路由类别；`P -> C` 从固定 `three_nn` 改为同路由类别 KNN message + `SparseRefineHead`。本文下方旧描述若与 [05-anchor-to-candidate-refine.md](05-anchor-to-candidate-refine.md) 或现行源码冲突，以第 05 阶段契约与源码为准。

## 1. 为什么重写计划

原 `tri_ligand_refine` 计划把“候选体素”与“伪原子”近似等同，容易在当前数据配置下产生“伪原子喧宾夺主”问题。

已知统计：

* BOX：`80^3`，`1.0Å` 分辨率。
* 标签阈值：`1.7Å`。
* 平均每 BOX：
  * small molecule 正类体素约 `3000`，由约 `300` 个真实配体原子产生。
  * metal ion 正类体素约 `100`，由约 `30` 个真实金属离子产生。
  * 真实原子约 `3000`，其中受体原子约 `200`。

如果直接为大量正类/候选体素创建伪原子，点云分支会从“受体结构 + 少量候选锚点互作”退化为“密集体素后处理器”，伪原子数量会接近或超过真实原子数量，并压制真实原子的结构语义。

因此本计划改为 sparse refine 范式：

```text
voxel logits -> 候选体素集合 C -> 从 C 采样锚点 P -> P 与真实原子点云互作 -> P feature 插值回 C -> 在 C 上输出 refined logits
```

其中：

* `C` 是 sparse candidate voxel set，可有 `3000~25000` 个。
* `P` 是少量 pseudo anchor points，约 `256~1024` 个。
* `P` 进入 point backbone / atom head。
* 最终 loss 和 logits 在 `C` 上计算，而不是只在 `P` 上计算。
* 未进入 `C` 的 GT 正体素在指标中显式算作漏检。

## 2. 总目标

实现三分类 ligand sparse refine：

1. 从 `voxel_logits_ligand` 生成逐类候选集合 `C`；长期支持二分类 `(B, 1, D, H, W)` sigmoid 与多分类 `(B, C_cls, D, H, W)` softmax。
2. 从 `C` 中逐类采样少量 pseudo anchors `P`。
3. `P` 以伪原子形式进入 point backbone / atom head，与真实原子互作。
4. 用 `three_nn` 将 `P` 的 refined feature 插值回 `C`。
5. 在 `C` 上通过 sparse logit head 输出 refined logits。
6. 训练 loss 在 `C` 内计算。
7. 验证输出 refined best F1 与 PR-AUC，并显式惩罚未进入 `C` 的漏检。

## 3. 核心决策

* 算法大改优先；当前 1.0Å / 1.7Å 数据配置作为压力测试。
* 1.5Å 数据集、更小标签阈值、较弱 expand factor 作为后续消融，不作为主修补方案。
* `C` 生成单独模块化，默认阈值扩张；`candidate_class_ids` 显式配置、严格校验、可扩展。
* `C` 生成长期支持 `voxel_ligand_logit_dim: 1` 的 sigmoid 二分类路径；多通道 ligand logits 使用 softmax。
* wrapper 维护 voxel ligand `p_best_by_class` 与 `p_sampling_by_class` 缓存；03 阶段只补 voxel 分支 best-F1 / sampling threshold，refined C 指标留到 06。
* 训练 checkpoint 可供推理直接复用 `p_best_by_class` / `p_sampling_by_class`：要求 checkpoint 字段存在、finite、长度匹配且 `candidate_class_ids` 与推理配置完全一致；`recorded_threshold` 直接用 `p_sampling_by_class`，`adaptive_threshold` 同时依赖 `p_best_by_class` 与当前 BOX 分布。
* `C` 的主键是 `(batch, voxel_zyx, candidate_class)`，允许同一 voxel 因多个前景类别同时入选而出现多行。
* `C` 阶段只输出离散 voxel index 与候选元数据，不输出 world/local 连续坐标；04 才把 C 转成 P anchor 坐标。
* `P` 采样单独模块化，支持 `weighted_fps`、`unweighted_fps`、`topk_nms` 三种显式 mode；实验配置可显式选择，默认配置文件提供 topk-NMS/FPS 变体。
* weighted-FPS 必须避免构造 `C×C` 距离矩阵；按 BOX/类别流式更新 min-distance。
* P 层对同一 `(batch, voxel_zyx)` 的多类别 C 记录去重，保留 `candidate_prob` 最大者，tie 保留原始 C 行号更小者；`anchor_counts_by_class` 为 `(B,K)`，`anchor_counts` 为 `(B,)`。
* `P -> C` 回写单独模块化，默认同类 three-nn，权重 `1 / d^p`，默认 `p=2`。
* `C` logit head 单独模块化，首版支持：
  * `voxel+interp` direct logits。
  * residual logits。
  * voxel 原始输出/特征默认 detach。
* density cube 保留给 `P` 的初始特征，04 默认使用 `cube_size=11`、两层 stride=2 下采样 Conv3d、普通 Conv3d blocks、GAP 和 Linear；`pseudo_feat` 不拼接 candidate class/prob。
* Attention 中 real 与 pseudo 互相可见。
* QKV/FFN 默认按 type 分参。
* CPE 默认只做 real-real 与 pseudo-pseudo，不做 real-pseudo CPE。
* 删除 sparseconv 风格 CPE，直接改造现有 Block/SerializedAttention，不新增长期包装层。
* `C` 生成 warmup 不新增专用用户参数，复用现有 scheduler warmup step；warmup 期间固定 per-class topk。
* `C` 生成 warmup 后通过 `selection_mode` 选择 `adaptive_threshold` 或 `recorded_threshold`。
* 删除 `max_candidate_voxels_per_box` 语义，只保留 `max_candidate_voxels_per_class`，避免跨类别总 cap 重新压制小类。
* `candidate_logits` 在 03 阶段默认 detach；05 若要对 voxel logits 反传，必须在 sparse refine head 中显式重新 gather non-detached logits。

## 4. 子计划清单

| 顺序 | 文件                                                                           | 目标                                                       |
| -- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1  | [01-clean-stage1-and-cpe.md](01-clean-stage1-and-cpe.md)                     | 清理 forward、删除旧伪原子逻辑、删除 sparseconv CPE、抽出 Stage1AtomHead  |
| 2  | [02-typed-point-core.md](02-typed-point-core.md)                             | 直接改造 Block/SerializedAttention，支持 type-aware QKV/FFN/CPE |
| 3  | [03-sparse-candidate-C.md](03-sparse-candidate-C.md)                         | 从 voxel logits 生成候选体素集合 C，维护阈值扩张策略                       |
| 4  | [04-anchor-P-sampling-and-features.md](04-anchor-P-sampling-and-features.md) | 从 C 逐类 weighted-FPS 采样 P，并为 P 构造 density cube 初始特征       |
| 5  | [05-anchor-to-candidate-refine.md](05-anchor-to-candidate-refine.md)         | P 点云互作、P->C three-nn 回写、C sparse logit head              |
| 6  | [06-loss-metrics-configs.md](06-loss-metrics-configs.md)                     | C 上 loss、best F1/PR-AUC、配置和消融矩阵                          |

## 5. 模块边界

建议新增模块：

```text
src/model/sparse_refine/
  candidate_set.py          # C 生成
  anchor_sampler.py         # C -> P
  density_cube.py           # P 初始特征
  interpolation.py          # P -> C
  sparse_refine_head.py     # C logits
```

伪原子行为统一由：

```text
src/model/pseudo_atoms.py
```

管理，包括：

* P 的 batch/mask/split 注入。
* P 的坐标字段。
* P 的 pseudo mask。
* P 与真实原子的 mixed layout。

`candidate_set.py` 管理的是 sparse voxel 候选 `C`，不是伪原子。

## 6. 关键变量对齐契约

> [!IMPORTANT]
> 本节契约是可维护的；但任何修改都必须全量同步更新对应源码文件开头的“对齐契约”、本节、调用点和相关测试，禁止只改其中一处。

### 6.1 维度记号

| 记号 | 意义 |
|---|---|
| `B` | batch 内 BOX 数量。 |
| `N_real` | 当前 batch 中真实原子总数，等于 `sum(real_counts)`。 |
| `N_pseudo` | 当前 batch 中 P anchor 总数，等于 `sum(pseudo_counts)`。 |
| `N_all` | mixed 点总数，`N_real + N_pseudo`。 |
| `F_atom` | atom/P anchor 输入特征维度，必须满足 real 与 pseudo 一致。 |
| `C_point` | point backbone 输出通道数。 |
| `C_hidden` | atom head shared hidden 维度。 |
| `C_pseudo` | `pseudo_feature` 输出通道数。 |

### 6.2 `src/model/sparse_refine/candidate_set.py`：候选集合 C 契约

源码同步位置：[src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py)。

`C` 是 sparse voxel candidate set，不是伪原子。`candidate_set.py` 只管理离散 voxel 候选行，不输出 P anchor 坐标，不进入 point backbone。

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `candidate_voxel_zyx` | `torch.Tensor`，`int64/long` | `(sumC, 3)` | 候选 voxel 离散坐标，轴顺序 `(z, y, x)`，可直接索引 `ligand_dist_map` 与 `voxel_valid_mask`。 |
| `candidate_batch_index` | `torch.Tensor`，`int64/long` | `(sumC,)` | 每个候选行所属 BOX 索引。 |
| `candidate_class` | `torch.Tensor`，`int64/long` | `(sumC,)` | 每个候选行的前景类别 ID，来自显式配置 `candidate_class_ids`。 |
| `candidate_prob` | `torch.Tensor`，floating | `(sumC,)` | 当前候选类概率；单通道 logits 使用 sigmoid，多通道 logits 使用 softmax 对应类。 |
| `candidate_logits` | `torch.Tensor`，floating | `(sumC, 1)` 或 `(sumC, C_cls)` | 对应 voxel 的原始 ligand logits，03 阶段默认 detach。 |
| `candidate_counts` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 的候选行数；按 `(voxel, class)` 行统计，不按唯一 voxel 统计。 |
| `candidate_counts_by_class` | `torch.Tensor`，`int64/long` | `(B, num_candidate_classes)` | 每个 BOX、每个候选类的候选行数。 |
| `candidate_p_sampling_by_class` | `torch.Tensor`，floating | `(B, num_candidate_classes)` | 当前 batch 每个 BOX、每类实际使用的候选截断阈值。 |
| `candidate_target_counts_by_class` | `torch.Tensor`，`int64/long` | `(B, num_candidate_classes)` | 当前 batch 每个 BOX、每类阈值扩张后、cap 前的目标候选数；warmup 时等于 warmup topc。 |

关键约束：

1. `C` 的主键是 `(candidate_batch_index, candidate_voxel_zyx, candidate_class)`；允许同一个 voxel 因多个前景类同时入选而出现多行。
2. `candidate_class_ids` 必须显式配置并严格校验。`voxel_logits_ligand.shape[1] == 1` 时只允许 `[1]`；多通道时每个 class id 必须满足 `1 <= class_id < C_cls`。
3. `selection_mode` 只允许 `"adaptive_threshold"` 或 `"recorded_threshold"`。
4. `p_best_by_class` 与 `p_sampling_by_class` 内部和日志均使用 `_by_class` 向量语义；二分类也不使用标量特例。
5. `p_sampling_by_class` 是 validation 全体有效体素 pos/neg histogram 近似分位/topk 阈值；`adaptive_threshold` 模式下当前 BOX 会产生局部 `candidate_p_sampling_by_class`。
6. 不使用 `max_candidate_voxels_per_box`；候选数量只由 `warmup_topc_per_class`、`adaptive_expand_factor` 和 `max_candidate_voxels_per_class` 控制。

wrapper 阈值缓存契约：

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `_cached_voxel_ligand_p_best_by_class` | `torch.Tensor | None` | `(num_candidate_classes,)` | validation best-F1 对应阈值；二分类也用向量。 |
| `_cached_voxel_ligand_p_sampling_by_class` | `torch.Tensor | None` | `(num_candidate_classes,)` | validation 全体有效体素 histogram 近似分位/topk 得到的全局候选截断阈值。 |
| `_cached_voxel_ligand_best_f1_by_class` | `torch.Tensor | None` | `(num_candidate_classes,)` | 当前缓存阈值对应的 best-F1。 |

缓存更新时机：每次 validation loop 结束时基于 per-class pos/neg histogram 更新并记录，不按 epoch 语义限制；项目支持 epoch 中间 validation。无有效统计类别保持旧缓存；无旧缓存时使用显式 initial；`p_best_by_class` 与 `p_sampling_by_class` 只有成对 finite 时才进入 Lightning checkpoint metadata，并在恢复后同步给 backbone candidate builder；checkpoint 中存在但 NaN/Inf 或长度不匹配的 cache 必须 fail-fast。DDP 下 histogram 做 sum 聚合，不收集逐体素概率数组。

### 6.3 `src/model/pseudo_atoms.py`：`pseudo_dict` 与 mixed layout

源码同步位置：[src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py)。

mixed layout 固定为每个 BOX 内 `[real_i..., pseudo_i...]`，不得把 pseudo 打散到 real 序列中间。

| 字段 | 类型 | shape | 必需 | 意义 |
|---|---|---|---|---|
| `pseudo_counts` | `torch.Tensor`，`int64/long` | `(B,)` | 是 | 每个 BOX 的 P anchor 数，`sum` 等于 `N_pseudo`。 |
| `pseudo_batch_index` | `torch.Tensor`，`int64/long` | `(N_pseudo,)` | 是 | 每个 P anchor 所属 BOX 索引，必须按 BOX 分组且与 `pseudo_counts` 一致。 |
| `pseudo_coord_centered_world` | `torch.Tensor`，floating | `(N_pseudo, 3)` | 是 | P anchor 以 BOX 中心为原点的世界坐标，轴顺序 `(x, y, z)`。 |
| `pseudo_coord_local_voxel` | `torch.Tensor`，floating | `(N_pseudo, 3)` | 是 | P anchor 的 corner 语义连续局部体素坐标，轴顺序 `(x, y, z)`。 |
| `pseudo_coord_world` | `torch.Tensor`，floating | `(N_pseudo, 3)` | 是 | P anchor 的绝对世界坐标，轴顺序 `(x, y, z)`。 |
| `pseudo_feat` | `torch.Tensor`，floating | `(N_pseudo, F_atom)` | 是 | P anchor 初始点特征，`F_atom` 必须等于 real batch 中 `atom_feat.shape[1]`。 |
| `pseudo_is_in_core_box` | `torch.Tensor`，bool | `(N_pseudo,)` | 否 | P anchor 是否在 core box 内；缺省时 `inject_pseudo_atoms()` 按全 True 处理。 |
| `pseudo_anchor_class` | `torch.Tensor`，`int64/long` | `(N_pseudo,)` | 否 | P anchor 的候选类别索引，类别顺序由候选 C 模块定义。 |
| `pseudo_anchor_voxel_zyx` | `torch.Tensor`，`int64/long` | `(N_pseudo, 3)` | 否 | P anchor 来源体素坐标，轴顺序 `(z, y, x)`。 |
| `pseudo_source_candidate_index` | `torch.Tensor`，`int64/long` | `(N_pseudo,)` | 否 | P anchor 对应的候选 C 行索引。 |

`inject_pseudo_atoms()` 输出字段：

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `real_mask` | `torch.Tensor`，bool | `(N_all,)` | mixed-only 字段，True 表示 real atom。 |
| `pseudo_mask` | `torch.Tensor`，bool | `(N_all,)` | mixed-only 字段，True 表示 P anchor。 |
| `atom_valid_mask` | `torch.Tensor`，bool | `(N_all,)` | P anchor 槽位固定为 False，不参与 real atom 监督。 |
| `atom_label` | `torch.Tensor`，`int64/long` | `(N_all,)` | P anchor 槽位固定为 0，仅作占位。 |
| `atom_global_indices` | `torch.Tensor`，`int64/long` | `(N_all,)` | P anchor 槽位固定为 -1，表示无真实原子全局索引。 |

### 6.4 `src/model/stage1_model.py`：forward 输入/输出契约

源码同步位置：[src/model/stage1_model.py](../../../../src/model/stage1_model.py)。

`forward()` 输入 batch 在 `_run_embed_head_once()` 之前必须是 real-only；伪原子不得进入 embed head。P anchors 只允许在最后一次 recycle 的 `_prepare_pseudo_batch()` 后进入 point backbone。03 阶段 `_prepare_pseudo_batch()` 可以生成并输出 C 元数据，但仍必须返回 real-only batch、`None` layout，不注入 P。

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `voxel_grid` | `torch.Tensor`，floating | `(B, C_in, D, H, W)` | voxel backbone 输入密度/特征体。 |
| `box_shape_zyx` | `torch.Tensor`，`int64/long` | `(B, 3)` | 每个 BOX 的体素尺寸，轴顺序 `(z, y, x)`。 |
| `voxel_size_world` | `torch.Tensor`，floating | `(B, 3)` | 每个 voxel 的世界坐标尺寸，轴顺序 `(x, y, z)`。 |
| `atom_feat` | `torch.Tensor`，floating | `(N_real, F_atom)` 或 mixed `(N_all, F_atom)` | 点分支输入原子/P anchor 特征。 |
| `atom_coord_centered_world` | `torch.Tensor`，floating | `(N_real, 3)` 或 mixed `(N_all, 3)` | 以 BOX 中心为原点的世界坐标，轴顺序 `(x, y, z)`。 |
| `atom_coord_local_voxel` | `torch.Tensor`，floating | `(N_real, 3)` 或 mixed `(N_all, 3)` | corner 语义连续局部体素坐标，轴顺序 `(x, y, z)`。 |
| `atom_coord_world` | `torch.Tensor`，floating | `(N_real, 3)` 或 mixed `(N_all, 3)` | 绝对世界坐标，轴顺序 `(x, y, z)`。 |
| `atom_batch_index` | `torch.Tensor`，`int64/long` | `(N_real,)` 或 mixed `(N_all,)` | 每个点所属 BOX 索引。 |
| `atom_offsets` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 在展平点序列中的结束偏移。 |
| `atom_counts` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 的点数；wrapper-facing 输出必须恢复为 real-only counts。 |
| `atom_label` | `torch.Tensor`，`int64/long` | `(N_real,)` 或 mixed `(N_all,)` | real atom 监督标签；P anchor 槽位只允许作为占位 0。 |
| `atom_valid_mask` | `torch.Tensor`，bool | `(N_real,)` 或 mixed `(N_all,)` | real atom 监督掩码；P anchor 槽位必须为 False。 |
| `atom_is_in_core_box` | `torch.Tensor`，bool | `(N_real,)` 或 mixed `(N_all,)` | 点是否在 core box 内。 |
| `atom_global_indices` | `torch.Tensor`，`int64/long` | `(N_real,)` 或 mixed `(N_all,)` | 真实原子全局索引；P anchor 槽位为 -1。 |
| `real_mask` | `torch.Tensor`，bool | `(N_all,)` | mixed-only 字段，True 表示 real atom。 |
| `pseudo_mask` | `torch.Tensor`，bool | `(N_all,)` | mixed-only 字段，True 表示 P anchor。 |

`forward()` 输出契约：

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `atom_logits` | `torch.Tensor | None`，floating | `(N_real, atom_logit_dim)` | wrapper-facing real-only atom logits。 |
| `atom_target` | `torch.Tensor | None`，`int64/long` | `(N_real,)` | wrapper-facing real-only atom 标签。 |
| `atom_valid_mask` | `torch.Tensor | None`，bool | `(N_real,)` | wrapper-facing real-only 监督掩码。 |
| `atom_counts` | `torch.Tensor | None`，`int64/long` | `(B,)` | wrapper-facing real-only counts。 |
| `atom_tokens` | `torch.Tensor | None`，floating | `(N_all, C_token)` | atom head token projection 前输入；mixed 路径保留全点。 |
| `atom_hidden` | `torch.Tensor | None`，floating | `(N_all, C_hidden)` | atom head shared attention 输出；mixed 路径保留全点。 |
| `pseudo_feature` | `torch.Tensor | None`，floating | `(N_pseudo, C_pseudo)` | P anchor refined feature；01/03 阶段或 real-only 路径为 None。 |
| `pseudo_logits` | `torch.Tensor | None`，floating | `(N_pseudo, pseudo_ligand_logit_dim)` | P 后置 ligand 区域归属 logits；`enable_pseudo_ligand_head=False` 或 real-only 路径为 None。 |
| `pseudo_logits_front` | `torch.Tensor | None`，floating | `(N_pseudo, pseudo_ligand_logit_dim)` | P 前置 ligand 区域归属 logits；`enable_pseudo_ligand_head` 且 `enable_atom_head_front` 且 mixed 路径存在，否则键缺失。 |
| `pseudo_voxel_zyx` | `torch.Tensor`，`int64/long` | `(N_pseudo, 3)` | P anchor home 体素离散索引，轴顺序 `(z, y, x)`；mixed 路径存在，供 wrapper 采样 P 监督。 |
| `pseudo_batch_index` | `torch.Tensor`，`int64/long` | `(N_pseudo,)` | P anchor 所属 BOX 索引；mixed 路径存在。 |
| `candidate_voxel_zyx` 等 C 字段 | `dict[str, torch.Tensor]` | 见 6.2 | 03 阶段开始可选输出；不改变 point/atom real-only 路径。 |

### 6.5 `src/model/stage1_atom_head.py`：atom head 输入/输出契约

源码同步位置：[src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py)。

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `pseudo_mask` | `torch.Tensor | None`，bool | `(N_all,)` | True 表示 P anchor，False 表示 real atom；None 表示 real-only 路径。 |
| `point_feat` | `torch.Tensor`，floating | `(N_all, C_point)` | 最后一轮 point backbone 输出特征，mixed 路径按 `pseudo_atoms.py` mixed layout 排列。 |
| `point_state["coord"]` | `torch.Tensor`，floating | `(N_all, 3)` | Point/Block 使用的点坐标，轴顺序 `(x, y, z)`。 |
| `point_state["batch"]` | `torch.Tensor`，`int64/long` | `(N_all,)` | 每个点所属 BOX 索引。 |
| `point_state["offset"]` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 在展平点序列中的结束偏移。 |
| `point_state["grid_size"]` | `float` | 标量 | PTV3 序列化/网格化使用的点云 grid size。 |
| `point_state["grid_coord"]` | `torch.Tensor`，int32/int64 | `(N_all, 3)` | 可选字段，离散网格坐标。 |
| `atom_coord_centered_world` | `torch.Tensor`，floating | `(N_all, 3)` | token 可选拼接的 centered-world 坐标，轴顺序 `(x, y, z)`。 |
| `atom_valid_mask` | `torch.Tensor`，bool | `(N_all,)` | real atom 监督掩码；P anchor 槽位必须为 False。 |

输出字段：

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `atom_tokens` | `torch.Tensor`，floating | `(N_all, C_point)` 或 `(N_all, C_point + 4)` | token projection 前输入；`append_coord_mask=True` 时追加 xyz 与 valid mask。 |
| `atom_hidden` | `torch.Tensor`，floating | `(N_all, C_hidden)` | shared attention stack 输出，保留 mixed 全点顺序。 |
| `atom_logits` | `torch.Tensor`，floating | `(N_real, atom_logit_dim)` | 只对 real atom 输出的监督 logits。 |
| `pseudo_feature` | `torch.Tensor | None`，floating | `(N_pseudo, C_pseudo)` | 只对 P anchor 输出；real-only 路径为 None。 |
| `pseudo_logits` | `torch.Tensor | None`，floating | `(N_pseudo, pseudo_ligand_logit_dim)` | P 后置 ligand 区域归属 logits（`pseudo_feature` 上的轻量 MLP）；`enable_pseudo_ligand_head=False` 或 real-only 路径为 None。 |

> P 监督深监督：`enable_pseudo_ligand_head` 开时，atom head 在 `pseudo_feature` 上接后置 `pseudo_logit_head`（→ `pseudo_logits`），`stage1_model` 在 `point_feat_raw` 的 P 槽位接前置 `pseudo_logit_head_front`（→ `pseudo_logits_front`）；两头单通道 sigmoid，用 `prior_prob_point_ligand` 初始化末层 bias（受 `prior_prob_init_enabled` 总开关控制）。P target 由 wrapper 用 `target_from_ligand_dist_map` 在 P home 体素采样、`voxel_valid_mask` gate；`pseudo_loss_front_weight>0` 而 `pseudo_logits_front` 缺失时 wrapper fail-fast raise。

> detach 路由（3 正交开关）：`detach_{real,pseudo}_point_feat_into_atomhead` 控制真实/伪原子槽位喂后置 atom head 是否 detach；`detach_pseudo_point_feat_into_refine` 单独控制 P 喂 refine 的 `P_point_backbone_feat`（从 `point_feat_raw` 取出再 detach）。real 进 refine 无消费者，故不引入该开关。

### 6.6 `src/model/stage1_point_backbone.py` 与 `src/model/PTV3bakcbone/model.py`：点分支契约

源码同步位置：[src/model/stage1_point_backbone.py](../../../../src/model/stage1_point_backbone.py)、[src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py)。

| 字段/配置 | 类型 | shape/允许值 | 意义 |
|---|---|---|---|
| `embedding_impl` | `str` | 只支持 `"pointconv"` | PTV3 embedding 实现；不得恢复 sparseconv stem。 |
| `cpe_impl` | `str` | `"pointconv"` 或 `"none"` | Block CPE 实现；不得恢复 sparseconv CPE。 |
| `enc_cpe_kernel_size` / `dec_cpe_kernel_size` | `Sequence[int]` | legacy 字段 | pointconv CPE 不消费这些 kernel size，仅为配置兼容保留。 |
| `atom_feat` | `torch.Tensor`，floating | `(N, F_atom)` | real atom 或 mixed 全点特征。 |
| `atom_coord_centered_world` | `torch.Tensor`，floating | `(N, 3)` | 点坐标，轴顺序 `(x, y, z)`。 |
| `atom_batch_index` | `torch.Tensor`，`int64/long` | `(N,)` | 每个点所属 BOX 索引。 |
| `atom_offsets` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 在展平点序列中的结束偏移。 |
| `recycle_in` | `torch.Tensor | None`，floating | `(N, C_recycle)` | 上一轮 point recycle 状态；mixed 最后一轮由 Stage1 主模型补齐 P anchor 槽位。 |
| `point_feat` | `torch.Tensor`，floating | `(N, C_point)` | 点分支最终输出特征。 |
| `point_state["coord"]` | `torch.Tensor`，floating | `(N, 3)` | atom head 复用的点坐标。 |
| `point_state["batch"]` | `torch.Tensor`，`int64/long` | `(N,)` | atom head 复用的 BOX 索引。 |
| `point_state["offset"]` | `torch.Tensor`，`int64/long` | `(B,)` | atom head 复用的结束偏移。 |
| `point_state["grid_size"]` | `float` | 标量 | atom head/PTV3 复用的点云 grid size。 |
| `point_state["grid_coord"]` | `torch.Tensor`，int32/int64 | `(N, 3)` | 可选字段，PTV3 离散网格坐标。 |
| `point_recycle_out` | `torch.Tensor`，floating | `(N, C_recycle)` | 下一轮 recycle 输入；mixed 最后一轮后由 Stage1 主模型裁成 real-only。 |
| `point_feature_dict` | `dict[str, torch.Tensor]` | 每项 `(N, C_name)` | 请求导出的命名点特征。 |

PTV3 `Point` 可用字段：

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `feat` | `torch.Tensor`，floating | `(N, C)` | 当前点特征。 |
| `coord` | `torch.Tensor`，floating | `(N, 3)` | 点坐标，轴顺序 `(x, y, z)`。 |
| `batch` | `torch.Tensor`，`int64/long` | `(N,)` | 每个点所属 BOX/样本索引。 |
| `offset` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX/样本在展平点序列中的结束偏移。 |
| `grid_size` | `float` | 标量 | 点云离散化 grid size。 |
| `grid_coord` | `torch.Tensor`，int32/int64 | `(N, 3)` | 可选字段，由 `coord/grid_size` 得到的离散坐标。 |
| `serialized_code` | `torch.Tensor`，`int64/long` | `(K, N)` | K 个序列化顺序对应的编码。 |
| `serialized_order` | `torch.Tensor`，`int64/long` | `(K, N)` | 每个序列化顺序下的排序后索引。 |
| `serialized_inverse` | `torch.Tensor`，`int64/long` | `(K, N)` | 每个序列化顺序下的逆索引。 |
| `pseudo_mask` | `torch.Tensor`，bool | `(N,)` | 可选字段，True 表示 P anchor；01 阶段只透传给后续 type-aware 改造。 |

### 6.7 `src/model/stage1_embed_head.py`：embed head real-only 契约

源码同步位置：[src/model/stage1_embed_head.py](../../../../src/model/stage1_embed_head.py)。

embed head 只允许处理 real-only atom，不允许接收 `pseudo_mask`、`real_mask` 或 P anchor 字段。

| 字段 | 类型 | shape | 意义 |
|---|---|---|---|
| `atom_feat` | `torch.Tensor`，floating | `(N_real, F_atom)` | real atom 原始特征。 |
| `atom_coord_centered_world` | `torch.Tensor`，floating | `(N_real, 3)` | 以 BOX 中心为原点的世界坐标，轴顺序 `(x, y, z)`。 |
| `atom_batch_index` | `torch.Tensor`，`int64/long` | `(N_real,)` | 每个 real atom 所属 BOX 索引。 |
| `atom_offsets` | `torch.Tensor`，`int64/long` | `(B,)` | 每个 BOX 在 real-only 展平序列中的结束偏移。 |
| `atom_coord_local_voxel` | `torch.Tensor`，floating | `(N_real, 3)` | corner 语义连续局部体素坐标，轴顺序 `(x, y, z)`。 |
| `box_shape_zyx` | `torch.Tensor`，`int64/long` | `(B, 3)` | BOX 体素尺寸，轴顺序 `(z, y, x)`。 |
| `voxel_size_world` | `torch.Tensor`，floating | `(B, 3)` | 每个 voxel 的世界坐标尺寸，轴顺序 `(x, y, z)`。 |
| `atom_is_in_core_box` | `torch.Tensor`，bool | `(N_real,)` | real atom 是否在 core box 内。 |
| `global_keep_mask` | `torch.Tensor`，bool | `(N_real,)` | 输出字段，True 表示原始 real atom 被 embed 裁剪后保留。 |
| `embed_point_feat` | `torch.Tensor | None`，floating | `(N_keep, embed_point_out_channels)` | 裁剪后点分支特征；未启用点输出时为 None。 |
| `voxel_pdb_embed_grid` | `torch.Tensor | None`，floating | `(B, C_embed, D, H, W)` | scatter 后体素嵌入；未启用体素输出时为 None。 |

## 7. 通用阅读要求

每个 agent 执行子计划前必须阅读：

1. [../../../../CLAUDE.md](../../../../CLAUDE.md)
2. 本总控计划。
3. 自己负责的子计划。
4. 子计划列出的源码。

涉及字段、shape、类别、mask、坐标语义时，以代码和真实配置为准。

## 7. 旧计划状态

旧目录 [../tri_ligand_refine/](../tri_ligand_refine/) 保留为历史和反馈来源，不再作为执行入口。

本目录 `tri_ligand_sparse_refine/` 是新的执行入口。
