# 03：候选体素集合 C 生成模块

> [!IMPORTANT]
> 本阶段初版允许同一 voxel 按类别保留多行；第 05 阶段实现后，`SparseCandidateSetBuilder` 已在输出前按 `(batch, voxel_zyx)` 合并为唯一 C，并对冲突提名随机选择路由类别。本文下方旧描述若与 [05-anchor-to-candidate-refine.md](05-anchor-to-candidate-refine.md) 或现行源码冲突，以第 05 阶段契约与源码为准。

## 背景 / 目标

本计划是三分类 ligand sparse refine 的第 3 个实施步骤。目标是在 01/02 已完成 Stage1 清理与 typed point core 后，从 `voxel_logits_ligand` 生成 sparse candidate voxel set `C`，并补齐 voxel ligand 分支的 best-F1 threshold 统计与缓存。

当前代码事实：

| 事实 | 位置 |
|---|---|
| final recycle 唯一 C/P 准备入口是 `_prepare_pseudo_batch()`，当前直接返回 real-only batch、`pseudo_layout=None`、空 `pseudo_outputs`。 | [src/model/stage1_model.py:695-713](../../../../src/model/stage1_model.py#L695-L713) |
| final 输出组装已经把 `**pseudo_outputs` 合入 `outputs`，因此 03 可只生成 C 元数据而不改变 point/atom 路径。 | [src/model/stage1_model.py:891-923](../../../../src/model/stage1_model.py#L891-L923) |
| voxel backbone 已输出 `voxel_logits_ligand`，其 channel 数由 `voxel_ligand_logit_dim` 配置控制。 | [src/model/stage1_voxel_backbone.py:304-323](../../../../src/model/stage1_voxel_backbone.py#L304-L323) |
| wrapper 已有 voxel ligand PR-AUC 更新、validation end compute/reset、checkpoint hook 与 warmup scheduler 解析点。 | [src/wrappers/voxel_point_stage1.py:615-648](../../../../src/wrappers/voxel_point_stage1.py#L615-L648), [src/wrappers/voxel_point_stage1.py:959-1019](../../../../src/wrappers/voxel_point_stage1.py#L959-L1019), [src/wrappers/voxel_point_stage1.py:1025-1053](../../../../src/wrappers/voxel_point_stage1.py#L1025-L1053) |
| collate 后 `voxel_valid_mask` 的主形状是 `(B,D,H,W)`；现有 wrapper/loss 有时也兼容 `(B,1,D,H,W)`。 | [src/datasets/box_point_collate.py:21-29](../../../../src/datasets/box_point_collate.py#L21-L29), [src/datasets/box_point_collate.py:52-59](../../../../src/datasets/box_point_collate.py#L52-L59) |
| 多分类 ligand hard label 的权威语义是“前景通道距离取最小，小于 `hard_label_threshold` 则为对应 class id，否则为 0”。 | [src/modules/losses.py:934-962](../../../../src/modules/losses.py#L934-L962), [src/wrappers/voxel_point_stage1.py:255-279](../../../../src/wrappers/voxel_point_stage1.py#L255-L279) |

`C` 是 sparse voxel 监督/候选集合，不是伪原子，不进入 point backbone。本阶段只让 [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 的 `_prepare_pseudo_batch()` 生成并输出 C 元数据，仍返回 real-only batch、`pseudo_layout=None`，不采样 P、不注入 P。

本阶段完成后应满足：

1. `SparseCandidateSetBuilder` 从 `voxel_logits_ligand` 与 `voxel_valid_mask` 生成 C。
2. `candidate_class_ids` 显式配置、严格校验、可扩展；默认 tri 配置为 `[1, 2]`。
3. 长期支持 `voxel_ligand_logit_dim: 1 + sigmoid()` 二分类路径；多通道路径使用 softmax。
4. wrapper 在每次 validation loop 结束时统计并缓存 `p_best_by_class`、`p_sampling_by_class` 与 best-F1。
5. `p_best_by_class` / `p_sampling_by_class` 成对 finite 后进入 Lightning checkpoint metadata；恢复后可用于训练续跑与推理。
6. fixed topk 只允许在 scheduler warmup step 内使用；warmup 后由配置选择 `adaptive_threshold` 或 `recorded_threshold`。
7. warmup 外若 checkpoint/显式 initial 没有提供完整 finite 阈值，训练/验证/推理都必须 fail-fast，不静默退回 fixed topk。
8. 03 只输出 voxel index 级 C 字段，不输出 world/local 连续坐标；C→P 坐标转换留到 04。

> [!IMPORTANT]
> 本计划不实现 P anchor 采样、density cube、P→C 插值、sparse C logit head、refined C loss 或 refined C 指标。refined best-F1 / PR-AUC / 漏检惩罚指标留到 06。

## AI review 处理结论

| review 项 | 结论 | 计划修改 |
|---|---|---|
| BUG-1：YAML 中 `mode: threshold_expand` 与 constructor 不一致 | 采纳。`mode` 是死字段，会让 Hydra `instantiate()` 报 unexpected keyword。 | 从 `binary.yaml` / `tri.yaml` 删除 `mode`；constructor 不接收 `mode`。 |
| BUG-2：F1 阈值计算中的 `tp` / `fp` 符号歧义 | 采纳。裸 `tp` 容易被误实现为单 bin count。 | 伪代码统一使用 `tp_at_threshold` / `fp_at_threshold` / `fn_at_threshold`。 |
| BUG-3：`hard_label_threshold is None` 未定义 | 采纳 fail-fast 方案。threshold stats 必须有距离阈值 hard label。 | `_update_voxel_ligand_best_f1_stats()` 在 candidate threshold stats 启用且 `hard_label_threshold is None` 时抛 `ValueError`；原 PR-AUC fallback 保持只服务旧非 candidate 用途。 |
| BUG-4：builder class ids 与 wrapper histogram class ids 可能不一致 | 采纳问题，调整方案。wrapper 不再自行推断 `1..C-1` 作为 threshold 类集合，而是读取 builder 的 `candidate_class_ids`。 | 新增 `_resolve_sparse_candidate_class_ids()`；二分类只允许 `[1]`，多分类要求每个 id 满足 `1 <= class_id < voxel_ligand_loss.num_classes`。允许显式子集。 |
| STYLE-1：`selection_mode` 默认参数 | 采纳。 | `SparseCandidateSetBuilder.__init__()` 中 `selection_mode` 必传；YAML 必须显式写。 |
| DESIGN-2：histogram 生命周期 | 采纳并收敛。 | builder 启用时在 wrapper `__init__()` 注册/初始化 histogram buffer；`on_validation_epoch_start()` 和 compute 后都清零。 |
| DESIGN-6：builder 内部 detach | 采纳。 | builder `forward()` 在 `torch.no_grad()` 下运行，内部对 gathered logits 和概率 detach；调用方不负责 detach。 |
| DESIGN-9：通用 metric 打印 callback 可能刷屏 | 采纳。 | 03 默认不新增 `ValidationMetricPrintCallback`；只在 wrapper rank0 打印 threshold cache 更新，普通 metric 继续交给 RichProgressBar / logger。 |
| 其余合理设计项 | 保留。 | 保留 histogram 近似、scheduler warmup 复用、同 voxel 多类重复、双 selection mode、checkpoint metadata 等决策。 |

## Grill 后校正

1. builder 启用后若 `voxel_logits_ligand is None`，必须 fail-fast，不静默返回空 C；只有 builder 未启用时才保持空 `pseudo_outputs`。
2. fixed topk 只允许在 scheduler warmup 内使用；`global_step >= candidate_warmup_steps` 后若 threshold cache 不完整，训练/验证/推理都必须 fail-fast，不做 bootstrap 回退。
3. 不新增 prefit threshold calibration；现有训练配置依赖 `warmup_ratio=0.025` 与 `val_per_epoch=3`，确保第一次正式 validation 在 warmup 内产生阈值。
4. Lightning sanity validation 与 batch-size tuning 不更新 candidate threshold cache，且结束后清空 histogram；tuning 只允许在 scheduler warmup window 内 fixed topk 通过显存探测。
5. checkpoint 只在 `p_best_by_class` 与 `p_sampling_by_class` 都完整 finite 时成对保存；加载 checkpoint 中存在但含 NaN/Inf 或长度不匹配的 cache 必须 fail-fast。
6. 计划更新只记录上述冲突与校正，不大幅重写后文，后续实现以本节为准。

## 必读文件

* [00-master.md](00-master.md)
* [src/model/stage1_model.py](../../../../src/model/stage1_model.py)
* [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)
* [src/model/stage1_voxel_backbone.py](../../../../src/model/stage1_voxel_backbone.py)
* [src/datasets/box_point_collate.py](../../../../src/datasets/box_point_collate.py)
* [src/modules/losses.py](../../../../src/modules/losses.py)
* [configs/base.yaml](../../../../configs/base.yaml)
* [configs/model/default.yaml](../../../../configs/model/default.yaml)
* [configs/experiment/tri001_tunedloss.yaml](../../../../configs/experiment/tri001_tunedloss.yaml)
* [tests/model/test_stage1_model.py](../../../../tests/model/test_stage1_model.py)
* [tests/test_multiclass_ligand_wrapper.py](../../../../tests/test_multiclass_ligand_wrapper.py)
* [tests/test_multiclass_voxel_backbone.py](../../../../tests/test_multiclass_voxel_backbone.py)

## 已有可复用代码

| 已有代码 | 位置 | 可复用能力 |
|---|---|---|
| final recycle 伪原子准备点 | [src/model/stage1_model.py:695-713](../../../../src/model/stage1_model.py#L695-L713) `_prepare_pseudo_batch()` | 03 在这里读取 `voxel_logits_ligand` 生成 C；仍返回 real-only batch 和 `None` layout。 |
| final 输出合并 | [src/model/stage1_model.py:906-923](../../../../src/model/stage1_model.py#L906-L923) | `**pseudo_outputs` 已进入最终输出，可直接透传 C 字段。 |
| voxel ligand logits 输出 | [src/model/stage1_voxel_backbone.py:304-323](../../../../src/model/stage1_voxel_backbone.py#L304-L323) `Stage1VoxelBackbone.forward()` | 已输出 `voxel_logits_ligand`，shape 可为 `(B,1,D,H,W)` 或 `(B,C,D,H,W)`。 |
| ligand hard label 构造 | [src/wrappers/voxel_point_stage1.py:255-279](../../../../src/wrappers/voxel_point_stage1.py#L255-L279) `_ligand_target_from_dist()` | validation best-F1 可复用同一距离阈值 hard-label 语义。 |
| 多分类距离图 hard label | [src/modules/losses.py:934-962](../../../../src/modules/losses.py#L934-L962) `AdaptiveClassificationCompositeLoss._target_from_multiclass_dist()` | 多分类 `ligand_dist_map` 的类别 ID 生成语义应保持一致。 |
| validation loop 汇总点 | [src/wrappers/voxel_point_stage1.py:959-987](../../../../src/wrappers/voxel_point_stage1.py#L959-L987) `on_validation_epoch_end()` | 每次 validation loop 结束后计算/记录 AP；03 在同一生命周期里更新 best-F1 threshold 与缓存。 |
| checkpoint metadata hook | [src/wrappers/voxel_point_stage1.py:989-1019](../../../../src/wrappers/voxel_point_stage1.py#L989-L1019) `on_save_checkpoint()` / `on_load_checkpoint()` | 03 在这里保存/恢复 `p_best_by_class` 与 `p_sampling_by_class`。 |
| validation 调度 | [src/train.py:638-647](../../../../src/train.py#L638-L647), [src/train.py:733-750](../../../../src/train.py#L733-L750) | 项目已支持每个 epoch 内多次 validation；03 的 threshold 更新按每次 validation loop 结束触发。 |
| scheduler warmup 解析 | [src/wrappers/voxel_point_stage1.py:1025-1053](../../../../src/wrappers/voxel_point_stage1.py#L1025-L1053) `_resolve_warmup_steps()` | candidate warmup 不新增用户参数，复用现有训练 scheduler warmup step。 |
| 现有 wrapper 测试风格 | [tests/test_multiclass_ligand_wrapper.py](../../../../tests/test_multiclass_ligand_wrapper.py) | threshold hard-label 与 wrapper 小单测放在 top-level [tests/](../../../../tests/) 下。 |

## 设计决策

### 1. `C` 的主键是 `(batch, voxel_zyx, candidate_class)`

允许同一个 `(batch, z, y, x)` 因多个前景类别都入选而出现多行。`candidate_counts` 与 `candidate_counts_by_class` 均按候选行统计，不按唯一 voxel 数统计。

理由：每类独立生成 C，避免 small molecule 与 metal 互相压制。同一 voxel 的类别冲突留给后续 sparse refine head 与指标分析处理。

> [!NOTE]
> 04 从 C 采样 P anchor 时需要显式决定是否对同坐标多类别候选去重；03 不在 C 层去重。

### 2. `p_sampling` 替代旧计划中的 `p_final`

命名统一为：

| 名称 | shape | 语义 |
|---|---|---|
| `p_best_by_class` | `(num_candidate_classes,)` | validation 上 best-F1 对应的 voxel ligand 阈值。 |
| `p_sampling_by_class` | `(num_candidate_classes,)` | validation 上满足 `count(prob > p_sampling) ~= adaptive_expand_factor * count(prob > p_best)` 的全局候选截断阈值。 |
| `candidate_p_sampling_by_class` | `(B, num_candidate_classes)` | 当前 batch 生成 C 实际使用的 per-box/class 截断值；`recorded_threshold` 下每个 BOX 复制全局 `p_sampling_by_class`。 |

内部状态、输出字段、日志名统一使用 `_by_class`，二分类也不实现标量特例。

### 3. wrapper threshold 类集合以 builder 的 `candidate_class_ids` 为准

`candidate_class_ids` 只在 [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py) 的 builder 配置里定义一次。wrapper 通过 backbone 读取同一 tuple 并只为这些类别维护 best-F1 / sampling threshold histogram。

| logits 路径 | 合法 `candidate_class_ids` | wrapper 校验 |
|---|---|---|
| 单通道 sigmoid `(B,1,D,H,W)` | `[1]` | `voxel_ligand_loss.num_classes` 若存在应为 2；threshold stats 只记录 class id 1。 |
| 多通道 softmax `(B,C,D,H,W)` | `1 <= class_id < C` 的显式列表，可为前景子集 | 每个 class id 必须落在 `1..voxel_ligand_loss.num_classes-1`；运行时 logits channel 数必须覆盖所有 id。 |

这样既解决 builder 与 histogram 不一致的问题，又保留未来只 refine 某个前景子集的能力。

### 4. 同时记录全局阈值，并支持两种候选选择模式

统计层始终记录：

1. `p_best_by_class`：best-F1 阈值。
2. `p_sampling_by_class`：全验证集有效体素 histogram 近似 topk / 分位阈值。

候选生成层在 fixed topk 阶段结束后用配置选择：

| `selection_mode` | 行为 | 推荐用途 |
|---|---|---|
| `adaptive_threshold` | 每个 BOX 内先计算 `n_best_box = count(prob > p_best_by_class[class])`，再令 `n_target_box = ceil(n_best_box * adaptive_expand_factor[class])`，用该 BOX topk 选出候选并记录局部 `p_sampling_box`。 | 训练默认，用每个 BOX 的分布自适应候选数量。 |
| `recorded_threshold` | 直接用全局缓存 `p_sampling_by_class[class]` 对当前 BOX 做阈值筛选；若超过 per-class cap，再按概率 topk 裁剪。 | 推理默认或消融，复用训练/验证统计得到的全局候选阈值。 |

fixed topk 阶段只包括：

1. fit/sanity/tuning lifecycle 中 `global_step < candidate_warmup_steps`。
2. `candidate_warmup_steps > 0`，且 wrapper 明确同步 `allow_warmup_fixed_topk=True`。

warmup 外不允许“阈值缺失时自动 fixed topk”作为隐式回退；需要 threshold 模式但阈值缺失时直接报错。

### 5. 全局 `p_sampling_by_class` 使用全验证集有效体素 histogram 近似 topk 定义

对每个候选类：

1. 在 validation 有效体素上用 `p_best_by_class[class]` 计算近似 `n_best_total = count(prob > p_best)`。
2. 令 `n_sampling_total = ceil(n_best_total * adaptive_expand_factor[class])`。
3. 在全验证集该类有效体素概率 histogram 中，从高概率 bin 向低概率 bin 累计，找到累计数首次 `>= n_sampling_total` 的 bin。
4. 取该 bin lower-edge 作为 `p_sampling_by_class[class]`，偏保守地多保留候选。
5. 若该类本轮没有正例或没有有效体素，保持旧缓存；旧缓存不存在但显式 initial 存在时使用 initial；仍不存在则该类保持 unavailable，非 fixed topk threshold 模式下 fail-fast。

该定义保证验证集整体候选量接近 expand factor，而不是对 per-box 阈值做均值或中位数。

### 6. 不再使用 `max_candidate_voxels_per_box`

删除 per-box 总 cap，只保留 `max_candidate_voxels_per_class`。三分类通过 per-class topc 与 per-class cap 近似体现 metal:small 的先验数量，不实现预算/候补逻辑。

### 7. `candidate_logits` 默认 detach，detach 在 builder 内完成

03 输出的 `candidate_logits` 是诊断/后续默认输入，默认 detach。builder 接收的 `voxel_logits_ligand` 是带梯度的原始 head 输出，但 `SparseCandidateSetBuilder.forward()` 必须在 `torch.no_grad()` 下完成概率计算、topk、阈值比较和 gathered logits 输出。

05 若需要 residual/direct logits 对 voxel 分支反传，应在 sparse refine head 中显式重新 gather non-detached logits，并由 head 配置控制。

### 8. candidate warmup 不新增用户参数

不新增 `warmup_candidate_steps`。wrapper 在 `configure_optimizers()` 解析已有 scheduler 时缓存：

```text
candidate_warmup_steps = _resolve_warmup_steps(cfg.train.scheduler)
```

规则：

| scheduler 情况 | `candidate_warmup_steps` |
|---|---|
| `name: warmup_only` | 与 LR warmup step 数一致。 |
| `name: warmup_plateau` | 与 LR warmup step 数一致。 |
| 其它 scheduler 或 `scheduler is None` | 0。 |

无 scheduler 或 warmup step 为 0 时不进入 scheduler warmup fixed topk；warmup 外首次有效 threshold cache 产生前也不允许 bootstrap fixed topk。

### 9. 03 默认不新增通用 validation metric 打印 callback

当前训练入口已经使用 `RichProgressBar` 与 WandB logger。03 不在 [src/train.py](../../../../src/train.py) 增加遍历 `trainer.callback_metrics` 的通用打印 callback，避免和 progress bar/logger 重复刷屏。

wrapper 在每次 threshold cache 产生或更新时，用 rank0 打印：

1. `p_best_by_class`。
2. `p_sampling_by_class`。
3. `best_f1_by_class`。

普通 AP/loss metric 继续通过 Lightning logger / progress bar 观察。

## Proposed Changes

### 1. 新增 sparse refine candidate 模块

#### [NEW] [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py)

新增 `SparseCandidateSetBuilder`。

```python
class SparseCandidateSetBuilder(nn.Module):
    def __init__(
        self,
        candidate_class_ids: Sequence[int],
        warmup_topc_per_class: Sequence[int],
        adaptive_expand_factor: Sequence[float],
        max_candidate_voxels_per_class: Sequence[int],
        selection_mode: str,
    ) -> None: ...

    def forward(
        self,
        voxel_logits_ligand: torch.Tensor,
        voxel_valid_mask: torch.Tensor,
        p_best_by_class: torch.Tensor | None,
        p_sampling_by_class: torch.Tensor | None,
        use_fixed_warmup: bool,
    ) -> dict[str, torch.Tensor]: ...
```

| 参数 | 类型 | shape / 允许值 | 意义 |
|---|---|---|---|
| `candidate_class_ids` | `Sequence[int]` | 长度 `num_candidate_classes` | 显式候选类别 ID。单通道 sigmoid 路径只允许 `[1]`；多通道 softmax 路径要求每个 ID 在 `[1, C-1]`。 |
| `warmup_topc_per_class` | `Sequence[int]` | 同 `candidate_class_ids`；每项 `>= 0` | fixed topk 阶段每个 BOX、每类固定 topk 数。 |
| `adaptive_expand_factor` | `Sequence[float]` | 同 `candidate_class_ids`；每项 `> 0` | `p_best` 正例数量扩张倍数。 |
| `max_candidate_voxels_per_class` | `Sequence[int]` | 同 `candidate_class_ids`；每项 `>= 0` | 每个 BOX、每类候选行上限。 |
| `selection_mode` | `str` | `"adaptive_threshold"` / `"recorded_threshold"`，必传 | fixed topk 结束后的候选阈值来源。 |
| `voxel_logits_ligand` | `torch.Tensor` | `(B,1,D,H,W)` 或 `(B,C,D,H,W)` | voxel ligand head 原始输出，可带梯度；builder 内部 no-grad 使用。 |
| `voxel_valid_mask` | `torch.Tensor` | `(B,D,H,W)` 或 `(B,1,D,H,W)` bool | 有效体素掩码，只在 True 位置产生候选。 |
| `p_best_by_class` | `torch.Tensor | None` | `(num_candidate_classes,)` | wrapper 缓存的 best-F1 阈值；`adaptive_threshold` 非 fixed topk 时必需。 |
| `p_sampling_by_class` | `torch.Tensor | None` | `(num_candidate_classes,)` | wrapper 缓存的全局候选阈值；`recorded_threshold` 非 fixed topk 时必需。 |
| `use_fixed_warmup` | `bool` | 标量 | True 时忽略阈值，使用 `warmup_topc_per_class`。 |

输出 `C`：

| 字段 | 类型 | shape | 语义 |
|---|---|---|---|
| `candidate_voxel_zyx` | `torch.Tensor` | `(sumC, 3)` long | 离散 voxel index，轴顺序 `(z, y, x)`；可直接索引 `ligand_dist_map` 与 `voxel_valid_mask`。 |
| `candidate_batch_index` | `torch.Tensor` | `(sumC,)` long | 每个候选行所属 BOX。 |
| `candidate_class` | `torch.Tensor` | `(sumC,)` long | 每个候选行的前景类别 ID，来自 `candidate_class_ids`。 |
| `candidate_prob` | `torch.Tensor` | `(sumC,)` floating | 当前候选类概率；单通道为 sigmoid，多通道为 softmax 对应类。 |
| `candidate_logits` | `torch.Tensor` | `(sumC, 1)` 或 `(sumC, C)` floating | 对应 voxel 的原始 ligand logits，builder 内部 detach。 |
| `candidate_counts` | `torch.Tensor` | `(B,)` long | 每个 BOX 的候选行数。 |
| `candidate_counts_by_class` | `torch.Tensor` | `(B, num_candidate_classes)` long | 每个 BOX、每个候选类的候选行数。 |
| `candidate_p_sampling_by_class` | `torch.Tensor` | `(B, num_candidate_classes)` floating | 当前 batch 每个 BOX、每类实际使用的候选截断值。 |
| `candidate_target_counts_by_class` | `torch.Tensor` | `(B, num_candidate_classes)` long | 当前 batch 每个 BOX、每类阈值扩张后、cap 前的目标候选数；fixed topk 时等于 warmup topc。 |

实现要求：

1. `__init__()` 不接受 `mode`，不为 `selection_mode` 提供 Python 默认值。
2. `candidate_class_ids`、`warmup_topc_per_class`、`adaptive_expand_factor`、`max_candidate_voxels_per_class` 长度必须一致；不一致 fail-fast。
3. `selection_mode` 只允许 `"adaptive_threshold"` 或 `"recorded_threshold"`。
4. `voxel_valid_mask` 若为 `(B,1,D,H,W)` 先 squeeze 到 `(B,D,H,W)`；若为 `(B,D,H,W)` 直接用；其它 shape 直接 fail-fast。
5. `voxel_logits_ligand.shape[1] == 1` 时使用 `torch.sigmoid(logits[:, 0])`；只允许 `candidate_class_ids == (1,)`。
6. `voxel_logits_ligand.shape[1] > 1` 时使用 `torch.softmax(logits, dim=1)`；`candidate_class_ids` 必须全部为前景类且 `< logits.shape[1]`。
7. 每个 BOX、每类独立筛选；允许同一个 voxel 以不同 `candidate_class` 出现多行。
8. fixed topk：在 valid 体素内按该类概率取 `warmup_topc_per_class[class]`，再套 `max_candidate_voxels_per_class[class]`。
9. `adaptive_threshold`：要求 `p_best_by_class` 非 None 且长度匹配；在当前 BOX valid 体素内计算 `n_best_box`、`n_target_box`，再取 top `n_target_box` 得到局部候选。
10. `recorded_threshold`：要求 `p_sampling_by_class` 非 None 且长度匹配；在当前 BOX valid 体素内取 `prob > p_sampling_by_class[class]`，若超过 cap 则按概率 topk 裁剪。
11. 两种非 fixed topk 模式均套 `max_candidate_voxels_per_class[class]`。
12. 若某 BOX/类没有候选，输出字段保持空行兼容，counts 为 0。
13. `candidate_logits = gathered_logits.detach()`；概率、topk、阈值比较、gather 均在 `torch.no_grad()` 下执行。
14. 不输出 world/local 连续坐标；04 再从 `candidate_voxel_zyx` 转 P anchor 坐标。

#### [NEW] [src/model/sparse_refine/__init__.py](../../../../src/model/sparse_refine/__init__.py)

若包导入需要，导出：

```python
from src.model.sparse_refine.candidate_set import SparseCandidateSetBuilder

__all__ = ["SparseCandidateSetBuilder"]
```

### 2. 配置更新

#### [NEW] [configs/model/candidate_set/none.yaml](../../../../configs/model/candidate_set/none.yaml)

```yaml
# @package _global_
model:
  backbone:
    candidate_set_cfg: null
```

#### [NEW] [configs/model/candidate_set/binary.yaml](../../../../configs/model/candidate_set/binary.yaml)

```yaml
# @package _global_
model:
  backbone:
    candidate_set_cfg:
      _target_: src.model.sparse_refine.candidate_set.SparseCandidateSetBuilder
      selection_mode: adaptive_threshold
      candidate_class_ids: [1]
      warmup_topc_per_class: [2048]
      adaptive_expand_factor: [10.0]
      max_candidate_voxels_per_class: [30000]
```

二分类语义：`candidate_class_ids=[1]` 表示单前景 ligand foreground，不写死 small molecule。

#### [NEW] [configs/model/candidate_set/tri.yaml](../../../../configs/model/candidate_set/tri.yaml)

```yaml
# @package _global_
model:
  backbone:
    candidate_set_cfg:
      _target_: src.model.sparse_refine.candidate_set.SparseCandidateSetBuilder
      selection_mode: adaptive_threshold
      candidate_class_ids: [1, 2]
      warmup_topc_per_class: [256, 2048]
      adaptive_expand_factor: [10.0, 10.0]
      max_candidate_voxels_per_class: [3000, 30000]
```

> [!IMPORTANT]
> 不配置 `mode: threshold_expand`。Hydra 会把该死字段传给 constructor 并导致实例化失败。

#### [MODIFY] [configs/base.yaml](../../../../configs/base.yaml)

在 model 子模块 defaults 中加入默认关闭项，位置放在 `model/pseudo_atom: none` 后或其它 model group 附近：

```yaml
- model/candidate_set: none
```

#### [MODIFY] 相关实验配置

后续 tri sparse refine 实验应显式引入：

```yaml
- override /model/candidate_set: tri
```

二分类 sparse refine 实验显式引入：

```yaml
- override /model/candidate_set: binary
```

不再配置 `max_candidate_voxels_per_box` 或 `p_final`。

### 3. 改造 Stage1 主模型接入 C 生成

#### [MODIFY] [src/model/stage1_model.py](../../../../src/model/stage1_model.py)

##### 3.1 `VolumePointStage1Model.__init__()` 新增 candidate builder 配置

新增参数：

```python
candidate_set_cfg: dict[str, Any] | nn.Module | None = None
```

行为：

| 条件 | 行为 |
|---|---|
| `candidate_set_cfg is None` | `self.candidate_set_builder = None`，保持当前 01/02 行为。 |
| `candidate_set_cfg` 是 `nn.Module` | 直接赋值为 `self.candidate_set_builder`。 |
| 其它 Hydra config | `instantiate(candidate_set_cfg)` 得到 builder。 |

新增 runtime 字段：

```python
self._candidate_p_best_by_class: torch.Tensor | None = None
self._candidate_p_sampling_by_class: torch.Tensor | None = None
self._candidate_warmup_steps: int = 0
self._candidate_global_step: int = 0
self._candidate_allow_warmup_fixed_topk: bool = False
```

##### 3.2 新增 builder 与阈值查询接口

新增方法：

```python
def get_sparse_candidate_class_ids(self) -> tuple[int, ...] | None: ...
```

行为：

1. `self.candidate_set_builder is None` 时返回 None。
2. 否则返回 `tuple(int(x) for x in self.candidate_set_builder.candidate_class_ids)`。
3. 若 builder 没有该属性，抛 `AttributeError`，不静默推断。

新增方法：

```python
def set_sparse_candidate_thresholds(
    self,
    p_best_by_class: torch.Tensor | None,
    p_sampling_by_class: torch.Tensor | None,
) -> None: ...
```

行为：

1. 保存 detached CPU 或 module-local tensor 状态到 `_candidate_p_best_by_class` 与 `_candidate_p_sampling_by_class`。
2. 接收 None 表示清空或尚未可用。
3. `_prepare_pseudo_batch()` 调用 builder 前按 selection mode 检查阈值可用性。
4. tensor 在 forward 前移动到 `voxel_logits_ligand.device`。

新增方法：

```python
def set_sparse_candidate_runtime(
    self,
    global_step: int,
    candidate_warmup_steps: int,
    allow_warmup_fixed_topk: bool,
) -> None: ...
```

| 参数 | 类型 | 意义 |
|---|---|---|
| `global_step` | `int` | wrapper 当前 `global_step`，用于判断 scheduler warmup fixed topk。 |
| `candidate_warmup_steps` | `int` | 从已有 scheduler warmup 解析出的 step 数；无 warmup 时为 0。 |
| `allow_warmup_fixed_topk` | `bool` | 当前 lifecycle 是否允许 scheduler warmup fixed topk；fit/sanity/tuning 可为 True，standalone validate/test/predict 为 False。 |

##### 3.3 `_prepare_pseudo_batch()` 生成 C 但不注入 P

当前 `_prepare_pseudo_batch()` 的 01 行为是返回 `(batch, None, {})`。03 修改为：

```python
if self.candidate_set_builder is None:
    return batch, None, {}
voxel_logits_ligand = voxel_output_dict.get("voxel_logits_ligand")
if voxel_logits_ligand is None:
    raise RuntimeError("candidate_set_builder 已启用，但 voxel_logits_ligand 为空。")
use_fixed_warmup = self._should_use_candidate_fixed_topk()
candidate_outputs = self.candidate_set_builder(
    voxel_logits_ligand=voxel_logits_ligand,
    voxel_valid_mask=batch["voxel_valid_mask"],
    p_best_by_class=self._candidate_p_best_by_class,
    p_sampling_by_class=self._candidate_p_sampling_by_class,
    use_fixed_warmup=use_fixed_warmup,
)
return batch, None, candidate_outputs
```

新增私有方法：

```python
def _should_use_candidate_fixed_topk(self) -> bool: ...
```

行为：

1. 若 `_candidate_allow_warmup_fixed_topk` 且 `_candidate_warmup_steps > 0` 且 `_candidate_global_step < _candidate_warmup_steps`，返回 True。
2. 其它情况返回 False；builder 会在 threshold 缺失时 fail-fast。

插入位置：仍只在 final recycle 调用 `_prepare_pseudo_batch()`，不改变非 final recycle 逻辑。

输出影响：`forward()` 最终 `outputs` 因 `**pseudo_outputs` 包含 C 字段，但 `point_batch` 仍是 real-only，`last_pseudo_layout=None`，atom head 输出仍 real-only。

### 4. wrapper 增加 voxel ligand best-F1 / threshold 统计

#### [MODIFY] [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

##### 4.1 新增 wrapper 参数

在 `VoxelPointStage1Wrapper.__init__()` 新增可选参数：

```python
initial_p_best_by_class: list[float] | tuple[float, ...] | None = None
initial_p_sampling_by_class: list[float] | tuple[float, ...] | None = None
```

| 参数 | shape | 默认 | 消费位置 |
|---|---|---|---|
| `initial_p_best_by_class` | `(num_candidate_classes,)` | `None` | checkpoint 无缓存时初始化 `_cached_voxel_ligand_p_best_by_class`。 |
| `initial_p_sampling_by_class` | `(num_candidate_classes,)` | `None` | checkpoint 无缓存时初始化 `_cached_voxel_ligand_p_sampling_by_class`。 |

若 builder 启用且任一 initial 非 None，长度必须等于 builder `candidate_class_ids` 长度。

##### 4.2 新增 runtime 缓存字段

在 `__init__()` 初始化：

```python
self._sparse_candidate_class_ids: tuple[int, ...] | None = self._resolve_sparse_candidate_class_ids()
self._cached_voxel_ligand_p_best_by_class: torch.Tensor | None = None
self._cached_voxel_ligand_p_sampling_by_class: torch.Tensor | None = None
self._cached_voxel_ligand_best_f1_by_class: torch.Tensor | None = None
self._candidate_warmup_steps: int = 0
```

新增方法：

```python
def _resolve_sparse_candidate_class_ids(self) -> tuple[int, ...] | None: ...
```

行为：

1. 从 eager backbone 读取 `get_sparse_candidate_class_ids()`；若 backbone 被 `torch.compile` 包装，使用 `_orig_mod` 解包。
2. builder 未启用时返回 None，不启用 threshold histogram。
3. 若 `voxel_ligand_loss is None` 且 builder 启用，抛 `ValueError`，因为无法生成 threshold stats。
4. 若 `voxel_ligand_loss.num_classes <= 2`，只允许 `(1,)`。
5. 若 `voxel_ligand_loss.num_classes > 2`，每个 class id 必须满足 `1 <= class_id < num_classes`。
6. 不从 loss 自动补全 class ids；缺失 builder 配置直接按 builder disabled 处理。

##### 4.3 新增 histogram 统计状态

复用 `voxel_ligand_pr_auc_thresholds` 作为 histogram bin 数。builder 启用时要求该值为正整数；若为 `None`，candidate threshold stats fail-fast，用户需显式设置整数。

新增字段：

```python
self._voxel_ligand_threshold_bin_count: int | None
self._voxel_ligand_threshold_grid: torch.Tensor | None
self._voxel_ligand_pos_hist_by_class: torch.Tensor | None
self._voxel_ligand_neg_hist_by_class: torch.Tensor | None
```

| 字段 | shape | 初始化 |
|---|---|---|
| `_voxel_ligand_threshold_bin_count` | 标量 int | builder 启用时等于 `voxel_ligand_pr_auc_thresholds`。 |
| `_voxel_ligand_threshold_grid` | `(num_bins,)` | builder 启用时在 `__init__()` 计算 `torch.arange(num_bins) / num_bins`。 |
| `_voxel_ligand_pos_hist_by_class` | `(num_candidate_classes, num_bins)` | builder 启用时注册/保存为全零 long tensor。 |
| `_voxel_ligand_neg_hist_by_class` | `(num_candidate_classes, num_bins)` | builder 启用时注册/保存为全零 long tensor。 |

新增方法：

```python
def _reset_voxel_ligand_threshold_histograms(self) -> None: ...
```

行为：builder 未启用时 no-op；启用时把 pos/neg histogram 原地清零。

在 `on_validation_epoch_start()` 调用 `_reset_voxel_ligand_threshold_histograms()`，并在 `_compute_log_update_voxel_ligand_best_f1_thresholds()` 结束后再次清零。

bin 规则：

```python
bin_idx = torch.floor(prob.clamp(0.0, 1.0) * num_bins).long().clamp(max=num_bins - 1)
threshold = bin_idx / num_bins
```

> [!IMPORTANT]
> 默认不保存全验证集概率 buffer。`p_best_by_class` 与 `p_sampling_by_class` 都从 pos/neg histogram 近似计算；1024 bins 时阈值量化误差约为 `1 / 1024`。如未来需要 exact threshold，应另做离线评估，不作为训练默认路径。

##### 4.4 validation step 更新 binned F1 状态

在 `_update_val_voxel_ligand_metric()` 中，现有 AP 更新后新增调用：

```python
self._update_voxel_ligand_best_f1_stats(
    logits=voxel_logits_ligand,
    ligand_dist_map=ligand_dist_map,
    valid_mask=batch["voxel_valid_mask"],
)
```

新增方法签名：

```python
def _update_voxel_ligand_best_f1_stats(
    self,
    logits: torch.Tensor,
    ligand_dist_map: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None: ...
```

行为：

1. `_sparse_candidate_class_ids is None` 时直接返回。
2. `voxel_ligand_loss.hard_label_threshold is None` 时抛 `ValueError`；candidate threshold stats 不使用 `batch["voxel_label"]` 回退。
3. 单通道 logits：`prob_by_class = sigmoid(logits[:, 0])`，候选类只能是 class id 1。
4. 多通道 logits：`prob = softmax(logits, dim=1)`，只统计 `_sparse_candidate_class_ids` 中列出的 class id。
5. hard label 语义复用 `_ligand_target_from_dist(ligand_dist_map, hard_label_threshold)`。
6. 只统计 `voxel_valid_mask=True` 的位置；`valid_mask` 支持 `(B,D,H,W)` 与 `(B,1,D,H,W)`。
7. 对每个候选类用 `torch.bincount()` 累加 `pos_hist_by_class` 与 `neg_hist_by_class`。
8. DDP 下不要同步或收集逐体素概率数组；只在 validation end 对 histogram 做 sum 聚合。

##### 4.5 validation end 计算并缓存阈值

在 `on_validation_epoch_end()` 中，现有 AP compute/reset 后、`_step_warmup_plateau_scheduler(computed_metrics)` 前新增：

```python
computed_metrics.update(self._compute_log_update_voxel_ligand_best_f1_thresholds())
```

新增方法签名：

```python
def _compute_log_update_voxel_ligand_best_f1_thresholds(self) -> dict[str, torch.Tensor]: ...
```

行为：

1. `_sparse_candidate_class_ids is None` 时返回 `{}`。
2. 复制本地 `pos_hist_by_class` / `neg_hist_by_class` 到 device float/long tensor。
3. DDP 下用 `self.all_gather()` 后按 rank 维求和，得到全局 histogram；单卡直接用本地 histogram。
4. 从高概率 bin 向低概率 bin 做 cumulative sum：

   ```python
   tp_at_threshold = torch.cumsum(pos_hist.flip(-1), dim=-1).flip(-1)
   fp_at_threshold = torch.cumsum(neg_hist.flip(-1), dim=-1).flip(-1)
   fn_at_threshold = pos_hist.sum(dim=-1, keepdim=True) - tp_at_threshold
   ```

5. 对每个候选类计算：

   ```python
   denominator = 2.0 * tp_at_threshold + fp_at_threshold + fn_at_threshold
   f1_by_threshold = torch.where(denominator > 0, 2.0 * tp_at_threshold / denominator, 0.0)
   ```

6. 取最大 F1 的 bin lower-edge threshold 作为 `p_best_by_class[class]`。
7. 基于同一 histogram 计算 `p_sampling_by_class[class]`：
   * `best_bin = argmax(f1_by_threshold[class_index])`。
   * `n_best_total = tp_at_threshold[class_index, best_bin] + fp_at_threshold[class_index, best_bin]`。
   * `n_sampling_total = ceil(n_best_total * adaptive_expand_factor[class_index])`。
   * `all_hist = pos_hist + neg_hist`。
   * 从高概率向低概率累计 `all_hist[class_index]`，找到第一个累计数 `>= n_sampling_total` 的 bin。
   * 取该 bin lower edge 作为 `p_sampling_by_class[class_index]`。
8. 对无有效统计类别：保持旧缓存；无旧缓存时使用显式 initial；仍不可用则该类保持 unavailable。
9. 更新 `_cached_voxel_ligand_p_best_by_class`、`_cached_voxel_ligand_p_sampling_by_class`、`_cached_voxel_ligand_best_f1_by_class`。
10. 调用 backbone 的 `set_sparse_candidate_thresholds(...)` 同步缓存。
11. 用 `self.log()` 记录每类：
    * `val/voxel_ligand_p_best_by_class_{class_name}`
    * `val/voxel_ligand_p_sampling_by_class_{class_name}`
    * `val/voxel_ligand_best_f1_by_class_{class_name}`
12. 记录 macro：`val/voxel_ligand_macro_best_f1_by_class`。
13. 每次产生或更新缓存时 rank0 打印 `p_best_by_class`、`p_sampling_by_class`、`best_f1_by_class`。
14. reset 本轮 histogram 状态。
15. 返回本方法现场计算出的 metric dict，供 scheduler monitor 若配置到这些指标时使用。

##### 4.6 checkpoint 保存/恢复

修改 `on_save_checkpoint()`：

```python
checkpoint["voxel_ligand_candidate_class_ids"] = self._sparse_candidate_class_ids
if is_finite_vector(p_best_by_class) and is_finite_vector(p_sampling_by_class):
    checkpoint["voxel_ligand_p_best_by_class"] = p_best_by_class
    checkpoint["voxel_ligand_p_sampling_by_class"] = p_sampling_by_class
if is_finite_vector(best_f1_by_class):
    checkpoint["voxel_ligand_best_f1_by_class"] = best_f1_by_class
```

修改 `on_load_checkpoint()`：

1. 若 checkpoint 中存在上述 tensor 字段，恢复到 wrapper runtime cache；`p_best_by_class` 与 `p_sampling_by_class` 必须成对出现，长度不匹配或 threshold cache 含 NaN/Inf 时 fail-fast。
2. 若 checkpoint 中存在 `voxel_ligand_candidate_class_ids`，必须与当前 `_sparse_candidate_class_ids` 一致；不一致 fail-fast。
3. 恢复后同步给 backbone。
4. 若 threshold 字段不存在，保持当前显式 initial 或 None；warmup 外仍不可用时由 builder fail-fast。

##### 4.7 同步 candidate runtime

新增方法：

```python
def _sync_sparse_candidate_runtime_to_backbone(self) -> None: ...
```

行为：

1. builder 未启用时 no-op。
2. 调用 backbone `set_sparse_candidate_runtime(global_step=..., candidate_warmup_steps=self._candidate_warmup_steps, allow_warmup_fixed_topk=...)`；fit/sanity/tuning lifecycle 可允许 warmup fixed topk，standalone validate/test/predict 不允许。
3. 调用 backbone `set_sparse_candidate_thresholds(...)` 同步当前 cache。

调用位置：

1. `training_step()` 的 forward 前。
2. `validation_step()` 的 forward 前。
3. `on_load_checkpoint()` 恢复 cache 后。
4. `_compute_log_update_voxel_ligand_best_f1_thresholds()` 更新 cache 后。

##### 4.8 解析 candidate warmup steps

在 `configure_optimizers()` 中：

1. 当 `sched_cfg is None` 时，`self._candidate_warmup_steps = 0`。
2. 当 `sched_cfg.get("name") in {"warmup_only", "warmup_plateau"}` 时，复用 `_resolve_warmup_steps(sched_cfg)` 的结果赋给 `self._candidate_warmup_steps`。
3. 其它 scheduler 设为 0。
4. 避免同一分支重复调用 `_resolve_warmup_steps()` 造成分歧；可先保存局部 `warmup_steps` 再传给 scheduler 构造函数。

### 5. 同步总控契约

#### [MODIFY] [CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md](00-master.md)

同步新增/更新：

1. C 输出字段契约。
2. `candidate_class_ids` 显式配置与校验，并说明 wrapper threshold 类集合来自 builder。
3. `p_best_by_class` / `p_sampling_by_class` 缓存语义。
4. `selection_mode: adaptive_threshold | recorded_threshold`。
5. 单通道 sigmoid 长期支持，多通道 softmax 支持。
6. C 主键为 `(batch, voxel_zyx, candidate_class)`，允许同 voxel 多类重复。
7. `candidate_logits` 03 默认 detach，detach 在 builder 内完成。
8. 03 不输出 world/local 坐标，04 才生成 P anchor 坐标。
9. `max_candidate_voxels_per_box` 删除，只保留 per-class cap。
10. 03 不新增通用 metric print callback，只打印 threshold cache 更新。
11. 配置组增加 `model/candidate_set: none | binary | tri`。

## 不修改的部分

1. 不实现 P anchor 采样；[src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 不在 03 中新增生成逻辑。
2. 不注入 P，不让 C 进入 point backbone。
3. 不实现 density cube。
4. 不实现 P→C three-nn 插值。
5. 不实现 sparse C logit head。
6. 不实现 refined C loss、refined best-F1、refined PR-AUC 或漏检惩罚指标；这些留到 06。
7. 不输出 C 的 world/local 连续坐标。
8. 不新增 `max_candidate_voxels_per_box` 或类别预算/候补裁剪逻辑。
9. 不新增 candidate 专用 warmup 参数。
10. 不为二分类实现标量阈值特例。
11. 不在 [src/train.py](../../../../src/train.py) 增加通用 `ValidationMetricPrintCallback`；除非后续用户明确要求终端打印全部 validation metrics。
12. 不修改现有 voxel ligand PR-AUC 的 `hard_label_threshold is None -> batch["voxel_label"]` fallback；该 fallback 只服务旧 AP 路径，不服务 candidate threshold stats。

## 改动文件汇总

| 文件 | 改动内容 |
|---|---|
| [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py) | 新增 `SparseCandidateSetBuilder`。 |
| [src/model/sparse_refine/__init__.py](../../../../src/model/sparse_refine/__init__.py) | 如需要，导出 candidate builder。 |
| [configs/model/candidate_set/none.yaml](../../../../configs/model/candidate_set/none.yaml) | 新增默认关闭配置。 |
| [configs/model/candidate_set/binary.yaml](../../../../configs/model/candidate_set/binary.yaml) | 新增二分类 foreground candidate 配置；不含 `mode`。 |
| [configs/model/candidate_set/tri.yaml](../../../../configs/model/candidate_set/tri.yaml) | 新增三分类 candidate 配置；不含 `mode`。 |
| [configs/base.yaml](../../../../configs/base.yaml) | defaults 增加 `model/candidate_set: none`。 |
| [src/model/stage1_model.py](../../../../src/model/stage1_model.py) | 实例化 candidate builder，接入 `_prepare_pseudo_batch()` 输出 C，新增 threshold/runtime 同步接口。 |
| [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) | 新增 best-F1 / p_best / p_sampling 统计、日志、缓存、checkpoint 保存恢复与 backbone 同步。 |
| [tests/model/test_sparse_candidate_set.py](../../../../tests/model/test_sparse_candidate_set.py) | 新增 candidate builder 单测。 |
| [tests/model/test_stage1_model.py](../../../../tests/model/test_stage1_model.py) | 增加 `_prepare_pseudo_batch()` C 输出接入测试。 |
| [tests/test_voxel_ligand_thresholds.py](../../../../tests/test_voxel_ligand_thresholds.py) | 新增 wrapper threshold/F1/cache/checkpoint 测试。 |
| [tests/test_multiclass_ligand_wrapper.py](../../../../tests/test_multiclass_ligand_wrapper.py) | 可补充 hard-label / class-id 小测试，或保留原有测试不动。 |
| [CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md](00-master.md) | 同步 C 与 threshold 契约。 |

## Verification Plan

### Automated Tests

```bash
pytest tests/model/test_sparse_candidate_set.py -q
```

覆盖点：

| 测试函数 | 构造 | 关键断言 |
|---|---|---|
| `test_binary_sigmoid_candidate_generation` | `(B,1,D,H,W)` logits，`candidate_class_ids=[1]` | 使用 sigmoid；`candidate_logits.shape[-1] == 1`；只在 valid mask 内取候选。 |
| `test_binary_rejects_non_one_class_id` | 单通道 logits + `candidate_class_ids=[2]` | fail-fast。 |
| `test_multiclass_softmax_candidate_generation` | `(B,3,D,H,W)` logits，`candidate_class_ids=[1,2]` | 使用 softmax；输出类别 ID 正确。 |
| `test_candidate_key_allows_same_voxel_multiple_classes` | 同一 voxel 两类概率都高 | 输出中同一 voxel 可出现两行不同 class。 |
| `test_warmup_uses_per_class_topc` | `use_fixed_warmup=True` | 每 BOX/类候选数不超过 `warmup_topc_per_class` 与 per-class cap。 |
| `test_adaptive_threshold_requires_p_best` | `selection_mode="adaptive_threshold"` 且 `p_best_by_class=None` | 非 fixed topk 下 fail-fast。 |
| `test_adaptive_threshold_uses_box_local_p_sampling` | `selection_mode="adaptive_threshold"` | `candidate_p_sampling_by_class` 随 BOX 分布变化。 |
| `test_recorded_threshold_requires_p_sampling` | `selection_mode="recorded_threshold"` 且 `p_sampling_by_class=None` | 非 fixed topk 下 fail-fast。 |
| `test_recorded_threshold_uses_cached_p_sampling` | `selection_mode="recorded_threshold"` | 每 BOX 使用同一全局 `p_sampling_by_class`。 |
| `test_constructor_rejects_unknown_mode_key` | 直接向 constructor 传 `mode="threshold_expand"` | Python 报 unexpected keyword 或配置测试确认 YAML 不含该字段。 |
| `test_candidate_logits_are_detached` | requires-grad logits | `candidate_logits.requires_grad is False`。 |
| `test_no_world_coordinates_in_candidate_output` | 正常 C 输出 | 不存在 `candidate_coord_world` / `candidate_coord_local_voxel` 字段。 |

```bash
pytest tests/model/test_stage1_model.py -q
```

新增或更新测试：

| 测试函数 | 覆盖点 |
|---|---|
| `test_prepare_pseudo_batch_outputs_candidates_without_pseudo_layout` | 配置 candidate builder 后，final recycle 输出 C 字段，但 `pseudo_layout is None`，point batch 仍 real-only。 |
| `test_stage1_model_candidate_builder_disabled_keeps_empty_pseudo_outputs` | `candidate_set_cfg=None` 时保持当前行为。 |
| `test_stage1_model_syncs_candidate_thresholds_to_builder_device` | wrapper/backbone 同步阈值后，forward 时移动到 logits device。 |
| `test_stage1_model_candidate_runtime_uses_training_warmup` | `global_step < candidate_warmup_steps` 时 builder 收到 `use_fixed_warmup=True`。 |
| `test_stage1_model_candidate_runtime_fails_without_thresholds_in_eval` | eval/推理且 threshold 缺失时非 fixed topk fail-fast。 |

```bash
pytest tests/test_voxel_ligand_thresholds.py -q
```

覆盖点：

| 测试函数 | 覆盖点 |
|---|---|
| `test_voxel_ligand_best_f1_updates_p_best_by_class` | validation stats 计算 best-F1 threshold。 |
| `test_voxel_ligand_sampling_threshold_uses_histogram_topk_definition` | `p_sampling_by_class` 由 pos/neg histogram 的全验证集近似 topk 定义得到，误差不超过一个 bin。 |
| `test_threshold_stats_reject_hard_label_threshold_none` | builder 启用且 loss `hard_label_threshold=None` 时 fail-fast。 |
| `test_threshold_class_ids_read_from_candidate_builder` | wrapper 只为 builder 的 `candidate_class_ids` 建 histogram。 |
| `test_threshold_class_ids_reject_loss_mismatch` | `candidate_class_ids=[1,2]` 但 loss `num_classes=2` 时 fail-fast。 |
| `test_threshold_histograms_reset_on_validation_epoch_start` | validation start 清零 histogram。 |
| `test_threshold_cache_keeps_old_value_when_class_has_no_positive_stats` | 无有效统计时保持旧缓存。 |
| `test_threshold_cache_uses_initial_when_no_old_value` | 无旧缓存时使用显式 initial。 |
| `test_threshold_cache_saved_and_loaded_from_checkpoint` | checkpoint metadata 保存/恢复 p_best/p_sampling/best_f1/class_ids。 |
| `test_threshold_cache_load_rejects_candidate_class_id_mismatch` | checkpoint class ids 与当前配置不一致时 fail-fast。 |
| `test_threshold_cache_syncs_to_backbone_after_update` | validation end 更新后调用 backbone setter。 |
| `test_candidate_warmup_reuses_scheduler_warmup_steps` | 不新增 candidate 参数，复用 scheduler warmup step 判断 fixed topk。 |
| `test_voxel_ligand_threshold_stats_do_not_store_probability_buffers` | wrapper 只维护 histogram 状态，不保存逐体素概率列表。 |
| `test_voxel_ligand_threshold_histograms_all_reduce_in_ddp` | 多 rank 场景下 histogram 用 sum 聚合，而不是收集概率数组。 |

```bash
pytest tests/test_multiclass_ligand_wrapper.py tests/test_multiclass_voxel_backbone.py -q
```

验证点：现有 voxel ligand 多分类 loss/metrics 与 voxel backbone 输出不回归。

### Config / Hydra Smoke Tests

```bash
python -m src.train +experiment=unet000 model/candidate_set=binary trainer.fast_dev_run=true
```

验证点：二分类 `voxel_ligand_logit_dim: 1 + sigmoid()` 路径可实例化，candidate builder 不要求三通道 logits。

```bash
python -m src.train +experiment=tri001_tunedloss model/candidate_set=tri trainer.fast_dev_run=true
```

验证点：三分类 `candidate_class_ids=[1,2]`、softmax、多类 threshold 统计可进入 fast-dev-run。

如果当前训练入口不支持 `trainer.fast_dev_run=true`，使用项目现有 fast-dev-run 参数；验证目标不变。

### Manual Code Checks

```bash
rg "max_candidate_voxels_per_box|p_final" CLAUDE/plans/implement/tri_ligand_sparse_refine src configs tests
```

预期：03 新实现和配置不再使用 `max_candidate_voxels_per_box` 或 `p_final`；如旧历史文档命中需确认不是执行入口。

```bash
rg "mode: threshold_expand" configs src tests
```

预期：源码、配置和测试中没有命中；计划文件中的 review 说明不作为执行入口。

```bash
rg "candidate_coord_world|candidate_coord_local_voxel" src tests CLAUDE/plans/implement/tri_ligand_sparse_refine/03-sparse-candidate-C.md
```

预期：03 不输出 C 的连续坐标；04 之前不应出现 candidate 坐标字段。

```bash
rg "p_best_by_class|p_sampling_by_class|candidate_p_sampling_by_class|selection_mode" src configs tests CLAUDE/plans/implement/tri_ligand_sparse_refine
```

预期：命名统一使用 `_by_class`，`selection_mode` 只允许 `adaptive_threshold` / `recorded_threshold`。

### Acceptance Criteria

1. `SparseCandidateSetBuilder` 可在二分类 sigmoid 与多分类 softmax 两条路径生成 C。
2. `candidate_class_ids` 显式配置并 fail-fast 校验，不写死三分类；wrapper threshold 类集合来自 builder。
3. C 输出字段完整，主键为 `(batch, voxel_zyx, candidate_class)`，允许同 voxel 多类重复。
4. C 只包含 voxel index 与行级元数据，不包含 world/local 连续坐标。
5. `candidate_logits` 默认 detach，detach 在 builder 内完成。
6. wrapper 每次 validation loop 结束后基于 pos/neg histogram 更新 best-F1、`p_best_by_class`、`p_sampling_by_class`，并记录/打印。
7. `hard_label_threshold=None` 时 candidate threshold stats fail-fast，不使用旧 AP fallback。
8. 无有效统计时保持旧缓存；无旧缓存时使用显式 initial；warmup 外无完整 finite cache 时 fail-fast，不允许 bootstrap fixed topk。
9. threshold 缓存仅在 `p_best_by_class` 与 `p_sampling_by_class` 成对 finite 时进入 checkpoint metadata，恢复后校验 candidate class ids 并同步到 backbone。
10. `_prepare_pseudo_batch()` 生成 C 但仍不注入 P，不改变 point/atom real-only 路径。
11. 删除 `max_candidate_voxels_per_box`，只使用 per-class cap。
12. 不新增 candidate 专用 warmup 参数，复用 scheduler warmup step。
13. `configs/base.yaml` 增加 `model/candidate_set: none`，binary/tri 配置不含 `mode` 死字段。
14. 00-master 与 03 子计划契约同步。
