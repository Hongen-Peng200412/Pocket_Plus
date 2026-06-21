# 06：sparse refine loss、best-F1、candidate recall 与配置

## 背景 / 目标

第 05 阶段之后，sparse refine 前向链路已经在唯一 C 上输出 refined logits：

```text
voxel logits -> 唯一候选体素 C -> P anchor -> P -> C KNN message -> ligand_refine_logits_C
```

本阶段在此基础上补齐训练与验证闭环：

1. 在所有有效 C 上计算 sparse refine loss。
2. 输出 refine 前 dense voxel best-F1。
3. 输出 refine 后 sparse C best-F1，且 FN 必须包含未进入 C 的 GT 正体素。
4. 输出 candidate recall，定义为 C 覆盖 GT 正体素的比例。
5. 输出 C/P 数量日志。
6. 清理 wrapper 指标记录逻辑，避免无有效样本时在 W&B 上产生无意义的 0 曲线。
7. 新增二分类与三分类 sparse refine 正式实验配置。

当前关键运行契约：

| 字段 | shape | 来源 | 本阶段用途 |
|---|---:|---|---|
| `candidate_voxel_zyx` | `(sumC, 3)` | [src/model/stage1_model.py](../../../../src/model/stage1_model.py) | C 的离散 z/y/x 坐标。 |
| `candidate_batch_index` | `(sumC,)` | [src/model/stage1_model.py](../../../../src/model/stage1_model.py) | C 所属 BOX。 |
| `candidate_class` | `(sumC,)` | [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py) | C 的随机路由类别；不作为最终监督标签。 |
| `candidate_counts` | `(B,)` | [src/model/sparse_refine/candidate_set.py](../../../../src/model/sparse_refine/candidate_set.py) | 记录每 BOX 候选 C 数量。 |
| `anchor_counts` | `(B,)` | [src/model/sparse_refine/anchor_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py) | 记录每 BOX P anchor 数量。 |
| `candidate_message_valid_mask` | `(sumC,)` | [src/model/sparse_refine/sparse_refine_head.py](../../../../src/model/sparse_refine/sparse_refine_head.py) | 保留诊断含义，不进入 loss mask。 |
| `ligand_refine_logits_C` | `(sumC, 1)` 或 `(sumC, C_cls)` | [src/model/sparse_refine/sparse_refine_head.py](../../../../src/model/sparse_refine/sparse_refine_head.py) | C 上 sparse refine loss 与 metric 的预测 logits。 |

## 已有可复用代码

| 已有代码 | 可复用能力 | 本阶段改造 |
|---|---|---|
| [src/modules/losses.py](../../../../src/modules/losses.py) `AdaptiveClassificationCompositeLoss` | 已支持 `(N,1)` 二分类 logits 与 `(N,C)` 多分类 logits，只要传入 `target=(N,)` 和 `hardmask=(N,)`。 | 增加 `target_from_ligand_dist_map()`，统一 dense voxel 与 C 级 target 语义。 |
| [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) `_ligand_target_from_dist()` | 当前 wrapper 内已有 ligand hard-label 构造逻辑。 | 改为调用 loss 模块 helper，避免 target 逻辑分叉。 |
| [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) `_update_voxel_ligand_best_f1_stats()` | 已有 dense voxel logits 的 per-class histogram。 | 用于输出 refine 前 best-F1，并停止输出旧 best-F1 名称。 |
| [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) `_compute_log_reset_metric_safe()` / `_compute_log_reset_multiclass_metrics()` | 当前统一计算与记录 PR-AUC。 | 改成无有效 update 时不 log、不返回 metric。 |
| [configs/loss/tri_7focal_3dice_hard_tuned.yaml](../../../../configs/loss/tri_7focal_3dice_hard_tuned.yaml) | 三分类 `AdaptiveClassificationCompositeLoss` 配置模板。 | 仿照新增二分类/三分类 sparse refine 四路 loss preset。 |

## 设计决策

### 1. C 上 loss mask

C 级 loss 只使用 C 位置采样到的 `voxel_valid_mask`：

```python
ligand_refine_valid_mask_C = dense_valid_mask[
    candidate_batch_index,
    candidate_voxel_zyx[:, 0],
    candidate_voxel_zyx[:, 1],
    candidate_voxel_zyx[:, 2],
]
```

`candidate_message_valid_mask` 不参与 loss mask。

### 2. candidate recall

每个候选类 `class_id` 的 candidate recall 定义为：

```text
count(C_valid & target_C == class_id)
/
count(dense_valid & dense_target == class_id)
```

C 已经按 `(batch, voxel_zyx)` 唯一化，因此不需要额外去重。

### 3. refine 后 best-F1

每个候选类 `class_id` 在全体唯一 C 上 one-vs-rest 统计：

```text
TP(th) = count(C_valid & score_C_class > th & target_C == class_id)
FP(th) = count(C_valid & score_C_class > th & target_C != class_id)
FN(th) = total_GT_positive_class_in_dense_valid_voxels - TP(th)
F1(th) = 2TP / (2TP + FP + FN)
```

未进入 C 的 GT 正体素进入 FN，因此 sparse refine best-F1 会惩罚 C 漏检。

### 4. 无有效样本日志规则

1. 单个类别在一个 validation loop 内没有 GT 正例：跳过该类别，不记录 0，不纳入 macro。
2. 所有 candidate 类别在整个 validation loop 内都没有 GT 正例：fail-fast，视为数据或配置 bug。
3. 某类有 GT 正例但 C 覆盖为 0：记录 `candidate_recall_{class_name}=0`，该类 sparse refine best-F1 也应为 0。
4. PR-AUC 等已有指标无有效 update 时不调用 `self.log()`，不向 `computed_metrics` 写入 0。

### 5. loss 组成与权重

二分类主实验与三分类正式配置都启用四路 loss：

| 分支 | 配置键 | target 来源 | 权重 |
|---|---|---|---:|
| 原子级结合原子辅助损失 | `model.atom_loss` | `outputs["atom_target"]` / `batch["atom_label"]` | `0.1` |
| 体素级结合体素辅助损失 | `model.voxel_aux_loss` | `batch["voxel_label"]`，mask 为 `hardmask & voxel_valid_mask` | `0.1` |
| coarse voxel ligand loss | `model.voxel_ligand_loss` | `ligand_dist_map` + `hard_label_threshold` | `1.0` |
| sparse refine loss | `model.ligand_sparse_refine_loss` | dense target 采样到 C | final weight `1.0` |

四路 loss 均使用 `AdaptiveClassificationCompositeLoss`。

主 loss 参数：

```yaml
w_focal: 0.7
w_tversky: 0.3
w_mse: 0.0
```

二分类 focal alpha：

```yaml
focal_alpha: [0.5, 0.5]
focal_alpha_neg: 0.5
focal_alpha_pos: 0.5
```

三分类 focal alpha 继承三分类 tuned 配置语义：

```yaml
focal_alpha: [0.2, 0.7, 0.1]
```

### 6. sparse refine loss 独立线性 warmup

`ligand_sparse_refine_loss` 的 effective weight 独立调度，不复用 LR scheduler warmup：

```yaml
model:
  ligand_sparse_refine_loss_weight: 1.0
  ligand_sparse_refine_loss_schedule:
    mode: linear_warmup
    start_weight: 0.0
    final_weight: ${model.ligand_sparse_refine_loss_weight}
    warmup_steps: null
    warmup_ratio: 0.15
```

字段含义：

| 字段 | 类型 | 默认 | 意义 |
|---|---|---:|---|
| `mode` | `str` | `linear_warmup` | 首版只支持线性 warmup。 |
| `start_weight` | `float` | `0.0` | 训练开始时 sparse refine loss 的有效权重。 |
| `final_weight` | `float` | `${model.ligand_sparse_refine_loss_weight}` | warmup 结束后的有效权重。 |
| `warmup_steps` | `int|null` | `null` | 显式 step 数；非 null 时优先使用。 |
| `warmup_ratio` | `float` | `0.15` | `warmup_steps=null` 时，使用总 optimizer step 的比例。 |

计算公式：

```python
if warmup_steps == 0:
    effective_weight = final_weight
else:
    progress = min(max(global_step / warmup_steps, 0.0), 1.0)
    effective_weight = start_weight + progress * (final_weight - start_weight)
```

validation loss 使用当前 `global_step` 下的 effective weight；所有 best-F1 / recall 指标不受 loss 权重影响。

### 7. 实验命名

| 任务 | experiment config | loss config | 说明 |
|---|---|---|---|
| 二分类主实验 | `configs/experiment/bi_sparse_refine_full.yaml` | `configs/loss/bi_sparse_refine_7focal_3dice.yaml` | foreground sparse refine 主线。 |
| 三分类正式配置 | `configs/experiment/tri_sparse_refine_full.yaml` | `configs/loss/tri_sparse_refine_7focal_3dice_hard_tuned.yaml` | 三分类功能验证与后续训练入口。 |

不新增临时 `tri_sparse_refine_smoke.yaml`；fast-dev-run 直接使用正式配置。

## Proposed Changes

### 1. 统一 ligand target helper

#### [MODIFY] [src/modules/losses.py](../../../../src/modules/losses.py)

在 `AdaptiveClassificationCompositeLoss` 中新增公开方法：

```python
def target_from_ligand_dist_map(
    self,
    ligand_dist_map: torch.Tensor,
    logit_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor: ...
```

| 参数 | 类型 / shape | 意义 |
|---|---|---|
| `ligand_dist_map` | 二分类 `(B,D,H,W)` 或 `(B,1,D,H,W)`；多分类 `(B,C,D,H,W)` | ligand 距离监督图。 |
| `logit_dim` | `int` | 当前 logits 通道数。二分类为 1，多分类为类别数。 |
| `device` | `torch.device` | 输出 target 所在设备。 |
| `dtype` | `torch.dtype` | 距离图计算 dtype。 |
| 返回值 | `(B,D,H,W)` long | hard-label target。 |

行为：

1. `self.hard_label_threshold is None` 时抛 `ValueError`。
2. `logit_dim == 1`：
   - 接受 `(B,D,H,W)`。
   - 接受 `(B,1,D,H,W)` 并 squeeze channel。
   - 返回 `(dist < hard_label_threshold).long()`。
   - 若传入 `(B,C,D,H,W)` 且 `C != 1`，抛 `ValueError`，不猜测多通道合并。
3. `logit_dim > 1`：
   - 要求 `logit_dim == self.num_classes`。
   - 要求 `ligand_dist_map.ndim == 5` 且 `ligand_dist_map.shape[1] == logit_dim`。
   - 前景 channel `1..C-1` 中距离最小且 `< hard_label_threshold` 的类别为 target，否则为 0。

同步修改：

1. `_target_from_multiclass_dist()` 内部可保留为私有实现，但 `_forward_multiclass()` 的 `target is None` 分支必须调用 `target_from_ligand_dist_map()`。
2. 不新增 `sparse_target_from_ligand_dist_map()`。

### 2. wrapper 新增 sparse refine loss 与 loss schedule

#### [MODIFY] [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

##### 2.1 `VoxelPointStage1Wrapper.__init__()` 新增参数

```python
ligand_sparse_refine_loss: nn.Module | None = None
ligand_sparse_refine_loss_weight: float = 0.0
ligand_sparse_refine_loss_schedule: dict[str, Any] | None = None
```

实现要求：

1. `save_hyperparameters(ignore=[...])` 中加入 `ligand_sparse_refine_loss`。
2. 对 `ligand_sparse_refine_loss` 执行与其它 loss 相同的 Hydra instantiate。
3. 若 `ligand_sparse_refine_loss is not None` 且不是 `AdaptiveClassificationCompositeLoss`，抛 `TypeError`。
4. 若 `ligand_sparse_refine_loss is not None` 且 `_sparse_candidate_class_ids is None`，抛 `ValueError`。
5. 校验 `ligand_sparse_refine_loss.num_classes` 与 `candidate_class_ids`：
   - `num_classes <= 2` 只允许 `(1,)`。
   - `num_classes > 2` 要求所有 class id 满足 `1 <= class_id < num_classes`。
6. 当 `ligand_sparse_refine_loss is not None` 时初始化 sparse refine metric buffers。

##### 2.2 新增 schedule 解析

新增方法：

```python
def _resolve_ligand_sparse_refine_loss_warmup_steps(self) -> int: ...
```

行为：

1. `ligand_sparse_refine_loss_schedule is None`：返回 `0`。
2. `mode` 只允许 `linear_warmup`。
3. `warmup_steps is not None`：返回非负整数。
4. `warmup_steps is None`：使用 `trainer.estimated_stepping_batches * warmup_ratio` 四舍五入得到 step 数。
5. `warmup_ratio` 必须在 `[0, 1)`。

新增方法：

```python
def _ligand_sparse_refine_loss_effective_weight(self) -> torch.Tensor: ...
```

返回当前 `global_step` 对应的 scalar tensor。该值用于 train/val total loss，并记录为：

```text
train/ligand_sparse_refine_loss_weight_effective
val/ligand_sparse_refine_loss_weight_effective
```

##### 2.3 新增 valid mask helper

```python
@staticmethod
def _normalize_voxel_valid_mask(
    voxel_valid_mask: torch.Tensor,
    spatial_shape_zyx: tuple[int, int, int],
) -> torch.Tensor: ...
```

行为：

1. 接受 `(B,D,H,W)` 或 `(B,1,D,H,W)`。
2. 输出 `(B,D,H,W)` bool。
3. 空间维度必须等于 `spatial_shape_zyx`，否则 fail-fast。

##### 2.4 新增 C 级监督采样

```python
def _sample_ligand_refine_supervision(
    self,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]: ...
```

返回字段：

| 字段 | shape | 意义 |
|---|---:|---|
| `ligand_dense_target` | `(B,D,H,W)` | dense hard-label target。 |
| `ligand_dense_valid_mask` | `(B,D,H,W)` | dense valid mask。 |
| `ligand_refine_target_C` | `(sumC,)` | C 上 hard-label target。 |
| `ligand_refine_valid_mask_C` | `(sumC,)` | C 上 valid mask。 |

伪代码：

```python
logits_C = outputs["ligand_refine_logits_C"]
logit_dim = int(logits_C.shape[1])
dense_target = self.ligand_sparse_refine_loss.target_from_ligand_dist_map(
    ligand_dist_map=batch["ligand_dist_map"],
    logit_dim=logit_dim,
    device=logits_C.device,
    dtype=logits_C.dtype,
)
dense_valid_mask = self._normalize_voxel_valid_mask(
    batch["voxel_valid_mask"],
    spatial_shape_zyx=tuple(dense_target.shape[-3:]),
).to(device=logits_C.device)
idx_b = outputs["candidate_batch_index"].to(device=logits_C.device)
idx_zyx = outputs["candidate_voxel_zyx"].to(device=logits_C.device)
target_C = dense_target[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
valid_C = dense_valid_mask[idx_b, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
```

该方法可以把 `ligand_refine_target_C` 与 `ligand_refine_valid_mask_C` 写回 `outputs`，但不要求 backbone 输出这些字段。

##### 2.5 新增 sparse refine loss 计算

```python
def _compute_ligand_sparse_refine_loss(
    self,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> torch.Tensor | None: ...
```

行为：

1. `ligand_sparse_refine_loss is None` 时返回 `None`。
2. `outputs.get("ligand_refine_logits_C") is None` 时返回 `None`。
3. 调用 `_sample_ligand_refine_supervision()`。
4. 调用：

```python
loss_out = self.ligand_sparse_refine_loss(
    logits=outputs["ligand_refine_logits_C"],
    target=supervision["ligand_refine_target_C"],
    hardmask=supervision["ligand_refine_valid_mask_C"],
)
```

5. 不传 `candidate_message_valid_mask`。
6. 返回 `_loss_output_to_tensor(loss_out)`。

##### 2.6 修改总 loss 汇总

在 `_compute_total_loss()` 中追加：

```python
ligand_sparse_refine_loss = self._compute_ligand_sparse_refine_loss(outputs, batch)
if ligand_sparse_refine_loss is not None:
    effective_weight = self._ligand_sparse_refine_loss_effective_weight()
    total_loss = total_loss + effective_weight * ligand_sparse_refine_loss
    loss_dict["ligand_sparse_refine_loss"] = ligand_sparse_refine_loss
    loss_dict["ligand_sparse_refine_loss_weight_effective"] = effective_weight
```

`training_step()` 与 `validation_step()` 新增日志：

```text
train/ligand_sparse_refine_loss
train/ligand_sparse_refine_loss_weight_effective
val/ligand_sparse_refine_loss
val/ligand_sparse_refine_loss_weight_effective
```

### 3. wrapper 局部整理 metric 状态与日志

#### [MODIFY] [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

新增轻量私有状态结构，可使用 `dataclass` 或普通 dict：

```python
@dataclass
class _MetricLogState:
    name: str
    updated: bool
    value: torch.Tensor | None = None
```

也可以使用现有 `_val_metric_update_counts`，但必须统一以下规则：

1. 无有效 update 的 metric 不调用 `self.log()`。
2. 无有效 update 的 metric 不写入 `computed_metrics`。
3. macro 只对有 update 的类别求平均。
4. 若 monitor metric 缺失，保持 scheduler / checkpoint fail-fast。

修改 `_compute_log_reset_metric_safe()`：

```python
def _compute_log_reset_metric_safe(self, metric_obj, metric_name: str) -> torch.Tensor | None: ...
```

行为：

1. `local_updates <= 0`：reset metric，清零 update count，返回 `None`，不 log。
2. 有 update：compute、reset、log、返回 scalar tensor。

修改 `on_validation_epoch_end()` 调用方式：

```python
metric_value = self._compute_log_reset_metric_safe(...)
if metric_value is not None:
    computed_metrics[metric_name] = metric_value
```

修改 `_compute_log_reset_multiclass_metrics()`：

1. 单类无 update：reset 该 metric，不 log，不写入 `computed_metrics`。
2. macro 无任何 class score：不 log、不写入 `computed_metrics`。
3. macro 有 class score：记录 macro。

### 4. dense before-refine best-F1

#### [MODIFY] [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

修改 `_compute_log_update_voxel_ligand_best_f1_thresholds()`：

1. 保留 threshold cache 指标：

```text
val/voxel_ligand_p_best_by_class_{class_name}
val/voxel_ligand_p_sampling_by_class_{class_name}
```

2. 只输出新的 before-refine best-F1 名称：

```text
val/voxel_ligand_best_f1_before_refine_{class_name}
val/voxel_ligand_best_f1_before_refine_macro
```

3. 不再输出旧名称：

```text
val/voxel_ligand_best_f1_by_class_{class_name}
val/voxel_ligand_macro_best_f1_by_class
```

4. 无 GT 正例的 class 不记录 before-refine best-F1。
5. 所有 candidate class 均无 GT 正例时 fail-fast。

### 5. sparse refine best-F1、candidate recall 与 C/P 数量

#### [MODIFY] [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

##### 5.1 新增 buffers

当 `ligand_sparse_refine_loss is not None` 时注册：

| 字段 | shape | dtype | 意义 |
|---|---:|---|---|
| `_ligand_sparse_refine_pos_hist_by_class` | `(K,num_bins)` | long | C 内正例 score histogram。 |
| `_ligand_sparse_refine_neg_hist_by_class` | `(K,num_bins)` | long | C 内负例 score histogram。 |
| `_ligand_sparse_refine_total_gt_pos_by_class` | `(K,)` | long | dense valid GT 正例数。 |
| `_ligand_sparse_refine_candidate_count_total` | `()` | long | validation loop 内 C 总数。 |
| `_ligand_sparse_refine_anchor_count_total` | `()` | long | validation loop 内 P 总数。 |
| `_ligand_sparse_refine_box_count_total` | `()` | long | validation loop 内 BOX 总数。 |

`num_bins` 复用 `voxel_ligand_pr_auc_thresholds`；启用 sparse refine metrics 时该值必须为正整数。

新增 reset：

```python
def _reset_ligand_sparse_refine_metric_histograms(self) -> None: ...
```

在 `on_validation_epoch_start()` 与 sparse refine metric compute 后调用。

##### 5.2 validation step 更新

新增：

```python
def _update_val_ligand_sparse_refine_metric(
    self,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> None: ...
```

调用位置：`validation_step()` 中 `_update_val_voxel_ligand_metric(...)` 之后。

更新流程：

```python
supervision = self._sample_ligand_refine_supervision(outputs, batch)
logits_C = outputs["ligand_refine_logits_C"]
valid_C = supervision["ligand_refine_valid_mask_C"]
target_C = supervision["ligand_refine_target_C"]
dense_valid = supervision["ligand_dense_valid_mask"]
dense_target = supervision["ligand_dense_target"]
```

对每个 `class_pos, class_id`：

```python
if logits_C.shape[1] == 1:
    score_C = torch.sigmoid(logits_C[:, 0]).detach().float()
else:
    score_C = torch.softmax(logits_C, dim=1)[:, class_id].detach().float()

score_valid = score_C[valid_C]
target_valid = target_C[valid_C]
positive_C = target_valid == class_id

bin_idx = torch.floor(score_valid.clamp(0.0, 1.0) * num_bins).long().clamp(max=num_bins - 1)
pos_hist += torch.bincount(bin_idx[positive_C], minlength=num_bins)
neg_hist += torch.bincount(bin_idx[~positive_C], minlength=num_bins)

total_gt_pos += ((dense_valid) & (dense_target == class_id)).sum()
```

> [!IMPORTANT]
> `((dense_valid) & (dense_target == class_id))` 必须显式加括号，避免 Python 运算符优先级导致多分类 GT 统计错误。

数量累计：

```python
candidate_count_total += outputs["candidate_counts"].sum()
anchor_count_total += outputs["anchor_counts"].sum()
box_count_total += dense_target.shape[0]
```

##### 5.3 validation end 计算

新增：

```python
def _compute_log_reset_ligand_sparse_refine_metrics(self) -> dict[str, torch.Tensor]: ...
```

在 `on_validation_epoch_end()` 中调用顺序：

```python
computed_metrics.update(self._compute_log_reset_multiclass_metrics())
computed_metrics.update(self._compute_log_update_voxel_ligand_best_f1_thresholds())
computed_metrics.update(self._compute_log_reset_ligand_sparse_refine_metrics())
self._step_warmup_plateau_scheduler(computed_metrics)
```

计算公式：

```python
tp_at_threshold = torch.cumsum(pos_hist.flip(-1), dim=-1).flip(-1)
fp_at_threshold = torch.cumsum(neg_hist.flip(-1), dim=-1).flip(-1)
fn_at_threshold = total_gt_pos[:, None] - tp_at_threshold
denominator = 2.0 * tp_at_threshold + fp_at_threshold + fn_at_threshold
f1_by_threshold = torch.where(
    denominator > 0.0,
    2.0 * tp_at_threshold / denominator,
    torch.zeros_like(denominator),
)
```

记录：

```text
val/ligand_sparse_refine_best_f1_{class_name}
val/ligand_sparse_refine_best_f1_macro
val/candidate_recall_{class_name}
val/candidate_recall_macro
val/num_candidate_voxels
val/num_anchor_points
```

规则：

1. `total_gt_pos[class_pos] == 0`：跳过该 class。
2. 所有 class 的 `total_gt_pos == 0`：抛 `RuntimeError`。
3. `total_gt_pos > 0` 但 `pos_hist.sum() == 0`：recall=0，best-F1=0，正常记录。
4. DDP 下对 hist 与 count buffer 做 `all_gather(...).sum(dim=0)`。
5. `val/num_candidate_voxels` 与 `val/num_anchor_points` 为 validation loop 内平均每 BOX 数量。

### 6. 配置更新

#### [MODIFY] [configs/model/default.yaml](../../../../configs/model/default.yaml)

新增默认关闭字段：

```yaml
ligand_sparse_refine_loss: null
ligand_sparse_refine_loss_weight: 0.0
ligand_sparse_refine_loss_schedule: null
```

#### [NEW] [configs/loss/bi_sparse_refine_7focal_3dice.yaml](../../../../configs/loss/bi_sparse_refine_7focal_3dice.yaml)

新增二分类四路 loss preset：

```yaml
# @package _global_
model:
  atom_loss:
    _target_: src.modules.losses.AdaptiveClassificationCompositeLoss
    num_classes: 2
    sigma: 2.0
    hard_label_threshold: null
    focal_gamma: 2.0
    focal_alpha: [0.5, 0.5]
    focal_alpha_neg: 0.5
    focal_alpha_pos: 0.5
    focal_eps: 1.0e-6
    tversky_alpha: 0.5
    tversky_beta: 0.5
    tversky_smooth: 1.0
    w_focal: 0.7
    w_tversky: 0.3
    w_mse: 0.0
    focal_soft_negative_suppression: false
    tversky_soft_target: false
  atom_loss_weight: 0.1

  voxel_aux_loss:
    _target_: src.modules.losses.AdaptiveClassificationCompositeLoss
    num_classes: 2
    sigma: 2.0
    hard_label_threshold: null
    focal_gamma: 2.0
    focal_alpha: [0.5, 0.5]
    focal_alpha_neg: 0.5
    focal_alpha_pos: 0.5
    focal_eps: 1.0e-6
    tversky_alpha: 0.5
    tversky_beta: 0.5
    tversky_smooth: 1.0
    w_focal: 0.7
    w_tversky: 0.3
    w_mse: 0.0
    focal_soft_negative_suppression: false
    tversky_soft_target: false
  voxel_aux_loss_weight: 0.1

  voxel_ligand_loss:
    _target_: src.modules.losses.AdaptiveClassificationCompositeLoss
    num_classes: 2
    sigma: 2.0
    hard_label_threshold: 1.7
    focal_gamma: 2.0
    focal_alpha: [0.5, 0.5]
    focal_alpha_neg: 0.5
    focal_alpha_pos: 0.5
    focal_eps: 1.0e-6
    tversky_alpha: 0.5
    tversky_beta: 0.5
    tversky_smooth: 1.0
    w_focal: 0.7
    w_tversky: 0.3
    w_mse: 0.0
    focal_soft_negative_suppression: false
    tversky_soft_target: false
  voxel_ligand_loss_weight: 1.0

  ligand_sparse_refine_loss:
    _target_: src.modules.losses.AdaptiveClassificationCompositeLoss
    num_classes: 2
    sigma: 2.0
    hard_label_threshold: 1.7
    focal_gamma: 2.0
    focal_alpha: [0.5, 0.5]
    focal_alpha_neg: 0.5
    focal_alpha_pos: 0.5
    focal_eps: 1.0e-6
    tversky_alpha: 0.5
    tversky_beta: 0.5
    tversky_smooth: 1.0
    w_focal: 0.7
    w_tversky: 0.3
    w_mse: 0.0
    focal_soft_negative_suppression: false
    tversky_soft_target: false
  ligand_sparse_refine_loss_weight: 1.0
  ligand_sparse_refine_loss_schedule:
    mode: linear_warmup
    start_weight: 0.0
    final_weight: ${model.ligand_sparse_refine_loss_weight}
    warmup_steps: null
    warmup_ratio: 0.15
```

#### [NEW] [configs/loss/tri_sparse_refine_7focal_3dice_hard_tuned.yaml](../../../../configs/loss/tri_sparse_refine_7focal_3dice_hard_tuned.yaml)

仿照 [configs/loss/tri_7focal_3dice_hard_tuned.yaml](../../../../configs/loss/tri_7focal_3dice_hard_tuned.yaml)，新增四路三分类 loss：

1. `atom_loss`：`AdaptiveClassificationCompositeLoss(num_classes=3, hard_label_threshold=null, focal_alpha=[0.2,0.7,0.1], w_focal=0.7, w_tversky=0.3)`，权重 `0.1`。
2. `voxel_aux_loss`：同现有三分类 aux loss，权重 `0.1`。
3. `voxel_ligand_loss`：同现有三分类 ligand loss，权重 `1.0`。
4. `ligand_sparse_refine_loss`：同三分类 ligand loss，权重 final `1.0`，并配置同样的 `ligand_sparse_refine_loss_schedule`。

#### [NEW] [configs/experiment/bi_sparse_refine_full.yaml](../../../../configs/experiment/bi_sparse_refine_full.yaml)

新增二分类主实验：

```yaml
# @package _global_
defaults:
  - override /model/task: binary
  - override /model/voxel_backbone: default
  - override /model/point_backbone: default
  - override /model/fusion: default
  - override /model/atom_head: default
  - override /model/embed_head: clipbig
  - override /model/sparse_refine/candidate_set: binary
  - override /model/sparse_refine/anchor_sampler: weighted_fps
  - override /model/sparse_refine/density_cube: default
  - override /model/sparse_refine/anchor_to_candidate: knn_message
  - override /model/sparse_refine/sparse_refine_head: default
  - override /model/sparse_refine/anchor_class_conditioning: none
  - override /loss: bi_sparse_refine_7focal_3dice
  - override /dataset: L_small_full_buffer
  - override /train: B40_L3

name: bi_sparse_refine_full
tag: "bi_sparse_refine_full"
experiment_group: "ligand_sparse_refine"
project_name: PV_ligand_sparse_refine

dataset:
  class_mapping: [0, 1, 1, 1, 1]
  class_names: [background, foreground]
  num_task_classes: 2

model:
  class_names: ${dataset.class_names}
  monitor_metric: val/ligand_sparse_refine_best_f1_macro
  monitor_mode: max
```

#### [NEW] [configs/experiment/tri_sparse_refine_full.yaml](../../../../configs/experiment/tri_sparse_refine_full.yaml)

新增三分类正式配置：

```yaml
# @package _global_
defaults:
  - override /model/task: tri
  - override /model/voxel_backbone: default
  - override /model/point_backbone: default
  - override /model/fusion: default
  - override /model/atom_head: default
  - override /model/embed_head: clipbig
  - override /model/sparse_refine/candidate_set: tri
  - override /model/sparse_refine/anchor_sampler: weighted_fps
  - override /model/sparse_refine/density_cube: default
  - override /model/sparse_refine/anchor_to_candidate: knn_message
  - override /model/sparse_refine/sparse_refine_head: default
  - override /model/sparse_refine/anchor_class_conditioning: none
  - override /loss: tri_sparse_refine_7focal_3dice_hard_tuned
  - override /dataset: L_small_full_buffer
  - override /train: B40_L3

name: tri_sparse_refine_full
tag: "tri_sparse_refine_full"
experiment_group: "ligand_sparse_refine"
project_name: PV_ligand_sparse_refine

dataset:
  class_mapping: [0, 1, 0, 0, 2]
  class_names: [background, metal_ion, small_molecule]
  num_task_classes: 3

model:
  class_names: ${dataset.class_names}
  monitor_metric: val/ligand_sparse_refine_best_f1_macro
  monitor_mode: max
```

### 7. 后续 train.py + wrapper 通用化方向

#### [NO CHANGE] [src/train.py](../../../../src/train.py)

本阶段不修改 [src/train.py](../../../../src/train.py) 行为。

后续方向记录：

1. [src/train.py](../../../../src/train.py) 长期只负责通用 orchestration：配置加载、DataModule、logger、checkpoint、Trainer、运行目录。
2. [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) 负责 loss/metric 注册、命名、更新、日志与 monitor 可用性。
3. wrapper 内部应继续收敛 metric 逻辑，减少指标散落在 `validation_step()` 与 `on_validation_epoch_end()` 的分支。
4. 不在 train.py 中写具体任务指标名、分支判断或可视化逻辑。

## 不修改的部分

1. 不实现 sparse refine PR-AUC。
2. 不把 `ligand_refine_logits_C` scatter 成 dense volume 训练或评估。
3. 不把 `candidate_message_valid_mask` 加入 loss mask。
4. 不新增复杂 message coverage 或 message-valid 分组指标。
5. 不改变 C 生成、P 采样、P -> C KNN message 或 sparse head 前向契约。
6. 不新增 `candidate_class` 限制下的 per-route 监督。
7. 不新增临时 tri smoke experiment；测试直接使用 `tri_sparse_refine_full`。
8. 不修改 [src/train.py](../../../../src/train.py) 运行行为。

## 改动文件汇总

| 文件 | 改动内容 |
|---|---|
| [src/modules/losses.py](../../../../src/modules/losses.py) | `AdaptiveClassificationCompositeLoss` 新增 `target_from_ligand_dist_map()`，并统一多分类 target 生成。 |
| [src/wrappers/voxel_point_stage1.py](../../../../src/wrappers/voxel_point_stage1.py) | 新增 sparse refine loss、独立 loss warmup、C 级监督采样、sparse refine metrics、无 update 不 log、wrapper metric 局部整理。 |
| [configs/model/default.yaml](../../../../configs/model/default.yaml) | 增加默认关闭的 sparse refine loss 与 schedule 字段。 |
| [configs/loss/bi_sparse_refine_7focal_3dice.yaml](../../../../configs/loss/bi_sparse_refine_7focal_3dice.yaml) | 新增二分类四路 loss preset。 |
| [configs/loss/tri_sparse_refine_7focal_3dice_hard_tuned.yaml](../../../../configs/loss/tri_sparse_refine_7focal_3dice_hard_tuned.yaml) | 新增三分类四路 loss preset。 |
| [configs/experiment/bi_sparse_refine_full.yaml](../../../../configs/experiment/bi_sparse_refine_full.yaml) | 新增二分类主实验。 |
| [configs/experiment/tri_sparse_refine_full.yaml](../../../../configs/experiment/tri_sparse_refine_full.yaml) | 新增三分类正式实验。 |
| [tests/test_ligand_sparse_refine_loss.py](../../../../tests/test_ligand_sparse_refine_loss.py) | 新增 target helper、C 级 loss、loss schedule 单测。 |
| [tests/test_ligand_sparse_refine_metrics.py](../../../../tests/test_ligand_sparse_refine_metrics.py) | 新增 sparse refine best-F1、candidate recall、C/P 数量、无 update 不 log 测试。 |
| [tests/test_voxel_ligand_thresholds.py](../../../../tests/test_voxel_ligand_thresholds.py) | 更新 before-refine best-F1 新日志名与旧日志名删除断言。 |

## Verification Plan

### Automated Tests

#### loss helper 与 C 级 loss

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\test_ligand_sparse_refine_loss.py -q
```

| 测试函数 | 构造 | 关键断言 |
|---|---|---|
| `test_target_from_ligand_dist_map_binary_accepts_4d_dist` | `(B,D,H,W)` binary dist | 返回 `(B,D,H,W)` 0/1 target。 |
| `test_target_from_ligand_dist_map_binary_accepts_single_channel_5d_dist` | `(B,1,D,H,W)` binary dist | squeeze 后 target 正确。 |
| `test_target_from_ligand_dist_map_binary_rejects_multiclass_dist` | `logit_dim=1` + `(B,3,D,H,W)` dist | fail-fast。 |
| `test_target_from_ligand_dist_map_multiclass_uses_nearest_foreground` | `(B,3,D,H,W)` dist | 最近前景且小于阈值得到 class id，否则 0。 |
| `test_sparse_refine_loss_uses_voxel_valid_mask_not_message_mask` | `candidate_message_valid_mask=False` 但 C valid=True | loss hardmask 仍包含该 C。 |
| `test_sparse_refine_loss_accepts_binary_C_logits` | logits `(sumC,1)` | 返回 finite scalar。 |
| `test_sparse_refine_loss_accepts_multiclass_C_logits` | logits `(sumC,3)` | 返回 finite scalar。 |
| `test_ligand_sparse_refine_loss_schedule_linear_warmup` | global_step 0/mid/end | effective weight 为 0/中间值/final。 |

#### sparse refine metrics

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\test_ligand_sparse_refine_metrics.py -q
```

| 测试函数 | 构造 | 关键断言 |
|---|---|---|
| `test_sparse_refine_best_f1_counts_missing_candidates_as_fn` | dense 有 2 个正体素，C 覆盖 1 个 | FN 包含漏检正体素，best-F1 低于全覆盖。 |
| `test_candidate_recall_is_covered_positive_over_total_positive` | dense 正体素 4 个，C 覆盖 3 个 | `candidate_recall_foreground == 0.75`。 |
| `test_candidate_recall_zero_when_gt_exists_but_C_misses_all` | dense 有 GT，C 无正例覆盖 | recall=0，best-F1=0，并正常 log。 |
| `test_sparse_refine_metrics_skip_single_class_without_gt` | 三分类中某一类无 GT | 跳过该类，不纳入 macro。 |
| `test_sparse_refine_metrics_fail_when_all_candidate_classes_have_no_gt` | 所有 candidate class 均无 GT | `RuntimeError`。 |
| `test_sparse_refine_metric_names_follow_binary_class_names` | `class_names=[background, foreground]` | 只输出 foreground 名称。 |
| `test_sparse_refine_metric_names_follow_tri_class_names` | `class_names=[background, metal_ion, small_molecule]` | 输出 metal_ion/small_molecule。 |
| `test_num_candidate_and_anchor_points_are_mean_per_box` | 两个 BOX 不同 C/P 数 | 数量日志为 per-box 均值。 |
| `test_metric_without_updates_is_not_logged` | metric update count 为 0 | 不调用 `self.log()`，不返回 `computed_metrics`。 |

#### dense before-refine threshold metrics

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\test_voxel_ligand_thresholds.py -q
```

| 测试函数 | 关键断言 |
|---|---|
| `test_voxel_ligand_best_f1_logs_before_refine_name` | 输出 `val/voxel_ligand_best_f1_before_refine_*`。 |
| `test_voxel_ligand_best_f1_does_not_log_old_by_class_name` | 不输出 `val/voxel_ligand_best_f1_by_class_*`。 |
| `test_threshold_cache_logs_p_best_and_p_sampling` | `p_best_by_class` / `p_sampling_by_class` 仍记录。 |
| `test_threshold_metrics_fail_when_all_candidate_classes_have_no_gt` | 全 candidate class 无 GT 时 fail-fast。 |

#### 05 前向链路回归

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m pytest tests\model\test_sparse_candidate_set.py tests\model\test_anchor_sampler.py tests\model\test_anchor_to_candidate.py tests\model\test_sparse_refine_head.py tests\model\test_stage1_model.py -q
```

预期：C 唯一化、P sampling、KNN message、`candidate_message_valid_mask`、`ligand_refine_logits_C` 不回归。

### Config / Hydra Smoke Tests

二分类主实验：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m src.train +experiment=bi_sparse_refine_full trainer.fast_dev_run=true
```

验证点：

1. 四路 loss 均实例化为 `AdaptiveClassificationCompositeLoss`。
2. dense logits 为 `(B,1,D,H,W)`，refine logits 为 `(sumC,1)`。
3. 指标名使用 `foreground`。
4. `val/ligand_sparse_refine_best_f1_macro` 可作为 monitor。
5. `ligand_sparse_refine_loss_weight_effective` 在 step 0 附近为 0，并随 step 上升。

三分类正式配置：

```powershell
& 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m src.train +experiment=tri_sparse_refine_full trainer.fast_dev_run=true
```

验证点：

1. 四路 loss 均为 `num_classes=3`。
2. `candidate_class_ids=[1,2]`。
3. `ligand_refine_logits_C.shape[-1] == 3`。
4. 指标名使用 `metal_ion` / `small_molecule`。

### Manual Checks

```powershell
rg "sparse_target_from_ligand_dist_map|candidate_interp_valid_mask|ligand_sparse_refine_pr_auc|sparse_refine_pr_auc" src configs tests CLAUDE\plans\implement\tri_ligand_sparse_refine
```

预期：新实现与新计划不使用这些旧字段或非目标功能。

```powershell
rg "voxel_ligand_best_f1_by_class|voxel_ligand_macro_best_f1_by_class" src tests configs
```

预期：不再输出旧 dense best-F1 名称；历史计划文件命中不算代码阻塞。

```powershell
rg "voxel_ligand_best_f1_before_refine|ligand_sparse_refine_best_f1|candidate_recall|num_candidate_voxels|num_anchor_points" src tests configs
```

预期：新指标名在 wrapper、测试与配置中可追踪。

## Acceptance Criteria

1. sparse refine loss 在二分类 `(sumC,1)` 与三分类 `(sumC,3)` 路径均可训练。
2. `ligand_refine_target_C.shape == (sumC,)`，`ligand_refine_valid_mask_C.shape == (sumC,)`。
3. sparse refine loss 只使用 C 位置采样到的 `voxel_valid_mask`，不使用 `candidate_message_valid_mask`。
4. 四路 loss 均使用 `AdaptiveClassificationCompositeLoss`，权重为 `0.1 / 0.1 / 1.0 / scheduled 1.0`。
5. refine loss effective weight 独立线性 warmup，默认 `warmup_ratio=0.15`。
6. 无有效 update 的指标不再向 W&B 记录 0。
7. `val/voxel_ligand_best_f1_before_refine_*` 输出 refine 前 dense best-F1，旧 dense best-F1 名称不再输出。
8. `val/ligand_sparse_refine_best_f1_*` 的 FN 包含未进入 C 的 GT 正体素。
9. `val/candidate_recall_*` 等于 C 覆盖 GT 正体素比例；有 GT 但覆盖为 0 时记录 0。
10. 单类无 GT 跳过，全 candidate 类无 GT fail-fast。
11. `val/num_candidate_voxels` 与 `val/num_anchor_points` 输出 validation loop 内平均每 BOX 数量。
12. 二分类主实验文件为 `bi_sparse_refine_full.yaml`，三分类正式配置为 `tri_sparse_refine_full.yaml`。
