# VoxelPointStage1Wrapper 分层重构与验证诊断计划

## 背景与目标

`src/wrappers/voxel_point_stage1.py` 当前约 2072 行，单个 `VoxelPointStage1Wrapper` 同时承担 Hydra 依赖构造、loss 计算、训练/验证 step、torchmetrics AP、candidate threshold histogram、sparse refine C/P 统计、checkpoint metadata 和 scheduler 管理。用户目标是大幅重构该文件，把大的逻辑分层拆开，降低初学者阅读压力，并清理二分类下重复记录的 `*_macro` 指标。

本次计划采用已确认方案：保留 `src.wrappers.voxel_point_stage1.VoxelPointStage1Wrapper` 作为唯一对外 LightningModule 入口，拆出同目录 helper 模块承接细节逻辑。日志层彻底切换到 `val_loss/*`、`val_score/*`、`val_pc/*` 三类，不保留 `val/*` 旧日志别名；旧训练 YAML 原样重跑时需要同步修改 `monitor_metric`。推理管线不依赖训练日志名，旧 checkpoint 仍可用于推理，只要不改 backbone 模块名与 state_dict key。

用户额外确认：`aux` 在业务语义上是 receptor head。日志、指标、计划与新配置命名统一使用 `receptor`；但模型内部输出 key `voxel_logits_aux`、模块名 `voxel_aux_head`、checkpoint 权重 key 暂不改名，避免破坏旧 checkpoint 的 `strict=True` 推理加载。

## 当前架构 / 已知约束

- `configs/model/default.yaml` 的 `_target_` 指向 `src.wrappers.voxel_point_stage1.VoxelPointStage1Wrapper`，入口类名和模块路径必须保留。
- `src/train.py` 从 `cfg.model.monitor_metric` 构造 `ModelCheckpoint.monitor`，文件名模板也引用该字段。日志彻底改名后，所有仍在使用的训练配置必须把 `monitor_metric` 改成新指标名。
- `src/inference/get_pred.py` 的推理加载只提取 `checkpoint["state_dict"]` 中的 `backbone.*` 权重，并剥去 `backbone.` 前缀后 `model.load_state_dict(..., strict=True)`。训练期 W&B/Lightning 指标名不参与推理。
- `src/inference/get_pred.py` 已把推理输出头 `"receptor"` 映射到 backbone 输出 `"voxel_logits_aux"`，说明日志层把 `voxel_aux` 命名为 `receptor` 与现有推理语义一致。
- `src/datasets/box_point_dataset.py` 在单样本中写入 `sample_dict["class_name"]`；`src/datasets/box_point_collate.py` 会把非 tensor 元信息保留为 batch 中的 `class_name: list[str]`。因此 validation diagnostics 可以按 `batch["class_name"]` 做 folder 级分组。
- `configs/dataset/emb_unet.yaml` 使用 `class_folder_names: ["metal_ion", "peptide", "nucleic", "small_molecule", "random_BOX"]`，未来可能新增如 `random_BOX2` 的目录。分组统计应从配置显式传入 group names，而不是运行时只靠当前 rank 动态发现。
- 现有测试覆盖 wrapper threshold cache、sparse refine metrics、多分类 wrapper、inference receptor 输出等路径，重构后必须迁移或扩展这些测试。

## 已有可复用代码

| 已有代码 | 位置 | 可复用能力 |
|---|---|---|
| `VoxelPointStage1Wrapper._compute_total_loss()` | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 已有 atom/receptor(l旧 aux)/ligand/sparse refine loss 汇总逻辑，可迁移到 loss helper。 |
| `VoxelPointStage1Wrapper._update_binary_or_multiclass_ap()` | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 已有二分类 AP 与多分类逐类 AP 统一更新逻辑，可迁移到 metric helper 并改名。 |
| `VoxelPointStage1Wrapper._update_voxel_ligand_best_f1_stats()` | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 已有 dense threshold histogram 累加逻辑，可扩展 sampling threshold TP/FP/FN 与分组 histogram。 |
| `VoxelPointStage1Wrapper._compute_log_update_voxel_ligand_best_f1_thresholds()` | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 已有 p_best/p_sampling/best-F1 计算和 checkpoint cache 更新逻辑，可拆到 candidate diagnostics helper。 |
| `VoxelPointStage1Wrapper._compute_log_reset_ligand_sparse_refine_metrics()` | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 已有 candidate recall、C/P 平均数和 sparse refine best-F1 统计，可迁移并改为 `val_pc/*`。 |
| `batch["class_name"]` metadata | `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\datasets\box_point_dataset.py`, `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\datasets\box_point_collate.py` | 可直接用于按 `class_folder_names` 分组统计，无需实现 box/voxel 级明细。 |

## 设计决策

### 1. 保留入口类，细节拆模块

保留 `src/wrappers/voxel_point_stage1.py` 和 `VoxelPointStage1Wrapper` 入口，避免 Hydra `_target_`、训练脚本和旧 checkpoint 恢复路径改变。新模块放在 `src/wrappers/` 同目录下，命名使用 `voxel_point_stage1_*.py`，让初学者可先读主 wrapper 生命周期，再按职责进入 helper。

> [!IMPORTANT]
> 不改 backbone 内部 `voxel_logits_aux`、`voxel_aux_head`、`voxel_aux_logit_dim` 等权重相关名字。本次只改日志、指标、配置面向用户的命名。

### 2. 日志彻底改名，不保留旧 `val/*` 别名

新日志分三类：

| 类别 | 前缀 | 内容 |
|---|---|---|
| loss 类 | `train_loss/*`, `val_loss/*` | total loss、atom loss、receptor loss、voxel ligand loss、sparse refine loss、effective weight |
| 得分类 | `val_score/*` | atom/receptor/voxel ligand PR-AUC、多分类逐类 AP、dense best-F1、sparse refine best-F1 |
| P-C 详细统计类 | `val_pc/*` | p_best、p_sampling、threshold TP/FP/FN、candidate recall、num_C、num_P、PR curve/table、按 folder 分组统计 |

二分类不记录 `*_macro`；多分类才记录 `macro`。例如二分类 `voxel_ligand` 只记录 `val_score/voxel_ligand_pr_auc`，不记录 `val_score/voxel_ligand_macro_ap`。

### 3. 诊断默认轻量开启，使用 histogram 而非 voxel 明细

默认记录标量和 histogram 曲线。每次 validation 约 1024 bins、少量类别和 folder 分组，30 次 validation 的本地 CSV/JSON 与 W&B table/plot 代价可接受。明确不实现 box 级和 voxel 级明细，避免 GB 级日志。

### 4. 按 `class_folder_names` 分组统计

按 `batch["class_name"]` 做 folder 级诊断。建议通过新配置显式传入 `validation_diagnostics.group_names: ${dataset.class_folder_names}`，保证 DDP 各 rank 拥有一致分组顺序。未来新增 `random_BOX2` 时，只需把它加入 dataset 配置和 diagnostics group names。

## Proposed Changes

### 模块拆分

#### [MODIFY] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py`

保留 `VoxelPointStage1Wrapper`，将其缩减为生命周期编排层：

- `__init__`: 只负责 instantiate backbone/loss、保存 hparams、初始化 helper state、注册必要 buffer。
- `forward`: 继续委托 backbone。
- `training_step`: 调用 loss helper 计算损失，再调用 logging helper 写 `train_loss/*`。
- `validation_step`: 调用 loss helper、metric helper 和 diagnostics helper，再写 `val_loss/*`。
- `on_validation_epoch_start`: 统一调用 diagnostics reset。
- `on_validation_epoch_end`: 统一 compute/log/reset AP、threshold diagnostics、candidate diagnostics、scheduler step。
- `on_save_checkpoint` / `on_load_checkpoint`: 仅保留 wrapper 级 checkpoint metadata 编排，threshold cache 的张量规范化可迁移到 diagnostics helper。
- `configure_optimizers`: 可暂时保留在主 wrapper 或迁移到 scheduler helper；如果迁移，主 wrapper 只转发。

主 wrapper 新增或整理这些私有属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `_val_metrics` | helper 对象 | 管理 AP torchmetrics、update counts、device policy 与 compute/reset/log。 |
| `_loss_engine` | helper 对象或函数集合 | 管理四路 loss 计算与 total loss 汇总。 |
| `_candidate_diagnostics` | helper 对象 | 管理 dense threshold histogram、p_best/p_sampling、PR curve/table、按 folder 分组 histogram。 |
| `_sparse_refine_diagnostics` | helper 对象 | 管理 C/P 统计、candidate recall、sparse refine best-F1、按 folder 分组 C/P 统计。 |

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_losses.py`

新增 loss helper，建议暴露函数而非复杂类：

```python
def compute_stage1_losses(wrapper: Any, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
```

参数：

| 参数 | 类型 | 意义 |
|---|---|---|
| `wrapper` | `Any` | 当前 LightningModule，提供 loss modules、hparams、device 与 `_sample_ligand_refine_supervision()` 等既有方法。 |
| `outputs` | `dict[str, Any]` | backbone 前向输出。 |
| `batch` | `dict[str, Any]` | 当前训练/验证 batch。 |

返回：

| 返回 | 类型 | 意义 |
|---|---|---|
| `total_loss` | `torch.Tensor` | 标量，加权总损失。 |
| `loss_dict` | `dict[str, torch.Tensor]` | 分项损失，内部 key 仍可用 `voxel_aux_loss`，日志阶段映射为 `receptor_loss`。 |

迁移现有方法：

- `_compute_atom_loss`
- `_compute_voxel_aux_loss`
- `_normalize_voxel_valid_mask`
- `_sample_ligand_refine_supervision`
- `_compute_voxel_ligand_loss`
- `_resolve_ligand_sparse_refine_loss_warmup_steps`
- `_ligand_sparse_refine_loss_effective_weight`
- `_compute_ligand_sparse_refine_loss`
- `_compute_total_loss`

其中 `_compute_voxel_aux_loss` 可在 helper 内重命名为 `_compute_receptor_loss`，但读取 outputs 时仍使用 `outputs["voxel_logits_aux"]`。

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_metrics.py`

新增 AP metric helper，建议包含轻量类：

```python
class ValidationMetricManager:
    def __init__(self, owner: Any, class_names: list[str], device_policy: str) -> None:
        ...
```

职责：

- 创建 atom/receptor/voxel_ligand 的 BinaryAveragePrecision 或逐类 AP metric。
- 维护 `_val_metric_update_counts`、metric specs、metric devices。
- 提供 `update_atom(...)`、`update_receptor(...)`、`update_voxel_ligand(...)`。
- 提供 `compute_log_reset_all() -> dict[str, torch.Tensor]`。
- 二分类只记录主 AP；多分类才记录逐类 AP 与 macro AP。

日志命名映射：

| 旧名 | 新名 |
|---|---|
| `val/atom_pr_auc` | `val_score/atom_pr_auc` |
| `val/voxel_aux_pr_auc` | `val_score/receptor_pr_auc` |
| `val/voxel_ligand_pr_auc` | `val_score/voxel_ligand_pr_auc` |
| `val/{prefix}_ap_{class}` | `val_score/{prefix}_ap_{class}`，其中 `voxel_aux` 改为 `receptor` |
| `val/{prefix}_macro_ap` | `val_score/{prefix}_macro_ap`，仅多分类记录 |

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_diagnostics.py`

新增 validation diagnostics helper，建议定义两个状态类：

```python
@dataclass
class ValidationDiagnosticsConfig:
    enabled: bool
    log_pr_curve: bool
    write_local_tables: bool
    group_by_meta_field: str
    group_names: tuple[str, ...]
    output_subdir: str
```

```python
class CandidateThresholdDiagnostics:
    ...
```

```python
class SparseRefineDiagnostics:
    ...
```

`CandidateThresholdDiagnostics` 管理 dense voxel ligand histogram：

| 字段 | 类型 | 意义 |
|---|---|---|
| `pos_hist_by_class` | `torch.Tensor` | `(K,num_bins)`，全局每类正例概率 histogram。 |
| `neg_hist_by_class` | `torch.Tensor` | `(K,num_bins)`，全局每类负例概率 histogram。 |
| `pos_hist_by_group_class` | `torch.Tensor` | `(G,K,num_bins)`，按 `class_folder_names` 分组的正例 histogram。 |
| `neg_hist_by_group_class` | `torch.Tensor` | `(G,K,num_bins)`，按 folder 分组的负例 histogram。 |

计算并记录这些新指标：

| 指标 | 新日志名示例 | 含义 |
|---|---|---|
| `total_gt_pos` | `val_pc/total_gt_pos_foreground` | dense 有效区域 GT 正例数。 |
| `p_best` | `val_pc/p_best_foreground` | best-F1 阈值。 |
| `best_threshold_tp/fp/fn` | `val_pc/best_threshold_tp_foreground` 等 | `p_best` 阈值下 TP/FP/FN。 |
| `n_best_total` | `val_pc/n_best_total_foreground` | `p_best` 阈值下 TP+FP。 |
| `sampling_target_total` | `val_pc/sampling_target_total_foreground` | `ceil(n_best_total * adaptive_expand_factor)`。 |
| `p_sampling` | `val_pc/p_sampling_foreground` | recorded-threshold 使用的采样阈值。 |
| `sampling_threshold_tp/fp/fn` | `val_pc/sampling_threshold_tp_foreground` 等 | `p_sampling` 阈值下 TP/FP/FN。 |
| `sampling_threshold_total` | `val_pc/sampling_threshold_total_foreground` | `p_sampling` 阈值下 TP+FP。 |

`SparseRefineDiagnostics` 管理 C/P 统计：

| 指标 | 新日志名示例 | 含义 |
|---|---|---|
| `candidate_tp` | `val_pc/candidate_tp_foreground` | 实际 C 候选中的 GT 正例数。 |
| `candidate_recall` | `val_pc/candidate_recall_foreground` | `candidate_tp / total_gt_pos`。与用户提出的 `C_recall` 同义，保留 `candidate_recall`。 |
| `candidate_cutoff_prob` | `val_pc/candidate_cutoff_prob_foreground` | C 生成时实际保留候选的最低概率，可由 `outputs["candidate_p_sampling_by_class"]` 汇总。 |
| `num_C` | `val_pc/num_C` | 平均每 box C 候选数，替代旧 `num_candidate_voxels`。 |
| `num_P` | `val_pc/num_P` | 平均每 box P anchor 数，替代旧 `num_anchor_points`。 |
| `sparse_refine_best_f1` | `val_score/ligand_sparse_refine_best_f1_foreground` | C 内 refined logits 的 best-F1。 |

按 folder 分组时，追加 `by_folder/{folder}` 路径，例如：

- `val_pc/by_folder/metal_ion/candidate_recall_foreground`
- `val_pc/by_folder/random_BOX/num_C`
- `val_pc/by_folder/random_BOX2/sampling_threshold_fp_foreground`

PR curve/table：

- 使用 histogram 的 threshold grid、TP、FP、FN、precision、recall、F1 生成 rows。
- W&B table/plot 每次 validation 都可记录，约 30 次 validation 的体量可接受。
- 本地文件建议写到 `trainer.default_root_dir/diagnostics/validation_epoch_{epoch:04d}_step_{global_step}.json` 或当前 run dir 下同名子目录。
- 不保存 voxel 坐标、voxel 概率明细或 per-box 明细。

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_logging.py`

新增日志命名与写入 helper：

```python
def log_loss_dict(owner: Any, prefix: str, total_loss: torch.Tensor, loss_dict: dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool) -> None:
    ...
```

日志映射：

| `loss_dict` key | train 日志 | val 日志 |
|---|---|---|
| `total_loss` | `train_loss/loss` | `val_loss/loss` |
| `atom_loss` | `train_loss/atom_loss` | `val_loss/atom_loss` |
| `voxel_aux_loss` | `train_loss/receptor_loss` | `val_loss/receptor_loss` |
| `voxel_ligand_loss` | `train_loss/voxel_ligand_loss` | `val_loss/voxel_ligand_loss` |
| `ligand_sparse_refine_loss` | `train_loss/ligand_sparse_refine_loss` | `val_loss/ligand_sparse_refine_loss` |
| `ligand_sparse_refine_loss_weight_effective` | `train_loss/ligand_sparse_refine_loss_weight_effective` | `val_loss/ligand_sparse_refine_loss_weight_effective` |

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_scheduler.py`

可选拆分 scheduler 逻辑。如果实施时想控制改动范围，可以暂时不拆 scheduler；若拆分，迁移：

- `_resolve_warmup_steps`
- `_build_warmup_only_scheduler`
- `_build_warmup_plateau_scheduler`
- `_sync_metric_for_scheduler`
- `_step_warmup_plateau_scheduler`

主 wrapper 保留 `configure_optimizers()` 入口并委托 helper。

### 配置变更

#### [MODIFY] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\configs\model\default.yaml`

新增默认 diagnostics 配置：

```yaml
monitor_metric: val_score/atom_pr_auc

validation_diagnostics:
  enabled: true
  log_pr_curve: true
  write_local_tables: true
  group_by_meta_field: class_name
  group_names: ${dataset.class_folder_names}
  output_subdir: validation_diagnostics
```

字段说明：

| 字段 | 类型 | 默认值 | 消费位置 |
|---|---|---|---|
| `validation_diagnostics.enabled` | `bool` | `true` | wrapper 初始化 diagnostics helper；关闭时只保留基础 AP/loss。 |
| `validation_diagnostics.log_pr_curve` | `bool` | `true` | validation end 是否写 W&B PR table/plot。 |
| `validation_diagnostics.write_local_tables` | `bool` | `true` | validation end 是否写本地 CSV/JSON。 |
| `validation_diagnostics.group_by_meta_field` | `str` | `class_name` | 从 batch 中读取分组元信息。 |
| `validation_diagnostics.group_names` | `list[str]` | `${dataset.class_folder_names}` | 固定 folder 分组顺序，支持未来新增 `random_BOX2`。 |
| `validation_diagnostics.output_subdir` | `str` | `validation_diagnostics` | 本地诊断文件子目录。 |

#### [MODIFY] 当前仍在使用的实验配置

需要同步当前非 old 的实验配置：

| 文件模式 | 旧值 | 新值 |
|---|---|---|
| `configs/experiment/unet*.yaml`, `configs/experiment/emb_unet.yaml` | `val/voxel_ligand_pr_auc` | `val_score/voxel_ligand_pr_auc` |
| `configs/experiment/sparse_refine*.yaml`, `configs/experiment/MINI_sparse_refine*.yaml` | `val/ligand_sparse_refine_best_f1_macro` | 二分类用 `val_score/ligand_sparse_refine_best_f1_foreground`；多分类才用 `val_score/ligand_sparse_refine_best_f1_macro` |
| UNet-only 配置 | `val/loss` | `val_loss/loss` |

`configs/experiment/old/**` 可不批量修改，除非用户明确要重跑旧配置。计划实现时应至少用 `rg -n "monitor_metric: val/" configs/experiment configs/model` 列出剩余项，确认是否都是历史配置。

### 测试变更

#### [MODIFY] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_voxel_ligand_thresholds.py`

- 更新所有旧日志名断言：
  - `val/voxel_ligand_best_f1_before_refine_macro` 改为多分类才出现的 `val_score/voxel_ligand_best_f1_before_refine_macro`。
  - 二分类新增断言不出现 macro。
- 新增 `test_sampling_threshold_counts_are_logged`：
  - 构造小 histogram。
  - 调用 threshold diagnostics compute。
  - 断言 `sampling_threshold_tp/fp/fn/total` 与手算一致。
- 更新 `test_sampling_threshold_saturates_when_expand_exceeds_all_voxels`：
  - 除 `p_sampling=0` 外，断言 `sampling_threshold_total` 等于全部有效 voxel 数。

#### [MODIFY] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_ligand_sparse_refine_metrics.py`

- 旧 `val/candidate_recall_foreground` 改为 `val_pc/candidate_recall_foreground`。
- 旧 `val/num_candidate_voxels` 改为 `val_pc/num_C`。
- 旧 `val/num_anchor_points` 改为 `val_pc/num_P`。
- sparse refine best-F1 改为 `val_score/ligand_sparse_refine_best_f1_*`。
- 新增按 folder 分组测试：
  - batch 中加入 `class_name=["metal_ion", "random_BOX"]`。
  - 构造两个 box 的 C/P 输出。
  - 断言 `val_pc/by_folder/metal_ion/num_C` 与 `val_pc/by_folder/random_BOX/num_C` 分别正确。

#### [MODIFY] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_multiclass_ligand_wrapper.py`

- 更新 `_ligand_target_from_dist` 调用方式；如果方法迁移到 helper，保留 wrapper 静态转发或修改测试 import。
- 保留多分类 target 行为断言。

#### [NEW] `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_voxel_point_stage1_logging_names.py`

新增日志命名测试：

- 二分类 receptor AP 记录为 `val_score/receptor_pr_auc`，不记录 `val_score/receptor_macro_ap`。
- loss key `voxel_aux_loss` 写出日志名 `train_loss/receptor_loss` / `val_loss/receptor_loss`。
- `monitor_metric` 使用新名时 `prog_bar` 判断仍可命中。

## 不修改的部分

- 不改 `src/model/stage1_voxel_backbone.py` 中的 `voxel_aux_head`、`voxel_aux_logit_dim`、`voxel_logits_aux`。
- 不改 `src/inference/get_pred.py` 的 `"receptor": "voxel_logits_aux"` 映射。
- 不实现 box 级或 voxel 级诊断明细，不保存候选坐标、每个 voxel 概率、每个 voxel 是否被 C/P 命中的表。
- 不保留旧 `val/*` 日志别名。
- 不批量改 `configs/experiment/old/**`，除非用户后续明确要求旧实验配置也可直接重跑。

## 改动文件汇总

| 文件 | 改动内容 |
|---|---|
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` | 缩减为 LightningModule 生命周期编排入口。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_losses.py` | 新增 loss helper。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_metrics.py` | 新增 AP metric manager。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_diagnostics.py` | 新增 dense threshold 与 C/P diagnostics。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_logging.py` | 新增日志命名与 loss logging helper。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1_scheduler.py` | 可选新增 scheduler helper。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\configs\model\default.yaml` | 更新默认 `monitor_metric`，新增 diagnostics 配置。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\configs\experiment\*.yaml` | 同步当前实验配置的 `monitor_metric` 新日志名。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_voxel_ligand_thresholds.py` | 更新 threshold 和 sampling counts 测试。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_ligand_sparse_refine_metrics.py` | 更新 C/P 指标名与 folder 分组测试。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_multiclass_ligand_wrapper.py` | 跟随 helper 迁移调整 target 测试。 |
| `C:\Users\15919\OneDrive\My_Project\Pocket_Plus\tests\test_voxel_point_stage1_logging_names.py` | 新增日志命名测试。 |

## Verification Plan

### Automated Tests

```bash
pytest tests/test_voxel_ligand_thresholds.py -v
```

验证点：
1. p_best/p_sampling cache 仍可更新、保存和恢复。
2. 新增 `sampling_threshold_tp/fp/fn/total` 与手算一致。
3. 二分类不再生成 dense best-F1 macro 指标。

```bash
pytest tests/test_ligand_sparse_refine_metrics.py -v
```

验证点：
1. `candidate_recall` 仍等于 C 覆盖正例数 / dense GT 正例数。
2. `num_C` / `num_P` 仍是每 box 平均数。
3. folder 分组统计按 `batch["class_name"]` 正确拆分。

```bash
pytest tests/test_multiclass_ligand_wrapper.py tests/test_multiclass_ligand_loss.py -v
```

验证点：
1. 多分类 ligand target 与 loss 行为未因 helper 拆分改变。
2. 多分类仍记录逐类与 macro 指标。

```bash
pytest tests/inference/test_get_voxel_pred.py -v
```

验证点：
1. 推理端 `"receptor"` 仍读取 `voxel_logits_aux`。
2. 不改 backbone key 后，receptor 输出逻辑保持兼容。

```bash
python -m compileall src/wrappers
```

验证点：
1. 新拆出的 helper 模块无语法错误。
2. import 关系没有循环导入。

### Manual Verification

1. 用一个小 batch 或已有 smoke 配置跑一次短 validation。
2. 检查 W&B 或 Lightning logs：
   - 只出现 `val_loss/*`、`val_score/*`、`val_pc/*` 新日志名。
   - 不出现旧 `val/voxel_aux_pr_auc`、`val/voxel_aux_loss`、二分类 `*_macro`。
   - receptor 相关日志名为 `receptor`，不是 `voxel_aux`。
3. 检查本地 diagnostics 输出：
   - 每次 validation 生成一份 PR curve/table JSON 或 CSV。
   - 文件包含 threshold、tp、fp、fn、precision、recall、f1。
   - 按 `metal_ion`、`peptide`、`nucleic`、`small_molecule`、`random_BOX` 分组的统计存在；未来加入 `random_BOX2` 后能自动按配置出现。
4. 用一个旧 checkpoint 跑 `src/inference/get_pred.py` 相关推理入口：
   - `load_state_dict(strict=True)` 不因本次 wrapper 日志改名失败。
   - 请求 `output_heads: ["ligand", "receptor"]` 时 receptor 仍可输出。

## 实施顺序建议

1. 先新增 `voxel_point_stage1_logging.py` 并只替换 loss 日志名，跑 logging names 测试。
2. 拆 `voxel_point_stage1_metrics.py`，完成二分类 macro 删除和 receptor AP 改名，跑 AP 相关测试。
3. 拆 `voxel_point_stage1_diagnostics.py`，先迁移原有 threshold/C/P 统计，再新增 sampling threshold TP/FP/FN、PR curve/table 和 folder 分组。
4. 拆 `voxel_point_stage1_losses.py`，保持 loss 数值不变。
5. 最后视改动规模决定是否拆 scheduler；如果测试压力较大，可把 scheduler 留在主 wrapper，后续单独重构。
6. 更新当前实验配置的 `monitor_metric`，再执行完整相关测试。
