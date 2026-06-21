# VoxelPointStage1 Wrapper 重构、CPC 诊断与 TopK 采样实施计划 v4

## Grill 决策修正记录（2026-05-29）

以下决策优先于本文后续原始描述中的冲突处：

1. 保留现有字段名 `warmup_topc_per_class`，不重命名为 `warmup_topk_per_class`；正式 `selection_mode=topk` 不新增 `topk_per_class`，直接使用 `max_candidate_voxels_per_class` 作为每 BOX/类别 top-k 请求数。
2. 新建干净的 `src/wrappers/voxel_point_stage1.py` 作为 thin coordinator，`src/wrappers/voxel_point_stage1_old.py` 仅作为迁移参考和行为对照。
3. 内部模型字段、配置参数和 loss 参数继续保留 `voxel_aux_*`；对外 metric、artifact、W&B key 和文档统一使用 `receptor`。
4. `random_BOX` 固定作为独立 `by_source_folder` diagnostics 组输出；单个 BOX 可无正类，但整个 `random_BOX` 验证集预期应有正类。
5. 正式 fit validation 中，若所有 candidate class 在整个 epoch 均无 GT 正例，可以 fail-fast；sanity check、tuner 或 smoke-only validation 不应因此中断。
6. 多分类候选的 `val_uncapped/sampling` 保留 per-class 提名阶段语义，`val_capped` 保留 unique+routed 后实际 C 语义，不强行令两者计数一致。
7. 二分类 sparse-refine active 配置的默认 `monitor_metric` 使用 `val_score/global/refined_F1`，不再使用 macro 指标。
8. 实施采用分阶段并行：每阶段可由多个 subagent 实现独立切片，但阶段之间必须安排 validator/transition subagent 检查已存在代码、接口漂移、bug 与计划一致性。
9. 本次验收保留 DDP 功能与本地 CPU/Gloo 多进程逻辑模拟；不依赖服务器双卡 DDP smoke 作为完成前置。

## 背景与目标

`src/wrappers/voxel_point_stage1.py` 当前同时承担 Lightning 生命周期、loss 计算、TorchMetrics 管理、候选 C 阈值缓存、sparse refine 诊断、W\&B 记录、optimizer/scheduler 构造等职责。文件规模已接近 2000 行，初学者很难按“模型实际运行时间顺序”理解 Stage1 的 C -> P -> C 流程，也很难在训练异常时判断问题发生在 dense 可分性、采样、cap、C 内局部判别还是 refine 分支。

本计划的目标是把 wrapper 大幅重构为分层结构，同时建立一套更清晰的验证诊断与日志命名体系。重构后，`VoxelPointStage1Wrapper` 仍是唯一对外 LightningModule/Hydra 入口，但大块逻辑会拆入同目录 helper 模块。验证日志会分为 loss、score、curve、四阶段 CPC 诊断与本地 artifact，二分类不再输出重复 macro 指标，对外日志统一使用 `receptor` 而不是 `aux`。

本计划还新增正式候选采样模式：

```text
selection_mode: topk
```

`topk` 不再只是 warmup 阶段的临时 fixed topk，而是可以从训练开始到结束都使用的正式 C 生成策略。正式 topk 模式不新增独立 `topk_per_class`，直接使用 `max_candidate_voxels_per_class` 作为每 BOX/类别 top-k 请求数，不从 `warmup_topc_per_class` fallback。

## 当前架构 / 已知约束

### Wrapper 现状

`src/wrappers/voxel_point_stage1.py` 当前包含以下关键逻辑：

| 逻辑               | 当前符号                                                                                                     | 说明                                                     |
| ---------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 类别名解析            | `VoxelPointStage1Wrapper._resolve_class_names()`                                                         | 当前从 `kwargs.get("class_names", None)` 读取，未配置时回退二分类默认名。 |
| 常规 AP 指标         | `_register_val_metric()`、`_init_multiclass_ap_metrics()`、`_update_val_metric()`、`_compute_log_reset_*()` | TorchMetrics 直接挂在 wrapper 上，含 AP 设备策略和 macro 输出。       |
| dense best-F1 阈值 | `_update_voxel_ligand_best_f1_stats()`、`_compute_log_update_voxel_ligand_best_f1_thresholds()`           | 维护 dense histogram，更新 `p_best` / `p_sampling` cache。   |
| sparse refine 诊断 | `_update_val_ligand_sparse_refine_metric()`、`_compute_log_reset_ligand_sparse_refine_metrics()`          | 当前 `fn` 使用 dense 全图 GT 正例数，语义是端到端 refined F1。          |
| 验证主流程            | `validation_step()`、`on_validation_epoch_start()`、`on_validation_epoch_end()`                            | 当前 step 内混合 loss、metrics、diagnostics、日志记录。             |
| 调度器              | `configure_optimizers()`、`_step_warmup_plateau_scheduler()`                                              | 与 `monitor_metric` 强耦合。                                |

### Candidate builder 现状

`src/model/sparse_refine/candidate_set.py` 中 `SparseCandidateSetBuilder` 当前只允许：

```text
selection_mode: adaptive_threshold
selection_mode: recorded_threshold
```

warmup fixed topk 由 `forward(..., use_fixed_warmup=True)` 临时覆盖正式模式。当前输出中已有可复用字段：

| 字段                                 | 形状                 | 可复用意义                                                                |
| ---------------------------------- | ------------------ | -------------------------------------------------------------------- |
| `candidate_logits`                 | `(sumC, C_logits)` | C 位置的原始 dense ligand logits，已 detach，可作为 `val_unrefined` 的 logit 来源。 |
| `candidate_prob`                   | `(sumC,)`          | 实际进入唯一 C 后的候选概率，可用于 `p_C_*`。                                         |
| `candidate_counts`                 | `(B,)`             | 每个 BOX 的实际唯一 C 数。                                                    |
| `candidate_counts_by_class`        | `(B, K)`           | 每个 BOX/候选类的实际唯一 C 数。                                                 |
| `candidate_p_sampling_by_class`    | `(B, K)`           | 每个 BOX/候选类实际采样边界概率。warmup/adaptive 为 topk cutoff，recorded 为全局阈值复制。   |
| `candidate_target_counts_by_class` | `(B, K)`           | 每个 BOX/候选类在 cap 前的目标或命中数量。                                           |

### 数据集分组约束

`src/datasets/box_point_dataset.py` 会写入 `sample_dict["class_name"]`，collate 后 `batch["class_name"]` 是 `list[str]`。该字段表示原始数据源文件夹，例如：

```text
metal_ion
peptide
nucleic
small_molecule
random_BOX
```

它不是经过 `class_mapping` 后的 task class。source folder 分组必须使用 `by_source_folder` 命名，task class 只作为 metric leaf suffix。

### 推理兼容边界

对外日志可彻底改名，不保留旧 W\&B metric alias。但不得改变旧 checkpoint 推理所依赖的模型权重 key。内部模型字段和输出仍可保留：

```text
voxel_logits_aux
voxel_aux_head
voxel_aux_logit_dim
```

对外日志、文档和配置说明统一称为 `receptor`。

### Metric 设备策略约束

`docs/验证时的metric计算问题.md` 中已经确立：

* `thresholds=None` 的非 binned AP 倾向 CPU，以避免保存大量 prediction/target 状态到显存。
* 固定阈值的 binned AP 倾向 GPU，因为状态小、阈值扫描快。
* batch size tuner 不应跳过验证 AP，否则会掩盖真实验证链路问题。
* 必须继续支持 `val_metric_device_policy: auto | cpu | gpu`。

本次重构必须保留这些策略，但不得在 `validation_step` 里用手动 `.to("cpu")` 造成同步阻塞。应优先依赖 TorchMetrics 的状态/`compute_on_cpu` 策略，并通过测试确认非 binned AP 不污染显存。

## 已有可复用代码

| 已有代码                                                       | 位置                                         | 可复用能力                                                                                                                   |
| ---------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `SparseCandidateSetBuilder.forward()`                      | `src/model/sparse_refine/candidate_set.py` | 已实现 per BOX/per class topk、threshold selection、cap、unique C 路由、`candidate_logits` 与 `candidate_p_sampling_by_class` 输出。 |
| `VolumePointStage1Model._prepare_pseudo_batch()`           | `src/model/stage1_model.py`                | 统一调用 candidate builder，并把 C/P 输出并入 backbone outputs。                                                                    |
| `VolumePointStage1Model.set_sparse_candidate_thresholds()` | `src/model/stage1_model.py`                | 接收 wrapper 写回的 `p_best_by_class` / `p_sampling_by_class` runtime cache。                                                 |
| `VolumePointStage1Model.set_sparse_candidate_runtime()`    | `src/model/stage1_model.py`                | 接收 `global_step`、`candidate_warmup_steps`、`allow_warmup_fixed_topk`，控制 warmup fixed topk。                               |
| `_ligand_target_from_dist()`                               | `src/wrappers/voxel_point_stage1.py`       | 从 ligand distance map 生成 hard target，可复用为 dense 与 C 内诊断的 target 来源。                                                     |
| `_sample_ligand_refine_supervision()`                      | `src/wrappers/voxel_point_stage1.py`       | 已能生成 C 级监督与 dense valid/target，可作为 `val_refined` 与端到端 score 的监督基础。                                                      |
| 当前 histogram buffer                                        | `src/wrappers/voxel_point_stage1.py`       | 已使用 `persistent=False` 注册 dense/refine histogram，可迁移为 diagnostics 模块的 buffer 策略。                                        |
| `tests/test_voxel_ligand_thresholds.py`                    | `tests/`                                   | 可扩展覆盖 `p_best`、`p_sampling`、warmup cache 更新、无正例处理。                                                                      |
| `tests/test_ligand_sparse_refine_metrics.py`               | `tests/`                                   | 可扩展覆盖 local/e2e refined F1 定义差异。                                                                                        |
| `tests/test_multiclass_ligand_wrapper.py`                  | `tests/`                                   | 可扩展覆盖多分类 suffix 与 macro 规则。                                                                                             |
| `tests/inference/test_get_voxel_pred.py`                   | `tests/inference/`                         | 可扩展保护旧 checkpoint 推理键名。                                                                                                 |

## 设计决策

### 1. Wrapper 仍是唯一对外入口

`VoxelPointStage1Wrapper` 的 import path 与 Hydra `_target_` 不变。外部训练脚本、checkpoint 加载、Lightning Trainer 不需要知道 helper 模块的存在。

> \[!IMPORTANT]
> 新 helper 可以拆很多，但 wrapper 本体必须保留训练/验证时间顺序，让读者能从 `validation_step()` 看见 dense -> C -> P -> refined 的主链路。

### 2. TorchMetrics manager 必须是 `nn.Module`

常规 validation metrics 会迁入 `ValidationMetricManager(nn.Module)`。所有 TorchMetrics 对象必须在模块树里可见，不能藏在普通 Python 类内部。

同时，validation metrics 不应污染 checkpoint。方案为：

* metric manager 是 `nn.Module`，保证 `.to(device)` 与 DDP 安全。
* metric 状态在 `on_validation_epoch_end` compute 后立即 reset。
* 实现 `on_save_checkpoint` 过滤 validation metric/diagnostics 状态，或在 manager 内使用不会持久化中间 state 的注册策略。
* 测试必须断言 checkpoint state\_dict 不包含大体积 validation prediction/target 状态。

### 3. DDP 聚合必须所有 rank 对称执行

所有 histogram、count、curve buffer 的跨卡同步必须满足：

1. 所有 rank 无条件进入 collective。
2. collective 的 tensor 形状固定。
3. 先做 `all_reduce(sum)`，再判断全局是否有正例、是否生成 warning、是否写 `nan`。
4. 不允许某 rank 因为本地无正例跳过同步。

> \[!WARNING]
> 无正例是数据统计状态，不是结构错误。未知 source folder、shape 不一致、缺少必须字段才是结构错误。

### 4. 持久化与 W\&B 只在 global zero 执行

DDP 下非 0 rank 只参与 batch update 和固定 tensor 聚合，不写文件，不上传 W\&B table，不 print 大段诊断。

必须使用：

```python
trainer.is_global_zero
```

保护：

* `write_validation_artifacts(...)`
* `log_wandb_curves(...)`
* W\&B table/media 上传
* 本地 `summary.json`、`warnings.json`、`curves/*.csv` 写入

Lightning scalar `self.log(..., sync_dist=True)` 可由所有 rank 调用，因为它是 Lightning 的同步日志路径。

### 5. `val_score` 保存端到端结果，四阶段面板保存诊断结果

`val_unrefined/*` 与 `val_refined/*` 保持 local 语义：只看实际 C 内，`fn` 只统计 C 内漏判。

端到端结果放入 `val_score`：

```text
val_score/global/unrefined_F1
val_score/global/refined_F1
val_score/by_source_folder/<folder>/unrefined_F1
val_score/by_source_folder/<folder>/refined_F1
```

其中端到端 `fn` 包含没有进入 C 的 dense GT 正例。这样 `val_score/refined_F1` 是最终链路分数，`val_refined/F1` 是 C 内 refined logit 局部判别能力。

### 6. 四阶段 CPC 面板定义

四个主面板：

| 面板                      | 统计空间                                      | 实际意义                      |
| ----------------------- | ----------------------------------------- | ------------------------- |
| `val_uncapped/best`     | global dense                              | dense 分支理论最佳阈值与上限质量。      |
| `val_uncapped/sampling` | global dense，按真实 per BOX/per class 采样边界聚合 | 当前采样策略想怎样切 C。             |
| `val_capped`            | global dense 中的实际 C                       | cap 后真正进入 C 的候选是否覆盖 GT。   |
| `val_unrefined`         | local C                                   | C 内用 dense logit 判别的质量。   |
| `val_refined`           | local C                                   | C 内用 refined logit 判别的质量。 |

`val_uncapped` 必须分 `best` 与 `sampling`，因为 `p_best` 与 `p_sampling` 是两个不同的系统。

### 7. `val_capped` 概率摘要改为 `p_C_*`

`val_uncapped/sampling` 记录采样边界概率：

```text
p_sampling_p5
p_sampling_p50
p_sampling_p75
p_sampling_p95
p_sampling_mean
```

`val_capped` 记录最终实际 C 的概率分布，不再复用 `p_sampling_*`，改为：

```text
p_C_p5
p_C_p50
p_C_p75
p_C_p95
p_C_mean
```

这样可以区分“采样边界”与“实际进入 C 的候选质量”。

### 8. Warmup 阶段允许更新候选阈值缓存

warmup fixed topk 阶段会生成 C，但不会读取 `p_best_by_class` / `p_sampling_by_class`。当前设计意图是 warmup 期间通过 validation 积累 dense histogram，写出可供 warmup 结束后正式采样使用的阈值缓存。

新版规则：

* fit warmup 阶段：尽可能完整记录诊断，并允许更新 `p_best_by_class` / `p_sampling_by_class` runtime cache。
* sanity check：可以做 smoke 统计，但不得写回正式 cache，不得影响 scheduler/monitor。
* batch size tuner：继续计算常规 AP 以暴露链路问题，但不得写回 candidate threshold cache，不得影响 scheduler/monitor。

> \[!IMPORTANT]
> warmup validation 写 cache 不是污染，而是候选采样启动设计的一部分。sanity/tuner 写 cache 才是污染。

### 9. 三种正式 selection mode

`SparseCandidateSetBuilder.selection_mode` 扩展为：

| 模式                   | 是否读取 cache               | C 生成方式                                                                               | `candidate_p_sampling_by_class` 含义    |
| -------------------- | ------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------- |
| `adaptive_threshold` | 读取 `p_best_by_class`     | 每 BOX 统计 `prob > p_best` 的数量，再乘 `adaptive_expand_factor` 得到 target count，按 topk 取 C。 | 每 BOX/类实际 topk cutoff probability。    |
| `recorded_threshold` | 读取 `p_sampling_by_class` | 每 BOX 直接按全局 `p_sampling` 阈值筛选，超过 cap 时在命中集合内 topk。                                   | 每 BOX/类复制使用的全局 `p_sampling`。          |
| `topk`               | 不读取 cache                | 每 BOX/类按 `max_candidate_voxels_per_class` 直接 topk。            | 每 BOX/类实际第 K 个候选的 cutoff probability。 |

正式 `topk` 模式不新增 `topk_per_class`，不允许从 `warmup_topc_per_class` fallback。

### 10. source folder unknown fail-fast

`batch["class_name"]` 中出现不在配置 `source_folder_names` 内的值时，立即 fail-fast。错误信息必须包含：

* unknown name
* allowed source folder names
* 当前 batch 中的位置 index

该检查发生在 validation step 编码阶段，早于任何 diagnostics collective。

## Proposed Changes

### A. 候选采样模式扩展

#### \[MODIFY] `src/model/sparse_refine/candidate_set.py`

**`SparseCandidateSetBuilder.__init__`**

修改签名：

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
```

| 参数                                                                                                   | 类型                | 意义                                                   |
| ---------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------- |
| `candidate_class_ids`                                                                                | `Sequence[int]`   | 候选前景 task class id，长度为 K。                            |
| :comment[warmup\_topc\_per\_class]{#comment-1780045414447 text="这个名字是否应该改成 warmup_topk_per_class ?"} | `Sequence[int]`   | warmup fixed topk 阶段每 BOX/类候选数。仅 warmup 覆盖正式模式时使用。   |
| `adaptive_expand_factor`                                                                             | `Sequence[float]` | `adaptive_threshold` 中相对 `p_best` 命中数的扩张倍数。          |
| `max_candidate_voxels_per_class`                                                                     | `Sequence[int]`   | 每 BOX/类实际 C 行数硬上限；正式 `topk` 模式也作为每 BOX/类 top-k 请求数。 |
| `selection_mode`                                                                                     | `str`             | 允许 `adaptive_threshold`、`recorded_threshold`、`topk`。 |

验证规则：

1. `selection_mode` 必须属于 `{"adaptive_threshold", "recorded_threshold", "topk"}`。
2. `candidate_class_ids`、`warmup_topc_per_class`、`adaptive_expand_factor`、`max_candidate_voxels_per_class` 长度必须一致。
3. `selection_mode == "topk"` 时，使用 `max_candidate_voxels_per_class[class_pos]` 作为 top-k 请求数。
4. 不允许用 `warmup_topc_per_class` 作为正式 topk 的 fallback。

不新增属性：

```python
# 不新增 self.topk_per_class
```

**`SparseCandidateSetBuilder.forward`**

保留签名：

```python
def forward(
    self,
    voxel_logits_ligand: torch.Tensor,
    voxel_valid_mask: torch.Tensor,
    p_best_by_class: torch.Tensor | None,
    p_sampling_by_class: torch.Tensor | None,
    use_fixed_warmup: bool,
) -> dict[str, torch.Tensor]: ...
```

修改 per BOX/per class 选择分支：

1. `use_fixed_warmup=True`：继续使用 `warmup_topc_per_class[class_pos]`。
2. `selection_mode == "adaptive_threshold"`：继续使用 `p_best_by_class` 与 `adaptive_expand_factor`。
3. `selection_mode == "recorded_threshold"`：继续使用 `p_sampling_by_class`。
4. `selection_mode == "topk"`：使用 `max_candidate_voxels_per_class[class_pos]`。

 topk 分支伪代码：

```python
elif self.selection_mode == "topk":
    target_before_cap = int(self.max_candidate_voxels_per_class[class_pos])
    target_count = min(target_before_cap, int(prob_valid.numel()))
    selected_order = torch.topk(prob_valid, k=target_count).indices if target_count > 0 else empty_long
    output["candidate_target_counts_by_class"][batch_idx, class_pos] = target_before_cap
```

 topk 分支也要写：

```python
output["candidate_p_sampling_by_class"][batch_idx, class_pos] = cutoff_prob
```

其中 `cutoff_prob = selected_prob.min()`，空选择时保留 `nan`。

#### \[NEW] `configs/model/sparse_refine/candidate_set/binary_topk.yaml`

新增配置：

```yaml
# @package _global_

model:
  backbone:
    candidate_set_cfg:
      _target_: src.model.sparse_refine.candidate_set.SparseCandidateSetBuilder
      selection_mode: topk
      candidate_class_ids: ${sparse_refine_task.candidate_class_ids}
      warmup_topc_per_class: ${sparse_refine_task.warmup_topc_per_class}
      adaptive_expand_factor: ${sparse_refine_task.adaptive_expand_factor}
      max_candidate_voxels_per_class: ${sparse_refine_task.max_candidate_voxels_per_class}
```

#### \[NEW] `configs/model/sparse_refine/candidate_set/tri_topk.yaml`

同 `binary_topk.yaml`，但用于三分类/多分类 task 覆盖。

#### \[MODIFY] `configs/model/task/*.yaml`

不新增 `topk_per_class`。正式 topk 复用已有 `max_candidate_voxels_per_class` 作为每 BOX/类 top-k 请求数。

### :comment[B. Wrapper 构造参数与配置边界&#xA;\[MODIFY\] src/wrappers/voxel\_point\_stage1.py&#xA;VoxelPointStage1Wrapper.\_\_init\_\_]{#comment-1780044964699 text="我已经把 src/wrappers/voxel_point_stage1.py 重命名为了 src/wrappers/voxel_point_stage1_old.py。你可以在 src/wrappers/voxel_point_stage1_old.py 的基础上做修改，也可以参考 old 版本，重新写一个 src/wrappers/voxel_point_stage1.py"}

新增显式参数：

```python
def __init__(
    ...,
    class_names: Sequence[str],
    validation_diagnostics: Mapping[str, Any] | None = None,
    ...
) -> None: ...
```

| 参数                       | 类型                   | 意义                                                   |                                                                |
| ------------------------ | -------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| `class_names`            | `Sequence[str]`      | task class 名，顺序必须与 loss/logit class id 对齐。必须由配置显式传入。 |                                                                |
| `validation_diagnostics` | \`Mapping\[str, Any] | None\`                                               | CPC 诊断与 artifact 配置。`None` 表示使用默认启用策略或由 helper 解析默认 dataclass。 |

移除或废弃：

```python
_resolve_class_names(kwargs: dict[str, Any]) -> list[str]
```

替代规则：

1. `class_names` 必须显式传入。
2. 若 `class_names` 长度与多分类 loss `num_classes` 不一致，fail-fast。
3. 二分类也必须传 `[background, foreground]`，不再隐式回退。

#### \[MODIFY] `configs/model/default.yaml`

新增：

```yaml
class_names: ${dataset.class_names}

validation_diagnostics:
  enabled: true
  source_folder_breakdown: true
  source_folder_names: ${dataset.class_folder_names}
  source_folder_meta_key: class_name
  num_bins: ${model.voxel_ligand_pr_auc_thresholds}
  write_local_artifacts: true
  log_wandb_curves: true
  wandb_curve_every_n_validation: 1
  output_subdir: validation_diagnostics
```

保留：

```yaml
val_metric_device_policy: auto
```

配置字段表：

| 字段                               | 类型          | 默认值                                       | 允许值/约束              | 消费位置                                   |
| -------------------------------- | ----------- | ----------------------------------------- | ------------------- | -------------------------------------- |
| `validation_diagnostics.enabled` | `bool`      | `true`                                    | `true/false`        | wrapper 构造 diagnostics manager。        |
| `source_folder_breakdown`        | `bool`      | `true`                                    | `true/false`        | 是否生成 `by_source_folder` 指标。            |
| `source_folder_names`            | `list[str]` | `${dataset.class_folder_names}`           | 固定 source folder 列表 | `SourceFolderRegistry`。                |
| `source_folder_meta_key`         | `str`       | `class_name`                              | batch metadata key  | `SourceFolderRegistry.encode_batch()`。 |
| `num_bins`                       | `int`       | `${model.voxel_ligand_pr_auc_thresholds}` | `>0`                | histogram 与 PRcurve grid。              |
| `write_local_artifacts`          | `bool`      | `true`                                    | `true/false`        | rank0 写本地 JSON/CSV。                    |
| `log_wandb_curves`               | `bool`      | `true`                                    | `true/false`        | rank0 上传 W\&B table。                   |
| `wandb_curve_every_n_validation` | `int`       | `1`                                       | `>=1`               | 控制 curve 上传频率。                         |
| `output_subdir`                  | `str`       | `validation_diagnostics`                  | 非空字符串               | run dir 下 artifact 子目录。                |

### C. 拆分 loss 逻辑

#### \[NEW] `src/wrappers/voxel_point_stage1_losses.py`

新增模块职责：只计算 loss term，不拼 W\&B key，不直接调用 logger。

新增 dataclass：

```python
@dataclass(frozen=True)
class LossTerm:
    name: str
    value: torch.Tensor
    weight: float
    logged_value: torch.Tensor
```

| 字段             | 类型       | 意义                                                                    |              |
| -------------- | -------- | --------------------------------------------------------------------- | ------------ |
| `name`         | `str`    | 内部 loss 名，使用 `atom`、`receptor`、`voxel_ligand`、`ligand_sparse_refine`。 |              |
| `value`        | `Tensor` | 原始 loss tensor，参与总 loss 加权。                                           |              |
| `weight`       | `float`  | 当前 loss 权重。                                                           |              |
| `logged_value` | `Tensor` | `value.detach()` 后的 tensor，用于 Lightning log，避免 \`float                | Tensor\` 分支。 |

新增函数：

```python
def compute_atom_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm: ...
```

```python
def compute_receptor_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm: ...
```

```python
def compute_voxel_ligand_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: nn.Module,
    weight: float,
) -> LossTerm: ...
```

```python
def compute_sparse_refine_loss_term(
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    loss_module: AdaptiveClassificationCompositeLoss,
    weight: float,
    effective_weight: float,
) -> tuple[LossTerm, torch.Tensor]: ...
```

| 返回                 | 类型        | 意义                                                                                 |
| ------------------ | --------- | ---------------------------------------------------------------------------------- |
| `LossTerm`         | dataclass | 单个 loss 的原始值、权重和日志值。                                                               |
| `effective_weight` | `Tensor`  | sparse refine schedule 后的有效权重，用于 `val_loss/ligand_sparse_refine_weight_effective`。 |

Wrapper 接入：

1. `_compute_total_loss()` 保留在 wrapper 或迁移为 thin coordinator。
2. `training_step()` 与 `validation_step()` 接收 `loss_terms` 后调用 logging helper。
3. 对外日志名使用 `receptor`，不再输出 `voxel_aux_loss`。

### D. 常规 validation metrics manager

#### \[NEW] `src/wrappers/voxel_point_stage1_metrics.py`

新增模块职责：管理常规分数指标，尤其是 AP/PRAUC。该模块不负责 CPC 四阶段 tp/fp/fn。

新增 dataclass：

```python
@dataclass(frozen=True)
class MetricBranchSpec:
    name: str
    enabled: bool
    num_classes: int
    class_names: tuple[str, ...]
    thresholds: int | Sequence[float] | None
```

新增 class：

```python
class ValidationMetricManager(nn.Module):
    def __init__(
        self,
        *,
        branches: Sequence[MetricBranchSpec],
        metric_device_policy: str,
    ) -> None: ...

    def update_branch(
        self,
        *,
        branch_name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        source_folder_idx: torch.Tensor | None = None,
    ) -> None: ...

    def compute_payload(self) -> dict[str, torch.Tensor]: ...

    def reset(self) -> None: ...

    def state_dict(
        self,
        *args: Any,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]: ...
```

| 参数                     | 类型                           | 意义                                 |                                                     |
| ---------------------- | ---------------------------- | ---------------------------------- | --------------------------------------------------- |
| `branches`             | `Sequence[MetricBranchSpec]` | atom/receptor/voxel\_ligand 等分支定义。 |                                                     |
| `metric_device_policy` | `str`                        | `auto/cpu/gpu`，沿用现有配置。             |                                                     |
| `branch_name`          | `str`                        | 当前更新的分支名，决定 metric set。            |                                                     |
| `logits`               | `Tensor`                     | 当前分支 logits。                       |                                                     |
| `target`               | `Tensor`                     | 当前分支 hard target。                  |                                                     |
| `mask`                 | `Tensor`                     | 参与统计的位置。                           |                                                     |
| `source_folder_idx`    | \`Tensor                     | None\`                             | 预留给 by\_source\_folder score；若第一阶段不对 AP 分组，可为 None。 |

实现要求：

1. `ValidationMetricManager` 必须继承 `nn.Module`。
2. TorchMetrics 存入 `nn.ModuleDict`。
3. 二分类只输出无 suffix 的 `PRAUC`，不输出 macro。
4. 多分类输出逐前景 class AP/PRAUC，必要时输出 macro；macro 仅在 `num_classes > 2` 时存在。
5. 避免 `validation_step` 手动 `.to("cpu")` 造成同步阻塞。
6. `compute_payload()` 返回新 key：

```text
val_score/global/atom_PRAUC
val_score/global/receptor_PRAUC
val_score/global/voxel_ligand_PRAUC
```

1. metric 状态不得进入 checkpoint。`state_dict()` override 或 wrapper `on_save_checkpoint()` 必须过滤 validation metric 中间状态。

### E. CPC diagnostics manager

#### \[NEW] `src/wrappers/voxel_point_stage1_diagnostics.py`

新增模块职责：维护 C/P/C 四阶段固定形状统计、source folder 分组统计、local 与 e2e F1、曲线表数据。

新增 dataclass：

```python
@dataclass(frozen=True)
class SourceFolderRegistry:
    names: tuple[str, ...]
    name_to_idx: Mapping[str, int]

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "SourceFolderRegistry": ...

    def encode_batch(self, class_names: Sequence[str], *, device: torch.device) -> torch.Tensor: ...
```

`encode_batch()` 契约：

| 参数            | 类型              | 意义                                   |
| ------------- | --------------- | ------------------------------------ |
| `class_names` | `Sequence[str]` | batch metadata 中的原始 source folder 名。 |
| `device`      | `torch.device`  | 输出 index tensor 所在 device。           |

| 返回                  | 类型                   | 意义                                 |
| ------------------- | -------------------- | ---------------------------------- |
| `source_folder_idx` | `Tensor`，`(B,)` long | 每个 BOX 对应的 source folder 静态 index。 |

错误：

* 若遇到未知 source folder，抛 `ValueError`，信息包含 unknown、allowed、batch position。
* 使用 dict lookup，不使用 `list.index()`。

新增 dataclass：

```python
@dataclass(frozen=True)
class CpcDiagnosticsConfig:
    enabled: bool
    source_folder_breakdown: bool
    num_bins: int
    write_local_artifacts: bool
    log_wandb_curves: bool
    wandb_curve_every_n_validation: int
    output_subdir: str
```

新增 payload：

```python
@dataclass(frozen=True)
class CurvePayload:
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
```

```python
@dataclass(frozen=True)
class DiagnosticsWarning:
    code: str
    message: str
    scope: str
    source_folder: str | None
    task_class_name: str | None
```

```python
@dataclass(frozen=True)
class CpcDiagnosticsPayload:
    scalars: dict[str, torch.Tensor]
    curves: dict[str, CurvePayload]
    warnings: tuple[DiagnosticsWarning, ...]
    local_tables: dict[str, CurvePayload]
```

新增 class：

```python
class CpcValidationDiagnostics(nn.Module):
    def __init__(
        self,
        *,
        config: CpcDiagnosticsConfig,
        class_names: Sequence[str],
        candidate_class_ids: Sequence[int],
        source_folders: SourceFolderRegistry,
        adaptive_expand_factor: Sequence[float],
        max_candidate_voxels_per_class: Sequence[int],
    ) -> None: ...

    def reset(self) -> None: ...

    def update_uncapped_best(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        source_folder_idx: torch.Tensor,
        allow_cache_update: bool,
    ) -> None: ...

    def update_uncapped_sampling(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        source_folder_idx: torch.Tensor,
        selection_mode: str,
        use_fixed_warmup: bool,
    ) -> None: ...

    def update_capped(
        self,
        *,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        source_folder_idx: torch.Tensor,
    ) -> None: ...

    def update_unrefined(
        self,
        *,
        candidate_outputs: Mapping[str, torch.Tensor],
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
        source_folder_idx: torch.Tensor,
    ) -> None: ...

    def update_refined(
        self,
        *,
        refined_logits_C: torch.Tensor,
        candidate_outputs: Mapping[str, torch.Tensor],
        target_C: torch.Tensor,
        valid_C: torch.Tensor,
        dense_num_gt: torch.Tensor,
        source_folder_idx: torch.Tensor,
    ) -> None: ...

    def compute_payload(
        self,
        *,
        sync_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> CpcDiagnosticsPayload: ...
```

Buffer 要求：

* 所有 histogram/count buffer 用 `register_buffer(..., persistent=False)`。
* buffer 形状固定，包含 global 与 source folder 维度。
* DDP 同步由 `compute_payload(sync_fn=...)` 内统一执行，所有 rank 无条件同步。
* 禁止在某个 rank 本地无正例时提前 return。

#### `val_uncapped/best` 计算

指标：

```text
val_uncapped/best/global/p_best
val_uncapped/best/global/best_F1
val_uncapped/best/global/best_precision
val_uncapped/best/global/best_recall
val_uncapped/best/global/best_tp
val_uncapped/best/global/best_fp
val_uncapped/best/global/best_fn
val_uncapped/best/global/num_gt
val_uncapped/best/global/numC_p_best_cutoff
```

source folder：

```text
val_uncapped/best/by_source_folder/<folder>/best_F1
```

多分类时 metric leaf 追加 task class suffix：

```text
best_F1_small_molecule
```

#### `val_uncapped/sampling` 计算

核心规则：`sampling_tp/fp/fn` 不允许用 `p_sampling_p50` 之类分位数反推。必须按真实 per BOX/per class sampling boundary 或 target count 聚合。

指标：

```text
val_uncapped/sampling/global/p_sampling_p5
val_uncapped/sampling/global/p_sampling_p50
val_uncapped/sampling/global/p_sampling_p75
val_uncapped/sampling/global/p_sampling_p95
val_uncapped/sampling/global/p_sampling_mean
val_uncapped/sampling/global/sampling_F1
val_uncapped/sampling/global/sampling_precision
val_uncapped/sampling/global/sampling_recall
val_uncapped/sampling/global/sampling_tp
val_uncapped/sampling/global/sampling_fp
val_uncapped/sampling/global/sampling_fn
val_uncapped/sampling/global/num_gt
val_uncapped/sampling/global/numC_sampling_target
val_uncapped/sampling/global/numC_sampling_cutoff
```

三种模式的 `sampling_tp/fp/fn`：

| 模式                   | 计算方式                                                                             |
| -------------------- | -------------------------------------------------------------------------------- |
| `recorded_threshold` | 每 BOX/类使用实际 `candidate_p_sampling_by_class` 中记录的全局阈值，在该 BOX dense valid 空间切割并累加。 |
| `adaptive_threshold` | 每 BOX/类使用 builder 实际 target count 或 cutoff probability 对应的局部 topk 集合累加，不用全局分位数。  |
| `topk`               | 每 BOX/类使用 `topk_per_class` 实际 topk 集合累加。                                         |
| warmup fixed topk    | 每 BOX/类使用 `warmup_topc_per_class` 实际 topk 集合累加。                                  |

空边界集合：

* `p_sampling_*` 返回 `nan`。
* 不抛错。
* warnings 写入 payload。

#### `val_capped` 计算

指标：

```text
val_capped/global/F1
val_capped/global/precision
val_capped/global/recall
val_capped/global/tp
val_capped/global/fp
val_capped/global/fn
val_capped/global/num_gt
val_capped/global/num_C
val_capped/global/num_P
val_capped/global/p_C_p5
val_capped/global/p_C_p50
val_capped/global/p_C_p75
val_capped/global/p_C_p95
val_capped/global/p_C_mean
```

定义：

* `tp`：实际进入 C 的 voxel 中 GT 为正的数量。
* `fp`：实际进入 C 的 voxel 中 GT 为负的数量。
* `fn`：dense 有效空间中 GT 为正但未进入 C 的数量。
* `p_C_*`：`candidate_prob` 的分位数和均值。
* `num_P`：优先从 `anchor_counts` 得到；无 anchor 输出时记录 `nan` 并写 warning。

#### `val_unrefined` 计算

logit 来源：

```python
candidate_outputs["candidate_logits"]
```

指标：

```text
val_unrefined/global/p
val_unrefined/global/F1
val_unrefined/global/precision
val_unrefined/global/recall
val_unrefined/global/tp
val_unrefined/global/fp
val_unrefined/global/fn
val_unrefined/global/num_C
val_unrefined/global/num_gt_in_C
```

`fn` 是 local C 内 FN，不包含 C 外 dense GT。

#### `val_refined` 计算

logit 来源：

```python
outputs["ligand_refine_logits_C"]
```

指标同 `val_unrefined`。`fn` 是 local C 内 FN。

#### `val_score` 端到端 F1 计算

从 `val_unrefined` 和 `val_refined` 的 C 内 histogram 复用 `tp_at_threshold` / `fp_at_threshold`，但 `fn` 使用 dense 全图 GT：

```text
fn_e2e = dense_num_gt - tp_at_threshold
```

输出：

```text
val_score/global/unrefined_F1
val_score/global/refined_F1
val_score/by_source_folder/<folder>/unrefined_F1
val_score/by_source_folder/<folder>/refined_F1
```

:comment[可选同步输出：]{#comment-1780045561630 text="让他们固定输出吧，不必加开关了"}

```text
val_score/global/unrefined_precision
val_score/global/unrefined_recall
val_score/global/refined_precision
val_score/global/refined_recall
```

计划默认输出 precision/recall，因为代价小且有助于判断 F1 变化来自过多 FP 还是漏召回。

### F. 日志 key 与 artifact 统一层

#### \[NEW] `src/wrappers/voxel_point_stage1_logging.py`

新增函数：

```python
def build_metric_key(
    *,
    panel: str,
    metric: str,
    num_classes: int,
    scope: str = "global",
    subpanel: str | None = None,
    source_folder: str | None = None,
    task_class_name: str | None = None,
) -> str: ...
```

| 参数                | 类型    | 意义                                                                                            |                                                 |
| ----------------- | ----- | --------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `panel`           | `str` | `val_loss`、`val_score`、`val_curve`、`val_uncapped`、`val_capped`、`val_unrefined`、`val_refined`。 |                                                 |
| `metric`          | `str` | 指标 leaf，例如 `F1`、`PRAUC`、`p_sampling_p50`。                                                     |                                                 |
| `num_classes`     | `int` | 用于内部判断二分类是否省略 suffix。                                                                         |                                                 |
| `scope`           | `str` | `global` 或 `by_source_folder`。                                                                |                                                 |
| `subpanel`        | \`str | None\`                                                                                        | `best` / `sampling`，仅 `val_uncapped` 使用。        |
| `source_folder`   | \`str | None\`                                                                                        | source folder 名。仅 `scope=by_source_folder` 时使用。 |
| `task_class_name` | \`str | None\`                                                                                        | 多分类 task class suffix 来源。二分类忽略。                 |

返回规则：

* 二分类不添加 task class suffix。
* 多分类对 task-class 相关指标追加 `_<task_class_name>`。
* source folder 必须出现在路径层级，不出现在 suffix。

新增函数：

```python
def log_scalar_payload(
    *,
    module: pl.LightningModule,
    payload: Mapping[str, torch.Tensor],
    monitor_metric: str,
    sync_dist: bool,
) -> None: ...
```

新增函数：

```python
def log_wandb_curves(
    *,
    module: pl.LightningModule,
    curves: Mapping[str, CurvePayload],
    validation_index: int,
    every_n: int,
) -> None: ...
```

要求：

* 只在 `trainer.is_global_zero` 上传 W\&B curves。
* key 固定，不带 epoch。
* 如果 logger 不是 W\&B，则 no-op。

新增函数：

```python
def write_validation_artifacts(
    *,
    run_dir: Path,
    output_subdir: str,
    epoch: int,
    global_step: int,
    payload: CpcDiagnosticsPayload,
) -> None: ...
```

要求：

* 只在 `trainer.is_global_zero` 调用。
* 创建目录：

```text
<run_dir>/<output_subdir>/epoch_000001/
  summary.json
  warnings.json
  curves/
  histograms/
```

### G. Wrapper 主流程接入

#### \[MODIFY] `src/wrappers/voxel_point_stage1.py`

**`__init__`**

1. 构造 `ValidationMetricManager`：

```python
self.val_metrics = ValidationMetricManager(...)
```

1. 构造 `SourceFolderRegistry`。
2. 若 candidate builder 启用，读取：

```python
candidate_class_ids
adaptive_expand_factor
max_candidate_voxels_per_class
selection_mode
```

1. 构造 `CpcValidationDiagnostics`：

```python
self.cpc_diagnostics = CpcValidationDiagnostics(...)
```

1. 所有 diagnostics buffer 都由 diagnostics manager 注册为 `persistent=False`。

**`validation_step` 时间顺序**

新版流程写成清晰的时间顺序：

1. `batch_dict = self._extract_batch(batch)`。
2. `source_folder_idx = self.source_folders.encode_batch(batch_dict["class_name"], device=self.device)`。
3. `_sync_sparse_candidate_runtime_to_backbone()`。
4. `outputs = self(batch_dict)`。
5. `total_loss, loss_terms = compute_total_loss(...)`。
6. 更新常规 dense/receptor/ligand AP。
7. 从 dense ligand logits 和 target 更新 `val_uncapped/best`。
8. 如果 candidate outputs 存在，更新 `val_uncapped/sampling`。
9. 更新 `val_capped`。
10. 如果 C 级 supervision 与 `candidate_logits` 存在，更新 `val_unrefined`。
11. 如果 `ligand_refine_logits_C` 存在，更新 `val_refined`。
12. 记录 step/epoch loss。

`allow_cache_update` 判断：

```python
allow_cache_update = (
    trainer is not None
    and not trainer.sanity_checking
    and not self._is_tuning_trainer(trainer)
)
```

fit warmup 阶段满足 `allow_cache_update=True`，因此可更新 candidate threshold cache。

**`on_validation_epoch_start`**

调用：

```python
self.val_metrics.reset()
self.cpc_diagnostics.reset()
```

说明 reset 时序：

```text
epoch_start reset -> batch updates -> epoch_end compute -> cache update/log/artifact -> reset
```

**`on_validation_epoch_end`**

执行：

1. `metric_payload = self.val_metrics.compute_payload()`。
2. `cpc_payload = self.cpc_diagnostics.compute_payload(sync_fn=self._all_reduce_sum)`。
3. 若 `allow_cache_update=True`，从 cpc payload 的 `val_uncapped/best` 与 `sampling` 更新：

```python
self._cached_voxel_ligand_p_best_by_class
self._cached_voxel_ligand_p_sampling_by_class
self._cached_voxel_ligand_best_f1_before_refine_by_class
```

1. `_sync_sparse_candidate_runtime_to_backbone()`。
2. 记录 scalar payload。
3. `trainer.is_global_zero` 时写 local artifact 和 W\&B curves。
4. reset managers。
5. 推进 warmup plateau scheduler。

新增 helper：

```python
def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
    ...
```

要求：

* 单卡直接返回 tensor。
* DDP 使用 `torch.distributed.all_reduce(..., ReduceOp.SUM)` 或 Lightning 等价 API。
* 所有 rank 调用形状一致。

**`on_save_checkpoint`**

新增或修改：

```python
def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    ...
```

要求：

* 保留 candidate threshold cache，因为它是训练继续所需 runtime state。
* 删除 validation metrics 中的大体积中间状态。
* diagnostics histogram/count 不应出现，因为 buffer `persistent=False`。

### H. Scheduler 与 optimizer 拆分

#### \[NEW] `src/wrappers/voxel_point_stage1_scheduler.py`

新增函数：

```python
def configure_stage1_optimizers(
    *,
    module: pl.LightningModule,
    optimizer_config: Mapping[str, Any] | None,
    scheduler_config: Mapping[str, Any] | None,
    interval: str,
    frequency: int,
    monitor_metric: str,
) -> Any: ...
```

| 参数                 | 类型                | 意义                             |                     |
| ------------------ | ----------------- | ------------------------------ | ------------------- |
| `module`           | `LightningModule` | 提供 parameters 与 trainer 上下文。   |                     |
| `optimizer_config` | \`Mapping         | None\`                         | Hydra optimizer 配置。 |
| `scheduler_config` | \`Mapping         | None\`                         | Hydra scheduler 配置。 |
| `interval`         | `str`             | Lightning scheduler interval。  |                     |
| `frequency`        | `int`             | Lightning scheduler frequency。 |                     |
| `monitor_metric`   | `str`             | plateau/checkpoint 监控指标新 key。  |                     |

第一阶段只做等价迁移，不改变 scheduler 语义。

### I. 监控指标与旧名迁移

#### \[MODIFY] active configs

必须更新当前 active experiment/config 中的 `monitor_metric`。

旧名到新名表：

| 旧名                                               | 新名                                                                  | 说明                          |
| ------------------------------------------------ | ------------------------------------------------------------------- | --------------------------- |
| `val/loss`                                       | `val_loss/global/total`                                             | 总验证 loss。                   |
| `val/atom_pr_auc`                                | `val_score/global/atom_PRAUC`                                       | atom AP。                    |
| `val/voxel_aux_pr_auc`                           | `val_score/global/receptor_PRAUC`                                   | 对外命名改 receptor。             |
| `val/voxel_ligand_pr_auc`                        | `val_score/global/voxel_ligand_PRAUC`                               | ligand dense AP。            |
| `val/voxel_ligand_macro_ap`                      | `val_score/global/voxel_ligand_PRAUC_macro`                         | 仅多分类保留 macro。               |
| `val/voxel_ligand_p_best_by_class_<class>`       | `val_uncapped/best/global/p_best_<class>`                           | 多分类 suffix；二分类无 suffix。     |
| `val/voxel_ligand_p_sampling_by_class_<class>`   | `val_uncapped/sampling/global/p_sampling_p50_<class>`               | 新版记录分位数；recorded 模式下分位数可相同。 |
| `val/voxel_ligand_best_f1_before_refine_<class>` | `val_uncapped/best/global/best_F1_<class>`                          | dense best-F1。              |
| `val/voxel_ligand_best_f1_before_refine_macro`   | `val_uncapped/best/global/best_F1_macro`                            | 仅多分类。                       |
| `val/ligand_sparse_refine_best_f1_<class>`       | `val_score/global/refined_F1_<class>`                               | 端到端 refined F1。             |
| `val/ligand_sparse_refine_best_f1_macro`         | `val_score/global/refined_F1` 或 `val_score/global/refined_F1_macro` | 二分类用无 suffix；多分类可用 macro。   |
| `val/candidate_recall_<class>`                   | `val_capped/global/recall_<class>`                                  | C 覆盖 recall。                |
| `val/candidate_recall_macro`                     | `val_capped/global/recall_macro`                                    | 仅多分类。                       |
| `val/num_candidate_voxels`                       | `val_capped/global/num_C`                                           | 实际 C 数。                     |
| `val/num_anchor_points`                          | `val_capped/global/num_P`                                           | P/anchor 数。                 |

更新范围：

* `configs/model/default.yaml`
* `configs/experiment/*.yaml`
* `configs/experiment/other/*.yaml`
* `configs/experiment/组会/*.yaml` 中仍在使用的 active 配置

`configs/experiment/old/**` 不批量修改，但计划执行时必须用 `rg -n "monitor_metric: val/" configs/experiment configs/model` 列出剩余项，并确认剩余项全部位于 `old/**` 或明确废弃目录。

### J. 文档

#### \[NEW] `CLAUDE/docs/wrapper_metrics_reference_ai.md`

面向 AI agent。内容必须包括：

1. Stage1 validation 的时间顺序。
2. C/P/C 各阶段输入输出字段。
3. W\&B key 命名规则。
4. 每个指标的精确定义，尤其：
   * `p_best`
   * `p_sampling_*`
   * `p_C_*`
   * `numC_p_best_cutoff`
   * `numC_sampling_target`
   * `numC_sampling_cutoff`
   * `num_C`
   * `num_P`
   * `val_score/refined_F1` 与 `val_refined/F1` 的区别
5. 训练运行产物位置和意义：

```text
<run_dir>/checkpoints/
<run_dir>/validation_diagnostics/
<run_dir>/validation_diagnostics/epoch_xxxxxx/summary.json
<run_dir>/validation_diagnostics/epoch_xxxxxx/warnings.json
<run_dir>/validation_diagnostics/epoch_xxxxxx/curves/*.csv
```

1. 服务器排查流程：
   * 如何找到最新 run。
   * 如何查看 `summary.json`。
   * 如何对比 W\&B key 与本地 CSV。
   * 如何判断问题发生在 dense、sampling、cap、local refined 还是 e2e。
2. 常见异常解释：
   * `val_uncapped/best` 高但 `val_capped/recall` 低。
   * `val_capped/recall` 高但 `val_score/refined_F1` 低。
   * `val_refined/F1` 高但 `val_score/refined_F1` 低。
   * 某 source folder 无正例。
   * `p_sampling_*` 全是 `nan`。

#### \[NEW] `docs/model/stage1_wrapper_cpc_guide.md`

面向用户。要求：

* 按实际运行时间顺序解释 CPC，不贴代码。
* 重点解释每一步的实际意义。
* 说明 W\&B 上的验证指标如何阅读。
* 明确区分：

```text
val_score/global/refined_F1        端到端最终分数
val_refined/global/F1              C 内局部判别分数
val_capped/global/recall           C 是否覆盖正例
val_uncapped/best/global/best_F1   dense 理论上限
```

建议章节：

1. dense voxel 先给整张空间打概率。
2. `val_uncapped/best` 看 dense 分支理论上限。
3. `val_uncapped/sampling` 看采样策略打算怎样切 C。
4. `val_capped` 看真正进入 C 的候选是否覆盖正例。
5. C 进入 P/point/refine 流程。
6. `val_unrefined` 看 C 内原始 dense logit 判别力。
7. `val_refined` 看 C 内 refined logit 判别力。
8. `val_score` 看端到端最终结果。
9. global 与 by\_source\_folder 的区别。
10. 二分类为什么不显示 macro。

## 不修改的部分

* 不改变 backbone state\_dict 中用于推理的旧 key。
* 不改变 `voxel_logits_aux` 等内部模型输出名。
* 不实现 box 级或 voxel 级样本明细表。
* 不上传样本级 dense voxel prediction 列表到 W\&B。
* 不为旧 W\&B metric 名保留 alias。
* 不把 source folder 动态扩展为未知 bucket。
* 不让正式 `topk` 模式从 `warmup_topc_per_class` fallback。
* 不改变 scheduler 数学语义，只迁移代码位置与 monitor key。

## 改动文件汇总

| 文件                                                      | 改动内容                                                                                |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `src/model/sparse_refine/candidate_set.py`              | 新增正式 `selection_mode=topk` 与 `topk_per_class`。                                      |
| `src/model/stage1_model.py`                             | 必要时暴露 candidate builder selection/topk 配置给 wrapper diagnostics；保持 forward 调用签名不扩散。  |
| `src/wrappers/voxel_point_stage1.py`                    | 瘦身为 Lightning 主流程 coordinator，接入 metrics/diagnostics/loss/logging/scheduler helper。 |
| `src/wrappers/voxel_point_stage1_losses.py`             | 新增 loss term helper。                                                                |
| `src/wrappers/voxel_point_stage1_metrics.py`            | 新增 `ValidationMetricManager(nn.Module)`。                                            |
| `src/wrappers/voxel_point_stage1_diagnostics.py`        | 新增 CPC diagnostics manager 与 source folder registry。                                |
| `src/wrappers/voxel_point_stage1_logging.py`            | 新增 key 构造、scalar/curve/artifact logging。                                            |
| `src/wrappers/voxel_point_stage1_scheduler.py`          | 新增 optimizer/scheduler 配置 helper。                                                   |
| `configs/model/default.yaml`                            | 新增 `class_names` 显式透传与 `validation_diagnostics` 配置。                                 |
| `configs/model/task/*.yaml`                             | 新增 `topk_per_class`。                                                                |
| `configs/model/sparse_refine/candidate_set/*_topk.yaml` | 新增 topk candidate set 配置。                                                           |
| `configs/experiment/*.yaml`                             | 更新 active monitor metric。                                                           |
| `tests/test_voxel_ligand_thresholds.py`                 | 更新阈值、warmup cache、no-positive 测试。                                                   |
| `tests/test_ligand_sparse_refine_metrics.py`            | 更新 local/e2e refined F1 测试。                                                         |
| `tests/test_multiclass_ligand_wrapper.py`               | 更新 macro/suffix 测试。                                                                 |
| `tests/test_sparse_candidate_set.py` 或同类测试              | 新增正式 topk selection mode 测试。                                                        |
| `tests/test_voxel_point_stage1_cpc_diagnostics.py`      | 新增 diagnostics manager 单测。                                                          |
| `tests/test_voxel_point_stage1_metric_logging.py`       | 新增 key 构造与 W\&B/artifact rank0 保护测试。                                                |
| `tests/inference/test_get_voxel_pred.py`                | 保持旧 checkpoint 推理兼容测试。                                                              |
| `CLAUDE/docs/wrapper_metrics_reference_ai.md`           | 新增 AI-facing 指标与产物说明。                                                               |
| `docs/model/stage1_wrapper_cpc_guide.md`                | 新增用户-facing CPC 与 W\&B 指标说明。                                                        |

## Verification Plan

### Automated Tests

候选 builder：

```bash
pytest tests/test_sparse_candidate_set.py -v
```

验证点：

1. `selection_mode="topk"` 且缺少 `topk_per_class` 时 fail-fast。
2. `selection_mode="topk"` 不读取 `p_best_by_class` / `p_sampling_by_class`。
3. topk 模式每 BOX/类候选数不超过 `topk_per_class` 与 `max_candidate_voxels_per_class`。
4. topk 模式正确写入 `candidate_p_sampling_by_class` cutoff probability。
5. adaptive/recorded 原有行为不变。

CPC diagnostics：

```bash
pytest tests/test_voxel_point_stage1_cpc_diagnostics.py -v
```

验证点：

1. `val_uncapped/best` 正确输出 `p_best`、`best_F1`、`numC_p_best_cutoff`。
2. `val_uncapped/sampling` 对 recorded/adaptive/topk/warmup 都按真实 per BOX 边界聚合 `sampling_tp/fp/fn`。
3. 空 p\_sampling 边界集合返回 `nan` 与 warning，不抛异常。
4. `val_capped` 输出 `p_C_*`、`num_C`、`num_P`。
5. `val_unrefined/F1` 使用 C 内 local fn。
6. `val_refined/F1` 使用 C 内 local fn。
7. `val_score/refined_F1` 使用 dense 全图 e2e fn。
8. source folder 维度与 global 同时存在。
9. unknown source folder fail-fast。

TorchMetrics 与设备策略：

```bash
pytest tests/test_voxel_point_stage1_metric_logging.py -v
```

验证点：

1. `ValidationMetricManager` 是 `nn.Module`。
2. 非 binned AP 在 `auto` 下不把整轮 prediction 堆到 GPU。
3. binned AP 在 `auto` 下使用当前训练 device。
4. batch size tuner 仍执行 AP update/compute。
5. checkpoint state\_dict 不包含大体积 validation metric 中间状态。

旧测试更新：

```bash
pytest tests/test_voxel_ligand_thresholds.py tests/test_ligand_sparse_refine_metrics.py tests/test_multiclass_ligand_wrapper.py -v
```

验证点：

1. 二分类不输出 `*_macro`。
2. 二分类 key 不带 task class suffix。
3. 多分类 key 带 task class suffix。
4. `PRAUC`、`PRcurve`、`F1` 大小写符合规范。
5. `aux` 对外日志改为 `receptor`。
6. warmup validation 会更新 candidate threshold cache。
7. sanity/tuner 不更新 candidate threshold cache。

推理兼容：

```bash
pytest tests/inference/test_get_voxel_pred.py -v
```

验证点：

1. 旧 checkpoint 中 `voxel_aux_*` / `voxel_logits_aux` 相关权重仍能加载。
2. 推理输出 key 映射不因日志改名而变化。
3. 不要求旧 W\&B metric 名存在。

配置扫描：

```bash
rg -n "monitor_metric: val/" configs/experiment configs/model
rg -n "selection_mode:" configs/model/sparse_refine/candidate_set
rg -n "topk_per_class" configs/model
```

验证点：

1. active 配置不再监控旧 `val/...` metric。
2. `old/**` 中旧 monitor 若保留，需要在执行记录里说明。
3. topk 配置显式包含 `topk_per_class`。

### Manual Verification

单卡 smoke：

```bash
python src/train.py experiment=MINI_sparse_refine000 trainer.max_epochs=1
```

观察：

1. W\&B 出现 `val_score/global/refined_F1`。
2. W\&B 出现 `val_refined/global/F1`，且文档中定义为 local。
3. W\&B 出现 `val_capped/global/recall` 与 `p_C_*`。
4. 二分类无 macro。
5. 本地 run dir 生成 `validation_diagnostics/epoch_xxxxxx/summary.json`。

topk smoke：

```bash
python src/train.py experiment=MINI_sparse_refine000 model/sparse_refine/candidate_set=binary_topk trainer.max_epochs=1
```

观察：

1. 不要求已有 `p_best` / `p_sampling` cache 才能 forward。
2. `val_uncapped/sampling/global/p_sampling_*` 有值。
3. `val_capped/global/num_C` 接近 `topk_per_class` 与 cap 约束后的规模。

DDP smoke：

```bash
python src/train.py experiment=MINI_sparse_refine000 trainer.devices=2 trainer.strategy=ddp trainer.max_epochs=1
```

观察：

1. 无 collective hang。
2. 只有 global zero 写本地 artifact。
3. W\&B curve 不重复上传。
4. 某 rank 本地无正例时训练不因 diagnostics 崩溃。

### Documentation Verification

检查：

1. `CLAUDE/docs/wrapper_metrics_reference_ai.md` 能让 AI agent 根据 run dir 找到 checkpoint、summary、warnings、curves。
2. `docs/model/stage1_wrapper_cpc_guide.md` 能让用户按时间顺序理解 dense -> C -> P -> refined，并能区分 `val_score/refined_F1` 与 `val_refined/F1`。
3. 两份文档都解释 source folder 与 task class 的区别。
4. 两份文档都说明二分类为什么没有 macro。

## 实施顺序

1. 新增 topk candidate builder 支持与配置文件，先用 builder 单测锁定行为。
2. 新增 logging key helper，迁移 loss 命名到 `val_loss` / `train_loss` 与 `receptor`。
3. 新增 `ValidationMetricManager`，迁移常规 AP/PRAUC，保留设备策略。
4. 新增 `CpcValidationDiagnostics`，先实现 global，再实现 by\_source\_folder。
5. 在 wrapper 中接入 diagnostics 主流程，保留时间顺序。
6. 接入 `val_score` 端到端 `unrefined_F1` / `refined_F1`。
7. 接入 W\&B curve 与本地 artifact，确保 rank0-only。
8. 迁移 optimizer/scheduler helper。
9. 更新 active configs 的 monitor metric。
10. 写 AI-facing 与用户-facing 文档。
11. 跑自动化测试与单卡/topk/DDP smoke。

## 验收标准

* `voxel_point_stage1.py` 明显变短，主流程能按时间顺序阅读。
* `selection_mode=topk` 可作为正式采样模式使用，且必须显式配置 `topk_per_class`。
* warmup validation 能更新 candidate threshold cache；sanity/tuner 不会污染 cache。
* DDP 下 diagnostics collective 对称执行，不因本地无正例 deadlock。
* W\&B curve/table 与本地 artifact 只在 global zero 执行。
* diagnostics buffer 使用 `persistent=False`，validation metric 状态不污染 checkpoint。
* 二分类无 macro。
* 对外日志无 `aux`，使用 `receptor`。
* `val_score/global/refined_F1` 是端到端 refined F1。
* `val_refined/global/F1` 是 C 内 local refined F1。
* global 与 by\_source\_folder 指标同时存在。
* 每个 source folder 使用独立 W\&B key。
* 两份文档完成并能解释运行产物、指标含义与 CPC 时间顺序。