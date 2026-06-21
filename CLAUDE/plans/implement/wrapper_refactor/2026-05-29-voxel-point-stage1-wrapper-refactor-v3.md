# VoxelPointStage1 Wrapper 重构与验证诊断实施计划 v3

## 1. 目标

本计划用于重构 `src/wrappers/voxel_point_stage1.py`，目标不是只改日志名，而是把 Stage1 训练 wrapper 拆成便于初学者分层阅读、便于调试、便于后续扩展的结构。重构后，wrapper 本体保留 Lightning 生命周期与训练/验证主流程，损失计算、验证指标、C/P/C 诊断统计、W&B/local artifact 记录、调度器构建等职责拆入独立模块。

同时，本计划会新增一套验证诊断体系，把候选采样与 refinement 的表现按时间顺序分成四个主要验证面板：

- `val_uncapped`：原始 dense 空间中关于 C 的预采样质量，属于 global 统计。
- `val_capped`：经过 cap 后实际进入 C 的候选集合质量，属于 global 统计。
- `val_unrefined`：在实际 C 内使用 dense logit 做局部判定的质量，属于 local 统计。
- `val_refined`：在实际 C 内使用 refined logit 做局部判定的质量，属于 local 统计。

其中 `val_uncapped` 进一步分成 `best` 与 `sampling` 两个子条目，因为 `p_best` 与 `p_sampling` 是两个不同的诊断系统：前者回答“dense 分支理论上能达到的最佳阈值在哪里”，后者回答“实际候选采样策略在用什么边界切出 C”。

## 2. 非目标

本次不改变模型主体结构，不改变 backbone 参数命名，不改变 checkpoint 中用于推理的权重键名。尤其是内部仍可保留 `voxel_logits_aux`、`voxel_aux_head`、`voxel_aux_logit_dim` 等历史模型键，以保证旧训练产物仍能通过现有推理管线加载。对外日志、文档、配置说明中则统一使用 `receptor` 表达业务含义，避免继续把 receptor 相关内容写成 `aux`。

本次不实现 box 级、voxel 级明细样本表，不上传每个样本的稠密 voxel 列表，也不保存可恢复单样本空间位置的超大诊断文件。诊断范围限定为 epoch 级聚合统计、固定 bin 直方图、source folder 分组统计、PR curve table/CSV/JSON。

本次不保留旧 W&B metric 名的兼容别名。训练配置中如果存在 `monitor`、early stopping、checkpoint callback、dashboard 查询等引用旧名的位置，必须同步更新。旧训练产出的 checkpoint 仍可用于推理，因为 W&B metric 名不是推理加载路径的一部分。

## 3. 现状约束

`src/wrappers/voxel_point_stage1.py` 当前约 2000 行，混合了以下职责：

- LightningModule 初始化、模型/损失/指标状态构造。
- dense voxel、sparse refine、receptor 辅助分支的 loss 计算。
- 训练 step、验证 step、hook、checkpoint 保存/加载。
- TorchMetrics AP/F1/Accuracy/Precision/Recall 等指标注册、更新、compute/reset。
- ligand candidate 的 best-F1 阈值搜索、candidate recall、p_sampling 更新。
- refined C 上的 sparse metrics。
- W&B 和 Lightning logger 的 key 命名。
- optimizer/scheduler 构造。

现有 `src/model/sparse_refine/candidate_set.py` 已经输出候选集合相关信息，包括但不限于：

- `candidate_logits`
- `candidate_prob`
- `candidate_counts`
- `candidate_counts_by_class`
- `candidate_p_sampling_by_class`

其中 `recorded_threshold`、`adaptive_threshold`、warmup/topk 三类候选策略的语义不同。新诊断层必须把它们统一为“采样边界概率分布”的可读统计，而不是假装它们都只有一个固定阈值。

`src/datasets/box_point_dataset.py` 会在样本中写入 `class_name`，`src/datasets/box_point_collate.py` 会把非 tensor 元数据保留为 batch 内 list，因此验证时可以读取 `batch["class_name"]`。这里的 `class_name` 是原始数据文件夹名，例如 `metal_ion`、`random_BOX`，不是经过 `class_mapping` 后的任务类别名。

`configs/dataset/emb_unet.yaml` 中当前存在：

```yaml
class_folder_names: ["metal_ion", "peptide", "nucleic", "small_molecule", "random_BOX"]
class_mapping: [0, 1, 1, 1, 1]
class_names: [background, foreground]
```

因此 source folder 分组统计必须显式区分：

- source folder：原始数据来源文件夹，如 `metal_ion`、`random_BOX`。
- task class：模型训练时的类别，如二分类时的 `background` 与 `foreground`。

`docs/验证时的metric计算问题.md` 已经给出了验证 AP 的关键约束：非 binned AP 默认应倾向 CPU 以节省显存；binned AP 默认应倾向 GPU，因为固定阈值扫描状态小、计算快；batch size tuner 不应跳过验证 AP，以便暴露真实验证链路问题。本次重构必须保留并制度化这套策略。

## 4. 关键设计原则

### 4.1 Lightning 与 TorchMetrics 注册安全

任何持有 `torchmetrics.Metric` 的 manager 都必须继承 `torch.nn.Module`，或者通过 `nn.ModuleDict` / `nn.ModuleList` 让 Lightning 能够从模块树中发现它。不得把 `BinaryAveragePrecision`、`MulticlassAveragePrecision` 等 metric 隐藏在普通 Python helper 内部。

计划新增的 `ValidationMetricManager` 必须是 `nn.Module`：

```python
class ValidationMetricManager(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        class_names: Sequence[str],
        metric_device_policy: str,
        binary_threshold: float,
        pr_auc_thresholds: int | Sequence[float] | None,
        ...
    ) -> None: ...
```

wrapper 在 `__init__` 中通过 `self.val_metrics = ValidationMetricManager(...)` 持有它。这样设备转移、DDP、checkpoint state_dict 都由 PyTorch/Lightning 的模块树处理。

### 4.2 DDP 分组统计使用静态 tensor 形状

source folder 分组诊断不得使用动态字符串 dict 跨 rank 合并。必须从配置中读取固定 `source_folder_names`，建立静态映射：

```python
source_folder_to_idx: dict[str, int]
```

所有分组 buffer 都使用固定形状 tensor，例如：

```text
(num_source_folders, num_task_classes, num_bins)
(num_source_folders, num_task_classes)
```

在 DDP epoch end 时，只对固定形状 tensor 做 `all_reduce(sum)` 或通过 Lightning 的同步策略聚合。若 batch 中出现不在 `source_folder_names` 的值，应 fail-fast，因为这代表数据配置与样本实际来源不一致。

### 4.3 诊断不因无正例中断训练

如果某个 class 或 source folder 在某轮验证中没有 GT 正例，诊断统计不应抛出致命异常。该类指标应记录为 `nan`、跳过曲线生成，或在 local summary JSON 中写 warning。结构性错误，例如未知 source folder、tensor 形状不一致、stage 输入缺失，则应明确报错。

### 4.4 日志不做旧名兼容

本次是彻底重命名。二分类不再输出任何 `*_macro`，也不再保留旧 `voxel_aux_*` 日志别名。二分类下 class suffix 省略；多分类下才添加 task class suffix。

## 5. 新日志分组与命名规范

### 5.1 基本路径结构

W&B key 使用路径层级表达语义：

```text
<panel>/<subpanel>/<scope>/<metric>
<panel>/<scope>/<metric>
<panel>/<scope>/<source_folder>/<metric>
```

其中：

- `panel` 包括 `val_loss`、`val_score`、`val_curve`、`val_uncapped`、`val_capped`、`val_unrefined`、`val_refined`。
- `subpanel` 当前仅用于 `val_uncapped/best` 与 `val_uncapped/sampling`。
- `scope` 包括 `global` 与 `by_source_folder`。
- `source_folder` 使用原始文件夹名，例如 `metal_ion`、`random_BOX`。
- `metric` 使用短名，例如 `F1`、`PRAUC`、`PRcurve`、`tp`、`fp`、`fn`、`precision`、`recall`、`p_best`。

### 5.2 task class suffix

二分类任务中省略 task class suffix：

```text
val_refined/global/F1
val_capped/by_source_folder/random_BOX/recall
```

多分类任务中对 task class 相关指标追加 `_<class_name>`：

```text
val_refined/global/F1_small_molecule
val_capped/by_source_folder/random_BOX/recall_metal_ion
```

这里的 suffix 是 task class 名，不是 source folder 名。source folder 已经在路径层级中表达。

### 5.3 loss 与 receptor 命名

对外 loss 日志使用 `receptor`，不再使用 `aux`：

```text
train_loss/receptor_loss_step
val_loss/receptor_loss
```

内部模型键仍可保留 `aux`，例如 `voxel_logits_aux`，因为这是 checkpoint 兼容边界。

### 5.4 score 与 curve

常规分数放在 `val_score`：

```text
val_score/global/voxel_ligand_PRAUC
val_score/global/voxel_ligand_F1
val_score/by_source_folder/random_BOX/voxel_ligand_PRAUC
```

PR 曲线放在 `val_curve`：

```text
val_curve/global/voxel_ligand_PRcurve
val_curve/by_source_folder/random_BOX/voxel_ligand_PRcurve
```

`PRAUC` 使用大写，`PRcurve` 使用 camel 风格。W&B 中每个 source folder 用独立 key，避免多个 folder 混在同一个 table/panel 内。

## 6. 四阶段 CPC 诊断定义

### 6.1 `val_uncapped/best`

实际意义：在整张 dense voxel 图上，不考虑候选数量 cap，仅询问 dense ligand 分支如果自由选择概率阈值，理论上能达到怎样的最佳 F1。这是 dense 分支本身可分性的上限诊断。

统计范围：global dense 空间。支持 `global` 与 `by_source_folder/<folder>`。

核心指标：

```text
p_best
best_F1
best_precision
best_recall
best_tp
best_fp
best_fn
num_gt
numC_p_best_cutoff
```

定义：

- `p_best`：使 F1 最优的 dense 概率阈值。
- `best_tp`：dense 概率大于等于 `p_best` 且 GT 为正的 voxel 数。
- `best_fp`：dense 概率大于等于 `p_best` 且 GT 为负的 voxel 数。
- `best_fn`：dense 概率低于 `p_best` 且 GT 为正的 voxel 数。
- `best_precision = best_tp / (best_tp + best_fp)`。
- `best_recall = best_tp / (best_tp + best_fn)`。
- `best_F1`：由 `best_precision` 与 `best_recall` 得到。
- `num_gt`：dense 空间内 GT 正例 voxel 数。
- `numC_p_best_cutoff`：若用 `p_best` 作为候选阈值，会产生多少候选 voxel。

### 6.2 `val_uncapped/sampling`

实际意义：在整张 dense voxel 图上，诊断实际采样策略想切出 C 时使用了什么概率边界，以及这个边界在未 cap 前对应怎样的 TP/FP/FN。这回答“采样策略本身是否把正例放进了候选集合附近”。

统计范围：global dense 空间。支持 `global` 与 `by_source_folder/<folder>`。

核心指标：

```text
p_sampling_p5
p_sampling_p50
p_sampling_p75
p_sampling_p95
p_sampling_mean
sampling_F1
sampling_precision
sampling_recall
sampling_tp
sampling_fp
sampling_fn
num_gt
numC_sampling_target
numC_sampling_cutoff
```

定义：

- `p_sampling_*`：当前候选策略形成的采样边界概率分布摘要。无论是 `adaptive_threshold`、`recorded_threshold` 还是 topk，都统一记录为 p5/p50/p75/p95/mean。
- `recorded_threshold` 下如果阈值固定，若干分位数可能相同，这是可接受且有意义的。
- `adaptive_threshold` 下不同 batch、box、class 的边界可不同，分位数展示其分布。
- topk 下没有显式固定概率阈值，但第 K 个被选中 voxel 的概率就是实际边界，因此仍可纳入 `p_sampling_*`。
- `sampling_tp/fp/fn`：以采样边界在 dense 空间中产生的候选判断为准计算。
- `sampling_precision/recall/F1`：由 `sampling_tp/fp/fn` 计算。
- `numC_sampling_target`：采样策略目标上的候选数量，例如 topk 或 adaptive 目标量。
- `numC_sampling_cutoff`：按实际采样边界在未 cap 语义下会切出的候选数量。

### 6.3 `val_capped`

实际意义：经过实际 candidate cap 后，真正进入 C 的 voxel 是否覆盖了 dense 空间中的 GT 正例。这是采样链路的最终质量，不关心 C 内后续分类器是否能判对。

统计范围：global dense 空间中的实际 C。支持 `global` 与 `by_source_folder/<folder>`。

核心指标：

```text
p_sampling_p5
p_sampling_p50
p_sampling_p75
p_sampling_p95
p_sampling_mean
F1
precision
recall
tp
fp
fn
num_gt
num_C
num_P
```

定义：

- `tp`：实际进入 C 的 voxel 中 GT 为正的数量。
- `fp`：实际进入 C 的 voxel 中 GT 为负的数量。
- `fn`：dense 空间中 GT 为正但没有进入 C 的数量。
- `precision = tp / (tp + fp)`。
- `recall = tp / (tp + fn)`。
- `F1`：由 capped precision/recall 得到。
- `num_C`：实际进入候选集合 C 的 voxel 总数。
- `num_P`：C 到 P 或 point/refine 分支实际处理的点/anchor 数量。若当前 batch 输出无法稳定提供该值，第一阶段只记录可可靠获得的 `num_C`，并在文档中标注 `num_P` 的来源限制。
- `p_sampling_*`：cap 后实际边界概率分布摘要。如果 cap 改变了候选边界，则该组指标应反映 cap 后实际 C 的边界。

### 6.4 `val_unrefined`

实际意义：只看实际 C 内的 voxel，使用 dense ligand logit 对 C 内候选做局部判定，衡量“进入 C 后，不经过 sparse refine，仅靠原始 dense 分支能否区分 C 内正负例”。

统计范围：local C 内。支持 `global` 与 `by_source_folder/<folder>`。

核心指标：

```text
p
F1
precision
recall
tp
fp
fn
num_C
num_gt_in_C
```

定义：

- `p`：C 内使用 dense logit/prob 进行 best-F1 sweep 得到的局部最佳阈值。
- `tp/fp/fn`：只在 C 内统计，不把 dense 图上没有进入 C 的 voxel 当作 local fn。
- `num_gt_in_C`：C 内 GT 正例数量。

### 6.5 `val_refined`

实际意义：只看实际 C 内的 voxel，使用 refined logit 对 C 内候选做局部判定，衡量 sparse refine 分支是否改善了 C 内正负例区分。

统计范围：local C 内。支持 `global` 与 `by_source_folder/<folder>`。

核心指标：

```text
p
F1
precision
recall
tp
fp
fn
num_C
num_gt_in_C
```

定义同 `val_unrefined`，但 logit 来源换成 refined ligand output。

## 7. source folder 分组策略

分组字段固定使用 batch metadata 中的 `class_name`，语义是数据源文件夹。配置字段命名避免使用容易混淆的 `by_class`：

```yaml
validation_diagnostics:
  source_folder_breakdown: true
  source_folder_names: ${dataset.class_folder_names}
  source_folder_meta_key: class_name
```

每个 source folder 都保留单独 W&B key：

```text
val_capped/by_source_folder/metal_ion/F1
val_capped/by_source_folder/random_BOX/F1
```

不把多个 source folder 放进同一个 W&B table 或同一条曲线。这样做会增加标量数量，但可读性最高，并且符合“每个 class name 一栏”的使用需求。

global 指标必须同时保留：

```text
val_capped/global/F1
val_refined/global/F1
```

## 8. W&B 与本地 artifact 策略

### 8.1 标量

标量每轮 validation 都记录。对于约 30 次验证的训练规模，主要代价是 W&B UI 面板数量与加载速度，而不是训练速度、内存或显存。

需要控制的是 key 数量和命名稳定性：

- key 名必须固定，不能把 epoch 写进 key。
- 二分类不输出 macro，不输出 class suffix。
- 多分类才输出 `_class_name` suffix。
- source folder 分组独立 key，但不动态生成未知 folder。

### 8.2 PRcurve table

PR curve 每轮 validation 可以记录，但必须使用稳定 key：

```text
val_curve/global/voxel_ligand_PRcurve
val_curve/by_source_folder/metal_ion/voxel_ligand_PRcurve
```

W&B 的 step 历史自然支持用 `>` 翻阅不同 validation epoch。不得生成 `.../epoch_001`、`.../epoch_002` 这类不断膨胀的新 key。

为了避免 W&B 网页过慢，PR curve table 只保存固定 bin 的聚合曲线，不保存样本级预测。若后续发现 W&B UI 仍过重，可通过配置把 W&B curve 上传频率降为每 N 次或仅 best epoch，但本计划默认每轮记录。

### 8.3 本地文件

每轮验证结束时，在 Lightning logger 的 run dir 下写入：

```text
<run_dir>/validation_diagnostics/
  epoch_000001/
    summary.json
    warnings.json
    curves/
      global_voxel_ligand_PRcurve.csv
      by_source_folder_metal_ion_voxel_ligand_PRcurve.csv
      by_source_folder_random_BOX_voxel_ligand_PRcurve.csv
    histograms/
      val_uncapped_best_global.json
      val_uncapped_sampling_global.json
      val_capped_global.json
      val_unrefined_global.json
      val_refined_global.json
```

`summary.json` 保存本轮所有标量、配置摘要、source folder 名单、metric device policy、candidate strategy。`warnings.json` 保存无正例、曲线跳过、`num_P` 来源缺失等非致命信息。

AI-facing 文档要明确说明这些文件在本地与服务器上的意义，方便 AI agent 登录服务器后直接定位 run 产物并判断异常。

## 9. 性能与资源评估

### 9.1 训练速度

新增统计以直方图、计数器、固定 bin sweep 为主，不做样本级全量保存。候选阶段本来已经产生 dense prob 与 C 相关输出，额外统计主要是 mask、histogram、sum/count。

预期训练速度影响较小。最明显的耗时来自 validation epoch end 的曲线表生成与 logger 上传，而不是 forward/backward。

### 9.2 CPU 内存

每个 source folder、每个 task class、每个 stage 保存固定长度 histogram。即使使用 512 或 1024 bins，内存仍在工程可接受范围。示意：

```text
5 source folders * 2 task classes * 1024 bins * 若干直方图 * 8 bytes
```

数量级通常是 KB 到低 MB，不会接近样本级 voxel 表的开销。

### 9.3 GPU 显存

显存风险主要来自 TorchMetrics AP 的状态策略，而不是新增 CPC 计数。必须遵守 `docs/验证时的metric计算问题.md`：

- `thresholds=None` 的非 binned AP 默认 CPU。
- 固定 bins 的 binned AP 默认 GPU。
- `val_metric_device_policy: auto|cpu|gpu` 继续可控。

CPC 诊断的 epoch buffer 可放 CPU；batch 内临时 tensor 在当前 device 上完成必要计算后，把聚合计数迁移到 CPU buffer。不得把整轮 validation 的 prediction/target 列表堆在 GPU。

### 9.4 W&B UI 与网络

最主要代价是 W&B 标量数量和 PRcurve table 数量。按照当前 `class_folder_names` 约 5 个 source folder、约 30 次 validation 的规模，工程上可接受。若未来 source folder 或 task class 数量显著增加，应通过配置关闭部分 by_source_folder curve 或降低 curve 上传频率。

## 10. 文件级修改计划

### 10.1 `src/wrappers/voxel_point_stage1.py`

保留为唯一 LightningModule 入口，外部 Hydra `_target_` 不变。

主要修改：

- `__init__` 只负责保存超参、构造 backbone、构造 helper/manager、注册 Lightning 可见 module。
- `training_step` 保留主流程，但调用 loss helper 与 logging helper。
- `validation_step` 保留时间顺序主流程：
  1. batch 前处理。
  2. backbone forward。
  3. loss 计算。
  4. dense/receptor 常规指标更新。
  5. candidate set 生成。
  6. `val_uncapped/best` 更新。
  7. `val_uncapped/sampling` 更新。
  8. `val_capped` 更新。
  9. sparse refine forward。
  10. `val_unrefined` 更新。
  11. `val_refined` 更新。
  12. step 级日志。
- `on_validation_epoch_end` 只协调：
  - metric manager compute/reset。
  - CPC diagnostics compute/reset。
  - W&B scalar/table logging。
  - local artifact 写入。
- 原 `_update_voxel_ligand_best_f1_stats`、`_compute_log_update_voxel_ligand_best_f1_thresholds`、`_update_val_ligand_sparse_refine_metric`、`_compute_log_reset_ligand_sparse_refine_metrics` 的实质逻辑迁移到 diagnostics 模块。
- 原 `_register_val_metric`、`_init_multiclass_ap_metrics`、`_update_val_metric`、`_update_binary_or_multiclass_ap`、`_compute_log_reset_metric_safe`、`_compute_log_reset_multiclass_metrics` 的实质逻辑迁移到 metric manager。

### 10.2 新增 `src/wrappers/voxel_point_stage1_losses.py`

职责：只管理 loss 计算，不做 W&B key 拼接，不直接写 logger。

建议函数：

```python
def compute_dense_ligand_loss(
    *,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    config: DenseLigandLossConfig,
) -> LossTerm: ...

def compute_receptor_loss(
    *,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    config: ReceptorLossConfig,
) -> LossTerm: ...

def compute_sparse_refine_loss(
    *,
    refine_outputs: Mapping[str, Tensor],
    candidate_batch: Mapping[str, Any],
    config: SparseRefineLossConfig,
) -> LossTerm: ...
```

`LossTerm` 至少包含：

```python
@dataclass
class LossTerm:
    name: str
    value: Tensor
    weight: float
    detached: float | Tensor
```

命名中使用 `receptor`，不再使用对外 `aux`。

### 10.3 新增 `src/wrappers/voxel_point_stage1_metrics.py`

职责：常规 validation metrics，包括 AP/PRAUC、Accuracy、Precision、Recall、F1 等。这里管理的是模型输出分数类指标，不管理 CPC 阶段计数。

关键类：

```python
class ValidationMetricManager(nn.Module):
    def update_dense_ligand(
        self,
        *,
        logits: Tensor,
        target: Tensor,
        source_folder_idx: Tensor | None,
    ) -> None: ...

    def update_receptor(
        self,
        *,
        logits: Tensor,
        target: Tensor,
        source_folder_idx: Tensor | None,
    ) -> None: ...

    def compute_log_payload(self) -> dict[str, float | Tensor | Any]: ...

    def reset(self) -> None: ...
```

实现要求：

- 继承 `nn.Module`。
- 使用 `nn.ModuleDict` 存放 TorchMetrics。
- 保留 `val_metric_device_policy`。
- 根据 AP 是否 binned 决定 CPU/GPU 策略。
- batch size tuner 期间仍计算验证 AP。
- 二分类不生成 macro。
- 多分类时只对有效前景 task class 生成 class suffix 与必要 macro。

### 10.4 新增 `src/wrappers/voxel_point_stage1_diagnostics.py`

职责：C/P/C 四阶段诊断统计。该模块不持有 TorchMetrics AP，主要持有固定形状 tensor buffer 和直方图。

建议数据结构：

```python
@dataclass(frozen=True)
class SourceFolderRegistry:
    names: tuple[str, ...]
    name_to_idx: dict[str, int]

    def encode_batch(self, class_names: Sequence[str]) -> Tensor: ...
```

```python
@dataclass
class CpcDiagnosticsConfig:
    enabled: bool
    source_folder_breakdown: bool
    num_bins: int
    probability_min: float
    probability_max: float
    write_local_artifacts: bool
    log_wandb_curves: bool
    wandb_curve_every_n_validation: int
```

```python
class CpcValidationDiagnostics(nn.Module):
    def update_uncapped_best(...): ...
    def update_uncapped_sampling(...): ...
    def update_capped(...): ...
    def update_unrefined(...): ...
    def update_refined(...): ...
    def compute(self) -> CpcDiagnosticsPayload: ...
    def reset(self) -> None: ...
```

虽然该类主要是 buffer 而非 TorchMetrics，也建议继承 `nn.Module` 并用 `register_buffer` 注册固定 tensor，便于设备管理与 DDP。

`compute()` 输出：

```python
@dataclass
class CpcDiagnosticsPayload:
    scalars: dict[str, float]
    curves: dict[str, CurvePayload]
    warnings: list[DiagnosticsWarning]
    local_tables: dict[str, TablePayload]
```

### 10.5 新增 `src/wrappers/voxel_point_stage1_logging.py`

职责：统一构造 metric key、过滤二分类 macro、把 payload 写入 Lightning/W&B/local files。

建议函数：

```python
def build_metric_key(
    *,
    panel: str,
    metric: str,
    scope: str = "global",
    subpanel: str | None = None,
    source_folder: str | None = None,
    task_class_name: str | None = None,
    binary_task: bool,
) -> str: ...
```

```python
def log_scalar_payload(
    *,
    lightning_module: pl.LightningModule,
    payload: Mapping[str, float | Tensor],
    sync_dist: bool,
) -> None: ...
```

```python
def log_wandb_curves(
    *,
    lightning_module: pl.LightningModule,
    curves: Mapping[str, CurvePayload],
    validation_index: int,
) -> None: ...
```

```python
def write_validation_artifacts(
    *,
    run_dir: Path,
    epoch: int,
    payload: CpcDiagnosticsPayload,
) -> None: ...
```

实现要求：

- key 稳定，不带 epoch。
- W&B table 使用固定 key。
- 本地文件路径带 epoch，方便离线对比。
- 若 logger 不是 W&B，则跳过 W&B table，但仍写本地文件。

### 10.6 新增 `src/wrappers/voxel_point_stage1_scheduler.py`

职责：迁移 optimizer/scheduler 构造中的长逻辑，减少 wrapper 尾部复杂度。

建议函数：

```python
def configure_stage1_optimizers(
    *,
    module: nn.Module,
    optimizer_config: Mapping[str, Any],
    scheduler_config: Mapping[str, Any] | None,
    trainer: pl.Trainer | None,
) -> Any: ...
```

第一阶段只做等价迁移，不改变调度器语义。

### 10.7 新增 AI-facing 文档

路径：

```text
CLAUDE/docs/wrapper_metrics_reference_ai.md
```

内容：

- wrapper 验证链路的时间顺序摘要。
- 每个 W&B panel/key 的含义。
- `p_best`、`p_sampling_*`、`numC_p_best_cutoff`、`numC_sampling_target`、`numC_sampling_cutoff`、`num_C`、`num_P` 的调试意义。
- source folder 与 task class 的区别。
- 本地/服务器训练产物位置：
  - Lightning/W&B run dir。
  - `validation_diagnostics/epoch_xxxxxx/summary.json`。
  - `curves/*.csv`。
  - `warnings.json`。
  - checkpoint 文件与推理加载关系。
- 常见异常排查：
  - `val_uncapped/best` 高但 `val_capped/recall` 低。
  - `val_capped/recall` 高但 `val_refined/F1` 低。
  - `val_unrefined/F1` 高于 `val_refined/F1`。
  - 某 source folder 无正例。
  - W&B 有 global 但缺 by_source_folder。

### 10.8 新增用户说明文档

路径：

```text
docs/model/stage1_wrapper_cpc_guide.md
```

内容写给用户，不贴代码，不拘泥实现细节。按时间顺序解释：

1. dense voxel 分支先在整张空间中给每个 voxel 打 ligand 概率。
2. `val_uncapped/best` 衡量 dense 分支理论上能切多好。
3. `val_uncapped/sampling` 衡量采样策略实际打算怎样切 C。
4. `val_capped` 衡量 cap 后真正进入 C 的候选是否覆盖正例。
5. C 内候选被送入 P/point/refine 相关流程。
6. `val_unrefined` 衡量不经过 refine 时 C 内 dense logit 的局部判断质量。
7. `val_refined` 衡量经过 sparse refine 后 C 内判断是否改善。
8. `val_score` 与 `val_curve` 如何阅读。
9. global 与 by_source_folder 的区别。
10. 二分类与多分类日志名为什么不同。

重点解释实际意义：这些指标如何帮助判断问题发生在 dense 可分性、采样、cap、C 内局部判定，还是 refine 分支。

## 11. 配置修改计划

在模型或训练配置中新增：

```yaml
validation_diagnostics:
  enabled: true
  source_folder_breakdown: true
  source_folder_names: ${dataset.class_folder_names}
  source_folder_meta_key: class_name
  num_bins: 1024
  write_local_artifacts: true
  log_wandb_curves: true
  wandb_curve_every_n_validation: 1
  output_subdir: validation_diagnostics
```

保留并检查：

```yaml
val_metric_device_policy: auto
```

需要全局搜索并更新旧 monitor metric，例如：

```text
val/...
val_score/voxel_ligand_pr_auc
val_pc/candidate_recall
...macro
...aux...
```

更新为新命名，例如：

```text
val_score/global/voxel_ligand_PRAUC
val_capped/global/recall
val_refined/global/F1
val_loss/receptor_loss
```

具体 monitor 选择建议：

- 主 checkpoint monitor 继续使用 `val_score/global/voxel_ligand_PRAUC` 或 `val_refined/global/F1`，二选一由当前训练目标决定。
- 采样质量 dashboard 使用 `val_capped/global/recall`、`val_uncapped/sampling/global/p_sampling_p50`。
- refine 改善 dashboard 同时看 `val_unrefined/global/F1` 与 `val_refined/global/F1`。

## 12. 测试计划

### 12.1 单元测试

新增或更新：

```text
tests/test_voxel_ligand_thresholds.py
tests/test_ligand_sparse_refine_metrics.py
tests/test_multiclass_ligand_wrapper.py
tests/test_voxel_point_stage1_cpc_diagnostics.py
tests/test_voxel_point_stage1_metric_logging.py
```

覆盖：

- 二分类不输出 `*_macro`。
- 二分类 key 不带 task class suffix。
- 多分类 key 带 task class suffix。
- `PRAUC`、`PRcurve`、`F1` 大小写符合规范。
- `aux` 对外日志改为 `receptor`。
- `SourceFolderRegistry` 能编码合法 source folder。
- 未知 source folder fail-fast。
- 某 source folder 无正例时不抛训练中断异常，只生成 warning/nan。
- `val_uncapped/best` 可计算 `p_best`、`numC_p_best_cutoff`。
- `val_uncapped/sampling` 对 recorded/adaptive/topk 都输出 `p_sampling_p5/p50/p75/p95/mean`。
- `val_capped` 中 `tp/fp/fn/precision/recall/F1` 按 global dense 定义计算。
- `val_unrefined` 与 `val_refined` 中 `fn` 只按 C 内 local 定义计算。
- W&B curve key 稳定，不随 epoch 改名。

### 12.2 DDP/同步测试

在可用环境下增加轻量 DDP smoke test，或至少用模拟两个 shard 的方式测试固定 tensor 合并：

- rank A 缺少某 source folder。
- rank B 包含该 source folder。
- 聚合后形状一致，结果正确。
- 无正例分组生成 warning 而非 RuntimeError。

### 12.3 metric device policy 测试

基于 `docs/验证时的metric计算问题.md` 的先例新增断言：

- `thresholds=None` AP 在 `auto` 下落 CPU。
- binned AP 在 `auto` 下落 GPU 或当前训练 device。
- batch size tuner 标志存在时仍执行 AP update/compute。
- `ValidationMetricManager` 是 `nn.Module`，其 metric 出现在 `state_dict()` 或模块树中。

### 12.4 推理兼容测试

更新或新增：

```text
tests/inference/test_get_voxel_pred.py
```

确认：

- 旧 checkpoint 中的 `voxel_logits_aux` / `voxel_aux_*` 相关权重仍可加载。
- 推理输出 key 映射不因日志改名而变化。
- 不要求旧 W&B metric 名存在。

## 13. 实施顺序

### 阶段一：建立 key 与配置边界

1. 新增 logging helper。
2. 新增 `validation_diagnostics` 配置读取。
3. 全局替换对外 `aux` 日志为 `receptor`。
4. 移除二分类 `*_macro` 输出。
5. 更新 monitor metric 引用。
6. 先跑现有测试，保证没有行为漂移。

### 阶段二：迁移常规 metrics

1. 新增 `ValidationMetricManager(nn.Module)`。
2. 把 AP/F1/Precision/Recall/Accuracy 的注册、update、compute/reset 从 wrapper 迁入 manager。
3. 保留 `val_metric_device_policy`。
4. 添加模块树与 state_dict 测试。

### 阶段三：实现 CPC diagnostics

1. 新增 `SourceFolderRegistry` 与静态 buffer。
2. 实现 `val_uncapped/best`。
3. 实现 `val_uncapped/sampling`，统一 recorded/adaptive/topk 的 `p_sampling_*`。
4. 实现 `val_capped`。
5. 实现 `val_unrefined`。
6. 实现 `val_refined`。
7. 添加 global 与 by_source_folder 输出。
8. 添加 warnings 与 no-positive 处理。

### 阶段四：W&B 与本地 artifact

1. 标量每轮记录。
2. PRcurve 使用稳定 W&B key 每轮记录。
3. 写入 `<run_dir>/validation_diagnostics/epoch_xxxxxx/`。
4. 非 W&B logger 下仍写本地文件。
5. 检查 30 次 validation 规模下文件数量与 W&B key 数量。

### 阶段五：拆分 loss 与 scheduler

1. 把 loss 计算迁入 `voxel_point_stage1_losses.py`。
2. 把 scheduler/optimizer 长逻辑迁入 `voxel_point_stage1_scheduler.py`。
3. wrapper 保持时间顺序主流程，减少细节噪声。

### 阶段六：文档与最终验证

1. 写 `CLAUDE/docs/wrapper_metrics_reference_ai.md`。
2. 写 `docs/model/stage1_wrapper_cpc_guide.md`。
3. 更新 README/CLAUDE 相关索引，如项目已有索引要求。
4. 跑单元测试、关键 smoke test、推理兼容测试。
5. 进行一次小验证集 dry run，检查 W&B key、local artifacts、warning 输出。

## 14. 验收标准

代码结构：

- `voxel_point_stage1.py` 明显变短，主流程可按训练/验证时间顺序阅读。
- metrics、diagnostics、loss、logging、scheduler 的职责边界清晰。
- TorchMetrics 不隐藏在普通 Python 类中。

日志：

- 二分类无 `*_macro`。
- 对外日志无 `aux`，使用 `receptor`。
- `PRAUC`、`PRcurve`、`F1` 命名统一。
- `val_uncapped/best`、`val_uncapped/sampling`、`val_capped`、`val_unrefined`、`val_refined` 都有 global 指标。
- source folder 分组独立 W&B key，同时保留 global。
- PRcurve table key 稳定，可在 W&B step 历史中翻阅。

诊断：

- `p_best` 与 `p_sampling_*` 并行存在，语义清楚。
- recorded/adaptive/topk 三种候选策略都能输出统一 `p_sampling_p5/p50/p75/p95/mean`。
- global 阶段与 local 阶段的 TP/FP/FN 定义不混淆。
- 无正例分组不导致 DDP 死锁式中断。

文档：

- AI-facing 文档能让 agent 快速定位服务器训练产物并解释指标异常。
- 用户文档能按时间顺序解释 C/P/C 架构实际意义与 W&B 指标阅读方式。

兼容：

- 旧 checkpoint 仍可用于推理。
- 旧 W&B metric 名不再保证兼容。

## 15. 主要风险与缓解

### 风险一：W&B UI key 数量过多

缓解：

- 默认保留当前最详细粒度，因为当前约 30 次 validation 且 source folder 数量有限。
- 配置保留 `source_folder_breakdown`、`log_wandb_curves`、`wandb_curve_every_n_validation`，未来可降采样。
- 本地 CSV/JSON 始终保存完整诊断，W&B 可按需减少。

### 风险二：C/P/C 阶段定义被误读

缓解：

- key 上显式区分 `uncapped`、`capped`、`unrefined`、`refined`。
- 文档强调前两个是 global，后两个是 local。
- `candidate_recall` 这类容易混淆的旧名不再作为主指标名。

### 风险三：source folder 与 task class 混淆

缓解：

- 使用 `by_source_folder` 命名，不使用 `by_class`。
- source folder 放路径层级，task class 放 metric suffix。
- 未知 source folder 直接报错。

### 风险四：metric device 策略回退

缓解：

- `ValidationMetricManager` 中显式实现 `val_metric_device_policy`。
- 测试覆盖 binned/non-binned AP 的设备选择。
- 参考 `docs/验证时的metric计算问题.md`，不跳过 batch size tuner 验证 AP。

### 风险五：重构范围大导致行为漂移

缓解：

- 先迁移 key/logging 与 tests，再迁移 metrics，再迁移 diagnostics。
- 每阶段保持小步测试。
- loss/scheduler 最后做等价迁移。
- 推理兼容测试单独保护 checkpoint 边界。
