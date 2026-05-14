# 体素分支三分类重构新版实施计划

## 目标

把当前 Stage1 体素相关任务从“只支持二分类配置”重构为“二分类与多分类并存”的实现：

* 保留现有二分类训练与推理逻辑：单通道 logits 使用 sigmoid，`ligand_dist_map` 仍保持当前单通道 `(D, H, W)` 监督，不强行改成 `(2, D, H, W)`。
* 新增三分类训练与推理逻辑：三通道 logits 使用 softmax。
* 三分类实验的类别映射为 `class_mapping=[0, 1, 0, 0, 2]`：
  * 原始 `0 background -> 0 background`
  * 原始 `1 metal_ion -> 1 metal_ion`
  * 原始 `2 peptide -> 0 background`
  * 原始 `3 nucleic -> 0 background`
  * 原始 `4 small_molecule -> 2 small_molecule`
* `peptide` 与 `nucleic` 映射到背景负类参与训练，不引入 ignore mask。
* `atom`、`voxel_aux`、`voxel_ligand` 三条监督路径都支持三分类。
* 推理管线根据最终 logits 通道数与 `class_mapping` 自动决定二分类或多分类处理方式。
* 多分类推理输出每个前景类别概率图，并允许 `metal_ion` 与 `small_molecule` 使用不同后处理参数。
* 新增一套三分类训练配置，放在 [configs/experiment/](../../configs/experiment/) 根目录，命名为 `exp000_baseline`，数据集使用 [configs/dataset/stardard\_full\_buffer.yaml](../../configs/dataset/stardard_full_buffer.yaml)。

## 非目标

* 不重新生成或重定义上游 `Make_Data/labels.npz` 的原始五类语义。
* 不修改 `processedPDB_EMDB_binder` 已有 BOX 文件夹命名规则。
* 不把 `peptide/nucleic` 从 loss 中 ignore；它们按背景处理。
* 不删除现有二分类配置、二分类 loss、二分类推理输出路径。
* 不把二分类 `ligand_dist_map` 改为二通道格式。

## 关键现状

* [Make\_Data/notes\_of\_dataset.md](../../Make_Data/notes_of_dataset.md) 说明上游 `labels.npz` 保存五类口袋标签。
* [processedPDB\_EMDB\_binder/notes\_of\_dataset.md](../../processedPDB_EMDB_binder/notes_of_dataset.md) 说明 `pdb_label_BOX` 保存原始体素标签，`ligand_dist_BOX` 保存四个原始前景类别距离通道。
* [具体工作的细节.md](../../具体工作的细节.md) 说明 Stage1 当前三条监督路径为 `atom_loss`、`voxel_aux_loss`、`voxel_ligand_loss`。
* [src/datasets/box\_point\_dataset.py](../../src/datasets/box_point_dataset.py) 当前已对 `voxel_label` 和 `atom_label` 应用 `class_mapping`，但 `ligand_dist_map` 会把所有映射后前景通道取 `min`，只生成单通道距离监督。
* [src/model/stage1\_voxel\_backbone.py](../../src/model/stage1_voxel_backbone.py) 当前 `voxel_aux_head` 和 `voxel_ligand_head` 输出固定为 1 通道。
* [src/wrappers/voxel\_point\_stage1.py](../../src/wrappers/voxel_point_stage1.py) 当前 loss 与 metric 主要按 sigmoid 二分类处理。
* [src/modules/losses.py](../../src/modules/losses.py) 当前 `UnifiedCompositeLoss` 是二分类复合 loss；已有 `MultiClassFocalLossWithAlpha`，但还没有覆盖三条 Stage1 监督的统一多分类复合入口。
* [src/inference/get\_pred.py](../../src/inference/get_pred.py) 当前固定使用 `torch.sigmoid(logits[:, 0])` 把 BOX logits 转概率。
* [configs/experiment/exp003\_stardardligand3.yaml](../../configs/experiment/exp003_stardardligand3.yaml) 是本次三分类 baseline 配置的主要模仿对象，但新配置应命名为 `exp000_baseline`，并使用 `stardard_full_buffer` 数据集。

## 实施步骤

### 1. 增加显式任务类别配置

修改 dataset 配置：

* [configs/dataset/stardard\_full.yaml](../../configs/dataset/stardard_full.yaml)
* [configs/dataset/stardard\_full\_buffer.yaml](../../configs/dataset/stardard_full_buffer.yaml)
* [configs/dataset/L\_small\_full.yaml](../../configs/dataset/L_small_full.yaml)
* [configs/dataset/L\_small\_full\_buffer.yaml](../../configs/dataset/L_small_full_buffer.yaml)
* [configs/dataset/L\_small\_full\_buffermargin.yaml](../../configs/dataset/L_small_full_buffermargin.yaml)
* [configs/dataset/noemdb\_small.yaml](../../configs/dataset/noemdb_small.yaml)

保留现有二分类配置的默认语义，不直接把所有 dataset yaml 都改成三分类。新增字段时应兼容旧配置：

```yaml
class_mapping: [0, 1, 1, 1, 1]
class_names: [background, foreground]
num_task_classes: 2
```

三分类实验配置中覆盖为：

```yaml
class_mapping: [0, 1, 0, 0, 2]
class_names: [background, metal_ion, small_molecule]
num_task_classes: 3
```

模型配置增加或显式传递：

```yaml
model:
  num_task_classes: 3
  task_activation: softmax
  backbone:
    atom_logit_dim: 3
    voxel_aux_logit_dim: 3
    voxel_ligand_logit_dim: 3
    prior_probs: [0.98, 0.01, 0.01]
```

约束：

* `logit_dim == 1` 表示二分类 sigmoid 路径。
* `logit_dim > 1` 表示多分类 softmax 路径。
* `class_mapping` 的最大值必须等于 `num_task_classes - 1`。
* 多分类时 `class_names` 长度必须等于 `num_task_classes`。
* 训练、验证、推理都通过 logits 通道数与 `class_mapping` 判断任务类型，不通过文件名判断。

### 2. 新增三分类 baseline 配置

新增文件：

* [configs/experiment/exp000\_baseline.yaml](../../configs/experiment/exp000_baseline.yaml)

配置应模仿 [configs/experiment/exp003\_stardardligand3.yaml](../../configs/experiment/exp003_stardardligand3.yaml)，但做以下调整：

```yaml
defaults:
  - override /model/voxel_backbone: default
  - override /model/point_backbone: zeros
  - override /model/fusion: none
  - override /model/atom_head: none
  - override /model/pseudo_atom: none
  - override /model/embed_head: clipbig
  - override /loss: tri_7focal_3dice_hard
  - override /dataset: stardard_full_buffer
  - override /train: B40_L3
```

实验标识建议：

```yaml
name: exp000_baseline
tag: "exp000_baseline"
experiment_group: "tri_ligand"
project_name: PV_ligand_tri
```

三分类覆盖字段：

```yaml
dataset:
  class_mapping: [0, 1, 0, 0, 2]
  class_names: [background, metal_ion, small_molecule]
  num_task_classes: 3

model:
  num_task_classes: 3
  task_activation: softmax
  backbone:
    enable_atom_head: false
    atom_logit_dim: 3
    prior_probs: [0.98, 0.01, 0.01]
    voxel_backbone:
      aux_head_hidden_channels: 32
      ligand_head_hidden_channels: 32
      num_conv3d_aux: 2
      num_conv3d_ligand: 2
      voxel_aux_logit_dim: 3
      voxel_ligand_logit_dim: 3
  monitor_metric: val/voxel_ligand_macro_ap
  monitor_mode: max
```

注意：虽然 baseline 关闭 atom head，但底层代码仍要支持 `atom_logit_dim=3`，供后续打开 atom head 时复用。

### 3. 数据集：保持二分类 ligand\_dist 单通道，新增多分类 ligand\_dist 多通道

修改 [src/datasets/box\_point\_dataset.py](../../src/datasets/box_point_dataset.py)。

当前行为：

* `ligand_dist_BOX` 原始形状为 `(4, D, H, W)`。
* 通道 `0..3` 对应原始 class id `1..4`。
* 当前会选取所有 `class_mapping[old_class_id] > 0` 的原始通道并取 `min`，得到单通道 `(D, H, W)`。

目标行为：

#### 二分类

当 `num_task_classes == 2` 或输出 head 为单通道时：

* 维持现有行为。
* `ligand_dist_map` 仍为 `(D, H, W)`。
* 不改成 `(2, D, H, W)`。

#### 多分类

当 `num_task_classes > 2` 时：

* `ligand_dist_map` 输出为 `(num_task_classes, D, H, W)`。
* 背景通道 `0` 填 `np.inf`，仅作为占位，loss 里不把它当真实 ligand 距离。
* 对每个原始前景类别：
  * 原始 class id `1` 使用 `ligand_dist_raw[0]`，映射到新 class id `1`。
  * 原始 class id `2` 使用 `ligand_dist_raw[1]`，映射到新 class id `0`，不写入前景距离通道。
  * 原始 class id `3` 使用 `ligand_dist_raw[2]`，映射到新 class id `0`，不写入前景距离通道。
  * 原始 class id `4` 使用 `ligand_dist_raw[3]`，映射到新 class id `2`。
* 如果多个原始类别映射到同一个新前景类别，对对应距离图取 `min`。

建议新增函数：

```python
def _build_ligand_dist_map(
    ligand_dist_raw: np.ndarray,
    class_mapping: list[int] | None,
    num_task_classes: int,
) -> np.ndarray:
    """二分类返回 (D,H,W)，多分类返回 (C,D,H,W)。"""
```

同步修改：

* [src/datasets/box\_point\_collate.py](../../src/datasets/box_point_collate.py)：更新 `ligand_dist_map` 可为 `(B,D,H,W)` 或 `(B,C,D,H,W)` 的说明。
* [src/datasets/box\_sample\_builder.py](../../src/datasets/box_sample_builder.py)：更新字段说明。
* [src/datasets/box\_point\_dataset.py](../../src/datasets/box_point_dataset.py) 的随机旋转：
  * 3D `ligand_dist_map` 按原逻辑旋转。
  * 4D `ligand_dist_map` 只旋转空间轴，类别轴不动。

### 4. Voxel backbone 输出通道与多类先验初始化

修改 [src/model/stage1\_voxel\_backbone.py](../../src/model/stage1_voxel_backbone.py)。

新增参数：

```python
voxel_aux_logit_dim: int = 1
voxel_ligand_logit_dim: int = 1
prior_probs: Sequence[float] | None = None
```

将 head 最后一层从固定 1 通道改为配置通道数：

```python
nn.Conv3d(_last_in, int(voxel_aux_logit_dim), kernel_size=1)
nn.Conv3d(_last_in_lig, int(voxel_ligand_logit_dim), kernel_size=1)
```

先验初始化规则：

#### 二分类 sigmoid

保留现有 `prior_prob`：

```python
bias = -log((1 - prior_prob) / prior_prob)
```

适用条件：

* head 输出通道为 1。
* 配置传入 `prior_prob`。

#### 多分类 softmax

新增 `prior_probs`，例如：

```yaml
prior_probs: [0.98, 0.01, 0.01]
```

含义：初始化 logits bias 后，`softmax(bias)` 应得到该概率分布。

实现：

```python
bias_i = log(prior_probs[i])
```

因为 softmax 对整体平移不敏感，直接使用 `log(p_i)` 即可满足：

```python
softmax(log([0.98, 0.01, 0.01])) == [0.98, 0.01, 0.01]
```

fail-fast 校验：

* `len(prior_probs) == logit_dim`。
* `sum(prior_probs)` 接近 1。
* 所有概率必须大于 0。
* 如果 head 是多通道但只传 `prior_prob`，报错或明确忽略并提示配置错误；不要静默套用二分类 bias。

`aux_head_hidden_channels <= 0` 的消融路径：

* 要求 `feature_channels == voxel_aux_logit_dim`，否则直接报错。

### 5.:comment[Atom head 输出与先验初始化]{#comment-1778766059813 text="先验初始化要同时作用于三个损失（ atom head 对受体原子的分类；体素分支对受体体素、ligand区域的损失）"}

修改 [src/model/stage1\_model.py](../../src/model/stage1_model.py)。

当前 `atom_logit_dim` 已存在。

目标：

* 二分类：`atom_logit_dim=1`，沿用 sigmoid 与 `prior_prob` 初始化。
* 多分类：`atom_logit_dim=3`，使用 `prior_probs=[0.98,0.01,0.01]` 初始化 `atom_logit_head` 最后一层 bias，使 softmax 后概率符合该向量。
* 文档从“二分类 logit 维度”改为“分类 logits 维度；1 表示二分类 sigmoid，多于 1 表示 softmax 多分类”。
* 如果 `enable_atom_head=false`，允许配置中保留 `atom_logit_dim=3`，但不构建 atom head。

### 6. Loss：新增同时适配二分类与多分类的分类复合 loss

修改 [src/modules/losses.py](../../src/modules/losses.py)。

新增模块，例如：

```python
class AdaptiveClassificationCompositeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hard_label_threshold: float | None,
        focal_gamma: float,
        focal_alpha: Sequence[float] | None,
        tversky_alpha: float,
        tversky_beta: float,
        tversky_smooth: float,
        w_focal: float,
        w_tversky: float,
        w_mse: float = 0.0,
    ) -> None:
        ...
```

行为：

#### 二分类路径

条件：`logits.shape[1] == 1`。

* 使用 sigmoid。
* target 为 0/1。
* `ligand_dist_map` 必须为 `(B,D,H,W)` 或等价展平形状。
* 复用现有 `UnifiedCompositeLoss` 的核心实现或保持完全等价行为。

#### 多分类路径

条件：`logits.shape[1] > 1`。

* 使用 softmax。
* `target` 为 class id，形状为 `(N,)` 或 `(B,D,H,W)`。
* `voxel_aux_loss` 使用 `voxel_label` 作为 target。
* `atom_loss` 使用 `atom_label` 作为 target。
* `voxel_ligand_loss` 从多通道 `ligand_dist_map` 生成 target：
  1. 忽略背景距离通道。
  2. 对前景类别找最小距离类别。
  3. 若最小距离 `< hard_label_threshold`，target 为该类别 id。
  4. 否则 target 为 0。
* :comment[ocal 可复用 MultiClassFocalLossWithAlpha 或实现等价的 per-class focal。]{#comment-1778766149717 text="不要服用 MultiClassFocalLossWithAlpha 了（它的很多参数比如自适应损失，是绝不会使用的）。直接重新写，并且直接写成hard_loss,不要写类似二分类focal loss 的平滑逻辑（因为用不到）"}
* :comment[Tversky/Dice 对 one-hot 后的前景类计算，默认排除背景类，对前景类平均。]{#comment-1778766195957 text="你需要仔细查阅相关论文与资料：多分类时 dice loss 如何作相应改动？"}
* `hardmask` 和 `valid_mask` 只控制有效位置，不改变类别。

配置文件：

新增 [configs/loss/tri\_7focal\_3dice\_hard.yaml](../../configs/loss/tri_7focal_3dice_hard.yaml)。

该配置应包含：

* `voxel_aux_loss` 使用新 adaptive loss，`num_classes: 3`。
* `voxel_ligand_loss` 使用新 adaptive loss，`num_classes: 3`，`hard_label_threshold: 1.7`。
* 如果 atom head 默认关闭，可保留 atom loss 配置为注释或 `null`；但代码路径应支持三分类 atom loss。

### 7. Wrapper：按 logits 通道数选择 sigmoid 或 softmax

修改 [src/wrappers/voxel\_point\_stage1.py](../../src/wrappers/voxel_point_stage1.py)。

#### Loss 路径

* `_compute_atom_loss`：
  * 单通道 logits 走二分类 target。
  * 多通道 logits 走 class id target。
* `_compute_voxel_aux_loss`：
  * 单通道 logits 接受二分类 `voxel_label`。
  * 多通道 logits 接受多分类 `voxel_label`。
* `_compute_voxel_ligand_loss`：
  * 单通道 logits 接受单通道 `ligand_dist_map`。
  * 多通道 logits 接受多通道 `ligand_dist_map`。

#### Metric 路径

保留二分类 metric：

* `val/atom_pr_auc`
* `val/voxel_aux_pr_auc`
* `val/voxel_ligand_pr_auc`

新增多分类 metric：

* `val/atom_ap_metal_ion`
* `val/atom_ap_small_molecule`
* `val/atom_macro_ap`
* `val/voxel_aux_ap_metal_ion`
* `val/voxel_aux_ap_small_molecule`
* `val/voxel_aux_macro_ap`
* `val/voxel_ligand_ap_metal_ion`
* `val/voxel_ligand_ap_small_molecule`
* `val/voxel_ligand_macro_ap`

多分类概率计算：

```python
prob = torch.softmax(logits, dim=1)
```

每个前景类 AP：

* 预测为该类 softmax 概率。
* target 为 `target == class_id`。
* 背景不作为前景 AP 计算。

:comment[monitor\_metric：]{#comment-1778766265044 text="训练时，让wandb画出每个通道的PR-AUC"}

* 二分类配置继续使用现有 PR-AUC。
* 三分类 baseline 使用 `val/voxel_ligand_macro_ap`。
* 如果配置的 monitor metric 没有被当前任务创建，应 fail-fast。

### 8. 推理：保留二分类路径，新增多分类路径

修改：

* [src/inference/get\_pred.py](../../src/inference/get_pred.py)
* [src/inference/main/voxel\_pipeline.py](../../src/inference/main/voxel_pipeline.py)
* [src/inference/voxel\_postprocess.py](../../src/inference/voxel_postprocess.py)
* [src/inference/utils/utils.py](../../src/inference/utils/utils.py)
* [configs/infer\_or\_eval/](../../configs/infer_or_eval/)

#### logits 转概率

```python
if logits.shape[1] == 1:
    probs = torch.sigmoid(logits[:, 0])          # (B,D,H,W)
else:
    probs = torch.softmax(logits, dim=1)         # (B,C,D,H,W)
```

#### 二分类输出保持不变

二分类时继续输出当前字段：

* `ligand_pred: (D,H,W)`
* `receptor_pred: (D,H,W)`
* `density_pred_prob.map`
* 当前单图后处理结果

#### 多分类输出新增字段

多分类时输出：

```python
ligand_pred: np.ndarray      # (C,D,H,W)
receptor_pred: np.ndarray    # (C,D,H,W)
class_names: list[str]
class_mapping: list[int]
```

保存时按前景类别拆分：

```text
pred/metal_ion/density_pred_prob.map
pred/metal_ion/postprocessed.map
pred/small_molecule/density_pred_prob.map
pred/small_molecule/postprocessed.map
```

背景通道不做前景后处理。

#### hardmask 约束

* receptor head 的前景类别概率乘 `hardmask`。
* ligand head 的前景类别概率乘 `1-hardmask`。
* 背景通道可保留原 softmax 概率用于调试，但不参与前景输出和后处理。

#### 多分类 BOX 合并

现有 `_merge_box_probability_into_full` 可继续用于单个 `(D,H,W)` 概率 patch。

新增外层逻辑：

* 二分类：按当前逻辑调用一次。
* 多分类：对每个类别通道分别调用。

合并输出形状：

* 二分类：`(D,H,W)`。
* 多分类：`(C,D,H,W)`。

### 9. 每类后处理参数

修改推理配置 [configs/infer\_or\_eval/](../../configs/infer_or_eval/) 与 [src/inference/main/voxel\_pipeline.py](../../src/inference/main/voxel_pipeline.py)。

新增配置结构示例：

```yaml
postprocess:
  default:
    threshold: 0.5
    min_component_size: 10
    gaussian_sigma: 1.0
  by_class:
    metal_ion:
      threshold: 0.3
      min_component_size: 1
    small_molecule:
      threshold: 0.5
      min_component_size: 10
```

规则：

* 如果 `by_class[class_name]` 存在，使用类别专属参数。
* 否则使用 `default`。
* 二分类配置继续兼容旧的平铺 `threshold` 等字段。
* 不要让 metal 与 small molecule 被迫共用同一阈值。

### 10. GT 与可视化同步

修改：

* [src/inference/voxel\_gt.py](../../src/inference/voxel_gt.py)
* [src/inference/parse\_input.py](../../src/inference/parse_input.py)
* [src/inference/utils/utils.py](../../src/inference/utils/utils.py)

目标：

* 二分类时保持当前 GT 与可视化输出。
* 多分类时按 `class_mapping` 生成每个前景类别 GT：
  * `metal_ion` GT。
  * `small_molecule` GT。
* `peptide/nucleic` 因映射为 0，不输出为前景 GT。
* 可视化目录与预测目录按类别对齐。

### 11. 文档同步

修改：

* [具体工作的细节.md](../../具体工作的细节.md)
* [processedPDB\_EMDB\_binder/notes\_of\_dataset.md](../../processedPDB_EMDB_binder/notes_of_dataset.md)

更新内容：

* Stage1 支持二分类 sigmoid 与多分类 softmax 两种任务形态。
* `class_mapping=[0,1,0,0,2]` 的三分类语义。
* `ligand_dist_map` 的二分类与多分类形状差异：
  * 二分类：`(D,H,W)`。
  * 多分类：`(C,D,H,W)`。
* 多分类推理按前景类别输出概率图和后处理结果。
* processed 数据本身不需要重生成，训练读取时完成类别折叠。

如果只改下游训练/推理，不改上游 `Make_Data` 落盘字段，则 [Make\_Data/notes\_of\_dataset.md](../../Make_Data/notes_of_dataset.md) 不需要同步修改字段定义。

## 验证计划

### Dataset 验证

1. 二分类配置：
   * `class_mapping=[0,1,1,1,1]`。
   * `voxel_label` 只包含 `{0,1}`。
   * `atom_label` 只包含 `{0,1}`。
   * `ligand_dist_map.shape == (D,H,W)`。
2. 三分类配置：
   * `class_mapping=[0,1,0,0,2]`。
   * `voxel_label` 只包含 `{0,1,2}`。
   * `atom_label` 只包含 `{0,1,2}`。
   * `ligand_dist_map.shape == (3,D,H,W)`。
   * `ligand_dist_map[0]` 为 `inf` 或背景占位。
   * `ligand_dist_map[1]` 来自 metal 原始通道。
   * `ligand_dist_map[2]` 来自 small molecule 原始通道。
3. 旋转增强：
   * 二分类 3D 距离图旋转结果 shape 不变。
   * 三分类 4D 距离图只旋转空间轴，类别轴不变。

### Model forward 验证

1. 二分类：
   * `atom_logits: (N,1)` 或 atom head 关闭。
   * `voxel_logits_aux: (B,1,D,H,W)`。
   * `voxel_logits_ligand: (B,1,D,H,W)`。
2. 三分类：
   * `atom_logits: (N,3)` 或 atom head 关闭。
   * `voxel_logits_aux: (B,3,D,H,W)`。
   * `voxel_logits_ligand: (B,3,D,H,W)`。
3. 多分类先验：
   * 初始化后取 head 最后一层 bias。
   * 验证 `softmax(bias) ≈ [0.98,0.01,0.01]`。

### Loss 验证

1. 二分类配置仍走 sigmoid，loss 与旧实现数值行为一致或在可接受范围内一致。
2. 三分类配置走 softmax。
3. `voxel_ligand_loss` 能从 `(B,3,D,H,W)` 距离图生成 class id target。
4. `peptide/nucleic` 区域在 target 中为背景 0。
5. mask 只影响有效位置，不改变类别。

### 训练 smoke test

用 [configs/experiment/exp000\_baseline.yaml](../../configs/experiment/exp000_baseline.yaml) 跑极小 batch：

* forward 成功。
* loss 非 NaN。
* backward 成功。
* validation 正常记录：
  * `val/voxel_ligand_ap_metal_ion`
  * `val/voxel_ligand_ap_small_molecule`
  * `val/voxel_ligand_macro_ap`

### 推理 smoke test

1. 二分类 checkpoint：
   * 输出字段和目录保持旧行为。
   * `density_pred_prob.map` 仍存在。
2. 三分类 checkpoint：
   * 输出 `metal_ion` 概率图。
   * 输出 `small_molecule` 概率图。
   * 两类分别使用自己的后处理参数。
   * 背景通道不做前景后处理。
   * GT 可视化按类别输出。

## 风险点

1. `ligand_dist_map` 在二分类和三分类下 shape 不同，所有消费方必须显式分支，不能隐式 squeeze。
2. 多分类 softmax 先验初始化必须用 `log(prior_probs)`，目标是 softmax 后概率等于配置向量。
3. 旧二分类推理路径不能被多分类改造破坏。
4. `BinaryAveragePrecision` 不能直接用于多分类整体 logits；多分类必须按前景类别分别算 AP。
5. `monitor_metric` 需要随任务类型切换，否则三分类 baseline 可能找不到监控指标。
6. 每类后处理参数不同，配置读取时不能只取全局 threshold。
7. 三分类 baseline 使用 `stardard_full_buffer`，不要误用 `stardard_medium`。