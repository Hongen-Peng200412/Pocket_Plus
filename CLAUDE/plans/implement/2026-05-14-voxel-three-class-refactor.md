# 体素分支三分类重构实施计划

## 目标

把当前体素分支从二分类任务扩展为三分类任务：

- 原始上游五类标签仍为：`0=background, 1=metal_ion, 2=peptide, 3=nucleic, 4=small_molecule`。
- 新训练映射为：`class_mapping=[0, 1, 0, 0, 2]`。
- 新任务类别为：
  - `0=background`
  - `1=metal_ion`
  - `2=small_molecule`
- `peptide` 与 `nucleic` 映射到 0，作为背景负类参与训练，不额外引入 ignore mask。
- 二分类输出使用 sigmoid；多分类输出使用 softmax。
- `atom`、`voxel_aux`、`voxel_ligand` 三条监督路径都支持三分类。
- 推理阶段输出每个前景类别概率图，并分别对 `metal_ion` 和 `small_molecule` 做后处理。

## 非目标

- 不重新定义上游 `Make_Data/labels.npz` 的五类语义。
- 不修改 `processedPDB_EMDB_binder` 已有 BOX 文件夹命名规则。
- 不把 `peptide/nucleic` 从 loss 中 ignore；它们按背景处理。
- 不引入兼容旧 checkpoint 的权重形状 shim；输出通道变化后旧 checkpoint 不应直接加载到新三分类 head。

## 现状依据

- [Make_Data/notes_of_dataset.md](../../Make_Data/notes_of_dataset.md) 说明上游 `labels.npz` 保留五类点云标签。
- [processedPDB_EMDB_binder/notes_of_dataset.md](../../processedPDB_EMDB_binder/notes_of_dataset.md) 说明体素标签和 ligand 距离图来自上游五类，并在 BOX 阶段保存。
- [具体工作的细节.md](../../具体工作的细节.md) 说明 Stage1 三条监督：`atom_loss`、`voxel_aux_loss`、`voxel_ligand_loss`。
- [src/datasets/box_point_dataset.py](../../src/datasets/box_point_dataset.py) 当前已对 `voxel_label` 和 `atom_label` 应用 `class_mapping`，但 `ligand_dist_map` 当前会把映射到前景的原始通道取 `min`，只生成单通道监督。
- [src/model/stage1_voxel_backbone.py](../../src/model/stage1_voxel_backbone.py) 当前 `voxel_aux_head` 和 `voxel_ligand_head` 输出固定为 1 通道。
- [src/wrappers/voxel_point_stage1.py](../../src/wrappers/voxel_point_stage1.py) 当前 metric 和 loss 路径按二分类 sigmoid 处理。
- [src/modules/losses.py](../../src/modules/losses.py) 已有二分类复合 loss 和多分类 focal loss，但当前统一复合 loss 是二分类形状契约。
- [src/inference/get_pred.py](../../src/inference/get_pred.py) 当前用 `torch.sigmoid(logits[:, 0])` 合并 BOX 概率。

## 实施步骤

### 1. 配置层增加任务类别控制面

修改配置文件：

- [configs/dataset/stardard_full.yaml](../../configs/dataset/stardard_full.yaml)
- [configs/dataset/stardard_full_buffer.yaml](../../configs/dataset/stardard_full_buffer.yaml)
- [configs/dataset/L_small_full.yaml](../../configs/dataset/L_small_full.yaml)
- [configs/dataset/L_small_full_buffer.yaml](../../configs/dataset/L_small_full_buffer.yaml)
- [configs/dataset/L_small_full_buffermargin.yaml](../../configs/dataset/L_small_full_buffermargin.yaml)
- [configs/dataset/noemdb_small.yaml](../../configs/dataset/noemdb_small.yaml)
- [configs/model/default.yaml](../../configs/model/default.yaml)
- 相关 [configs/experiment/](../../configs/experiment/) yaml。

新增或统一字段：

```yaml
class_mapping: [0, 1, 0, 0, 2]
class_names: [background, metal_ion, small_molecule]
num_task_classes: 3
```

模型配置新增或显式设置：

```yaml
model:
  num_task_classes: 3
  task_activation: softmax
  backbone:
    atom_logit_dim: 3
    voxel_aux_logit_dim: 3
    voxel_ligand_logit_dim: 3
```

保留二分类配置能力：

- 二分类可继续使用 `atom_logit_dim=1`、`voxel_aux_logit_dim=1`、`voxel_ligand_logit_dim=1`。
- 当输出通道为 1 时，训练与推理使用 sigmoid。
- 当输出通道大于 1 时，训练与推理使用 softmax。

### 2. 改造数据集中的 ligand 距离监督

修改 [src/datasets/box_point_dataset.py](../../src/datasets/box_point_dataset.py)。

当前逻辑：

- `ligand_dist_BOX` 原始形状为 `(4, D, H, W)`。
- 通道 `0..3` 对应原始 class id `1..4`。
- 当前根据 `class_mapping` 选择所有映射后 `>0` 的通道并取 `min`，得到 `(D, H, W)`。

目标逻辑：

- 对三分类任务生成多类距离图：`ligand_dist_map` 形状改为 `(num_task_classes, D, H, W)` 或只保存前景 `(num_task_classes-1, D, H, W)`，推荐使用完整 `(C, D, H, W)`，背景通道可填 `inf`。
- 对每个原始 class id：
  - `1 metal_ion -> 新 class 1`
  - `2 peptide -> 新 class 0`
  - `3 nucleic -> 新 class 0`
  - `4 small_molecule -> 新 class 2`
- 对映射到同一个新前景类别的多个原始通道取 `min`。
- 映射到背景的原始通道不生成前景距离监督。
- `peptide/nucleic` 不参与 ligand 前景类别监督。

建议新增内部函数：

```python
def _map_ligand_dist_channels(
    ligand_dist_raw: np.ndarray,
    class_mapping: list[int] | None,
    num_task_classes: int,
) -> np.ndarray:
    ...
```

输出契约：

- 二分类：可继续输出 `(D, H, W)`，也可以统一输出 `(2, D, H, W)`；若选择统一多类格式，需要同步 loss。
- 三分类：输出 `(3, D, H, W)`。

同步修改：

- [src/datasets/box_point_collate.py](../../src/datasets/box_point_collate.py) 中 `ligand_dist_map` 的 docstring 和 stack 形状说明。
- [src/datasets/box_sample_builder.py](../../src/datasets/box_sample_builder.py) 中相关字段说明。
- 旋转增强路径中，若 `ligand_dist_map` 为 4D `(C,D,H,W)`，只旋转空间轴 `(D,H,W)`，不旋转类别轴。

### 3. 改造 voxel backbone 输出通道

修改 [src/model/stage1_voxel_backbone.py](../../src/model/stage1_voxel_backbone.py)。

当前：

```python
aux_layers.append(nn.Conv3d(_last_in, 1, kernel_size=1))
ligand_layers.append(nn.Conv3d(_last_in_lig, 1, kernel_size=1))
```

目标：

- `__init__` 增加：

```python
voxel_aux_logit_dim: int = 1
voxel_ligand_logit_dim: int = 1
```

- 最后一层输出改为：

```python
nn.Conv3d(_last_in, int(voxel_aux_logit_dim), kernel_size=1)
nn.Conv3d(_last_in_lig, int(voxel_ligand_logit_dim), kernel_size=1)
```

- 文档和 shape 说明从固定 `(B,1,D,H,W)` 改为 `(B,C,D,H,W)`。
- `prior_prob` 偏置初始化只适用于单通道 sigmoid head；多类 softmax head 不使用同一正类先验初始化，避免给所有类别同一 bias 造成语义混乱。
- `aux_head_hidden_channels <= 0` 的消融路径要求 `feature_channels == voxel_aux_logit_dim`，否则 fail-fast。

### 4. 改造 atom head 输出配置

涉及：

- [src/model/stage1_model.py](../../src/model/stage1_model.py)
- [configs/model/default.yaml](../../configs/model/default.yaml)
- 各 experiment yaml。

当前 `atom_logit_dim` 已存在，默认是 1。

目标：

- 三分类实验中设为 `atom_logit_dim: 3`。
- `prior_prob` 初始化逻辑只在 `atom_logit_dim == 1` 时应用。
- 文档说明从“二分类 logit”改成“分类 logits，二分类可为单通道”。

### 5. 新增统一分类 loss 适配层

修改 [src/modules/losses.py](../../src/modules/losses.py)。

当前：

- `UnifiedCompositeLoss` 假定 logits 为 `(*,1,...)`，内部使用 sigmoid。
- `MultiClassFocalLossWithAlpha` 已有 softmax 多分类 focal，但没有直接覆盖三条现有 wrapper 入口。

目标：

新增一个明确的 loss 适配模块，例如：

```python
class ClassificationCompositeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hard_label_threshold: float | None,
        w_focal: float,
        w_tversky: float,
        ...,
    ) -> None: ...

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor | None = None,
        hardmask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        ligand_dist_map: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

行为：

- `logits.shape[1] == 1`：
  - 使用 sigmoid。
  - 复用当前 `UnifiedCompositeLoss` 二分类逻辑。
- `logits.shape[1] > 1`：
  - 使用 softmax。
  - `target` 为 `(B,D,H,W)` 或 `(N,)` 的 class id。
  - focal 使用 multi-class focal 或 `F.cross_entropy` 风格实现。
  - Dice/Tversky 按类别 one-hot 后计算，默认排除背景类，只对前景类求均值。
  - `hardmask`/`valid_mask` 只作为有效位置 mask，不改变类别 id。
- `voxel_ligand_loss` 多分类 target：
  - 从 `ligand_dist_map` 生成 class id target。
  - 对每个体素，在前景类别距离小于阈值的类别中选择最近类别。
  - 若所有前景类别距离都大于等于阈值，则 target 为背景 0。

不要给三分类 loss 增加隐式 fallback；输入形状不符合配置时直接报错。

### 6. Wrapper 中区分二分类和多分类路径

修改 [src/wrappers/voxel_point_stage1.py](../../src/wrappers/voxel_point_stage1.py)。

需要改的点：

1. `_compute_atom_loss`
   - 当前可传 `atom_logits` 和 `atom_target`。
   - 增加 logits 通道判断，不再假设单通道。
   - 多分类时 target 保持 int64 class id。

2. `_compute_voxel_aux_loss`
   - 当前 `voxel_logits_aux` 形状按 `(B,1,D,H,W)`。
   - 多分类时接受 `(B,3,D,H,W)`。
   - `voxel_label` 仍为 `(B,D,H,W)` class id。

3. `_compute_voxel_ligand_loss`
   - 当前传入单通道 `ligand_dist_map`。
   - 多分类时传入 `(B,C,D,H,W)` 或 `(B,C-1,D,H,W)` 的距离图，由 loss 生成 class id target。

4. metrics
   - 二分类保留 `BinaryAveragePrecision`。
   - 多分类新增按类别 AP：
     - `val/atom_ap_metal_ion`
     - `val/atom_ap_small_molecule`
     - `val/voxel_aux_ap_metal_ion`
     - `val/voxel_aux_ap_small_molecule`
     - `val/voxel_ligand_ap_metal_ion`
     - `val/voxel_ligand_ap_small_molecule`
   - 可额外记录 macro AP。
   - 多分类概率由 `torch.softmax(logits, dim=1)` 得到。

5. 日志与 monitor
   - 默认 monitor 可以改为 `val/voxel_ligand_macro_ap` 或保留用户实验配置指定。
   - 若 monitor 指向不存在 metric，应 fail-fast。

### 7. 推理阶段输出多类概率图

修改：

- [src/inference/get_pred.py](../../src/inference/get_pred.py)
- [src/inference/main/voxel_pipeline.py](../../src/inference/main/voxel_pipeline.py)
- [src/inference/voxel_postprocess.py](../../src/inference/voxel_postprocess.py)
- [src/inference/utils/utils.py](../../src/inference/utils/utils.py)
- [configs/infer_or_eval/*.yaml](../../configs/infer_or_eval/)

当前关键问题：

- [src/inference/get_pred.py](../../src/inference/get_pred.py) 当前固定 `torch.sigmoid(logits[:, 0])`。
- 合并函数 `_merge_box_probability_into_full` 当前只合并单张 `(D,H,W)` 概率图。
- 后处理 `postprocess_ligand_probability_map` 当前面向单张概率图。

目标：

1. logits 转概率：

```python
if logits.shape[1] == 1:
    probs = torch.sigmoid(logits[:, 0])
else:
    probs = torch.softmax(logits, dim=1)
```

2. 多分类合并：
   - 对每个类别通道分别做 BOX 到全图合并。
   - 输出结构从单个 `ligand_pred: (D,H,W)` 扩展为：

```python
ligand_pred_by_class: dict[str, np.ndarray]
receptor_pred_by_class: dict[str, np.ndarray]
```

或数组：

```python
ligand_pred: np.ndarray  # (C,D,H,W)
receptor_pred: np.ndarray  # (C,D,H,W)
```

推荐数组主存储，另在保存阶段按 `class_names` 拆文件。

3. hardmask 约束：
   - receptor head 概率乘 `hardmask`。
   - ligand head 概率乘 `1-hardmask`。
   - 背景通道不参与 ligand/receptor 前景后处理。

4. 后处理：
   - 对 `metal_ion` 和 `small_molecule` 各自调用现有连通域后处理函数。
   - 输出目录中区分类别，例如：

```text
pred/metal_ion/density_pred_prob.map
pred/metal_ion/postprocessed.map
pred/small_molecule/density_pred_prob.map
pred/small_molecule/postprocessed.map
```

5. 配置：
   - `postprocess` 参数允许按类别覆盖阈值。
   - 没有按类别参数时使用全局默认阈值。

### 8. GT 与可视化同步

修改：

- [src/inference/voxel_gt.py](../../src/inference/voxel_gt.py)
- [src/inference/parse_input.py](../../src/inference/parse_input.py)
- [src/inference/utils/utils.py](../../src/inference/utils/utils.py)

目标：

- `class_mapping=[0,1,0,0,2]` 后，GT 可生成：
  - metal 前景 GT。
  - small molecule 前景 GT。
  - 背景不单独作为前景 GT 输出。
- 可视化同时保存每类预测概率图与每类 GT 图。
- `peptide/nucleic` 在 GT 中作为背景，不显示为前景。

### 9. processedPDB_EMDB_binder 是否需要改

本次不必须重新生成 processed 数据，因为：

- `pdb_label_BOX` 已保存原始五类体素标签。
- `ligand_dist_BOX` 已保存原始四个前景类别距离通道。
- 三分类映射可以在训练 dataset 读取时完成。

但需要更新说明文档：

- [processedPDB_EMDB_binder/notes_of_dataset.md](../../processedPDB_EMDB_binder/notes_of_dataset.md)

写明：

- 原始 BOX 仍是五类/四前景通道。
- 训练可通过 `class_mapping=[0,1,0,0,2]` 折叠为三分类。
- ligand 距离图在读取时按映射归并为新类别距离监督。

### 10. 文档同步

按项目说明，机制变化后需要更新：

- [具体工作的细节.md](../../具体工作的细节.md)
- [processedPDB_EMDB_binder/notes_of_dataset.md](../../processedPDB_EMDB_binder/notes_of_dataset.md)

如果只改下游训练/推理，不改上游 `Make_Data` 落盘字段，则 [Make_Data/notes_of_dataset.md](../../Make_Data/notes_of_dataset.md) 不需要改数据字段，只可选补充“下游可重映射”。

## 验证计划

### 单元/小样本验证

1. Dataset 读取单样本：
   - `voxel_label` 只包含 `{0,1,2}`。
   - `atom_label` 只包含 `{0,1,2}`。
   - `ligand_dist_map` 三分类形状符合设计。
   - `peptide/nucleic` 被映射为 0。

2. Collate：
   - batch 后 `voxel_label: (B,D,H,W)`。
   - batch 后 `ligand_dist_map` 保留类别维。
   - 旋转增强不会旋转类别轴。

3. Model forward：
   - `atom_logits: (sumN,3)`。
   - `voxel_logits_aux: (B,3,D,H,W)`。
   - `voxel_logits_ligand: (B,3,D,H,W)`。

4. Loss：
   - 二分类配置仍走 sigmoid。
   - 三分类配置走 softmax。
   - `voxel_ligand_loss` 能从多类距离图生成 class id target。

### 训练 smoke test

运行一个极小 batch：

- forward 成功。
- loss 非 NaN。
- backward 成功。
- validation metric 正常记录每类 AP。

### 推理 smoke test

对单个样本运行：

- 输出 `metal_ion` 概率图。
- 输出 `small_molecule` 概率图。
- 两类分别完成后处理。
- 可视化 GT 与预测类别一致。

## 风险点

1. `ligand_dist_map` 从单通道变多通道会影响 dataset、collate、loss、旋转增强、推理 GT 的多个契约。
2. 旧 checkpoint 的单通道 head 无法直接加载到三通道 head。
3. 当前 AP metric 是二分类 `BinaryAveragePrecision`，多分类必须按类别拆分，否则指标含义错误。
4. `prior_prob` 的二分类 bias 初始化不能直接套到 softmax 多分类 head。
5. 推理后处理原本假设单张概率图，多类输出后目录结构和配置阈值都要显式化。
