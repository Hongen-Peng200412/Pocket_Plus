# 04：从 C 采样 P anchors 与 density cube 特征

> [!IMPORTANT]
> 本阶段初版允许 P sampler 通过 `deduplicate_candidates` 处理重复 C；第 05 阶段实现后，C 已由 builder 唯一化，sampler 直接继承随机路由类别且不再提供该去重开关。本文下方既有校正说明与正文若与 [05-anchor-to-candidate-refine.md](05-anchor-to-candidate-refine.md) 或现行源码冲突，以第 05 阶段契约与源码为准。

## 已做最小修改，请以此处为准

本节是实现审查后的最小校正说明；下文旧计划若与本节冲突，以本节和实际代码为准，不全篇重写旧措辞。

1. `deduplicate_candidates` 仍是显式开关且默认 `false`；只有开启时才保证同 `(batch, voxel_zyx)` 去重，并在去重阶段按 `candidate_prob` 优先、同概率取原始 C 行号更小者。
2. 采样阶段为节省计算不保证同概率稳定 tie-break，尤其 `topk_nms` 不再构造局部最大候选之间的 O(P²) tie 抑制矩阵。
3. `DensityCubeEncoder.in_channels` 支持 `null` lazy 初始化；训练入口会从首个 dataset 样本读取 raw `voxel_grid` 通道数，分别初始化 voxel backbone 的实际输入通道和 density cube 的 raw 输入通道。
4. 训练启动时由 `src/train.py::_initialize_lazy_modules_before_ddp()` 在 rank zero 打印一次 lazy 初始化结果，包括 `raw_voxel_grid_channels`、`voxel_backbone_in_channels` 和 `density_cube_in_channels`。
5. `configs/model/sparse_refine/density_cube/default.yaml` 使用 `in_channels: null`，不再写死 56 通道；若实验需要固定通道数，可在实验配置中显式覆盖。
6. P/C 相关 Hydra 配置已统一迁移到 `configs/model/sparse_refine/`，默认由 `model/sparse_refine: default` 与其子配置组控制。
7. 任务类别 preset 使用 `configs/model/task/{binary,tri}.yaml`；binary 采用单通道 sigmoid + `prior_prob`，tri 采用三通道 softmax + `prior_probs`。
8. `anchor_class_conditioning_cfg.mode=add_embedding` 可在 density cube 输出后、P 注入前向 `pseudo_feat` 加入来源类别 embedding；默认关闭，且不改变 `pseudo_feat` 末维。

## 背景 / 目标

本计划是关于二分类和多分类同时支持的 ligand sparse refine 第 4 个实施步骤。01/02 已完成 Stage1 清理、anchor-based mixed layout 和 typed point core；03 已在 final recycle 的 `_prepare_pseudo_batch()` 中生成 sparse candidate voxel set `C`，但仍返回 real-only point batch。本阶段目标是在同一入口中从 C 采样少量 pseudo anchors `P`，为 P 构造 density cube 初始点特征，并通过 `inject_pseudo_atoms()` 注入最后一轮 point backbone / atom head。

本阶段完成后应满足：

1. `P` 从 03 的 candidate 输出中生成，`P` 数量远小于 `C`。
2. `deduplicate_candidates` 是 anchor sampler 的显式开关，默认 `false`；只有设为 `true` 时，才保证同一 `(batch, voxel_zyx)` 最多生成一个 P anchor，并在 P 层去重 C 中同 voxel 多类别重复记录。
3. 支持三种一等采样模式：`weighted_fps`、`unweighted_fps`、`topk_nms`，每种模式都有独立配置。
4. `max_anchors_per_class` 是唯一 per-class 上限，不再使用 `max_anchors_per_box`。
5. 二分类路径显式支持 `candidate_class_ids: [1]` 与 `max_anchors_per_class: [1024]`。
6. `P` 坐标使用候选体素中心，local voxel 坐标为 `(x+0.5, y+0.5, z+0.5)` 的 corner 语义。
7. `pseudo_feat` 首版只来自 density cube encoder；candidate class/prob 只作为 metadata 透传，不拼入初始点特征。
8. `inject_pseudo_atoms()` 后，P 只在最后一轮 recycle 进入 point backbone，`atom_logits` / `atom_target` / `atom_valid_mask` 仍保持 real-only 对齐。

> \[!IMPORTANT]
> 本计划不实现 P→C 插值、sparse C logits、refine loss、refine metrics，也不把 P feature 回写到 C。

## 当前代码事实

| 事实                                                                                                                                                                          | 位置                                                                                                 | 对 04 的影响                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `_prepare_pseudo_batch()` 当前在 final recycle 生成 C，并返回 `batch, None, candidate_outputs`。                                                                                      | [src/model/stage1\_model.py](../../../../src/model/stage1_model.py) `_prepare_pseudo_batch()`      | 04 在这里继续生成 P、density cube feature 并注入 mixed batch。                       |
| `_run_point_backbone()` 已支持 `pseudo_layout`、`pseudo_mask`、mixed recycle 槽位补零和 typed fusion fail-fast。                                                                       | [src/model/stage1\_model.py](../../../../src/model/stage1_model.py) `_run_point_backbone()`        | 04 只需保证 mixed batch 来自 `inject_pseudo_atoms()` 并包含 `pseudo_mask`。        |
| `_run_atom_head()` 已支持 mixed 输入，并把监督字段裁成 real-only。                                                                                                                         | [src/model/stage1\_model.py](../../../../src/model/stage1_model.py) `_run_atom_head()`             | 04 注入 P 后不改 wrapper-facing atom 监督契约。                                    |
| `inject_pseudo_atoms()` 要求 pseudo dict 包含 `pseudo_coord_centered_world`、`pseudo_coord_local_voxel`、`pseudo_coord_world`、`pseudo_feat`、`pseudo_batch_index`、`pseudo_counts`。 | [src/model/pseudo\_atoms.py](../../../../src/model/pseudo_atoms.py)                                | 04 的 anchor sampler + density cube 必须构造这些字段。                             |
| 03 C 输出字段为 `candidate_voxel_zyx`、`candidate_batch_index`、`candidate_class`、`candidate_prob`、`candidate_logits`、`candidate_counts`、`candidate_counts_by_class` 等。            | [src/model/sparse\_refine/candidate\_set.py](../../../../src/model/sparse_refine/candidate_set.py) | anchor sampler 直接消费这些字段。                                                 |
| batch 中已有 `box_origin_world` `(B,3)`、`voxel_size_world` `(B,3)`、`box_shape_zyx` `(B,3)`、`voxel_grid` `(B,C,D,H,W)`。                                                         | [src/datasets/box\_point\_collate.py](../../../../src/datasets/box_point_collate.py)               | 04 用这些字段把 voxel index 转为 local/world/centered-world 坐标，并抽取 density cube。 |

## 设计决策

### 1. P 层对同 voxel 多类别候选去重开关

03 允许同一 `(batch, z, y, x)` 因多个前景类别入选而出现多条 C 记录。04 通过 `deduplicate_candidates` 显式控制是否把这种 C 重复合并到一个 P。

实现规则：

1. `deduplicate_candidates: false` 为默认值，anchor sampler 直接保留全部 C 行参与 P 采样；同一 voxel 的不同类别候选可以分别生成 P。
2. `deduplicate_candidates: true` 时，anchor sampler 先对 C 做 P 级 canonicalization。
3. key 为 `(candidate_batch_index, candidate_voxel_zyx[:, 0], candidate_voxel_zyx[:, 1], candidate_voxel_zyx[:, 2])`。
4. 同 key 多条候选类别记录时保留 `candidate_prob` 最大的一条。
5. 概率相等时保留原始 C 行号更小的一条。
6. 被保留记录的 `candidate_class` 决定该 P 所属类别和占用的 per-class quota。
7. 不为被跳过类别做预算补偿，不做跨类别候补重分配。

> \[!IMPORTANT]
> 这里即使启用去重，也只去重 P 层输入，不回改 03 的 C 主键语义；C 仍允许同 voxel 多类重复。

### 2. 三种采样模式都是显式 mode，不做 fallback

| mode             | 实现                                               | 推荐用途       | 关键约束                                                       |
| ---------------- | ------------------------------------------------ | ---------- | ---------------------------------------------------------- |
| `weighted_fps`   | 纯 PyTorch，按 `min_dist * prob**weight_power` 串行选点 | 高质量采样 / 消融 | 固定使用 voxel L2 距离；不构造 `C×C` 或完整 `C×P`；默认沿用张量 device，推荐 GPU。 |
| `unweighted_fps` | 调用已有 `torch_cluster.fps`                         | 高效 FPS 近似  | 不新增依赖；不做额外概率预筛；概率不进入 FPS score。                            |
| `topk_nms`       | 纯 PyTorch dense grid + `max_pool3d` 局部极大值        | 训练吞吐优先     | BOX 约 `80^3` 时显存可控；空间分散由 `nms_radius_voxel` 控制。            |

不配置 `fallback_mode`。如果训练吞吐受 `weighted_fps` 影响，用户显式把 `mode` 切到 `topk_nms` 或 `unweighted_fps`。

### 3. weighted-FPS 默认在 GPU 上跑

`weighted_fps` 不主动把 C 搬到 CPU。H100 上 `C=30000, P=3000` 的主要风险是串行选择的 kernel launch / 同步延迟，而不是显存。04 不新增 micro-benchmark；计划只要求实现不构造大矩阵，并保留更快的 `unweighted_fps` / `topk_nms` 配置。

### 4. density cube feature 不拼 candidate metadata

`pseudo_feat = DensityCubeEncoder(voxel_grid, anchor_voxel_zyx, anchor_batch_index)`。

`anchor_class`、`anchor_prob`、`anchor_source_candidate_index` 作为 metadata 输出和透传，不拼接到 `pseudo_feat`，也不影响 density cube encoder。

### 5. density cube encoder 使用 downsample-conv-GAP baseline

实现前确认：首版 density cube encoder 保留 `conv_gap` 类型名，但默认结构调整为“两层 stride=2 下采样 Conv3d → 普通 Conv3d blocks → Global Average Pooling(GAP) → Linear”。`cube_size` 默认从 7 改为 11；15 只作为显式实验配置值，不作为默认。这个选择扩大初始局部视野，同时用 downsample 控制后续 conv blocks 的空间开销；GAP 继续保留，避免 Linear 输入维度绑定具体 cube size。残差块、learned pooling 等变体不在 04 实现。

网络结构：

```text
[重复 num_downsample 次]
Conv3d(in_channels/hidden_channels, hidden_channels, 3, padding=1, stride=2)
GroupNorm(num_groups, hidden_channels)
SiLU/ReLU/GELU
[重复 num_conv 次]
Conv3d(hidden_channels, hidden_channels, 3, padding=1, stride=1)
GroupNorm(num_groups, hidden_channels)
SiLU/ReLU/GELU
AdaptiveAvgPool3d(1)  # GAP
Flatten
Linear(hidden_channels, out_dim)
```

默认推荐 `encoder_type: conv_gap`、`act: silu`、`norm: group`、`num_groups: 8`、`hidden_channels: 64`、`num_downsample: 2`、`num_conv: 2`、`cube_size: 11`、`chunk_size: 512`。

显存说明：`P≈3000`、density channel `56` 时，`cube_size=11` 的原始 cube fp32 临时张量约 0.9GB，`cube_size=15` 约 2.3GB；实际训练还要叠加 autograd 保存和卷积激活。因此实现必须分块抽取 cube，默认 `chunk_size=512`，不一次性处理全部 P。

> \[!IMPORTANT]
> 04 只支持 `norm: group | none`，默认 `group`。不支持 LayerNorm / BatchNorm。

> \[!WARNING]
> `hidden_channels % num_groups != 0` 时构造函数直接 fail-fast，不自动猜测新 group 数。

## 已有可复用代码

| 可复用代码                                           | 位置                                                                                                        | 用法                                                                      |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `inject_pseudo_atoms()` / `PseudoAtomLayout`    | [src/model/pseudo\_atoms.py](../../../../src/model/pseudo_atoms.py)                                       | 将 P pseudo dict 注入 real batch，生成 mixed batch、`pseudo_mask`、`real_mask`。 |
| `interleave_real_and_pseudo_tensor()`           | [src/model/pseudo\_atoms.py](../../../../src/model/pseudo_atoms.py)                                       | mixed recycle 输入中为 P 槽位补零，已有 `_run_point_backbone()` 调用。                |
| `Stage1PointBackbone.forward(..., pseudo_mask)` | [src/model/stage1\_point\_backbone.py](../../../../src/model/stage1_point_backbone.py)                    | 接收 mixed P 点并透传 `pseudo_mask`。                                          |
| `Stage1AtomHead.forward(..., pseudo_mask)`      | [src/model/stage1\_atom\_head.py](../../../../src/model/stage1_atom_head.py)                              | 输出 real-only `atom_logits` 和 pseudo-only `pseudo_feature`。              |
| `torch_cluster.fps`                             | [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) 已直接 import `torch_cluster` | `unweighted_fps` mode 调用已有依赖，不新增安装步骤。                                   |

## Proposed Changes

### 1. 新增 anchor sampler 模块

#### \[NEW] [src/model/sparse\_refine/anchor\_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py)

新增 `SparseAnchorSampler`，负责从 03 的 C 输出中选择 P anchors，并计算 P 的三套坐标与 metadata。

文件开头直接导入：

```python
import torch
import torch_cluster
from torch import nn
```

新增类：

```python
class SparseAnchorSampler(nn.Module):
    def __init__(
        self,
        candidate_class_ids: Sequence[int],
        max_anchors_per_class: Sequence[int],
        mode: str,
        weight_power: float = 1.0,
        chunk_size: int = 8192,
        nms_radius_voxel: int = 2,
        random_start: bool = False,
        deduplicate_candidates: bool = False,
    ) -> None: ...

    def forward(
        self,
        candidate_outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]: ...
```

| 参数                      | 类型              | 默认      | 允许值 / shape                                    | 意义                                      |
| ----------------------- | --------------- | ------- | ---------------------------------------------- | --------------------------------------- |
| `candidate_class_ids`   | `Sequence[int]` | 必填      | `(K,)`                                         | 与 candidate builder 对齐的前景类别 ID。         |
| `max_anchors_per_class` | `Sequence[int]` | 必填      | `(K,)`，每项 `>=0`                                | 每个 BOX、每个候选类最多产生多少 P；不再有 per-box 总 cap。 |
| `mode`                  | `str`           | 必填      | `weighted_fps` / `unweighted_fps` / `topk_nms` | 采样模式。                                   |
| `weight_power`          | `float`         | `1.0`   | `>=0`                                          | `weighted_fps` 的概率权重指数。                 |
| `chunk_size`            | `int`           | `8192`  | `>0`                                           | `weighted_fps` 距离更新分块大小。                |
| `nms_radius_voxel`      | `int`           | `2`     | `>=0`                                          | `topk_nms` 的局部最大池化半径，单位为 voxel。         |
| `random_start`          | `bool`          | `False` | bool                                           | 传给 `torch_cluster.fps`。                 |
| `deduplicate_candidates` | `bool`         | `False` | bool                                           | 是否在 P 层按同 BOX/voxel 去重 C 候选。       |

构造函数校验：

1. `candidate_class_ids` 与 `max_anchors_per_class` 长度必须一致。
2. `mode` 必须是三种允许值之一。
3. `weight_power >= 0`。
4. `chunk_size > 0`。
5. `nms_radius_voxel >= 0`。
6. `deduplicate_candidates` 转为 bool 使用，默认关闭。
7. 不接受 `max_anchors_per_box`、`fallback_mode`、`distance_metric`、`unweighted_prefilter_per_class` 等旧字段；Hydra 传入即让 Python 报 unexpected keyword。

`forward()` 输入字段要求：

| 字段                      | shape            | 必需 | 来源                   |
| ----------------------- | ---------------- | -- | -------------------- |
| `candidate_voxel_zyx`   | `(sumC, 3)` long | 是  | 03 candidate builder |
| `candidate_batch_index` | `(sumC,)` long   | 是  | 03 candidate builder |
| `candidate_class`       | `(sumC,)` long   | 是  | 03 candidate builder |
| `candidate_prob`        | `(sumC,)` float  | 是  | 03 candidate builder |
| `box_origin_world`      | `(B, 3)` float   | 是  | batch                |
| `voxel_size_world`      | `(B, 3)` float   | 是  | batch                |
| `box_shape_zyx`         | `(B, 3)` long    | 是  | batch                |

`forward()` 输出字段：

| 字段                              | shape             | 语义                                             |
| ------------------------------- | ----------------- | ---------------------------------------------- |
| `anchor_voxel_zyx`              | `(sumP, 3)` long  | P 来源 voxel index，轴顺序 `(z, y, x)`。              |
| `anchor_coord_local_voxel`      | `(sumP, 3)` float | P 中心 local voxel 坐标，轴顺序 `(x, y, z)`，corner 语义。 |
| `anchor_coord_world`            | `(sumP, 3)` float | P 世界坐标 `(x, y, z)`。                            |
| `anchor_coord_centered_world`   | `(sumP, 3)` float | P 相对 BOX 中心的世界坐标 `(x, y, z)`。                  |
| `anchor_batch_index`            | `(sumP,)` long    | P 所属 BOX。                                      |
| `anchor_class`                  | `(sumP,)` long    | P 继承的候选类别 ID。                                  |
| `anchor_prob`                   | `(sumP,)` float   | P 对应候选记录的概率。                                   |
| `anchor_source_candidate_index` | `(sumP,)` long    | P 对应原始 C 行号。                                   |
| `anchor_counts`                 | `(B,)` long       | 每个 BOX 的 P 总数。                                 |
| `anchor_counts_by_class`        | `(B, K)` long     | 每个 BOX、每个候选类的 P 数量。                            |

> \[!IMPORTANT]
> `anchor_counts_by_class` 是类别维度统计；`anchor_counts` 只是每个 BOX 的总数，不能替代 per-class 统计。

##### 1.1 C canonicalization helper

新增私有函数：

```python
def _deduplicate_candidates_by_voxel(
    candidate_voxel_zyx: torch.Tensor,
    candidate_batch_index: torch.Tensor,
    candidate_prob: torch.Tensor,
) -> torch.Tensor: ...
```

返回：`kept_candidate_index: torch.Tensor`，shape `(sumC_unique,)`，表示保留的原始 C 行号。

行为：

1. 对每个 `(batch,z,y,x)` 只保留一个候选类别记录。
2. 优先保留 `candidate_prob` 最大的记录。
3. tie 时保留原始行号更小的记录。
4. 返回 index 按原始 C 行号升序排列，后续按 BOX/类别分组时再局部排序。

实现要求：

1. 不使用 Python dict 逐候选循环作为主路径。
2. 可使用 `torch.argsort()` 和组合 key 实现稳定分组。
3. 若 `sumC == 0`，返回 shape `(0,)` 的 long tensor。

##### 1.2 weighted-FPS

新增私有方法：

```python
def _sample_weighted_fps_one_group(
    self,
    candidate_index: torch.Tensor,
    voxel_zyx: torch.Tensor,
    prob: torch.Tensor,
    max_count: int,
) -> torch.Tensor: ...
```

| 参数                | shape    | 意义                        |
| ----------------- | -------- | ------------------------- |
| `candidate_index` | `(M,)`   | 当前 BOX/类别 canonical C 行号。 |
| `voxel_zyx`       | `(M, 3)` | 当前 BOX/类别候选 voxel 坐标。     |
| `prob`            | `(M,)`   | 当前 BOX/类别候选概率。            |
| `max_count`       | 标量       | 当前类别上限。                   |

返回：`selected_candidate_index: torch.Tensor`，shape `(P_group,)`，`P_group <= max_count`。

算法：

1. 若 `M <= max_count`，直接返回该组全部 `candidate_index`，顺序按 `prob` 降序、原始行号升序稳定排列。
2. 第一枚 anchor 选择最高概率候选。
3. 维护 `min_dist_to_selected: (M,)`，初始为 `inf`。
4. 每轮用新 anchor 按 `chunk_size` 更新 `min_dist_to_selected`。
5. 选择 `score = min_dist_to_selected * prob.pow(weight_power)` 最大的位置。
6. 距离固定使用 voxel index 空间 L2 距离平方。
7. 不构造 `M×M` 或完整 `M×P` 距离矩阵。
8. 张量在哪个 device 就在哪个 device 计算；不主动 `.cpu()`。

##### 1.3 unweighted-FPS

新增私有方法：

```python
def _sample_unweighted_fps_one_group(
    self,
    candidate_index: torch.Tensor,
    voxel_zyx: torch.Tensor,
    prob: torch.Tensor,
    max_count: int,
) -> torch.Tensor: ...
```

行为：

1. 使用文件开头导入的 `torch_cluster.fps`。
2. 若 `M <= max_count`，直接返回该组全部 `candidate_index`，顺序按 `prob` 降序排列。
3. 对当前 BOX/类别全部 canonical C 调用 FPS，不额外做概率预筛。
4. `voxel_xyz_float = voxel_zyx[:, [2, 1, 0]].float()`，FPS 坐标使用 `(x,y,z)` 或 `(z,y,x)` 都只影响维度命名、不影响欧氏距离；实现中固定为 `(x,y,z)` 以对齐其它坐标语义。
5. `ratio = min(1.0, max_count / M)`，其中 `M = int(candidate_index.shape[0])`。
6. 调用 `torch_cluster.fps(x=voxel_xyz_float, batch=None, ratio=ratio, random_start=random_start)`。
7. 返回数量若超过 `max_count`，按 FPS 输出顺序截断。
8. 若返回数量少于 `max_count`，不补点；`max_anchors_per_class` 是上限，不是必须凑满的目标。

> \[!NOTE]
> `unweighted_fps` 不实现 probability-weighted score。它是高效近似模式，不是 `weighted_fps` 的等价替代。

##### 1.4 topk-NMS

新增私有方法：

```python
def _sample_topk_nms_one_group(
    self,
    candidate_index: torch.Tensor,
    voxel_zyx: torch.Tensor,
    prob: torch.Tensor,
    max_count: int,
    box_shape_zyx: torch.Tensor,
) -> torch.Tensor: ...
```

行为：

1. 对当前 BOX/类别候选按 `candidate_prob` 写入 dense score grid，shape `(1, 1, D, H, W)`。
2. 非候选位置填 `-inf`。
3. 用 `F.max_pool3d(score_grid, kernel_size=2*r+1, stride=1, padding=r)` 得到局部最大值。
4. 保留 `score == pooled_score` 的候选作为局部极大值；tie 时用原始 C 行号更小者优先。
5. 从局部极大候选按概率降序、原始行号升序取前 `max_count`。
6. 若局部极大数量小于 `max_count`，不从被抑制候选补点；保持 NMS 语义简单。

> \[!WARNING]
> `topk_nms` 是吞吐优先模式。它保证高概率局部极大和空间去重，但不是全局覆盖最优。

##### 1.5 坐标转换 helper

新增函数：

```python
def build_anchor_coordinates(
    anchor_voxel_zyx: torch.Tensor,
    anchor_batch_index: torch.Tensor,
    box_origin_world: torch.Tensor,
    voxel_size_world: torch.Tensor,
    box_shape_zyx: torch.Tensor,
) -> dict[str, torch.Tensor]: ...
```

输出：

| 字段                            | shape       | 公式                                                                |
| ----------------------------- | ----------- | ----------------------------------------------------------------- |
| `anchor_coord_local_voxel`    | `(sumP, 3)` | `torch.stack([x+0.5, y+0.5, z+0.5], dim=-1)`                      |
| `anchor_coord_world`          | `(sumP, 3)` | `origin_xyz[batch] + local_xyz * voxel_size_xyz[batch]`           |
| `anchor_coord_centered_world` | `(sumP, 3)` | `world_xyz - (origin_xyz + 0.5 * box_shape_xyz * voxel_size_xyz)` |

其中 `box_shape_xyz = box_shape_zyx[:, [2, 1, 0]]`。

### 2. 新增 density cube encoder 模块

#### \[NEW] [src/model/sparse\_refine/density\_cube.py](../../../../src/model/sparse_refine/density_cube.py)

新增 `DensityCubeEncoder`。

```python
class DensityCubeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cube_size: int,
        hidden_channels: int,
        num_conv: int,
        out_dim: int,
        encoder_type: str = "conv_gap",
        norm: str = "group",
        num_groups: int = 8,
        act: str = "silu",
        chunk_size: int = 512,
        num_downsample: int = 2,
    ) -> None: ...

    def forward(
        self,
        voxel_grid: torch.Tensor,
        anchor_voxel_zyx: torch.Tensor,
        anchor_batch_index: torch.Tensor,
    ) -> torch.Tensor: ...
```

| 参数                | 类型  | 默认         | 约束                                  | 意义                                                            |
| ----------------- | --- | ---------- | ----------------------------------- | ------------------------------------------------------------- |
| `in_channels`     | int | 必填         | `>0`                                | `voxel_grid.shape[1]`，来自 dataset density channel 配置。          |
| `cube_size`       | int | 必填         | 奇数且 `>=1`                           | 每个 P 周围抽取的 cube 边长。                                           |
| `hidden_channels` | int | 必填         | `>0`                                | Conv3d hidden 通道数。                                            |
| `num_conv`        | int | 必填         | `>=1`                               | Conv3d block 数。                                               |
| `out_dim`         | int | 必填         | `>0`                                | 输出 `pseudo_feat` 维度，必须等于当前 point backbone `atom_feature_dim`。 |
| `encoder_type`    | str | `conv_gap` | 只支持 `conv_gap`                      | density cube 小模块类型。                                           |
| `norm`            | str | `group`    | `group` / `none`                    | Conv3d 后归一化；不支持 LayerNorm / BatchNorm。                        |
| `num_groups`      | int | `8`        | `hidden_channels % num_groups == 0` | GroupNorm group 数。                                            |
| `act`             | str | `silu`     | `silu` / `relu` / `gelu`            | 激活函数。                                                         |
| `chunk_size`      | int | `512`      | `>0`                                | 分块抽取 cube，限制临时显存。                                             |
| `num_downsample`  | int | `2`        | `>=0`                               | stride=2 下采样 Conv3d 层数。                                             |

构造函数要求：

1. 保存 `self.out_dim = int(out_dim)`，供 Stage1 初始化时自动校验。
2. `encoder_type != "conv_gap"` 直接 `ValueError`。
3. `norm not in {"group", "none"}` 直接 `ValueError`。
4. `norm == "group"` 且 `hidden_channels % num_groups != 0` 时直接 `ValueError`。
5. 模块中不得使用 `nn.BatchNorm3d` 或 LayerNorm。

网络结构：

```text
for i in range(num_downsample):
    Conv3d(in_channels if i == 0 else hidden_channels, hidden_channels, 3, padding=1, stride=2)
    [GroupNorm(num_groups, hidden_channels) if norm == "group"]
    activation
for i in range(num_conv):
    Conv3d(hidden_channels, hidden_channels, 3, padding=1, stride=1)
    [GroupNorm(num_groups, hidden_channels) if norm == "group"]
    activation
AdaptiveAvgPool3d(1)
Flatten
Linear(hidden_channels, out_dim)
```

`forward()` 行为：

1. 校验 `voxel_grid.ndim == 5`，shape 为 `(B, C, D, H, W)`。
2. 校验 `C == in_channels`。
3. 校验 `anchor_voxel_zyx` 为 `(sumP, 3)` long，`anchor_batch_index` 为 `(sumP,)` long。
4. `sumP == 0` 时返回 `voxel_grid.new_empty((0, out_dim))`。
5. 使用 zero padding 抽取边界 cube；不对越界 anchor 做 wrap 或 clamp。
6. 按 `chunk_size` 处理 P，临时 cube shape 为 `(P_chunk, C, cube_size, cube_size, cube_size)`。
7. 输出 `pseudo_feat: (sumP, out_dim)`。

> \[!IMPORTANT]
> density cube 只读 `batch["voxel_grid"]`，不读 voxel backbone feature，也不读 candidate logits/prob/class。

### 3. 更新 sparse\_refine 包导出

#### \[MODIFY] [src/model/sparse\_refine/**init**.py](../../../../src/model/sparse_refine/__init__.py)

导出新增类：

```python
from src.model.sparse_refine.anchor_sampler import SparseAnchorSampler
from src.model.sparse_refine.candidate_set import SparseCandidateSetBuilder
from src.model.sparse_refine.density_cube import DensityCubeEncoder

__all__ = ["SparseCandidateSetBuilder", "SparseAnchorSampler", "DensityCubeEncoder"]
```

### 4. Stage1 主模型接入 P anchor pipeline

#### \[MODIFY] [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)

##### 4.1 构造函数新增配置参数

在 `VolumePointStage1Model.__init__()` 增加：

```python
anchor_sampler_cfg: dict[str, Any] | nn.Module | None = None,
density_cube_cfg: dict[str, Any] | nn.Module | None = None,
```

新增成员：

```python
self.anchor_sampler = ...
self.density_cube_encoder = ...
```

构造规则：

| 条件                                                        | 行为                                                |
| --------------------------------------------------------- | ------------------------------------------------- |
| `anchor_sampler_cfg is None`                              | `self.anchor_sampler = None`，保持 03 行为，只输出 C。      |
| `anchor_sampler_cfg` 是 `nn.Module`                        | 直接赋值。                                             |
| 其它 Hydra config                                           | `instantiate(anchor_sampler_cfg)`。                |
| `density_cube_cfg is None` 且 `anchor_sampler is None`     | `self.density_cube_encoder = None`。               |
| `density_cube_cfg is None` 且 `anchor_sampler is not None` | fail-fast：启用 P anchors 必须配置 density cube encoder。 |
| `density_cube_cfg` 是 `nn.Module`                          | 直接赋值。                                             |
| 其它 Hydra config                                           | `instantiate(density_cube_cfg)`。                  |

额外校验：

1. `anchor_sampler is not None` 时，`candidate_set_builder` 也必须启用。
2. 若 `candidate_set_builder` 和 `anchor_sampler` 都有 `candidate_class_ids`，二者必须完全一致。
3. `density_cube_encoder` 必须有 `out_dim` 属性。
4. `density_cube_encoder.out_dim` 必须等于 `self.point_backbone.atom_feature_dim`，在初始化阶段直接 fail-fast。
5. forward 阶段仍保留 `pseudo_feat.shape[-1] == batch["atom_feat"].shape[-1]` 检查，作为运行时防线。

##### 4.2 `_prepare_pseudo_batch()` 改为 C→P→inject

当前流程：

```python
candidate_outputs = self.candidate_set_builder(...)
return batch, None, candidate_outputs
```

修改为：

```python
candidate_outputs = self.candidate_set_builder(...)
if self.anchor_sampler is None:
    return batch, None, candidate_outputs

anchor_outputs = self.anchor_sampler(candidate_outputs=candidate_outputs, batch=batch)
pseudo_feat = self.density_cube_encoder(
    voxel_grid=batch["voxel_grid"],
    anchor_voxel_zyx=anchor_outputs["anchor_voxel_zyx"],
    anchor_batch_index=anchor_outputs["anchor_batch_index"],
)
if pseudo_feat.shape[-1] != batch["atom_feat"].shape[-1]:
    raise RuntimeError(...)

pseudo_dict = {
    "pseudo_coord_centered_world": anchor_outputs["anchor_coord_centered_world"],
    "pseudo_coord_local_voxel": anchor_outputs["anchor_coord_local_voxel"],
    "pseudo_coord_world": anchor_outputs["anchor_coord_world"],
    "pseudo_feat": pseudo_feat,
    "pseudo_batch_index": anchor_outputs["anchor_batch_index"],
    "pseudo_counts": anchor_outputs["anchor_counts"],
    "pseudo_anchor_class": anchor_outputs["anchor_class"],
    "pseudo_anchor_voxel_zyx": anchor_outputs["anchor_voxel_zyx"],
    "pseudo_source_candidate_index": anchor_outputs["anchor_source_candidate_index"],
}
point_batch, pseudo_layout = inject_pseudo_atoms(batch, pseudo_dict)
pseudo_outputs = {**candidate_outputs, **anchor_outputs}
return point_batch, pseudo_layout, pseudo_outputs
```

输出影响：

| 输出字段                                              | 有 anchor sampler                 | 无 anchor sampler |
| ------------------------------------------------- | -------------------------------- | ---------------- |
| `candidate_*`                                     | 保留 03 输出                         | 保留 03 输出或无       |
| `anchor_*`                                        | 新增 P anchor metadata             | 无                |
| `pseudo_feature`                                  | atom head 输出 pseudo-only feature | `None`           |
| `atom_logits` / `atom_target` / `atom_valid_mask` | real-only                        | real-only        |

> \[!IMPORTANT]
> `pseudo_feat` 是 point backbone 输入特征；`outputs["pseudo_feature"]` 是 atom head 对 P 的输出特征，二者不是同一个张量。

##### 4.3 小修：atom\_head none 注释

#### \[MODIFY] [configs/model/atom\_head/none.yaml](../../../../configs/model/atom_head/none.yaml)

当前注释“不进行任何运算”容易误导，因为该配置只是 `atom_head_num_layers: 0`，并未设置 `enable_atom_head: false`。同步改注释为：

```yaml
# atom head 无 attention block 变体: 仍构造 token projection 和 logit head, 只是 atom_head_num_layers=0
```

不改变该配置的实际行为。

##### 4.4 小修：checkpoint best\_f1 finite 校验

#### \[MODIFY] [src/wrappers/voxel\_point\_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)

在 `on_load_checkpoint()` 恢复 `voxel_ligand_best_f1_by_class` 时，增加与 `p_best_by_class` / `p_sampling_by_class` 一致的 finite 校验：

1. shape 长度必须等于 candidate class 数量。
2. 若存在 NaN/Inf，直接 fail-fast。

### 5. 配置更新

#### \[MODIFY] [configs/base.yaml](../../../../configs/base.yaml)

在 model defaults 中增加默认关闭项，放在 `model/candidate_set: none` 附近：

```yaml
- model/anchor_sampler: none
- model/density_cube: none
```

#### \[NEW] [configs/model/anchor\_sampler/none.yaml](../../../../configs/model/anchor_sampler/none.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_sampler_cfg: null
```

#### \[NEW] [configs/model/anchor\_sampler/weighted\_fps.yaml](../../../../configs/model/anchor_sampler/weighted_fps.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_sampler_cfg:
      _target_: src.model.sparse_refine.anchor_sampler.SparseAnchorSampler
      mode: weighted_fps
      candidate_class_ids: [1, 2]
      max_anchors_per_class: [256, 768]
      weight_power: 1.0
      chunk_size: 8192
      nms_radius_voxel: 2
      random_start: false
      deduplicate_candidates: false
```

二分类使用同 mode 时的配置值必须写为：

```yaml
candidate_class_ids: [1]
max_anchors_per_class: [1024]
```

#### \[NEW] [configs/model/anchor\_sampler/unweighted\_fps.yaml](../../../../configs/model/anchor_sampler/unweighted_fps.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_sampler_cfg:
      _target_: src.model.sparse_refine.anchor_sampler.SparseAnchorSampler
      mode: unweighted_fps
      candidate_class_ids: [1, 2]
      max_anchors_per_class: [256, 768]
      weight_power: 1.0
      chunk_size: 8192
      nms_radius_voxel: 2
      random_start: false
      deduplicate_candidates: false
```

二分类使用同 mode 时的配置值必须写为：

```yaml
candidate_class_ids: [1]
max_anchors_per_class: [1024]
```

#### \[NEW] [configs/model/anchor\_sampler/topk\_nms.yaml](../../../../configs/model/anchor_sampler/topk_nms.yaml)

```yaml
# @package _global_
model:
  backbone:
    anchor_sampler_cfg:
      _target_: src.model.sparse_refine.anchor_sampler.SparseAnchorSampler
      mode: topk_nms
      candidate_class_ids: [1, 2]
      max_anchors_per_class: [256, 768]
      weight_power: 1.0
      chunk_size: 8192
      nms_radius_voxel: 2
      random_start: false
      deduplicate_candidates: false
```

二分类使用同 mode 时的配置值必须写为：

```yaml
candidate_class_ids: [1]
max_anchors_per_class: [1024]
```

#### \[NEW] [configs/model/density\_cube/none.yaml](../../../../configs/model/density_cube/none.yaml)

```yaml
# @package _global_
model:
  backbone:
    density_cube_cfg: null
```

#### \[NEW] [configs/model/density\_cube/default.yaml](../../../../configs/model/density_cube/default.yaml)

```yaml
# @package _global_
model:
  backbone:
    density_cube_cfg:
      _target_: src.model.sparse_refine.density_cube.DensityCubeEncoder
      in_channels: ${dataset.density_channel_config.num_channels}
      cube_size: 11
      hidden_channels: 64
      num_downsample: 2
      num_conv: 2
      out_dim: ${model.backbone.point_backbone.atom_feature_dim}
      encoder_type: conv_gap
      norm: group
      num_groups: 8
      act: silu
      chunk_size: 512
```

> \[!IMPORTANT]
> sparse refine 实验启用 anchor sampler 时必须同时启用 `model/density_cube: default`。`out_dim` 跟随 point backbone 当前 `atom_feature_dim`，以兼容 embed head 把点特征维度从 49 改到 64 的情况。

#### \[MODIFY] sparse refine 实验配置

三分类 sparse refine 实验应显式组合：

```yaml
- override /model/candidate_set: tri
- override /model/anchor_sampler: topk_nms      # 或 weighted_fps / unweighted_fps
- override /model/density_cube: default
```

二分类 sparse refine 实验应显式组合：

```yaml
- override /model/candidate_set: binary
- override /model/anchor_sampler: topk_nms
- override /model/density_cube: default
```

并覆盖：

```yaml
model:
  backbone:
    anchor_sampler_cfg:
      candidate_class_ids: [1]
      max_anchors_per_class: [1024]
```

### 6. 同步总控契约

#### \[MODIFY] [CLAUDE/plans/implement/tri\_ligand\_sparse\_refine/00-master.md](00-master.md)

同步新增 / 更新：

1. 04 的 P anchor 输出字段契约，尤其 `anchor_counts_by_class: (B, K)`。
2. P 层同 voxel 去重规则：保留最高 `candidate_prob` 候选类别记录，tie 保留原始行号更小者。
3. 三种采样模式：`weighted_fps` / `unweighted_fps` / `topk_nms`。
4. `max_candidate_voxels_per_box` 删除，只保留 `max_anchors_per_class`。
5. `pseudo_feat` 只来自 density cube，不拼接 class/prob。
6. density cube 使用 `conv_gap`，GroupNorm 或 no norm，不使用 LayerNorm / BatchNorm。
7. density cube 使用 `voxel_grid`，输出维度对齐 point backbone `atom_feature_dim`。
8. P 只在 final recycle 注入，wrapper-facing atom supervised 字段保持 real-only。

## 不修改的部分

1. 不实现 P→C three-nn 插值。
2. 不实现 sparse C logits 或 refined C head。
3. 不实现 refine loss / metrics / wrapper 新损失分支。
4. 不把 P feature 回写到 C。
5. 不新增 `point_kind` / `atom_point_kind` 三态字段。
6. 不修改 03 的 C 主键；C 仍允许同 voxel 多类别重复。
7. 不安装新依赖；`unweighted_fps` 只使用环境中已有的 `torch_cluster`。
8. 不新增 micro-benchmark 测试。
9. 不改变 `configs/model/atom_head/none.yaml` 的实际行为，只修正注释。
10. 不实现 density cube residual、strided downsample、learned pooling 或其它 encoder 变体。

## 改动文件汇总

| 文件                                                                                                                 | 改动内容                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| [src/model/sparse\_refine/anchor\_sampler.py](../../../../src/model/sparse_refine/anchor_sampler.py)               | 新增 `SparseAnchorSampler`，实现 C 去重、三种采样模式、P 坐标转换与 anchor metadata 输出。                              |
| [src/model/sparse\_refine/density\_cube.py](../../../../src/model/sparse_refine/density_cube.py)                   | 新增 `DensityCubeEncoder`，抽取 P 周围 density cube 并编码为 `pseudo_feat`。                                 |
| [src/model/sparse\_refine/**init**.py](../../../../src/model/sparse_refine/__init__.py)                            | 导出 `SparseAnchorSampler` 与 `DensityCubeEncoder`。                                                 |
| [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)                                                | 新增 `anchor_sampler_cfg` / `density_cube_cfg`，在 `_prepare_pseudo_batch()` 中执行 C→P→density→inject。 |
| [src/wrappers/voxel\_point\_stage1.py](../../../../src/wrappers/voxel_point_stage1.py)                             | 补齐 checkpoint load 中 `best_f1_by_class` finite 校验。                                               |
| [configs/base.yaml](../../../../configs/base.yaml)                                                                 | defaults 增加 `model/anchor_sampler: none` 和 `model/density_cube: none`。                           |
| [configs/model/anchor\_sampler/none.yaml](../../../../configs/model/anchor_sampler/none.yaml)                      | 默认关闭 anchor sampler。                                                                             |
| [configs/model/anchor\_sampler/weighted\_fps.yaml](../../../../configs/model/anchor_sampler/weighted_fps.yaml)     | 新增 weighted-FPS 主配置。                                                                             |
| [configs/model/anchor\_sampler/unweighted\_fps.yaml](../../../../configs/model/anchor_sampler/unweighted_fps.yaml) | 新增 torch\_cluster FPS 主配置。                                                                       |
| [configs/model/anchor\_sampler/topk\_nms.yaml](../../../../configs/model/anchor_sampler/topk_nms.yaml)             | 新增 topk-NMS 主配置。                                                                                 |
| [configs/model/density\_cube/none.yaml](../../../../configs/model/density_cube/none.yaml)                          | 默认关闭 density cube。                                                                               |
| [configs/model/density\_cube/default.yaml](../../../../configs/model/density_cube/default.yaml)                    | 新增 density cube encoder 默认配置。                                                                    |
| [configs/model/atom\_head/none.yaml](../../../../configs/model/atom_head/none.yaml)                                | 修正注释，说明仍构造 atom head，只是无 attention block。                                                        |
| [CLAUDE/plans/implement/tri\_ligand\_sparse\_refine/00-master.md](00-master.md)                                    | 同步 P anchor / density cube 契约。                                                                   |
| [tests/model/test\_anchor\_sampler.py](../../../../tests/model/test_anchor_sampler.py)                             | 新增 anchor sampler 单测。                                                                            |
| [tests/model/test\_density\_cube.py](../../../../tests/model/test_density_cube.py)                                 | 新增 density cube encoder 单测。                                                                      |
| [tests/model/test\_stage1\_model.py](../../../../tests/model/test_stage1_model.py)                                 | 增加 C→P→mixed 注入、real-only supervised 输出测试。                                                       |
| [tests/test\_voxel\_ligand\_thresholds.py](../../../../tests/test_voxel_ligand_thresholds.py)                      | 补 checkpoint `best_f1_by_class` NaN/Inf load fail-fast 测试。                                       |

## Verification Plan

### Automated Tests

```bash
pytest tests/model/test_anchor_sampler.py -q
```

覆盖点：

| 测试函数                                                           | 构造                                                         | 关键断言                                                                        |
| -------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| `test_anchor_sampler_rejects_max_anchors_length_mismatch`      | `candidate_class_ids=[1,2]`、`max_anchors_per_class=[1024]` | 构造时报 `ValueError`。                                                          |
| `test_anchor_sampler_rejects_unknown_mode`                     | `mode="bad"`                                               | 构造时报 `ValueError`。                                                          |
| `test_anchor_sampler_deduplicates_same_voxel_by_highest_prob`  | 同一 `(batch,z,y,x)` 两类 C 记录，prob 不同                         | 只输出一个 P，保留高 prob 记录。                                                        |
| `test_anchor_sampler_deduplicate_tie_keeps_earlier_candidate`  | 同 key 同 prob                                               | `anchor_source_candidate_index` 为较小行号。                                      |
| `test_anchor_sampler_outputs_counts_by_class_and_total_counts` | B\=2、K\=2                                                  | `anchor_counts.shape == (B,)`，`anchor_counts_by_class.shape == (B,K)`，总和一致。 |
| `test_anchor_coordinates_use_xyz_center_corner_semantics`      | 已知 `box_origin_world`、`voxel_size_world`、`box_shape_zyx`   | local/world/centered-world 坐标公式正确。                                          |
| `test_weighted_fps_respects_per_class_cap`                     | 每类 C 数量大于 cap                                              | 每 BOX/类 P 数不超过 `max_anchors_per_class`。                                     |
| `test_weighted_fps_first_anchor_is_highest_prob`               | 构造概率最高点                                                    | 第一枚选中高概率点。                                                                  |
| `test_weighted_fps_does_not_return_duplicate_voxels`           | C 中有重复 voxel                                               | P 中同 `(batch,zyx)` 唯一。                                                      |
| `test_unweighted_fps_uses_torch_cluster`                       | 正常 `torch_cluster` 环境                                      | mode 可运行，输出不超过 cap。                                                         |
| `test_unweighted_fps_uses_all_candidates_without_prefilter`    | 构造低概率远端点                                                   | 不因概率预筛被提前删除。                                                                |
| `test_topk_nms_selects_local_maxima`                           | 相邻高/低概率候选                                                  | 只保留局部最大。                                                                    |
| `test_topk_nms_radius_zero_matches_topk_cap`                   | `nms_radius_voxel=0`                                       | 行为退化为 per-class topk。                                                       |
| `test_binary_anchor_sampler_config_shape`                      | `candidate_class_ids=[1]`、`max_anchors_per_class=[1024]`   | 输出 `anchor_counts_by_class.shape == (B,1)`。                                 |

```bash
pytest tests/model/test_density_cube.py -q
```

覆盖点：

| 测试函数                                                                  | 构造                                   | 关键断言                                 |
| --------------------------------------------------------------------- | ------------------------------------ | ------------------------------------ |
| `test_density_cube_encoder_output_shape`                              | `voxel_grid=(B,C,D,H,W)`、`sumP>0`    | 输出 `(sumP, out_dim)`。                |
| `test_density_cube_encoder_empty_anchor_returns_empty_feature`        | `sumP=0`                             | 输出 `(0, out_dim)`。                   |
| `test_density_cube_encoder_rejects_even_cube_size`                    | `cube_size=6`                        | 构造时报 `ValueError`。                   |
| `test_density_cube_encoder_rejects_channel_mismatch`                  | `in_channels != voxel_grid.shape[1]` | forward 报错。                          |
| `test_density_cube_encoder_uses_zero_padding_at_border`               | anchor 位于边界                          | cube 抽取不越界，边界外为 0。                   |
| `test_density_cube_encoder_uses_groupnorm_not_batchnorm_or_layernorm` | 默认配置                                 | 模块中不存在 `nn.BatchNorm3d` 和 LayerNorm。 |
| `test_density_cube_encoder_rejects_bad_group_count`                   | `hidden_channels=60,num_groups=8`    | 构造时报 `ValueError`。                   |
| `test_density_cube_encoder_rejects_non_conv_gap_encoder_type`         | `encoder_type="residual"`            | 构造时报 `ValueError`。                   |

```bash
pytest tests/model/test_stage1_model.py -q
```

新增或更新测试：

| 测试函数                                                              | 覆盖点                                                                                                                        |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `test_prepare_pseudo_batch_injects_anchor_pseudo_atoms`           | candidate builder + anchor sampler + density cube 启用后，`_prepare_pseudo_batch()` 返回 mixed batch 和非 None `PseudoAtomLayout`。 |
| `test_stage1_model_anchor_sampler_requires_candidate_builder`     | 启用 anchor sampler 但 candidate builder 为 None 时构造 fail-fast。                                                                |
| `test_stage1_model_anchor_sampler_requires_density_cube`          | 启用 anchor sampler 但 density cube 为 None 时构造 fail-fast。                                                                     |
| `test_stage1_model_anchor_class_ids_must_match_candidate_builder` | candidate builder 和 anchor sampler class ids 不一致时报错。                                                                       |
| `test_stage1_model_density_out_dim_checked_at_init`               | density cube `out_dim` 与 `point_backbone.atom_feature_dim` 不一致时构造时报错。                                                      |
| `test_stage1_model_anchor_outputs_reach_final_outputs`            | final `outputs` 含 `anchor_voxel_zyx`、`anchor_counts_by_class` 等字段。                                                         |
| `test_stage1_model_mixed_anchor_keeps_atom_supervision_real_only` | 注入 P 后 `atom_logits`、`atom_target`、`atom_valid_mask` 长度仍等于真实 atom 数。                                                       |
| `test_stage1_model_pseudo_feature_shape_matches_anchor_count`     | `outputs["pseudo_feature"].shape[0] == outputs["anchor_counts"].sum()`。                                                    |

```bash
pytest tests/test_voxel_ligand_thresholds.py -q
```

补充测试：

| 测试函数                                                   | 覆盖点                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `test_threshold_cache_load_rejects_best_f1_nan_or_inf` | checkpoint 中 `voxel_ligand_best_f1_by_class` 含 NaN/Inf 时 fail-fast。 |

### Config / Hydra Smoke Tests

```bash
python -m src.train +experiment=tri001_tunedloss model/candidate_set=tri model/anchor_sampler=weighted_fps model/density_cube=default trainer.fast_dev_run=true
```

验证点：三分类 C→P→density→mixed point 路径可实例化，`weighted_fps` 配置字段可被 Hydra 正确传入。

```bash
python -m src.train +experiment=tri001_tunedloss model/candidate_set=tri model/anchor_sampler=unweighted_fps model/density_cube=default trainer.fast_dev_run=true
```

验证点：`torch_cluster.fps` mode 可实例化并前向。

```bash
python -m src.train +experiment=tri001_tunedloss model/candidate_set=tri model/anchor_sampler=topk_nms model/density_cube=default trainer.fast_dev_run=true
```

验证点：`topk_nms` mode 可实例化并前向。

二分类 smoke 可使用：

```bash
python -m src.train +experiment=unet000 model/candidate_set=binary model/anchor_sampler=topk_nms model/density_cube=default model.backbone.anchor_sampler_cfg.candidate_class_ids=[1] model.backbone.anchor_sampler_cfg.max_anchors_per_class=[1024] trainer.fast_dev_run=true
```

若当前训练入口不支持 `trainer.fast_dev_run=true`，改用项目现有 fast-dev-run 参数；验证目标不变。

### Manual Code Checks

```bash
rg "max_anchors_per_box|fallback_mode|distance_metric|unweighted_prefilter_per_class" src configs CLAUDE/plans/implement/tri_ligand_sparse_refine
```

预期：04 新实现和配置不再使用这些字段；旧计划说明若命中需确认不是执行配置。

```bash
rg "anchor_counts_by_class|anchor_counts" src tests CLAUDE/plans/implement/tri_ligand_sparse_refine
```

预期：`anchor_counts_by_class` 用于 per-class 统计；`anchor_counts` 仅表示 per-BOX 总 P 数。

```bash
rg "BatchNorm3d|LayerNorm" src/model/sparse_refine/density_cube.py
```

预期：density cube encoder 不使用 BatchNorm 或 LayerNorm。

```bash
rg "torch_cluster.fps|mode: unweighted_fps|mode: weighted_fps|mode: topk_nms" src configs tests
```

预期：三种采样模式都有实现和配置；`unweighted_fps` 使用已有 `torch_cluster.fps`。

### Acceptance Criteria

1. `SparseAnchorSampler` 支持 `weighted_fps`、`unweighted_fps`、`topk_nms` 三种 mode。
2. `max_anchors_per_box` 已删除；所有 P 上限由 `max_anchors_per_class` 控制。
3. 不再配置 `distance_metric` 或 `unweighted_prefilter_per_class`。
4. 二分类配置语义明确：`candidate_class_ids=[1]`、`max_anchors_per_class=[1024]`、`anchor_counts_by_class.shape == (B,1)`。
5. 同一 `(batch, voxel_zyx)` 最多产生一个 P，保留最高 `candidate_prob` 候选类别记录，tie 保留原始行号更小者。
6. `anchor_counts_by_class` 为 `(B,K)`，`anchor_counts` 为 `(B,)` 总数，二者总和一致。
7. P 坐标 local/world/centered-world 公式与 dataset 坐标语义一致。
8. density cube encoder 输出 `pseudo_feat.shape == (sumP, point_backbone.atom_feature_dim)`。
9. density cube encoder 首版只实现 `conv_gap`，默认两层 stride=2 下采样 Conv3d 后接普通 Conv3d blocks 和 GAP，使用 GroupNorm 或 no norm，不使用 LayerNorm / BatchNorm。
10. `pseudo_feat` 不拼接 candidate class/prob；class/prob 只通过 `anchor_class` / `anchor_prob` metadata 透传。
11. `_prepare_pseudo_batch()` 在 final recycle 中完成 C→P→density→inject，返回 mixed batch 和 `PseudoAtomLayout`。
12. point backbone 收到 mixed `pseudo_mask`，atom head 输出 `pseudo_feature`，wrapper-facing atom supervised 字段仍 real-only 对齐。
13. `unweighted_fps` 不新增依赖，只使用环境已有 `torch_cluster`。
14. `topk_nms` 使用 GPU 友好的 dense grid / max-pool 实现，不写 Python 逐体素 suppression 主循环。
15. 00-master 与 04 子计划契约同步。
