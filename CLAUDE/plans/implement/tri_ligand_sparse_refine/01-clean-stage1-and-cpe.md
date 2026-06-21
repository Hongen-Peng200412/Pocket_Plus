# 01：清理 Stage1、旧伪原子逻辑与 PTV3 sparseconv 路径

## 背景与目标

本计划是三分类 ligand sparse refine 的第 1 个实施步骤。目标不是实现 C/P sparse refine，而是先清理会阻碍后续 sparse refine 的 Stage1 旧路径：旧随机伪原子生成器、伪原子进入 embed head 的生命周期逻辑、atom head 分散在主模型里的实现、以及 PTV3 点分支中的 sparseconv embedding / CPE 路径。

当前 [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 同时承担 embed head、voxel backbone、point backbone、旧随机伪原子 lifecycle/recycle、mixed/real 视图裁剪、atom head 构建与 atom head 前向。旧 [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 内置随机采样、密度加权、KDTree 距离过滤、`neighbor_mean` 初始化和 recycle policy；这些语义与后续“从 C 采样得到少量 P anchors，再作为伪原子进入点分支”的 sparse refine 范式冲突。

用户已确认：模型旧实现已全量备份到 [src/legacy/](../../../../src/legacy/)，原位置 [src/model/](../../../../src/model/) 可以任意修改；旧随机伪原子配置直接删除，只保证 [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml) 继续可用于 [configs/experiment/unet000.yaml](../../../../configs/experiment/unet000.yaml)。

本阶段完成后应满足：

1. [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 不再定义 `Stage1SerializedAttentionStack`，也不再直接持有 `atom_token_proj` / `atom_attention_stack` / `atom_logit_head` 三个零散成员，只持有 `self.atom_head: Stage1AtomHead | None`。
2. 伪原子不能进入 embed head；旧 `lifecycle=[embed_head, point_backbone, atom_head]` 语义删除。
3. 新版 P anchors 只在最后一次 recycle 创建/注入；前面的 recycle 全部 real-only。
4. [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 只管理 anchor-based P 伪原子的 mixed layout、mask、注入、移除和 tensor split/interleave 工具，不再生成伪原子。
5. `Stage1AtomHead` 使用双尾部：真实原子输出 atom logits，伪原子输出 `pseudo_feature`，不让伪原子输出 atom logits。
6. PTV3 点分支删除 sparseconv embedding 和 sparseconv CPE，只保留 pointconv / none 相关路径；voxel backbone 的普通 3D conv / UNet 不受影响。
7. 当前 voxel baseline / Unet-only 路径仍可实例化和前向。

> [!IMPORTANT]
> 本计划不实现候选集合 C、P weighted-FPS、density cube、P→C 插值、sparse C logits、refine loss，也不实现 type-aware QKV/FFN/CPE。

## 当前架构 / 已知约束

| 现状 | 位置 | 约束 |
|---|---|---|
| `Stage1SerializedAttentionStack` 定义在主模型文件内 | [src/model/stage1_model.py:29-161](../../../../src/model/stage1_model.py#L29-L161) | 迁入新文件 [src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py)，继续复用现有 `Block`。 |
| atom head 构造分散为 `atom_token_proj` / `atom_attention_stack` / `atom_logit_head` | [src/model/stage1_model.py:406-467](../../../../src/model/stage1_model.py#L406-L467) | 整体收口到 `Stage1AtomHead`。 |
| `forward()` 内存在 embed 阶段注入伪原子、`embed_split_info` 和裁剪同步 | [src/model/stage1_model.py:797-845](../../../../src/model/stage1_model.py#L797-L845) | 本阶段删除；embed head 永远只处理 real atoms。 |
| `forward()` 内存在旧 pseudo recycle 缓存和 mixed/real 视图切换 | [src/model/stage1_model.py:781-1037](../../../../src/model/stage1_model.py#L781-L1037) | 删除旧 random pseudo/recycle policy；改为“最后一次 recycle 才准备 P”的清晰时序。 |
| atom head 前向在 `forward()` 末尾手写 | [src/model/stage1_model.py:1039-1079](../../../../src/model/stage1_model.py#L1039-L1079) | 改为 `self.atom_head(...)` 一次调用；监督字段 real-only 对齐。 |
| 旧伪原子生成器内置随机采样、密度采样、KDTree 过滤、`neighbor_mean` | [src/model/pseudo_atoms.py:150-651](../../../../src/model/pseudo_atoms.py#L150-L651) | 全部删除，不保留 fallback。 |
| PTV3 `Block` 支持 sparseconv CPE | [src/model/PTV3bakcbone/model.py:1119-1210](../../../../src/model/PTV3bakcbone/model.py#L1119-L1210) | 删除 sparseconv CPE 分支，`Block.cpe_impl` 只允许 `none` / `pointconv`。 |
| PTV3 `Embedding` 支持 sparseconv stem | [src/model/PTV3bakcbone/model.py:1663-1746](../../../../src/model/PTV3bakcbone/model.py#L1663-L1746) | 删除 sparseconv embedding 分支，点分支 embedding 只允许 `pointconv`。 |
| PTV3 `Point.sparsify()` / `PointSequential` 仍维护 spconv tensor | [src/model/PTV3bakcbone/model.py:212-255](../../../../src/model/PTV3bakcbone/model.py#L212-L255)、[src/model/PTV3bakcbone/model.py:360-405](../../../../src/model/PTV3bakcbone/model.py#L360-L405) | 删除点分支 spconv 依赖后，这些 sparseconv 专用逻辑应同步移除或不再调用。 |
| `Stage1EmbedHead` 也复用 PTV3 `Block` | [src/model/stage1_embed_head.py:637-720](../../../../src/model/stage1_embed_head.py#L637-L720) | 因 `Block` 删除 sparseconv CPE，embed head 的 `cpe_impl` 也只能是 `none` / `pointconv`。 |
| Unet-only 实验依赖 `model/pseudo_atom: none` | [configs/experiment/unet000.yaml:16-24](../../../../configs/experiment/unet000.yaml#L16-L24) | [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml) 必须保留并可用。 |

## 已有可复用代码

| 已有代码 | 位置 | 可复用的能力 |
|---|---|---|
| `Stage1SerializedAttentionStack` | [src/model/stage1_model.py:29-161](../../../../src/model/stage1_model.py#L29-L161) | 直接迁入 [src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py)，本阶段不改 attention 算法。 |
| atom token 构造逻辑 | [src/model/stage1_model.py:1040-1059](../../../../src/model/stage1_model.py#L1040-L1059) | 移入 `Stage1AtomHead.forward()`，保持 token 拼接语义。 |
| atom logit head 与 prior bias 初始化 | [src/model/stage1_model.py:446-497](../../../../src/model/stage1_model.py#L446-L497) | 移入 `Stage1AtomHead`，保持多分类 prior bias 初始化语义。 |
| point state 契约 | [src/model/stage1_point_backbone.py:292-315](../../../../src/model/stage1_point_backbone.py#L292-L315) | `Stage1AtomHead` 继续消费 `coord` / `batch` / `offset` / `grid_size` / optional `grid_coord`。 |
| 旧 `inject/remove/interleave` 的布局语义 | [src/model/pseudo_atoms.py:794-1056](../../../../src/model/pseudo_atoms.py#L794-L1056) | 可借鉴 `[real_i, pseudo_i]` mixed layout，但实现应重写为 anchor-based 工具，不保留旧生成器类。 |

## 设计决策

### 1. `pseudo_atoms.py` 清空重建，而不是兼容旧生成器

旧随机伪原子与 sparse refine 的 P anchors 语义冲突。P anchors 的坐标、类别和初始特征将由后续 03/04 计划从候选体素集合 C 生成，不应由 [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 自行随机采样。

> [!IMPORTANT]
> [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 中不得再出现 `generate()`、`base_count`、`scale_factor`、`max_sample_rounds`、density random sampling、KDTree collision/deletion、`neighbor_mean`、`recycle_policy`、`lifecycle`。

### 2. P anchors 只在最后一次 recycle 注入

新版 recycle 时序为：前 `recycle_steps - 1` 轮只运行 real-only voxel/point；最后一轮在 voxel backbone 输出后调用 `_prepare_pseudo_batch(...)`，由后续 03/04 在此生成 C、采样 P、注入 P。

这样可以避免每轮都为 P anchors 构图、融合和 attention，节省时间与显存，把资源留给最后一轮更重的 P 初始化和后续 sparse refine。

### 3. `atom_valid_mask` 保持 bool，不新增 `point_kind`

本阶段保留 `atom_valid_mask: torch.bool`，继续只表达“真实原子是否参与 atom-level 监督”。不新增 `atom_point_kind` / `point_kind` 三态字段，避免 01 阶段扩大工作量。伪原子类型由 `pseudo_mask: torch.bool` 表达。

### 4. `Stage1AtomHead` 使用 real logits + pseudo feature 双尾部

`Stage1AtomHead` 的 shared attention stack 输出 mixed 全点 `atom_hidden`。尾部分离为：

- `real_atom_logit_head`：只对真实原子输出 `atom_logits`。
- `pseudo_feature_head`：只对 P anchors 输出 `pseudo_feature`。

`pseudo_feature_dim` 可配置，默认等于 `hidden_dim`。01 不实现伪原子监督，但这个结构允许未来新增 pseudo-specific supervision head。

### 5. wrapper 监督字段保持 real-only 对齐

对 wrapper/loss 暴露的 `atom_logits`、`atom_target`、`atom_valid_mask` 必须始终同长度。无 P 时它们等于当前 real-only 行为；有 P 时它们统一裁成 real-only，不让 wrapper 处理 mixed mask。`atom_tokens` / `atom_hidden` 可保持 mixed 全点，`pseudo_feature` 单独给后续 refine。

### 6. 删除 PTV3 点分支 sparseconv，不影响 voxel backbone

删除范围是 PTV3 点分支中的 sparseconv embedding、sparseconv CPE、`spconv.SparseConvTensor` 构造与 spconv module 处理路径。voxel backbone / RAUNet 中的普通 3D convolution 不属于本计划的删除范围。

### 7. 流程专用函数需要写明上下游语义

新增/重写函数中，只有“流程专用、调用点少、上下游语义强”的函数需要在中文 docstring 中说明参数来自哪个阶段、返回值被哪个下游消费。例如 `_prepare_pseudo_batch(...)`、`_run_atom_head(...)` 需要写清楚；`build_real_mask(...)` 这类通用工具只需写清 shape 和语义。

## Proposed Changes

### 1. 抽出并改造 Atom Head

#### [NEW] [src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py)

**新增模块职责**：承载 Stage1 atom head 的 token projection、serialized attention stack、real atom logit head、pseudo feature head 和 prior bias 初始化。

新增类一：

```python
class Stage1SerializedAttentionStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        atom_head_ffn_type: str,
        mlp_ratio: int,
        act_layer: type[nn.Module],
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,
    ) -> None: ...

    def forward(
        self,
        point_state: dict[str, Any],
        token_feat: torch.Tensor,
        pseudo_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

| 参数 | 类型 | 意义 |
|---|---|---|
| `point_state` | `dict[str, Any]` | 来自 point backbone 的点状态，至少包含 `coord`、`batch`、`offset`、`grid_size`，可包含 `grid_coord`。 |
| `token_feat` | `torch.Tensor` | `(N, C_hidden)`，atom token projection 后的 mixed 或 real-only hidden feature。 |
| `pseudo_mask` | `torch.Tensor | None` | `(N,)`，True 表示 P anchors；01 只写入 `Point`，02 才消费 type-aware 行为。 |

实现要求：

1. 从 [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 迁移当前 `Stage1SerializedAttentionStack`，保持 `Block(...)` 调用和 `output_norm` 行为。
2. `forward()` 重建 `Point` 时，如果 `pseudo_mask is not None`，写入 `point_dict["pseudo_mask"] = pseudo_mask`。
3. `cpe_impl` 只允许 `none` / `pointconv`，具体 fail-fast 由 `Block` 保证。

新增类二：

```python
class Stage1AtomHead(nn.Module):
    def __init__(
        self,
        point_channels: int,
        hidden_dim: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        atom_logit_dim: int,
        pseudo_feature_dim: int | None,
        atom_head_ffn_type: str,
        mlp_ratio: int,
        act_layer: type[nn.Module],
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,
        append_coord_mask: bool,
        prior_prob: float | None = None,
        prior_probs: Sequence[float] | None = None,
    ) -> None: ...

    def forward(
        self,
        point_feat: torch.Tensor,
        point_state: dict[str, Any],
        atom_coord_centered_world: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        pseudo_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]: ...
```

| 参数 | 类型 | 意义 |
|---|---|---|
| `point_channels` | `int` | point backbone 输出通道数，来自 `self.point_backbone.out_channels`。 |
| `hidden_dim` | `int` | shared atom head hidden 维度，对应旧 `atom_head_hidden_dim`。 |
| `atom_logit_dim` | `int` | 真实原子的 atom logits 输出维度，对应旧 `atom_logit_dim`。 |
| `pseudo_feature_dim` | `int | None` | 伪原子高阶特征输出维度；None 表示使用 `hidden_dim`。 |
| `append_coord_mask` | `bool` | 是否把 centered-world 坐标和 bool `atom_valid_mask` 拼入 token，保持旧语义。 |
| `prior_prob` / `prior_probs` | `float | Sequence[float] | None` | 只用于 `real_atom_logit_head` 最后一层 bias 初始化；二者不能同时配置。 |
| `point_feat` | `torch.Tensor` | `(N_all, C_point)`，最后一轮 point backbone 输出；有 P 时为 mixed 全点。 |
| `point_state` | `dict[str, Any]` | 与 `point_feat` 同布局的点状态。 |
| `atom_coord_centered_world` | `torch.Tensor` | `(N_all, 3)`，用于可选 token 拼接。 |
| `atom_valid_mask` | `torch.Tensor` | `(N_all,) bool`，用于可选 token 拼接；P anchors 应为 False。 |
| `pseudo_mask` | `torch.Tensor | None` | `(N_all,) bool`；None 表示当前为 real-only 路径。 |

返回契约：

| 返回字段 | 类型 | shape | 语义 |
|---|---|---|---|
| `atom_tokens` | `torch.Tensor` | `(N_all, C_token_in)` | shared token projection 前的 token；有 P 时为 mixed 全点。 |
| `atom_hidden` | `torch.Tensor` | `(N_all, hidden_dim)` | shared attention stack 后的 hidden；有 P 时为 mixed 全点。 |
| `atom_logits` | `torch.Tensor` | `(N_real, atom_logit_dim)` | 只对真实原子输出，不包含伪原子。 |
| `pseudo_feature` | `torch.Tensor | None` | `(N_pseudo, pseudo_feature_dim)` | 只对伪原子输出；`pseudo_mask is None` 时为 None。 |

实现步骤：

1. `self.atom_token_proj` 迁移旧 token projection 结构。
2. `self.atom_attention_stack` 使用迁移后的 `Stage1SerializedAttentionStack`。
3. `self.real_atom_logit_head` 迁移旧 `atom_logit_head` 结构。
4. 新增 `self.pseudo_feature_head`：

```python
self.pseudo_feature_dim = int(pseudo_feature_dim) if pseudo_feature_dim is not None else int(hidden_dim)
self.pseudo_feature_head = nn.Sequential(
    nn.Linear(int(hidden_dim), int(hidden_dim)),
    act_layer(),
    nn.Linear(int(hidden_dim), self.pseudo_feature_dim),
)
```

5. 将 `_init_linear_multiclass_prior_bias()` 迁入 `Stage1AtomHead`，只初始化 `real_atom_logit_head` 最后一层。
6. `forward()` 中：
   - 构造 `atom_tokens`。
   - 得到 mixed/all `atom_hidden`。
   - `pseudo_mask is None` 时，`real_hidden = atom_hidden`，`pseudo_feature = None`。
   - `pseudo_mask is not None` 时，`real_hidden = atom_hidden[~pseudo_mask]`，`pseudo_feature = self.pseudo_feature_head(atom_hidden[pseudo_mask])`。
   - `atom_logits = self.real_atom_logit_head(real_hidden)`。

> [!IMPORTANT]
> `atom_logits` 不再表示 mixed 全点 logits；它永远对应真实原子监督分支。伪原子只通过 `pseudo_feature` 向后续 sparse refine 暴露。

#### [MODIFY] [src/model/stage1_model.py](../../../../src/model/stage1_model.py)

**导入与成员收口**：

- 删除本文件内的 `Stage1SerializedAttentionStack` 类定义。
- 删除 `Point` / `SerializedAttention` / `GatedTransition` / `MLP` 等仅供旧 atom head 使用的导入。
- 新增：

```python
from src.model.stage1_atom_head import Stage1AtomHead
```

**构造函数参数新增**：

```python
atom_head_pseudo_feature_dim: int | None = None
```

| 参数 | 类型 | 默认值 | 消费位置 |
|---|---|---|---|
| `atom_head_pseudo_feature_dim` | `int | None` | `None` | 传给 `Stage1AtomHead(pseudo_feature_dim=...)`；None 表示等于 `atom_head_hidden_dim`。 |

**`__init__()` atom head 构造区**：

- 删除 `self.atom_token_proj`、`self.atom_attention_stack`、`self.atom_logit_head` 直接构造。
- `enable_atom_head=True` 时构造：

```python
self.atom_head = Stage1AtomHead(
    point_channels=int(self.point_backbone.out_channels),
    hidden_dim=int(atom_head_hidden_dim),
    num_heads=int(atom_head_num_heads),
    patch_size=int(atom_head_patch_size),
    num_layers=int(atom_head_num_layers),
    serialization_orders=atom_head_serialization_orders,
    shuffle_orders=bool(atom_head_shuffle_orders),
    qkv_bias=bool(atom_head_qkv_bias),
    qk_scale=atom_head_qk_scale,
    attn_drop=float(atom_head_attn_drop),
    proj_drop=float(atom_head_proj_drop),
    enable_rpe=bool(atom_head_enable_rpe),
    enable_flash=bool(atom_head_enable_flash),
    upcast_attention=bool(atom_head_upcast_attention),
    upcast_softmax=bool(atom_head_upcast_softmax),
    atom_logit_dim=int(atom_logit_dim),
    pseudo_feature_dim=atom_head_pseudo_feature_dim,
    atom_head_ffn_type=str(atom_head_ffn_type),
    mlp_ratio=int(atom_head_mlp_ratio),
    act_layer=act_cls,
    cpe_impl=str(atom_head_cpe_impl),
    cpe_kernel_size=int(atom_head_cpe_kernel_size),
    cpe_receptive_field=float(atom_head_cpe_receptive_field),
    pointconv_block_max_neighbors=int(atom_head_pointconv_max_neighbors),
    drop_path=float(atom_head_drop_path),
    pre_norm=bool(atom_head_pre_norm),
    append_coord_mask=bool(atom_head_append_coord_mask),
    prior_prob=prior_prob,
    prior_probs=prior_probs,
)
```

- `enable_atom_head=False` 时：

```python
self.atom_head = None
self.atom_head_append_coord_mask = False
```

- 删除 `__init__()` 尾部对 `self.atom_logit_head[2]` 的 prior 初始化。

### 2. 重写 `pseudo_atoms.py` 为 anchor-based layout 工具

#### [MODIFY] [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py)

删除旧 `PseudoAtomGenerator` 及其生成逻辑，重写为轻量 tensor 工具模块。模块不再 import `numpy`、`scipy.spatial.cKDTree`、`detect_diff_posdiff_indices`。

新增数据结构：

```python
@dataclass(frozen=True)
class PseudoAtomLayout:
    real_counts: torch.Tensor
    pseudo_counts: torch.Tensor

    @property
    def mixed_counts(self) -> torch.Tensor: ...

    @property
    def batch_size(self) -> int: ...
```

| 字段 | 类型 | shape | 不变量 |
|---|---|---|---|
| `real_counts` | `torch.Tensor` | `(B,) long` | 每个 BOX 的真实原子数，sum 等于 real-only batch 点数。 |
| `pseudo_counts` | `torch.Tensor` | `(B,) long` | 每个 BOX 的 P anchor 数，sum 等于 pseudo dict 点数。 |
| `mixed_counts` | property | `(B,) long` | `real_counts + pseudo_counts`。 |

anchor-based pseudo dict 字段契约：

| 字段 | 类型 | shape | 必需 | 语义 |
|---|---|---|---|---|
| `pseudo_coord_centered_world` | `torch.Tensor` | `(sumP, 3)` | 是 | P anchors centered-world 坐标。 |
| `pseudo_coord_local_voxel` | `torch.Tensor` | `(sumP, 3)` | 是 | P anchors local voxel corner 坐标。 |
| `pseudo_coord_world` | `torch.Tensor` | `(sumP, 3)` | 是 | P anchors world 坐标。 |
| `pseudo_feat` | `torch.Tensor` | `(sumP, F_atom)` | 是 | P anchors 初始点特征；04 由 density cube 或 zero feat 构造。 |
| `pseudo_batch_index` | `torch.Tensor` | `(sumP,)` | 是 | 每个 P anchor 所属 BOX。 |
| `pseudo_counts` | `torch.Tensor` | `(B,)` | 是 | 每个 BOX 的 P anchor 数。 |
| `pseudo_anchor_class` | `torch.Tensor` | `(sumP,)` | 否 | P anchor 类别，04 后透传。 |
| `pseudo_anchor_voxel_zyx` | `torch.Tensor` | `(sumP, 3)` | 否 | P anchor 来源体素坐标，04 后透传。 |
| `pseudo_source_candidate_index` | `torch.Tensor` | `(sumP,)` | 否 | P anchor 对应的 C index，04 后透传。 |

新增函数：

```python
def compute_atom_counts(batch: dict[str, Any]) -> torch.Tensor: ...
def build_layout(real_batch: dict[str, Any], pseudo_dict: dict[str, Any]) -> PseudoAtomLayout: ...
def build_real_mask(layout: PseudoAtomLayout, *, device: torch.device | None = None) -> torch.Tensor: ...
def build_pseudo_mask(layout: PseudoAtomLayout, *, device: torch.device | None = None) -> torch.Tensor: ...
def interleave_real_and_pseudo_tensor(
    real_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
    pseudo_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None: ...
def extract_real_tensor_from_mixed(
    mixed_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
) -> torch.Tensor | None: ...
def extract_pseudo_tensor_from_mixed(
    mixed_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
) -> torch.Tensor | None: ...
def inject_pseudo_atoms(
    real_batch: dict[str, Any],
    pseudo_dict: dict[str, Any],
) -> tuple[dict[str, Any], PseudoAtomLayout]: ...
def remove_pseudo_atoms(
    mixed_batch: dict[str, Any],
    layout: PseudoAtomLayout,
) -> dict[str, Any]: ...
def filter_point_state_with_mask(
    point_state: dict[str, Any],
    keep_mask: torch.Tensor,
    counts: torch.Tensor,
) -> dict[str, Any]: ...
def extract_real_point_output(
    mixed_batch: dict[str, Any],
    fused_point_feat: torch.Tensor,
    point_output_dict: dict[str, Any],
    layout: PseudoAtomLayout,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], dict[str, Any]]: ...
```

关键行为：

1. mixed layout 固定为每个 BOX 内 `[real_i, pseudo_i]`。
2. `interleave_real_and_pseudo_tensor()` 不接受已 mixed 的 real tensor；长度不等于 `sum(layout.real_counts)` 时直接 `RuntimeError`。
3. `pseudo_tensor is None` 时，用 `real_tensor.new_zeros((sumP,) + real_tensor.shape[1:])` 填充 pseudo slots。
4. `inject_pseudo_atoms()` 中 `atom_valid_mask` 的 pseudo slots 填 False，`atom_label` 的 pseudo slots 填 0，`atom_global_indices` 的 pseudo slots 填 -1。
5. `inject_pseudo_atoms()` 在 mixed batch 中新增 `pseudo_mask` 和 `real_mask`，均为 `(sumN + sumP,) bool`。
6. `remove_pseudo_atoms()` 删除 mixed-only 的 `pseudo_mask` / `real_mask`，恢复 real-only `atom_counts` / `atom_offsets` / `atom_batch_index`。
7. `extract_real_point_output()` 裁剪 `point_feat`、`point_state`、`point_recycle_out` 和 `point_feature_dict` 中第一维等于 mixed 点数的 tensor。

通用工具函数如 `build_real_mask()` 只需写清 shape 和语义；`inject_pseudo_atoms()` / `extract_real_point_output()` 这类流程专用函数需要在 docstring 中写明输入来自后续 anchor sampler / point backbone，返回给 `_run_point_backbone()` / wrapper 的意义。

### 3. 清理 `VolumePointStage1Model.forward()` 并固定最后一轮 P 时序

#### [MODIFY] [src/model/stage1_model.py](../../../../src/model/stage1_model.py)

**旧 pseudo config 处理**：

保留 `pseudo_atom_cfg: dict | None = None` 参数以兼容 [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml)，但只允许 None：

```python
if pseudo_atom_cfg is not None:
    raise ValueError("旧 pseudo_atom_cfg 已删除；P anchors 将由 sparse refine anchor pipeline 提供。")
```

删除 `self.pseudo_atom_gen = PseudoAtomGenerator(...)` 和 `_validate_pseudo_recycle_policy()` 调用。

**删除 embed-head 伪原子路径**：

删除以下旧逻辑：

- `lifecycle` 读取。
- `embed_split_info`。
- embed 前 `_prepare_aligned_pseudo_dict()` / `inject()`。
- embed 裁剪后 `_update_split_info_after_trim()`。
- embed 后 `_capture_pseudo_dict_from_batch()` / `remove()`。

替换后的 embed 流程：

1. `batch` 始终是 real-only。
2. `embed_output = self.embed_head(...)` 只接收真实原子字段。
3. embed head 裁剪后只同步 real atom 字段和 `atom_counts`。
4. 不再存在 embed mixed layout。

**拆分 forward 阶段函数**：

```python
def _run_embed_head_once(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]: ...
def _build_voxel_input(self, batch: dict[str, Any], embed_output: dict[str, Any] | None) -> torch.Tensor: ...
def _run_voxel_backbone(self, voxel_input: torch.Tensor, voxel_recycle_in: torch.Tensor | None) -> dict[str, Any]: ...
def _prepare_pseudo_batch(
    self,
    batch: dict[str, Any],
    voxel_output_dict: dict[str, Any],
) -> tuple[dict[str, Any], PseudoAtomLayout | None, dict[str, Any]]: ...
def _run_point_backbone(
    self,
    batch: dict[str, Any],
    voxel_output_dict: dict[str, Any],
    point_recycle_in: torch.Tensor | None,
    pseudo_layout: PseudoAtomLayout | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]: ...
def _run_atom_head(
    self,
    outputs: dict[str, Any],
    atom_head_batch: dict[str, Any],
    pseudo_layout: PseudoAtomLayout | None,
) -> None: ...
```

| 方法 | 01 阶段行为 | 后续扩展点 |
|---|---|---|
| `_run_embed_head_once` | real-only embed head 和裁剪。 | 无伪原子扩展。 |
| `_build_voxel_input` | 只构造 voxel backbone 输入张量。 | 不放 C 生成逻辑。 |
| `_run_voxel_backbone` | 包装 `self.voxel_backbone(...)`。 | 03/04 从其输出读取 `voxel_logits_ligand`。 |
| `_prepare_pseudo_batch` | 01 返回 `(batch, None, {})`。 | 03/04 在最后一次 recycle 中生成 C、采样 P、注入 P。 |
| `_run_point_backbone` | 01 只处理 real-only；若未来 `pseudo_layout` 非 None，则 interleave real recycle 输入并运行 mixed 点分支。 | 04 接入 mixed batch 和 `pseudo_mask`。 |
| `_run_atom_head` | 调用 `self.atom_head`，并保证 supervised atom 字段 real-only 对齐。 | 05 消费 `pseudo_feature`。 |

`_prepare_pseudo_batch()` 是流程专用函数，docstring 必须写明：输入 `voxel_output_dict` 来自当前 recycle 的 `_run_voxel_backbone()`；后续 03/04 将在这里用 `voxel_logits_ligand` 生成 C 并注入 P；01 暂时只返回 real-only batch。

**forward 时序**：

```python
batch, embed_output = self._run_embed_head_once(batch)
voxel_input = self._build_voxel_input(batch, embed_output)

voxel_recycle_in = None
point_recycle_in = None
outputs: dict[str, Any] = {}
last_pseudo_layout: PseudoAtomLayout | None = None

for recycle_idx in range(recycle_steps):
    voxel_output_dict = self._run_voxel_backbone(voxel_input, voxel_recycle_in)
    is_final_recycle = recycle_idx == recycle_steps - 1

    if is_final_recycle:
        point_batch, pseudo_layout, pseudo_outputs = self._prepare_pseudo_batch(batch, voxel_output_dict)
    else:
        point_batch, pseudo_layout, pseudo_outputs = batch, None, {}

    point_output_dict = self._run_point_backbone(
        batch=point_batch,
        voxel_output_dict=voxel_output_dict,
        point_recycle_in=point_recycle_in,
        pseudo_layout=pseudo_layout,
    )

    voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
    point_recycle_in = point_output_dict["point_recycle_out"]
    if self.detach_recycle_states:
        ...

    if is_final_recycle:
        last_pseudo_layout = pseudo_layout
        outputs = {..., **pseudo_outputs}

self._run_atom_head(outputs, atom_head_batch=point_batch, pseudo_layout=last_pseudo_layout)
outputs["recycle_passes_used"] = recycle_steps
return outputs
```

01 阶段 `pseudo_layout` 始终为 None，但调用位置必须固定为最后一轮。

**监督字段对齐要求**：

- `outputs["atom_tokens"]`：mixed 全点；无 P 时等于 real-only。
- `outputs["atom_hidden"]`：mixed 全点；无 P 时等于 real-only。
- `outputs["pseudo_feature"]`：pseudo-only；无 P 时 None。
- `outputs["atom_logits"]`：real-only。
- `outputs["atom_target"]`：real-only，和 `atom_logits.shape[0]` 一致。
- `outputs["atom_valid_mask"]`：real-only，和 `atom_logits.shape[0]` 一致。
- `outputs["atom_counts"]`：real-only counts。

`_run_atom_head()` 如果收到 `pseudo_layout is not None`，应使用 [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 的 real mask 工具裁剪 `atom_target` / `atom_valid_mask` / `atom_counts` 到 real-only；wrapper 不处理 mixed mask。

**删除旧辅助方法**：

从 [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 删除：

- `_align_pseudo_features_to_batch()`。
- `_prepare_aligned_pseudo_dict()`。
- `_capture_pseudo_dict_from_batch()`。
- `_expand_real_tensor_with_pseudo_slots()`。
- `_build_real_views_from_mixed_point_output()`。
- `_validate_pseudo_recycle_policy()`。
- `_update_split_info_after_trim()`。

### 4. 删除 PTV3 点分支 sparseconv 路径

#### [MODIFY] [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py)

**删除或停用 spconv import 与 sparse tensor 路径**：

- 删除 `import spconv.pytorch as spconv`。
- 删除 `Point.sparsify()` 中创建 `spconv.SparseConvTensor` 的逻辑；如果保留 `sparsify()` 方法用于兼容调用，也必须变成 no-op 或只维护非 spconv 字段，且不得生成 `sparse_conv_feat`。
- 删除 `PointSequential.forward()` 中 `spconv.modules.is_spconv_module(...)` 和 `spconv.SparseConvTensor` 分支。
- 删除所有 `sparse_conv_feat.replace_feature(...)` 同步逻辑。
- 删除 `Stage1PointBackbone._forward_ptv3()` 和 `PointTransformerV3.forward()` 中对 `point.sparsify()` 的依赖；如果方法保留 no-op，调用也可删除以避免误导。

**`Block.__init__()`**：

- 默认 `cpe_impl` 从 `"sparseconv"` 改为 `"pointconv"`。
- 删除 `elif self.cpe_impl == "sparseconv"` 分支。
- 允许值只保留 `none` / `pointconv`。
- 错误信息改为：

```python
raise ValueError(f"Block: cpe_impl 必须是 'pointconv' 或 'none', 当前为 '{self.cpe_impl}'")
```

**`Embedding.__init__()`**：

- 默认 `embedding_impl` 从 `"sparseconv"` 改为 `"pointconv"`。
- 删除 sparseconv stem 分支。
- 只构造 `PointConvEmbedding`。
- 若保留 `embedding_impl` 参数，则只允许 `"pointconv"`：

```python
if self.embedding_impl != "pointconv":
    raise ValueError(f"Embedding: embedding_impl 只支持 'pointconv', 当前为 '{self.embedding_impl}'")
```

**`PointTransformerV3.__init__()`**：

- 默认 `embedding_impl="pointconv"`。
- 默认 `cpe_impl="pointconv"`。
- 删除 sparseconv CPE kernel size 校验分支。
- 保留 pointconv receptive field 校验。
- `enc_cpe_kernel_size` / `dec_cpe_kernel_size` 可以暂时保留在签名中以减少配置连锁修改，但注释必须说明当前 pointconv CPE 不消费这些字段。

#### [MODIFY] [src/model/stage1_point_backbone.py](../../../../src/model/stage1_point_backbone.py)

- `_forward_ptv3()` 中删除 `point.sparsify()` 调用或让其不再依赖 spconv。
- `embedding_impl` 参数保留但只允许 `pointconv`；传入其它值 fail-fast。
- docstring 中 `embedding_impl` 允许值改为只支持 `pointconv`。
- `cpe_impl` 允许值改为 `pointconv` / `none`。

#### [MODIFY] [src/model/stage1_embed_head.py](../../../../src/model/stage1_embed_head.py)

- 更新 `cpe_impl` docstring，允许值从 `none` / `sparseconv` / `pointconv` 改为 `none` / `pointconv`。
- `cpe_kernel_size` 注释改为 legacy/no-op for pointconv；不要再写“sparseconv CPE 卷积核”。
- 依赖 `Block` 的 fail-fast，不在 embed head 内保留 sparseconv 分支。

> [!WARNING]
> 删除的是 PTV3 点分支的 sparseconv 依赖；不要改 voxel backbone / RAUNet 的普通 Conv3d。

### 5. 配置更新

#### [VERIFY ABSENT] [configs/model/pseudo_atom/64_2_non.yaml](../../../../configs/model/pseudo_atom/64_2_non.yaml)

该文件用户已手动删除。实现时确认不要重新创建。

#### [MODIFY] [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml)

保持：

```yaml
# @package _global_
model:
  backbone:
    pseudo_atom_cfg: null
```

该配置继续用于 [configs/experiment/unet000.yaml](../../../../configs/experiment/unet000.yaml)。

#### [MODIFY] [configs/model/atom_head/default.yaml](../../../../configs/model/atom_head/default.yaml)、[configs/model/atom_head/none.yaml](../../../../configs/model/atom_head/none.yaml)、[configs/model/atom_head/stardard.yaml](../../../../configs/model/atom_head/stardard.yaml)

新增扁平字段：

```yaml
atom_head_pseudo_feature_dim: null  # int|null, null 表示等于 atom_head_hidden_dim
```

更新 `atom_head_cpe_impl` 注释：允许值只写 `"none" / "pointconv"`。

#### [MODIFY] [configs/model/point_backbone/default.yaml](../../../../configs/model/point_backbone/default.yaml)、[configs/model/point_backbone/zeros.yaml](../../../../configs/model/point_backbone/zeros.yaml)、[configs/model/point_backbone/stardard.yaml](../../../../configs/model/point_backbone/stardard.yaml)

- `embedding_impl` 注释改为只支持 `pointconv`。
- `cpe_impl` 注释改为只支持 `pointconv` / `none`。
- `embedding_kernel_size` 注释改为 legacy/no-op for pointconv，或保留为旧字段但明确当前不消费。
- `enc_cpe_kernel_size` / `dec_cpe_kernel_size` 注释改为 legacy/no-op for pointconv。

#### [MODIFY] [configs/model/embed_head/*.yaml](../../../../configs/model/embed_head/)

所有 embed head 配置中的 `cpe_impl` 注释只允许 `none` / `pointconv`。当前值为 `pointconv` 或 `none` 的配置不需要改值。

#### [NO CHANGE] [configs/experiment/old/](../../../../configs/experiment/old/)

不修复其中引用旧 pseudo atom 配置的历史实验。

### 6. 删除旧测试与旧检查脚本，重写新测试

#### [DELETE] [scripts/check_stage1_pseudo_atom_pipeline.py](../../../../scripts/check_stage1_pseudo_atom_pipeline.py)

删除旧 pseudo recycle regression runner，不做兼容改写。

#### [NEW] [tests/model/test_pseudo_atoms.py](../../../../tests/model/test_pseudo_atoms.py)

新增 anchor-based layout 单测。

| 测试函数 | 构造 | 关键断言 |
|---|---|---|
| `test_build_layout_counts_match` | real batch 两个 BOX，pseudo_counts `[1, 2]` | `real_counts`、`pseudo_counts`、`mixed_counts` 正确。 |
| `test_inject_pseudo_atoms_interleaves_fields` | real `[2,1]`，pseudo `[1,2]` | mixed 顺序为 `[real0, pseudo0, real1, pseudo1]`；`atom_offsets`、`atom_batch_index` 正确。 |
| `test_build_real_and_pseudo_mask` | 同上 | real/pseudo mask 与 mixed 顺序一致。 |
| `test_remove_pseudo_atoms_restores_real_batch` | 先 inject 再 remove | real atom 字段、counts、offsets 与原始 batch 一致。 |
| `test_extract_real_and_pseudo_tensor_from_mixed` | 构造 mixed tensor | real/pseudo 裁剪结果正确。 |
| `test_interleave_rejects_wrong_lengths` | 错误 real/pseudo 长度 | 抛出 `RuntimeError`。 |
| `test_no_legacy_generator_symbols` | import 新模块 | 无 `PseudoAtomGenerator`、`generate`、`recycle_policy`。 |

#### [NEW] [tests/model/test_stage1_atom_head.py](../../../../tests/model/test_stage1_atom_head.py)

新增 atom head 测试。

| 测试函数 | 构造 | 关键断言 |
|---|---|---|
| `test_stage1_atom_head_real_only_shapes` | `pseudo_mask=None`，N=5 | `atom_logits.shape == (5, atom_logit_dim)`；`pseudo_feature is None`。 |
| `test_stage1_atom_head_mixed_outputs_split_heads` | N=6，2 个 pseudo | `atom_hidden.shape[0] == 6`；`atom_logits.shape[0] == 4`；`pseudo_feature.shape[0] == 2`。 |
| `test_stage1_atom_head_pseudo_feature_dim_configurable` | `pseudo_feature_dim=32` | `pseudo_feature.shape[-1] == 32`。 |
| `test_stage1_atom_head_append_coord_mask_shape` | `append_coord_mask=True` | `atom_tokens.shape[-1] == point_channels + 4`。 |
| `test_stage1_atom_head_multiclass_prior_bias` | `prior_probs=[0.8, 0.1, 0.1]` | real logit head 最后一层 bias 等于 `log(prior_probs)`。 |

#### [NEW] [tests/model/test_stage1_model.py](../../../../tests/model/test_stage1_model.py)

新增 Stage1 结构和时序测试。

| 测试函数 | 覆盖点 |
|---|---|
| `test_stage1_model_holds_single_atom_head_member` | `enable_atom_head=True` 时有 `model.atom_head`，没有 `atom_token_proj` / `atom_attention_stack` / `atom_logit_head` 零散成员。 |
| `test_stage1_model_rejects_non_null_legacy_pseudo_atom_cfg` | 传入旧字段 dict 时 fail-fast。 |
| `test_stage1_model_unet_only_allows_pseudo_atom_none` | `enable_atom_head=False`、`pseudo_atom_cfg=None` 可构造。 |
| `test_prepare_pseudo_batch_called_only_on_final_recycle` | monkeypatch `_prepare_pseudo_batch` 计数，`max_recycles=3` 时只调用 1 次且发生在最后一轮。 |
| `test_atom_supervision_outputs_are_real_only_aligned` | 构造 fake mixed final outputs | `atom_logits`、`atom_target`、`atom_valid_mask` 同长度。 |

#### [NEW] [tests/model/test_ptv3_no_sparseconv.py](../../../../tests/model/test_ptv3_no_sparseconv.py)

新增 PTV3 sparseconv 删除测试。

| 测试函数 | 覆盖点 |
|---|---|
| `test_block_rejects_sparseconv_cpe` | `Block(cpe_impl="sparseconv")` 抛 `ValueError`。 |
| `test_embedding_rejects_sparseconv_impl` | `Embedding(embedding_impl="sparseconv")` 抛 `ValueError` 或无该参数路径。 |
| `test_point_transformer_defaults_pointconv` | `PointTransformerV3` 默认 `embedding_impl` / `cpe_impl` 为 `pointconv`。 |
| `test_ptv3_module_has_no_spconv_dependency` | 新 PTV3 模块不再暴露 `spconv.SubMConv3d` 相关成员。 |

## 不修改的部分

- 不实现 `src/model/sparse_refine/candidate_set.py`、`anchor_sampler.py`、`density_cube.py`、`interpolation.py`、`sparse_refine_head.py`。
- 不实现 C 生成、P weighted-FPS、P density cube 特征、P→C three-nn 插值、C sparse logits。
- 不实现 type-aware QKV / attention proj / FFN / CPE；本阶段只传递 `pseudo_mask`，不消费 type-aware 行为。
- 不新增 `atom_point_kind` / `point_kind`。
- 不修改 [configs/experiment/old/](../../../../configs/experiment/old/) 中的历史实验。
- 不改 wrapper 的 loss/metrics 逻辑；模型输出必须保证 wrapper 看到的 atom supervised 字段 real-only 对齐。
- 不改 voxel backbone / RAUNet 的普通 3D convolution。

## 改动文件汇总

| 文件 | 改动内容 |
|---|---|
| [src/model/stage1_atom_head.py](../../../../src/model/stage1_atom_head.py) | 新增 `Stage1SerializedAttentionStack`、`Stage1AtomHead`、real/pseudo 双尾部。 |
| [src/model/stage1_model.py](../../../../src/model/stage1_model.py) | 删除内联 atom head、旧伪原子 lifecycle/recycle/split 逻辑；固定最后一轮 `_prepare_pseudo_batch()` 时序。 |
| [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) | 清空重写为 anchor-based P mixed layout 工具。 |
| [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) | 删除 PTV3 点分支 sparseconv embedding/CPE/SparseConvTensor 路径。 |
| [src/model/stage1_point_backbone.py](../../../../src/model/stage1_point_backbone.py) | 删除 point.sparsify 依赖，embedding/cpe 只允许 pointconv/none。 |
| [src/model/stage1_embed_head.py](../../../../src/model/stage1_embed_head.py) | 更新 CPE 允许值和注释，跟随 `Block` 删除 sparseconv CPE。 |
| [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml) | 保持 `pseudo_atom_cfg: null` 可用。 |
| [configs/model/atom_head/*.yaml](../../../../configs/model/atom_head/) | 新增 `atom_head_pseudo_feature_dim`，更新 CPE 注释。 |
| [configs/model/point_backbone/*.yaml](../../../../configs/model/point_backbone/) | 更新 embedding/CPE 注释，标记 kernel size 字段不再消费。 |
| [configs/model/embed_head/*.yaml](../../../../configs/model/embed_head/) | 更新 CPE 注释。 |
| [scripts/check_stage1_pseudo_atom_pipeline.py](../../../../scripts/check_stage1_pseudo_atom_pipeline.py) | 删除旧 pseudo recycle runner。 |
| [tests/model/test_pseudo_atoms.py](../../../../tests/model/test_pseudo_atoms.py) | 新增 layout 工具测试。 |
| [tests/model/test_stage1_atom_head.py](../../../../tests/model/test_stage1_atom_head.py) | 新增 atom head 双尾部测试。 |
| [tests/model/test_stage1_model.py](../../../../tests/model/test_stage1_model.py) | 新增 Stage1 结构/时序测试。 |
| [tests/model/test_ptv3_no_sparseconv.py](../../../../tests/model/test_ptv3_no_sparseconv.py) | 新增 PTV3 sparseconv 删除测试。 |

## Verification Plan

### Automated Tests

```bash
pytest tests/model/test_pseudo_atoms.py -q
```

验证点：anchor-based pseudo layout、inject/remove、mask、split/interleave 工具正确，旧生成器入口不存在。

```bash
pytest tests/model/test_stage1_atom_head.py -q
```

验证点：`Stage1AtomHead` real-only 与 mixed 输入 shape 正确，real logits / pseudo feature 双尾部正确，`pseudo_feature_dim` 可配置，prior bias 语义保持。

```bash
pytest tests/model/test_stage1_model.py -q
```

验证点：主模型只持有 `self.atom_head`，旧 pseudo cfg fail-fast，`_prepare_pseudo_batch()` 只在最后一次 recycle 调用，wrapper-facing atom supervised 字段 real-only 对齐。

```bash
pytest tests/model/test_ptv3_no_sparseconv.py -q
```

验证点：PTV3 点分支 sparseconv embedding/CPE 被删除，`sparseconv` 配置 fail-fast，默认路径为 pointconv。

```bash
pytest tests/model/test_online_pdb_feature.py tests/test_multiclass_voxel_backbone.py tests/test_multiclass_ligand_wrapper.py -q
```

验证点：online PDB scatter、voxel ligand baseline 和 wrapper 基础路径不受本次清理影响。

### Config / Hydra Smoke Tests

```bash
python -m src.train +experiment=unet000 trainer.fast_dev_run=true
```

验证点：

1. [configs/experiment/unet000.yaml](../../../../configs/experiment/unet000.yaml) 仍可通过 [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml) 组合。
2. `pseudo_atom_cfg: null` 不触发旧伪原子路径。
3. `enable_atom_head=false` 时不构造 `Stage1AtomHead`，Unet-only 前向仍可跑通。

如果当前训练入口不支持 `trainer.fast_dev_run=true`，改用项目现有 fast-dev-run 参数；验证目标不变。

### Manual Code Checks

```bash
rg "PseudoAtomGenerator|neighbor_mean|recycle_policy|embed_split_info|_align_pseudo_features_to_batch|_update_split_info_after_trim" src/model configs/model
```

预期：除 [src/legacy/](../../../../src/legacy/) 留档外，新模型和配置中无旧符号。

```bash
rg "spconv|SparseConvTensor|SubMConv3d|sparse_conv_feat|embedding_impl=.*sparseconv|cpe_impl=.*sparseconv" src/model/PTV3bakcbone src/model/stage1_point_backbone.py configs/model
```

预期：PTV3 点分支和模型配置不再含 sparseconv embedding/CPE 入口；若其它非点分支模块命中，需要逐项确认不是本计划范围。

### Acceptance Criteria

1. [src/model/stage1_model.py](../../../../src/model/stage1_model.py) 不再定义 `Stage1SerializedAttentionStack`，不再包含旧 pseudo helper。
2. [src/model/pseudo_atoms.py](../../../../src/model/pseudo_atoms.py) 不再生成伪原子，只提供 anchor-based layout 工具。
3. `_prepare_pseudo_batch()` 只在最后一次 recycle 调用；01 阶段返回 real-only batch 和空 metadata。
4. `Stage1AtomHead` 输出 mixed `atom_tokens` / `atom_hidden`、real-only `atom_logits`、pseudo-only `pseudo_feature`。
5. `atom_logits` / `atom_target` / `atom_valid_mask` 对 wrapper 始终同长度。
6. PTV3 点分支不再依赖 spconv，`sparseconv` embedding/CPE 配置 fail-fast。
7. [configs/model/pseudo_atom/none.yaml](../../../../configs/model/pseudo_atom/none.yaml) 仍可用于 [configs/experiment/unet000.yaml](../../../../configs/experiment/unet000.yaml)。
8. 新测试覆盖 layout、atom head、Stage1 时序和 PTV3 sparseconv 删除。
