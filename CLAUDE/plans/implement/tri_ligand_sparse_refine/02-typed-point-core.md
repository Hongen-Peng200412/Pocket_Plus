# 02：typed point core 与 real/pseudo 参数分离

## 背景 / 目标

本计划是三分类 ligand sparse refine 的第 2 个实施步骤，目标是在 01 已清理完成的 Stage1 / PTV3 点分支上，加入 mixed real atom / pseudo anchor 点云的核心支持，并在关键点特征路径上按 `pseudo_mask` 使用 real/pseudo 两套参数。

01 阶段已完成：

1. `Stage1AtomHead` 已从 `stage1_model.py` 抽出。
2. `pseudo_atoms.py` 已提供 anchor-based mixed layout 工具，mixed 顺序为每个 BOX 内 `[real_i..., pseudo_i...]`。
3. P anchors 只允许在最后一次 recycle 后进入 point backbone。
4. PTV3 sparseconv embedding / CPE 路径已删除，只保留 `pointconv` / `none`。
5. `Stage1SerializedAttentionStack.forward(..., pseudo_mask)` 已能把 `pseudo_mask` 写入 atom head 内部重建的 `Point`。

本阶段完成后应满足：

1. mixed 点云中保留一次 mixed attention，real 与 pseudo 在 attention 中互相可见。
2. 默认启用 `typed_point_cfg.enabled=true`，并默认分离 QKV、attention output projection、FFN、CPE、embedding、pooling projection、unpooling projection、voxel-to-point fusion、point input projection 和 atom token projection。
3. CPE 与 embedding 的 typed 图语义固定为同类子图：real 只看 real 邻居，pseudo 只看 pseudo 邻居；跨类型交互统一交给 mixed attention。
4. 所有会重建 `Point` / `point_state` / `point_dict` 的位置都必须维护已有 `pseudo_mask`，包括 `PointTransformerV3.forward()`、`Stage1PointBackbone._forward_ptv3()`、`PointConvEmbedding`、`SerializedPooling`、`SerializedUnpooling`、`Stage1PointBackbone._export_point_state()`、zeros backend、fusion hook 和 atom head attention stack。
5. `SerializedPooling` 必须使用 type-aware cluster key 防止同一 spatial cell 中的 real 与 pseudo 被合并；但 `serialized_code` 仍必须保持纯空间序列化编码，不能混入 type bit。
6. 显式支持 all-real / all-pseudo / mixed 三种运行状态；配置启用分参不代表每个 batch、每个 stage、每个子图都同时存在 real 与 pseudo。

> \[!IMPORTANT]
> 本计划不实现候选集合 C、P 采样、density cube、P→C 插值、sparse C logits、refine loss 或 refine metrics。这里只实现 mixed real/P 点云核心能力。

> \[!IMPORTANT]
> [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) 中已有注释是用户手写的重要说明。实施本计划时不得私自删除或大幅重写既有注释；只有代码语义变化导致原注释不准确或误导时，才做最小化同步更新。

## 本次 grill 后的覆盖性决策

若本节与下文任何细节冲突，以本节为准。

1. 默认仍使用 `typed_point_cfg.enabled=true`；同时需要提供一套显式打开 typed point 和一套显式关闭 typed point 的配置，便于后续对照。当前项目此前没有跑过正式训练，因此本阶段不需要为旧 shared checkpoint 保持权重兼容，也不要为旧权重加载增加 fallback。
2. `PointConvEmbedding` 必须保留每个点自身信息：shared path 与 typed real/pseudo path 都实现 `neighbor_agg + self_proj(input_feat)`，其中 `self_proj` 为 `nn.Linear(in_channels, embed_channels)`。radius graph 仍保持 `loop=False`，typed 子图仍只看同类邻居；孤立 real/pseudo 点不应被 embedding 清零。
3. CPE typed 模式不增加 self/residual 投影；`PointConvCPE` 仍只输出邻域 delta，孤立同类子图 delta 为零，外层 `Block.forward()` 的 CPE 残差负责保留输入特征。
4. `SerializedPooling` 的 type-aware `cluster_key` 只用于 pooling 分组，`serialized_code` 必须继续保持纯空间编码，不写入 type bit；同一 spatial cell 内的 real/pseudo pooled token 通过稳定 tie-break 保持确定顺序。
5. 本阶段接受 `interleave_real_and_pseudo_tensor(point_recycle_in, layout)` 对 P anchor recycle 槽位补零；后续 P→C / density 阶段再替换为更有信息的 P anchor 初始化。本阶段不要额外引入 density/C 依赖。

## 记号与命名约定

| 记号           | 意义                                                    |
| ------------ | ----------------------------------------------------- |
| `B`          | batch 内 BOX 数量。                                       |
| `N_all`      | 当前层 / 当前 point 对象中的全部点数，包含 real atom 与 pseudo anchor。 |
| `N_real`     | 当前层 / 当前 point 对象中的 real atom 点数。                     |
| `N_pseudo`   | 当前层 / 当前 point 对象中的 pseudo anchor 点数。                 |
| `N_pool_all` | pooling 后当前 stage 的全部 pooled token 数。                 |
| `C_in`       | 当前模块输入通道数。                                            |
| `C_out`      | 当前模块输出通道数。                                            |
| `C_point`    | point backbone 输出通道数。                                 |
| `C_hidden`   | atom head hidden 通道数。                                 |

所有新增接口、计划表格和测试描述中，mixed 点数统一写 `N_all`，真实点写 `N_real`，伪点写 `N_pseudo`。除非是泛型 tensor helper 的内部局部变量，否则不使用模糊的 `N` 表达 mixed 点数。

## 已有可复用代码

| 已有代码                       | 位置                                                                             | 可复用能力                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pseudo_mask` mixed 契约     | [src/model/pseudo\_atoms.py](../../../../src/model/pseudo_atoms.py)            | `inject_pseudo_atoms()` 输出 `(N_all,) bool` 的 `pseudo_mask` / `real_mask`，mixed 顺序为每个 BOX 内 `[real_i..., pseudo_i...]`；`filter_point_state_with_mask()` 已能裁剪 `point_state["pseudo_mask"]`。 |
| atom head pseudo mask 入口   | [src/model/stage1\_atom\_head.py](../../../../src/model/stage1_atom_head.py)   | `Stage1SerializedAttentionStack.forward(..., pseudo_mask)` 已在重建 `Point` 时写入 `point_dict["pseudo_mask"]`。                                                                                  |
| atom head 双尾部              | [src/model/stage1\_atom\_head.py](../../../../src/model/stage1_atom_head.py)   | `Stage1AtomHead` 已输出 real-only `atom_logits` 和 pseudo-only `pseudo_feature`。                                                                                                              |
| final recycle mixed 时序     | [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)            | `VolumePointStage1Model._run_point_backbone()` 已有 `pseudo_layout` 参数，适合向 point backbone 透传 `pseudo_mask`。                                                                                 |
| voxel-to-point fusion hook | [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)            | `_fuse_point_variable()` 在命名 point 变量处执行 voxel-to-point fusion，可按当前 `point_like["pseudo_mask"]` 分参。                                                                                       |
| pointconv embedding        | [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) | `PointConvEmbedding` 已有 radius graph、`w_v`、`mlp_w`、`norm`、`act` 流程，可扩展为同类子图 typed embedding。                                                                                              |
| pointconv CPE              | [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) | `PointConvCPE` 已有 radius graph + cache key，可扩展为 real/pseudo subset graph。                                                                                                                 |
| pooling trace              | [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) | `SerializedPooling` 已保存 `pooling_parent` / `pooling_inverse`，`SerializedUnpooling` 可回到父分辨率。                                                                                               |
| 现有测试骨架                     | [tests/model/](../../../../tests/model/)                                       | 已有 `test_ptv3_no_sparseconv.py`、`test_stage1_atom_head.py`、`test_stage1_model.py`、`test_pseudo_atoms.py` 可扩展。                                                                             |

## 设计决策

### 1. 默认启用 type-aware 分参

`typed_point_cfg.enabled` 默认值为 `true`。`enabled=false` 仍保留为显式 baseline / ablation 开关。所有 `use_separate_*` effective property 都必须被 `enabled` gate：

```python
use_separate_qkv = enabled and separate_qkv
```

默认配置：

```yaml
typed_point_cfg:
  enabled: true
  separate_qkv: true
  separate_attn_proj: true
  separate_ffn: true
  separate_cpe: true
  separate_embedding: true
  separate_pooling_proj: true
  separate_unpooling_proj: true
  separate_fusion: true
  separate_point_input_proj: true
  separate_atom_token_proj: true
```

> \[!IMPORTANT]
> 旧字段 `separate_token_proj` 不保留兼容映射。若配置或调用传入该 key，`normalize_typed_point_cfg()` 必须按未知字段 fail-fast。

### 2. `typed_point_cfg` 使用全局共享字典

`typed_point_cfg` 放在 `model.backbone.typed_point_cfg`，作为 point backbone、atom head 和 fusion 的统一语义来源。

由于 Hydra 默认递归实例化 nested `_target_`，`point_backbone` 可能在 `VolumePointStage1Model.__init__()` 前已经被构造。因此 point backbone 配置文件仍需通过插值显式接收同一份全局字典：

```yaml
model:
  backbone:
    typed_point_cfg:
      enabled: true
      separate_qkv: true
      separate_attn_proj: true
      separate_ffn: true
      separate_cpe: true
      separate_embedding: true
      separate_pooling_proj: true
      separate_unpooling_proj: true
      separate_fusion: true
      separate_point_input_proj: true
      separate_atom_token_proj: true

    point_backbone:
      typed_point_cfg: ${model.backbone.typed_point_cfg}
```

`VolumePointStage1Model` 负责把同一份 parsed config 传给 atom head 和 fusion；`Stage1PointBackbone` 自己解析通过 Hydra interpolation 收到的配置。

### 3. 参数能分离就分离，明确例外

本阶段原则：只要 real 与 pseudo 在同一个模块或同一条点特征路径中共处，且该模块有可学习参数，就默认支持 real/pseudo 分参。

`PointTransformerV3` 作为底层 PTV3 core 继续接收拆开的 bool flags，不直接接收 `TypedPointConfig` 对象或字典。PTV3 core 可以 import 纯 helper 函数（`validate_pseudo_mask()`、`apply_type_aware_tensor_module()`），但不得消费项目级配置 dataclass。

本阶段纳入分参的模块：

| 模块 / 路径                                                  | 分参字段                        |   默认 | 说明                                                                             |
| -------------------------------------------------------- | --------------------------- | ---: | ------------------------------------------------------------------------------ |
| `SerializedAttention` QKV                                | `separate_qkv`              | true | QKV projection 分参，但 attention 本体仍 mixed 一次计算。                                  |
| `SerializedAttention` output projection                  | `separate_attn_proj`        | true | attention 输出后按 type 投影。                                                        |
| `Block` FFN                                              | `separate_ffn`              | true | `MLP` 或 `GatedTransition` 分参。                                                  |
| `Block` CPE                                              | `separate_cpe`              | true | 同类子图 CPE 分参；atom head CPE 也跟随该字段。                                              |
| `PointConvEmbedding`                                     | `separate_embedding`        | true | 同类子图 embedding 分参。                                                             |
| `SerializedPooling.proj` projection path                 | `separate_pooling_proj`     | true | pooling 前 Linear 分参；pooling cluster 也 type-aware；pooled 后 typed LayerNorm/act。 |
| `SerializedUnpooling.proj` / `proj_skip` projection path | `separate_unpooling_proj`   | true | decoder unpooling 输入与 skip projection 分参。                                      |
| voxel-to-point fusion MLP                                | `separate_fusion`           | true | 每个 fusion hook 的 MLP 分参，mask 只来自当前 `point_like`。                               |
| point input projection                                   | `separate_point_input_proj` | true | `Stage1PointBackbone` 的 atom feature -> point input projection 分参。             |
| atom token projection                                    | `separate_atom_token_proj`  | true | `Stage1AtomHead` 的 atom token projection 分参。                                   |

明确不分离的例外：

1. **Block 级 LayerNorm / atom stack output norm**：不新增 `separate_layernorm`，不拆 `Block.norm1` / `Block.norm2` / atom stack `output_norm`。
2. **Dropout / DropPath**：不分离随机正则路径，不新增两套 drop path。
3. **Attention 本体**：不复制两份 attention 计算；real 与 pseudo 必须在同一次 mixed attention 中互相可见。
4. **CPE / embedding cross-type graph**：不实现 mixed graph typed-param 版本。CPE 与 embedding 的 typed 图都只允许同类邻域。
5. **shared old path 的归一化体系**：`typed_point_cfg.enabled=false` 或对应 `separate_*` 为 false 时，旧 shared path 继续使用当前 BN/PDNorm/LayerNorm 选择，不随 typed path 改动。

### 4. all-real / all-pseudo / mixed 必须全部正确

启用分参只表示模块拥有 real 与 pseudo 两套参数，不表示每次 forward 都同时存在两类点。

必须支持：

| 情况          | `pseudo_mask`   | 行为                                                                                  |
| ----------- | --------------- | ----------------------------------------------------------------------------------- |
| real-only   | `None` 或全 False | 只调用 real 分支，不调用 pseudo 分支。                                                          |
| pseudo-only | 全 True          | 只调用 pseudo 分支，不调用 real 分支。                                                          |
| mixed       | 同时有 True/False  | 分别 slice 两类点，调用各自分支，再 scatter 回原顺序。                                                 |
| 空点          | `N_all=0`       | tensor helper 调 real 分支自然得到空输出；已有明确空 batch fast path 可保留，只要 shape 正确且不调用 pseudo 分支。 |

`pseudo_mask is None` 与全 False mask 的字段契约保持区别：

* `None` 表示当前路径未携带 type 标注，输出不强制补全 False mask。
* 全 False 表示显式携带 type mask 但当前全 real，所有重建 `Point` / `point_state` 的路径必须继续透传全 False mask。

### 5. CPE 与 embedding 都使用同类子图

`separate_cpe=True`：

```text
real subset   -> cpe_real   -> scatter 回 real 槽位
pseudo subset -> cpe_pseudo -> scatter 回 pseudo 槽位
```

`separate_embedding=True`：

```text
real subset   -> embedding_real   -> scatter 回 real 槽位
pseudo subset -> embedding_pseudo -> scatter 回 pseudo 槽位
```

同类子图定义：

| 子图        | 目标点           | 可见邻居          | 不允许       |
| --------- | ------------- | ------------- | --------- |
| real 子图   | real points   | real points   | pseudo 邻居 |
| pseudo 子图 | pseudo points | pseudo points | real 邻居   |

`PointConvEmbedding.forward_subset()` 与 `PointConvCPE.forward_subset()` 继续使用现有 `_radius_graph_eager(..., loop=False)` 语义，不新增 self-loop。若某个点没有同类邻居，来自邻域的聚合信息为零，而由外部的残差连接维持信息；这是本阶段显式语义，不在 typed path 中偷偷补 residual 或 self-loop。

CPE graph cache 只缓存 `edge_index`，不缓存 delta。多个同 stage Block 可以复用同一个 real/pseudo graph，但每个 Block 使用自己的参数重新计算 delta。

### 6. typed path 小分支 norm 统一用普通 LayerNorm

typed 分支中可能出现 all-pseudo、单个 pseudo pooled token 或极小同类子图。为避免 `BatchNorm1d` 在训练态或推理边缘路径中因小样本统计爆掉：

1. `separate_embedding=True` 时，`PointConvEmbedding` 的 real/pseudo 两套分支 norm 固定使用普通 `nn.LayerNorm(embed_channels)`。
2. `separate_pooling_proj=True` 时，pooling 后 real/pseudo pooled token 的 norm 固定使用普通 `nn.LayerNorm(out_channels)`。
3. `separate_unpooling_proj=True` 时，child projection 与 parent skip projection 的 typed path norm 固定使用普通 `nn.LayerNorm(out_channels)`。
4. typed path 不接入 `PDNorm`；shared old path 继续按当前 `pdnorm_bn` / `pdnorm_ln` 配置走。

> \[!IMPORTANT]
> 这是 typed path 专属变更。不要把 shared old path 的 BN/PDNorm 体系一起改成 LayerNorm，否则会破坏 `enabled=false` baseline 的可比性。

### 7. Pooling 使用 type-aware cluster key，但 serialized code 保持纯空间语义

`SerializedPooling` 中必须严格区分四个概念：

| 名称                     | shape                   | 来源                                             | 用途                                                                                   | 是否含 type bit |
| ---------------------- | ----------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ | ------------ |
| `spatial_code`         | `(K_order, N_all)`      | `point.serialized_code >> (pooling_depth * 3)` | 每个序列化 order 下的纯空间 pooled cell code。                                                  | 否            |
| `spatial_code_primary` | `(N_all,)`              | `spatial_code[0]`                              | 当前 pooling 使用第 0 个 order 的纯空间 cell code。                                             | 否            |
| `type_bit`             | `(N_all,)`              | `pseudo_mask.long()`，real\=0、pseudo\=1         | 区分 real/pseudo。                                                                      | 是            |
| `cluster_key`          | `(N_all,)`              | `spatial_code_primary * 2 + type_bit`          | 仅用于 `torch.unique(..., return_inverse=True, return_counts=True)` 形成 pooling cluster。 | 是            |
| `serialized_code`      | `(K_order, N_pool_all)` | `spatial_code[:, head_indices]`                | pooling 后新 `Point` 的序列化编码。                                                           | 否            |

关键约束：

1. `cluster_key` 只用于 `torch.unique()` 得到 `cluster` / `counts` / `indices` / `head_indices`。
2. `cluster_key` 和 `code_` 不能写入 `point_dict["serialized_code"]`。
3. `point_dict["serialized_code"]` 必须使用纯空间 `spatial_code[:, head_indices]`。
4. typed pooling 前必须检查 `spatial_code_primary` 非负且 `< 1 << 62`；否则 fail-fast。
5. 同一 spatial cell 中 real 和 pseudo 必须成为两个 pooled token。
6. real/pseudo 同 spatial code 会产生 tie；排序时 `serialized_code` 仍保持纯空间，但使用 `head_indices` 作为 secondary tie-breaker 生成稳定 `serialized_order` / `serialized_inverse`，避免不确定 tie 顺序。

> \[!WARNING]
> 这里最容易出错的是把 `cluster_key` 当成新的空间 code。`cluster_key` 是 pooling 分组键，不是几何序列化 code。后续 attention serialization 必须继续使用纯空间 `serialized_code`。

### 8. Pooling 分参保持当前数学顺序

当前 shared `SerializedPooling.forward()` 顺序是：

```text
Linear(point.feat) -> segment_csr pooling -> pooled Point -> norm -> act
```

typed path 必须保持这个数学顺序，不得把 norm/act 提前到 pooling 前。

`separate_pooling_proj=True` 时顺序为：

```text
apply typed Linear to point.feat
-> segment_csr pooling
-> 得到 pooled_feat 与 pooled pseudo_mask
-> 对 pooled_feat 按 pooled pseudo_mask 做 typed LayerNorm/act
```

> \[!IMPORTANT]
> 不采用 `Linear -> norm -> act -> segment_csr` 的 pre-pooling 方案。`BatchNorm/LayerNorm/activation` 与 `max/mean/sum` pooling 不可交换，提前 norm/act 会改变当前 pooling 数学语义。

### 9. fusion mask 只来自当前 point\_like

`VolumePointStage1Model._fuse_point_variable()` 做 type-aware fusion 时，`pseudo_mask` 必须只从当前 `point_like` 读取：

```python
pseudo_mask = point_like.get("pseudo_mask", None) if hasattr(point_like, "get") else None
```

`_run_point_backbone()` 构造 `point_feature_hook` 时必须把 `require_pseudo_mask_for_fusion = pseudo_layout is not None` 固定进 hook 闭包。若 `typed_point_cfg.use_separate_fusion` 为 true、`require_pseudo_mask_for_fusion` 为 true 且当前 `point_like` 缺少 `pseudo_mask`，直接 fail-fast。不得从原始 `batch["pseudo_mask"]` 兜底，因为 fusion hook 可能挂在 pooling 后的点变量上，此时 mask 分辨率已经变成 `N_pool_all`。

### 10. attention padding cache 本阶段保持现状

:comment[SerializedAttention.get\_padding\_and\_inverse() 现有缓存 key 只按 patch\_size 命名。本阶段 typed 改造不改变 offset / patch\_size 的缓存语义，也不扩大 scope 修改该缓存 key。]{#comment-1779199408064 text="客观上这是否会造成，在某些情况下出现缓存误用的可能？"}

需要补测试覆盖现有行为：同一 `Point`、同一 `patch_size`、不同 `order_index` 的 Block 仍能 forward，且 typed 改造不把 `pseudo_mask` 纳入 padding cache key。

## Proposed Changes

### 1. 新增 typed point 配置解析与通用 helper

#### \[NEW] [src/model/typed\_point.py](../../../../src/model/typed_point.py)

新增轻量配置与 helper。该文件不新增长期 PTV3 wrapper 类，只提供 dataclass 和通用 tensor/Point 辅助函数。

##### 1.1 `TypedPointConfig`

```python
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

@dataclass(frozen=True)
class TypedPointConfig:
    enabled: bool = True
    separate_qkv: bool = True
    separate_attn_proj: bool = True
    separate_ffn: bool = True
    separate_cpe: bool = True
    separate_embedding: bool = True
    separate_pooling_proj: bool = True
    separate_unpooling_proj: bool = True
    separate_fusion: bool = True
    separate_point_input_proj: bool = True
    separate_atom_token_proj: bool = True

    @property
    def use_separate_qkv(self) -> bool: ...
    @property
    def use_separate_attn_proj(self) -> bool: ...
    @property
    def use_separate_ffn(self) -> bool: ...
    @property
    def use_separate_cpe(self) -> bool: ...
    @property
    def use_separate_embedding(self) -> bool: ...
    @property
    def use_separate_pooling_proj(self) -> bool: ...
    @property
    def use_separate_unpooling_proj(self) -> bool: ...
    @property
    def use_separate_fusion(self) -> bool: ...
    @property
    def use_separate_point_input_proj(self) -> bool: ...
    @property
    def use_separate_atom_token_proj(self) -> bool: ...
```

| 字段                          | 类型     |    默认值 | 语义                                                                         |
| --------------------------- | ------ | -----: | -------------------------------------------------------------------------- |
| `enabled`                   | `bool` | `True` | 全局 typed point 开关；False 时所有 `use_separate_*` 为 False。                      |
| `separate_qkv`              | `bool` | `True` | enabled 后 QKV real/pseudo 分参。                                              |
| `separate_attn_proj`        | `bool` | `True` | enabled 后 attention 输出投影 real/pseudo 分参。                                   |
| `separate_ffn`              | `bool` | `True` | enabled 后 Block FFN real/pseudo 分参。                                        |
| `separate_cpe`              | `bool` | `True` | enabled 后 CPE 使用同类子图 + 分参数。                                                |
| `separate_embedding`        | `bool` | `True` | enabled 后 `PointConvEmbedding` 使用同类子图 + 分参数。                               |
| `separate_pooling_proj`     | `bool` | `True` | enabled 后 `SerializedPooling` projection path real/pseudo 分参。              |
| `separate_unpooling_proj`   | `bool` | `True` | enabled 后 `SerializedUnpooling` input/skip projection path real/pseudo 分参。 |
| `separate_fusion`           | `bool` | `True` | enabled 后 voxel-to-point fusion MLP real/pseudo 分参。                        |
| `separate_point_input_proj` | `bool` | `True` | enabled 后 point backbone input projection real/pseudo 分参。                  |
| `separate_atom_token_proj`  | `bool` | `True` | enabled 后 atom head token projection real/pseudo 分参。                       |

##### 1.2 `normalize_typed_point_cfg()`

```python
def normalize_typed_point_cfg(cfg: Mapping[str, Any] | TypedPointConfig | None) -> TypedPointConfig: ...
```

| 参数    | 类型                   | 意义               |        |                                          |
| ----- | -------------------- | ---------------- | ------ | ---------------------------------------- |
| `cfg` | \`Mapping\[str, Any] | TypedPointConfig | None\` | Hydra/OmegaConf 字典、已解析 dataclass 或 None。 |

返回：`TypedPointConfig`。

行为：

1. `cfg is None` 时返回默认 `TypedPointConfig(enabled=True, ...)`。
2. 对未知 key 直接 `ValueError`；旧 key `separate_token_proj` 也按未知 key 报错。
3. 所有提供字段必须已经是 `bool`；非 bool 直接 `ValueError`。不得使用 `bool(value)` 宽松转换。
4. 支持 OmegaConf `DictConfig` 的 mapping 行为；不得依赖 OmegaConf 专用 API。

##### 1.3 `validate_pseudo_mask()`

```python
def validate_pseudo_mask(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> torch.Tensor | None: ...
```

| 参数            | 类型             | 意义              |                                   |
| ------------- | -------------- | --------------- | --------------------------------- |
| `pseudo_mask` | \`torch.Tensor | None\`          | `(N_all,) bool`，True 表示 P anchor。 |
| `point_count` | `int`          | 应匹配的点数 `N_all`。 |                                   |
| `name`        | `str`          | 报错信息中的调用点名称。    |                                   |

返回：校验后的 bool mask 或 None。

行为：

1. `None` 表示 real-only / 当前点对象未携带伪点。
2. 非 None 时必须一维、长度等于 `point_count`。
3. dtype 不是 bool 时直接 `RuntimeError`。
4. `point_count == 0` 时只接受 None 或 shape `(0,)` 的 bool mask。

##### 1.4 `apply_type_aware_tensor_module()`

```python
def apply_type_aware_tensor_module(
    x: torch.Tensor,
    pseudo_mask: torch.Tensor | None,
    real_module: nn.Module,
    pseudo_module: nn.Module,
) -> torch.Tensor: ...
```

| 参数              | 类型             | 意义                                    |                                      |
| --------------- | -------------- | ------------------------------------- | ------------------------------------ |
| `x`             | `torch.Tensor` | `(N_all, C_in)`，mixed 或 real-only 特征。 |                                      |
| `pseudo_mask`   | \`torch.Tensor | None\`                                | `(N_all,) bool`；None 时全部按 real 分支处理。 |
| `real_module`   | `nn.Module`    | real 分支模块。                            |                                      |
| `pseudo_module` | `nn.Module`    | pseudo 分支模块。                          |                                      |

返回：`torch.Tensor`，`(N_all, C_out)`，保持输入点顺序。

行为：

1. `N_all == 0` 时直接调用 `real_module(x)`，用模块自身得到 `(0, C_out)`；不得推断 `C_out`，不得调用 pseudo 分支。
2. `pseudo_mask is None` 或全 False：只调用 `real_module(x)`。
3. `pseudo_mask` 全 True：只调用 `pseudo_module(x)`。
4. mixed：分别 slice 后调用各自模块，再 scatter 回原顺序。
5. 空 real 子集不调用 real module；空 pseudo 子集不调用 pseudo module。

##### 1.5 可选 helper：`split_mask_state()`

```python
def split_mask_state(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> tuple[torch.Tensor | None, bool, bool]: ...
```

返回：`(validated_mask, has_real, has_pseudo)`。用于 CPE / embedding / pooling / unpooling 中统一判断 all-real、all-pseudo、mixed。

##### 1.6 可选 helper：`make_typed_linear_norm_act()`

如实现 pooling / unpooling typed path 时需要避免重复代码，可新增：

```python
def make_typed_linear_norm_act(
    in_channels: int,
    out_channels: int,
    act_layer: type[nn.Module] | None,
) -> nn.Sequential: ...
```

返回：`Linear -> LayerNorm -> optional act`。该 helper 只用于 typed path，不替代 shared old path 的 `norm_layer` / `PointSequential` 构造。

### 2. 同步更新总控契约

#### \[MODIFY] [CLAUDE/plans/implement/tri\_ligand\_sparse\_refine/00-master.md](00-master.md)

需要同步更新以下契约：

1. `src/model/stage1_point_backbone.py` forward 输入增加 `pseudo_mask: torch.Tensor | None`，shape `(N_all,) bool`。
2. `point_state` 输出增加可选 `point_state["pseudo_mask"]`；当输入 / 中间 `Point` 携带 `pseudo_mask` 时必须透传。
3. 新增 `typed_point_cfg` 配置契约，字段与 [src/model/typed\_point.py](../../../../src/model/typed_point.py) 保持一致，使用 `separate_point_input_proj` / `separate_atom_token_proj`，不得再写 `separate_token_proj`。
4. PTV3 `Point` 契约补充：所有重建 `Point` 的路径必须维护已有 `pseudo_mask`。
5. PTV3 `SerializedPooling` 契约补充：
   * `cluster_key` 只用于 pooling 分组。
   * `serialized_code` 仍是纯空间 code。
   * pooling 不得合并同一 spatial cell 的 real 与 pseudo。
   * typed pooling 保持当前 `Linear -> pool -> norm/act` 数学顺序。
6. PTV3 `PointConvEmbedding` 与 `PointConvCPE` 契约补充：typed 模式下只做同类子图，不做 cross-type graph，不新增 self-loop；其中 `PointConvEmbedding` 额外保留 `self_proj(input_feat)`，`PointConvCPE` 不增加 self/residual 投影。

> \[!IMPORTANT]
> 00-master、本 02 子计划、源码文件头部“对齐契约”和测试必须同步更新；禁止只改其中一处。

### 3. 改造 PTV3 core

#### \[MODIFY] [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py)

##### 3.1 更新文件头部 `Point` 契约

将 `pseudo_mask` 更新为正式字段：

| 字段            | 类型             | shape           | 意义                                          |
| ------------- | -------------- | --------------- | ------------------------------------------- |
| `pseudo_mask` | `torch.Tensor` | `(N_all,) bool` | 可选字段，True 表示 P anchor；所有重建 `Point` 的路径必须维护。 |

同时补充 typed pooling 约束：`cluster_key` 含 type bit，但 `serialized_code` 不含 type bit；typed pooling 保持 `Linear -> pool -> typed LayerNorm/act` 顺序。

##### 3.2 `SerializedAttention.__init__()` 增加参数

修改签名：

```python
class SerializedAttention(PointModule):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        order_index: int = 0,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        separate_qkv: bool = False,
        separate_attn_proj: bool = False,
    ) -> None: ...
```

构造逻辑：

| 条件                         | 构造                                                                |
| -------------------------- | ----------------------------------------------------------------- |
| `separate_qkv=False`       | 保持 `self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)`。 |
| `separate_qkv=True`        | 构造 `self.qkv_real` / `self.qkv_pseudo`。                           |
| `separate_attn_proj=False` | 保持 `self.proj = nn.Linear(channels, channels)`。                   |
| `separate_attn_proj=True`  | 构造 `self.proj_real` / `self.proj_pseudo`。                         |

##### 3.3 `SerializedAttention.forward()` 保持一次 mixed attention

修改点：

1. forward 开始校验 `point.get("pseudo_mask", None)`。
2. QKV 构造时先对原始顺序 `point.feat` 做 type-aware projection，再用 `order` 重排。
3. flash path 与 non-flash path 均保持一次 mixed attention，不按 type 拆 attention。
4. attention 输出 `feat = feat[inverse]` 后，再做 type-aware output projection。
5. :comment[get\_padding\_and\_inverse() 的缓存 key 本阶段保持现状，不加入 pseudo\_mask。]{#comment-1779200355087 text="缓存不冗余是重要的，但是不误用更是重要，这需要仔细考察"}

> \[!IMPORTANT]
> `separate_qkv=True` 不能禁用 flash attention。QKV 分参发生在 attention 之前，scatter 后仍走原 flash/non-flash attention。

##### 3.4 `PointConvEmbedding` 支持同类子图分参

`embedding_impl` 不扩展新取值，仍只支持 `"pointconv"`。`separate_embedding=True` 只在 `embedding_impl="pointconv"` 下生效。

新增方法：

```python
class PointConvEmbedding(PointModule):
    def forward_subset(
        self,
        point: Point,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor: ...
```

| 参数          | 类型             | 意义                                     |
| ----------- | -------------- | -------------------------------------- |
| `point`     | `Point`        | 原 mixed `Point`，包含 `feat/coord/batch`。 |
| `keep_mask` | `torch.Tensor` | `(N_all,) bool`，当前类型子集。                |

返回：`embed_subset: torch.Tensor`，`(N_keep, C_out)`。

行为：

1. 校验 `keep_mask.shape == (N_all,)` 且 dtype 为 bool。
2. `N_keep == 0` 时返回 `(0, embed_channels)` 空张量，不调用 `_radius_graph_eager()`。
3. radius graph 只使用 `point.coord[keep_mask]`、`point.batch[keep_mask]` 与 `point.feat[keep_mask]`。
4. 使用 `_radius_graph_eager(..., loop=False)` 现有语义；无同类邻居时该点聚合为零。
5. 必须包含完整 embedding 语义：邻域聚合、typed path 固定 `LayerNorm`、act 都在 `forward_subset()` 内完成。
6. `forward_subset()` 是 pure helper，只返回子集 tensor，不修改原 `point.feat`。

`Embedding.__init__()` 增加参数：

```python
separate_embedding: bool = False
```

构造逻辑：

| 条件                         | 构造                                                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `separate_embedding=False` | 保持 `self._pointconv_embed = PointConvEmbedding(..., norm_layer=bn_layer, act_layer=act_layer)`。                             |
| `separate_embedding=True`  | 构造 `self._pointconv_embed_real` 与 `self._pointconv_embed_pseudo` 两套 `PointConvEmbedding`，typed 分支 norm 固定普通 `nn.LayerNorm`。 |

##### 3.5 `PointConvCPE` 增加 subset delta 计算

新增方法：

```python
class PointConvCPE(PointModule):
    def forward_subset(
        self,
        point: Point,
        keep_mask: torch.Tensor,
        *,
        type_name: str,
    ) -> torch.Tensor: ...
```

| 参数          | 类型             | 意义                                                          |
| ----------- | -------------- | ----------------------------------------------------------- |
| `point`     | `Point`        | 原 mixed `Point`，作为 graph cache owner，包含 `feat/coord/batch`。 |
| `keep_mask` | `torch.Tensor` | `(N_all,) bool`，当前类型子集。                                     |
| `type_name` | `str`          | 只能是 `"real"` 或 `"pseudo"`，用于 graph cache key 后缀。            |

返回：`delta_subset: torch.Tensor`，`(N_keep, C_out)`，只包含子集点的 CPE delta，不修改 `point.feat`。

实现要求：

1. `N_keep == 0` 时返回 `(0, channels)` 空张量，不建图。
2. `forward_subset()` 不得调用原 `_get_or_build_graph(point)`，因为原方法使用 mixed/shared cache key。
3. graph cache 存在原 mixed `point` 上，使用 `f"_pointconv_graph_{self.cache_key}_{type_name}"`；`self.cache_key is None` 时不缓存。
4. `Block` 构造 CPE 时继续传原始 `cpe_indice_key`（如 `"stage0"`），不得预先追加 `_real` / `_pseudo`，避免 `_real_real` 双重后缀。
5. radius graph 只使用 `point.coord[keep_mask]` 与 `point.batch[keep_mask]`，并保持 `loop=False`。
6. 复用当前 `w_v`、`mlp_w`、`w_o`、`norm`；CPE typed path 的 `norm` 仍由 Block 的 `norm_layer` 决定，不接入 BN。
7. 保留现有 `forward(self, point)` mixed CPE 行为，供 `separate_cpe=False` 使用。

##### 3.6 `Block.__init__()` 增加 typed 参数

新增参数：

```python
separate_qkv: bool = False
separate_attn_proj: bool = False
separate_ffn: bool = False
separate_cpe: bool = False
```

新增属性：

| 属性                                  | 条件                                          | 意义                                                 |
| ----------------------------------- | ------------------------------------------- | -------------------------------------------------- |
| `self.separate_ffn`                 | always                                      | 是否 FFN 分参。                                         |
| `self.separate_cpe`                 | always                                      | 是否 CPE 同类子图分参。                                     |
| `self.cpe`                          | `cpe_impl="pointconv" and not separate_cpe` | shared mixed CPE。                                  |
| `self.cpe_real` / `self.cpe_pseudo` | `cpe_impl="pointconv" and separate_cpe`     | typed subset CPE，cache\_key 均为原始 `cpe_indice_key`。 |
| `self.mlp`                          | `ffn_type != "none" and not separate_ffn`   | shared FFN。                                        |
| `self.mlp_real` / `self.mlp_pseudo` | `ffn_type != "none" and separate_ffn`       | typed FFN。                                         |

`SerializedAttention(...)` 调用透传 `separate_qkv` / `separate_attn_proj`。

##### 3.7 `Block.forward()` 拆出 typed CPE / FFN helper

新增内部方法：

```python
def _run_cpe(self, point: Point, pseudo_mask: torch.Tensor | None) -> torch.Tensor: ...
def _run_ffn(self, feat: torch.Tensor, pseudo_mask: torch.Tensor | None) -> torch.Tensor: ...
```

`_run_cpe()` 返回 `(N_all, C_out)` delta，供外层残差使用；shared 与 typed 路径都不得改变进入 CPE 残差的原始 shortcut 语义。`_run_ffn()` 返回 `(N_all, C_out)` FFN delta。

`forward()` 顺序保持不变：

```text
CPE + residual
norm1 -> mixed attention -> shared drop_path -> residual
norm2 -> FFN(shared or typed) -> shared drop_path -> residual
```

##### 3.8 `SerializedPooling.__init__()` 支持 projection path 分参

新增参数：

```python
separate_pooling_proj: bool = False
```

构造逻辑：

| 条件                            | 构造                                                                                                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `separate_pooling_proj=False` | 保持 `self.proj`、现有 `self.norm` / `self.act`。                                                                                                                                             |
| `separate_pooling_proj=True`  | 构造 `self.pool_proj_real` / `self.pool_proj_pseudo` 两个 `nn.Linear`；构造 pooled 后使用的 `self.pool_norm_real` / `self.pool_norm_pseudo` 为普通 `nn.LayerNorm(out_channels)`；act 可复用同类无参激活或两套激活模块。 |

##### 3.9 `SerializedPooling.forward()` 使用 type-aware cluster key，并保持 pooling 顺序

修改流程：

1. 计算 `spatial_code = point.serialized_code >> (pooling_depth * 3)`。
2. 校验 `pseudo_mask = validate_pseudo_mask(point.get("pseudo_mask", None), int(point.feat.shape[0]), name="SerializedPooling.forward")`。
3. 若 `pseudo_mask is None`，`cluster_key = spatial_code_primary`。
4. 若 `pseudo_mask is not None`：
   * 检查 `spatial_code_primary` 非负且 `< 1 << 62`。
   * `cluster_key = spatial_code_primary * 2 + pseudo_mask.to(spatial_code_primary.dtype)`。
5. 用 `cluster_key` 做 `torch.unique(..., return_inverse=True, return_counts=True)`。
6. `serialized_code` 使用 `spatial_code[:, head_indices]`，不使用 `cluster_key`。
7. 对 `serialized_order` / `serialized_inverse` 做稳定 tie-breaker：primary key 是纯空间 `pooled_spatial_code`，secondary key 是 `head_indices`。
8. `point_dict["pseudo_mask"] = pseudo_mask[head_indices]` 仅在输入 mask 非 None 时写入。
9. 分参顺序：

```python
if self.separate_pooling_proj:
    projected_feat = apply_type_aware_tensor_module(point.feat, pseudo_mask, self.pool_proj_real, self.pool_proj_pseudo)
else:
    projected_feat = self.proj(point.feat)

pooled_feat = torch_scatter.segment_csr(projected_feat[indices], idx_ptr, reduce=self.reduce)

if self.separate_pooling_proj:
    pooled_mask = pseudo_mask[head_indices] if pseudo_mask is not None else None
    pooled_feat = apply_type_aware_tensor_module(
        pooled_feat,
        pooled_mask,
        nn.Sequential(self.pool_norm_real, real_act_or_identity),
        nn.Sequential(self.pool_norm_pseudo, pseudo_act_or_identity),
    )
```

shared path 继续保持当前 `point = self.norm(point)` / `point = self.act(point)`。

##### 3.10 `SerializedUnpooling.__init__()` 支持 projection path 分参

新增参数：

```python
separate_unpooling_proj: bool = False
```

构造逻辑：

| 条件                              | 构造                                                                                                                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `separate_unpooling_proj=False` | 保持 `self.proj` / `self.proj_skip` 两个 shared `PointSequential`。                                                                                                                                            |
| `separate_unpooling_proj=True`  | 构造裸 `nn.Sequential` tensor modules：`self.proj_real` / `self.proj_pseudo` 处理 child tensor，`self.proj_skip_real` / `self.proj_skip_pseudo` 处理 parent skip tensor。每套为 `Linear -> LayerNorm -> optional act`。 |

> \[!IMPORTANT]
> shared 旧路径继续使用 `PointSequential`，因为它接收 `Point` 并原地更新 `point.feat`；typed 分支必须使用裸 tensor module，不能把 `PointSequential` 误传给 tensor helper。

##### 3.11 `SerializedUnpooling.forward()` typed 行为

现有 `pop` 行为保持不变，但顺序必须是：

1. 先 validate child pooled mask：`point.get("pseudo_mask", None)`，shape `(N_pool_all,)`。
2. 再 `parent = point.pop("pooling_parent")` 与 `inverse = point.pop("pooling_inverse")`。
3. 再 validate parent mask：`parent.get("pseudo_mask", None)`，shape `(N_all_parent,)`。
4. `separate_unpooling_proj=True` 时：
   * child projection 使用 child mask。
   * parent skip projection 使用 parent mask。
5. `parent.feat = parent.feat + point.feat[inverse]` 后返回 parent，parent mask 保持原始分辨率。

> \[!IMPORTANT]
> 不要把 child pooled mask 覆盖到 parent 上。unpooling 输出分辨率是 parent 分辨率，mask 必须来自 parent。

##### 3.12 `PointTransformerV3.__init__()` 透传 typed flags

新增参数：

```python
separate_qkv: bool = False
separate_attn_proj: bool = False
separate_ffn: bool = False
separate_cpe: bool = False
separate_embedding: bool = False
separate_pooling_proj: bool = False
separate_unpooling_proj: bool = False
```

消费位置：

1. `Embedding(...)` 传 `separate_embedding`。
2. 每个 `SerializedPooling(...)` 传 `separate_pooling_proj`。
3. 每个 `SerializedUnpooling(...)` 传 `separate_unpooling_proj`。
4. 每个 encoder / decoder `Block(...)` 传 `separate_qkv`、`separate_attn_proj`、`separate_ffn`、`separate_cpe`。

`PointTransformerV3.forward()` 本身不新增参数；如果 `data_dict` 中有 `pseudo_mask`，`Point(data_dict)` 会携带，并由 embedding / pooling / block / unpooling 维护。

### 4. 改造 Stage1 point backbone

#### \[MODIFY] [src/model/stage1\_point\_backbone.py](../../../../src/model/stage1_point_backbone.py)

##### 4.1 文件头契约更新

更新 forward 输入 / 输出契约：

| 字段/参数                       | 类型             | shape             | 意义                               |                                      |
| --------------------------- | -------------- | ----------------- | -------------------------------- | ------------------------------------ |
| `atom_feat`                 | `torch.Tensor` | `(N_all, F_atom)` | real-only 或 mixed 全点特征。          |                                      |
| `atom_coord_centered_world` | `torch.Tensor` | `(N_all, 3)`      | 点坐标。                             |                                      |
| `atom_batch_index`          | `torch.Tensor` | `(N_all,)`        | 点所属 BOX。                         |                                      |
| `atom_offsets`              | `torch.Tensor` | `(B,)`            | 每个 BOX mixed 或 real-only 展平结束偏移。 |                                      |
| `recycle_in`                | \`torch.Tensor | None\`            | `(N_all, C_recycle)`             | 当前轮 point recycle 输入。                |
| `pseudo_mask`               | \`torch.Tensor | None\`            | `(N_all,) bool`                  | True 表示 P anchor；None 表示未携带 type 标注。 |

输出补充：`point_state["pseudo_mask"]` 为可选字段；输入或中间 `Point` 携带 mask 时必须输出。

##### 4.2 构造函数新增 `typed_point_cfg`

修改签名：

```python
class Stage1PointBackbone(nn.Module):
    def __init__(
        ...,
        typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None,
    ) -> None: ...
```

内部：

```python
self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)
```

传给 `PointTransformerV3(...)`：

```python
separate_qkv=self.typed_point_cfg.use_separate_qkv,
separate_attn_proj=self.typed_point_cfg.use_separate_attn_proj,
separate_ffn=self.typed_point_cfg.use_separate_ffn,
separate_cpe=self.typed_point_cfg.use_separate_cpe,
separate_embedding=self.typed_point_cfg.use_separate_embedding,
separate_pooling_proj=self.typed_point_cfg.use_separate_pooling_proj,
separate_unpooling_proj=self.typed_point_cfg.use_separate_unpooling_proj,
```

##### 4.3 point input projection 支持 `separate_point_input_proj`

新增 helper：

```python
def _build_atom_input_proj(self, act_cls: type[nn.Module], input_embed_hidden_dim: int) -> nn.Module: ...
```

返回当前结构：

```python
nn.Sequential(
    nn.Linear(self.atom_feature_dim, int(input_embed_hidden_dim)),
    nn.LayerNorm(int(input_embed_hidden_dim)),
    act_cls(),
    nn.Linear(int(input_embed_hidden_dim), self.input_embed_dim),
)
```

构造逻辑：

```python
if self.typed_point_cfg.use_separate_point_input_proj:
    self.atom_input_proj_real = self._build_atom_input_proj(_act_cls, int(input_embed_hidden_dim))
    self.atom_input_proj_pseudo = self._build_atom_input_proj(_act_cls, int(input_embed_hidden_dim))
    self.atom_input_proj = None
else:
    self.atom_input_proj = self._build_atom_input_proj(_act_cls, int(input_embed_hidden_dim))
```

`_build_point_input_feat()` 增加参数 `pseudo_mask: torch.Tensor | None = None`，在非空 batch 下按 mask 调 `apply_type_aware_tensor_module()`。

现有 `atom_count == 0` fast path 保留：直接返回 `(0, input_embed_dim)` 空张量，不强行调用 real projection。

##### 4.4 `_forward_ptv3()` 透传 `pseudo_mask`

修改签名增加 `pseudo_mask: torch.Tensor | None = None`。构造初始 `Point` 时：

```python
point_dict = {
    "feat": point_input_feat,
    "coord": atom_coord_centered_world,
    "batch": atom_batch_index,
    "offset": atom_offsets,
    "grid_size": self.point_grid_size,
}
if pseudo_mask is not None:
    point_dict["pseudo_mask"] = pseudo_mask
point = Point(point_dict)
```

##### 4.5 `build_zeros_output()` 透传 `pseudo_mask`

修改签名增加 `pseudo_mask: torch.Tensor | None = None`。输出 `point_state` 时，如果输入 mask 非 None，写入 `point_state["pseudo_mask"] = pseudo_mask`。空 batch 分支同样处理 shape `(0,)` 的 mask。

##### 4.6 `forward()` 新增 `pseudo_mask`

修改签名增加 `pseudo_mask: torch.Tensor | None = None`。

forward 行为：

1. `pseudo_mask = validate_pseudo_mask(pseudo_mask, atom_count, name="Stage1PointBackbone.forward")`。
2. `_build_point_input_feat(..., pseudo_mask=pseudo_mask)`。
3. 空 batch、zeros backend、ptv3 backend 均透传 `pseudo_mask`。

##### 4.7 `_export_point_state()` 维护 `pseudo_mask`

当前只导出 `coord/batch/offset/grid_size/grid_coord`。新增：

```python
if "pseudo_mask" in point_like:
    point_state["pseudo_mask"] = point_like["pseudo_mask"]
```

### 5. 改造 Stage1 atom head

#### \[MODIFY] [src/model/stage1\_atom\_head.py](../../../../src/model/stage1_atom_head.py)

##### 5.1 文件头契约更新

将所有 mixed shape 写为 `N_all`，real 输出写 `N_real`，pseudo 输出写 `N_pseudo`。补充：atom head `Block` 默认跟随全局 `separate_cpe`；当 `atom_head_cpe_impl="pointconv"` 时，atom head CPE 也使用同类子图，不为 atom head 开 cross-type CPE 例外。

##### 5.2 `Stage1SerializedAttentionStack.__init__()` 新增 `typed_point_cfg`

修改签名增加：

```python
typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None
```

内部解析 `self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)`。

每个 `Block(...)` 调用增加：

```python
separate_qkv=self.typed_point_cfg.use_separate_qkv,
separate_attn_proj=self.typed_point_cfg.use_separate_attn_proj,
separate_ffn=self.typed_point_cfg.use_separate_ffn,
separate_cpe=self.typed_point_cfg.use_separate_cpe,
```

`forward()` 开始用 `validate_pseudo_mask()` 校验 `pseudo_mask`。

##### 5.3 `Stage1AtomHead.__init__()` 新增 `typed_point_cfg`

修改签名增加：

```python
typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None
```

内部解析 `self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)`，并传给 `Stage1SerializedAttentionStack`。

##### 5.4 atom token projection 支持 `separate_atom_token_proj`

新增 helper：

```python
def _build_atom_token_proj(
    self,
    token_input_dim: int,
    act_layer: type[nn.Module],
) -> nn.Module: ...
```

返回当前结构：

```python
nn.Sequential(
    nn.Linear(token_input_dim, self.hidden_dim),
    nn.LayerNorm(self.hidden_dim),
    act_layer(),
)
```

构造逻辑：

```python
if self.typed_point_cfg.use_separate_atom_token_proj:
    self.atom_token_proj_real = self._build_atom_token_proj(token_input_dim, act_layer)
    self.atom_token_proj_pseudo = self._build_atom_token_proj(token_input_dim, act_layer)
    self.atom_token_proj = None
else:
    self.atom_token_proj = self._build_atom_token_proj(token_input_dim, act_layer)
```

`forward()` 中：

```python
pseudo_mask = validate_pseudo_mask(pseudo_mask, int(point_feat.shape[0]), name="Stage1AtomHead.forward")
if self.typed_point_cfg.use_separate_atom_token_proj:
    atom_hidden = apply_type_aware_tensor_module(
        atom_tokens,
        pseudo_mask,
        self.atom_token_proj_real,
        self.atom_token_proj_pseudo,
    )
else:
    atom_hidden = self.atom_token_proj(atom_tokens)
```

输出契约保持：

| 输出               | shape                               | 说明                           |              |
| ---------------- | ----------------------------------- | ---------------------------- | ------------ |
| `atom_tokens`    | `(N_all, C_token)`                  | projection 前输入，mixed 全点。     |              |
| `atom_hidden`    | `(N_all, C_hidden)`                 | attention stack 输出，mixed 全点。 |              |
| `atom_logits`    | `(N_real, atom_logit_dim)`          | real-only。                   |              |
| `pseudo_feature` | \`(N\_pseudo, pseudo\_feature\_dim) | None\`                       | pseudo-only。 |

### 6. 改造 Stage1 主模型

#### \[MODIFY] [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)

##### 6.1 构造函数新增 `typed_point_cfg`

修改 `VolumePointStage1Model.__init__()` 签名增加：

```python
typed_point_cfg: dict[str, Any] | TypedPointConfig | None = None
```

内部：

```python
self.typed_point_cfg = normalize_typed_point_cfg(typed_point_cfg)
```

传给 atom head：

```python
self.atom_head = Stage1AtomHead(
    ...,
    typed_point_cfg=self.typed_point_cfg,
)
```

##### 6.2 fusion MLP 支持分参

新增 helper：

```python
def _build_point_fusion_module(
    self,
    fusion_input_dim: int,
    fusion_hidden_dim: int,
    point_channels: int,
    act_cls: type[nn.Module],
    fusion_proj_drop: float,
) -> nn.Module: ...
```

返回当前结构：

```python
nn.Sequential(
    nn.Linear(fusion_input_dim, fusion_hidden_dim),
    nn.LayerNorm(fusion_hidden_dim),
    act_cls(),
    nn.Dropout(fusion_proj_drop),
    nn.Linear(fusion_hidden_dim, point_channels),
)
```

构造逻辑：

```python
if self.typed_point_cfg.use_separate_fusion:
    self.point_fusion_modules[point_name] = nn.ModuleDict({
        "real": self._build_point_fusion_module(...),
        "pseudo": self._build_point_fusion_module(...),
    })
else:
    self.point_fusion_modules[point_name] = self._build_point_fusion_module(...)
```

##### 6.3 `_fuse_point_variable()` 按当前 `point_like` mask 分参

修改签名，增加由 `_run_point_backbone()` hook 闭包传入的显式要求：

```python
def _fuse_point_variable(
    self,
    feature_name: str,
    point_like: Any,
    voxel_output_dict: dict[str, Any],
    batch: dict[str, Any],
    *,
    require_pseudo_mask: bool = False,
) -> Any: ...
```

保留现有 voxel sampling 逻辑。拼接后：

```python
fusion_input = torch.cat([point_like.feat, sampled_voxel_feat], dim=-1)
```

新增：

```python
pseudo_mask = point_like.get("pseudo_mask", None) if hasattr(point_like, "get") else None
pseudo_mask = validate_pseudo_mask(
    pseudo_mask,
    int(fusion_input.shape[0]),
    name="VolumePointStage1Model._fuse_point_variable",
)
fusion_module = self.point_fusion_modules[feature_name]
if self.typed_point_cfg.use_separate_fusion:
    if require_pseudo_mask and pseudo_mask is None:
        raise RuntimeError("typed fusion 的 mixed point_like 必须携带 pseudo_mask。")
    point_like.feat = apply_type_aware_tensor_module(
        fusion_input,
        pseudo_mask,
        fusion_module["real"],
        fusion_module["pseudo"],
    )
else:
    point_like.feat = fusion_module(fusion_input)
```

实现时不需要额外传 batch mask；`require_pseudo_mask=False` 且当前 point\_like 缺 mask 时按 real-only 路径处理；`require_pseudo_mask=True` 且缺 mask 时 fail-fast。

##### 6.4 `_run_point_backbone()` 向 point backbone 透传 `pseudo_mask`

当前 `_run_point_backbone()` 已有 `pseudo_layout`。修改：

```python
if pseudo_layout is None:
    pseudo_mask = None
else:
    if "pseudo_mask" not in batch:
        raise RuntimeError("mixed point batch 必须包含 pseudo_mask。")
    pseudo_mask = batch["pseudo_mask"]
```

zeros backend 与 ptv3 backend 均传 `pseudo_mask=pseudo_mask`。构造 `point_feature_hook` 时同时固定 `require_pseudo_mask_for_fusion = pseudo_layout is not None`，并把它传给 `_fuse_point_variable(..., require_pseudo_mask=require_pseudo_mask_for_fusion)`。

> \[!IMPORTANT]
> `pseudo_layout is not None` 时，`batch` 必须来自 `inject_pseudo_atoms()` 并包含 `pseudo_mask`。缺失时 fail-fast，不得静默按 real-only 处理。

### 7. 配置更新

#### \[MODIFY] [configs/model/default.yaml](../../../../configs/model/default.yaml)

在 `model.backbone` 下新增：

```yaml
typed_point_cfg:
  enabled: true                       # bool, 全局 typed point 开关; false 时所有 use_separate_* 为 false
  separate_qkv: true                  # bool, QKV real/pseudo 分参
  separate_attn_proj: true            # bool, attention 输出投影 real/pseudo 分参
  separate_ffn: true                  # bool, Block FFN real/pseudo 分参
  separate_cpe: true                  # bool, CPE 同类子图 real/pseudo 分参
  separate_embedding: true            # bool, PointConvEmbedding 同类子图 real/pseudo 分参
  separate_pooling_proj: true         # bool, SerializedPooling projection path real/pseudo 分参
  separate_unpooling_proj: true       # bool, SerializedUnpooling projection path real/pseudo 分参
  separate_fusion: true               # bool, voxel-to-point fusion MLP real/pseudo 分参
  separate_point_input_proj: true     # bool, point backbone input projection real/pseudo 分参
  separate_atom_token_proj: true      # bool, atom head token projection real/pseudo 分参
```

#### \[MODIFY] [configs/model/point\_backbone/default.yaml](../../../../configs/model/point_backbone/default.yaml)

#### \[MODIFY] [configs/model/point\_backbone/stardard.yaml](../../../../configs/model/point_backbone/stardard.yaml)

#### \[MODIFY] [configs/model/point\_backbone/zeros.yaml](../../../../configs/model/point_backbone/zeros.yaml)

在 `model.backbone.point_backbone` 下新增：

```yaml
typed_point_cfg: ${model.backbone.typed_point_cfg}
```

说明：这是同一份全局配置的 Hydra 插值，不是第二套独立配置。

#### \[NO CHANGE] [configs/model/atom\_head/\*.yaml](../../../../configs/model/atom_head/)

`Stage1AtomHead` 由 `VolumePointStage1Model` 内部构造，直接消费 `model.backbone.typed_point_cfg`，不在 atom head config group 中重复定义。

> \[!NOTE]
> 当前 [configs/model/atom\_head/default.yaml](../../../../configs/model/atom_head/default.yaml) 中 `atom_head_cpe_impl: "pointconv"` 时，atom head CPE 会默认跟随 `separate_cpe=true` 并使用同类子图。

#### \[NO CHANGE] [configs/model/fusion/\*.yaml](../../../../configs/model/fusion/)

fusion 拓扑仍由 `point_fusion_map` / `point_fusion_modes` / `sampler_modes` 控制；是否分参由全局 `typed_point_cfg.separate_fusion` 控制，不在 fusion config group 中重复定义。

### 8. 测试更新

#### \[NEW] [tests/model/test\_typed\_point\_core.py](../../../../tests/model/test_typed_point_core.py)

新增 typed point core 单测。

| 测试函数                                                                   | 构造                                                                                    | 关键断言                                                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `test_typed_point_config_defaults_enabled`                             | `normalize_typed_point_cfg(None)`                                                     | `enabled is True`，所有 `use_separate_*` 默认为 True。                            |
| `test_typed_point_config_unknown_key_fails_fast`                       | `normalize_typed_point_cfg({"enabled": True, "bad": True})`                           | 抛 `ValueError`。                                                            |
| `test_typed_point_config_rejects_legacy_separate_token_proj`           | `{"separate_token_proj": True}`                                                       | 抛 `ValueError`，不兼容旧字段。                                                     |
| `test_typed_point_config_rejects_non_bool`                             | `{"enabled": "false"}`                                                                | 抛 `ValueError`，不得宽松 bool 转换。                                               |
| `test_typed_point_config_enabled_false_gates_all_flags`                | `enabled=False` 且所有 separate\=true                                                    | 所有 `use_separate_*` 为 False。                                               |
| `test_apply_type_aware_tensor_module_scatters_to_original_order`       | real module 输出 `+1`，pseudo module 输出 `-1`，mask 混排                                     | 输出顺序与输入一致。                                                                 |
| `test_apply_type_aware_tensor_module_skips_empty_branch`               | all-real / all-pseudo mask，另一分支 module 设计为一调用就报错                                      | 空分支不被调用。                                                                   |
| `test_serialized_attention_separate_qkv_keeps_single_mixed_attention`  | `SerializedAttention(separate_qkv=True, separate_attn_proj=True, enable_flash=False)` | forward shape `(N_all, C)`；attention 仍一次 mixed。                            |
| `test_serialized_attention_padding_cache_unchanged_by_pseudo_mask`     | 同一 `Point`、同一 `patch_size`、带/不带 `pseudo_mask`                                         | padding cache key 不加入 mask，typed 改造不破坏现有 forward。                          |
| `test_pointconv_embedding_separate_uses_same_type_graphs`              | 同时构造 same-type 与 cross-type 邻居，并将 cross-type 特征设为极大值                                  | typed embedding 输出只受 same-type 邻居影响。                                       |
| `test_pointconv_embedding_typed_norm_is_layernorm`                     | `Embedding(separate_embedding=True)`                                                  | real/pseudo 分支 norm 为 `nn.LayerNorm`，shared path 不改。                       |
| `test_block_separate_ffn_has_distinct_real_pseudo_modules`             | `Block(separate_ffn=True, ffn_type="mlp")`                                            | `mlp_real is not mlp_pseudo`，drop\_path 仍共享。                               |
| `test_block_separate_cpe_uses_no_cross_type_edges`                     | same-type 与 cross-type 邻居                                                             | typed CPE 输出只受 same-type 邻居影响，cache 中无跨类型边。                                |
| `test_typed_cpe_cache_uses_single_type_suffix`                         | 连续两个 `Block(cpe_indice_key="stage0", separate_cpe=True)`                              | 缓存 key 为 `_pointconv_graph_stage0_real` / `_pseudo`，不存在 `_real_real`。      |
| `test_serialized_pooling_splits_same_cell_by_type`                     | 同一 spatial cell 内一个 real 一个 pseudo                                                    | pooling 后点数为 2，不合并。                                                        |
| `test_serialized_pooling_serialized_code_stays_spatial`                | 构造同 cell real/pseudo                                                                  | `serialized_code` 等于纯空间 `spatial_code[:, head_indices]`，不等于 `cluster_key`。 |
| `test_serialized_pooling_order_tie_breaks_by_head_indices`             | real/pseudo pooled token 同 spatial code                                               | `serialized_order` 对 tie 稳定，secondary key 为 `head_indices`。                |
| `test_serialized_pooling_rejects_negative_or_too_large_spatial_code`   | 构造非法 `spatial_code_primary`                                                           | typed pooling fail-fast。                                                   |
| `test_serialized_pooling_keeps_linear_pool_norm_order`                 | monkeypatch Linear/norm/act 记录调用顺序                                                    | typed path 顺序为 Linear -> segment\_csr -> typed norm/act。                   |
| `test_serialized_pooling_typed_norm_is_layernorm`                      | `SerializedPooling(separate_pooling_proj=True)`                                       | typed pooled norm 为 `nn.LayerNorm`。                                        |
| `test_serialized_unpooling_preserves_parent_pseudo_mask`               | pooling + unpooling                                                                   | 返回 parent 分辨率 mask 与原始 mask 一致。                                            |
| `test_serialized_unpooling_uses_child_and_parent_masks_separately`     | child pooled mask 与 parent 原始 mask shape 不同                                           | child projection 用 child mask，skip projection 用 parent mask。               |
| `test_serialized_unpooling_typed_norm_is_layernorm`                    | `SerializedUnpooling(separate_unpooling_proj=True)`                                   | typed projection path norm 为 `nn.LayerNorm`。                               |
| `test_point_transformer_preserves_pseudo_mask_through_encoder_decoder` | tiny `PointTransformerV3` mixed 输入                                                    | 输出 `Point` 仍含 `(N_all,) bool` mask。                                        |
| `test_point_transformer_typed_flags_reach_all_core_modules`            | tiny `PointTransformerV3(separate_*=True)`                                            | Embedding、pooling、unpooling、Block attention/FFN/CPE flags 均为 True。         |

依赖 `torch_cluster.radius_graph` 的图测试必须使用：

```python
pytest.importorskip("torch_cluster")
```

并默认 `enable_flash=False`，避免本地缺 flash attention 时失败。纯 helper / config / 构造类测试不得跳过。

#### \[MODIFY] [tests/model/test\_stage1\_atom\_head.py](../../../../tests/model/test_stage1_atom_head.py)

更新 `_make_head(..., typed_point_cfg=None)`，传给 `Stage1AtomHead`。

新增测试：

| 测试函数                                                                          | 构造                                                         | 关键断言                                                                           |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `test_stage1_atom_head_default_typed_flags_enabled`                           | 不传 `typed_point_cfg`，`num_layers=1`，`cpe_impl="pointconv"` | 默认 typed enabled；Block 中 `separate_qkv/separate_ffn/separate_cpe` 为 True。      |
| `test_stage1_atom_head_separate_atom_token_proj_uses_real_and_pseudo_modules` | mixed mask                                                 | 存在 `atom_token_proj_real` / `atom_token_proj_pseudo`；输出 shape 不变。              |
| `test_stage1_atom_head_all_real_skips_pseudo_token_proj`                      | 全 False mask，pseudo token module 调用即报错                     | pseudo 分支不被调用。                                                                 |
| `test_stage1_atom_head_all_pseudo_skips_real_token_proj`                      | 全 True mask，real token module 调用即报错                        | real 分支不被调用；`atom_logits.shape[0] == 0`，`pseudo_feature.shape[0] == N_pseudo`。 |
| `test_stage1_atom_head_pointconv_cpe_follows_global_separate_cpe`             | `cpe_impl="pointconv"`，默认 typed cfg                        | atom head Block 使用 typed CPE。                                                  |

保留现有 real-only / mixed 双尾部 shape、pseudo\_feature\_dim、append\_coord\_mask、prior bias 测试。

#### \[MODIFY] [tests/model/test\_stage1\_model.py](../../../../tests/model/test_stage1_model.py)

更新 `_PointBackboneStub.build_zeros_output()` / forward stub 签名以接收 `pseudo_mask`。

新增测试：

| 测试函数                                                                      | 构造                                                                                    | 关键断言                                                               |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `test_stage1_model_passes_pseudo_mask_to_point_backbone`                  | monkeypatch final `_prepare_pseudo_batch()` 注入 P；point backbone stub 记录 `pseudo_mask` | mixed final 轮调用 point backbone 时收到 `(N_all,) bool` mask。           |
| `test_stage1_model_separate_fusion_builds_real_pseudo_modules_by_default` | fusion map 非空，默认 typed config                                                         | `point_fusion_modules[point_name]` 是 `ModuleDict`，含 `real/pseudo`。 |
| `test_stage1_model_shared_fusion_when_typed_disabled`                     | `typed_point_cfg={"enabled": False}`                                                  | `point_fusion_modules[point_name]` 是单个 `nn.Sequential`。            |
| `test_stage1_model_fusion_all_real_skips_pseudo_module`                   | all-real point\_like，pseudo fusion module 调用即报错                                       | pseudo fusion 分支不被调用。                                              |
| `test_stage1_model_fusion_uses_point_like_mask_not_batch_mask`            | point\_like mask 与 batch mask shape 不同                                                | fusion 使用 point\_like mask；不访问 batch mask 兜底。                      |
| `test_stage1_model_missing_pseudo_mask_in_mixed_batch_fails_fast`         | 伪造 `pseudo_layout` 但 batch 无 `pseudo_mask`                                            | `_run_point_backbone()` 抛错。                                        |

#### \[MODIFY] [tests/model/test\_ptv3\_no\_sparseconv.py](../../../../tests/model/test_ptv3_no_sparseconv.py)

保持 sparseconv 删除测试，补充：

| 测试函数                                                         | 构造                                                                | 关键断言             |
| ------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------- |
| `test_block_rejects_sparseconv_cpe_with_typed_flags`         | `Block(cpe_impl="sparseconv", separate_cpe=True)`                 | 仍抛 `ValueError`。 |
| `test_embedding_rejects_non_pointconv_impl_with_typed_flags` | `Embedding(embedding_impl="sparseconv", separate_embedding=True)` | 仍抛 `ValueError`。 |

#### \[MODIFY] [tests/model/test\_pseudo\_atoms.py](../../../../tests/model/test_pseudo_atoms.py)

如 `filter_point_state_with_mask()` 仍负责 point\_state 裁剪，确认已有或新增：

| 测试函数                                          | 构造                                          | 关键断言                                          |
| --------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| `test_filter_point_state_filters_pseudo_mask` | point\_state 含 `pseudo_mask`，keep mask 选择子集 | 输出 `point_state["pseudo_mask"]` 与 keep 后顺序一致。 |

## 不修改的部分

1. 不实现 C 生成、P weighted-FPS、density cube、P→C three-nn 插值、C sparse logit head。
2. 不实现 refine loss / metrics / wrapper 新分支。
3. 不新增 `separate_layernorm`，不分离 `Block.norm1` / `Block.norm2` / atom stack `output_norm`。
4. 不分离 dropout / drop path。
5. 不复制两份 attention 计算；real 与 pseudo 仍在同一次 mixed attention 中互相可见。
6. 不实现 CPE / embedding 的 mixed graph typed-param 版本；typed 图只做同类子图。
7. 不扩展 `embedding_impl` 取值；它仍只支持 `pointconv`，`separate_embedding` 只在该实现下生效。
8. 不恢复 sparseconv embedding / CPE。
9. 不新增 `atom_point_kind` / `point_kind` 三态字段；类型仍由 bool `pseudo_mask` 表达。
10. 不把 typed path 的 LayerNorm 变更扩散到 shared old path。
11. 不修改 `SerializedAttention.get_padding_and_inverse()` 的缓存 key 设计，只补测试守住现有行为。
12. 不私自删除 [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py) 中既有用户注释；只在语义过期时最小化更新。

## 改动文件汇总

| 文件                                                                                              | 改动内容                                                                                            |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [src/model/typed\_point.py](../../../../src/model/typed_point.py)                               | 新增 `TypedPointConfig`、严格配置解析、mask 校验和 type-aware helper。                                        |
| [CLAUDE/plans/implement/tri\_ligand\_sparse\_refine/00-master.md](00-master.md)                 | 同步 typed config、shape 命名、`pseudo_mask`、pooling key、typed embedding/CPE 契约。                      |
| [src/model/PTV3bakcbone/model.py](../../../../src/model/PTV3bakcbone/model.py)                  | 改造 attention、embedding、CPE、Block、pooling、unpooling、PointTransformerV3 typed flags；保留既有注释。       |
| [src/model/stage1\_point\_backbone.py](../../../../src/model/stage1_point_backbone.py)          | 接收 `typed_point_cfg` / `pseudo_mask`，分离 point input projection，导出 `point_state["pseudo_mask"]`。 |
| [src/model/stage1\_atom\_head.py](../../../../src/model/stage1_atom_head.py)                    | 接收 `typed_point_cfg`，分离 atom token projection，向 Block 透传 typed flags。                           |
| [src/model/stage1\_model.py](../../../../src/model/stage1_model.py)                             | 接收全局 `typed_point_cfg`，fusion 分参，向 point backbone 透传 `pseudo_mask`。                             |
| [configs/model/default.yaml](../../../../configs/model/default.yaml)                            | 新增默认 enabled 的全局 `model.backbone.typed_point_cfg`。                                              |
| [configs/model/point\_backbone/\*.yaml](../../../../configs/model/point_backbone/)              | 增加 `typed_point_cfg: ${model.backbone.typed_point_cfg}`。                                        |
| [tests/model/test\_typed\_point\_core.py](../../../../tests/model/test_typed_point_core.py)     | 新增 typed core、embedding、pooling、unpooling 测试。                                                   |
| [tests/model/test\_stage1\_atom\_head.py](../../../../tests/model/test_stage1_atom_head.py)     | 增加 typed atom token projection、atom head typed CPE 和 all-real/all-pseudo 测试。                    |
| [tests/model/test\_stage1\_model.py](../../../../tests/model/test_stage1_model.py)              | 增加 mask 透传、fusion 分参、fusion mask 来源、mixed 缺 mask fail-fast 测试。                                  |
| [tests/model/test\_ptv3\_no\_sparseconv.py](../../../../tests/model/test_ptv3_no_sparseconv.py) | 覆盖 typed flags 下 sparseconv 仍不可用。                                                               |
| [tests/model/test\_pseudo\_atoms.py](../../../../tests/model/test_pseudo_atoms.py)              | 补充或确认 point\_state `pseudo_mask` 裁剪测试。                                                          |

## Verification Plan

### Automated Tests

```bash
pytest tests/model/test_typed_point_core.py -q
```

验证点：默认 enabled、unknown key fail-fast、旧 `separate_token_proj` fail-fast、非 bool fail-fast、enabled gate、empty branch 不调用、QKV/proj/FFN/CPE/embedding/pooling/unpooling 分参、pooling key 不污染 `serialized_code`、stable tie-breaker、typed path LayerNorm、PTV3 mixed forward 不丢 `pseudo_mask`。

```bash
pytest tests/model/test_stage1_atom_head.py -q
```

验证点：atom head 原有 real/pseudo 双尾部不变，默认 typed flags 生效，atom token projection 分参，atom head pointconv CPE 跟随 `separate_cpe`，all-real / all-pseudo 不调用空分支。

```bash
pytest tests/model/test_stage1_model.py -q
```

验证点：Stage1 final mixed 轮向 point backbone 透传 `pseudo_mask`，fusion 默认分参，显式 disabled 后共享，fusion mask 只来自当前 `point_like`，mixed 缺 mask fail-fast。

```bash
pytest tests/model/test_ptv3_no_sparseconv.py tests/model/test_pseudo_atoms.py -q
```

验证点：PTV3 sparseconv 删除不回归；typed flags 下仍拒绝 sparseconv；pseudo layout / point\_state mask 裁剪不回归。

### Config / Hydra Smoke Tests

```bash
python -m src.train +experiment=unet000 trainer.fast_dev_run=true
```

验证点：默认 `typed_point_cfg.enabled=true` 时，Unet-only / pseudo\_atom none 组合仍可实例化和前向。若该实验禁用 point/atom head，则至少验证配置解析不破坏禁用路径。

```bash
python -m src.train +experiment=unet000 model.backbone.typed_point_cfg.enabled=false trainer.fast_dev_run=true
```

验证点：显式关闭 typed 后仍可走旧 shared 参数路径，用于 baseline / ablation。

如果当前训练入口不支持 `trainer.fast_dev_run=true`，改用项目现有 fast-dev-run 参数；验证目标不变。

### Manual Code Checks

```bash
rg "separate_token_proj|separate_layernorm|mixed_graph_typed|point_kind|atom_point_kind" src/model configs/model CLAUDE/plans/implement/tri_ligand_sparse_refine
```

预期：无旧 `separate_token_proj`、无新增 `separate_layernorm`、无 mixed graph typed CPE/embedding 配置、无三态 point kind 字段。

```bash
rg "cluster_key|spatial_code|serialized_code|pseudo_mask" src/model/PTV3bakcbone/model.py
```

预期：`cluster_key` 仅用于 pooling unique/cluster；`serialized_code` 仍来自纯空间 `spatial_code[:, head_indices]`；所有 Point 重建路径都维护已有 `pseudo_mask`。

```bash
rg "BatchNorm1d|PDNorm|LayerNorm" src/model/PTV3bakcbone/model.py src/model/typed_point.py
```

预期：typed embedding / pooling / unpooling path 使用普通 `nn.LayerNorm`；shared old path 仍可使用 BN/PDNorm。