# 特征/梯度通路优化 —— 实施计划（implement-plan-writer 版）

> [!IMPORTANT]
> **决策增补（grill 第二轮收口，若与下文任何处冲突，以本块为准）：**
> 1. **真实原子 density 支持 standalone（选 B）**：保留独立 encoder 路径——`real_atom_density_share_encoder=False` 时用 `real_density_cube_cfg` 建独立属性 `real_density_cube_encoder`，并把它注册进 [`set_input_channels`:834](src/model/stage1_model.py#L834) 与 [train.py lazy-init 门:96-101](src/train.py#L96)，保证 DDP 前 materialize。`share_encoder=True` 仍要求 `anchor_sampler`/`density_cube_encoder` 在位（fail-fast 信息写明"想独立用请设 share_encoder=false"）。[313-316 不变量](src/model/stage1_model.py#L313)**不动**（独立 encoder 是另一属性，不冲突；故第 4 条 agent 建议的"放宽不变量/提升为通用 token encoder"不采纳）。
> 2. **density 注入位置**：抽独立方法 `_apply_real_atom_density_to_atom_feat(batch)`，在 forward 里 `_run_embed_head_once` 返回**之后无条件调用**（`embed_head=None` 也生效），不塞进 `_run_embed_head_once` 末尾、也不复制进 early-return 分支。
> 3. **Stage1AtomHead fail-fast**：`__init__` 加 `concat_receptor_base_logit and receptor_base_dim is None → raise`。
> 4. **通用构建收进 [src/model/utils.py](src/model/utils.py)**：`build_zero_init_residual_mlp` / `FiLMCombine` / `CubeWeightingParams`；点 3 combine 与点 5 cube 权重均委托此处。

## 背景 / 目标

围绕 Stage1 模型的 voxel↔point 信息流与 logit 残差做五处解耦/增强。设计经 grill 收口，本文是按 `implement-plan-writer` 规范撰写的形式化规格。

> [!NOTE]
> 本文是本特征的**权威完整规格**；同目录 [plan-native.md](plan-native.md) 已降级为指针(只记 native 视角差异)，修订只改本文即可，无需再双写。行号为撰写时锚点，实施以实际代码为准。

五点目标：

1. 把 voxel→point 融合的唯一 detach 开关 `detach_voxel_feat_into_point` 拆成真实/伪原子两个。
2. 解耦 `SparseRefineHead` 的"base logit 进 MLP"与"末尾加残差"。
3. 新增"把 base 拼进 atom 头输入"开关，与现有"末尾加残差"配对，统一作用于前置头与后置头。
4. 为真实原子新增可选 density cube 特征，与 embed 输出零初始化融合。
5. voxel→point 融合采样新增 `weighted_cube`(可学习 3³ softmax 加权)/`cube_mean`(无参数 3³ 均值) 两种 sampler，并把现有 per-BOX 循环采样向量化。

## 设计决策

| # | 决策 | 取值 |
| --- | --- | --- |
| 1 | fusion detach 拆分 | `detach_voxel_feat_into_real_point=False`、`detach_voxel_feat_into_pseudo_point=True`；删旧统一键 |
| 2a | sparse refine 两轴 | 保留 `mode∈{direct,residual}` 与 `inputs.use_voxel_logits`，删二者耦合约束；4 组合合法 |
| 2b | atom 头两轴 | 新 `atom_head_concat_receptor_base_logit`(concat 进输入) 与 `refine_receptor_from_voxel`(末尾加残差) 独立；统一管前/后置头；base 恒 detach |
| 3 | 真实原子密度 | `real_atom_density_cube` + `share_encoder(默认 True)` + `cube_size(默认 7)` + `combine_mode∈{concat_mlp(默认),film,mini_residue}`；零初始化恒等；embed 阶段算一次复用所有 recycle |
| 4 | fusion 采样器 | `sampler_modes` 扩 `weighted_cube`/`cube_mean`；per-hook `(a,b,c,d)` = home/含原子/其他 log-bias + 温度(单一 softmax)；`point_feat` 默认 `(1.5,1.0,0.0,1.0)`、`d` 固定 1.0 不随分辨率；现有 per-BOX 循环向量化(行为不变) |

> [!IMPORTANT]
> 所有新开关在 `configs/model/default.yaml` 仅给**中性占位默认**，行为切换由 experiment（或其子配置 override）控制，不要求改根 default.yaml（见项目记忆 `feedback_switches_experiment_controlled`）。

## 已有可复用代码

| 复用项 | 位置 | 用途 |
| --- | --- | --- |
| `_apply_point_feat_detach_routing(point_feat, pseudo_mask, detach_real, detach_pseudo)` | [stage1_model.py:674-704](src/model/stage1_model.py#L674-L704) | 第 1 点直接复用：both→`.detach()`、real-only→按 real、mixed→`torch.where` |
| `_gather_voxel_aux_logit_at_atom_home_voxel(...)` | [stage1_model.py:706-734](src/model/stage1_model.py#L706-L734) | 第 2b 点采集 base（恒返回 `.detach()`）；其 floor/clamp 逻辑供第 3 点 home voxel 复用 |
| `extract_real_tensor_from_mixed` | `_run_atom_head` 内多处（如 [:1277-1287](src/model/stage1_model.py#L1277)） | 第 2b 点把 base 取成 real 序，与 `real_hidden`/`real_point_feat_raw` 对齐 |
| `DensityCubeEncoder`（conv+GAP，尺寸无关） | [density_cube.py:8-252](src/model/sparse_refine/density_cube.py#L8) | 第 3 点共享 encoder；GAP 使其可吃不同 cube 尺寸 |
| `interface_norm_embed_to_point` / `interface_norm_density_to_point` | [stage1_model.py:471-474](src/model/stage1_model.py#L471) | 第 3 点对称新增 `interface_norm_real_density_to_point` |
| `pseudo_density_residual` 零初始化残差范式 | [stage1_atom_head.py:361-372](src/model/stage1_atom_head.py#L361) | 第 3 点 `mini_residue`/combine 零初始化的参照 |
| `_extract_cube_chunk`(b,z,y,x 混合高级索引) | [density_cube.py:162-195](src/model/sparse_refine/density_cube.py#L162) | 第 5 点抽 3³ 邻居的索引骨架；需并行补 in-bounds mask |
| `nn.init.zeros_` 末层零初始化范式 | [sparse_refine_head.py:166-169](src/model/sparse_refine/sparse_refine_head.py#L166) / [stage1_atom_head.py:371-372](src/model/stage1_atom_head.py#L371) | 第 3/5 点收敛进 utils.py 的零初始化残差/FiLM 构建器 |

## 改动文件汇总

| 文件 | 标签 | 摘要 |
| --- | --- | --- |
| [src/model/stage1_model.py](src/model/stage1_model.py) | MODIFY | 第 1/2b/3 点主改：开关入参、fusion detach 路由、base 提前采集与传参、真实原子密度 combine 构造与注入 |
| [src/model/stage1_atom_head.py](src/model/stage1_atom_head.py) | MODIFY | 第 2b 点：后置头 concat base 入参、首层扩维、forward concat |
| [src/model/sparse_refine/sparse_refine_head.py](src/model/sparse_refine/sparse_refine_head.py) | MODIFY | 第 2a 点：删耦合约束、forward 残差解耦 |
| [src/model/sparse_refine/density_cube.py](src/model/sparse_refine/density_cube.py) | MODIFY | 第 3 点：`cube_size` 改 forward 入参 |
| [configs/model/default.yaml](configs/model/default.yaml) | MODIFY | 删旧 fusion detach 键、加新键与 2b/3 中性默认 |
| `configs/experiment/MINI_*.yaml`(8) + `detach_main.yaml` | MODIFY | 替换 `detach_voxel_feat_into_point` 为两新键 |
| `configs/model/sparse_refine/density_cube/real_default.yaml` | NEW | 仅 `share_encoder=false` 独立 encoder 时需要 |
| `tests/model/test_stage1_atom_head.py` 等 | MODIFY | 同步 2b 新签名/扩维，保留旧行为用例 |
| [src/model/utils.py](src/model/utils.py) | NEW | 第 3/5 点通用构建器：零初始化残差 MLP、FiLM、cube 加权参数容器；点 3 combine 委托此处 |
| `configs/model/fusion/*.yaml` | MODIFY | 第 5 点：`sampler_modes` 可填新值 + 新增等长 `sampler_cube_init` |

> [!NOTE]
> 第 5 点的 [stage1_model.py](src/model/stage1_model.py)(sampler 路由+向量化) 与 [density_cube.py](src/model/sparse_refine/density_cube.py)(cube 骨架复用+mask) 与上表第 1/3 点改动同文件，分段落地。

## Proposed Changes

### 第 1 点：拆分 voxel→point 融合 detach

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `__init__` 入参

- **删除**入参 `detach_voxel_feat_into_point: bool = False`（[:171](src/model/stage1_model.py#L171)）及 `self.detach_voxel_feat_into_point`（[:269](src/model/stage1_model.py#L269)）。
- **新增**入参与属性：

| 入参 | 类型 | 默认 | 属性 | 语义 |
| --- | --- | --- | --- | --- |
| `detach_voxel_feat_into_real_point` | bool | `False` | `self.detach_voxel_feat_into_real_point` | 真实原子接收采样 voxel 特征时是否 detach；False=带梯度回流（实测提升体素分支结合体素检测） |
| `detach_voxel_feat_into_pseudo_point` | bool | `True` | `self.detach_voxel_feat_into_pseudo_point` | 伪原子接收采样 voxel 特征时是否 detach；True=噪声不污染共享 backbone |

- 同步更新 docstring [:242](src/model/stage1_model.py#L242) 与文件顶部契约 [:56-57](src/model/stage1_model.py#L56)。

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `_fuse_point_variable`（[:973-990](src/model/stage1_model.py#L973-L990)）

把单开关 detach（[:973-974](src/model/stage1_model.py#L973)）替换为按类型路由。`pseudo_mask` 的取用从 [:985](src/model/stage1_model.py#L985) **上移**到 sample 之后：

```python
# 在 _sample_voxel_feature_trilinear(...) 之后:
pseudo_mask = point_like.get("pseudo_mask", None) if hasattr(point_like, "get") else None
pseudo_mask = validate_pseudo_mask(pseudo_mask, int(sampled_voxel_feat.shape[0]),
                                   name="VolumePointStage1Model._fuse_point_variable")
sampled_voxel_feat = self._apply_point_feat_detach_routing(
    sampled_voxel_feat, pseudo_mask,
    self.detach_voxel_feat_into_real_point, self.detach_voxel_feat_into_pseudo_point)
```

> [!NOTE]
> 下游原 [:985-990](src/model/stage1_model.py#L985) 的 `pseudo_mask`/`validate_pseudo_mask` 改为复用上移后的结果，避免二次校验。`_apply_point_feat_detach_routing` 作用对象从 `point_feat` 换成 `sampled_voxel_feat`，语义同构。

### 第 2a 点：SparseRefineHead 残差解耦

#### [DELETE] [src/model/sparse_refine/sparse_refine_head.py](src/model/sparse_refine/sparse_refine_head.py) — 耦合约束（[:103-104](src/model/sparse_refine/sparse_refine_head.py#L103-L104)）

删除 `if self.mode == "residual" and not self.inputs.get("use_voxel_logits", False): raise ValueError(...)`。

#### [MODIFY] [src/model/sparse_refine/sparse_refine_head.py](src/model/sparse_refine/sparse_refine_head.py) — `forward` 末段（[:323-341](src/model/sparse_refine/sparse_refine_head.py#L323-L341)）

| 行为 | 触发条件 |
| --- | --- |
| 取出 `base_logits = voxel_logits` | `use_voxel_logits` **或** `mode=="residual"` |
| 把 `base_logits` 拼进 `final_parts`（MLP 输入） | 仅 `use_voxel_logits` |
| `output_logits = base_logits + output_logits`（末尾加残差） | 仅 `mode=="residual"` |

```python
final_parts = [candidate_message_delta]
need_base = self.inputs.get("use_voxel_logits", False) or self.mode == "residual"
base_logits = None
if need_base:
    if voxel_logits is None:
        raise RuntimeError("use_voxel_logits 或 residual 模式要求 voxel_logits 非空。")
    base_logits = voxel_logits  # detach 由调用方负责
if self.inputs.get("use_voxel_logits", False):
    final_parts.insert(0, base_logits)
if self.inputs.get("use_C_voxel_backbone_feat", False):
    final_parts.insert(1 if self.inputs.get("use_voxel_logits", False) else 0, C_voxel_backbone_feat)
output_logits = self.output_mlp(torch.cat(final_parts, dim=1))
if self.mode == "residual":
    output_logits = base_logits + output_logits
```

> [!WARNING]
> `use_C_voxel_backbone_feat` 的插入下标判断必须从原 `1 if base_logits is not None else 0`（[:335](src/model/sparse_refine/sparse_refine_head.py#L335)）改为 `1 if use_voxel_logits else 0`——因为 residual-only 时 `base_logits` 非空但**不进** `final_parts`。

> [!NOTE]
> `final_input_dim`（[:162-165](src/model/sparse_refine/sparse_refine_head.py#L162)）与 `zero_init_residual`（[:166-169](src/model/sparse_refine/sparse_refine_head.py#L166)）不变；调用方 [stage1_model.py:1414-1415](src/model/stage1_model.py#L1414) 始终传 `voxel_logits_C`，None 检查仅为防御。

### 第 2b 点：atom 前/后置头 concat base 入参

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `__init__` 与前置头构造

- 新增入参 `atom_head_concat_receptor_base_logit: bool = False` → `self.atom_head_concat_receptor_base_logit`。
- 前置头构造（[:519-527](src/model/stage1_model.py#L519)）首层扩维：

```python
front_in = point_out + (int(atom_logit_dim) if self.atom_head_concat_receptor_base_logit else 0)
self.atom_logit_head_front = nn.Sequential(
    nn.Linear(front_in, point_out), act_cls(), nn.Linear(point_out, int(atom_logit_dim)))
```

- atom_head 实例化（[:505-512](src/model/stage1_model.py#L505) 附近）新增 kwargs：`concat_receptor_base_logit=self.atom_head_concat_receptor_base_logit`、`receptor_base_dim=int(atom_logit_dim)`。

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `_run_atom_head`（[:1256-1326](src/model/stage1_model.py#L1256)）：base 提前采集

把 base 采集从 atom_head 调用**之后**（现 [:1296-1313](src/model/stage1_model.py#L1296)）移到**之前**，条件取并集：

```python
need_base = self.refine_receptor_from_voxel or self.atom_head_concat_receptor_base_logit
real_base = None
if need_base:
    real_coord_local, real_batch_index = <现 1300-1307 的 real 坐标/index 提取>
    real_base = self._gather_voxel_aux_logit_at_atom_home_voxel(
        outputs["voxel_logits_aux"], real_coord_local, real_batch_index,
        atom_head_batch["box_shape_zyx"])  # (N_real, C_aux), detached
```

atom_head 调用（[:1260-1267](src/model/stage1_model.py#L1260)）新增入参 `real_receptor_base_logit=(real_base if self.atom_head_concat_receptor_base_logit else None)`。

前置头（[:1314-1323](src/model/stage1_model.py#L1314)）：

```python
front_in_feat = real_point_feat_raw
if self.atom_head_concat_receptor_base_logit:
    front_in_feat = torch.cat([real_point_feat_raw, real_base], dim=1)
front_logits = self.atom_logit_head_front(front_in_feat)
outputs["atom_logits_front"] = front_logits + real_base if self.refine_receptor_from_voxel else front_logits
```

后置头末尾相加（[:1325-1326](src/model/stage1_model.py#L1325)）维持：`refine_receptor_from_voxel` 时 `outputs["atom_logits"] += real_base`。

> [!NOTE]
> 顺序对齐：`real_base` 为 real 序（`extract_real_tensor_from_mixed`），与前置头 `real_point_feat_raw`、后置头 `real_hidden=atom_hidden[~pseudo_mask]` 同序——现 [:1326](src/model/stage1_model.py#L1326) 的末尾相加已依赖该对齐，安全。

> [!WARNING]
> 维度等式 `atom_logit_dim == voxel_aux_logit_dim` 仅"末尾相加"(`refine_receptor_from_voxel`)必需（现有断言不变）；纯 concat 不强制（二分类下 base 维=1）。

#### [MODIFY] [src/model/stage1_atom_head.py](src/model/stage1_atom_head.py) — `Stage1AtomHead`

`__init__`（[:222-240](src/model/stage1_atom_head.py#L222)）新增：

| 入参 | 类型 | 默认 | 消费 |
| --- | --- | --- | --- |
| `concat_receptor_base_logit` | bool | `False` | `real_atom_logit_head` 首层是否扩 `receptor_base_dim` |
| `receptor_base_dim` | int \| None | `None` | concat 时 base 通道数(=atom_logit_dim) |

`real_atom_logit_head` 构造（[:344-353](src/model/stage1_atom_head.py#L344)）首层扩维：

```python
back_in = self.hidden_dim + (int(receptor_base_dim) if concat_receptor_base_logit else 0)
self.real_atom_logit_head = nn.Sequential(
    nn.Linear(back_in, self.hidden_dim), act_layer(), nn.Linear(self.hidden_dim, self.atom_logit_dim)
) if self.enable_atom_head_back else None
```

`forward`（[:440-505](src/model/stage1_atom_head.py#L440)）新增入参 `real_receptor_base_logit: torch.Tensor | None = None`；`atom_logits` 产出（[:499](src/model/stage1_atom_head.py#L499)）前 concat：

```python
real_in = real_hidden
if self.concat_receptor_base_logit:
    if real_receptor_base_logit is None:
        raise RuntimeError("concat_receptor_base_logit=True 时必须传入 real_receptor_base_logit。")
    real_in = torch.cat([real_hidden, real_receptor_base_logit], dim=1)
atom_logits = self.real_atom_logit_head(real_in) if self.real_atom_logit_head is not None else None
```

> [!NOTE]
> prior bias 初始化（[:374-385](src/model/stage1_atom_head.py#L374)）仍作用末层 `real_atom_logit_head[2]`，首层扩维不影响。real-only 路径下 `real_hidden=atom_hidden`，base 同为全 real 序。

### 第 3 点：真实原子 density cube

#### [MODIFY] [src/model/sparse_refine/density_cube.py](src/model/sparse_refine/density_cube.py) — `cube_size` 入参化

`forward`（[:197-252](src/model/sparse_refine/density_cube.py#L197)）签名加 `cube_size: int | None = None`：

```python
def forward(self, voxel_grid, anchor_voxel_zyx, anchor_batch_index, cube_size=None):
    cube_size = self.cube_size if cube_size is None else int(cube_size)
    if cube_size < 1 or cube_size % 2 == 0:
        raise ValueError("cube_size 必须为 >=1 的奇数。")
    radius = cube_size // 2
    ...  # _extract_cube_chunk 接收 cube_size 形参替代 self.cube_size([:180])
```

`_extract_cube_chunk`（[:162-195](src/model/sparse_refine/density_cube.py#L162)）增加 `cube_size` 形参；`__init__` 的 `self.cube_size` 保留为默认值。

> [!NOTE]
> conv+GAP 对空间尺寸无关；7³ 经 `num_downsample=2`→2³ 仍合法。共享 encoder 由此可同时吃伪原子默认 cube 与真实原子 7³。

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `__init__`：开关、encoder、combine

新增入参：

| 入参 | 类型 | 默认 | 允许值 |
| --- | --- | --- | --- |
| `real_atom_density_cube` | bool | `False` | — |
| `real_atom_density_share_encoder` | bool | `True` | — |
| `real_atom_density_cube_size` | int | `7` | 奇数 ≥1 |
| `real_atom_density_combine_mode` | str | `concat_mlp` | `concat_mlp` / `film` / `mini_residue` |
| `real_density_cube_cfg` | dict\|nn.Module\|None | `None` | 仅 `share_encoder=false` 时必填 |

构造逻辑（`F = atom_feature_dim`）：

```python
self.real_density_cube_encoder = None
self.real_density_combine = None
self.interface_norm_real_density_to_point = None
if self.real_atom_density_cube:
    if self.real_atom_density_share_encoder:
        if self.density_cube_encoder is None:
            raise ValueError("share_encoder=True 需已配置 density_cube_encoder。")
    else:
        self.real_density_cube_encoder = instantiate(real_density_cube_cfg)
        if int(self.real_density_cube_encoder.out_dim) != int(F):
            raise ValueError("real density encoder out_dim 必须等于 atom_feature_dim。")
    self.real_density_combine = self._build_real_density_combine(self.real_atom_density_combine_mode, int(F))
    if self.enable_interface_norm:
        self.interface_norm_real_density_to_point = nn.LayerNorm(int(F))
```

`_build_real_density_combine(mode, F) -> Callable[[embed, density], atom_feat]`（全部**末层零初始化→开局恒等**；`concat_mlp`/`film`/`mini_residue` 的底层构建**委托** [src/model/utils.py](src/model/utils.py) 通用构建器，见第 5 点 utils 小节，本处仅做模式分派）：

| mode | 公式 | 零初始化对象 |
| --- | --- | --- |
| `concat_mlp`(默认) | `embed + MLP([embed, density])` | MLP 末层 Linear |
| `film` | `embed*(1+gamma) + beta`，`[gamma,beta]=Linear(density)` | 该 Linear（→gamma=beta=0） |
| `mini_residue` | `embed + Linear(density)`（轻量，省显存保底） | 该 Linear |

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — `_run_embed_head_once`（[:1037-1050](src/model/stage1_model.py#L1037)）：注入

在 embed 裁剪后（`batch["atom_coord_local_voxel"]` 已写，[:1045](src/model/stage1_model.py#L1045)）、return 前插入：

```python
if self.real_atom_density_cube:
    enc = self.density_cube_encoder if self.real_atom_density_share_encoder else self.real_density_cube_encoder
    home_zyx = self._atom_home_voxel_zyx(batch["atom_coord_local_voxel"],
                                         batch["atom_batch_index"], batch["box_shape_zyx"])  # (N_real,3) z,y,x
    real_density_feat = enc(batch["voxel_grid"], home_zyx, batch["atom_batch_index"],
                            cube_size=self.real_atom_density_cube_size)  # (N_real, F)
    if self.interface_norm_real_density_to_point is not None:
        real_density_feat = self.interface_norm_real_density_to_point(real_density_feat)
    batch["atom_feat"] = self.real_density_combine(batch["atom_feat"], real_density_feat)
```

新增 helper `_atom_home_voxel_zyx(atom_coord_local_voxel, atom_batch_index, box_shape_zyx) -> (N,3) long`：抽取 `_gather_voxel_aux_logit_at_atom_home_voxel`（[:725-731](src/model/stage1_model.py#L725)）的 `floor → [2,1,0] → clamp(min=0) → minimum(shape-1)` 逻辑，两处共用。

> [!IMPORTANT]
> `_run_embed_head_once` 在 recycle 循环前**只跑一次**（[:1010](src/model/stage1_model.py#L1010) docstring），`batch["atom_feat"]` 跨 recycle 复用，故真实原子密度**算一次、复用所有 recycle**，无需额外缓存。此处 `batch["atom_feat"]` 已是 `interface_norm_embed_to_point`（[:1038-1040](src/model/stage1_model.py#L1038)）之后的 embed 特征，combine 作用其上。

### 第 5 点：weighted_cube / cube_mean 采样器 + 采样向量化

把 voxel→point 融合的**采样阶段**从"仅 trilinear/nearest"扩成可选的 3³ 邻域加权池化，并把现有 per-BOX 循环采样向量化（行为不变）。新增两个 `sampler_modes` 取值：`weighted_cube`(可学习 softmax 加权)、`cube_mean`(无参数均值)；二者共用同一 cube 抽取，仅 logit 来源不同。

#### 设计公式

对每个点取以其 home 体素(`floor`)为中心的 3³=27 邻域，邻居 i 的权重：

> `w_i = softmax_{有效邻居}( cat_bias(i) − dist_i² / d )`

- `cat_bias(i) ∈ {a, b, c}`：home / 含原子 / 其他 三类 log-bias(无约束实数, 0=中性)。
- `dist_i`：连续点位到第 i 个邻居体素中心的距离(voxel 单位)。
- `d`：温度, `exp(log_d)` 参数化保正。
- 越界/padding 邻居：logit 置 `-inf`(等价"从归一化剔除")。
- `cube_mean`：`cat_bias≡0` 且去掉 dist 项(logit 恒 0) → 有效邻居均匀, 无参数。

类别判定优先级 **home > 含原子 > 其他**：home 体素恒取 `a`；"含原子"= 该 hook 这一级**池化后的真实原子**(`point_like.coord[~pseudo_mask]`) scatter 得到的占据图；占据图每级 scatter 一次、跨 recycle 复用(原子坐标不变)。

> [!NOTE]
> weighted_cube 与第 1 点 detach、typed fusion 正交：采样产物 shape/语义不变, 下游 detach 路由 / interface_norm / `concat_linear` 全部零改动。

#### [NEW] [src/model/utils.py](src/model/utils.py) — 通用构建器

集中放与具体 head 无关的通用构建, 供 stage1_model / atom_head / sparse_refine 复用(只做"移动 + 收敛同义", **不改既有数值行为**)：

| 函数 | 返回 | 用途 |
| --- | --- | --- |
| `build_zero_init_residual_mlp(in_dim, out_dim, hidden_dim, act)` | `nn.Sequential` | 末层零初始化残差 MLP；点 3 `concat_mlp`/`mini_residue` 委托 |
| `build_film(cond_dim, feat_dim)` | `nn.Module` | FiLM：`[gamma,beta]=Linear(cond)`(零初始化→gamma=beta=0), 前向 `feat*(1+gamma)+beta`；点 3 `film` 委托 |
| `build_cube_weighting_params(a, b, c, d)` | `nn.Module` | weighted_cube 的 per-hook `(a,b,c,log_d)` 参数容器 + 前向把 27 邻居的类别/dist² 组成 logit |

> 点 3 的 `_build_real_density_combine` 改为委托上面前两个构建器；本表三个构建器都遵循"末层/条件零初始化→开局恒等"。

#### [MODIFY] [src/model/sparse_refine/density_cube.py](src/model/sparse_refine/density_cube.py) — cube 索引骨架可复用 + in-bounds mask

把 `_extract_cube_chunk`（[:162-195](src/model/sparse_refine/density_cube.py#L162)）的 `(b,z,y,x)` 混合高级索引抽成可复用 helper，新增并行返回 `valid_mask: (P_chunk, k, k, k) bool`(未越界邻居为 True)。density 原路径走 zero-pad、忽略 mask；weighted_cube 用 mask 喂 `-inf`。第 3 点的 `cube_size` 入参化在同文件，一并落地。

#### [MODIFY] [src/model/stage1_model.py](src/model/stage1_model.py) — sampler 路由 + 向量化

- `__init__`：`sampler_modes` 合法集由 `{trilinear, nearest}` 扩为 `{trilinear, nearest, weighted_cube, cube_mean}`（[:375-377](src/model/stage1_model.py#L375)）。新增入参 `sampler_cube_init: Sequence | None`(等长 hook 列表, `weighted_cube` 填 `(a,b,c,d)`, 其余 `None`)；对 `weighted_cube` 的 hook 按 point_name 建 `nn.ModuleDict`(用 utils `build_cube_weighting_params`, 存 `a/b/c` 标量与 `log_d=log(d)`)。
- `_sample_voxel_feature_trilinear`（[:891-935](src/model/stage1_model.py#L891-L935)）：删 per-BOX `for` 循环，改"逐点算归一化 grid + 单次 `F.grid_sample`"(trilinear/nearest 数值不变)；按 sampler_mode 分派 cube 分支。
- 新 helper `_sample_voxel_feature_cube(voxel_feat, point_like, batch, point_name, weighted)`：抽 3³ + valid_mask + dist² + 占据类别 → 组 logit → softmax → 加权和，分块(仿 density_cube `chunk_size`)。占据图按该级 voxel 特征**实际空间尺寸**、用 real 子集坐标 scatter，每级缓存。
- 注入点不变：仍在 `_fuse_point_variable`（[:964](src/model/stage1_model.py#L964)）内替换原 `_sample_voxel_feature_trilinear` 调用；输出 `(N, C_voxel)` 后续路径零改动。

> [!WARNING]
> cube 的整数 home 索引与越界判断必须用该 hook 这一级 voxel 特征的**实际 D/H/W**(非 `box_shape_zyx` 全分辨率)。这是相对 density_cube(单分辨率输入图)唯一新增的坐标处理。粗分辨率 hook 上 3³ 覆盖比例大、home/含原子/其他三类会退化, 故默认只在满分辨率 `point_feat` 启用(见配置)。

## 配置改动

#### [MODIFY] [configs/model/default.yaml](configs/model/default.yaml) — backbone 块（[:38-46](configs/model/default.yaml#L38)）

- 删 `detach_voxel_feat_into_point: false`（[:42](configs/model/default.yaml#L42)）。
- 加：

```yaml
  detach_voxel_feat_into_real_point: false    # bool, 真实原子接收 voxel 特征是否 detach; 默认 False(回流提升体素分支检测)
  detach_voxel_feat_into_pseudo_point: true   # bool, 伪原子接收 voxel 特征是否 detach; 默认 True(噪声隔离)
  atom_head_concat_receptor_base_logit: false # bool, 前/后置头是否把 home 体素 aux base logit 拼进输入(与 refine_receptor_from_voxel 正交)
  real_atom_density_cube: false               # bool, 真实原子是否加 density cube 特征
  real_atom_density_share_encoder: true       # bool, 真实原子 density 是否复用伪原子 DensityCubeEncoder
  real_atom_density_cube_size: 7              # int, 真实原子 cube 边长(奇数)
  real_atom_density_combine_mode: concat_mlp  # str, embed↔density 融合; concat_mlp/film/mini_residue
  real_density_cube_cfg: null                 # dict|null, 独立 encoder 配置; 仅 share_encoder=false 需要
```

#### [MODIFY] 实验配置 fan-out（第 1 点必改）

下表 9 个文件各自 backbone 块里的 `detach_voxel_feat_into_point: <v>` 替换为两新键：

| 文件 | 原 `detach_voxel_feat_into_point` |
| --- | --- |
| [configs/experiment/MINI_both_front.yaml](configs/experiment/MINI_both_front.yaml#L56) | true |
| `configs/experiment/MINI_both_both.yaml` | (核对) |
| `configs/experiment/MINI_none_back.yaml` | (核对) |
| `configs/experiment/MINI_none_both.yaml` | (核对) |
| `configs/experiment/MINI_none_front.yaml` | (核对) |
| `configs/experiment/MINI_real_back.yaml` | (核对) |
| `configs/experiment/MINI_real_both.yaml` | (核对) |
| `configs/experiment/MINI_real_front.yaml` | (核对) |
| `configs/experiment/detach_main.yaml` | (核对) |

替换为：

```yaml
    detach_voxel_feat_into_real_point: false
    detach_voxel_feat_into_pseudo_point: true
```

> [!WARNING]
> 原值 `true` 含义是"真实+伪都 detach"；新默认是"真实不 detach、伪 detach"。逐个核对每个实验意图：若某实验确需真实也 detach，显式写 `detach_voxel_feat_into_real_point: true`。

> [!NOTE]
> 2b/3 的新开关默认中性，仅在需要的新实验里按需开启；既有 9 个实验不必新增这些键。

#### [MODIFY] `configs/model/fusion/*.yaml` — sampler 取值与 cube 初值（第 5 点）

- `sampler_modes` 可填 `weighted_cube` / `cube_mean`(基线 [d4321/e234d4](configs/model/fusion/d4321.yaml) 仍全 `trilinear`, 不动)。
- 新增等长 `sampler_cube_init`：`weighted_cube` 的 hook 填 `[a, b, c, d]`, 其余 hook 填 `null`。
- 推荐起步：仅把满分辨率 `point_feat` 这一个 hook 设为 `weighted_cube`，初值 `[1.5, 1.0, 0.0, 1.0]`(a>b>c log-bias 灌先验；d=温度, 固定 1.0、不随分辨率)。

```yaml
    # list[str], 采样器: trilinear/nearest/weighted_cube(可学习 3³ softmax)/cube_mean(无参数 3³ 均值)
    sampler_modes: ["trilinear", "trilinear", "trilinear", "trilinear", "weighted_cube"]
    # list[list|null], 与 sampler_modes 等长; weighted_cube 的 hook 填 [a,b,c,d]: a/b/c=home/含原子/其他 log-bias, d=温度
    sampler_cube_init: [null, null, null, null, [1.5, 1.0, 0.0, 1.0]]
```

> [!NOTE]
> `sampler_modes` / `sampler_cube_init` 与 `point_fusion_map` / `point_fusion_modes` 同序等长(校验见 [stage1_model.py:373](src/model/stage1_model.py#L373))。

#### [NEW] `configs/model/sparse_refine/density_cube/real_default.yaml`（可选）

仿 [density_cube/default.yaml](configs/model/sparse_refine/density_cube/default.yaml) 写入 `cfg.model.backbone.real_density_cube_cfg`；`cube_size: 7`、`num_conv` 可减为 1 省显存、`out_dim: ${model.backbone.point_backbone.atom_feature_dim}`。仅 `real_atom_density_share_encoder=false` 时需要。

## 不修改的部分

- candidate/anchor/sparse-refine 既有算法、PTV3 backbone、wrapper loss/metric 框架。
- 伪原子 density 路径（`density_cube_encoder` 对 P 的调用、`pseudo_density_residual`）保持不变。
- `zero_init_residual`、`refine_receptor_from_voxel` 的"末尾加残差"与维度断言语义不变。
- 不引入"让前置/后置头梯度回流 voxel aux"的开关（base 恒 detach）。

## Verification Plan

| # | 验证 | 覆盖改动 | 关键断言 |
| --- | --- | --- | --- |
| V1 | mixed batch `torch.autograd.grad` | 第 1 点 | `pseudo=True/real=False` 时伪原子点 loss **不**触达 voxel backbone、真实原子点 loss **触达**；both/real-only 退化与 `_apply_point_feat_detach_routing` 一致 |
| V2 | 旧→新等价 | 第 1 点 | 单开关旧值的等价取值下 forward 逐元素一致 |
| V3 | 四组合 `(use_voxel_logits, mode)` | 第 2a 点 | 全部 forward 通过；`residual+use_voxel_logits=False` MLP 不含 base、末尾加 base、`zero_init` 下初值==base；`direct+use_voxel_logits=True` 不加残差但 base 进 MLP |
| V4 | 2b concat × refine 4 组合 | 第 2b 点 | 前/后置头首层维正确扩张、forward 通过；base detach 无回流；front/back 与 `real_base` 同序；`enable_atom_head_back=False` 后置 None 不崩 |
| V5 | combine 恒等 | 第 3 点 | 三种 `combine_mode` 开局 `atom_feat` 数值==未开时（零初始化）；share/独立 encoder 均可实例化 forward；`cube_size=7` 经共享 encoder 正常 |
| V6 | 计算次数 | 第 3 点 | `max_recycles>1` 时真实原子密度只在 embed 阶段调用一次 |
| V7 | 实例化跑通 | 全部 | `+experiment=...` 各开关组合跑 1 个 train+val step |
| V8 | 回归 | 全部 | 全部新开关中性默认时 forward/loss/metric 键与改动前一致；四 voxel-only 配置零改动通过 |
| V9 | weighted_cube/cube_mean | 第 5 点 | `cube_mean`==有效邻居均值；`weighted_cube` 数值符合 softmax 公式、越界邻居被 `-inf` 剔除；`a/b/c/log_d` 有梯度而占据图无梯度；`max_recycles>1` 时占据图只 scatter 一次 |
| V10 | trilinear 向量化等价 | 第 5 点 | 向量化后 `trilinear`/`nearest` forward 与旧 per-BOX 循环逐元素一致(含空点 box、dtype 转换、`padding_mode=zeros`) |

> [!NOTE]
> 执行顺序：第 1 → 2a → 2b → 3 → 5（风险递增）；第 1、2a 小且独立，可作第一批快速落地。第 5 点内部：先做 per-BOX 向量化(行为不变重构, 过 V10 当作无回归基线)，再上 cube_mean，最后 weighted_cube(探索性最强)。每段过对应 V# 再进入下一段。

## 对齐契约同步（随实现更新）

- [src/model/stage1_model.py](src/model/stage1_model.py#L56) 顶部契约：fusion detach 双开关、`atom_head_concat_receptor_base_logit`、真实原子 density 字段、`sampler_modes` 新值 `weighted_cube`/`cube_mean` 与 `sampler_cube_init`。
- [src/model/utils.py](src/model/utils.py)：新文件顶部说明通用构建器清单与"仅移动/收敛同义、不改数值"的边界。
- [src/model/stage1_atom_head.py](src/model/stage1_atom_head.py#L4) 顶部"对齐契约"：`forward` 新增 `real_receptor_base_logit`、`real_atom_logit_head` 首层扩维。
- `CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md` 相关段。
- `tests/model/test_stage1_atom_head.py`：覆盖 concat base 新签名与扩维，保留旧行为用例（记忆 `feedback_test_no_regression`）。
