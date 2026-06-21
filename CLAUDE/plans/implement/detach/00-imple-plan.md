# detach 梯度路由 / 前后二分类头 / 接口 Norm / density 残差 / refine 调度 —— 实施计划（implement plan）

> 设计动机与梯度语义见同目录 `00-grad-routing-heads-norm.md`。本文是**可执行的 file-by-file 改动清单**：精确到函数、edit 位置、新签名、控制流、配置文件内容与验证步骤。行号为撰写时锚点，实施时以实际代码为准。

## 0. 目标与非目标

**目标**：
1. 在 `VolumePointStage1Model` 加入 9 个梯度路由/头/Norm 开关 + atom head 内 density 残差。
2. 把受体原子二分类做成"前置头(point backbone 末端)"与"后置头(atom head 末端)"两路，各自可开关、各出一个 loss term、可选 `refine_receptor_from_voxel` 残差。
3. `refine` 吃 voxel 信息统一 detach；voxel→point 融合可 detach；refine loss 调度新增 `start_on_ratio` 硬 0 延迟。
4. 全局默认**中性**（复现旧行为），新建 1 个主实验 + 8 个消融实验配置承载"最大 detach"及其网格。
5. 保证 `emb_unet / unet_c1 / unet_c2 / unet000` 四个 voxel-only 配置**零改动**继续训练。

**非目标**：
- 不引入 GradNorm / PCGrad 等多任务再加权。
- 不重构 wrapper / 不改 PTV3 / 不动 candidate/anchor/density cube 既有算法。
- 不为退化角（point backbone 完全无监督）写 fail-fast。
- 不兼容旧 `atom_loss_weight` / 旧 `detach_voxel_logits`（直接删）。

## 1. 改动文件总览

| 文件 | 改动 |
| --- | --- |
| `src/model/stage1_model.py` | 新增 9 开关入参 + density 残差透传；detach 路由；前置头；refine_receptor；fusion detach；refine voxel detach 统一 |
| `src/model/stage1_atom_head.py` | 后置头开关门控；`pseudo_density_residual` 残差块 |
| `src/model/sparse_refine/sparse_refine_head.py` | 删 `detach_voxel_logits`；接口 Norm（per-source LN）门控 |
| `src/wrappers/voxel_point_stage1_losses.py` | `compute_atom_loss_term` 参数化 `logits_key`/`name` |
| `src/wrappers/voxel_point_stage1.py` | atom 权重改名 front/back；front/atom metric 分支与 None 守卫；refine 调度加 `start_on` |
| `configs/model/default.yaml` | backbone 块加 9 个中性默认开关 |
| `configs/loss/sparse_refine.yaml` | `atom_loss_weight`→front/back；schedule 加 `start_on_*` |
| `configs/model/sparse_refine/sparse_refine_head/default.yaml` | 删 `detach_voxel_logits` 行 |
| `configs/experiment/detach_main.yaml` | 新建：主实验（大数据集 + 最大 detach） |
| `configs/experiment/detach_abl_*.yaml` | 新建 8 个：消融网格（小数据集） |

## 2. 配置架构与默认值策略

- **中性全局默认**写进 `configs/model/default.yaml::backbone`，使所有现有/未来非 detach 配置零改动（见 §4.1）。
- **最大 detach 主实验**与消融网格只存在于新建实验配置（§4.4 / §4.5）。
- **全二分类**：`aux=1 / atom=1 / ligand=1`（`task: binary`），`refine_receptor` 维度天然一致。
- 四个 voxel-only 配置（`point_backbone: zeros`、`enable_atom_head: false`、`atom_loss=None`）继承中性默认：前置头不建、`refine_receptor` 关（不触发维度断言）、其余 detach/norm 关 → 行为不变。

## 3. 代码改动（file-by-file）

### 3.1 `src/model/stage1_model.py`

**(a) `__init__` 新增入参**（追加到现有签名末尾，全部给 Python 默认值）：
```
detach_real_point_feat: bool = False,
detach_pseudo_point_feat: bool = False,
detach_voxel_feat_into_point: bool = False,
detach_voxel_into_refine: bool = True,
enable_atom_head_front: bool = False,
enable_atom_head_back: bool = True,
refine_receptor_from_voxel: bool = False,
pseudo_density_residual: bool = False,
enable_interface_norm: bool = False,
```
存为 `self.<name>`。`enable_atom_head_back`、`pseudo_density_residual`、`enable_interface_norm` 透传给 `Stage1AtomHead`（见 3.2）；`enable_interface_norm` 也透传给 sparse refine head（见 3.3）。

**(b) 前置头构造**（在现有 atom head 构造块 [stage1_model.py:410-447](src/model/stage1_model.py#L410) 之后）：
```
self.atom_logit_head_front = None
if self.enable_atom_head_front:
    point_out = int(self.point_backbone.out_channels)
    self.atom_logit_head_front = nn.Sequential(
        nn.Linear(point_out, point_out), act_cls(), nn.Linear(point_out, int(atom_logit_dim)),
    )
    # 复用与后置头一致的 prior_prob/prior_probs bias 初始化（refine_receptor 开时由 (c) 覆盖为零）
```

**(c) `refine_receptor_from_voxel` 维度断言 + 零初始化**（`__init__` 末尾，sparse refine head 构造之后）：
```
if self.refine_receptor_from_voxel:
    if int(atom_logit_dim) != int(self.voxel_backbone.voxel_aux_logit_dim):
        raise ValueError("refine_receptor_from_voxel 要求 atom_logit_dim == voxel_aux_logit_dim。")
    # 末层 Linear 零初始化(weight+bias), 初值 = 照抄 voxel_aux base; 同时覆盖 prior bias
    for head in (self.atom_logit_head_front, getattr(self.atom_head, "real_atom_logit_head", None)):
        if head is not None:
            last = next(m for m in reversed(head) if isinstance(m, nn.Linear))
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
```
`refine_receptor` 要求 voxel aux head 已启用（`aux_head_hidden_channels>0`），否则 forward 时无 `voxel_logits_aux`。本实验 `aux=32` 满足。

**(d) detach 路由纯函数**（新增静态方法；用条件分支避免冗余 `where`）：
```
@staticmethod
def _apply_point_feat_detach_routing(point_feat, pseudo_mask, detach_real, detach_pseudo):
    if detach_real and detach_pseudo:
        return point_feat.detach()
    if detach_real:
        real_mask = torch.ones(point_feat.shape[0], dtype=torch.bool, device=point_feat.device) if pseudo_mask is None else ~pseudo_mask
        return torch.where(real_mask[:, None], point_feat.detach(), point_feat)
    if detach_pseudo and pseudo_mask is not None:
        return torch.where(pseudo_mask[:, None], point_feat.detach(), point_feat)
    return point_feat
```

**(e) home-voxel base 抽取 helper**（供 refine_receptor）：
```
@staticmethod
def _gather_voxel_aux_logit_at_atom_home_voxel(voxel_logits_aux, atom_coord_local_voxel, atom_batch_index, box_shape_zyx):
    idx_xyz = torch.floor(atom_coord_local_voxel).to(torch.long)        # (N,3) x,y,z
    idx_zyx = idx_xyz[:, [2, 1, 0]].clamp(min=0)                        # (N,3) z,y,x
    shape_zyx = box_shape_zyx.to(idx_zyx.device)[atom_batch_index]      # (N,3) z,y,x
    idx_zyx = torch.minimum(idx_zyx, shape_zyx - 1)
    base = voxel_logits_aux[atom_batch_index, :, idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]  # (N, C_aux)
    return base.detach()
```

**(f) `forward` 最后一轮改动**（[stage1_model.py:1234-1266](src/model/stage1_model.py#L1234)）：组装 `outputs` 时
- 保留原始 `outputs["point_feat_raw"] = point_output_dict["point_feat"]`；
- `outputs["fused_point_feat"] = self._apply_point_feat_detach_routing(point_output_dict["point_feat"], pseudo_mask, self.detach_real_point_feat, self.detach_pseudo_point_feat)`（`pseudo_mask` 取 `point_batch.get("pseudo_mask")`，real-only 路径为 None）；
- 后置头(atom head)与 `_run_sparse_refine_head` 的 `P_point_backbone_feat` 均消费 `fused_point_feat`（已按开关 detach）。

**(g) `_fuse_point_variable`**（[stage1_model.py:787](src/model/stage1_model.py#L787)）：在 `torch.cat([point_like.feat, sampled_voxel_feat])` 前
- 若 `self.detach_voxel_feat_into_point`：`sampled_voxel_feat = sampled_voxel_feat.detach()`；
- 若 `self.enable_interface_norm`：对 `point_like.feat` 与 `sampled_voxel_feat` 各过一个 `nn.LayerNorm`（在 `__init__` 按 fusion 变量名建好 `ModuleDict`，关时不建、走恒等）。

**(h) `_run_atom_head`**（[stage1_model.py:1035-1090](src/model/stage1_model.py#L1035)）。函数签名是 `(self, outputs, atom_head_batch, pseudo_layout)`，**没有 `batch` 形参**，凡 BOX 级字段一律取自 `atom_head_batch`（mixed 路径下它由 `inject_pseudo_atoms` 复制了 real batch 的所有 BOX 级字段，含 `box_shape_zyx`）。
- 透传 `pseudo_density_feat = outputs.get("pseudo_density_feat")` 给 atom head（见 3.2）。
- atom head 返回、且 `outputs["atom_logits"]` / 真实原子 BOX 级字段写好之后，计算 refine_receptor base 并加到前/后置头。real 坐标与 batch index 自包含地从 `atom_head_batch` + `pseudo_layout` 取，不依赖 outputs 写入顺序：
```
base = None
if self.refine_receptor_from_voxel:
    if pseudo_layout is not None:
        real_coord_local = extract_real_tensor_from_mixed(atom_head_batch["atom_coord_local_voxel"], pseudo_layout)
        real_batch_index = extract_real_tensor_from_mixed(atom_head_batch["atom_batch_index"], pseudo_layout)
    else:
        real_coord_local = atom_head_batch["atom_coord_local_voxel"]
        real_batch_index = atom_head_batch["atom_batch_index"]
    base = self._gather_voxel_aux_logit_at_atom_home_voxel(
        outputs["voxel_logits_aux"], real_coord_local, real_batch_index, atom_head_batch["box_shape_zyx"])
if self.atom_logit_head_front is not None:
    real_point_feat_raw = extract_real_tensor_from_mixed(outputs["point_feat_raw"], pseudo_layout) if pseudo_layout is not None else outputs["point_feat_raw"]
    front_logits = self.atom_logit_head_front(real_point_feat_raw)
    outputs["atom_logits_front"] = front_logits + base if base is not None else front_logits
if outputs.get("atom_logits") is not None and base is not None:
    outputs["atom_logits"] = outputs["atom_logits"] + base
```
- 后置 `atom_logits` 当 `enable_atom_head_back=False` 时为 None（atom head 不产出，见 3.2）。

**(i) `_run_sparse_refine_head`**（[stage1_model.py:1145-1158](src/model/stage1_model.py#L1145)）：用 `self.detach_voxel_into_refine` 统一控制三处：
- `voxel_logits_C`：`True` 用已 detach 的 `outputs["candidate_logits"]`（[candidate_set.py:405](src/model/sparse_refine/candidate_set.py#L405) 已 `.detach()`）；`False` 从 `voxel_output_dict["voxel_logits_ligand"]` 现 gather（带梯度）。
- `C_voxel_backbone_feat` / `P_voxel_backbone_feat`：`True` 时 `.detach()`。
- 删除原 `getattr(self.sparse_refine_head, "detach_voxel_logits", True)` 分支。

**(j) `_prepare_pseudo_batch`**（[stage1_model.py:945-966](src/model/stage1_model.py#L945)）：把 density cube 输出存入透传字典：`pseudo_outputs["pseudo_density_feat"] = pseudo_feat`（即 `_condition_anchor_pseudo_feat` 之后、inject 所用的同一张量；anchor 顺序，与 `atom_hidden[pseudo]` 一致）。

### 3.2 `src/model/stage1_atom_head.py`

**(a) `Stage1AtomHead.__init__` 新增入参** `enable_atom_head_back: bool = True`、`pseudo_density_residual: bool = False`、`pseudo_density_in_dim: int | None = None`（= `point_backbone.atom_feature_dim`，由 stage1_model 注入）、`enable_interface_norm: bool = False`。
- 后置头门控：`real_atom_logit_head` 仅在 `enable_atom_head_back` 时构造（[stage1_atom_head.py:335-339](src/model/stage1_atom_head.py#L335)）；否则置 None。prior bias 初始化随之只在构造时执行。
- density 残差块（仅 `pseudo_density_residual` 时构造；**先 LayerNorm 后 Linear，对末层 Linear 零初始化**，保证初值为 0、不改变 `P_atom_head_feat`）：
```
self.density_residual = nn.Sequential(
    nn.LayerNorm(int(pseudo_density_in_dim)),
    nn.Linear(int(pseudo_density_in_dim), self.pseudo_feature_dim),
)
nn.init.zeros_(self.density_residual[-1].weight); nn.init.zeros_(self.density_residual[-1].bias)
```

**(b) `forward` 新增可选入参** `pseudo_density_feat: torch.Tensor | None = None`（[stage1_atom_head.py:411-419](src/model/stage1_atom_head.py#L411)）：
- pseudo_feature 计算处（[stage1_atom_head.py:463](src/model/stage1_atom_head.py#L463)）：
```
pseudo_feature = self.pseudo_feature_head(atom_hidden[pseudo_mask])
if self.density_residual is not None and pseudo_density_feat is not None:
    pseudo_feature = pseudo_feature + self.density_residual(pseudo_density_feat)
```
- atom_logits 计算（[stage1_atom_head.py:466](src/model/stage1_atom_head.py#L466)）：`real_atom_logit_head is None`（后置头关）时 `atom_logits = None`。

### 3.3 `src/model/sparse_refine/sparse_refine_head.py`

- **删除** `detach_voxel_logits` 入参（[sparse_refine_head.py:38](src/model/sparse_refine/sparse_refine_head.py#L38)）、`self.detach_voxel_logits`（[:85](src/model/sparse_refine/sparse_refine_head.py#L85)）；forward 内 `base_logits = voxel_logits.detach() if ... else voxel_logits`（[:285](src/model/sparse_refine/sparse_refine_head.py#L285)）改为直接 `base_logits = voxel_logits`（detach 由调用方 §3.1(i) 负责）。
- **接口 Norm**（`enable_interface_norm` 注入）：在拼接前对**学习特征源**各加 `nn.LayerNorm`，且**先归一化再 expand/index_select**（计算量降到约 `1/K_nn`，数学等价）：
  - P content：进 `P_content_mlp`（[:221](src/model/sparse_refine/sparse_refine_head.py#L221)）前对 `P_point_backbone_feat`、`P_atom_head_feat` 各 LN（二者各只此一处消费，可就地归一化）。
  - edge：`P_voxel_backbone_feat` 在 index_select（[:247](src/model/sparse_refine/sparse_refine_head.py#L247)）**之前**对源张量 `(sumP,C)` LN；`C_voxel_backbone_feat` 在 expand（[:240](src/model/sparse_refine/sparse_refine_head.py#L240)）**之前**对源张量 `(sumC,C)` LN。`relative_coords`、`candidate_class_embedding` 不 LN。
  - **`C_voxel_backbone_feat` 有两个消费者**（edge_mlp 与末层 head [:287-290](src/model/sparse_refine/sparse_refine_head.py#L287)）：只对喂给 edge 的那份做 LN（用一个局部变量 `C_voxel_edge`），**末层 head 仍用原始未 LN 的 `C_voxel_backbone_feat`**，以保 `base_logits` 残差语义。
  - 末层 head 拼接整体不再额外 LN。
  - 关时不建 LN、走恒等。

### 3.4 `src/wrappers/voxel_point_stage1_losses.py`

`compute_atom_loss_term`（[voxel_point_stage1_losses.py:53-97](src/wrappers/voxel_point_stage1_losses.py#L53)）新增参数 `logits_key: str = "atom_logits"`、`name: str = "atom"`；内部 `atom_logits = outputs[logits_key]`，`LossTerm(name=name, ...)`。target/mask 仍取 `outputs["atom_target"]`/`outputs["atom_valid_mask"]`（前后置共享真实原子监督）。

### 3.5 `src/wrappers/voxel_point_stage1.py`

**(a) `__init__`**（[:40](src/wrappers/voxel_point_stage1.py#L40)）：删 `atom_loss_weight`；加 `atom_loss_front_weight: float = 0.0`、`atom_loss_back_weight: float = 1.0`。`save_hyperparameters` 自动收录新名。

**(b) `_compute_total_loss`**（[:493-494](src/wrappers/voxel_point_stage1.py#L493)）替换 atom 段，前后置各用 `outputs.get(...) is not None` 守卫：
```
if self.atom_loss is not None:
    if outputs.get("atom_logits") is not None:
        loss_terms.append(compute_atom_loss_term(outputs=outputs, batch=batch, loss_module=self.atom_loss,
            weight=float(self.hparams.atom_loss_back_weight), logits_key="atom_logits", name="atom"))
    if outputs.get("atom_logits_front") is not None:
        loss_terms.append(compute_atom_loss_term(outputs=outputs, batch=batch, loss_module=self.atom_loss,
            weight=float(self.hparams.atom_loss_front_weight), logits_key="atom_logits_front", name="atom_front"))
```

**(c) `_build_metric_branch_specs`**（[:222-226](src/wrappers/voxel_point_stage1.py#L222)）：原 `atom` 分支的 enabled 改为同时受后置头控制；新增 `atom_front` 分支受前置头控制（两个 metric 键分开，不合并）：
```
back_on  = bool(getattr(self._unwrap_backbone(), "enable_atom_head_back", True))
front_on = bool(getattr(self._unwrap_backbone(), "enable_atom_head_front", False))
MetricBranchSpec("atom",       self.atom_loss is not None and back_on,  int(getattr(self.atom_loss, "num_classes", 2)), self.class_names, None),
MetricBranchSpec("atom_front", self.atom_loss is not None and front_on, int(getattr(self.atom_loss, "num_classes", 2)), self.class_names, None),
```

**(d) validation 更新**（[:685-688](src/wrappers/voxel_point_stage1.py#L685)）：把原 `if self.atom_loss is not None and "atom_logits" in outputs:` 的判据改成 `outputs.get("atom_logits") is not None`（后置头关时 `atom_logits=None`，旧的 `in` 判据会把 None 传给 metric manager 而崩）；并追加前置头分支：
```
if self.atom_loss is not None and outputs.get("atom_logits") is not None:
    ...  # 原 atom 分支 update_branch 不变
if self.atom_loss is not None and outputs.get("atom_logits_front") is not None:
    front_mask = outputs.get("atom_valid_mask", batch_dict.get("atom_valid_mask", torch.ones_like(batch_dict["atom_label"], dtype=torch.bool)))
    self.val_metrics.update_branch(branch_name="atom_front", logits=outputs["atom_logits_front"],
        target=outputs.get("atom_target", batch_dict["atom_label"]), mask=front_mask)
```

**(e) refine 调度 `start_on`**：
- 新增 `_resolve_sparse_refine_loss_start_on_steps(self, sched_cfg)`，仿 [:423-443](src/wrappers/voxel_point_stage1.py#L423)：`start_on_steps` 非 null 优先返回；否则 `start_on_ratio` 为 null → 返回 0；否则 `round(total_steps * start_on_ratio)`。
- `_compute_sparse_refine_loss_effective_weight`（[:445-472](src/wrappers/voxel_point_stage1.py#L445)）：**先解析 start_on 并校验，再处理 warmup==0**（删掉原 `if warmup_steps == 0: return final` 短路，统一到下面的 `>= warmup` 分支，避免 `start_on>0, warmup==0` 静默忽略延迟）：
```
warmup_steps   = self._resolve_sparse_refine_loss_warmup_steps(sched_cfg)
start_on_steps = self._resolve_sparse_refine_loss_start_on_steps(sched_cfg)
if start_on_steps > warmup_steps:
    raise ValueError("ligand_sparse_refine_loss_schedule: start_on 必须 <= warmup。")
if global_step < start_on_steps:
    return torch.tensor(0.0, device=self.device, dtype=torch.float32)
if global_step >= warmup_steps:                     # 同时覆盖 warmup_steps==0 与 start_on==warmup 的阶跃
    return torch.tensor(final_weight, device=self.device, dtype=torch.float32)
progress = (float(global_step) - start_on_steps) / float(warmup_steps - start_on_steps)
return torch.tensor(start_weight + progress * (final_weight - start_weight), device=self.device, dtype=torch.float32)
```

## 4. 配置改动

### 4.1 `configs/model/default.yaml`（backbone 块 [:14-33](configs/model/default.yaml#L14) 末尾追加中性默认）
```yaml
  # --- detach / heads / norm 开关 (中性默认: 复现旧行为) ---
  enable_atom_head_front: false
  enable_atom_head_back: true
  detach_real_point_feat: false
  detach_pseudo_point_feat: false
  detach_voxel_feat_into_point: false
  detach_voxel_into_refine: true
  refine_receptor_from_voxel: false
  pseudo_density_residual: false
  enable_interface_norm: false
```

### 4.2 `configs/loss/sparse_refine.yaml`
- 删 `atom_loss_weight: 0.1`（[:29](configs/loss/sparse_refine.yaml#L29)），加：
```yaml
  atom_loss_front_weight: 0.1
  atom_loss_back_weight: 0.1
```
- schedule（[:93-98](configs/loss/sparse_refine.yaml#L93)）补两行（中性默认 null）：
```yaml
    start_on_steps: null
    start_on_ratio: null
```

### 4.3 `configs/model/sparse_refine/sparse_refine_head/default.yaml`
删 `detach_voxel_logits: true`（[:15](configs/model/sparse_refine/sparse_refine_head/default.yaml#L15)）。

### 4.4 新建主实验 `configs/experiment/detach_main.yaml`
```yaml
# @package _global_
# 主实验: 全二分类 sparse refine + 最大 detach + density 残差关闭 + 大数据集
defaults:
  - override /model/voxel_backbone: stardard
  - override /model/point_backbone: stardard
  - override /model/fusion: e234d4
  - override /model/atom_head: stardard
  - override /model/task: binary
  - override /model/typed_point: "on"
  - override /model/embed_head: clipbig
  - override /model/sparse_refine/candidate_set: binary_adaptive
  - override /model/sparse_refine/anchor_sampler: unweighted_fps
  - override /model/sparse_refine/density_cube: default
  - override /model/sparse_refine/anchor_to_candidate: knn_message
  - override /model/sparse_refine/sparse_refine_head: default
  - override /model/sparse_refine/anchor_class_conditioning: none
  - override /loss: sparse_refine
  - override /dataset: emb_unet
  - override /train: B40_L3

name: detach_main
tag: "detach_main"
experiment_group: "detach"
project_name: PV_detach

dataset:
  class_mapping: [0, 1, 1, 1, 1]
  class_names: [background, foreground]
  num_task_classes: 2

model:
  backbone:
    atom_logit_dim: 1
    enable_recycling: true
    max_recycles: 3
    randomize_recycles: true
    detach_recycle_states: true
    act_layer_name: "gelu"
    ffn_type: "gated"
    atom_head_ffn_type: "gated"
    enable_atom_head: true
    # === 最大 detach ===
    enable_atom_head_front: true
    enable_atom_head_back: false
    detach_real_point_feat: true
    detach_pseudo_point_feat: true
    detach_voxel_into_refine: true
    detach_voxel_feat_into_point: true
    refine_receptor_from_voxel: true
    pseudo_density_residual: false
    enable_interface_norm: true
    voxel_backbone:
      ligand_head_hidden_channels: 32
      aux_head_hidden_channels: 32
      num_conv3d_aux: 1
      num_conv3d_ligand: 1
  class_names: ${dataset.class_names}
  monitor_metric: val_score/global/refined_F1
  monitor_mode: max
  validation_diagnostics:
    enabled: true
  ligand_sparse_refine_loss_schedule:
    warmup_ratio: 0.2
    start_on_ratio: 0.1

train:
  enable_batch_size_tuning: false
  batch_size: 5
  val_per_epoch: 5
```

### 4.5 新建 8 个消融 `configs/experiment/detach_abl_<combo>.yaml`
与 `detach_main.yaml` 同 `defaults`，仅以下差异：`dataset` 换成 `MINI_emb_unet`、train 换成 MINI 块、refine 调度换成 `(0.2, 0.3)`、`name/tag/experiment_group` 改名、以及 4 个**消融轴 bool**。固定开关（`detach_voxel_into_refine=true, detach_voxel_feat_into_point=true, refine_receptor_from_voxel=true, pseudo_density_residual=false, enable_interface_norm=true`）与主实验一致。

模板差异块：
```yaml
defaults:
  # 同 detach_main, 但:
  - override /dataset: MINI_emb_unet
model:
  backbone:
    # 固定开关同 detach_main, 仅下面 4 个按下表:
    detach_real_point_feat: <R>
    detach_pseudo_point_feat: <P>
    enable_atom_head_front: <F>
    enable_atom_head_back: <B>
  ligand_sparse_refine_loss_schedule:
    warmup_ratio: 0.3
    start_on_ratio: 0.2
train:
  enable_batch_size_tuning: false
  strict_global_batch_size: false
  global_batch_size: 39
  batch_size: 3
  val_per_epoch: 1
  scheduler:
    warmup_ratio: 0.06
```

8 个组合（3×3−1，去掉 `both detach × 只放后面`）：

| 文件 `detach_abl_*` | detach 档 | loss 档 | R | P | F | B |
| --- | --- | --- | --- | --- | --- | --- |
| `none_front` | none | 前 | F | F | T | F |
| `none_back`  | none | 后 | F | F | F | T |
| `none_both`  | none | 前后 | F | F | T | T |
| `real_front` | real-only | 前 | T | F | T | F |
| `real_back`  | real-only | 后 | T | F | F | T |
| `real_both`  | real-only | 前后 | T | F | T | T |
| `both_front` | both | 前 | T | T | T | F |
| `both_both`  | both | 前后 | T | T | T | T |

> `both × 只放后面`(R=T,P=T,F=F,B=T) 已剔除。8 个组合中均无 backbone/density-cube 成 unused（前置 loss 经注意力、或 refine 经 pseudo 在 `detach_pseudo=False` 时总能训到 point/density），`ddp_find_unused_parameters: false` 全部安全。

## 5. 验证关卡与测试

按 Stage 顺序逐段验证（A 路由 → B 头/残差 → C Norm → D 配置/wrapper → E 调度）。

1. **回归**：四个 voxel-only 配置 + 中性默认 → forward 通过、loss/metric 键与改动前一致。
2. **路由 autograd 断言**：构造带 sparse refine 的小 batch，遍历 §4.5 的 8 组（+ `detach_voxel_*` 开关），用 `torch.autograd.grad` 断言各 loss 是否触达 voxel/point/density-cube 参数，与 `00-grad-routing-heads-norm.md` §4 一致；特别验证主实验取值下三块解耦、且 density cube 经前置 loss 注意力**有**梯度。
3. **头开关**：`enable_atom_head_back=false` → `atom_logits is None`、sparse refine 正常、**validation 不崩**（§3.5(c)(d) 已守卫）；前置头开/关产出 `atom_logits_front` 与 `atom_front` loss/metric。
4. **refine_receptor**：零初始化下前/后置头初始 logits == `voxel_aux` home 体素 detach 值；维度不匹配 fail-fast；base 无梯度回流 voxel。
5. **density 残差**：`pseudo_density_residual=true` 且末层零初始化时 `P_atom_head_feat` 初值=无残差、且 `P_point_backbone_feat`/前置 loss 数值不变。
6. **Norm**：`enable_interface_norm=false` 与未改前逐元素一致；`=true` 下 forward/backward 正常、`C_voxel` 末层 head 用未 LN 版本。
7. **refine 调度**：`(start_on_ratio, warmup_ratio)=(0.1,0.2)` 下若干 `global_step` 返回 0 / 斜坡 / final 三段正确；`start_on>warmup` 抛错；`start_on_ratio=null` 复现旧斜坡；`warmup_steps==0` 且 `start_on==0` 返回 final。
8. **实例化**：`+experiment=detach_main` 与任一 `detach_abl_*` 能实例化并跑 1 个 train+val step。

## 6. 对齐契约同步（随实现更新）

`stage1_model.py` / `stage1_atom_head.py` / `stage1_point_backbone.py` 顶部"对齐契约"docstring、`tri_ligand_sparse_refine/00-master.md`、相关 `tests/model/test_*.py` 注释：新增 `point_feat_raw`、`atom_logits_front`、detach 路由、前/后置头、`refine_receptor`、`pseudo_density_feat`、删除 `detach_voxel_logits`。

## 7. 风险与注意

- **退化角不拦截**：`前置头关 ∧ 后置头开 ∧ R=T ∧ P=T ∧ 残差关` 会使 point/density 成 unused（DDP 崩）；本计划 9 配置均不触发，实验者自负。
- **`binary_adaptive` 运行前提（已核 OK）**：adaptive 在阈值缓存填充前依赖 scheduler warmup 的 fixed-topk；`B40_L3` 的 `scheduler.name=warmup_plateau`（[B40_L3.yaml:72](configs/train/B40_L3.yaml#L72)）满足；且 warmup 窗口内会发生 validation 填缓存（主实验 `val_per_epoch=5`、消融 `=1` 均在 warmup 比例内完成首个 validation）。实现后建议跑一两个 step 实测无阈值缺失报错。
- **两个"warmup"勿混**：`train.scheduler.warmup_ratio`(优化器 LR) 与 `model.ligand_sparse_refine_loss_schedule.warmup_ratio`(refine 权重) 是不同字段。
- **`refine_receptor` 前提**：需 voxel aux head 启用（`aux_head_hidden_channels>0`）且 `atom_logit_dim==voxel_aux_logit_dim`（本实验均=1）。
- **`detach_voxel_into_refine` 默认 True** 有意偏离已发散的 exp3。
- **minimal-code**：全部为现有 forward/`__init__`/wrapper 内插钩子与复用，不新建大模块。

## 8. 执行顺序

A 路由开关（含删 `detach_voxel_logits`）→ B 前/后置头 + `refine_receptor` + density 残差 → C 接口 Norm → D 配置(默认/loss/实验 9 个) + wrapper loss/metric/None 守卫 → E refine 调度 `start_on` → 同步契约/测试。每段过 §5 对应验证再进入下一段。
