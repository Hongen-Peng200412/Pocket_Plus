# Stage1 梯度路由 / 前后二分类头 / 接口 Norm / density 残差 实施计划

> 本文是实现计划子文档。阅读前请先回根目录 `CLAUDE.md` 看总指针与权威层级；遇到字段/shape/路径冲突，以**实际代码 > Hydra/checkpoint 配置 > 真实产物 > 本文**为准。

## 0. 背景与动机

已验证实验（出发点）：

1. 纯 UNet（体素分支）预测结合体素或 ligand 区域都收敛良好。
2. UNet+embed head，以及 UNet+embed head+point backbone+atom head 同时预测受体（体素/原子）与配体，结果都优于 1。
3. 一旦启用 `configs/model/sparse_refine` 让 sparse refine head 对 voxel 分支的 ligand 预测做 refine，**所有 loss 初期收敛、随后随 refine loss 权重提高全部发散**。

工作假设：发散源于 **refine loss 经多条路径回流污染 voxel/point backbone（负迁移）**。本计划把"哪条梯度回流、回流到哪个 backbone"做成**显式可配置可解释**的开关，并补上稳定化措施（接口 Norm、把原子二分类做成对 voxel 预测的残差 refine、density cube 直通残差）。

不引入 GradNorm（已确认删除）：`.detach()` 是梯度路由问题、GradNorm 是多任务 loss 再加权问题，二者不可在单接口互换，复杂度也不符合 minimal-code 目标。

## 1. refine loss 当前回流路径（实现前必须心里有数）

`src/model/stage1_model.py::_run_sparse_refine_head` 当前把 refine loss 回流到 backbone：

- → **voxel backbone**：`voxel_logits_C`（residual base）、`C_voxel_backbone_feat`、`P_voxel_backbone_feat`（从 `voxel_final` / `voxel_logits_ligand` gather）。当前只有 logits 一处可 detach（`SparseRefineHead.detach_voxel_logits`）。
- → **point backbone**：`P_point_backbone_feat`（直接抽 `outputs["fused_point_feat"]` 的 pseudo 槽位）与 `P_atom_head_feat`（穿过 atom head 回到 point backbone pseudo 槽位）。
- → **voxel backbone（间接二次回流）**：point backbone 在 `_fuse_point_variable` 内 `grid_sample` 吸收 voxel 特征，故 `refine → point → voxel-fusion` 是活路径。

## 2. 开关总览

> **本轮新增（refine loss 调度，独立于下表开关）**：给 `ligand_sparse_refine_loss_schedule` 增加 `start_on_ratio`（及对称的绝对版 `start_on_steps`）。语义——`global_step < start_on` 时 refine 权重**硬等于 0**（refine 块完全不训练、真正推迟开始）；随后在 `[start_on, warmup]` 区间从 `start_weight`(=0) 线性升到 `final_weight`；`≥ warmup` 取 `final_weight`。默认 `null` 完全复现旧行为。动机：在"最大 detach"下三块（voxel / point+front / refine）解耦、Adam 对单一 loss 的整体缩放不变，原 warmup 从第 0 步起的微小权重等价于"没推迟 refine"；只有一段硬 0 才能真正延后 refine 开始。要求 `start_on ≤ warmup`，否则 fail-fast。主实验取 `(start_on_ratio, warmup_ratio)=(0.1, 0.2)`，消融取 `(0.2, 0.3)`。详细实现见 `00-imple-plan.md`。

所有 knob 作为**扁平字段**挂在 `cfg.model.backbone.*`，透传进 `VolumePointStage1Model.__init__`。下表"全局默认"是 `configs/model/default.yaml` 的取值，刻意保持中性以复现旧行为、不影响任何现有配置；主实验对它们的覆盖见 §2.1。

| 开关 | 类型 | 全局默认 | 语义 |
| --- | --- | --- | --- |
| `detach_real_point_feat` | bool | False | detach point backbone 输出的**真实原子**槽位，再喂给（后置）atom head。前置头取 detach 之前的特征，不受影响。 |
| `detach_pseudo_point_feat` | bool | False | detach point backbone 输出的 **pseudo(P)** 槽位，统一作用于"进 atom head"与"直接 `P_point_backbone_feat` 进 refine"两处。 |
| `detach_voxel_feat_into_point` | bool | False | 全局：`_fuse_point_variable` 里把采样到的 `sampled_voxel_feat` detach 后再融合进 point。 |
| `detach_voxel_into_refine` | bool | True | 统一覆盖 refine 吃的三处 voxel 信息：base logits、`C_voxel_backbone_feat`、`P_voxel_backbone_feat`。**取代并删除**旧 `SparseRefineHead.detach_voxel_logits`。 |
| `enable_atom_head_front` | bool | False | 在 point backbone 末端（真实原子）增设二分类"前置头"。 |
| `enable_atom_head_back` | bool | True | 是否启用 atom head 末端的 `real_atom_logit_head`（"后置头"）。atom head 模块本体在需要 `pseudo_feature` 时仍构造。 |
| `refine_receptor_from_voxel` | bool | False | 前/后置头 logits 都 `+= voxel_logits_aux` 在该原子 home 体素处的 `.detach()`；同一 bool 同时作用于前/后置头。 |
| `pseudo_density_residual` | bool | False | 在 atom head 伪原子分支输出上加 density cube 直通残差：`pseudo_feature = pseudo_feature_head(...) + density_residual(density_cube_feat)`；**只作用于 `P_atom_head_feat` 这一条进 refine 的路**。 |
| `enable_interface_norm` | bool | False | 单开关，在若干 channel-last 融合接口"先各自 LayerNorm 再拼接"。 |

Loss 权重（`configs/loss/*` → `cfg.model.*`）：

| 字段 | 语义 |
| --- | --- |
| `atom_loss_front_weight` | 前置头 atom loss 权重 |
| `atom_loss_back_weight` | 后置头 atom loss 权重 |

> 旧 `atom_loss_weight` **直接删除不兼容**。前/后置头**共用同一个 `atom_loss` 模块**，各出一个 loss term（后置=`atom`，前置=`atom_front`）。

### 2.1 主实验取值（最大隔离起点）

原基线 exp3 已发散，主实验不追求"复现 exp3"，而是从"最大隔离"切入：让 voxel、point+前置头、refine 三块尽量互不干扰。主实验对 §2 各开关的覆盖取值为

`enable_atom_head_front=True, enable_atom_head_back=False, detach_real_point_feat=True, detach_pseudo_point_feat=True, detach_voxel_feat_into_point=True, detach_voxel_into_refine=True, refine_receptor_from_voxel=True, pseudo_density_residual=False, enable_interface_norm=True`，sparse candidate 取 `binary_adaptive`，全二分类（`aux=atom=ligand=1`）。

此取值下 voxel backbone 只由自身两个体素 loss 训练（refine 与 point 分支的梯度都被 detach 挡在外面，含 `detach_voxel_feat_into_point=True` 切掉的 point→voxel 融合回流）；point backbone、density cube 与前置头只由前置原子 loss 训练（density cube 借真实–伪原子注意力拿到梯度，`pseudo_density_residual` 本轮关闭）；atom head 的伪分支与 sparse refine head 只由 refine loss 训练。三块解耦之后，除体素分支内部 `voxel_aux : voxel_ligand` 之外的跨块相对 loss 权重基本失效，因此 refine 的时间控制改由 `start_on_ratio` 的硬 0 段承担（见上文）。8 个消融在此基础上扫 detach×loss 网格，详见 `00-imple-plan.md`。

## 3. 组合说明（按需求方要求**不设 fail-fast、不硬编码矩阵，由实验者自负**）

detach 轴 =（`detach_real_point_feat`, `detach_pseudo_point_feat`）两独立 bool；loss 轴 =（`enable_atom_head_front`, `enable_atom_head_back`）两独立 bool。代码层面不做任何排除断言。

**仅作记录的退化角**（不拦截）：`前置头关 且 后置头开 且 detach_real=True 且 detach_pseudo=True 且 pseudo_density_residual=False`。此时 point backbone 与 density cube 都拿不到监督梯度（成为 unused 参数），在 `ddp_find_unused_parameters=false` 下会 DDP 报错。默认不落在此角；若要跑它，临时把 `ddp_find_unused_parameters` 置 `true`（见 §6）。

## 4. 各模块梯度来源（写进 docstring，便于 sweep 解释）

**关键认知**：mixed point backbone 中真实原子与伪原子是同一片点云，PTV3 序列化注意力 + pointconv CPE 让空间相邻的 real/pseudo 互相 attend。因此存在一条**输入侧**回流：

```
前置/后置 atom loss → point_feat[real] →（point backbone 内 real↔pseudo 注意力）→ pseudo 输入特征 → density_cube
```

`detach_real/detach_pseudo` 施加在 point backbone **输出**，挡不住这条经由**输入**的回流。

各模块梯度来源汇总：

| 模块 | 梯度来源（默认配置下是否有） |
| --- | --- |
| voxel backbone | 自身 receptor/ligand voxel loss（恒有）；`detach_voxel_into_refine=False` 再加 refine 直接路径；`detach_voxel_feat_into_point=False` 再加 atom/refine 经 point-fusion 的二次回流 |
| point backbone | 前置 loss（前置头开，恒在 detach 之前）；后置 loss（后置头开且 `detach_real=False`）；refine（`detach_pseudo=False`） |
| density cube | 前置/后置 loss 经注意力（对应头开、对应 detach 不阻断真实原子输出梯度时）；refine 经 `pseudo_density_residual`（残差开）；refine 经 point backbone pseudo（`detach_pseudo=False`） |
| atom head | 后置 loss（后置头开）；refine（恒有，atom head 自身参数即便输入被 detach 也被 refine 训练） |
| sparse refine head | refine（恒有） |
| 前置头 | 前置 loss |
| 后置头 | 后置 loss |

refine 对两个 backbone 的隔离程度，由 `detach_voxel_into_refine`（→voxel 直接）、`detach_pseudo`（→point，及经 point 的→voxel 间接）、`detach_voxel_feat_into_point`（→voxel 经 point-fusion）三者共同决定，是一条从"协同训练"到"完全隔离"的光谱。

## 5. 分阶段实现

每个 Stage 结尾给验证关卡；建议 Stage 间各跑小 batch forward smoke test + 相关单测。所有改动均为现有 forward/`__init__` 内插钩子 + 复用现有结构，不新建大模块、不重构 wrapper。逐文件精确改动见 `00-imple-plan.md`。

- **Stage A 梯度路由开关**：`stage1_model` 新增 detach 路由纯函数与四个 detach 开关；`_fuse_point_variable` 加 `detach_voxel_feat_into_point`；`_run_sparse_refine_head` 用 `detach_voxel_into_refine` 统一三处 voxel detach；删除 `SparseRefineHead.detach_voxel_logits` 与其配置行。
- **Stage B 前/后置头 + refine_receptor + density 残差**：后置头用 `enable_atom_head_back` 门控、关时 `atom_logits=None`；新增前置头浅 MLP；`refine_receptor_from_voxel` 用 home 体素 `voxel_logits_aux.detach()` 作残差 base，开启时末层零初始化并跳过 prior bias，且要求 `atom_logit_dim==voxel_aux_logit_dim`；`pseudo_density_residual` 在 atom head 伪分支输出上加 `LayerNorm→Linear`（末层 Linear 零初始化）残差，仅作用于 `P_atom_head_feat`。
- **Stage C 接口 Norm**：`enable_interface_norm` 在 density cube 输出、voxel→point 融合两源、sparse refine 的 P/edge 学习特征源处"先各自 LayerNorm 再拼接"；几何量与 embedding 不归一化；末层 head 不归一化以保 `base_logits` 残差语义。
- **Stage D 配置 + wrapper**：`default.yaml` 写中性默认；`sparse_refine.yaml` 改 atom 权重为前/后置并加 `start_on_*`；删 `sparse_refine_head` 配置里的 `detach_voxel_logits`；wrapper 拆出前/后置 atom loss term 与 metric 分支；新建 1 主实验 + 8 消融实验配置。
- **Stage E refine 调度 + 契约/测试**：refine 权重调度加 `start_on` 硬 0 段；同步契约 docstring 与测试。

## 6. DDP `find_unused_parameters` 说明（已查阅现状）

- 现状：`configs/train/B40_L3.yaml`、`baseline配置.yaml` 均为 `ddp_find_unused_parameters: false` → **任何被构造但无梯度的参数都会让 DDP 报错**。
- density cube：`DensityCubeEncoder` 有可训练参数（`encoder` 卷积栈含可能的 `LazyConv3d` + `proj` Linear）；`src/train.py::_initialize_lazy_modules_before_ddp` 在 DDP 包裹前 materialize lazy 参数。
- **结论**：主实验下 density cube **不会冻结**——前置 atom loss 经 point backbone 的 real↔pseudo 注意力回流到 pseudo 输入特征即可训练它（§4）。9 个配置（主实验 + 8 消融）经核都没有任何 backbone/density-cube 成 unused，故 `find_unused_parameters=false` 全部安全。
- 代价（若确需开 true）：**显存**基本可忽略（仅多一张参数使用位图 + 反向前一次自动求导图遍历，不增 GPU 激活显存）；**速度**每步多一次全图遍历标记 unused，典型开销 **~3–10%**（图越大越明显）。
- 策略：**默认保持 false**；仅当跑 §3 那个退化角时临时置 true。

## 7. 对齐契约同步点

随实现一并更新以下 docstring/文档：

- `src/model/stage1_model.py` 顶部契约块、`forward`/`_run_atom_head`/`_run_sparse_refine_head`/`_fuse_point_variable`（新增 `point_feat_raw`、detach 路由、前置头、refine_receptor、density 残差、各开关语义）。
- `src/model/stage1_atom_head.py` 顶部契约块（后置头可关、`atom_logits` 可为 None、新增 `pseudo_density_feat` 入参与残差输出语义）。
- `src/model/sparse_refine/sparse_refine_head.py`（删除 `detach_voxel_logits`，detach 由调用方负责）。
- `CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md`（若其契约表引用 `detach_voxel_logits` 或 atom head 单一 logits 头，需同步）。
- 对应 `tests/model/test_stage1_model.py`、`test_stage1_atom_head.py`、sparse refine 相关 test 的契约注释。

## 8. 测试计划（no-regression + 细粒度）

1. **回归**：四个 voxel-only 配置 + 中性默认 → forward 通过、loss/metric 键与改动前一致。
2. **路由 autograd 断言**：构造带 sparse refine 的小 batch，遍历消融 8 组（+ `detach_voxel_*` 开关），用 `torch.autograd.grad` 断言各 loss 是否触达 voxel/point/density-cube 参数，与 §4 一致；特别验证"最大 detach"下三块解耦、且 density cube 经前置 loss 注意力**有**梯度。
3. **头开关**：`enable_atom_head_back=false` → `atom_logits is None`、sparse refine 正常、validation 不崩；前置头开/关产出 `atom_logits_front` 与 `atom_front` loss/metric。
4. **refine_receptor**：零初始化下前/后置头初始 logits == `voxel_aux` home 体素 detach 值；维度不匹配 fail-fast；base 无梯度回流 voxel。
5. **density 残差**：`pseudo_density_residual=true` 时 `P_atom_head_feat` 含残差、`P_point_backbone_feat`/前置 loss 数值不变；末层零初始化时初值=无残差。
6. **Norm**：`enable_interface_norm` 开/关 forward 均通过；关时与不加 Norm 实现一致。
7. **refine 调度**：`(start_on_ratio, warmup_ratio)=(0.1,0.2)` 下若干 step 返回 0 / 斜坡 / final 三段正确；`start_on>warmup` 抛错；`start_on_ratio=null` 复现旧斜坡。

## 9. 风险与注意

- **退化角 DDP**：§3 那个组合会使 point/density 成 unused → `find_unused_parameters=false` 下崩；不设断言，由实验者负责，必要时临时置 true。
- **`detach_voxel_into_refine` 默认 True** 有意偏离已发散的 exp3。
- **空 P / 空 batch**：所有新 LayerNorm、detach、gather、density 残差在 `sumP==0`/`N==0` 时走安全分支（sparse refine head 已有 `numel()==0` 处理，新代码须对齐）。
- **prior bias 与 zero-init 冲突**：`refine_receptor_from_voxel` 开启时跳过 `prior_prob/prior_probs` bias 初始化。
- **`binary_adaptive` 运行前提**：阈值缓存填充前依赖 scheduler warmup 的 fixed-topk；`B40_L3` 的 `scheduler.name=warmup_plateau` 已满足，且 warmup 窗口内会发生 validation 来填缓存（主实验 `val_per_epoch=5`、消融 `=1` 均满足）。
- **minimal-code**：全部为现有 forward/`__init__` 内插钩子与复用，不新建大模块。

## 10. 执行顺序

A 路由开关（含删 `detach_voxel_logits`）→ B 前/后置头 + `refine_receptor` + density 残差 → C 接口 Norm → D 配置 + wrapper → E refine 调度 `start_on` → 同步契约/测试。每段过 §8 对应验证再进入下一段。
