# Handoff: CPC / sparse refine / stage1 重构（最后重构代码.md 落地中）

Date: 2026-06-15

## Current State

正在按 `CLAUDE/plans/implement/最后重构代码.md`（本会话写的实施计划）逐步实现 stage1 / sparse refine 重构。计划经 3 个 style skill 约束（code-comment-style-cn / python-writing-style-cn / yaml-config-style-cn），实现时必须严格遵守。**Step 1、Step 2 已完成并 `py_compile` 验证通过；Step 3 进行中（已读 atom head 构造，尚未动笔）。** 用户授权"实现剩余所有部分，并安排 subagent 在合适时机补测试"。

诊断文档 `CLAUDE/docs/sparse_refine_f1_regression_diagnosis_v2.md` 已按两轮 MD Human Review 评审指令改完；计划文档 `最后重构代码.md` 已按一轮评审指令改完。

## Completed

- **诊断文档 v2**：两轮评审逐条修改（§1 预算/§3.1 普遍低/§4.2 残差vs直接/§4.5 anchor 不改/§5.3 主决策点/§5.5 Tversky 0.3-0.7/§6.2 不新增日志/§6.3 损失设计/§6.4 仅调 P 规模/§6.6 schedule/§6.7 公共配置/§6.8 删除/§7 主实验+消融 等），并补充文献到 §10。
- **计划文档** `最后重构代码.md`：完整写出（7 节 + 事实锚点），并按评审改了 CPC 配置位置（`configs/experiment/CPC/`）、dataset=`MINI_emb_unet`、batch 3/global 39、density 主=both/消融=pseudo、point_clipbig 直接改、省略 `detach_real_point_feat_into_refine`。
- **Step 1 detach 三开关重构（验证通过）**：
  - `src/model/stage1_model.py`：旧 `detach_real_point_feat`/`detach_pseudo_point_feat` → `detach_real_point_feat_into_atomhead`/`detach_pseudo_point_feat_into_atomhead`/`detach_pseudo_point_feat_into_refine`（参数/字段/docstring/调用点）。refine 的 `P_point_backbone_feat` 改从 `point_feat_raw` 取出并按 `into_refine` 单独 detach（与进 atom head 解耦）。
  - `configs/model/detach_residual/max_detach.yaml`：3 开关（进 atomhead 不 detach×2、进 refine detach=true）；`sparse_refine_residual_mode: direct`（用户已手动设）。
  - `configs/model/default.yaml`：3 个中性 false 开关（§2.6 整体删除留到 Step 5）。
- **Step 2 gauss27 scatter（验证通过）**：
  - `src/model/stage1_embed_head.py`：新增 `gauss_scatter_to_voxel_grid`（27邻域各向同性高斯、σ默认0.7、到体素中心距离、逐原子归一化保质量守恒、occupancy 2 + 加权 centroid 3，通道序 [特征]++[occupancy]++[centroid]）。
  - `src/model/stage1_model.py`：`__init__` 加 `online_pdb_feature_scatter_kernel/sigma_voxel/add_occupancy/add_centroid` 4 参数 + docstring；`_build_voxel_input` online 分支加 `gauss27` 路径，legacy(soft/hard)温存。import 加 `gauss_scatter_to_voxel_grid`。
  - `configs/model/embed_head/point_clipbig.yaml`：`online_pdb_feature:true` + `scatter_kernel:gauss27` + sigma 0.7 + occupancy/centroid true（零可学习参数投影）。

## Decisions（已锁定，勿重新讨论）

- 范围(b)：P 监督在主实验开、可消融关；stage2/diffusion 不在范围；anchor sampler 不改（保 `unweighted_fps`，因 P 要被 stage2 diffusion 复用为虚拟原子）。
- 主实验 head = **`direct`**（A1 消融才 `residual`）；residual scale gate 不上。
- detach：进 atom head 不 detach（real+pseudo，受体/P 监督训练 trunk）、进 refine detach（refine 隔墙只读）。`detach_real_point_feat_into_refine` 直接省略（无消费者）。
- 深监督权重 PSPNet 惯例：atom 前/后 0.4/1.0、pseudo 前/后 0.4/1.0。
- P 监督目标 = 二值 ligand 区域归属（`target_from_ligand_dist_map` 在 P 体素取值，单通道 sigmoid）；独立 pseudo 头接在 `pseudo_feature` 上。
- sparse refine 损失：`w_focal=0`、Tversky `0.3/0.7`、+ `w_pos=w_rank=0.3`、`m_pos=0`、`m_rank≈0.5`；P_keep 用验证集 `p_best`（`_cached_voxel_ligand_p_best_by_class`）；Q_C = 全正例 × per-sample TopK 硬负例。
- 预算(CPC)：`task/binary.yaml` 改 expand 7.0→5.0（cap/warmup/max_anchors 已 25000/25000/1024）；`task/binary_less.yaml` 在 binary 基础上 expand→2.5（供 A5）；`knn_message.yaml` num_neighbors 4→8。
- schedule：start_on_ratio 0.1 / warmup_ratio 0.2（refine 三损失共用；P 损失从头开、不 warmup）。
- fusion 主=e234d4（A6 消融 d4321）；density 主=both（A7 消融 pseudo）。
- prior schema（落 `default.yaml`，总开关 `prior_prob_init_enabled:true` 默认开、被各头 residual 压制）：`prior_prob_voxel_receptor:0.1`、`prior_prob_point_receptor:0.1`、`prior_prob_voxel_ligand:0.01`、`prior_prob_point_ligand:0.1`、`prior_prob_sparse_refine:0.1`。
- embed head 投影：用 **online_pdb_feature 纯 scatter 路径**（零可学习参数），不是 embed_voxel_out_channels（那条有 input_proj+voxel_out_proj 两个 MLP 且 scatter 的是嵌入特征）。
- 实验集：8 个（1 主 + A1~A7），`configs/experiment/CPC/`，单 seed。
- 互換保全：仅需保证 `emb_unet / emb_unet_abl0 / emb_unet_abl1 / unet000 / unet_c1 / unet_c2` 行为不变（各加一行 `prior_prob_init_enabled:false`）；`old_1/old_2/old_3/other` 不必兼容。
- 验收：主实验三条（C recall 回升 / global 朝 0.63 / refined≥unrefined）；消融看末 3-5 epoch 均值超抖动。
- 配置原则：尽量直接改同栏目共享子配置，少用 experiment override。
- 工作流硬规定：写正式 plan 前必须 grill + 取得明确许可（见用户级 memory `plan-writing-workflow`）。

## Open Questions

- 无阻塞项。P front 头能否看到 pseudo 槽位：用"`pseudo_loss_front_weight>0` 而 `pseudo_logits_front` 缺失则 raise"的契约兜住，smoke run 暴露，无需手动核验。

## Next Actions

1. **Step 3**：`stage1_atom_head.py` 加 `pseudo_logit_head`(pseudo_feature→1ch) + `enable_pseudo_ligand_head/pseudo_ligand_logit_dim/prior_prob_point_ligand` 参数 + forward 出 `pseudo_logits`；`stage1_model.py` 加 `pseudo_logit_head_front` + `_run_atom_head` 出 `pseudo_logits_front`；wrapper 加 `ligand_pseudo_loss` + 前/后权重 0.4/1.0 + 缺失 raise 契约。
2. **Step 4**：`src/modules/losses.py` 新增 `LigandSparseRefineDeltaLoss`（preservation `max(0, base-refined+m_pos)` 仅 P_keep；ranking `max(0, m_rank-r_i+r_j)` Q_C=全正例×per-sample TopK 硬负例）；`voxel_point_stage1_losses.py::compute_sparse_refine_loss_term` 扩成 `L_cls + w_pos·L_pos + w_rank·L_rank`，base 用 `candidate_logits/candidate_prob`、p_best 用 `_cached_voxel_ligand_p_best_by_class`、分组 `candidate_batch_index`。
3. **Step 5**：prior schema 5 字段 + 总开关落 `default.yaml`，统一 direct sparse refine 头 / pseudo 头 / voxel 头先验初始化（复用现有 `_init_atom_logit_head_prior_bias` / `_init_linear_multiclass_prior_bias`，residual 模式零初始化跳过）；§2.6 删 default.yaml 31-49 行（注意 `enable_atom_head_front/back` 等需 __init__ 有匹配默认，逐项核对，避免破坏 6 个互換 config）。
4. **配置层**：改 `binary.yaml`(expand 5.0)、`binary_less.yaml`(expand 2.5)、`knn_message.yaml`(num_neighbors 8)、`sparse_refine.yaml`(w_focal 0/Tversky 0.3-0.7/atom 0.4-1.0/+ligand_pseudo_loss/+delta 权重/+pseudo 权重)；新建 8 个 `configs/experiment/CPC/*.yaml`（参照 `other/detach_main.yaml` 结构，dataset MINI_emb_unet、batch 3/global 39、embed_head point_clipbig、density both、fusion e234d4、enable_pseudo_ligand_head true、prior schema 值）；6 个互換 config 各加 `prior_prob_init_enabled:false`。
5. **契约同步**：`stage1_atom_head.py` 顶部契约段（已部分改）、`_run_atom_head`、`CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md`、`tests/test_stage1_atom_head.py` / `tests/test_ligand_sparse_refine_loss.py`。
6. **subagent 补测试**：gauss27 scatter（质量守恒/各向同性/空输入/occupancy+centroid 维度）、detach 三开关路由、pseudo 头 forward、delta loss（preservation/ranking 数值与边界）、prior 初始化（direct→先验 / residual→零初始化）；并 `py_compile` + 跑现有测试确认不回归。

## Files To Reopen

- `CLAUDE/plans/implement/最后重构代码.md`（实施计划，单一事实源）
- `CLAUDE/docs/sparse_refine_f1_regression_diagnosis_v2.md`（诊断与决策依据）
- `src/model/stage1_model.py`（detach 路由、_build_voxel_input、_run_atom_head、前置头、__init__ 参数）
- `src/model/stage1_atom_head.py`（pseudo_feature_head 旁加 pseudo_logit_head；构造约 356-396、forward 约 498-520）
- `src/model/stage1_embed_head.py`（已加 gauss_scatter_to_voxel_grid）
- `src/modules/losses.py`（新增 LigandSparseRefineDeltaLoss）
- `src/wrappers/voxel_point_stage1.py` / `voxel_point_stage1_losses.py`（loss 组合、p_best 缓存）
- `configs/model/task/binary.yaml` / `binary_less.yaml`、`configs/model/sparse_refine/anchor_to_candidate/knn_message.yaml`、`configs/loss/sparse_refine.yaml`、`configs/model/default.yaml`、`configs/experiment/other/detach_main.yaml`（CPC 模板参照）
