# Handoff: CPC / sparse refine / stage1 重构 —— 端到端实现完成

Date: 2026-06-16

## Current State

`CLAUDE/plans/implement/最后重构代码.md` 的全部剩余工作(Step 3~5 + 配置层 + 契约同步 + 测试)已端到端实现并本地验证。严格遵守 code-comment-style-cn / python-writing-style-cn / yaml-config-style-cn。**未 git commit**(等用户指示)。

本地验证(Windows, `D:/Anaconda/envs/baseline_env/python.exe`, 有 torch2.5.1+hydra+yaml, 无 torch_cluster/PTV3):
- 全部改动源码 `py_compile` 通过。
- Hydra compose 8 个 CPC + 6 互換 config 全部成功, 关键值正确(CPC_main: direct/pseudo on/prior_init true/expand 5.0/w_pos 0.3/pseudo_back 1.0; A1 residual; A5 expand 2.5; A6 fusion d4321; A2/A3/A4 各自轴; emb_unet/unet_c1 prior_init False)。
- pytest: **134 passed**。新增 16 个纯 torch 用例全绿。失败/报错仅剩 env 限制(12× test_stage1_atom_head 需 PTV3 Block、15 collection error 需 torch_cluster、1× 无关 inference postprocess)——这些在 Linux 服务器有依赖时运行。

## Completed(本会话)

- **Step 3 P 监督头 + ligand_pseudo_loss**:
  - `stage1_atom_head.py`: 后置 `pseudo_logit_head`(pseudo_feature→1ch) + 参数 `enable_pseudo_ligand_head/pseudo_ligand_logit_dim/prior_prob_point_ligand` + forward 出 `pseudo_logits`; 顶部契约同步。
  - `stage1_model.py`: 前置 `pseudo_logit_head_front`(point_feat_raw 的 P 槽位→1ch); `_run_atom_head` 出 `pseudo_logits_front` + 暴露 `pseudo_voxel_zyx`(x,y,z floor 重排为 z,y,x)/`pseudo_batch_index`; 顶部契约 + forward 输出契约同步。
  - `voxel_point_stage1.py`: `_sample_ligand_pseudo_supervision`(dense ligand 在 P home 体素 gather + voxel_valid_mask gate) + loss 调用(前置缺失 raise); `voxel_point_stage1_losses.py`: `compute_pseudo_loss_term`。
- **Step 4 LigandSparseRefineDeltaLoss**:
  - `src/modules/losses.py` 末尾新增 `LigandSparseRefineDeltaLoss`: preservation 仅 P_keep(valid 且正且 base_prob>p_best) `mean(max(0,base-refined+m_pos))`; ranking 对称难例挖掘(每 BOX base_prob 最低 Kp 正例 × 最高 Kn 负例, `mean(max(0,m_rank-r_pos+r_neg))`); 选择口径统一 base_prob; 空集合安全返回 0。
  - `compute_sparse_refine_loss_term` 扩成 `L_cls + w_pos·L_pos + w_rank·L_rank` 单一组合点(返回 3-tuple: term, eff_weight, component_logs); wrapper 传 base=candidate_logits/candidate_prob、batch_index=candidate_batch_index、p_best=cache[0]、w_pos/w_rank; delta_on 守卫(cache 非空且权重>0)。
  - 现有 `tests/test_ligand_sparse_refine_loss.py` 4 处调用更新为 3-tuple(契约同步)。
- **Step 5 prior schema**(模块侧 + default.yaml):
  - `stage1_model.__init__` 加 `prior_prob_init_enabled` 主开关 + 5 命名先验 + `enable_pseudo_ligand_head/pseudo_ligand_logit_dim`; 计算 `_eff_*`(主开关关→全 None); atom 头按 `atom_logit_dim` 多/单通道路由(多通道→prior_probs、单通道→命名 point_receptor 回退 legacy prior_prob); voxel backbone 注入 `prior_prob_voxel_receptor/ligand`; refine head 注入 `prior_prob_sparse_refine`。
  - `stage1_voxel_backbone.py`: aux/ligand 先验拆分(各自独立, 多通道 prior_probs 优先; 报错信息保留"多通道 softmax head" 子串以兼容现有测试)。
  - `sparse_refine_head.py`: direct 模式末层 bias=logit(prior_prob); residual 保持零初始化跳过。
  - `default.yaml`: 替换旧 prior_prob/prior_probs 为新 prior schema(主开关 true + 5 命名先验); 删除 31-49 行 detach/头/refine 中性默认(改为由 detach_residual 组 + __init__ 默认决定)。
- **配置层**: `task/binary.yaml` expand 7→5; `task/binary_less.yaml` 对齐主预算仅 expand=2.5(供 A5); `knn_message.yaml` num_neighbors 4→8; `loss/sparse_refine.yaml`(refine L_cls 改 w_focal0/Tversky0.3-0.7、atom 0.4/1.0、+delta 模块+w_pos/w_rank 0.3、+ligand_pseudo_loss、+pseudo 0.4/1.0、schedule start_on_ratio 0.1); 新建 8 个 `configs/experiment/CPC/CPC_{main,A1_residual,A2_no_pseudo,A3_no_delta,A4_no_trunk_shaping,A5_budget,A6_fusion,A7_density}.yaml`; 6 互換 config 各加 `prior_prob_init_enabled: false`。
- **契约同步**: `tri_ligand_sparse_refine/00-master.md` forward/atom-head 输出表加 pseudo_logits/_front/voxel_zyx/batch_index + P 监督与 detach 三开关说明。
- **测试(本人写并本地跑通)**: `tests/test_ligand_sparse_refine_delta_loss.py`(6)、`tests/model/test_gauss_scatter.py`(5)、扩充 `tests/test_multiclass_voxel_backbone.py`(+4 prior 拆分)、扩充 `tests/model/test_stage1_atom_head.py`(+2 pseudo 头, 仅服务器可跑)。

## Decisions(本会话新锁定, 覆盖计划模糊处)

- prior 机制: **主开关 `prior_prob_init_enabled`**, false=全跳过; 6 互換 config 各加一行 false(用户接受 atom 头不再 logit(0.01), 仅 init 差异)。
- `ligand_pseudo_loss` = 召回偏置纯 Tversky(w_focal0/Tversky0.3-0.7), 与 refine L_cls 同源(非 atom focal 风格)。
- `L_rank` 改**对称难例挖掘** Kp=Kn=**512**(用户从 256 上调), m_rank0.5。
- §2.6 "detach_residual 必选" **不执行**: 会破坏 6 互換 config(它们不选该组); 已核对 __init__ 中性默认与旧 default.yaml 逐项一致(仅把 `detach_voxel_feat_into_pseudo_point` __init__ 默认 True→False 对齐), 删除中性默认后回退 __init__ 默认即可, 无"未定义报错"。
- 续航/watchdog: 不做(用户撤销)。
- 提问用聊天文本不用弹窗(已存 user memory `ask-in-chat-not-popups`)。

## Open / Next

- **未 commit**: 等用户指示再提交/push。
- **服务器测试**: PTV3/torch_cluster 相关用例(test_stage1_atom_head 含新增 pseudo 头、test_sparse_refine_head 的 direct prior、test_stage1_model 的 detach 路由)需在 Linux 服务器跑全。建议 smoke run 一个 CPC_main 前向/反向(前置 pseudo 契约、lazy in_channels、通道数会在此暴露)。
- **subagent**: 本会话派的"代码理论审查 + 测试撰写"两个 subagent 撞额度上限(resets 15:10 Asia/Shanghai)未执行; 已由主会话自行完成等价工作。如需更深审查/更多服务器端用例, 额度恢复后可重派。

## Files To Reopen

- `CLAUDE/plans/implement/最后重构代码.md`(单一事实源)
- `src/model/stage1_model.py` / `stage1_atom_head.py` / `stage1_voxel_backbone.py` / `sparse_refine/sparse_refine_head.py` / `stage1_embed_head.py`(gauss27, 上会话)
- `src/modules/losses.py`(LigandSparseRefineDeltaLoss) / `src/wrappers/voxel_point_stage1.py` / `voxel_point_stage1_losses.py`
- `configs/model/default.yaml` / `configs/loss/sparse_refine.yaml` / `configs/experiment/CPC/*.yaml`
