# Stage1 网络阅读说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是训练/模型链路中的子说明。

本文只做当前 Stage1 网络、wrapper 和验证诊断的阅读路径与检查点，不替代代码阅读。字段、shape、路径、类别语义和 mask 语义以实际代码、Hydra 配置、checkpoint 中保存的配置和真实 batch 为准。如果本文与实现冲突，把冲突当作潜在 bug 或待更新文档处理。

## 1. 推荐阅读顺序

建议先从运行入口和配置确定当前实验实际启用了哪些分支，再进入模型实现：

1. `src/train.py`
   - 看 Hydra 如何实例化 DataModule、`VoxelPointStage1Wrapper`、Trainer、logger、checkpoint。
2. `configs/model/default.yaml` 与当前 `configs/experiment/*.yaml`
   - 看 wrapper、backbone、loss、optimizer/scheduler、`class_names`、`validation_diagnostics` 和 `monitor_metric` 的实际配置。
3. `configs/dataset/stage1_find.yaml` 或 `configs/dataset/stage1_unet_c1.yaml`
   - 看 A—G 整图资产根目录、冻结 BOX 请求目录、密度通道、请求抽样和增强配置。
4. `src/datasets/stage1_dataset.py`
   - 看 `Stage1Dataset` 如何读取 A—G 整图资产与冻结 BOX 请求，并物化一个 80³ 模型样本。
5. `src/datasets/stage1_collate.py`
   - 看固定 80³ 体素字段如何堆叠，以及不同 BOX 的受体原子如何通过 `atom_counts`、`atom_offsets` 和 `atom_batch_index` 拼成批次。
6. `src/wrappers/voxel_point_stage1.py`
   - 当前正式 LightningModule。先读 `training_step()`、`validation_step()`、`on_validation_epoch_end()`，按运行时间顺序理解 forward、loss、metric、CPC diagnostics、candidate threshold cache。
7. `src/wrappers/voxel_point_stage1_losses.py`
   - 看 atom / receptor / dense ligand / sparse refine 各 loss term 的输入字段。
8. `src/wrappers/voxel_point_stage1_metrics.py`
   - 看常规 AP/PRAUC 分支、二分类/多分类 key 规则和 metric 设备策略。
9. `src/wrappers/voxel_point_stage1_diagnostics.py`
   - 看 dense -> C -> refined 的 CPC 验证统计、global-only 指标和 DDP all-reduce 约束。
10. `src/model/stage1_model.py`
    - 看 `VolumePointStage1Model.forward()` 如何串起 embed head、voxel backbone、point backbone、pseudo atoms 和 sparse refine。
11. `src/model/stage1_embed_head.py`
    - 看真实 atom 如何进入预编码和可选 voxel scatter/embed grid。
12. `src/model/stage1_voxel_backbone.py`
    - 看 dense voxel 分支输入输出，尤其 `voxel_logits_aux` 与 `voxel_logits_ligand`。
13. `src/model/stage1_point_backbone.py`
    - 看真实点 / pseudo 点混合后的 point backbone 输出。
14. `src/model/pseudo_atoms.py`
    - 看 mixed point layout、pseudo atom 注入、real/pseudo 输出拆分。
15. `src/model/sparse_refine/`
    - 看候选 C 生成、P anchor 采样、density cube 编码、P-to-C 邻接和 sparse refine head。

`src/wrappers/voxel_point_stage1_old.py` 只作为旧行为对照，不是当前正式入口。不要按旧 wrapper 私有 metric API 设计新调用。

## 2. 当前 wrapper 职责边界

`VoxelPointStage1Wrapper` 现在是 thin coordinator：它保留 Lightning 生命周期和训练/验证时间顺序，大块逻辑拆到同目录 helper。

主要职责：

- 实例化或接收 backbone 与 loss module。
- 显式接收 `class_names`，未传入时 fail-fast。
- 从 backbone 读取 sparse candidate class ids，并校验 sparse refine loss 与 candidate builder 是否匹配。
- 维护 candidate threshold runtime cache：`p_best_by_class`、`p_sampling_by_class`、`best_f1_before_refine_by_class`。
- 在训练和验证前同步 candidate runtime/cache 到 backbone。
- 汇总 loss term 并记录 `train_loss/*`、`val_loss/*`。
- 将常规 AP/PRAUC 交给 `ValidationMetricManager`。
- 将 dense -> C -> refined CPC 诊断交给 `CpcValidationDiagnostics`。
- 在 validation epoch end 写回 candidate threshold cache、记录 scalar、按 rank0 写 artifact / W&B curves、推进 warmup plateau scheduler。
- checkpoint 中保留训练继续所需 candidate cache 和 warmup plateau state；validation metric/diagnostics 中间状态不应进入 checkpoint。

对外日志使用 `receptor` 表达 receptor/auxiliary voxel 分支；内部模型 key 仍可能保留 `voxel_logits_aux`、`voxel_aux_head`、`voxel_aux_logit_dim`，不要为了日志名改动 checkpoint 兼容 key。

## 3. Stage1 forward 主链路

当前 `VolumePointStage1Model` 的核心时间顺序可以按下面读：

1. `_run_embed_head_once(batch)`
   - 只处理真实 atom，不让 P anchor/pseudo atom 进入 embed head。
   - 可能裁剪 buffer atom，并同步更新 atom feature、坐标、offset、count、label、mask。
   - 输出可选 `embed_point_feat` 和 `voxel_pdb_embed_grid`。
2. `_build_voxel_input(batch, embed_output)`
   - 以 `batch["voxel_grid"]` 为基础构造 dense voxel 输入。
   - 如果 embed head 输出 `voxel_pdb_embed_grid`，在 channel 维拼接。
   - 否则在启用 online PDB feature 时，把 atom feature scatter 到 voxel grid。
3. recycle loop
   - 每轮先跑 voxel backbone。
   - 非最终轮通常跑 real-only point backbone，并更新或 detach recycle state。
   - 最终轮才准备 pseudo batch。
4. `_prepare_pseudo_batch(...)`
   - 如果没有 candidate builder，返回 real-only point batch。
   - 如果启用 candidate builder，从 `voxel_logits_ligand` 和 `voxel_valid_mask` 生成候选 C。
   - 如果启用 anchor sampler，从 C 中采样 P anchor，抽取 density cube，编码为 `pseudo_feat`，并通过 `inject_pseudo_atoms()` 拼成 mixed point batch。
5. `_run_point_backbone(...)`
   - real-only 或 mixed `[real_i..., pseudo_i...]` point batch 进入 point backbone。
   - mixed 模式下，real-only recycle state 会扩展出 pseudo slot。
6. `_run_atom_head(...)`
   - atom head 在最终 point feature 上输出 atom 监督字段。
   - `atom_logits`、`atom_target`、`atom_valid_mask` 面向真实 atom；`atom_tokens` / `atom_hidden` 可能仍保留 mixed 信息。
   - 如果存在 P anchor，会抽取 pseudo feature 供 sparse refine 使用。
7. `_run_sparse_refine_head(...)`
   - 可选 sparse refine 分支。
   - 需要 P anchor、atom head 的 `pseudo_feature`、`anchor_to_candidate` 和 sparse refine head。
   - 汇集 C/P 位置的 voxel feature，按 P-to-C 邻接聚合 P 消息，输出 `ligand_refine_logits_C`。

## 4. 关键输入字段和 shape

常见 batch 字段包括：

| 字段 | 常见 shape | 说明 |
| --- | --- | --- |
| `density_input` | `(B, C_density, 80, 80, 80)` | `Stage1Dataset` 返回的密度通道；模型入口把它转换为内部名称 `voxel_grid` |
| `hardmask` | `(B, 80, 80, 80)` | 当前 BOX 内受体原子的 home voxel 掩码 |
| `voxel_label` | `(B, 80, 80, 80)` | 当前 BOX 内结合受体原子的 home voxel 标签 |
| `ligand_area_target` | `(B, 80, 80, 80)` | 完整配体区域并集在当前 BOX 中的裁剪 |
| `protein_mainchain_target` | `(B, C_protein, 80, 80, 80)` | Find_1 与 unet_c1 使用的蛋白主链原子类别标签 |
| `nucleic_mainchain_target` | `(B, C_nucleic, 80, 80, 80)` | Find_1 与 unet_c1 使用的核酸主链原子类别标签 |
| `ligand_inverse_distance_target` | `(B, 80, 80, 80)` | `1 / (1 + distance_Å)` 配体距离回归目标 |
| `box_shape_zyx` | `(B, 3)` | 每个 BOX 的 z/y/x 空间形状 |
| `box_origin_world` | `(B, 3)` | BOX 原点世界坐标，常按 x/y/z 解释 |
| `voxel_size_world` | `(B, 3)` | voxel 物理尺寸，常按 x/y/z 解释 |
| `pdb_id` / `request_role` | `list[str]`, 长度 B | 每个 BOX 对应的 PDB 编号与冻结请求职责 |
| `atom_feat` | `(N, F_atom)` | atom 输入特征，mixed 后可含 pseudo slot |
| `atom_coord_centered_world` | `(N, 3)` | 中心化世界坐标，通常 x/y/z |
| `atom_coord_local_voxel` | `(N, 3)` | 连续 local voxel 坐标，通常 x/y/z |
| `atom_coord_world` | `(N, 3)` | 世界坐标，通常 x/y/z |
| `atom_batch_index` | `(N,)` | atom 所属 BOX index |
| `atom_offsets` | `(B,)` | batch 内 atom 累计 end offset |
| `atom_counts` | `(B,)` | 每个 BOX 的 atom 数 |
| `atom_label` | `(N_real,)` | 真实 atom 监督标签；pseudo slot 不参与监督 |
| `atom_is_in_core_box` | `(N_real,)` | 每个受体原子是否位于 80³ 核心 BOX 内；其余原子来自外扩缓冲区 |

涉及字段 shape 时，不要只看本文。优先读 dataset/collate，然后抽样真实 batch。

## 5. 坐标、索引和 mixed point 约定

当前代码里需要特别区分：

- 稠密体素输入在 Dataset 外部使用 `density_input`，形状为 `(B, C, D, H, W)`；模型内部把同一张量命名为 `voxel_grid`。
- 稠密标签和掩码按 `(B, D, H, W)` 读取；蛋白与核酸多分类标签额外带类别维。
- voxel 整数索引通常使用 `zyx`，例如 `candidate_voxel_zyx`、`anchor_voxel_zyx`。
- 世界坐标和点坐标通常使用 `xyz`。
- local voxel 连续坐标通常使用 `xyz`，P anchor 的 local voxel 坐标按 voxel center 表示，即 `(x+0.5, y+0.5, z+0.5)`。
- mixed point layout 必须按每个 BOX 分块保持 `[real_i..., pseudo_i...]`。
- mixed batch 中常见辅助字段包括 `real_mask`、`pseudo_mask`、`pseudo_anchor_class`、`pseudo_anchor_voxel_zyx`、`pseudo_source_candidate_index`。
- 从 mixed 输出回到真实 atom 监督时，需要通过 `extract_real_point_output()`、`extract_real_tensor_from_mixed()` 等工具拆分。

## 6. 关键模型输出字段

常见 backbone 输出包括：

| 字段 | 常见 shape | 说明 |
| --- | --- | --- |
| `atom_logits` | `(N_real, C_atom)` | 真实 atom 分类 logits |
| `atom_target` | `(N_real,)` | 真实 atom 监督 target |
| `atom_valid_mask` | `(N_real,)` | 真实 atom 有效监督掩码 |
| `voxel_features` | `(B, C_feat, D, H, W)` | voxel backbone 中间/最终特征 |
| `voxel_logits_aux` | `(B, C_aux, D, H, W)` | receptor/auxiliary dense logits |
| `voxel_logits_ligand` | `(B, C_ligand, D, H, W)` | dense ligand logits |
| `voxel_recycle_out` | 依实现而定 | voxel recycle state |
| `point_feat` / `point_state` / `point_recycle_out` | 依 point backbone 而定 | point backbone 输出与 recycle state |
| `candidate_batch_index` | `(sumC,)` | 每个 C 候选 voxel 所属 BOX |
| `candidate_voxel_zyx` | `(sumC, 3)` | 每个 C 候选 voxel 的 z/y/x index |
| `candidate_logits` | `(sumC, C_ligand)` | C 位置对应的 dense ligand logits |
| `candidate_prob` | `(sumC,)` | 实际进入 C 的候选概率 |
| `candidate_counts` | `(B,)` | 每个 BOX 实际 C 数 |
| `candidate_counts_by_class` | `(B, K)` | 每个 BOX/候选类实际 C 数 |
| `candidate_p_sampling_by_class` | `(B, K)` | 每个 BOX/候选类实际 sampling boundary 概率 |
| `candidate_target_counts_by_class` | `(B, K)` | cap 前目标或命中候选数 |
| `anchor_voxel_zyx` | `(sumP, 3)` | P anchor 的 z/y/x index |
| `anchor_counts` | `(B,)` | 每个 BOX 的 P anchor 数 |
| `pseudo_feature` | `(sumP, C_pseudo)` | P anchor 对应 pseudo atom feature |
| `ligand_refine_logits_C` | `(sumC, C_ligand)` | sparse refine 后的 C 级 ligand logits |

不是每个实验都会输出所有字段。UNet-only、未启用 candidate builder、未启用 sparse refine head 时，C/P/refine 字段可能不存在；当前 wrapper 会先检查核心 candidate 字段再更新对应 diagnostics。

## 7. Candidate C、Anchor P 与 sparse refine

`sparse_refine` 子目录建议按以下顺序读：

1. `candidate_set.py`
   - `SparseCandidateSetBuilder` 从 dense ligand logits 生成候选 C。
   - 当前正式 selection mode 包括 `adaptive_threshold`、`recorded_threshold`、`topk`。
   - `topk` 正式模式使用 `max_candidate_voxels_per_class` 作为每 BOX/类 top-k 请求数；不要新增或依赖单独的 `topk_per_class`。
   - warmup fixed-topk 使用 `warmup_topc_per_class`，只在 wrapper 同步 runtime 允许时覆盖正式模式。
2. `anchor_sampler.py`
   - 从 C 中选 P anchor，常见策略包括 weighted FPS、unweighted FPS、topk NMS。
3. `density_cube.py`
   - 围绕 P anchor 抽取局部 raw density cube，并编码成 pseudo atom feature。
4. `interpolation.py`
   - 处理 dense voxel feature 与稀疏 C/P 位置之间的采样或插值。
5. `sparse_refine_head.py`
   - 用 P anchor 信息更新 C 级 ligand logits，输出 `ligand_refine_logits_C`。

Wrapper 维护的 candidate cache 有两个语义：

- `p_best_by_class`：来自 `val_uncapped_best` 的 dense best-F1 阈值。
- `p_sampling_by_class`：来自 diagnostics 计算的 sampling 阈值，用于 recorded/adaptive 采样 runtime。

sanity check 和 tuner 不应污染正式 cache；普通 fit validation 可以写回 cache，用于 warmup 后的正式 candidate sampling。

## 8. Loss、metric 与 diagnostics 的分工

当前 wrapper helper 分工如下：

- `voxel_point_stage1_losses.py`
  - 只计算 loss term，不构造 W&B key，不写日志。
  - `compute_atom_loss_term()` 消费 atom logits/target/mask。
  - `compute_receptor_loss_term()` 消费 `voxel_logits_aux`、`voxel_label`、`hardmask`/valid mask 相关字段。
  - `compute_voxel_ligand_loss_term()` 消费 `voxel_logits_ligand` 与 `ligand_dist_map` 派生 target。
  - `compute_sparse_refine_loss_term()` 消费 `ligand_refine_logits_C`、C 级 target 和 C 级 valid mask。
- `voxel_point_stage1_metrics.py`
  - 管理常规 validation AP/PRAUC。
  - 二分类不输出 macro；多分类对前景类输出 suffix，必要时输出 macro。
  - 保留 `val_metric_device_policy: auto | cpu | gpu` 语义。
- `voxel_point_stage1_diagnostics.py`
  - 管理 CPC 面板的 histogram/count buffer。
  - 所有 buffer 固定形状，并通过 `persistent=False` 避免进入 checkpoint。
  - DDP 下所有 rank 必须对称调用 all-reduce，不能因本地无正例提前跳过同步。
- `voxel_point_stage1_logging.py`
  - 构造统一 metric key，记录 scalar payload，rank0 写 artifact / W&B curves。
- `voxel_point_stage1_scheduler.py`
  - 构造 optimizer/scheduler，并处理 warmup-only / warmup-plateau 相关状态。

## 9. CPC validation 指标怎么读

当前 validation 里，dense ligand 分支和 sparse refine 分支按时间顺序更新以下面板：

1. `val_uncapped_best`
   - dense 全空间统计。
   - 看 dense ligand 分支理论 best-F1、`p_best` 和 best-F1 cutoff 下的候选数量。
2. `val_uncapped_sampling`
   - dense 全空间统计，但按真实 per BOX/per class sampling boundary 聚合。
   - 看当前 sampling 策略打算怎样切 C，输出 `p_sampling_*`、`sampling_F1`、`numC_sampling_target`、`numC_sampling_cutoff`。
3. `val_capped`
   - 实际进入 C 的候选统计。
   - `recall` 看 C 是否覆盖 dense GT 正例，`num_C` / `num_P` 看候选和 anchor 规模。
4. `val_unrefined`
   - 只在 C 内，用 C 位置原始 dense logit 做 local 判别统计。
   - `fn` 是 C 内 local FN，不包含 C 外漏掉的 dense GT。
5. `val_refined`
   - 只在 C 内，用 sparse refine 后的 `ligand_refine_logits_C` 做 local 判别统计。
6. `val_score`
   - 端到端分数。
   - `refined_F1` / `unrefined_F1` 只允许 C 内体素预测为正，C 外 GT 正例计入 FN。
   - `refined_p` / `unrefined_p` 各自按对应 score histogram 独立选择，使 `val_score` 端到端 F1 最大；它和 `val_refined/global/F1` 的 local 语义不同。

当前 Stage1 batch 不携带旧式样本目录名。`dataset.class_names` 表示模型任务类别，并决定多分类指标名称；`pdb_id` 与 `request_role` 分别保存结构编号和冻结请求职责。

## 10. 修改前检查清单

改 Stage1 网络、wrapper、dataset、collate、loss 或配置前，至少完成：

1. 定位当前实验实际使用的 Hydra 配置和 checkpoint 配置。
2. 读 dataset/collate，确认 batch 字段、shape、mask 和 metadata。
3. 读 `src/wrappers/voxel_point_stage1.py`，确认模型输出如何进入 loss、metric、diagnostics 和 cache。
4. 读 `src/model/stage1_model.py`，确认 batch 字段如何被 embed/voxel/point/pseudo/sparse refine 消费。
5. 如果涉及 candidate C/P/refine，读 `src/model/sparse_refine/` 对应模块。
6. 抽样检查真实 `.npz` / `.json` 或一个真实 batch。
7. 如果改字段、shape、类别、mask 或日志 key，回头更新对应数据说明、配置说明和本文。

本文保持克制：只记录阅读路线和检查点。实现细节请直接看当前代码。
