# Handoff: Stage1 Wrapper Refactor 当前状态

Date: 2026-05-29

## Current State

本轮围绕 `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md` 做了 grill 决策、第一阶段实现与审计。当前不是完整完成状态：只完成了 topk candidate builder、若干 helper 骨架、新 wrapper import path 恢复，以及部分 loss/scheduler helper 接入。CPC diagnostics 新体系、完整 `val_score` / `val_unrefined` / `val_refined` / artifact / W&B 曲线 / active monitor metric 迁移还没有完成。

当前 focused suite 结果：`46 passed`，但没有跑 `python src/train.py ...` 单卡 smoke、topk smoke 或 DDP smoke。

## Completed

- 在计划顶部加入了 grill 决策修正记录；后续按顶部修正优先，不再逐段修正文档正文。
- `SparseCandidateSetBuilder` 支持 `selection_mode="topk"`。
  - 按最新决策：不新增 `topk_per_class`，正式 topk 直接使用 `max_candidate_voxels_per_class` 作为每 BOX/类别 top-k 请求数。
  - 没有从 `warmup_topc_per_class` fallback。
- 新增 topk candidate set 配置：
  - `configs/model/sparse_refine/candidate_set/binary_topk.yaml`
  - `configs/model/sparse_refine/candidate_set/tri_topk.yaml`
- `configs/model/default.yaml` 新增：
  - `class_names: ${dataset.class_names}`
  - `validation_diagnostics` 基础配置块
- 恢复新 wrapper import path：`src/wrappers/voxel_point_stage1.py`。
  - 当前由 `voxel_point_stage1_old.py` 复制后小步接入 helper，仍不是 thin coordinator。
- 新增 helper 模块：
  - `src/wrappers/voxel_point_stage1_losses.py`
  - `src/wrappers/voxel_point_stage1_logging.py`
  - `src/wrappers/voxel_point_stage1_scheduler.py`
  - `src/wrappers/voxel_point_stage1_metrics.py`
  - `src/wrappers/voxel_point_stage1_diagnostics.py`
- 新 wrapper 已部分接入：
  - loss helper：atom / receptor(voxel_aux) / voxel_ligand / sparse_refine loss 计算
  - scheduler helper：warmup step、warmup-only、warmup-plateau 构造
  - 对外日志名：新 wrapper 中 `val/voxel_aux*` / `train/voxel_aux*` 已迁移为 `receptor`；old 参考文件未改。
- 新增/扩展 focused tests：
  - `tests/model/test_sparse_candidate_set.py`
  - `tests/test_voxel_point_stage1_metric_logging.py`
  - `tests/test_voxel_point_stage1_cpc_diagnostics.py`

## Decisions

- 保留 `warmup_topc_per_class` 名称，不重命名。
- 正式 `selection_mode=topk` 复用 `max_candidate_voxels_per_class`，不新增 `topk_per_class`。
- 新建/恢复 `src/wrappers/voxel_point_stage1.py` 作为正式入口，`src/wrappers/voxel_point_stage1_old.py` 仅作参考。
- 内部字段与配置继续保留 `voxel_aux_*`；对外日志、artifact、文档使用 `receptor`。
- `random_BOX` 固定作为独立 `by_source_folder` diagnostics 组；单个 BOX 可无正例，但整体 random_BOX 验证集预期有正例。
- 正式 fit validation 中，所有 candidate class 整个 epoch 无 GT 正例可以 fail-fast；sanity/tuner/smoke 不应因此中断。
- 多分类 sampling 意图与 capped unique+routed 实际 C 是两个语义层，不强行令计数一致。
- 二分类 sparse-refine active monitor 应迁移为 `val_score/global/refined_F1`。
- 对计划产生相反决策时，只维护顶部修正记录，不浪费 token 逐段修正文档正文。

## Open Questions

- 是否继续按当前“复制 old wrapper 后逐步替换”的路线，还是下一轮改为更彻底地重写 thin coordinator？当前路线更稳，但短期内文件不会明显变短。
- `ValidationMetricManager` 和 `CpcValidationDiagnostics` 是否先完整接入 wrapper，再统一迁移旧 metric key；还是先把旧 metric key 保持到 smoke 通过后再集中迁移？
- `formal validation no GT fail-fast` 的实现条件需要明确：用 trainer.sanity_checking / tuning 判断，还是额外加 diagnostics config 开关？

## Next Actions

1. 接入 `ValidationMetricManager` 到新 wrapper，替代旧 TorchMetrics 管理逻辑。
2. 接入 `CpcValidationDiagnostics` 到 validation 主流程：
   - source folder encode
   - `val_uncapped/best`
   - `val_uncapped/sampling`
   - `val_capped`
   - `val_unrefined`
   - `val_refined`
   - `val_score`
3. 完成 DDP 对称 `sync_fn` / `_all_reduce_sum` 路径。
4. 接入 `write_validation_artifacts` 与 `log_wandb_curves`，只在 `trainer.is_global_zero` 执行。
5. 迁移 active configs 的 `monitor_metric`：
   - 二分类 sparse-refine 改为 `val_score/global/refined_F1`
   - unet/ligand 相关旧 `val/...` key 需按新 key 表迁移或明确保留范围
6. 检查 checkpoint state：validation metric/diagnostics 中间状态不得进入 checkpoint，candidate threshold cache 保留。
7. 创建计划要求的两份文档：
   - `CLAUDE/docs/wrapper_metrics_reference_ai.md`
   - `docs/model/stage1_wrapper_cpc_guide.md`
8. 跑最终验证：
   - focused tests
   - `tests/inference/test_get_voxel_pred.py`
   - 单卡 smoke: `python src/train.py experiment=MINI_sparse_refine000 trainer.max_epochs=1`
   - topk smoke: `python src/train.py experiment=MINI_sparse_refine000 model/sparse_refine/candidate_set=binary_topk trainer.max_epochs=1`
   - DDP smoke（服务器/合适 GPU 环境）

## Files To Reopen

- `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md`
- `src/wrappers/voxel_point_stage1.py`
- `src/wrappers/voxel_point_stage1_old.py`
- `src/wrappers/voxel_point_stage1_losses.py`
- `src/wrappers/voxel_point_stage1_logging.py`
- `src/wrappers/voxel_point_stage1_scheduler.py`
- `src/wrappers/voxel_point_stage1_metrics.py`
- `src/wrappers/voxel_point_stage1_diagnostics.py`
- `src/model/sparse_refine/candidate_set.py`
- `configs/model/default.yaml`
- `configs/model/sparse_refine/candidate_set/binary_topk.yaml`
- `configs/model/sparse_refine/candidate_set/tri_topk.yaml`
- `tests/model/test_sparse_candidate_set.py`
- `tests/test_voxel_point_stage1_metric_logging.py`
- `tests/test_voxel_point_stage1_cpc_diagnostics.py`
