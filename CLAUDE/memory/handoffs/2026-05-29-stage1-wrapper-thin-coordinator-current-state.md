# Handoff: Stage1 Wrapper Thin Coordinator 续作状态

Date: 2026-05-29

## Current State

本轮继续 `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md`，并确认采用 thin coordinator 路线：不再复制 `voxel_point_stage1_old.py` 的 2000 行实现，也不保留旧 wrapper 私有 metric API。当前 `src/wrappers/voxel_point_stage1.py` 已从空文件恢复为 thin coordinator 草版，接入 loss / logging / scheduler / metric manager / CPC diagnostics helper。focused refactor suite 目前通过，但旧 `test_voxel_ligand_thresholds.py` 与 `test_ligand_sparse_refine_metrics.py` 仍有大量过时断言，需要继续迁移到 helper/manager 新接口。

当前通过的 focused suite：

```powershell
& "C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe" -m pytest tests/test_voxel_point_stage1_thin_coordinator.py tests/test_voxel_point_stage1_metric_logging.py tests/test_voxel_point_stage1_cpc_diagnostics.py tests/test_ligand_sparse_refine_loss.py -q
# 25 passed, 6 warnings
```

## Completed

- 明确回答：handoff 文件不会自动知道全文，必须读取；已读取旧 handoff 与 v4 plan。
- 决策继续 thin coordinator 路线，不恢复 old wrapper 复制版。
- 决策 formal fit validation 的 all-no-GT fail-fast 不新增 config 开关，直接按 trainer 状态判断；sanity/tuner 不 fail-fast。
- 决策旧 wrapper 私有 metric API 不保留：
  - `_update_voxel_ligand_best_f1_stats`
  - `_compute_log_update_voxel_ligand_best_f1_thresholds`
  - `_update_val_ligand_sparse_refine_metric`
  - `_compute_log_reset_ligand_sparse_refine_metrics`
  - `_register_val_metric`
- 新增 memory：`feedback_test_no_regression.md`，要求新 refactor tests 必须覆盖旧测试功能且只能更细，不能退化。
- 新增 `tests/test_voxel_point_stage1_thin_coordinator.py`：
  - wrapper 必须实例化 `ValidationMetricManager` / `CpcValidationDiagnostics`
  - wrapper 不恢复旧 metric 私有 API
  - `validation_step` 必须保持 readable time order
  - checkpoint 中保留 candidate threshold cache 且不保存 metric/diagnostics state
- `src/wrappers/voxel_point_stage1.py` 从空文件恢复为 thin coordinator 草版：
  - 显式 `class_names`，未传入 fail-fast
  - 接入 `ValidationMetricManager`
  - 接入 `CpcValidationDiagnostics`
  - 接入 loss helper、logging helper、scheduler helper
  - 保留 candidate threshold cache 的 save/load
  - 保留 sparse refine supervision 采样的最小 coordinator 方法 `_sample_ligand_refine_supervision`
- `tests/test_ligand_sparse_refine_loss.py` 已迁移：
  - 不再调用旧 `_compute_ligand_sparse_refine_loss`
  - 改为 `_sample_ligand_refine_supervision` + `compute_sparse_refine_loss_term`
  - warmup 权重测试改用 `_compute_sparse_refine_loss_effective_weight`
- `tests/test_voxel_point_stage1_metric_logging.py` 已迁移 receptor 测试：
  - 不再检查 `_val_metric_specs`
  - 改查 `wrapper.val_metrics.branch_specs` 与 `wrapper.val_metrics.metrics`
- `src/wrappers/voxel_point_stage1_diagnostics.py` 进展：
  - 按用户提示，把逐行 `register_buffer()` 改为 `_register_stat_buffers()` 规格表集中注册
  - `_stat_buffer_descriptions` 保存每个 buffer 的语义说明
  - 已新增最小 `val_capped`、`val_unrefined`、`val_refined`、`val_score` 统计输出
  - 新增 diagnostics 测试覆盖 candidate recall、C/P 均值、local F1 vs e2e F1

## Decisions

- 新测试必须具备所有旧测试功能，只能更细化，不能出现功能覆盖退化。
- 旧 wrapper 私有 API 不作为兼容目标；旧测试应迁移到 helper/manager 或少量 coordinator 集成测试。
- `voxel_point_stage1.py` 可以保留少量 coordinator 必需方法，例如 batch extraction、candidate runtime/cache 同步、sparse refine supervision 采样、checkpoint cache save/load，但不能恢复旧 metric 计算大块逻辑。
- Diagnostics buffer 注册应集中管理，避免大量手写 `self.register_buffer()`；每个 buffer 的意义应在规格表或注释中说明。

## Open Questions

- `src/wrappers/voxel_point_stage1_diagnostics.py` 当前只完成了最小 `val_capped` / `val_unrefined` / `val_refined` / `val_score`，还未完成完整 `val_uncapped/sampling`、分位数、curve、local tables、warnings 全体系。
- `formal validation no GT fail-fast` 还没有在新 diagnostics/epoch_end 中完整实现；目前 no-positive 多数仍是 warning/nan。
- `p_sampling` cache 目前在 wrapper `_update_candidate_threshold_cache_from_payload()` 中临时用 `p_best.clone()` 写回，不是最终 sampling 语义；后续必须改成来自 `val_uncapped/sampling` 的真实 sampling boundary。
- `validation_diagnostics=None` 时 wrapper 当前用 `source_folder_names=("unknown",)`，这只是保持构造可用的临时默认；Hydra active config 已有显式 `validation_diagnostics`，后续可考虑更严格 fail-fast。
- `docs` 两份计划要求文档还未写。
- active configs 的 monitor metric 迁移未完成。

## Next Actions

1. 继续迁移旧测试：
   - `tests/test_voxel_ligand_thresholds.py`
   - `tests/test_ligand_sparse_refine_metrics.py`
   把仍绑定旧 wrapper 私有 API 的断言改为 diagnostics/helper/coordinator 新接口测试。
2. 完成 `CpcValidationDiagnostics.update_uncapped_sampling()`：
   - recorded/adaptive/topk/warmup 都按真实 per BOX/per class 边界或 target count 聚合
   - 输出 `p_sampling_*`、`sampling_F1/precision/recall/tp/fp/fn`、`numC_sampling_target`、`numC_sampling_cutoff`
3. 修正 candidate threshold cache 写回：
   - `p_best` 来自 `val_uncapped/best`
   - `p_sampling` 来自 `val_uncapped/sampling`，不要再 clone `p_best`
4. 补完整 no-GT fail-fast 条件：
   - regular fit validation 可 raise
   - sanity/tuner/smoke-only 不 raise
   - per-source-folder no-positive 只 warning/nan
5. 跑更广 focused suite：
   - `tests/test_voxel_ligand_thresholds.py`
   - `tests/test_ligand_sparse_refine_metrics.py`
   - `tests/test_multiclass_ligand_wrapper.py`
   - `tests/model/test_sparse_candidate_set.py`
6. 后续再做单卡/topk smoke 与 DDP smoke。

## Files To Reopen

- `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md`
- `CLAUDE/memory/handoffs/2026-05-29-stage1-wrapper-refactor-current-state.md`
- `src/wrappers/voxel_point_stage1.py`
- `src/wrappers/voxel_point_stage1_diagnostics.py`
- `src/wrappers/voxel_point_stage1_metrics.py`
- `src/wrappers/voxel_point_stage1_logging.py`
- `src/wrappers/voxel_point_stage1_losses.py`
- `src/wrappers/voxel_point_stage1_scheduler.py`
- `src/wrappers/voxel_point_stage1_old.py`
- `tests/test_voxel_point_stage1_thin_coordinator.py`
- `tests/test_voxel_point_stage1_cpc_diagnostics.py`
- `tests/test_voxel_point_stage1_metric_logging.py`
- `tests/test_ligand_sparse_refine_loss.py`
- `tests/test_voxel_ligand_thresholds.py`
- `tests/test_ligand_sparse_refine_metrics.py`
