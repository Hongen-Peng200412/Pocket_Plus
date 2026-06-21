# Handoff: Stage1 Wrapper Refactor 本地完成状态

Date: 2026-05-30

## Current State

本轮继续执行 `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md` 的剩余任务。当前代码层面的 wrapper refactor、diagnostics sampling、active monitor metric 迁移、旧测试迁移、本地 DDP 逻辑模拟与文档补充已完成一轮。核心 focused / 迁移 / 推理兼容测试通过；本地 `src/train.py` smoke 被 Hydra dataset 配置缺失阻塞，而不是本轮 wrapper 代码错误。

注意：本轮新增的测试、文档和 learning 记忆被当前 `.gitignore` 规则忽略，`git status` 默认不显示；如需纳入提交，需要显式处理 ignore 规则或 `git add -f`。

## Completed

- 计划顶部只新增一条简短修正：DDP 验收保留功能与本地 CPU/Gloo 逻辑模拟，不依赖服务器双卡 smoke。
- 记录 Read offset gotcha：
  - `CLAUDE/memory/learnings/gotcha-2026-05-30-read-offset-real-line-number.md`
  - 持久 auto-memory `feedback_read_offset_real_lines.md`
- `src/wrappers/voxel_point_stage1_diagnostics.py`：
  - 新增 `val_uncapped/sampling` 统计 buffer。
  - 实现 `update_uncapped_sampling()`，按真实 per-BOX/per-class `candidate_p_sampling_by_class` 边界聚合。
  - 输出 `p_sampling_p5/p50/p75/p95/mean`、`sampling_F1/precision/recall/tp/fp/fn`、`num_gt`、`numC_sampling_target`、`numC_sampling_cutoff`。
- `src/wrappers/voxel_point_stage1.py`：
  - candidate outputs 不存在时不再无条件调用 C/P/C diagnostics，避免 UNet-only 或无 sparse refine 路径崩溃。
  - candidate cache 写回拆分：`p_best` 来自 `val_uncapped/best`，`p_sampling` 来自 `val_uncapped/sampling`，不再 `p_sampling = p_best.clone()`。
- 迁移 active monitor metric：
  - sparse refine 二分类改为 `val_score/global/refined_F1`。
  - tri sparse refine 改为 `val_score/global/refined_F1_macro`。
  - unet/emb_unet ligand monitor 改为 `val_score/global/voxel_ligand_PRAUC`。
  - default atom monitor 改为 `val_score/global/atom_PRAUC`。
  - `exp004_zeros` 改为 `val_loss/global/total`。
- 旧测试迁移到新接口：
  - `tests/test_voxel_ligand_thresholds.py`
  - `tests/test_ligand_sparse_refine_metrics.py`
  - `tests/test_multiclass_ligand_wrapper.py`
- 新增本地 CPU/Gloo DDP 逻辑模拟测试：
  - `tests/test_voxel_point_stage1_ddp_diagnostics.py`
  - Windows PyTorch 需设置 `USE_LIBUV=0` 才能使用当前 Gloo 初始化。
- 新增文档：
  - `CLAUDE/docs/wrapper_metrics_reference_ai.md`
  - `docs/model/stage1_wrapper_cpc_guide.md`

## Decisions

- 不恢复旧 wrapper 私有 metric API；测试迁移到 `CpcValidationDiagnostics`、`ValidationMetricManager`、loss/helper 新接口。
- DDP 验收采用本地 CPU/Gloo 多进程逻辑模拟，覆盖 all-reduce 对称性与本地无服务器双卡场景；不把服务器 2 GPU smoke 作为完成前置。
- 正式 topk 继续使用 `max_candidate_voxels_per_class`，不引入 `topk_per_class`。
- active 配置中旧 `monitor_metric: val/...` 已迁移；`configs/experiment/old/**` 下旧项按计划保留。

## Open Questions

- `src/train.py experiment=MINI_sparse_refine000 ...` 仍依赖缺失的 `dataset/fused` config。需要决定是恢复/补充该 dataset config，还是更新 MINI sparse refine experiment 到当前可用 dataset group。
- 新增测试/文档是否应纳入 Git 提交：当前被 `.gitignore` 忽略，需显式 `git add -f` 或调整 ignore 规则。
- `CpcValidationDiagnostics` 仍是以 scalar 统计为主，curve/local table 产物目前为空；计划中完整 curve/table 体系如需严格实现，后续还可继续补。

## Next Actions

1. 处理 smoke 阻塞：检查 `configs/base.yaml` 和 `configs/dataset/`，决定补 `dataset/fused` 还是改 `MINI_sparse_refine000` 的 dataset 默认。
2. 若要提交新增测试/文档，确认 `.gitignore` 策略并显式纳入这些文件。
3. 运行更大范围测试或全量 pytest（视耗时和数据依赖决定）。
4. 若继续完善 diagnostics artifact，补 curve/local table payload 生成与对应测试。
5. 若恢复 smoke：重新运行
   - `python src/train.py experiment=MINI_sparse_refine000 trainer.max_epochs=1`
   - `python src/train.py experiment=MINI_sparse_refine000 model/sparse_refine/candidate_set=binary_topk trainer.max_epochs=1`

## Files To Reopen

- `CLAUDE/plans/implement/wrapper_refactor/2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md`
- `src/wrappers/voxel_point_stage1.py`
- `src/wrappers/voxel_point_stage1_diagnostics.py`
- `src/wrappers/voxel_point_stage1_metrics.py`
- `src/wrappers/voxel_point_stage1_logging.py`
- `tests/test_voxel_point_stage1_cpc_diagnostics.py`
- `tests/test_voxel_point_stage1_ddp_diagnostics.py`
- `tests/test_voxel_ligand_thresholds.py`
- `tests/test_ligand_sparse_refine_metrics.py`
- `tests/test_multiclass_ligand_wrapper.py`
- `tests/model/test_sparse_candidate_set.py`
- `configs/model/default.yaml`
- `configs/experiment/MINI_sparse_refine000.yaml`
- `configs/experiment/MINI_sparse_refine001.yaml`
- `configs/experiment/MINI_sparse_refine002.yaml`
- `configs/experiment/MINI_sparse_refine003.yaml`
- `configs/experiment/MINI_sparse_refine004.yaml`
- `configs/experiment/MINI_sparse_refine005.yaml`
- `configs/experiment/MINI_sparse_refine006.yaml`
- `configs/experiment/sparse_refine000.yaml`
- `configs/experiment/other/tri_sparse_refine_full.yaml`
- `CLAUDE/docs/wrapper_metrics_reference_ai.md`
- `docs/model/stage1_wrapper_cpc_guide.md`
