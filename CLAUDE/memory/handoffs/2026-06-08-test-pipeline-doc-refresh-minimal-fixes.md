# Handoff: 测试 pipeline 文档刷新与最小修复

Date: 2026-06-08

## Current State

本轮完成了 `docs/test_pipline` 文档向前看的刷新，并对前面只读检查发现的四个高置信不一致做了最小修复。文档现在以当前流程为主，不再用“过去/现在/旧/历史”等方式解释口径迁移；`general.md`、`运行指导.md`、`phenix.md`、`实施计划.md`、`收尾实施计划.md` 和 `phenix/` 下用户说明/执行计划/诊断记录均已重写或精简为当前可执行口径。

需要注意：`docs/` 目录被 `.gitignore` 忽略，`docs/test_pipline` 的改动不会出现在普通 `git status` / `git diff` 中；文件内容已在工作区实际写入。

## Completed

- 修复正式 DL 配置直跑风险：`configs/infer_or_eval/DL/*.yaml` 的 `filter_strength` 从 `advanced` 改为 `basic`，注释说明 advanced 需要 `receptor_pred` 和显式 receptor head。
- 补齐 test 阶段 per-sample PR-AUC 落盘：`src/inference/main/voxel_pipeline.py` 在二分类 `compute_pr_auc=true` 时复用 `evaluate_voxel_pr_auc()`，使 `per_sample_best_metrics.json` 和 `best_outputs/<sample>/metrics.json` 中有 `pr_auc`。
- 修正 `configs/infer_or_eval/voxel_single.yaml` 与 `voxel_batch.yaml` 的覆盖率阈值注释：`0.4/0.6` -> `0.3/0.6`。
- 将 `docs/test_pipline/phenix/phenix_smoke_driver.py` 精简为立即退出的 stub，提示使用正式 `generate_phenix_diff_maps.py` + `run_baseline_two_stage.py`。
- 重写 `docs/test_pipline/general.md`：当前目标、`cov03/cov06` 指标、PR-AUC per-sample 落盘、DL/basic 后处理默认、baseline cache 契约、系统语义与查验清单。
- 修订 `docs/test_pipline/运行指导.md`：去掉向后看的表述，补充 per-sample `pr_auc`，保留服务器运行命令、结果读取和排查命令。
- 重写 `docs/test_pipline/phenix.md`：明确 Phenix 预生成、A2 派生路径、输入选择、resolution、对齐和 baseline cache 衔接。
- 重写 `docs/test_pipline/实施计划.md` 与 `收尾实施计划.md`：作为当前工程边界、模块分工和验证计划，而不是旧计划记录。
- 重写 `docs/test_pipline/phenix/USER_NOTES.md`、`RUN_LOG.md`、`EXECPLAN.md`：保留诊断价值，但改成当前 Phenix baseline 诊断、归因和执行梯度说明。

## Decisions

- 不在 `voxel_postprocess.py` 做 `advanced -> basic` 自动 fallback；正式 DL 配置显式默认 `basic`，advanced 仍 fail-fast 要求 `receptor_pred`。
- PR-AUC 只补 per-sample 落盘，不改搜索、objective、`best_summary` macro 汇总逻辑，不支持多分类 PR-AUC。
- `voxel_single/batch` 只改注释，不新增 `coverage_thresholds` 配置项。
- `phenix_smoke_driver.py` 不继续维护诊断原型逻辑，只保留指向正式入口的 fail-fast stub。
- 文档语气改为面向当前流程，不强调口径迁移历史。

## Open Questions

- 还未在 Linux 服务器环境运行完整 DL smoke、phenix 预生成 smoke 或 baseline smoke；当前验证为本地静态检查与 Python 编译。
- `docs/` 被 `.gitignore` 忽略；如果这些文档需要进入版本控制，需要用户决定是否调整 ignore 或强制 add。

## Next Actions

1. 同步到服务器后，在 `Pocket_Plus_centos7_cu121_allgpu` 环境跑一个 DL smoke：`two_stage_basic.py --val_config DL/emb_unet_stardard_40 --test_config DL/emb_unet_stardard`。
2. 检查 test `best_summary.json` 中 `pr_auc_macro`、loose instance、top-K 和 `cov03/cov06` 字段。
3. 检查 `per_sample_best_metrics.json` 与 `best_outputs/<sample>/metrics.json` 中是否有 per-sample `pr_auc`。
4. 跑 `generate_phenix_diff_maps.py` 少量样本 smoke，再跑 `run_baseline_two_stage.py --baselines phenix_real_space_diff_map` smoke。
5. 若文档要纳入 git，处理 `docs/` ignore 规则或用显式强制 add。

## Files To Reopen

- `docs/test_pipline/general.md`
- `docs/test_pipline/运行指导.md`
- `docs/test_pipline/phenix.md`
- `docs/test_pipline/实施计划.md`
- `docs/test_pipline/收尾实施计划.md`
- `docs/test_pipline/phenix/USER_NOTES.md`
- `docs/test_pipline/phenix/RUN_LOG.md`
- `docs/test_pipline/phenix/EXECPLAN.md`
- `docs/test_pipline/phenix/phenix_smoke_driver.py`
- `src/inference/main/voxel_pipeline.py`
- `configs/infer_or_eval/DL/*.yaml`
- `configs/infer_or_eval/voxel_single.yaml`
- `configs/infer_or_eval/voxel_batch.yaml`
