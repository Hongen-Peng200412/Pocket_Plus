# Handoff: 测试 pipeline 最终配置与文档收口

Date: 2026-06-08

## Current State

本轮完成了测试 pipeline 的最终配置落地与文档总入口整理。当前正式口径以 `docs/test_pipline/general.md` 和 `docs/test_pipline/运行指导.md` 为准；早期 `docs/test_pipline/实施计划.md`、`docs/test_pipline/收尾实施计划.md`、`docs/test_pipline/phenix/` 下的诊断记录只保留历史设计/诊断上下文。

最终评估口径已从早期 `cov04/cov06` 改为 `cov03/cov06`，即 `DEFAULT_COVERAGE_THRESHOLDS = (0.3, 0.6)`。所有正式输出字段、默认 objective、Excel/CSV 表头、stdout 打印都已更新到 `cov03/cov06`。

DL 配置现在统一位于 `configs/infer_or_eval/DL/`，共 16 个；non-DL baseline 基础配置统一位于 `configs/infer_or_eval/non_DL/`，共 2 个。DL 配置保留旧 `feedback_plus` 的 checkpoint 与 GPU forward cache，输出/报错/可视化改到 `/home/penghongen/My_Project/EVAL_OUT`。

## Completed

- 代码侧评估口径更新：
  - `src/inference/voxel_evaluator.py`：`DEFAULT_COVERAGE_THRESHOLDS` 改为 `(0.3, 0.6)`，相关 docstring 更新。
  - `src/inference/main/two_stage_basic.py`：默认 Stage2 objective 改为 `avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06`，最终指标打印改为 `cov03/cov06`。
  - `src/inference/main/voxel_pipeline.py`：param-search Excel 表头改为 `cov03/cov06`。
  - `src/inference/utils/utils.py`：batch Excel 指标列改为 `tp_cov03`、`top*_success_cov03`。
  - `src/inference/voxel_tuning.py`：逐类 CSV 字段和 docstring 改为 `cov03/cov06`。
- baseline 配置组织更新：
  - `src/inference/main/run_baseline_two_stage.py` 读取 `configs/infer_or_eval/non_DL/baseline_base_<system>.yaml`。
  - `src/inference/main/build_baseline_cache.py` docstring 示例改为 `configs/infer_or_eval/non_DL/...`。
  - `baseline_base_stardard.yaml` / `baseline_base_strict.yaml` 已移动到 `configs/infer_or_eval/non_DL/`。
- DL 配置整理：
  - 16 个配置位于 `configs/infer_or_eval/DL/`。
  - `output_root`、`error_dir`、`vis_output_root` 的 `feedback_plus` 改为 `EVAL_OUT`。
  - `ckpt_path` 和 `cache_root` 保留 `feedback_plus`，用于复用模型与旧 GPU forward cache。
  - 删除旧死字段 `alpha`、`beta`、`delete_cache_after_search`。
  - `objective_expr` 改为 `avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06`，并更新注释说明 two-stage 会按阶段覆盖此字段。
- 文档收口：
  - `docs/test_pipline/general.md` 成为总入口，记录最终指标、配置组织、路径策略、baseline 契约和查验清单。
  - `docs/test_pipline/运行指导.md` 记录服务器运行命令、运行意义、产物说明、常见错误排查和最小汇总脚本。
  - `docs/test_pipline/实施计划.md`、`docs/test_pipline/收尾实施计划.md` 顶部添加指向 `general.md` / `运行指导.md` 的最终口径提醒。
  - `docs/test_pipline/phenix.md` 和 `docs/test_pipline/phenix/EXECPLAN.md`、`USER_NOTES.md`、`RUN_LOG.md` 增加提醒：若出现 `cov04/cov06`，只是旧草稿字段或早期诊断口径，不代表当前正式评估结果。
- 校验：
  - 7 个改动 `.py` 文件 `py_compile` 通过。
  - 16 个 DL + 2 个 non_DL 配置使用 PyYAML 解析并断言通过。
  - `docs/test_pipline/` 内不再引用旧文件名 `新评估指标-最终配置.md`。

## Decisions

- 正式 coverage thresholds 为 `(0.3, 0.6)`，字段命名统一为 `cov03/cov06`。
- DL 配置只迁移输出、错误和可视化目录到 `EVAL_OUT`；checkpoint 和 cache 继续复用 `feedback_plus`。
- baseline 不复用旧 cache（用户确认不存在）；运行时用 `--base_dir /home/penghongen/My_Project/EVAL_OUT`，baseline cache/output/vis/error 全落在 `EVAL_OUT`。
- `configs/infer_or_eval/DL/` 放 DL 配置；`configs/infer_or_eval/non_DL/` 放 baseline 配置。
- `general.md` 是 `docs/test_pipline` 的总入口；`运行指导.md` 是实际执行入口。
- 文档中对 `cov04/cov06` 的历史提法仅标注为旧草稿/早期诊断口径，不声称已经跑出正式结果。

## Open Questions

- 服务器端尚未实际跑完整 smoke / batch。需要在 `Pocket_Plus_centos7_cu121_allgpu` 环境中验证：
  1. 一个 DL smoke 是否命中旧 `feedback_plus` cache 并写入 `EVAL_OUT`。
  2. `best_summary.json` 是否包含 `cov03/cov06` 和 `pr_auc_macro`。
  3. phenix 预生成和 baseline smoke 是否按 `EVAL_OUT` 路径正常落地。

## Next Actions

1. 同步代码到服务器。
2. 按 `docs/test_pipline/运行指导.md` 先跑一个 DL smoke：
   `python src/inference/main/two_stage_basic.py --val_config DL/emb_unet_stardard_40 --test_config DL/emb_unet_stardard device=cuda:0`
3. 检查 `/home/penghongen/My_Project/EVAL_OUT/infer_out/emb_unet_stardard/best_summary.json` 中是否存在 `global_instance_f1_cov03`、`global_instance_f1_cov06`、`pr_auc_macro`。
4. 预生成 phenix 差图到 `/home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps`。
5. 跑 baseline smoke：先 `diff_clipnorm_nopost`，再 `phenix_real_space_diff_map`。
6. smoke 通过后批量跑 8 对 DL 配置和 12 个 baseline。

## Files To Reopen

- `docs/test_pipline/general.md`
- `docs/test_pipline/运行指导.md`
- `configs/infer_or_eval/DL/emb_unet_stardard.yaml`
- `configs/infer_or_eval/non_DL/baseline_base_stardard.yaml`
- `src/inference/voxel_evaluator.py`
- `src/inference/main/two_stage_basic.py`
- `src/inference/main/run_baseline_two_stage.py`
- `src/inference/main/build_baseline_cache.py`
- `CLAUDE/memory/projects/pocket-plus.json`
