# Handoff: 测试 pipeline 评估改造 + baseline 链路 + 收尾项

Date: 2026-06-07

## Current State

`docs/test_pipline/实施计划.md` 的主体改造**已实现并通过本地轻量检查**（py_compile + YAML 解析 + 无残留引用）。
目前处于**第二轮 grill**（收尾三项），尚未实现，待 grill 完写"改动计划"落盘后再编码：

1. 体素语义级 PR-AUC（**Q1 已锁**）。
2. two_stage_basic.py 末尾 print 最终指标（**Q2 已锁**）。
3. receptor-only phenix 彻底纳入管线（**Q3 框架问题已抛出，等用户选 A/B/C**）。

数值验证只能在 Linux 服务器 conda 环境 `Pocket_Plus_centos7_cu121_allgpu` 跑（本地无 numpy/scipy）。

## Completed

第一轮已实现（9 项，全部 py_compile / yaml 通过）：
- `src/inference/voxel_evaluator.py`：删 `evaluate_instance_mask`/`alpha`/`beta`；新增 `build_instance_overlap_stats`、`evaluate_global_instance_matching`（单次 Hungarian，`sqrt(pred_cover*gt_cover)` 最大化，双阈值 cov04/cov06 计 tp）、`evaluate_topk_success`（按 `score_mean` 降序取 `min(K,n)`）；常量 `DEFAULT_COVERAGE_THRESHOLDS=(0.4,0.6)`、`DEFAULT_TOPK_VALUES=(3,4,5)`。
- `src/inference/voxel_tuning.py`：`eval_params` 改携 `compute_instance_metrics/coverage_thresholds/topk_values`；叶子评估按开关产 instance 计数+topK；`_summarize_postprocess_metrics`+`_aggregate_global_instance_and_topk` 求和出 `global_instance_*_cov04/06`、`top{3,4,5}_success_ratio_*`；`_extract_best_metrics` 让新指标进 best_metrics；`write_best_by_class_csv` 列改 global。
- `src/inference/main/voxel_pipeline.py`：`_evaluate_single_post_result` 统一新口径（single/batch + best_outputs 共用）；`_eval_params_from_cfg`（缺省 compute_instance=True）；Excel 列换 global/topK；`_resolve_extra_map_paths` + 两处 `build_infer_vis_bundle(extra_map_paths=...)`。
- `src/inference/utils/utils.py`：`build_infer_vis_bundle`/`write_pymol_vis_script` 加可选 `extra_map_paths`；batch Excel 列换新指标。
- `src/inference/main/two_stage_basic.py`：`--val_config --test_config` 双必填；`run_two_stage_then_fixed_test`（Stage1 vis=false/阈值 0.00→1.00/compute_instance=False；Stage2 新 objective/compute_instance=True；110 `search_space={}` 固定测试写测试 `output_root` 根目录）；`DEFAULT_STAGE2_OBJECTIVE_EXPR="avg_voxel_f1 + global_instance_f1_cov04 + global_instance_f1_cov06"`；新增 `build_test_cfg`/`read_best_params`。
- `src/inference/main/build_baseline_cache.py`（新）：整卷 `build_density_channels`（receptor_mask=hardmask>0）或读 `phenix_diff_map_path`；`per_sample_rank_equalize`（rankdata average + `(r-1)/(N-1)`，atom 区置 0）；复用 `load_from_raw_cif`+GT loader+`save_voxel_prediction_cache`；receptor_pred=None；raw 差图 sidecar MRC 记入 `meta["raw_diff_map_path"]`。
- `src/inference/main/run_baseline_two_stage.py`（新）：6×2 枚举 + `derive_baseline_paths` 一处规则 + `run_voxel_param_search(model=None)`（命中预生成 cache 绝不 forward）。
- `configs/infer_or_eval/baseline_base_stardard.yaml` / `baseline_base_strict.yaml`（新）；清理 `voxel_single.yaml`/`voxel_batch.yaml` 的 alpha/beta（改 `compute_instance_metrics: true`）。
- `docs/test_pipline/phenix.md`（写）；`实施计划.md` 顶部加 10 条定稿决议 note。

## Decisions

第一轮 10 条见 `实施计划.md` 顶部 NOTE。第二轮：
- **Q1 PR-AUC（锁）**：加；阈值无关、手写（无 sklearn）、只在 `hardmask==0` 有效区算；**macro**（每蛋白一票）；独立开关 `compute_pr_auc` **只在 110 fixed test 置 True**（不进搜索循环）；`pr_auc` 进 per_sample，`pr_auc_macro` 进 test best_summary 并参与 print。
- **Q2 print（锁）**：在 `run_two_stage_then_fixed_test` 末尾用 `test_result["best_summary"]` 打印紧凑块（test_threshold + avg_voxel_f1 + pr_auc_macro + global_instance F1/P/R cov04/06 + topK cov04/06），只到 stdout，不写额外文件。

## Open Questions

- **Q3 phenix 框架（待用户选）**：A=committed 预生成脚本 `generate_phenix_diff_maps.py`（独立前置、幂等、消费端按系统解析，**我推荐**）/ B=runner 自动触发 / C=build_baseline_cache 内 subprocess。
- Q3 后续待 grill 细节：① **per-system 字段命名**（建议 `phenix_map_source` 选择器 + `phenix_diff_map_path_stardard/_strict`，套 `structure_input_source` 模式）；② **receptor-only strip 方式**（stardard 的 `cif_gt_path` 含 ligand 必须剥离；strict 的 `cif_path` 通常已无 ligand；用什么 preset/工具剥离）；③ **resolution 来源**（JSON 字段 / 全局默认 / EMDB metadata）；④ **对齐**（复刻 `load_from_raw_cif` 的 `make_model_grid`，target_voxel_size=1.0）。
- 已知缺口：当前 `build_baseline_cache` 读单字段 `phenix_diff_map_path`，未按系统区分 → Q3 落地时必须改成 per-system 解析。

## Next Actions

1. 等用户选 Q3 A/B/C，继续 grill per-system / receptor-only / resolution / 对齐。
2. grill 完把"改动计划"落盘（markdown 到 `docs/test_pipline/`）。
3. 实现 PR-AUC（`voxel_evaluator` 加 AP 函数 + `voxel_tuning`/`voxel_pipeline` 接 `compute_pr_auc`）、print、phenix 全链路（generate 脚本 + per-system 消费 + build_baseline_cache 修正）。
4. 服务器 `Pocket_Plus_centos7_cu121_allgpu` 跑最小链路验证（先 1 个已存在 cache 的 DL 配置，再 phenix smoke）。

## Files To Reopen

- `docs/test_pipline/实施计划.md`、`docs/test_pipline/phenix.md`
- `src/inference/voxel_evaluator.py`、`src/inference/voxel_tuning.py`
- `src/inference/main/voxel_pipeline.py`、`src/inference/main/two_stage_basic.py`
- `src/inference/main/build_baseline_cache.py`、`src/inference/main/run_baseline_two_stage.py`
- `configs/infer_or_eval/baseline_base_stardard.yaml` / `baseline_base_strict.yaml`
