# Handoff: 测试 pipeline 收尾三项实现完成(待服务器数值验证)

Date: 2026-06-07

## Current State

`docs/test_pipline/收尾实施计划.md` 的全部改动**已实现并通过本地轻量检查**(py_compile + YAML 解析 + 无残留 `phenix_diff_map_path` JSON 字段读取 / 无 alpha|beta)。
本地无 numpy/scipy, **数值验证只能在服务器 conda `Pocket_Plus_centos7_cu121_allgpu` 跑**, 尚未跑。
grill 已结束, 16 条决议见计划文档顶部「定稿决议」。

## Completed(本轮)

- **PR-AUC = AP, 仅 110 test 一次**: `src/inference/voxel_evaluator.py` 新增 `evaluate_voxel_pr_auc`(有效区 hardmask==0, 阶梯式 AP, 空 GT 返回 None)。
  `voxel_tuning.py`: 每样本算(二分类 ndim==3, 空 GT warn)、`_summarize_postprocess_metrics` 出 `pr_auc_macro`+`pr_auc_num_valid`、
  `_extract_best_metrics` 放行 pr_auc*、多分类 class 路径强制 `compute_pr_auc=False`。`voxel_pipeline._eval_params_from_cfg` 加 `compute_pr_auc`(缺省 False)。
  AP 对单调变换不变 → DL 与 baseline(rank 均衡)可直接比。单/批 `_evaluate_single_post_result` 路径**未**加 PR-AUC(范围最小)。
- **two_stage_basic.py**: `build_stage1_cfg` 读 `stage1_search_space`、`build_stage2_cfg`/`_threshold_window` 读 `stage2_threshold_window_halfwidth`+`stage2_threshold_step`(DL 缺省 0..1/0.01 与 ±0.10/0.01 不变);
  `build_test_cfg` 设 `compute_pr_auc=True`; `run_two_stage_then_fixed_test` 捕获 test 结果并 `_print_final_test_metrics`(test_threshold+avg_voxel_f1+pr_auc_macro+global cov04/06+topK)。
- **baseline 高分位网格**: `baseline_base_stardard/strict.yaml` 加 `stage1_search_space {0.97,1.0,0.0001}`、`stage2_threshold_window_halfwidth 0.001`、`stage2_threshold_step 0.0001`;移除 `phenix_diff_map_path` 占位, 加 `phenix_output_root`。
- **phenix A2 约定派生**: `build_baseline_cache.derive_phenix_map_path(root,system,sample)`=`{root}/{system}/{sample}/phenix_diff_aligned.mrc`;消费端按此读取(缺文件 fail-fast), meta 记派生路径。
- **新 `generate_phenix_diff_maps.py`**: 核心 `generate_one_phenix_diff_map`(可被用户层 import)+ 实验批壳(val/test×两系统, 去重幂等);
  resolution 链 json>csv>3.5(warn, 无在线 API/无全局 CLI);receptor 剔除仅 stardard(model=cif_gt_path), strict 直接喂 cif_path;phenix 吃 mmCIF;
  `make_model_grid(1.0)` 对齐 + 断言 shape & origin;机器常量全 CLI。
- **共享 util**: 新 `src/inference/utils/receptor_strip.py`(`ReceptorOnlySelect`+`extract_receptor_cif`);`Bundle_of_Maps/simulated_map/get_receptor_from_PDB.py` 改为 import 它。
- **run_baseline_two_stage.py**: 加 `--phenix_output_root`(默认 `.../feedback_plus/phenix_diff_maps`), 注入每个 cfg。

## Decisions(grill 锁定, 详见计划文档 16 条)

PR-AUC 只 test 一次/AP/二分类/DL+baseline 都算/空 GT warn 不计 macro;phenix 选 A 预生成;receptor 剔除仅 cif_gt_path 无开关;
resolution json>csv>3.5;A2 约定派生不碰 JSON;机器常量收 CLI(面向用户 b);本轮只做实验/复现层, 核心可 import, 不写 predict_one.py。

## 关键技术发现(来自另一 agent 的 phenix 诊断)

- full model 会把 ligand 密度扣进差图 → **必须 receptor-only**。
- rank 均衡后分布近似 [0,1] 均匀, 信号集中在极高分位(7xy7 最佳 0.9999)→ baseline 必须高分位网格(本轮已实现)。
- 之前 top-K≈0 很可能是旧粗网格假象, 不是 phenix 失效 → 服务器验证时重点看高分位网格下是否改善。

## Next Actions

1. 服务器同步代码后, 先用 1 个已存在 cache 的 DL 配置跑 `two_stage_basic` 三段, 确认 PR-AUC 进 test best_summary 与末尾 print。
2. 跑 `generate_phenix_diff_maps.py`(1 系统 3 样本 smoke), 再 `run_baseline_two_stage.py` 单 baseline, 确认高分位网格 + A2 派生命中。
3. 全量: 16 DL 配置 + 6×2 baseline。
4. `docs/test_pipline/phenix/phenix_smoke_driver.py` 是诊断 tmp 原型, 已被 `generate_phenix_diff_maps.py` 取代, 不再维护。

## Files To Reopen

- `docs/test_pipline/收尾实施计划.md`(本轮 spec)
- `src/inference/voxel_evaluator.py` / `voxel_tuning.py` / `main/voxel_pipeline.py` / `main/two_stage_basic.py`
- `src/inference/main/build_baseline_cache.py` / `main/run_baseline_two_stage.py` / `main/generate_phenix_diff_maps.py`
- `src/inference/utils/receptor_strip.py`
- `configs/infer_or_eval/baseline_base_stardard.yaml` / `baseline_base_strict.yaml`
