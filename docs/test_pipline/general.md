# 测试 pipeline 总说明

本文是 `docs/test_pipline` 的总入口，用于说明测试 pipeline 的目标、指标口径、配置组织和查验清单。字段、路径、shape 或运行行为以实际代码和配置为准；阅读时建议同时核对 `src/inference/notes_of_infereval.md`、`src/inference/` 与 `configs/infer_or_eval/`。

- `general.md`：指标、配置组织、系统语义和查验清单。
- `运行指导.md`：服务器运行命令、产物位置和结果读取方式。
- `phenix.md`：Phenix 差图 baseline 的预生成和消费契约。
- `实施计划.md`：工程实现边界和模块分工。
- `收尾实施计划.md`：PR-AUC、Phenix、baseline 高分位网格和验证顺序。

## 1. 目标

测试 pipeline 产出一套可重复、可比较的 held-out test 结果：

1. 4 个 DL 模型：`unet_base`、`unet_c1`、`unet_c2`、`emb_unet`。
2. 2 个系统：`stardard` 与 `strict`。
3. 每个 DL 模型和系统都有一对配置：`protein_40` 校准配置和 `protein_110` 测试配置。
4. 6 个非 DL baseline：`phenix_real_space_diff_map` 与 5 个 density-channel baseline。
5. DL 与 baseline 共享同一套后处理、阈值选择、instance/top-K/PR-AUC 评估和输出格式。

固定流程为：

```text
protein_40 Stage1：只用 voxel F1 粗扫 threshold
protein_40 Stage2：在 Stage1 best threshold 附近选择测试用 threshold
protein_110 fixed test：固定 Stage2 best params，只评估，不选择参数
```

`protein_110.json` 不参与任何参数选择。

## 2. 评估指标口径

### 2.1 覆盖率阈值

覆盖率阈值固定为：

```text
coverage_thresholds = (0.3, 0.6)
```

代码位置：`src/inference/voxel_evaluator.py` 中的 `DEFAULT_COVERAGE_THRESHOLDS`。

所有 instance 和 top-K 指标后缀为：

```text
cov03
cov06
```

### 2.2 voxel 指标

voxel 级指标使用二值 mask 评估：

```text
voxel_precision
voxel_recall
voxel_f1
voxel_iou
voxel_dice
tp / fp / fn / tn
```

搜索汇总中对应平均字段为：

```text
avg_voxel_precision
avg_voxel_recall
avg_voxel_f1
avg_voxel_iou
avg_voxel_dice
```

Stage1 objective：

```text
avg_voxel_f1
```

### 2.3 global instance matching

instance 匹配使用预测 instance 与 GT instance 的双向覆盖率：

```text
pred_cover[i,j] = overlap(pred_i, gt_j) / size(pred_i)
gt_cover[i,j]   = overlap(pred_i, gt_j) / size(gt_j)
```

有效 pair 判定：

```text
pred_cover >= t AND gt_cover >= t
其中 t in {0.3, 0.6}
```

一对一匹配用 Hungarian matching，匹配分数为：

```text
sqrt(pred_cover * gt_cover)
```

数据集级汇总字段：

```text
global_instance_precision_cov03
global_instance_recall_cov03
global_instance_f1_cov03

global_instance_precision_cov06
global_instance_recall_cov06
global_instance_f1_cov06
```

补充输出一组 loose instance 指标，用于观察宽松命中情况，不参与 Stage2 objective：

```text
global_instance_precision_loose_cov03
global_instance_recall_loose_cov03
global_instance_f1_loose_cov03

global_instance_precision_loose_cov06
global_instance_recall_loose_cov06
global_instance_f1_loose_cov06
```

loose 口径不做 Hungarian、不做一对一约束：

```text
precision_loose: 一个预测 instance 被 hit 当且仅当存在任意 GT instance，使 overlap(pred_i, gt_j) / size(pred_i) >= t
recall_loose:    一个 GT instance 被 hit 当且仅当存在任意预测 instance，使 overlap(pred_i, gt_j) / size(gt_j) >= t
f1_loose:        用数据集级 precision_loose 与 recall_loose 计算 2PR/(P+R)
```

### 2.4 top-K successful ratio

top-K 只做 per-sample successful ratio，K 固定为 3、4、5。

排序只用预测候选的 `score_mean`，不使用 `score_max`、`voxel_count` 或 fallback 排序。

数据集级汇总字段：

```text
top3_success_ratio_cov03
top4_success_ratio_cov03
top5_success_ratio_cov03

top3_success_ratio_cov06
top4_success_ratio_cov06
top5_success_ratio_cov06
```

### 2.5 PR-AUC

PR-AUC 是 voxel 语义级 Average Precision，只在 `protein_110 fixed test` 阶段计算一次，不进入 Stage1/Stage2 搜索循环。

口径：

```text
valid = hardmask == 0
score = ligand_pred[valid]
label = gt_ligand_mask[valid]
AP = Σ (R_n - R_{n-1}) * P_n
```

空 GT 样本：有效区无 GT 正类时，该样本 `pr_auc = None`，不计入 `pr_auc_macro`。

输出字段：

```text
per_sample / per_sample_best_metrics.json / best_outputs/<sample>/metrics.json: pr_auc
best_summary: pr_auc_macro, pr_auc_num_valid
```

PR-AUC 只接二分类前景路径；多分类 class-view 路径显式关闭 `compute_pr_auc`。

## 3. 三段流程与 objective

入口：`src/inference/main/two_stage_basic.py`。

### 3.1 Stage1：protein_40 threshold 粗扫

固定参数：

```text
filter_strength = basic
min_component_voxels = 10
connectivity_policy = 7_none
merge_dist = 0.0
vis_enable = false
compute_instance_metrics = false
```

DL 搜索空间：

```text
threshold: 0.00 到 1.00, step = 0.01
```

baseline 搜索空间由 `stage1_search_space` 覆盖为高分位网格：

```text
threshold: 0.97 到 1.00, step = 0.0001
```

objective：

```text
avg_voxel_f1
```

### 3.2 Stage2：protein_40 局部完整评估

固定参数：

```text
filter_strength = basic
min_component_voxels = 10
connectivity_policy = 7_none
merge_dist = 5.0
vis_enable = true
compute_instance_metrics = true
```

DL 阈值窗口：

```text
[Stage1 best - 0.10, Stage1 best + 0.10]
step = 0.01
```

baseline 阈值窗口由配置覆盖：

```text
[Stage1 best - 0.001, Stage1 best + 0.001]
step = 0.0001
```

objective：

```text
avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06
```

### 3.3 Stage3：protein_110 fixed test

测试阶段不搜索参数。它读取 Stage2 的 `best_params.json`，固定后处理参数后只评估一次。

额外开启：

```text
compute_pr_auc = true
```

stdout 打印：

```text
test_threshold
avg_voxel_f1
pr_auc_macro
global_instance cov03/cov06 F1/P/R
global_instance loose cov03/cov06 F1/P/R
top3/top4/top5 cov03/cov06 successful ratio
```

## 4. DL 配置组织

DL 配置统一放在：

```text
configs/infer_or_eval/DL/
```

共 16 个：

```text
emb_unet_stardard.yaml
emb_unet_stardard_40.yaml
emb_unet_strict.yaml
emb_unet_strict_40.yaml
unet_base_stardard.yaml
unet_base_stardard_40.yaml
unet_base_strict.yaml
unet_base_strict_40.yaml
unet_c1_stardard.yaml
unet_c1_stardard_40.yaml
unet_c1_strict.yaml
unet_c1_strict_40.yaml
unet_c2_stardard.yaml
unet_c2_stardard_40.yaml
unet_c2_strict.yaml
unet_c2_strict_40.yaml
```

命名约定：

- 带 `_40`：`protein_40.json` 校准配置。
- 不带 `_40`：`protein_110.json` 测试配置。
- `stardard`：真实结构系统。
- `strict`：cryoatom/预测结构系统。

### 4.1 DL 路径策略

```text
ckpt_path  使用 /home/penghongen/My_Project/feedback_plus/logs/...
cache_root 使用 /home/penghongen/My_Project/feedback_plus/infer_cache/...
output_root 使用 /home/penghongen/My_Project/EVAL_OUT/infer_out/...
error_dir   使用 /home/penghongen/My_Project/EVAL_OUT/infer_error/...
vis_output_root 使用 /home/penghongen/My_Project/EVAL_OUT/infer_vis/...
```

含义：

- 模型 checkpoint 使用训练输出。
- GPU forward `.npz` 缓存使用 `feedback_plus/infer_cache`，避免重复 forward。
- 评估输出、报错和可视化写入 `EVAL_OUT`。

### 4.2 DL 后处理默认值

DL 配置的 `output_heads` 默认为：

```text
output_heads: ["ligand"]
```

对应的直接运行后处理默认值为：

```text
filter_strength: basic
```

`advanced` 后处理需要 `receptor_pred`；如需使用 advanced，应显式让模型输出 receptor head，并确认 checkpoint 支持该 head。

DL 配置中的 `objective_expr` 为：

```text
avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06
```

`two_stage_basic.py` 会在 Stage1/Stage2/test 内部覆盖 `objective_expr`；配置中的 `objective_expr` 主要用于直接 `run.py --config ... mode=voxel_param_search` 单阶段运行。

## 5. baseline 配置组织

非 DL baseline 配置统一放在：

```text
configs/infer_or_eval/non_DL/
```

两个基础配置：

```text
baseline_base_stardard.yaml
baseline_base_strict.yaml
```

`src/inference/main/run_baseline_two_stage.py` 会按 system 自动读取：

```text
configs/infer_or_eval/non_DL/baseline_base_<system>.yaml
```

baseline 的实际路径不写死在 YAML 中，而由 `run_baseline_two_stage.py` 的 `derive_baseline_paths(base_dir, baseline_name, system, split)` 派生。

推荐运行时使用：

```text
--base_dir /home/penghongen/My_Project/EVAL_OUT
```

这样 baseline cache、输出、可视化和错误目录全部落在 `EVAL_OUT` 下。baseline cache 首次运行会在 `EVAL_OUT/infer_cache/baseline/` 下生成 DL-compatible `.npz`。

## 6. baseline score 与缓存契约

baseline 不直接进入评估，而是先生成 DL-compatible `.npz` cache。

缓存字段与 DL 预测缓存兼容：

```text
ligand_pred          = baseline score map after per-sample rank equalization
receptor_pred        = None
hardmask             = 当前系统结构输入对应的 receptor hardmask
resampled_emdb       = 重采样实验密度图
origin / voxel_size  = 体素网格信息
gt_ligand_mask       = 与 DL 同口径的 GT ligand mask
gt_instance_label    = 与 DL 同口径的 GT instance label
meta                 = baseline_name / system / input paths / raw_diff_map_path 等
```

### 6.1 per-sample rank equalization

所有 baseline score map 都在每个样本内部做保序 rank 均衡：

```text
valid_mask = hardmask == 0
score_eq[valid_mask] = (rank - 1) / (N_valid - 1)
score_eq[hardmask > 0] = 0
```

threshold 统一作用在 `[0,1]` 区间，baseline 的高分位阈值网格因此具有可比性。

### 6.2 六组 baseline

固定六组：

```text
phenix_real_space_diff_map
diff_clipnorm_nopost
posdiff_clipnorm_DoG1
posdiff_clipnorm_DoG2
posdiff_clipnorm_smooth1
posdiff_clipnorm_smooth2
```

Phenix 差图需要先由 `generate_phenix_diff_maps.py` 预生成并对齐，再由 baseline cache 生成脚本消费。density-channel baseline 直接用 `src/datasets/density_channel_builder.py` 中的通道实现整卷计算。

## 7. 系统语义

`stardard` 与 `strict` 是两个不同系统，不混合比较。

```text
stardard:
  structure_input_source = cif_gt_path
  sim_map_source = sim_map_path
  gt_receptor_source = cif_gt_path

strict:
  structure_input_source = cif_path
  sim_map_source = sim_map_path_cryoatom
  gt_receptor_source = cif_path
```

这套语义同时作用于 DL forward、density-channel baseline 和 structure GT 构造。

## 8. 查验清单

修改或新增配置后，至少确认：

1. DL 配置数量为 16，baseline 基础配置数量为 2。
2. DL 的 `ckpt_path` 与 `cache_root` 指向 `feedback_plus`。
3. DL 的 `output_root`、`error_dir`、`vis_output_root` 指向 `EVAL_OUT`。
4. DL 配置直接运行时使用 `filter_strength: basic`；如启用 `advanced`，必须同时提供 `receptor_pred`。
5. objective 使用 `avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06`。
6. 输出字段使用 `cov03/cov06`。
7. baseline runner 读取 `configs/infer_or_eval/non_DL/baseline_base_<system>.yaml`。
8. `protein_110` 只做 fixed test，不参与 threshold 选择。
9. loose instance 指标只作为补充输出，不写入 Stage2 objective。
10. baseline/phenix 默认产物根目录为 `/home/penghongen/My_Project/EVAL_OUT`。
