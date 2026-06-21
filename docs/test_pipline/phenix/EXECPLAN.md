# Phenix Baseline 诊断 ExecPlan

本文用于接手 Phenix baseline 的诊断与归因。正式运行命令见 `docs/test_pipline/运行指导.md`，预生成契约见 `docs/test_pipline/phenix.md`。

## 1. 目标

验证 Phenix baseline 从差图生成到 held-out test 指标的完整链路：

1. 从 `protein_40.json` / `protein_110.json` 样本资源生成 real-space difference map。
2. 将差图对齐到 DL-compatible cache 网格。
3. 生成 baseline `.npz` cache。
4. 进入 `run_voxel_param_search()`、`voxel_tuning.py`、`voxel_evaluator.py` 完成 threshold 搜索和 fixed test。
5. 对低分结果进行可追溯归因。

## 2. 边界

- 本地 Windows 负责文档、代码编辑和轻量静态检查。
- Linux 服务器负责 phenix、完整 cache 生成和批量评估。
- 不修改 `src/inference/utils/protein_40.json` / `protein_110.json`。
- Phenix 路径通过 `phenix_output_root/system/sample_name` 派生。
- `protein_110` 不参与参数选择。

## 3. 系统输入

```text
stardard:
  phenix model = cif_gt_path -> receptor-only mmCIF
  map          = map_path

strict:
  phenix model = cif_path
  map          = map_path
```

Phenix 输入使用 mmCIF，避免 PDB chain id 限制。

## 4. 跑通标准

### 4.1 生成层

- Phenix 命令成功生成差图。
- `phenix_diff_aligned.mrc` 存在。
- 差图非全 0、非全 NaN、非常数。
- `generate_summary.json` 记录 resolution、resolution_source、model_path、raw_diff_path、out_path。

### 4.2 cache 层

- cache 中 `ligand_pred/hardmask/resampled_emdb/gt_ligand_mask/gt_instance_label` 形状一致。
- `ligand_pred` 有效区经过 rank equalization 后位于 `[0,1]`。
- hardmask 区 `ligand_pred` 为 0。
- meta 中记录 `baseline_name/system/raw_diff_map_path`。

### 4.3 评估层

- `run_voxel_param_search()` 输出：

```text
best_params.json
best_summary.json
per_sample_best_metrics.json
param_search_results.xlsx
best_outputs/
```

- `best_summary.json` 含 `global_instance_*_cov03/cov06`、loose instance、top-K 与 `pr_auc_macro`。
- `per_sample_best_metrics.json` 与 `best_outputs/<sample>/metrics.json` 含 per-sample `pr_auc`。

## 5. 结果质量标准

第一轮不设置绝对 F1 门槛，使用“非退化 + 可归因”标准：

- raw diff 有动态范围。
- threshold grid 产生不同候选数或 voxel 指标。
- voxel、instance、top-K、PR-AUC 字段完整。
- 可视化能展示 raw diff map 与预测/GT 对应关系。

## 6. 失败归因规则

- Phenix 命令、结构输入、resolution 或输出定位失败：归因生成层。
- 差图存在但 shape/origin 不一致：归因对齐层。
- cache 字段缺失或 shape 不一致：归因 cache 契约。
- cache 正常但扫参/评估报错：归因评估代码或配置。
- cache 和指标均正常但结果差：检查结构/map 匹配、resolution、差图符号、阈值曲线、blob 排名、merge 和 min_component。

## 7. 执行梯度

1. `stardard` 少量样本 smoke。
2. `strict` 少量样本 smoke。
3. 扩到更多样本。
4. 完整 `protein_40/protein_110`。
5. 汇总与 DL、density-channel baseline 的 held-out 指标。

## 8. 下一步分析

Phenix voxel 有信号但 top-K 或 instance 弱时，重点分析：

1. GT overlap 最大 blob 的 `score_mean` 排名。
2. `score_mean` 与覆盖率的关系。
3. `threshold` 对候选数量和覆盖率的影响。
4. `merge_dist` 对 instance 合并的影响。
5. `min_component_voxels` 对碎片候选的影响。
6. 可视化中 raw diff map、filtered mask、instance label 与 GT 的空间关系。
