# Phenix Baseline 诊断记录

本文保留 Phenix baseline 诊断中可复用的观察和执行线索，帮助后续解释结果。正式流程以 `docs/test_pipline/general.md`、`docs/test_pipline/运行指导.md` 和 `docs/test_pipline/phenix.md` 为准。

## 1. 生成层观察

- Phenix baseline 需要把生成层、cache 层和评估层分开看，避免“流程跑通但差图或对齐错误”。
- 服务器可用 phenix 二进制：

```text
/home/yangjy/software/phenix/build/bin/phenix.real_space_diff_map
```

- Phenix help 示例偏向 PDB 输入，但实测 mmCIF 可运行。使用 mmCIF 能避免 strict 预测结构中的多字符 chain id 被 PDB 格式截断。
- `stardard` 使用 `cif_gt_path` 时，需要将 model 剔除 ligand/HETATM 后再喂 phenix，避免 ligand 已被 model 解释进 model map。
- `strict` 使用 `cif_path`，按系统语义直接作为 phenix model 输入。

## 2. resolution 观察

- resolution CSV 路径：

```text
/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv
```

- CSV 列：

```text
emdb_id,resolution,fitted_pdbs
```

- 逐样本 resolution 来源需要在 summary 中记录，尤其是 fallback。

## 3. 阈值观察

- Phenix 差图经过 rank equalization 后，有用信号可能集中在极高分位。
- baseline Stage1 使用 `0.97..1.00 step=0.0001`。
- baseline Stage2 使用 `best ± 0.001 step=0.0001`。
- smoke 中 `0.995` 以上阈值对 voxel F1 有明显影响，因此不建议只看粗阈值点。

## 4. 指标观察

- Phenix 可出现 voxel 有信号但 instance/top-K 弱的情况。
- top-K 弱时，优先看 GT-overlap 最大候选的 `score_mean` 排名，而不是只看整体 voxel F1。
- instance 弱时，检查候选切分、merge、min_component 和 threshold 对 blob 的影响。

## 5. 推荐记录字段

每次 smoke 或批量运行，建议保留：

```text
system
sample_name
resolution
resolution_source
phenix model path
raw diff path
aligned diff path
aligned shape/origin/voxel_size
raw diff min/max/mean/std
best threshold
avg_voxel_f1
global_instance_f1_cov03/cov06
top3/4/5_success_ratio_cov03/cov06
pr_auc_macro / per-sample pr_auc
```

## 6. 继续诊断方向

1. 对 `stardard/strict` 的 top-K 失败做候选排名归因。
2. 扩到更多样本，避免结论由单个样本主导。
3. 对低分样本同时查看 raw diff map、filtered mask、instance label 和 GT。
4. 汇总 DL、density-channel baseline、phenix baseline 的 held-out 指标，区分方法差异和系统差异。
