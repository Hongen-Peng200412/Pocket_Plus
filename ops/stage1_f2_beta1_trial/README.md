# F2 语义 blobs 的 beta=1 basic 增量实验

本目录保存一次可复现的 Stage1 CPU 增量实验。实验复用已经按 semantic macro
F2 选出的概率阈值及其 `F2_blobs.npz`，只把 basic 后处理调参目标改为 beta=1，
然后用同一套冻结参数评估 calibration 和 validation。

## 科学边界

- 语义候选固定为原 macro 结果根中的 `F2_blobs.npz`。
- basic 调参仍使用绝对 `score_threshold` 与最终 `min_voxels`。
- `prefiltered_min_voxel` 固定为 8，与原 F1/F2 basic 实验一致。
- 调参目标是 semantic、coverage@0.3 与 one-to-one@0.3 三项 PDB 等权 macro
  F1 之和。
- 参数只在 100 个 calibration PDB 上拟合，再原样用于 200 个 validation
  PDB；validation 不重新调参。
- evaluate 继续发布 semantic、coverage、one-to-one 的 micro/macro 指标和
  semantic micro/macro PRAUC。

本入口不计算 probability，不重新拟合 F2 语义阈值，不重新生成 F2 blobs，
也不读取或生成 centered。现有 `F2_basic.json` 始终保持原位；新参数单独写为
`F2_basic_beta1.json`。

## 目录与文件

正式实验根为：

```text
/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950
```

新增正式产物为：

```text
artifacts/unet_c1/tuning/F2_basic_beta1.json
artifacts/unet_c1/calibration/evaluation/f2_blobs_basic_beta1_selected.*
artifacts/unet_c1/validation/evaluation/f2_blobs_basic_beta1_selected.*
```

`run_trial.sh` 在 Slurm 临时目录建立一个只读 calibration 目录映射，让现有
`stage1_v3.sh tune` 把默认名称 `F2_basic.json` 写到临时根。脚本随后把该文件
原子发布为正式根中的 `F2_basic_beta1.json`，因此调参期间也不会覆盖原来的
beta=2 参数。临时目录在脚本退出时删除，不属于正式产物。

## 运行参数

脚本只接收两个位置参数：

1. `experiment_root`：上述正式实验根，内部必须已有 `inputs/calibration.json`、
   `inputs/validation.json` 和完整的 F2 blobs。
2. `data_root`：Stage1 数据根，用于读取 `ligand_area.npz` 与 `union_mask.npy`。

正式作业经项目通用 `训练与运行/submit_task.sh` 提交；完整提交命令、Job 编号
和结果记录在 `EXECUTION.md`。作业原始输出同时写入正式实验根的
`feedback/F2_basic_beta1.log`。
