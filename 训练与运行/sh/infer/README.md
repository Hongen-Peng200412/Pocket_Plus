# Stage1 V3 推理入口

本目录只保留一个正式脚本 `stage1_v3.sh`。它从训练运行目录恢复 checkpoint 与最终 `config.yaml`，生成完整图概率、F1/F3 连通区域、`F1_basic.npz`、`F3_centered.npz`、逐 PDB 交集矩阵和全局指标。

旧 Selector、CLG、组件森林、Li 和七套 Fα 脚本已经退出活动代码树；需要考察旧实现时使用 Git 历史。

## 两个命令

`calibrate` 对固定 calibration PDB 清单执行完整流程，冻结：

- 语义 micro-F1 与 micro-F3 的完整图阈值；
- F1 basic 的来源平均概率阈值和最小体素数；
- U-Net F3 的来源平均概率阈值和最小体素数；
- Find F3 的 A 原子 Gaussian 参数、分数阈值和最小体素数。

冻结文件位于 `<output_root>/<producer>/calibration/stage1_v3.json`。

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/infer/stage1_v3.sh \
  --resource h100 --gpus 1 --cpus 16 \
  --job-name unet_c1_stage1_calibration \
  -- calibrate \
  --producer unet_c1 \
  --checkpoint /绝对路径/checkpoints/BEST.ckpt \
  --resolved-config /绝对路径/config.yaml \
  --pdb-list /绝对路径/calibration_pdb_ids.txt \
  --split calibration \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3 \
  --allow-current-workspace-code false
```

`run` 读取同一 producer 的冻结文件，对 validation 或 train 清单执行相同科学契约：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/infer/stage1_v3.sh \
  --resource h100 --gpus 1 --cpus 16 \
  --job-name unet_c1_stage1_validation \
  -- run \
  --producer unet_c1 \
  --checkpoint /绝对路径/checkpoints/BEST.ckpt \
  --resolved-config /绝对路径/config.yaml \
  --pdb-list /绝对路径/validation_pdb_ids.txt \
  --split validation \
  --output-root /storage/penghongen/AdaLigand_stage1_inference_v3 \
  --calibration /storage/penghongen/AdaLigand_stage1_inference_v3/unet_c1/calibration/stage1_v3.json \
  --allow-current-workspace-code false
```

四个 producer 名称固定为 `unet_c1`、`Find_0`、`Find_1` 和 `Find_2`。checkpoint、训练配置和 producer 必须互相对应。

## 显式科学参数

共享配置位于 `configs/inference/stage1_v3.yaml`。以下字段没有 Python 默认值，运行前必须在配置或命令中可见：

- 完整图 `stride_zyx`、`gaussian_sigma`、batch、CPU 线程数和预取深度；
- centered batch、CPU 线程数、前向模式和预取深度；
- F1/F3 是否保存 `voxel_final` 和 48³ 稠密数组；
- 语义阈值分母、最小体素数搜索范围、F3 blob 上限和超限行为；
- coverage 阈值与 top-K 清单。

正式示例使用 `stride_zyx=[50,50,50]` 和规范化 Gaussian `sigma=0.5`。完整图概率和 centered 概率都不乘受体 hardmask。

## 产物路径

一个 PDB 的正式目录是：

```text
<output_root>/<producer>/<split>/<pdb_id>/
├── probability/probability_map.npz
├── blobs/F1_blobs.npz
├── blobs/F3_blobs.npz
├── centered/F1_basic.npz
├── centered/F3_centered.npz
├── evaluation/F1_basic.npz
├── evaluation/F3_centered.npz
└── status/<role>/_COMPLETE
```

`F3_centered` 的 eligible 数量严格大于配置中的 `blob_limit` 时，额外写 `status/F3_centered/_BLOB_EXCEED`。是否继续由配置中的显式布尔字段决定。

数据划分级评估位于 `<output_root>/<producer>/<split>/evaluation/`，每个角色同时保存逐 PDB JSONL 和全局 metrics JSON。逐 PDB NPZ 保存预测候选与真实 occurrence 的完整交集矩阵。

## 发布和恢复

NPZ、JSON 和 JSONL 都先写同目录临时文件，再原子替换最终路径。`_COMPLETE` 只在最终 NPZ 已经重读后建立。默认配置 `overwrite=true` 会重算产物；需要断点复用完整概率图时，显式改为 `false`。

正式任务仍经 `训练与运行/submit_task.sh` 提交，从而保留 release、launch、运行命令、资源身份和 Slurm 日志。脚本本身不申请资源，也不操作 `pre_lock`、`try_lock`、`kill_lock` 或 `after_lock`。
