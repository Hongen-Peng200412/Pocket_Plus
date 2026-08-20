# Stage1 V3 推理入口

本目录只保留一个正式脚本 `stage1_v3.sh`。它从训练运行目录恢复 checkpoint 与最终 `config.yaml`，生成完整图概率、F1/F3 连通区域、`F1_basic.npz`、`F3_centered.npz`、逐 PDB 交集矩阵和全局指标。

旧 Selector、CLG、组件森林、Li 和七套 Fα 脚本已经退出活动代码树；需要考察旧实现时使用 Git 历史。

## 两个命令

`calibrate` 对固定 calibration PDB 清单执行完整流程，冻结：

- 语义 micro-F1 与 micro-F3 的完整图阈值；
- F1 basic 的来源平均概率阈值和最小体素数；
- U-Net F3 的来源平均概率阈值和最小体素数；
- Find F3 的 A 原子 Gaussian 参数、分数阈值和最小体素数。

冻结目录同时包含 `stage1_v3.json`、`semantic_threshold_scan.npz`、`stage1_v3.metrics.json` 和最后发布的 `_COMPLETE`。冻结 JSON 保存 checkpoint 的规范化绝对路径以及冻结后的科学参数；`run` 读取 `--calibration` 指定的 JSON 和同目录完成标记，要求两者与当前命令的 checkpoint 路径一致。calibration 目录可以不同于当前 `--output-root`。

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
  --model-code-source current_workspace
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
  --model-code-source current_workspace
```

四个 producer 名称固定为 `unet_c1`、`Find_0`、`Find_1` 和 `Find_2`。checkpoint、训练配置和 producer 必须互相对应。`--model-code-source` 没有默认值：`current_workspace` 使用本轮 V3 Dataset 与模型代码，`training_snapshot` 使用训练 run 内的完整 `src_snapshot/src`。同一版本目录内必须保持该选择一致，并在 release/launch 记录中明确保存。

`--output-root` 表示一个完整推理版本目录。若同一 checkpoint 分别使用 F3 最优目标和 F2 最优目标，两个命令应传入两个不同目录，例如 `.../unet_c1_f3` 与 `.../unet_c1_f2`。代码不读取目录名含义，也不为 checkpoint、配置或代码计算摘要；版本差异由命令和执行记录明确保存。

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
├── probability/geometry.json
├── blobs/F1_blobs.npz
├── blobs/F3_blobs.npz
├── centered/F1_basic.npz
├── centered/F3_centered.npz
├── evaluation/F1_basic.npz
├── evaluation/F3_centered.npz
├── status/<role>/performance.json
└── status/<role>/_COMPLETE
```

`F3_centered` 的 eligible 数量严格大于配置中的 `blob_limit` 时，额外写 `status/F3_centered/_BLOB_EXCEED`，并继续生产 centered；不存在跳过 PDB 的特殊终态。

数据划分级评估位于 `<output_root>/<producer>/<split>/evaluation/`，每个角色同时保存逐 PDB JSONL 和全局 metrics JSON。逐 PDB NPZ 保存交集矩阵、阈值轴、覆盖命中掩码和一对一匹配下标；`topk_winning_candidate_rank` 是已选候选序列中从 0 开始的获胜名次，`topk_winning_occurrence_index` 是 occurrence 轴下标。

## 发布和恢复

NPZ、JSON 和 JSONL 都先写同目录临时文件，再原子替换最终路径。大型 NPZ 不做重复解压重读；`_COMPLETE` 只在最终 NPZ 已经替换后建立。概率压缩、CPU blobs 与下一个 PDB 的 GPU 前向重叠；centered 压缩也与下一 PDB 前向重叠。两个有界 PDB 队列防止 train 清单积累完整数组。正式运行总是为当前 checkpoint 和配置重新计算概率图；角色 `_COMPLETE` 不含生产身份，不能用来跨 checkpoint 复用概率。

正式任务仍经 `训练与运行/submit_task.sh` 提交，从而保留 release、launch、运行命令、资源身份和 Slurm 日志。脚本本身不申请资源，也不操作 `pre_lock`、`try_lock`、`kill_lock` 或 `after_lock`。
