# Handoff：unet_c1 pdb-centric-v2 已完成独立校准

Date: 2026-09-03

## Current State

Job `358384` 在 `hnode01` 使用 H100×1、32 CPU 运行 pdb-centric-v2 固定流水线。100-PDB calibration、F1 semantic 阈值拟合、F1 blobs 和 F1 basic 选参均已完成，流水线已经进入 179-PDB held-out `test_0.json` 的 probability 阶段。`/home/penghongen/Feedback/Pocket_Plus/allocations/358384/after_lock_358384` 必须在整个推理期间及推理完成后继续保留。

Job `367332` 的 pdb-centric-v1 已完整结束并停在 `try_lock_367332`；其 `after_lock_367332` 同样保留，不再复用。

## Inputs And Runtime

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/checkpoints/TOP_epoch_03_score_0.5957.ckpt`
- calibration 清单：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json`，100 个 PDB
- held-out 清单：`/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json`，179 个 PDB
- 正式输出根：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts`
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus`
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260903T161305_a2`
- 正式命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_2`

## Frozen Calibration Parameters

- F1 semantic：`threshold_value=0.3990478515625`、`threshold_grid_index=13076`、macro F1=`0.43239013451144453`
- F1 basic：`score_threshold=0.5232318639755249`、`prefiltered_min_voxel=8`、`min_voxels=21`、`objective_beta=1`
- F1 semantic 文件 SHA-256：`1cc4fd9dabfbf304b9158501cda6276a10ee1f8dccf6b247afe54d9bfbf90425`
- F1 basic 文件 SHA-256：`3f6c1b12d4a420db7425039b565c76ebcd7c49e28a7a87caff7f5b2a27be9db6`

## Calibration Closure Evidence

固定 calibration 清单、probability 完成标记和 F1 blobs 完成标记各包含同一组 100 个 PDB；没有缺失、额外或重复标识。两个参数 JSON 的全部浮点均为有限值，且明确保存 `alpha=1`、`objective_beta=1` 和 `score_mode=basic`。

2026-09-03 21:19 +08:00，held-out probability 已启动；H100 利用率为 100%，显存为 79,478/81,559 MiB，进程使用上述固定 checkpoint、训练 `config.yaml`、training snapshot、held-out 清单和 v2 输出根。Job `358384` 的 traceback 与 CUDA OOM 计数均为 0。

## Next Actions

1. 稳定阶段每轮以连续 `Start-Sleep -Seconds 300` 组成 60 或 120 分钟静默等待。
2. 每次醒来同时核验 Job `367332` 的两把锁，以及 Job `358384` 的 Slurm 状态、锁、错误、H100 活动与 held-out 完成数。
3. v2 held-out 完成后核验 probability、F1 blobs、逐 PDB evaluation NPZ 和 JSONL 的 179-PDB 集合闭合，并确认汇总指标均为有限值。
4. 完成三模型结果汇总与最终 handoff；未经用户新授权不得删除任何 `after_lock`。
