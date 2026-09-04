# Handoff：unet_c1 pdb-centric-v1 固定流水线已完成

Date: 2026-09-03

## Current State

Job `367332` 已完成 pdb-centric-v1 的独立 calibration、F1 semantic/basic 选参与 held-out `test_0.json` 评估。第 2 次执行于 2026-09-03 17:40 +08:00 成功结束并停在 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_367332`；`/home/penghongen/Feedback/Pocket_Plus/allocations/367332/after_lock_367332` 继续保留。未经用户明确授权，不删除两把锁，不复用或释放该 Job。

## Inputs And Runtime

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal/checkpoints/TOP_epoch_01_score_0.4918.ckpt`
- calibration 清单：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json`，100 个 PDB
- held-out 清单：`/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json`，179 个 PDB
- 正式输出根：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts`
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_766bf257504e/Pocket_Plus`
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/367332/unet_c1_job367332_20260903T004831_a2`
- 正式命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_1`

## Frozen Calibration Parameters

- F1 semantic：`threshold_value=0.46185302734375`，calibration macro F1=`0.392776324846336`
- F1 basic：`score_threshold=0.5705057382583618`、`prefiltered_min_voxel=8`、`min_voxels=26`、`objective_beta=1`

## Held-out Closure Evidence

清单、probability、F1 blobs、逐 PDB evaluation NPZ 和 JSONL 各包含同一组 179 个 PDB；没有缺失、额外或重复标识。汇总 JSON 中全部浮点指标均为有限值。

- 汇总文件：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts/unet_c1/held_out_test_0/evaluation/f1_blobs_basic_macro_selected.metrics.json`
- 汇总 SHA-256：`97a70ee8da6c03b798db182accaac88a570a34fa937b799f638cc0bbf18d3334`
- 逐 PDB JSONL SHA-256：`303e1d12a871df396e963fcc36dba3c7cfc59c86f7ef91217e6050f0fc7ca941`
- semantic micro/macro F1：`0.5380389712545107 / 0.33924360596529357`
- semantic micro/macro PRAUC：`0.47141677152214273 / 0.3404087819676567`
- coverage 0.3 micro/macro F1：`0.5160819065672164 / 0.3878505296169011`
- one-to-one 0.3 micro/macro F1：`0.5066598360655739 / 0.3836095662271395`

## Next Actions

1. 每次醒来确认 Job `367332` 仍停在 `try_lock_367332` 且 `after_lock_367332` 未被误删；不再启动新执行。
2. 继续守护 Job `358384` 上的 pdb-centric-v2 calibration、选参与 held-out 测试。
3. pdb-centric-v2 完整闭合后，从执行记录读取三套 held-out 指标完成最终比较与收口。
