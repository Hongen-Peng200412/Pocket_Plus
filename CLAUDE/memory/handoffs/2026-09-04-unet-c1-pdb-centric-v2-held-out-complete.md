# Handoff：unet_c1 pdb-centric-v2 固定流水线已完成

Date: 2026-09-04

## Current State

Job `358384` 已完成 pdb-centric-v2 的训练、独立 calibration、F1 semantic/basic 选参与 held-out `test_0.json` 评估。第 2 次执行于 2026-09-04 06:03 +08:00 成功结束并停在 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_358384`；06:06 完成最终产物核验。`/home/penghongen/Feedback/Pocket_Plus/allocations/358384/after_lock_358384` 继续保留，`kill_lock_358384` 不存在。未经用户明确授权，不删除两把锁，不复用或释放该 H100 allocation。

## Inputs And Runtime

- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/checkpoints/TOP_epoch_03_score_0.5957.ckpt`
- checkpoint SHA-256：`8e8bd068ebbebeb8b49db3ddd40c6c485bad322e75f3308b39298b8cea7bec3a`
- calibration 清单：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json`，100 个 PDB
- held-out 清单：`/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json`，179 个 PDB
- 正式输出根：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts`
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus`
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260903T161305_a2`
- 正式命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_2`
- 资源：H100×1、32 CPU；完整图 batch 为 24，窗口物化线程为 18，calibration 并发为 24

## Frozen Calibration Parameters

- F1 semantic：`threshold_value=0.3990478515625`，calibration macro F1=`0.43239013451144453`
- F1 basic：`score_threshold=0.5232318639755249`、`prefiltered_min_voxel=8`、`min_voxels=21`、`objective_beta=1`
- F1 semantic 参数文件 SHA-256：`1cc4fd9dabfbf304b9158501cda6276a10ee1f8dccf6b247afe54d9bfbf90425`
- F1 basic 参数文件 SHA-256：`3f6c1b12d4a420db7425039b565c76ebcd7c49e28a7a87caff7f5b2a27be9db6`

## Held-out Closure Evidence

清单、probability、F1 blobs、逐 PDB evaluation NPZ 和 JSONL 各包含同一组 179 个 PDB；没有缺失、额外或重复标识。汇总 JSON 中全部浮点指标均为有限值。第 2 次执行没有 traceback 或 CUDA OOM。

- 汇总文件：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts/unet_c1/held_out_test_0/evaluation/f1_blobs_basic_macro_selected.metrics.json`
- 汇总 SHA-256：`da0c4302cb05a2f73c8d5079d16eab2b29507937d76279368fc4c5f1f03bd362`
- 逐 PDB JSONL SHA-256：`0ff2d75b896c16534f856beae75b03ae530d81b1ebbbb815adda1db3c6877518`
- semantic micro/macro F1：`0.552666440099597 / 0.39443718585894233`
- semantic micro/macro PRAUC：`0.5014972424689892 / 0.4016180431955654`
- coverage 0.3 micro/macro F1：`0.562123209922334 / 0.4829127244915479`
- coverage 0.3 micro/macro PRAUC：`0.409504912848759 / 0.45232694067266693`
- one-to-one 0.3 micro/macro F1：`0.5522673031026253 / 0.4781325170565899`
- one-to-one 0.3 micro/macro PRAUC：`0.3977168529069711 / 0.4459768237952151`

## Comparison Result

在固定的 F1 blobs+basic 口径下，pdb-centric-v2 的 semantic micro/macro F1、semantic micro/macro PRAUC、coverage 0.3 macro F1 和 one-to-one 0.3 macro F1 均为三种模型最高。occurrence-centric 在部分 coverage/one-to-one PRAUC 上仍略高，因此结论限于：pdb-centric-v2 是本次 F1 主口径下的最佳模型，不是所有候选排序 PRAUC 都全面最优。

完整三模型指标表见 `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`。

## Next Actions

1. 保留 Job `358384` 的 `try_lock_358384` 与 `after_lock_358384`，等待用户决定 H100 后续用途。
2. 不重新运行已经冻结的 calibration 或 held-out 流水线；后续分析直接读取三个正式输出根与执行记录。

## 09:41 Resource Follow-up

本文件前文记录的是推理刚完成时的资源状态。Slurm 随后于 08:35:30 将 Job `358384` 记录为外部取消，`JobState=CANCELLED`、`ExitCode=143:0`；本训练守护没有执行取消、删除锁或其他资源写操作。09:41 再次检查时，Job 已不在队列，`try_lock_358384`、`after_lock_358384` 与 allocation 活动文件均已由调度包装器完成退出清理，hnode01 的第三张 H100 已实际空出。未经用户新的明确授权，不自行重提 Job 或占用该资源。完整事件链见 `CLAUDE/memory/handoffs/2026-09-04-stage1-four-training-validation-milestones.md`。
