# Handoff：unet_c1 三种采样模型比较已完成

Date: 2026-09-04

## Current State

occurrence-centric、pdb-centric-v1 与 pdb-centric-v2 的既定 Stage1 calibration-test 流水线均已完成。三者都已对 `/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json` 的同一组 179 个 PDB 生成 probability、F1 blobs、逐 PDB evaluation NPZ 和汇总 JSONL；集合闭合且汇总浮点指标均为有限值。

Job `358384` 的第 2 次执行于 2026-09-04 06:03 +08:00 成功结束并停在 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_358384`；06:06 完成最终产物核验。`/home/penghongen/Feedback/Pocket_Plus/allocations/358384/after_lock_358384` 继续保留，`kill_lock_358384` 不存在。未经用户新授权，不删除 `try_lock_358384` 或 `after_lock_358384`，不释放或复用该 H100 allocation。pdb-centric-v1 与 occurrence-centric 的运行职责已经结束，不再启动新执行。

Job `367332` 与 `367411` 的第 2 次执行共同使用 release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_766bf257504e/Pocket_Plus`。A800 launch 是 `/home/penghongen/Feedback/Pocket_Plus/launches/367332/unet_c1_job367332_20260903T004831_a2`；A100 launch 是 `/home/penghongen/Feedback/Pocket_Plus/launches/367411/unet_c1_job367411_20260903T005135_a2`。Job `358384` 的推理是第 2 次执行，使用 release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus` 与 launch `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260903T161305_a2`。

## Completed

- 推理代码、配置、正式 shell、契约和测试已完成两轮全面审查及窄口径复核。用户最终要求只严格返工相对原始提交新增的函数和类；该范围已经批准。
- 实现端点 `92b4cdbab970d0bc975acb65eee2bff67ee4dd90` 与学习端点 `43f86c5e3cbaba6c3949f9766f9ed5c9f0803d94` 的 tree 均为 `bd302b98c54c8500d3305caddb64edd389232d04`。正式 release 从 `Learn/CUMULATIVE` 生成。
- 首次正式前向发现 A800 batch 32 与 A100 batch 16 显存溢出。按用户事先授权将正式完整图 batch 降为 24/12，资源线程数和输出契约未改变；学习端两份推理测试为 `45 passed in 4.93s`。
- A800 已完成 calibration PDB `6bgi`，写出 64,078,005 字节的 `probability_map.npz` 与 `_COMPLETE`；A100 已完成 held-out PDB `9ter`，写出 79,378,570 字节的概率图与 `_COMPLETE`。两张 GPU 利用率均为 100%，第 2 次执行没有新增 traceback。
- occurrence-centric 已完成 held-out 清单要求的 179 个 probability、179 个 F1 blobs、179 个逐 PDB evaluation NPZ 与 179 行 JSONL；集合完全一致且全部汇总指标为有限值。
- Job `358384` 于 2026-09-03 15:46 正常结束训练。最终最高完整 checkpoint 为 `TOP_epoch_03_score_0.5957.ckpt`；它与 `BEST.ckpt`、`last.ckpt` 均为 499,678,098 字节且 SHA-256 均为 `8e8bd068ebbeb8b49db3ddd40c6c485bad322e75f3308b39298b8cea7bec3a`。
- 2026-09-03 16:13，Job `358384` 的动态命令已改为极简 pdb-centric-v2 正式入口，并只删除 `try_lock_358384` 启动推理。16:18 时 calibration 已完成首个 PDB；H100 利用率 100%、显存 79,478/81,559 MiB，没有 traceback 或 CUDA OOM，`after_lock_358384` 保留。
- 2026-09-03 17:40，pdb-centric-v1 第 2 次执行成功。held-out 清单、probability、F1 blobs、逐 PDB evaluation NPZ 与 JSONL 各为同一组 179 个 PDB，全部汇总浮点指标有限；汇总与 JSONL 的 SHA-256 分别为 `97a70ee8da6c03b798db182accaac88a570a34fa937b799f638cc0bbf18d3334` 与 `303e1d12a871df396e963fcc36dba3c7cfc59c86f7ef91217e6050f0fc7ca941`。Job `367332` 已停在 `try_lock`，两把锁保持原状。
- 2026-09-03 21:18，pdb-centric-v2 的 100-PDB calibration probability、F1 blobs 与 basic 选参完成并通过集合闭合核验。F1 semantic threshold 为 `0.3990478515625`，F1 basic 为 score threshold `0.5232318639755249`、prefilter 8、min voxels 21；参数文件 SHA-256 分别为 `1cc4fd9dabfbf304b9158501cda6276a10ee1f8dccf6b247afe54d9bfbf90425` 和 `3f6c1b12d4a420db7425039b565c76ebcd7c49e28a7a87caff7f5b2a27be9db6`。
- 2026-09-04 06:03，pdb-centric-v2 完成 held-out 固定评估并安全进入 `try_lock_358384`；06:06 的最终门控确认清单、probability、F1 blobs、逐 PDB evaluation NPZ 与 JSONL 各包含同一组 179 个 PDB，全部汇总浮点指标有限。汇总与 JSONL 的 SHA-256 分别为 `da0c4302cb05a2f73c8d5079d16eab2b29507937d76279368fc4c5f1f03bd362` 与 `0ff2d75b896c16534f856beae75b03ae530d81b1ebbbb815adda1db3c6877518`。第 2 次执行没有 traceback 或 CUDA OOM。
- 三模型对照已写入执行记录。以冻结的 F1 blobs+basic 参数为准，pdb-centric-v2 的 semantic micro/macro F1 为 `0.552666440/0.394437186`，semantic micro/macro PRAUC 为 `0.501497242/0.401618043`，均为三者最高；coverage 0.3 与 one-to-one 0.3 的 macro F1 也最高。occurrence-centric 在部分候选排序 PRAUC 上仍略高，因此没有把结论扩大为所有 PRAUC 全面占优。

## Decisions

- occurrence-centric 复用 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950` 中既有 calibration、F1 semantic 和 F1 basic 参数，只补 held-out test。
- pdb-centric-v1 使用 Job `356953` 的 `TOP_epoch_01_score_0.4918.ckpt`，在 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts` 独立 calibration 和测试。
- pdb-centric-v2 使用 Job `358384` 的显式固定 checkpoint `TOP_epoch_03_score_0.5957.ckpt`，在 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts` 独立 calibration 和测试；不再运行时动态选择 checkpoint。
- 三模型比较只使用 `objective_beta=1` 的 blobs+basic，不使用 Gaussian 或 centered 打分。
- 正式命令保持为 `unet_c1_sampling_comparison.sh` 加单个模式参数；探测和门控命令不混入正式入口。

## Open Questions

- 当前没有待用户裁决的科学口径，也没有未完成的推理或评估步骤。

## Next Actions

1. 保留 Job `358384` 的 `try_lock_358384` 与 `after_lock_358384`；只有用户明确授权后才可释放或复用 H100。
2. 后续比较直接读取执行记录与三个正式输出根，不重新计算已经冻结的 calibration 或 held-out 产物。

## Files To Reopen

- `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`
- `CLAUDE/memory/handoffs/2026-09-04-unet-c1-pdb-centric-v2-held-out-complete.md`
