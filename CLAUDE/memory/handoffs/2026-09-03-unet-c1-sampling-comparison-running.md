# Handoff：unet_c1 三模型 × 两测试集收口完成

Date: 2026-09-04

## Current State

occurrence-centric、pdb-centric-v1 与 pdb-centric-v2 的既定 Stage1 calibration-test 流水线和六套正式汇总均已完成。三者都已对 `/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json` 的同一组 179 个 PDB 生成 probability、F1 blobs 和逐 PDB evaluation NPZ；`test_1.json` 是其中按原顺序筛出的 149-PDB 严格子集，边界为 `1 < ligand occurrence 数 < 100`，没有重复执行模型前向、blobs 或 evaluate。

三套 `test_0` canonical JSON/JSONL 已只增加新版 P/R 字段，旧字段逐项零漂移；三套 `test_1` 已发布全局 JSON、逐 PDB JSONL 与 provenance。服务器数据发布及独立验收均完成。Job `358384` 当前已不在队列中，其 allocation 目录只保留 `out/err`；此前完成并验收的训练、calibration 与 held-out 产物未受影响，也没有待管理的资源锁。

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
- Pocket Plus 的 `aggregate_stage1_metrics()` 已在 unstaged 工作区最小增加 semantic micro/macro P/R 与 coverage、one-to-one 三档 macro P/R；CLI、`pipeline.py`、既有 F1/F2/PRAUC/top-K 口径未变。最终定向复跑为 `40 passed in 11.34s`。
- 只读回填门控 Job `368996` 使用 16 CPU，于 20 分 50 秒 `COMPLETED/0:0`；三模型各 179 个 PDB 均通过 canonical 旧字段零漂移门控。
- 正式回填 Job `369051` 使用 16 CPU，于 20 分 29 秒 `COMPLETED/0:0`；原始 test_0 JSON/JSONL 已备份到 `/storage/penghongen/tmp/unet_c1_series_closure_20260904/preimage/`，正式文件通过同目录临时文件与 `os.replace` 原子发布。
- 独立验收 Job `369081` 使用 16 CPU，于 1 分 57 秒 `COMPLETED/0:0`；重新哈希 1,799 个唯一输入文件，并核验 179/149 行及顺序、严格 occurrence 子集、P/R-F1、macro、top-K、备份、代码、输出和临时文件残留。
- 任务级文档来源为 `/storage/penghongen/tmp/unet_c1_series_closure_20260904/documentation_source.json`，SHA-256 为 `ed49ae3d5cb60a2b312e9e4036c56df85b6b63fc58f5ffd870c956051fec19f9`。三份 test_1 provenance 保存完整输入清单与输出哈希。
- 八份待审 Markdown 已写入 `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\`；它们与 Pocket Plus 指标契约改动都保持 unstaged，尚未同步服务器。
- 八份文档通过确定性渲染、336 个主指标边界、micro F1、top-K、LF 与占位符门控；独立文档、科学逻辑和 Git 布局窄审均批准。科学审查另核对六套主表的 672 个显示值与三模型分层、Spearman 表的 636 个显示值，并从服务器 canonical JSONL 和身份文件独立重算一致。

## Decisions

- occurrence-centric 复用 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950` 中既有 calibration、F1 semantic 和 F1 basic 参数，只补 held-out test。
- pdb-centric-v1 使用 Job `356953` 的 `TOP_epoch_01_score_0.4918.ckpt`，在 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts` 独立 calibration 和测试。
- pdb-centric-v2 使用 Job `358384` 的显式固定 checkpoint `TOP_epoch_03_score_0.5957.ckpt`，在 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts` 独立 calibration 和测试；不再运行时动态选择 checkpoint。
- 三模型比较只使用 `objective_beta=1` 的 blobs+basic，不使用 Gaussian 或 centered 打分。
- 正式命令保持为 `unet_c1_sampling_comparison.sh` 加单个模式参数；探测和门控命令不混入正式入口。
- `test_1` 是 `test_0` 的派生保序子集，不能作为独立重复实验；它只保存汇总 JSON/JSONL/provenance，不复制 probability、blobs 或逐 PDB evaluation NPZ。
- 用户审阅前，不提交 Pocket Plus 的指标契约改动，不改写既有推理提交，也不把八份 AdaLigand 文档同步服务器。

## Open Questions

- 当前没有未完成的计算或科学口径阻塞。唯一待办是用户审阅本地代码、README、测试与八份收口文档。

## Next Actions

1. 请用户审阅 `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\` 的八份 Markdown；审阅前保持 unstaged 且不上传服务器。
2. 请用户审阅 Pocket Plus 的 `src/inference/evaluation.py`、`src/inference/README.md` 与 `tests/inference/test_stage1_v3.py`；批准后再按双线历史要求融入既有推理提交。
3. 后续比较直接读取三个正式输出根及 test_1 provenance，不重新计算已经冻结的 calibration、probability、blobs 或逐 PDB评估事实。

## Files To Reopen

- `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`
- `src/inference/evaluation.py`
- `src/inference/README.md`
- `tests/inference/test_stage1_v3.py`
- `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\总の说明.md`
- `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\总の结果.md`
