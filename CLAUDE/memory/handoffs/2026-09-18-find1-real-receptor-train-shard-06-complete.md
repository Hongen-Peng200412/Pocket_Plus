# Find_1 真实受体训练第 6 片完成

## 完成结论

- Job `378587` attempt a3 已完成固定训练第 6 片，即 seed `3407`、50 片定义中的 CLI `shard-index=5`。275 个唯一成员与第 1—5 片无交集，成员序列 SHA-256 为 `73e7ddf1cf621d97852d2065e66cf5efb8b213a711d86f60c7d3306cabc4ab9a`。
- 全部成员均完成 probability、冻结阈值 F2 blobs、centered 与冻结 Gaussian score-only。275 个 `F2_centered.npz` 均含 `score/selected`，正式 attempt 没有严格错误。
- 全量只读验收精确复算正式 blobs 和 Gaussian 分数，并核对候选顺序、四组 offsets、来源体素、48³ 数组和最终选择。11,652 个来源 blob、11,319 个 centered 候选、11 个超框候选和 8,159 个最终选择全部通过。

## 产物与证据

- 正式产物根：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train`
- 第 6 片 preflight：`/storage/penghongen/tmp/find1_real_receptor_train_shard_06_20260917/preflight.json`，SHA-256 为 `90762216b324d985f9ee07ed72cb5d28cd77ab753e736e4b45e4ea60ae9816d2`
- 最终验收报告：`/storage/penghongen/tmp/find1_real_receptor_train_shard_06_20260917/final_verification.json`，SHA-256 为 `bad622fa678b29e107d995ebbd708241814307a24351e4d54856a35f72b56cec`
- 正式 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_268637d112a5/Pocket_Plus`
- 正式 launch：`/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260917T135451_a3`
- 收口集合门控确认 train 根目录的 1,650 个直接成员恰好等于固定第 1—6 片并集，与第 7 片交集为 0。

## allocation 终态

- Job `378587` 仍为 RUNNING 并停在生效的 `/home/penghongen/Feedback/PocketXMol/allocations/try_lock_378587`。
- `/home/penghongen/Feedback/PocketXMol/allocations/378587/after_lock_378587` 保留；pre lock 和 kill lock 不存在。本阶段没有创建新的 kill lock，也没有使用 `scancel`。
- H100 显存占用 1 MiB、利用率 0%，没有残留推理进程。未经新授权不得删除 after lock 或启动第 7 片。

## 本地记录

- 共享总日志：`文档/exec_plan/2026-09-16_Find_1真实受体训练分片推理总日志.md`
- 第 6 片实验日志：`文档/exec_plan/2026-09-17_Find_1真实受体训练第06片推理.md`
- 计划映射：`文档/mapping/计划执行映射.md`
