# Find_1 真实受体训练第 6 片运行中

## 已完成的边界

- 第 5 片 275 个成员已经完成四阶段正式流程，并通过全量产物与数组契约验收；最终报告为 `/storage/penghongen/tmp/find1_real_receptor_train_shard_05_20260916/final_verification.json`，SHA-256 为 `2626a2e26193182ee85a1da007328d674d1f290c055fb6237d78aada5ba95537`。
- 第 6 片固定为用户片号 6、CLI `shard-index=5`。其 275 个唯一成员与第 1—5 片无交集，成员序列 SHA-256 为 `73e7ddf1cf621d97852d2065e66cf5efb8b213a711d86f60c7d3306cabc4ab9a`；preflight 位于 `/storage/penghongen/tmp/find1_real_receptor_train_shard_06_20260917/preflight.json`。
- 第 6 片代码已经按双线历史收口。`Learn/CUMULATIVE` 为 `9bbd2494ef54343967643b7d58c0809508207965`，实现端点和学习端点的 Git tree 都是 `d4c151a7129b872d83ac1e8810ca3352503613d7`；干净 Learn release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_268637d112a5/Pocket_Plus`。

## 当前运行状态

- Job `378587` 位于 `hnode02`，资源为 H100×1、32 CPU、QOS `h100g2`、不限时。attempt a3 的 launch 为 `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260917T135451_a3`。
- 正式命令只调用 `find1_real_receptor_train_shard_06.sh 6`；实际 probability 进程使用固定 checkpoint、真实受体训练清单、`--shard-count 50 --shard-index 5` 和 `stage1_v3_h100_32cpu.yaml`。
- 2026-09-18 07:27，275/275 个成员的 probability 与冻结阈值 F2 blobs 已完整落盘，正式入口已进入 centered；实际命令保留 `alpha=2`、`forward_min_voxels=8`、`--continue-on-blob-exceed` 和 CLI `shard-index=5`，严格错误为 0。
- 四锁执行器的生效 try lock 位于 `/home/penghongen/Feedback/PocketXMol/allocations/try_lock_378587`；它已为本次 attempt 删除。`/home/penghongen/Feedback/PocketXMol/allocations/378587/after_lock_378587` 始终保留，没有创建新的 kill lock，也没有使用 `scancel`。

## 后续动作

1. 按 60 或 90 分钟静默窗口守护，等待 probability 完成，再核验冻结 F2 blobs、centered 和 Gaussian score-only 的阶段切换。
2. 正式流程完成并停回生效的 `try_lock_378587` 后，对 275 个成员执行与第 5 片同口径的全量只读验收：冻结函数重算、候选顺序、offsets、来源体素、48³ 数组和 `score/selected` 必须全部一致。
3. 完成时更新第 6 片实验日志、共享总日志、映射、项目记忆与完成 handoff；不得扩大到第 7 片，不得触碰 `after_lock_378587`。

## 记录位置

- 第 6 片实验日志：`文档/exec_plan/2026-09-17_Find_1真实受体训练第06片推理.md`
- 共享总日志：`文档/exec_plan/2026-09-16_Find_1真实受体训练分片推理总日志.md`
- 计划映射：`文档/mapping/计划执行映射.md`
