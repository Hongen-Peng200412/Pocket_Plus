# Handoff: Find_1 双节点 A800 历史续训已由用户暂时停止

Date: 2026-09-12

## Current State

Find_1 历史续训 Job `366277` 已由用户主动执行 `scancel 366277` 暂时停止，未来可能继续训练。Slurm 记录训练步骤于 2026-09-12 12:07:29 +08:00 被取消，主 Job 于 12:10:40 +08:00 结束；终态为 `CANCELLED by 1351`、`ExitCode=0:0`。原 allocation 使用 `gnode09,gnode10` 各 A800×1、CPU×17；资源现已释放，`pre_lock_366277`、`try_lock_366277` 和 `after_lock_366277` 均不存在。

这不是训练故障。本守护没有执行取消、删锁或其他资源控制，只完成只读核验和恢复端点固化。Job `366277` 不再是活动轮询目标；只有用户明确要求续训后，才重新申请资源、生成新的 release/launch 并启动。

## Durable Resume Boundary

若未来目标是从本次取消前最后一个可持久位置延续轨迹，默认恢复源为：

`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1/checkpoints/USER_SCANCEL_20260912T121040_epoch_00_step_37248.ckpt`

- 大小：1,466,253,917 bytes；
- SHA-256：`191795e33432110dd5ea2137143069484fcb0bc209fe33afbc11b614cac7d405`；
- 与取消时的 `checkpoints/last.ckpt` 逐字节一致；
- epoch 0、`global_step=37248`；
- 每个 rank 已处理 148,995 个训练 microbatches；
- 最近一次 validation 为每个 rank 完整 551/551 batches，`is_last_batch=true`，全局 `num_gt=14,802,569`；
- optimizer 学习率为 `2.0000000000000054e-6`；
- `LearningRateReductionStopper.lr_reduction_count=2`；
- plateau 状态为 `best=0.6733377575874329`、`num_bad_epochs=3`、`validation_index=30`、`last_stepped_validation=[37248,0]`；
- 候选阈值缓存为 `p_best=0.3935546875`；
- 当前 validation 的配体体素 PRAUC 为 `0.6744711995124817`，ModelCheckpoint 历史最佳为 `0.675399661064148`。

对应的固化 validation 摘要为：

`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1/validation_diagnostics/USER_SCANCEL_20260912T121040_epoch_00_step_37248_summary.json`

大小为 1,993 bytes，SHA-256 为 `d72bcb06c647ddbb4d513f4ad4bf0bdcb3619dfc278e60afe25efdf5df2cd73f`；它与取消时的 `validation_diagnostics/epoch_000000/summary.json` 逐字节一致。

## Non-durable Progress At Cancellation

W&B run `pencounkdual-111/AdaLigand_Stage1/9kj96nld` 最后写入的摘要时间为 11:25:14 +08:00，记录 `global_step=38375`、epoch 0、学习率 `2e-6`。它比 checkpoint 的 `global_step=37248` 多 1,127 个 optimizer steps，但这些参数更新没有进入可恢复 checkpoint；未来不得把 W&B 的 38375 当作恢复位置。

本运行使用 `accumulate_grad_batches=4`。checkpoint 的 rank-local 训练计数 148,995 满足 `148995 mod 4 = 3`，说明循环状态位于累积周期的第 3 个 microbatch 之后；普通 Lightning checkpoint 不保存参数的内存 `.grad`。未来恢复前必须重新审计这个一次性优化动力学边界，不得宣称与未中断轨迹逐位等价，也不得用修改 validation、patience、阈值或学习率掩盖该边界。

## Frozen Run Identity

- 原始历史源 checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt`，SHA-256 `d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c`。
- 历史恢复实现提交：`082905874d88e37799232f1a5d383510356f9a4a`。
- 部署归档：`/home/penghongen/Feedback/Pocket_Plus_Find1/uploads/find1-historical-resume-082905874d88.tar`，SHA-256 `b3e6345dc09aff527eb7f0b177ece5f4d2fa8fab4cf51bc953d982cbfbea0c5d`。
- 隔离项目根：`/home/penghongen/Feedback/Pocket_Plus_Find1/task_roots/find1-historical-resume-082905874d88/Pocket_Plus`。
- runner release：`/home/penghongen/Feedback/Pocket_Plus_Find1/releases/Pocket_Plus_Find1_7b0d38481c0f/Pocket_Plus_Find1`。
- attempt a3 launch：`/home/penghongen/Feedback/Pocket_Plus_Find1/launches/366277/Find_1_job366277_20260902T182829_a3`。
- `launch.json` SHA-256：`b4b269f47110f350f7b4436fd4c5b4a5428a9f4f104c1bf0f08aab77d33e8670`。
- launch 内 `run_cmd.sh` SHA-256：`ee1b76a946618dfa1bc10f56b5d6351db0deaa71bbbfae8cefcbac7e1b24cef3`。
- 正式训练与全部实验产物根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1`。
- W&B 本地目录：上述产物根的 `wandb/run-20260902_183724-9kj96nld/`。
- allocation 标准输出：`/home/penghongen/Feedback/Pocket_Plus_Find1/allocations/366277/out`，SHA-256 `8f5c0749089a0a2aa757bb6d8e9ac7fb34c7e3551933168ff333d7edb4298070`。
- allocation 错误输出：`/home/penghongen/Feedback/Pocket_Plus_Find1/allocations/366277/err`，SHA-256 `61d23303d6b683730de0c1670524e68a0491e29da0501fd547084ab2b43caebf`。

## Future Resume Requirements

1. 先由用户重新授权续训和资源规格；当前不得自动重提 Job。
2. 默认从固化的 `USER_SCANCEL_20260912T121040_epoch_00_step_37248.ckpt` 开始，不使用 W&B step 38375，也不继续使用较早的 a1/a2 污染 checkpoint。若用户届时明确选择其他 checkpoint，须先重新核对其模型、优化器、调度器、循环与 validation 状态。
3. 在既有历史续训隔离实现上做最小适配：固定新源路径与 SHA-256，并把 epoch 0、每 rank 148,995 个已处理训练 microbatches、完整 validation 551/551 写入新的 rebase manifest。
4. 保持模型、数据、validation selection、optimizer、scheduler、`expected_threshold=0.003`、`patience=3`、学习率、全局梯度裁剪 0.5 和科学配置不变；workers 可以按新资源适配。
5. 启动前验证模型、optimizer、scheduler、candidate cache 和 callback 状态等价；明确记录累积周期中 3 个待累积 microbatches 不可恢复的影响。
6. 启动后首先确认不会立即重放已完成 validation；下一次真实 validation 必须每 rank 完整处理 551 batches，并且 plateau、candidate cache、ModelCheckpoint 和停止计数只能推进一次。
7. 新任务必须使用独立 release、launch 和输出根，保留本次 a3 目录及固化 checkpoint 只读。

## Other Active Work

- 双 H100 Find_1 PDB-centric-2 Job `368455` 仍由当前守护负责；其第二次实际学习率衰减端点已固化，训练继续运行。
- A800 Find_1 PDB-centric-1 Job `371591` 的 `kill_lock`、重启或结束可能由另一个 AI agent 执行；这是正常协作。当前守护只读观察其归属，不操作、不把正常接管记录为故障。
- 已完成或由用户处理的 Jobs `366071`、`350305`、`358384` 不得重新纳入资源操作。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `CLAUDE/memory/handoffs/2026-09-02-find1-a800-timeout-recovered.md`
- `ops/find1_historical_resume/rebase_checkpoint.py`
- `训练与运行/sh/Find_1.sh`
