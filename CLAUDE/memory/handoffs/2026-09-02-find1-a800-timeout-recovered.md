# Handoff: Find_1 A800 validation 恢复边界已修复，a3 正常训练

Date: 2026-09-02

## Current State

Find_1 两项训练均在运行，未经用户新指令不得释放资源或操作其锁：

- A800 历史续训：Job `366277`，`gnode09,gnode10`，每节点 A800×1、CPU×17、每 rank 16 workers；修复版 attempt a3 已从原始 checkpoint 跨过完成态 validation 边界，并从 `global_step=11287` 推进到至少 `11291`。W&B run 为 `9kj96nld`。
- H100 PDB-centric-1：Job `366071`，`hnode02`，H100×1、CPU×32，正式 attempt a11；W&B run `wi4gcvcs` 的最新摘要已到 `global_step=3668`，第三次 validation 的 `voxel_ligand_PRAUC=0.46331578493118286`。
- 两个 Job 的 allocation 内 `after_lock` 均存在；根级 `try_lock` 与 allocation 内 `kill_lock` 均不存在。

## Critical Correction

先前把 A800 a1 的主要问题记录成下一次 validation 的 1,800 秒 NCCL watchdog 超时，只解释了 a1 后半段停止，没有识别恢复瞬间已经发生的伪 validation。原始 checkpoint 的 Lightning validation 进度为 `current.completed=551`、`is_last_batch=true`；双卡、物理 batch size 6 时，每个 rank 的完整 validation 正是 551 batches。

- a1 在载入的 551 上只执行一个 batch，写回 552；`global_step` 仍为 `11287`，`num_gt=44,405`，PRAUC 为 `0.571532`。`num_bad_epochs` 从 2 变为 3，候选 `p_best` 从 `0.1826171875` 变为 `0.1572265625`。
- a2 从 a1 的受污染 checkpoint 再执行一个 batch，写回 553；`global_step` 仍为 `11287`，`num_gt=44,405`，PRAUC 为 `0.5718138217926025`。学习率从 `5e-5` 降为 `1e-5`，学习率衰减停止器从 0 变为 1，候选 `p_best` 变为 `0.130859375`。
- 原完整 validation 的 `num_gt=14,802,569`、`voxel_ligand_PRAUC=0.618439674`。a1/a2 均不得续训，其目录保持只读故障证据。

根因有两项：Lightning 2.2.5 在恢复点恰好也是 validation 调度边界时重新进入已完成的 validation；项目原 plateau 去重键包含重启后自增的 `validation_index`，无法识别相同 `global_step` 的重复 validation。

## Frozen Source And Validation

唯一允许使用的源 checkpoint：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt`

SHA-256：`d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c`。源状态为 `global_step=11287`、每 rank 训练进度 45,150 batches、validation 551 batches、学习率 `5e-5`、`num_bad_epochs=2`、`p_best=0.1826171875`。

validation selection 没有修改：

`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection.npz`

它包含 200 个 PDB、3,305 个 bias BOX 和 3,305 个 context BOX，SHA-256 为 `91af9c01538e6da1a0a11d2109eeb28673b97cf4c89b1e7e813f7f2b843f5ac7`。

## Read-only Failure Evidence

- a1 输出根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260901T074338_a1_CPC1`；其 `last.ckpt` SHA-256 为 `7ef7eb5af9eb0a8e68c61ed9605bee7b7f6da440b967305b613511de3beeb117`。
- a2 输出根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T142603_a2_CPC1`；其 `last.ckpt` SHA-256 为 `55cc5c39462aa7a4ebcd05d7aaccff68baa2deeb2c1e351c4039711bd19f733c`，W&B run 为 `diieig3m`。

## Stop And Repair

2026-09-02 17:56:54 +08:00，在核对 Job、最新 launch 和三种锁后，只创建：

```bash
touch -- /home/penghongen/Feedback/Pocket_Plus_Find1/allocations/366277/kill_lock_366277
```

allocation runner 只停止 a2，17:57:24 恢复根级 `try_lock_366277`；Job、两节点 A800 allocation 与 `after_lock_366277` 均保留，没有触碰其他任务或资源。准确本地执行脚本是：

`C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\tmp\find1-stop-a800-a2-366277.sh`

修复只在历史续训工作树 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\find1_historical_resume` 的 `codex/find1-historical-resume` 上实现；提交为 `082905874d88e37799232f1a5d383510356f9a4a`。改动只包括完成态 validation 首次恢复保护、plateau 的 `(global_step, current_epoch)` 幂等键、源 SHA/循环边界门禁及对应测试。当前生产 Find_1 代码线没有吸收这些历史恢复逻辑。

本地定向测试为 60 passed；服务器定向测试为 16 passed。服务器以原始 1.466 GB checkpoint 验证 45,150/551 完成边界，并断言学习率、`num_bad_epochs` 与 `p_best` 保持指定原值。端到端测试覆盖同一 checkpoint 连续恢复两次，以及 1 节点×2 devices 与 2 节点×1 device 的等价结果。

## A800 Attempt a3 Identity

- 归档：`/home/penghongen/Feedback/Pocket_Plus_Find1/uploads/find1-historical-resume-082905874d88.tar`，6,604,800 bytes，SHA-256 `b3e6345dc09aff527eb7f0b177ece5f4d2fa8fab4cf51bc953d982cbfbea0c5d`。
- 隔离项目根：`/home/penghongen/Feedback/Pocket_Plus_Find1/task_roots/find1-historical-resume-082905874d88/Pocket_Plus`。
- deployment identity SHA-256：`11ff2299e833d5dc130df0affcba6034d335797fc7ae2d53db0158f54f922bbb`。
- launch：`/home/penghongen/Feedback/Pocket_Plus_Find1/launches/366277/Find_1_job366277_20260902T182829_a3`。
- 输出根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1`。
- W&B run：`9kj96nld`。
- `resume_state.ckpt` SHA-256：`eec9cb12570e8ed41ddd766a7203046b4d39d5c1fe1420f9129612a18c3b0825`。
- 动态命令与 launch 快照 SHA-256：`ee1b76a946618dfa1bc10f56b5d6351db0deaa71bbbfae8cefcbac7e1b24cef3`。

动态命令完整内容为：

```bash
#!/usr/bin/env bash
set -euo pipefail
export EXPERIMENT_FEEDBACK_ROOT=/home/penghongen/Feedback/Pocket_Plus_Find1
export FIND1_NUM_WORKERS=16
exec bash /home/penghongen/Feedback/Pocket_Plus_Find1/task_roots/find1-historical-resume-082905874d88/Pocket_Plus/训练与运行/sh/Find_1.sh
```

登录节点在 18:31:05 +08:00 执行：

```bash
rm -- /home/penghongen/Feedback/Pocket_Plus_Find1/allocations/try_lock_366277
```

截至 18:45:24，a3 已到 `global_step=11291`，`warmup_lr=5.000000000000013e-05`，训练总损失为 `0.2594981789588928`；W&B 摘要没有任何新 validation 字段，目录没有新的 validation 诊断或由 a3 写出的 checkpoint。这证明恢复瞬间先进入训练，没有重放伪 validation。

## Next Actions

1. 继续只读监视 A800 a3；下一次正常 validation 预计在再完成约 4,515 个 rank-local 训练 batches 后触发。验收必须看到每 rank 551 batches、`num_gt` 约 1,480 万，并核对 scheduler、候选缓存、checkpoint 和停止器只更新一次。
2. 同时只读监视 H100 a11 的第三次 validation checkpoint 与后续训练；没有故障时不操作资源或锁。
3. 稳定期使用 12 或 18 个彼此独立的 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待，不用 heartbeat；只在 validation、checkpoint、错误、任务结束或锁变化等明确事件写记录。
4. 历史实现分支已经只有一个本轮实质提交。执行记录、计划和本 handoff 统一纳入 `Learn/CUMULATIVE` 的日志提交，不与正式代码逻辑混合。
5. 两项训练正常完成后，才执行学习历史、tree 等价和 `Learn/CUMULATIVE` 收口；当前持久 goal 不应提前标记完成。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `训练与运行/sh/Find_1.sh`
- `ops/find1_historical_resume/resume_guard.py`
- `ops/find1_historical_resume/rebase_checkpoint.py`
- `tests/test_find1_historical_resume.py`
