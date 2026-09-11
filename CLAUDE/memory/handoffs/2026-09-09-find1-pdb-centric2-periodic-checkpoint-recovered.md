# Handoff: Find_1 PDB-centric-2 周期 checkpoint 死锁已修复并恢复训练

Date: 2026-09-09

## Current State

双 H100 Job `368455` 的 attempt a1 在 epoch 2 末尾写周期 checkpoint 时发生 DDP collective 顺序死锁。提交 `fa01b8d1fbf4a43c1d3c34c7018a2cd7882f4bb1` 已按最小范围修复：所有 DDP rank 在到达周期时都调用 Lightning 的 `trainer.save_checkpoint()`。修复版 attempt a2 已从 a1 的可靠 `last.ckpt` 恢复，成功写出新的 `PERIODIC_epoch_02.ckpt` 并继续进入 epoch 3；07:46 的 W&B 摘要为 `global_step=27326`、学习率约 `5e-5`，没有新错误。Job 仍在 hnode01 使用 H100×2、CPU×64，只保留自己的 `after_lock_368455`。

A800 PDB-centric-1 恢复 Job `371591` 与双节点 A800 历史续训 Job `366277` 同期继续正常训练。旧 Jobs `366071`、`350305`、`358384` 已关闭，不再轮询或执行资源动作。

## Completed

- a1 故障定位：NCCL 序号 `2565749` 上，rank 0 为元素数 1 的 `ALLREDUCE`，rank 1 为元素数 8,205 的 `BROADCAST`，watchdog 在 1,800,000 ms 后终止进程。根因是 `PeriodicCheckpointSaver.on_train_epoch_end()` 只让 rank 0 进入包含分布式同步的 `trainer.save_checkpoint()`。
- a1 运行根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job368455_20260903T233745_a1_pdb_centric_2`；W&B 为 `pencounkdual-111/AdaLigand_Stage1/g7ul4r8p`。
- 恢复源：a1 运行根的 `checkpoints/last.ckpt`，SHA-256 `8e118679583ffdce09c916cb4649c0f04e362a31a662dbddd4cd38b69bc146b7`。边界为 epoch 2、`global_step=27262`、每 rank 训练 batch 36,344、validation 543/543；学习率 `5e-5`、plateau best `0.6386887431144714`、`num_bad_epochs=2`、实际衰减计数 0。
- 不采用 a1 的 `PERIODIC_epoch_02.ckpt`；其 SHA-256 为 `490534ba9969bfeefee8b99b7011f5ca5544423501211fb35e5335e59ec8bbf3`，由故障的不对称保存调用产生，只保留为证据。
- 实现提交：分支 `codex/find1-pdb-centric-1-recovery` 上的 `fa01b8d1fbf4a43c1d3c34c7018a2cd7882f4bb1`。65 项相关测试、Python 编译、shell 语法和差异检查全部通过。
- 部署归档：`/home/penghongen/Feedback/Pocket_Plus/task_roots/find1-pdb-centric-2-recovery-20260909/upload/implementation_fa01b8d.tar.gz`，SHA-256 `f1da030f1879a48aab946b458101f2e3a6ee943f2bcad479b9d7620dbfbcd4b9`。隔离项目根为同级 `Pocket_Plus/`；共享 `/home/penghongen/My_Project/Pocket_Plus` 未修改。
- attempt a2 launch：`/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260909T071535_a2`。`launch.json` 与 `run_cmd.sh` SHA-256 分别为 `0c2d075dd399cdab2b29d051d23058c67792e303ee887b3fe4bc16a172574644`、`141fd66cba79cf8d980fa42deedf83da51739ca34ac51fcd33f895542077b69b`。
- attempt a2 全部新增实验产物根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume`；checkpoint 位于其 `checkpoints/`；W&B 本地目录为其 `wandb/run-20260909_072823-0bydo3jg/`，在线身份为 `pencounkdual-111/AdaLigand_Stage1/0bydo3jg`。
- `resume_state.ckpt` SHA-256 为 `9c6fccba6eddf00e8a6332b338387d2ebb8b4a3d2785dba1608bbd6b1dab546a`；`resume_state_manifest.json` SHA-256 为 `b43afd7626108484d37687c4cd3d2eaa5843f1a501ac2be267cfe274443509eb`，并记录 24 个被 ModelCheckpoint 状态引用的历史 checkpoint 副本。
- 07:15:11 的唯一放行动作是删除 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_368455`；`after_lock_368455` 未删除。07:30:46，新运行成功写出大小 1,466,252,642 bytes 的 `PERIODIC_epoch_02.ckpt` 后继续训练，证明死锁修复生效。

## Decisions

- 从可靠 `last.ckpt` 恢复会重复处理 epoch 2 末尾 7 个 microbatch。checkpoint 不保存待累积梯度，也不能保证随机状态逐位续接；用户接受这一次性非逐位偏差，以优先完成训练。不得把 a2 描述为与不中断训练逐位等价。
- 本次没有改变模型、数据、损失、optimizer、scheduler、学习率、batch、workers、validation 或停止条件。PDB-centric-2 继续使用双 H100、CPU×64、每 rank 30 workers、每卡 batch 6、全局 batch 48 和 `stop_after_lr_reductions=3`。
- 普通 step、validation、epoch 和 checkpoint 不建立 handoff。仅在新故障及修复、正式重启、训练完成、资源交接或科学契约变化时更新执行记录与 handoff。
- 未经用户新授权，不删除三个活动 Job 的 `after_lock`，不取消或释放资源。

## Next Actions

1. 固定醒来检查 Jobs `371591`、`368455` 和 `366277` 的 Slurm 状态、训练推进、错误信号和各自控制锁；不再检查已经关闭的旧 Job。
2. 重点确认 `368455` a2 持续通过后续 epoch 末周期保存；如果再次出现 NCCL、OOM、DataLoader 或非有限值错误，在不改变科学契约的范围内修复并记录。
3. 训练稳定时以多个独立 300 秒睡眠组成 60 或 90 分钟等待，不使用 heartbeat；无变化时不发送消息、不写日志。
4. 任一训练正常结束后验收最终 checkpoint、BEST/TOP、W&B 和停止条件，再按用户授权处理资源。
5. 所有训练结束后，完成实现分支与学习分支的 tree 等价核验、`Learn/CUMULATIVE` 推进和文档收口。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\src\train.py`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\训练与运行\sh\Find_1_pdb_centric_2_resume.sh`
- `/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260909T071535_a2`
- `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume`
