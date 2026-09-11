# Handoff: Find_1 PDB-centric-1 A800 恢复训练已启动

Date: 2026-09-07

## Current State

Find_1 PDB-centric-1 的替代 Job `371591` 已在 gnode09 使用 A800×1、CPU×16 正式恢复训练。attempt a1 已从节点故障前 Job `366071` 的字面最后 checkpoint 恢复到 W&B `pencounkdual-111/AdaLigand_Stage1/gg8y6s0i`，22:16 左右推进到 `global_step=14291`，学习率保持 `5e-5`，训练损失有限。恢复后没有立即执行伪 validation，Job 当前仅保留自己的 `after_lock_371591`。

双 H100 PDB-centric-2 Job `368455` 与双节点 A800 历史续训 Job `366277` 同期继续训练；本轮没有操作它们的进程或锁。旧 Job `366071` 已由用户主动取消，不再是活动监视目标。

## Completed

- Slurm 权威身份核实为 Job `371591`、gnode09、A800×1、CPU×16、QOS `h200g4`、无时限。
- 放行前核对最终实现提交 `e91721fcfa2f3b0fd2582cb321f7f559260f6eef` 的部署归档、五个关键文件、一次性入口、源 checkpoint 和动态命令摘要；全部与已验收记录一致。
- 唯一启动动作是删除 `/home/penghongen/Feedback/Pocket_Plus/allocations/pre_lock_371591`；`/home/penghongen/Feedback/Pocket_Plus/allocations/371591/after_lock_371591` 始终保留。
- 正式 release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_752e383bc074/Pocket_Plus`。
- 正式 launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/371591/run_job371591_20260907T215604_a1`。`launch.json` SHA-256 为 `ff3c111b82e7d06c07bcb5a7e2aedefe8c7d05e58e6509254d0384550928ba09`，`run_cmd.sh` SHA-256 为 `24f215439894e619f6999c76179922528ba63113b3db98dc068c64ef15a548c8`。
- 新运行与全部产物根为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_1/Find_1-pdb_centric_1_resume_job366071____run_job371591_20260907T215604_a1_pdb_centric_1_resume`。
- 权威源 checkpoint 为旧 a11 运行根下的 `checkpoints/last.ckpt`，SHA-256 `bdaf83e3183d368d9569b22fefc84ce1c3bb781be88048f2628010899d707b5a`。
- 重建的 `checkpoints/resume_state.ckpt` SHA-256 为 `9309e34fa7fcda54124a5b26aa6c80397e513fa8de1bdc41bc997d2f094c37f2`；同目录 `resume_state_manifest.json` 记录 `global_step=14287`、已完成训练 batch 114300、完整 validation 1250/1250，以及 12 个被 top-k 状态引用的历史 checkpoint 副本。
- W&B 新运行目录为上述产物根下的 `wandb/run-20260907_220759-gg8y6s0i/`。首批摘要为 `global_step=14291`、`warmup_lr=5e-5`，没有 `val_*` 字段；GPU 进程核验确认 PID `33008` 已使用 Job 分配的 GPU 0。

## Decisions

- 用户接受源 checkpoint 缺少 4 个待累积 microbatch 梯度和完整 Python、NumPy、torch、CUDA RNG 状态造成的一次性非逐位偏差，以尽快完成训练。不得把该运行描述为与未中断轨迹逐位相同。
- CPU16 使用 15 个 DataLoader workers；workers 只适配资源，不改变科学契约。
- 源 checkpoint 的模型、optimizer、scheduler、callback、候选缓存和 sampler 位置保持不变。源审计值为 LR `5e-5`、plateau `num_bad_epochs=0`、候选 `p_best=0.2802734375`。
- 未经用户新授权，不删除 `after_lock_371591`，也不释放或取消任何活动资源。

## Next Actions

1. 固定醒来检查 Jobs `371591`、`368455` 和 `366277`；稳定状态使用多个 300 秒命令组成 60 或 90 分钟静默等待，不使用 heartbeat。
2. 监视 `371591` 的训练步骤、有限损失、GPU、W&B、错误和 checkpoint。新 epoch 第 9,525 个训练 batch 后，首次正式 validation 必须完整处理 1,250 batches。
3. 第一次完整 validation 后核对 LR、plateau `num_bad_epochs`、候选 `p_best`、一次性 validation 副作用和 checkpoint 路径；这是恢复修复的关键验收事件，应写入执行记录。
4. 普通 epoch、常规 checkpoint 和无异常轮询不新增 handoff；仅在故障修复与重启、任务提交或替换、训练完成、资源交接或科学契约变化时再建立 handoff。
5. 所有训练结束后，再完成实现分支与学习分支等价、`Learn/CUMULATIVE` 推进和最终文档收口。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `CLAUDE/memory/handoffs/2026-09-06-find1-pdb-centric1-node-recovery-a800-queued.md`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\ops\find1_historical_resume\README.md`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\tmp\find1_job366071_recovery\probe_371591_launch.sh`
