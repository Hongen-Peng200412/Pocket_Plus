# Handoff：Find_1 PDB-centric-1 节点故障恢复与 A800 替代任务排队

Date: 2026-09-06

## Current State

Find_1 PDB-centric-1 的 H100 Job `366071` 遭遇 hnode02 重启。Slurm 自动恢复后的 allocation 只有 CPU TRES，没有可用 CUDA 设备；自动 attempt a1 因 FlashAttention 收到 CPU 张量失败。用户于 2026-09-07 09:24:27 +08:00 主动取消该无 GPU allocation；Slurm 最终状态为 `CANCELLED by 1351`，主 Job `ExitCode=0:0`，包装器于 09:24:47 清理 `try_lock`、`after_lock` 和动态命令。该取消不是新的运行故障，也不是本守护执行的资源动作。旧运行和 checkpoint 均未改写。

替代 Job `371591` 已于 2026-09-06 23:22:07 +08:00 提交，申请 nvlink 分区的 A800×1、CPU×16，QOS 为 `h200g4`，无时限，同时启用 `pre_hold` 与 `after_hold`。2026-09-07 09:31:26 时状态为 `PENDING (Resources)`；排队阶段没有 release、launch、训练目录、W&B run 或锁文件。最终恢复实现已经部署到隔离任务根。用户已接受字面最后 checkpoint 缺失 4 个待累积 microbatch 梯度与完整 RNG 状态所导致的一次性非逐位偏差；Job 获得资源并通过资源、文件与动态命令核验后，可以直接删除本任务自己的 `pre_lock_371591` 启动训练。

其余活动训练保持正常：

- Job `368455`：hnode01 的 H100×2、CPU×64，Find_1 PDB-centric-2；2026-09-06 21:23 写出 PRAUC `0.611733` 的 checkpoint 后继续训练，未见错误。
- Job `366277`：gnode09、gnode10 各 A800×1、CPU×17，Find_1 历史续训；2026-09-06 13:27 写出 PRAUC `0.649774` 的 checkpoint 后继续训练，未见错误。

## Completed

- 权威恢复源固定为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_1/Find_1-pdb_centric_1____Find_1_job366071_20260901T102343_a11_pdb_centric_1/checkpoints/last.ckpt`，SHA-256 为 `bdaf83e3183d368d9569b22fefc84ce1c3bb781be88048f2628010899d707b5a`。
- checkpoint 只读审计保存于 `/storage/penghongen/tmp/job366071-node-fail-20260906/last_checkpoint_state.json`，SHA-256 为 `ee2f59d564479e7cceefb8d6277a5c0e6d0590c50cd347ba07098990300499db`。
- 精确边界为每个 rank 共 114,309 个训练 batch，已完成 114,300，恢复 epoch 尚余 9；此前 12 次 validation 均完整，每次 1,250 batch，下一次 validation 在新 epoch 的第 9,525 个训练 batch 运行。
- 恢复实现位于 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra` 的 `codex/find1-pdb-centric-1-recovery`。实现加入完整 Lightning checkpoint 恢复、sampler 跳过、完成态 validation 防重放、plateau 幂等键及 ModelCheckpoint 路径迁移。
- 逻辑审查把 a11 `config.yaml` 与恢复配置逐字段比较；排除有意新增的恢复字段和新运行身份后，差异数为 0。规范审查完成一轮全面检查和一次窄口径复核，未留下阻断项。
- 相关本地回归为 `34 passed`；Python 编译、一次性 shell 语法和差异格式检查通过。`tests/test_adaligand_stage1_configs.py` 的既有 PDB-centric-2 workers 断言仍期望 24，而用户当前正式配置为 30；该单项失败与本恢复实现无关，本轮未修改。
- 用于排队的隔离任务根为 `/home/penghongen/Feedback/Pocket_Plus/task_roots/find1-pdb-centric-1-recovery-20260906/Pocket_Plus`。初始归档 SHA-256 为 `fb8089eaa4cc820ab657e559ae3169abc983dfba819cabfeaf1cb0fd4b38dee2`；当前一次性 `tmp/find1_job366071_recovery/run.sh` SHA-256 为 `03158133fd534b9677c18cdc579826887eab790498e811db1e012e69ece69181`。
- 恢复实现已收成单一真实实现提交 `e91721fcfa2f3b0fd2582cb321f7f559260f6eef`，提交说明为 `fix: 保证 Find_1 历史 checkpoint 恢复幂等`。提交前使用 `D:\Anaconda\envs\Pocket_Plus_windows\python.exe` 复跑同一组回归，结果仍为 `34 passed`；规范审查指出的最后一处 DDP 测试 Docstring 表达已通过 amend 融入同一提交，定向测试为 `1 passed`。此前短暂部署的 `3f6bbce91ad46b7a12e6be0d065fec96115377d1` 从未启动训练，已被该提交取代。
- 最终提交归档已部署为 `/home/penghongen/Feedback/Pocket_Plus/task_roots/find1-pdb-centric-1-recovery-20260906/Pocket_Plus/deployments/implementation_e91721fcfa2f3b0f.tar.gz`，大小 `2,665,339` bytes，SHA-256 为 `df8295ba26cce9c071950de9752f000e1755e8e0896c453fbd51d8d3dadfafcb`。归档在服务器校验通过后覆盖展开到同一隔离任务根；共享 `/home/penghongen/My_Project/Pocket_Plus` 未改写。前一归档 `implementation_3f6bbce91ad46b7a.tar.gz` 只作为 amend 前部署证据保留，不得用于正式 release。
- 最终服务器文件中，`tests/test_find1_training_resume.py` SHA-256 为 `51ff0cbd3fe6166679b9c8c2dcb675d5ce1603364c5cc4c4f8db66fba68c0c7d`；其余四个提交文件的 SHA-256 与 amend 前相同。launcher 仍为 `03158133fd534b9677c18cdc579826887eab790498e811db1e012e69ece69181`。

## Decisions

- 旧 Job `366071` 已由用户取消；本守护不重提该 Job，也不把用户资源处置归因为故障。
- A800 Job `371591` 使用 CPU×16；一次性启动脚本按 `SLURM_CPUS_PER_TASK-1` 设置 DataLoader workers，因此本次为 15。workers 调整只适配资源，不改变科学契约。
- 恢复保持 a11 的模型、Dataset、损失、batch 6、全局 batch 48、`val_per_epoch=12`、学习率、plateau、分组梯度裁剪和 `save_top_k=30`。
- 初始隔离归档只用于尽快建立排队身份，不是最终 release。通过测试的最终提交 `e91721fcfa2f3b0fd2582cb321f7f559260f6eef` 已部署到同一隔离任务根；allocation 执行器在实际放行时才冻结正式 release 和 launch。
- 普通 validation、epoch 和常规 checkpoint 不更新 handoff。后续 handoff 只在替代训练正式启动、故障修复、训练完成或资源交接等事件发生时创建。

## Accepted Recovery Boundary

源 checkpoint 位于 8 个 microbatch 梯度累积周期的第 4 个 microbatch 之后，但 Lightning 不保存参数 `.grad`。因此恢复时会丢失此前 4 个尚未执行 `optimizer.step` 的 microbatch 梯度，第一次新 optimizer step 只能使用余下 4 个 microbatch。源文件也没有 Python、NumPy、torch 和 CUDA 的完整随机状态，随机旋转、随机 recycle 次数和 dropout 不能逐位续接。用户于 2026-09-07 明确接受这两项一次性非逐位偏差，以尽快完成训练；后续仍须保持模型、optimizer、scheduler、callback、sampler 位置、科学配置和下一次完整 validation 不变。

## Next Actions

1. 监视 Job `371591`；取得 allocation 后核对 A800、CPU16、`pre_lock`、最终归档与部署文件哈希、Job 身份和动态命令，再只删除 `pre_lock_371591`。
2. 启动后核对 checkpoint SHA、`global_step=14287`、LR `5e-5`、plateau bad epochs 0、`p_best=0.2802734375`、9 个 epoch 尾段 batch、无伪 validation，以及下一次完整 1,250-batch validation。
3. 固定醒来检查 Jobs `368455`、`366277` 和 `371591`；稳定时使用多个 300 秒命令组成 60 或 90 分钟静默等待，不使用 heartbeat。Job `366071` 已由用户取消，不再作为活动轮询目标。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-training-global-watch.md`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\src\train.py`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\ops\find1_historical_resume\README.md`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\ops\find1_historical_resume\rebase_checkpoint.py`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\tests\test_find1_training_resume.py`
- `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\tmp\find1_job366071_recovery\run.sh`
