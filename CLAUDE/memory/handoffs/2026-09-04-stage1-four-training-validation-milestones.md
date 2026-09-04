# Handoff：Stage1 四条训练线完成新一轮 validation

Date: 2026-09-04

## Current State

截至 2026-09-04 18:06 +08:00，Jobs `368455`、`366071`、`350305`、`366277` 均保持 `RUNNING`，各自只保留预期的 `after_lock`，没有 `try_lock` 或 `kill_lock`。Job `368455` 已完成第三次完整 validation、写出 `TOP` 与 `last` checkpoint，并在写盘后继续推进训练；其余三条训练线本轮没有新的 validation、checkpoint、错误或资源变化。最近 500 行日志未发现 traceback、CUDA OOM、NCCL、DataLoader worker 退出或非有限值错误。

Find_1 PDB-centric-2 Job `368455` 继续采用用户确认的 `stop_after_lr_reductions=3`。本次首个 checkpoint 仍处于学习率 warmup，实际学习率衰减次数为 0。若以后希望取得“实际衰减 2 次后停止”的等价端点，应保留并核验第二次实际衰减完成后写出的完整 checkpoint；当前不修改配置、不重启训练。

Job `358384` 的 pdb-centric-v2 推理已于 06:03 成功结束并写回 `try_lock_358384`。08:35:30，Slurm 把该 Job 记录为外部取消，`JobState=CANCELLED`、`ExitCode=143:0`；本守护没有执行取消、删除锁或任何资源写操作。08:36 时磁盘上暂时仍有 `try_lock_358384` 与 `after_lock_358384`，但 hnode01 的节点级 `AllocTRES` 已只剩 Job `368455` 的 64 CPU，说明 `358384` 不再持有 H100 allocation。09:41 再次检查时，Job 已不在队列，两把锁及 allocation 活动文件均已由调度包装器的退出清理删除。除非用户重新授权，不重新提交任务，也不接管空出的第三张 H100。

## Training Identities And New Milestones

| Job | 身份与资源 | 正式产物根 | 新 validation | checkpoint 与调度状态 |
| ---: | --- | --- | --- | --- |
| `368455` | Find_1 PDB-centric-2；hnode01，H100×2、CPU×64；W&B `g7ul4r8p` | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job368455_20260903T233745_a1_pdb_centric_2` | 第三次完整 validation：每 rank 543 batches；累计 1,629；总损失 `0.3351006210`；配体体素 PRAUC `0.5468277931`；受体 PRAUC `0.6241850853`；`num_gt=13,835,754` | checkpoint `global_step=3407`；warmup LR `3.9336034e-5`；plateau 尚未启动，实际衰减 0 次。`TOP...0.5468` 与 `last` SHA `f85b1beb...63620b`；`BEST` 已追上 `0.5264`，SHA `63c236bd...a5625`。18:06 已推进到 step 3635 |
| `366071` | Find_1 PDB-centric-1；hnode02，H100×1、CPU×32；W&B `wi4gcvcs` | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_1/Find_1-pdb_centric_1____Find_1_job366071_20260901T102343_a11_pdb_centric_1` | 第六次完整 validation：1,250 batches；总损失 `0.2843468189`；配体体素 PRAUC `0.5093884468`；受体 PRAUC `0.6373867393`；`num_gt=8,156,584` | checkpoint `global_step=7143`；LR `5e-5`；plateau best `0.5093884468`、bad epochs 0、实际衰减 0 次。`TOP...0.5094` 与 `last` SHA 均为 `7b331145...3cdcb`；`BEST` 仍是上一轮文件 `d86bd106...02c2`。08:35 已推进到 step 8123 |
| `350305` | occurrence-centric `unet_diff`；gnode10，A800×1、CPU×16；W&B `nf93buae` | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal` | 新一轮完整 validation：827 batches；总损失 `0.1617104411`；配体体素 PRAUC `0.6343953609`；受体 PRAUC `0.7158595920` | 原始分数创新高，但只比 plateau best 高 `0.0024929643`，低于阈值 `0.003`；plateau best 保持 `0.6319023967`、bad epochs 1、LR `4e-6`、实际衰减 2 次。`TOP...0.6344` 与 `last` SHA 均为 `8777e7a5...51409`；`BEST` 已追上上一轮 `0.6319`，SHA `269d9c2b...ead89`。08:35 已推进到 step 43493 |
| `366277` | Find_1 历史 checkpoint 两节点续训；gnode09、gnode10 各 A800×1、CPU×17；W&B `9kj96nld` | `/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1` | 第二次真实续训 validation：每 rank 551 batches；总损失 `0.2849375606`；配体体素 PRAUC `0.6297937632`；受体 PRAUC `0.7051408887`；`num_gt=14,802,569` | checkpoint `global_step=13545`；LR `5e-5`；plateau best `0.6297937632`、bad epochs 0、实际衰减 0 次；候选 `p_best=0.4482421875`。`TOP...0.6298` 与 `last` SHA 均为 `eadba47d...7964`；历史 `BEST` SHA `d633de05...fd4c`。08:31 已推进到 step 14360 |

## 11:23—11:50 Follow-up

Job `368455` 完成第二次双卡 validation：每 rank 543 batches、累计 1,086，`num_gt=13,835,754`，总损失 `0.3797462285`、配体体素 PRAUC `0.5264484882`、受体 PRAUC `0.5785223842`。`TOP...0.5264` 与 `last` SHA 均为 `63c236bd...a5625`；checkpoint `global_step=2271`、LR `3.1721789e-5`，仍处于 warmup，plateau 未启动、实际衰减 0 次。磁盘 `BEST` 已追上首次 `0.4649`，SHA `c3ad5b38...945df1`。训练随后推进到至少 step 2369。

Job `366071` 完成第七次 validation：1,250 batches、累计 8,750，`num_gt=8,156,584`，总损失 `0.2750830948`、配体体素 PRAUC `0.5385515094`、受体 PRAUC `0.6454816461`。`TOP...0.5386` 与 `last` SHA 均为 `bbf388a2...e7b1a`；checkpoint `global_step=8334`、LR `5e-5`。相对上一 plateau best 提高 `0.0291631`，超过 `0.003`，因此 plateau best 更新、bad epochs 归零，实际衰减仍为 0。磁盘 `BEST` 已追上上一轮 `0.5094`，SHA `7b331145...3cdcb`。训练随后推进到至少 step 8378。

Jobs `350305` 与 `366277` 同期继续训练，没有新的 validation、checkpoint、错误或资源变化。本轮检查与 checkpoint 审计均为只读，没有操作任何训练、锁或资源。

13:24 的下一轮检查发现 Job `350305` 已于 12:50 完成又一次 827-batch validation：总损失 `0.1607980728`、配体体素 PRAUC `0.6314629316`、受体 PRAUC `0.7157904506`。该分数低于 plateau best `0.6319023967`，因此 bad epochs 从 1 增至 2；LR 仍为 `4e-6`，实际衰减仍为 2 次。`TOP...0.6315` 与 `last` SHA 均为 `70150347...cc1c3`，磁盘 `BEST` 已追上原始最高的 `0.6344`，SHA `8777e7a5...51409`。训练随后推进到至少 step 44087。下一次仍未达到 `best+0.003` 的 validation 将触及 `patience=3`，需要重点核验第三次实际学习率衰减和停训行为。

14:57 的下一轮检查发现 Job `366277` 已于 13:21 完成第三次真实续训 validation：每 rank 551 batches、累计 7,163，`num_gt=14,802,569`，总损失 `0.2847497165`、配体体素 PRAUC `0.6125640869`、受体 PRAUC `0.7039208412`。该分数低于 plateau best `0.6297937632`，因此 bad epochs 为 1；LR 仍为 `5e-5`，实际衰减仍为 0。候选 `p_best=0.0595703125`，`last_stepped_validation=[14673,0]`，没有重复推进 validation 副作用。`TOP...0.6126` 与 `last` SHA 均为 `5c320e83...dc065`，磁盘 `BEST` 已追上 `0.6298`，SHA `eadba47d...7964`。训练随后推进到至少 step 14777。

18:04 的下一轮检查发现 Job `368455` 已于 17:04 完成第三次完整双卡 validation：每 rank 543 batches、累计 1,629，`num_gt=13,835,754`，总损失 `0.3351006210`、配体体素 PRAUC `0.5468277931`、受体 PRAUC `0.6241850853`。候选 `p_best=0.134765625`。checkpoint `global_step=3407`、LR `3.9336034e-5`，仍处于 4,998-step warmup；plateau 尚未启动，bad epochs 与实际衰减均为 0。`TOP...0.5468` 与 `last` SHA 均为 `f85b1beb...63620b`，磁盘 `BEST` 已追上 `0.5264`，SHA `63c236bd...a5625`。18:06 已继续到至少 step 3635；其余三个活动 Job 本轮没有新事件。

## Interpretation Notes

- Job `366071` 的 `0.5093884468` 只比所有历史原始分数中的 `0.5078241825` 高 `0.0015642643`，但这不构成调度器阈值错误。`0.5078241825` 发生在 warmup 阶段，没有进入 plateau；当前 checkpoint 的 `validation_index=2`，本轮相对上一条进入 plateau 的结果具有超过 `0.003` 的改进，因此 plateau 接受新 best 是预期行为。
- Job `350305` 的 ModelCheckpoint 仍按原始分数保存 `0.6344` 新 TOP；plateau 则按 `expected_threshold=0.003` 拒绝把它作为新调度基准。两套状态分别承担“保留最高原始 checkpoint”和“控制学习率”的职责，不矛盾。
- Job `366277` 的 `validation_batch_progress.current.completed=551`、`is_last_batch=true`，累计 validation 进度只增加一个完整周期；`num_gt` 也维持约 1,480 万的完整量级。此前修复的 Lightning 完成态 validation 恢复边界仍然有效。
- Job `368455` 的首轮 validation 发生在 warmup 中，因此 `WarmupPlateauController.best=-Infinity`、`validation_index=0`、`lr_reduction_count=0` 均为预期。第二次实际学习率衰减 checkpoint 尚未出现。

## Immutable Execution Identities

- Job `368455`：release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_2ca40d551603/Pocket_Plus`；launch `/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260903T233745_a1`；`run_cmd.sh` SHA `26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。
- Job `366071`：release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_8a3ba3ce8a2c/Pocket_Plus`；launch `/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T102343_a11`；当前 `run_cmd_366071.sh` SHA `30c77bb9619c1817106db095fe4e56f566782c88e5813d5ac188045faadc62ce`。
- Job `350305`：release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_fdb8a30fa196/Pocket_Plus`；launch `/home/penghongen/Feedback/Pocket_Plus/launches/350305/unet_diff_job350305_20260822T171529_a1`；`run_cmd.sh` SHA `29c6433271a5f46c8403d5d2a9b9d7de6c3f1ce460b12a9e4c4714b738cedfc5`。
- Job `366277`：release `/home/penghongen/Feedback/Pocket_Plus_Find1/releases/Pocket_Plus_Find1_7b0d38481c0f/Pocket_Plus_Find1`；launch `/home/penghongen/Feedback/Pocket_Plus_Find1/launches/366277/Find_1_job366277_20260902T182829_a3`；`run_cmd.sh` SHA `ee1b76a946618dfa1bc10f56b5d6351db0deaa71bbbfae8cefcbac7e1b24cef3`。

## Audit Commands And Side Effects

大型 checkpoint 的 SHA-256 与 Lightning 状态均在对应保留 allocation 内只读检查，命令形态为：

```bash
srun --jobid=<job-id> --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --gres=none sha256sum <BEST> <newest-TOP> <last>
srun --jobid=<job-id> --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --gres=none \
  /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python \
  <checkpoint-inspector> <last.ckpt>
```

本轮没有启动、重启或停止任何训练，没有删除、创建或修改任何控制锁。Job `358384` 的取消发生在只读守护之外；当前 Slurm 记录不能证明取消者身份，因此不得把它归因于本任务或任一特定外部操作者。

## Next Actions

1. 固定醒来检查 Jobs `368455`、`366071`、`350305`、`366277`；只在 validation/checkpoint、实际学习率衰减、错误、重启、完成或资源变化时集中记录。
2. Job `368455` 达到第二次实际学习率衰减后，立即核对并保留完整 checkpoint 的模型、optimizer、scheduler、回调与训练进度状态；当前训练仍按 3 次衰减停训。
3. Job `350305` 下一轮 validation 若仍未超过 plateau best 加 `0.003`，关注 bad epoch 与第三次实际衰减；不因原始 TOP 创新高而误判 plateau 已更新。
4. Job `358384` 的活动锁已经随外部取消完成清理；不自行重建 allocation。如用户希望重新占用第三张 H100，需要新的明确提交授权。
5. 稳定状态继续使用多个独立 300 秒睡眠组成 60 或 90 分钟静默等待，不创建 heartbeat。
