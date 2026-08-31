# Handoff: Find_1 A800 两节点历史续训已启动

Date: 2026-09-01

## Current State

Find_1 历史续训 Job `366277` 正在 `gnode09,gnode10` 运行，每节点使用 A800×1、CPU×17，每个 DDP rank 使用 16 个 DataLoader workers。根级 `pre_lock_366277` 已于登录节点时间 2026-09-01 07:46:03 +08:00 删除；allocation 内 `after_lock_366277` 保持存在，没有操作 `kill_lock_366277`。attempt a1 已建立 release、launch 与训练输出根，当前正在迁移字面 `last.ckpt` 及其 ModelCheckpoint 历史文件，尚未观察到训练 step 或失败。

H100 Job `366071` 的 schema 4 动力学门禁 attempt a7 同时继续运行；最近一次事实是第一条 AUTO controlled 轨迹的稠密 float32 旁车仍在增长，没有 `_FAILED` 或 `_COMPLETE`。两项任务互相使用独立节点、反馈根和隔离代码根。

## Completed

- `nvlink` 没有 PENDING 作业后，先提交并核验 Job `366277`，再只取消授权目标 Jobs `350302`、`356946`。新 Job 于 07:35:20 开始占用 `gnode09,gnode10`，每节点一张 A800 和 17 CPU。
- 历史代码固定为提交 `5b4aa5f52d699d904fa82e97e22099047e607e4c`，本地 Git 归档为 6,584,320 bytes，SHA-256 `9739367a598bccb2387729cec1f92ce7b70555795ad6bd5ba6db29e6f1833092`。
- 服务器隔离根为 `/home/penghongen/Feedback/Pocket_Plus_Find1/task_roots/find1-historical-resume-5b4aa5f52d69/Pocket_Plus`；共享项目根没有被训练代码部署覆盖。
- 字面 checkpoint 为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt`，1,466,227,120 bytes，SHA-256 `d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c`。
- 正式动态命令为 `/home/penghongen/Feedback/Pocket_Plus_Find1/allocations/366277/run_cmd_366277.sh`，1,322 bytes，SHA-256 `c476077d76d62d86e522929de9e8e44ccc68c076eba392757dff683bde33fc11`。它显式固定隔离代码根、checkpoint、checkpoint 摘要、反馈根和 16 workers。
- 放行前的 Job、锁、部署摘要、checkpoint 摘要、Shell 语法和服务器 Python 语法门禁全部通过。
- 本次接管、完整提交命令、取消命令、部署命令身份、完整动态命令和所有路径已写入 `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`。

## Formal Artifacts

- runner release：`/home/penghongen/Feedback/Pocket_Plus_Find1/releases/Pocket_Plus_Find1_7b0d38481c0f/Pocket_Plus_Find1`
- launch：`/home/penghongen/Feedback/Pocket_Plus_Find1/launches/366277/Find_1_job366277_20260901T074338_a1`
- 训练输出根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260901T074338_a1_CPC1`
- checkpoint 迁移目录：上述训练输出根下的 `checkpoints/`
- 部署身份：`/home/penghongen/Feedback/Pocket_Plus_Find1/task_roots/find1-historical-resume-5b4aa5f52d69/DEPLOYMENT_IDENTITY.txt`

## Decisions

- 接受 `last.ckpt` 没有保存最后两个部分累积梯度；从它记录的 `global_step=11287` 恢复，并只在 epoch 0 跳过每 rank 已完成的 45,150 个 microbatch。
- 历史续训保持原全局梯度裁剪 0.5，不使用生产 Find_1 新增的两组裁剪。
- node rank 0 负责复制历史 top-k 并迁移 ModelCheckpoint 路径，node rank 1 只等待原子发布结果；两者共用同一 run stamp。
- 当前不释放 A800；训练正常后保留 `after_lock_366277` 并持续监视。

## Next Actions

1. 读取 `366277` allocation 输出和训练目录，确认 `resume_state.ckpt`、`resume_state_manifest.json`、历史 top-k 复制和两节点 NCCL 初始化完成。
2. 核对 Lightning 的恢复 epoch、global step、首轮跳批边界和第一个真实训练 step；出现错误时在历史隔离分支内做最小修复并完整记录。
3. 同时观察 H100 a7 的 `_FAILED`、`_COMPLETE`、轨迹 JSON 和四份比较结果。动力学门禁通过后才允许 Job `366071` 进入 PDB-centric-1 正式训练。
4. 只有明确事件才追加执行记录与 handoff；两项任务稳定后，用多个独立 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `CLAUDE/memory/handoffs/2026-09-01-find1-schema4-a7-running.md`
- `训练与运行/sh/Find_1.sh`
