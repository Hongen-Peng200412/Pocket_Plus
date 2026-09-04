# Handoff：Find_1 PDB-centric-2 已转换为双 H100 正式训练

Date: 2026-09-03

## Current State

Find_1 PDB-centric-2 的权威训练已从单卡 Job `367928` 转换为同节点双卡 Job `368455`。`368455` 于 2026-09-03 23:37:06 +08:00 在 hnode01 获得 H100×2、CPU×64，使用 NCCL 建立两个训练 rank 并从头训练；23:57 时 W&B 已推进到 `global_step=50`。allocation 只有 `/home/penghongen/Feedback/Pocket_Plus/allocations/368455/after_lock_368455`，没有 `pre_lock`、`try_lock` 或 `kill_lock`。

用户明确确认 `train.scheduler.stop_after_lr_reductions=3` 是手动修改后的最新科学配置。当前运行保持 3；第二次实际学习率衰减后的完整 checkpoint 仍需保留，可在需要时作为“衰减 2 次”版本的选择端点。

## Resource Conversion

- 单卡 Job `367928` 于 22:42:57 在 hnode01 获得 H100×1、CPU×32，并停在 `pre_lock_367928`。
- 23:28:31 按已有授权核验 Job、节点、资源、锁和 `run_cmd` 哈希后，只删除 `/home/penghongen/Feedback/Pocket_Plus/allocations/pre_lock_367928`；原有正式命令直接启动，没有加入 GPU 门禁。
- 用户于 23:35:39 先提交双卡 Job `368455`。Slurm 在 23:37:06 为其实际分配 hnode01 的 H100×2、CPU×64；用户随后取消 `367928`。`sacct` 记录 `367928` 于 23:37:00 被 UID 1351 取消，转换顺序符合“先取得双卡、后取消单卡”。
- `367928.batch` 的 `FAILED/15:0` 来自外部取消，不是训练代码故障。单卡 a1 尚未创建 W&B 或 checkpoint，不存在应恢复的训练状态。
- 已取消的 `367928` 仅保留证据，不再属于固定醒来检查，也不得重新操作其资源。

## Job 367928 Evidence

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/367928/Find_1_pdb_centric_2_job367928_20260903T232856_a1`。
- 运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job367928_20260903T232856_a1_pdb_centric_2`。
- immutable `launch.json` SHA-256：`4e992b6ef9f797275cfa537bf89b4be172bf1fe4b49a05baf2529a0859872719`。
- immutable `run_cmd.sh` SHA-256：`26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。
- `config.yaml`、`train.yaml`、`src_snapshot/manifest.json` SHA-256：`d743998d0ffc2f7a5d1c7e3b8038080896737b1fc78fac355a3cd90dfc20647b`、`e64eea2cebcc00a2220dbb7311d2c6b75beef8a2bd80bb4e8dccb1b5a06101db`、`360d76a2c9951c00408d2e48f5c32d3b797b2a1e12c3d0f5d7c191d9e4b9590f`。

## Job 368455 Identity

- Slurm：h100 partition、`h100g2` QOS、hnode01、H100×2、CPU×64、单节点、无时限、`after_hold=1`。
- Slurm 分配的 GPU 索引：0、1。
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_2ca40d551603/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/368455/Find_1_pdb_centric_2_job368455_20260903T233745_a1`。
- 正式运行与全部实验产物根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job368455_20260903T233745_a1_pdb_centric_2`。
- checkpoint 目录：上述运行根下的 `checkpoints/`。
- W&B：`pencounkdual-111/AdaLigand_Stage1/g7ul4r8p`。
- immutable `launch.json` SHA-256：`89147969e47d4f561d88faad8732eb615ba9e4661d5ac517c853c6c72fb448ab`。
- immutable `run_cmd.sh` 与 allocation 当前 `run_cmd_368455.sh` SHA-256：`26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。
- `config.yaml`、`train.yaml`、`src_snapshot/manifest.json` SHA-256：`b7bb25254502e6753a3239a08b8cf37a5cbad25dc123476b5dc2be9c8329ebd5`、`9a655a76e8e9efb47f9160c7065cc3a0f01d309a4fb24349e3c1a55700498f98`、`ee159ed4d9cb9083d3bdc018e9f0cf469cf29784ee67fcb474c145715f746be2`。
- release 中 `Find_1_pdb_centric_2.sh`、Dataset YAML、训练 YAML、实验 YAML SHA-256：`8b8d3aa8fc4806d554e35c58ba5ff6946b9718577ca083755b5855a4b63e44d6`、`54e9e18685c0e9dadd249de4a061dc3a47657c25b17445be15d4c47b1a42b6a2`、`ad09b9bfa746909fba9645fe7a7fb324d6f03971c6df8ce58c5ae85f980b446a`、`2ff911aae5fa7b7ba465ae12b0181ba784c6bf645724d8ecb960a120f5e6d8ad`。
- immutable 启动命令：`exec bash "${TASK_PROJECT_ROOT}"/训练与运行/sh/Find_1_pdb_centric_2.sh`。

## Configuration Audit

与已核验的旧单卡 resolved config 逐字段递归比较，只有以下三处差异：

1. `train.devices`：1 → 2，这是单卡到双卡转换本身。
2. `train.scheduler.stop_after_lr_reductions`：2 → 3，这是用户手动修改并再次确认的最新值。
3. `output.save_top_k`：30 → 50，只增加最佳 checkpoint 的保留数量，不改变优化动力学。

两份运行的 `src_snapshot/manifest.json` 中全部 `src/` 文件哈希一致。其余科学契约保持不变：

- Dataset：`stage1_find_pdb_centric_2`。
- validation：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric_v2.npz`，SHA-256 `0c92a731a7676f8083a250ff54e13dd2356e2cc462e383f6ad45c4c30de5fda2`；200 个 PDB、3,305 个 bias BOX、3,200 个 context BOX。
- 训练采样：每个 PDB 目标 50 个 bias BOX、bias 比例 25/33、每个 occurrence 最多 1 个 bias BOX，并补充 16 个 context BOX。
- 每卡 batch 6、全局 batch 48、每 rank 30 workers、梯度累积 4、学习率 `5e-5`、绝对改进阈值 `0.003`、`patience=3`、每 epoch 8 次 validation、110 epochs。
- Find_1 voxel/point 两个互斥参数组分别裁剪到 0.5，仍使用一个 AdamW optimizer。

## Runtime Evidence

- 两条 NCCL rank 均已注册，日志明确给出 `world_size=2` 和 `distributed_backend=nccl`。
- 23:49 的三次 GPU 抽样中，GPU 0/1 均占用约 80 GiB 显存并观察到计算活动；第三张 H100 继续由 Job `358384` 的其他工作使用。
- W&B 从 23:49 的 `global_step=20` 继续推进到 23:57 的 `global_step=50`；训练损失保持有限。
- 没有发现 traceback、CUDA OOM、NCCL、DataLoader worker 退出或非有限值错误。

## Next Actions

1. 每次醒来固定检查 Jobs `368455`、`366071`、`350305`、`366277` 的训练、validation、checkpoint、错误与资源状态；无变化轮询不写日志。
2. 对 Job `368455` 重点检查双 rank 是否持续推进、第一次完整 validation、TOP/BEST/last checkpoint、第二次实际学习率衰减对应 checkpoint 与最终第三次衰减停训。
3. Job `358384` 不在固定训练检查清单，但未经新的明确授权不得释放 `after_lock_358384` 或中断其 allocation 内工作。
4. 稳定状态不使用 heartbeat；使用多个独立 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待。
5. Job `368455` 没有自动释放授权。训练完成或进入 `try_lock` 后先核验最终产物并报告，不擅自删除锁或取消 allocation。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-training-global-watch.md`
