# Handoff: Find_1 真实受体训练第 5 片运行中

Date: 2026-09-16

## Current State

- Job `378587` 在 hnode02 使用 H100×1、32 CPU 运行固定训练第 5 片，用户片号 5 对应 CLI `shard-index=4`。
- 第 2 次执行 launch 为 `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260916T162710_a2`。控制器仍属于 PocketXMol，但其 `run_cmd.sh` 以绝对路径调用冻结的 Pocket Plus release。
- probability、冻结阈值 F2 blobs 与 centered 前向均已完成 275/275。主进程已按顺序进入冻结 Gaussian score-only；最近检查为 173/275 个 `F2_centered.npz` 同时具备 `score/selected`，命令继续使用 `F2_gaussian.json`、`--score-only`、`--shard-count 50 --shard-index 4` 和同一冻结配置。
- `after_lock_378587` 始终保留；`try_lock_378587` 已为启动删除；`kill_lock_378587` 不存在。未使用 `scancel`。

## Completed

- 通过 `kill_lock_378587` 停止原 PocketXMol `F-5` 训练，原进程组退出后确认 GPU 空闲，并由四锁执行器自动停在 `try_lock_378587`。
- 固定分片门控确认训练清单共 13,717 个唯一 PDB；第 5 片含 275 个唯一 PDB，成员序列 SHA-256 为 `70ff0eb03dbd8bcb9ce1cc99a69a7afb2191dc68d041cb93f9d2be13c8e120b2`，与第 1—4 片无交集，并与既有完整 50 片定义逐片一致。
- 新增只接受用户片号 5 的正式入口、H100/32 CPU 配置、README 和测试。两轮独立全面审查及窄复核均通过；`tests/inference` 为 57 项全部通过。
- 实现端点 `71004c6` 与 Learn 端点逐文件等价；`Learn/CUMULATIVE` 已快进到 `691674b1930f4dbc220541be0d1cad1985105ab1`。
- 正式 release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_b354f7562994/Pocket_Plus`，内容 SHA-256 为 `b354f75629946bcf7550e29b37443bce9781719f773eab72d488e2c93993e171`。
- 动态命令 SHA-256 为 `70b02f7ad33dc53246dadc64809ea9ca600274ce1571931f3735c1bcba772e46`；旧命令已保留在 `/storage/penghongen/tmp/find1_real_receptor_train_shard_05_20260916/run_cmd_378587.before_find1_shard_05.sh`。

## Decisions

- 正式流程仅为 probability → 冻结 F2 blobs → centered → 冻结 Gaussian score-only，不调参、不评估、不使用 overwrite。
- 每次运行先核验训练清单、checkpoint、resolved config、F2 语义参数和 F2 Gaussian 参数的冻结 SHA-256。
- H100/32 CPU 配置保持完整图 batch 18、centered batch 12、两阶段各 26 个请求物化线程，并把 blobs worker 限制为 30。
- 输出非破坏性续写到 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train`。
- 用户第 6 片已获严格后继授权，但只有第 5 片 275 个成员通过全量产物与数组契约验收、且 Job `378587` 停回 `try_lock_378587` 后才能启动；用户第 6 片固定为 CLI `shard-index=5`，不得创建新的 `kill_lock_378587` 或扩大到第 7 片。

## Next Actions

1. 以单次 `Start-Sleep -Seconds 300` 短间隔核查 score-only 是否给 275 个 NPZ 全部写入 `score/selected`，并确认 275 个 centered 完成标记恢复。
2. score-only 完成后等待四锁执行器停回 `try_lock_378587`，同时确认 `after_lock_378587` 始终保留。
3. 对第 5 片 275 个 PDB 的三类 NPZ、完成标记、候选顺序、offsets、`score/selected`、超量标记和成员集合执行全量验收。
4. 仅在第 5 片最终门控通过后，准备用户第 6 片的独立实验日志、固定身份门控、最小正式入口和新 Learn release，再原子改写动态命令并删除 `try_lock_378587` 启动 CLI `shard-index=5`。

## Files To Reopen

- `文档/exec_plan/2026-09-16_Find_1真实受体训练第05片推理.md`
- `文档/exec_plan/2026-09-16_Find_1真实受体训练分片推理总日志.md`
- `训练与运行/sh/infer/find1_real_receptor_train_shard_05.sh`
- `configs/inference/stage1_v3_h100_32cpu.yaml`
- `CLAUDE/memory/handoffs/2026-09-16-find1-real-receptor-train-shard-05-running.md`
