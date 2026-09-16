# Handoff: Find_1 真实受体训练第 5 片运行中

Date: 2026-09-16

## Current State

- Job `378587` 在 hnode02 使用 H100×1、32 CPU 运行固定训练第 5 片，用户片号 5 对应 CLI `shard-index=4`。
- 第 2 次执行 launch 为 `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260916T162710_a2`。控制器仍属于 PocketXMol，但其 `run_cmd.sh` 以绝对路径调用冻结的 Pocket Plus release。
- 主进程已经进入 probability 阶段，命令含 `--shard-count 50 --shard-index 4`，配置为 `stage1_v3_h100_32cpu.yaml`。最近一次检查仍处于模型与数据初始化，GPU 尚未形成稳定前向负载。
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
- 输出非破坏性续写到 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train`。不处理用户第 6 片。

## Next Actions

1. 确认 probability 阶段开始正常 GPU 前向并持续增加第 5 片完成标记；检查 stderr 是否出现 traceback、CUDA OOM 或冻结输入错误。
2. 稳定后不停止任务、不创建 heartbeat，以连续 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待，再核查阶段进度。
3. 顺序守护 F2 blobs、centered 与 Gaussian score-only；阶段变化、失败、锁状态变化或完成时更新实验日志和总日志。
4. 完成后对 275 个 PDB 的三类 NPZ、完成标记、候选顺序、offsets、`score/selected`、超量标记和成员集合执行全量验收；Job 最终停回 `try_lock_378587`，保留 `after_lock_378587`。

## Files To Reopen

- `文档/exec_plan/2026-09-16_Find_1真实受体训练第05片推理.md`
- `文档/exec_plan/2026-09-16_Find_1真实受体训练分片推理总日志.md`
- `训练与运行/sh/infer/find1_real_receptor_train_shard_05.sh`
- `configs/inference/stage1_v3_h100_32cpu.yaml`
- `CLAUDE/memory/handoffs/2026-09-16-find1-real-receptor-train-shard-05-running.md`
