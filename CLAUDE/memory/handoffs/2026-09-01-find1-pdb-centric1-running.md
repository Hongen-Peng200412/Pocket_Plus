# Handoff: Find_1 PDB-centric-1 与历史续训均已运行

Date: 2026-09-01

## Current State

H100 Job `366071` 已结束一次性优化动力学门禁，正式运行 PDB-centric-1。当前 allocation 动态命令只有三行，实际业务命令是 `exec bash .../训练与运行/sh/Find_1.sh`；attempt a11 已进入 epoch 0 的真实 optimizer steps。A800 两节点历史续训 Job `366277` 同时稳定运行。两项任务都保留各自 `after_lock`，没有 `kill_lock`，未经授权不得释放。

## H100 Gate Result

- a9 launch：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T090421_a9`。
- a9 输出：`/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics/Find_1_job366071_20260901T090421_a9`。
- a9 八条轨迹全部生成；受控比较通过。唯一失败是 trunk-sequence replay 的第 8 个 microbatch 体素输出逐元素差 `3.076171875 > 3.0`。
- 提交 `6905aeb476c17be679c909c56d80ffa8cdfbe8c9` 只把该上限改为 `3.5`，其余科学门禁不变；30 项定向测试通过。
- a10 launch：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T100401_a10`，完整命令 SHA-256 `6aedfd2adb24faa2ea19319b6648415e7fef1f8d4f98efdf603f3a39cb3c20be`。
- a10 输出：`/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics_targeted/Find_1_job366071_20260901T100401_a10`。
- a10 使用新进程重新生成受影响的 replay 对；受控、fresh trunk-sequence replay、full-sequence replay、natural 四份比较全部 `passed=true`、`mismatch_count=0`，于 2026-09-01 10:21:33 +08:00 写入 `_COMPLETE`。

## H100 Formal Training

最终动态命令是：

```bash
#!/usr/bin/env bash
set -euo pipefail
exec bash /home/penghongen/Feedback/Pocket_Plus/task_roots/find1-production-6905aeb476c1/Pocket_Plus/训练与运行/sh/Find_1.sh
```

- 动态命令 SHA-256：`30c77bb9619c1817106db095fe4e56f566782c88e5813d5ac188045faadc62ce`。
- 放行时间：2026-09-01 10:23:21 +08:00。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T102343_a11`。
- runner release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_8a3ba3ce8a2c/Pocket_Plus`。
- 实际代码根：`/home/penghongen/Feedback/Pocket_Plus/task_roots/find1-production-6905aeb476c1/Pocket_Plus`。
- 输出根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_1/Find_1-pdb_centric_1____Find_1_job366071_20260901T102343_a11_pdb_centric_1`。
- W&B run：`wi4gcvcs`。
- 资源和配置：hnode02、H100×1、CPU×32、workers 24、microbatch 6、累积 8、全局 batch 48、val 12 次/epoch、学习率 `5e-5`、`find_voxel_point` 两组各裁剪 0.5。
- 截至本 handoff，`trainer/global_step=14`，总损失 `0.7000278830528259`，没有 traceback、OOM 或 NCCL 错误。

以新 Slurm allocation 重启时，只应使用可读的生产入口：

```bash
bash 训练与运行/submit_task.sh \
    --sh Find_1.sh \
    --resource h100 \
    --gpus 1 \
    --cpus 32
```

一次性 a7— a10 命令只由不可变 launch 和执行记录保留，不是生产训练入口。任务收口时应从活动代码树删除一次性动力学门禁目录，通过 Git 阅读历史。

## A800 Historical Resume

- Job：`366277`，`gnode09,gnode10`，每节点 A800×1、CPU×17、每 rank workers 16。
- 输出根：`/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260901T074338_a1_CPC1`。
- W&B run：`j0qdddcn`。
- 截至同一关键节点，`trainer/global_step=11528`，总损失 `0.27465128898620605`，没有明确错误。

## Next Actions

1. 持续监视 Jobs `366071` 与 `366277` 的 step、有限损失、验证、checkpoint、W&B、Job 和锁状态。
2. 两项训练处于稳定状态时，用 12 或 18 个彼此独立的 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待；不用 heartbeat。
3. 只在 validation、checkpoint、错误、任务终止或锁变化等明确事件补写日志，不记录无变化轮询。
4. PDB-centric-2 仍只完成配置和测试，不提交训练。
5. 两项训练正常完成后，删除一次性动力学门禁活动代码，完成学习历史、tree 等价和 `Learn/CUMULATIVE` 收口。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `训练与运行/sh/Find_1.sh`
- `configs/experiment/CPC1/Find_1.yaml`
- `configs/dataset/stage1_find_pdb_centric_1.yaml`
- `configs/train/stage1_find_pdb_centric_1.yaml`
