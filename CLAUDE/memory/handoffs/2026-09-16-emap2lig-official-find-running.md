# Handoff: Emap2lig 官方 Find held-out 评估运行中

Date: 2026-09-16

## Current State

Job `383986` 正在 `gnode09` 使用 A800×1、4 CPU 执行 179-PDB `test_0` 的 Emap2lig v0.3.4 官方 Find。第 2 次执行于 09:40 启动，首个 `9ter` 已正常进入 512 个 ROI 的 GPU 前向；`after_lock_383986` 保留，禁止删除。

## Completed

- Emap2lig 实现提交为 `6d793a5`；唯一预测行为修改是超过 100 个 blob 后不再退出。
- Pocket Plus 真实实现端点 `ac028bc` 与 Learn 端点 `3966fde` 整树等价；`Learn/CUMULATIVE` 已推进到 `3966fde`。
- 服务器定向测试通过。Job `383986` 第 1 次隔离门控完整保留 `9ter` 的 279 个官方 blob，并成功完成概率映射、实例映射和标准指标聚合。
- 正式 Pocket Plus release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_5570cf8a0b7a/Pocket_Plus`。
- 正式 Emap2lig release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_60e34f30b989/Emap2lig`。
- 正式 launch 为 `/home/penghongen/Feedback/AdaLigand/launches/383986/allocation_runner_job383986_20260916T094030_a2`。

## Decisions

- `test_0` 只执行一次官方 Find；`test_1` 从 `test_0` 的逐 PDB 评估事实保序派生。
- 每个官方不少于 32 体素的 blob 均为 `selected=True`、`prauc_eligible=True`；分数为原生 ligand probability 的实例内均值。
- 概率采用线性映射，实例采用最近体素映射并保留官方编号，允许不同实例映射后重叠。
- 正式评估使用 4 个 CPU worker；不增加双卡分片。

## Next Actions

1. 以 12 次连续 `Start-Sleep -Seconds 300` 组成 60 分钟静默守护窗口，醒来后检查完成数、GPU 活动、日志和锁。
2. Find 完成后核验 179 个官方状态，再等待 4-worker CPU 评估与 149-PDB `test_1` 派生。
3. 执行全量只读验收，生成 AdaLigand `收口の结果/Stage1/Emap2lig/` 下三份文档。
4. 更新执行日志、映射和本 handoff；任务结束后 Job 必须停回 `try_lock_383986`，并保留 `after_lock_383986`。

## Files To Reopen

- `文档/exec_plan/2026-09-16_Emap2lig官方Find_held_out评估.md`
- `src/inference/baseline/emap2lig_find.py`
- `训练与运行/sh/infer/emap2lig_official_find_li.sh`
- `CLAUDE/memory/handoffs/2026-09-16-emap2lig-official-find-running.md`
