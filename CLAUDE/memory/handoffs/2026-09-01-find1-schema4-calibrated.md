# Handoff: Find_1 schema 4 已完成同角色标定

Date: 2026-09-01

## Current State

H100 Job `366071` 仍在 `hnode02` 占用 H100×1、CPU×32，根级 `try_lock_366071` 与 allocation 内 `after_lock_366071` 均存在，`kill_lock_366071` 不存在。attempt a7 已完整生成八条 schema 4 轨迹，但因四项旧数值上限越界而明确失败；attempt a8 已只读完成同角色底噪标定。提交 `827d47030b63d9162ff05d7747cc13dac506eb1b` 仅更新四项标定后的包络，尚未部署或启动新的独立验收 attempt。

A800 两节点历史续训 Job `366277` 同时保持正常运行，使用 `gnode09,gnode10`、每节点 A800×1、CPU×17、每 rank 16 workers；两项任务互相使用独立反馈根和代码根。

## Evidence

- a7 输出根：`/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics/Find_1_job366071_20260901T072743_a7`。
- a7 完整启动脚本：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T072743_a7/run_cmd.sh`，8,477 bytes，SHA-256 `5102edab331d149ab3731213caa510cf1e58a52793f991c1d80cc3b104a5c62b`。
- a7 于 2026-09-01 08:16:18 +08:00 写入 `_FAILED`；16 个受控 microbatches 身份和 recycle 均一致，失败只涉及两个体素输出逐元素上限和两个参数更新逐元素上限。
- a8 输出根：`/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics_calibration/Find_1_job366071_20260901T083454_a8`。
- a8 完整启动脚本：`/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T083454_a8/run_cmd.sh`，5,538 bytes，SHA-256 `9f0095a408f2762066b8b711dc8b4bc2d7937294e22d9149247c2f2c45169cb0`。
- a8 于 2026-09-01 08:44:11 +08:00 写入 `_COMPLETE`；`summary.json` SHA-256 为 `30b34c2257c7bf6d344c40da4061bce380e1a37e6dae3c78208c9781fc2ca96a`，两份同角色身份检查均通过。
- 同角色最大值证明跨角色差异未出现系统性放大：体素输出逐元素最大 `2.5`，参数更新逐元素最大 `3.9689243e-5`，单 microbatch 梯度 `l2` 相对差 P95 最大 `0.084122`，梯度最大值相对差 P95 最大 `0.120805`。
- 新冻结值为 `3.0`、`5e-5`、`0.10` 和 `0.16`；其余门禁不变。动力学测试为 30 passed。

## Decisions

- a7 是失败事实，a8 是标定事实，二者都不能作为最终验收通过。
- 下一次 a9 必须在新隔离代码根中从头生成八条轨迹，再运行受控、双向 replay 与 natural 全套比较。
- a9 通过以前，不启动 PDB-centric-1 正式训练；H100 继续由 `try_lock` 安全保持。
- a9 若仍失败，只能结合新的局部诊断和同角色证据判断；不得自动继续放宽阈值。

## Next Actions

1. 归档并部署提交 `827d47030b63d9162ff05d7747cc13dac506eb1b`，核对归档、服务器文件和项目源码摘要。
2. 在 Job `366071` 安装带完整身份的 a9 动态命令，通过门禁后只删除根级 `try_lock_366071`。
3. a9 通过后补齐执行记录和 handoff，再把 Job `366071` 切换为 PDB-centric-1 正式训练命令；保留 `after_lock`。
4. A800 与 H100 均进入稳定训练后，以多个独立 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/规划文档/Find_1训练与历史续训.md`
- `tmp/find1_optimization_dynamics/README.md`
- `tmp/find1_optimization_dynamics/compare_optimization_traces.py`
- `训练与运行/sh/Find_1_pdb_centric_1.sh`
