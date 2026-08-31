# Handoff: Find_1 schema 4 动力学门禁 a7 正在运行

Date: 2026-09-01

## Current State

Find_1 生产实现工作树为 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra`，分支为 `codex/find1-training`。当前实现提交是 `2fc8f15b142a5ada879b47399434ed4c3f256a98`；该提交只改动一次性优化动力学门禁，把可能碰撞的统计草图替换为可逐元素核验的 float32 稠密旁车。完整活动测试集合为 `396 passed, 11 warnings in 89.39s`，最后一个逻辑审查 P1 已经由原审查子任务窄口径复核为 PASS。

H100 Job `366071` 仍在 `hnode02` 使用 H100×1、CPU×32。attempt a7 已于 2026-09-01 07:27:48 +08:00 开始运行真实门禁；当前处于第一条 AUTO controlled 轨迹。根级 `try_lock_366071` 已被本次触发消费，allocation 内 `after_lock_366071` 保持存在，没有操作 `kill_lock`，也没有释放 H100。

## Completed

- attempt a6 在 `/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics/Find_1_job366071_20260901T044525_a6` 生成八条真实轨迹，但旧逐字节判定被 CUDA `scatter_add_` 的同角色非确定性否定。a6 于 05:10:27 写入 `_FAILED` 并返回 `try_lock`；其数值差异只用于确定逐字节相等不是可行契约，不是最终通过证据。
- schema 4 为所有跨角色共同体素输出、累计梯度和 optimizer-step 体素张量保存可恢复 float32；大数组写入每轨迹唯一 `*.dense-float32.bin`，JSON 保存 offset、长度和 SHA-256。比较器核对旁车后分块计算逐元素最大绝对差。
- 审查反例为长度 1,000,000、位置 9004/22749 的 `±50` 交换。真实旁车测试得到 `element_absolute_max=100.0`、`passed=false`；原审查子任务确认任一位置超过 `element_absolute_max` 必然失败。
- 归档 `C:\Users\15919\Desktop\Pocket_Plus_worktrees\cross_node_ddp_infra\tmp\find1-production-2fc8f1517f07.tar` 为 7,045,120 bytes，SHA-256 `35bc28f00945b3efc54185351df83ccc447d9ac16224511121e4c8610509ac57`。
- 新隔离根为 `/home/penghongen/Feedback/Pocket_Plus/task_roots/find1-production-2fc8f1517f07/Pocket_Plus`；未修改共享 `/home/penghongen/My_Project/Pocket_Plus`。AUTO/生产 `src` 与 `configs` 摘要仍为 `5c6fb1ea...fc75` 与 `90c0874d...27b7`。
- 动态命令 `/home/penghongen/Feedback/Pocket_Plus/allocations/366071/run_cmd_366071.sh` 及 a7 launch 副本的 SHA-256 均为 `5102edab331d149ab3731213caa510cf1e58a52793f991c1d80cc3b104a5c62b`。
- a7 launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/366071/Find_1_job366071_20260901T072743_a7`；输出根为 `/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics/Find_1_job366071_20260901T072743_a7`。完整十二条命令保存在 launch 的 `run_cmd.sh` 中。
- 上述提交、归档、部署哈希、触发命令、锁和全部产物地址已追加到 `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`。

## Decisions

- 共同体素路径不再要求非初始张量逐字节相等；初始 353 个共同参数仍要求原始 dtype 字节 SHA-256 完全相同。
- 非初始张量同时受传统统计包络与逐元素绝对差硬上限约束。a7 若只因新增 `element_absolute_max` 超限而失败，应先用同角色重复与跨角色结果标定实际逐元素范围，再在相同科学契约内调整上限并运行新的正式 attempt；不能把 a6 追认为通过。
- PDB-centric-1 只有在真实动力学门禁通过后才能替换动态命令并启动。PDB-centric-2 仍只允许配置与测试，不提交训练。
- A800 监视仍由 `/root/a800_takeover_executor` 每 300 秒执行。只有 `nvlink` 没有任何 PENDING 作业时才按授权先提交并核验双节点新任务，再取消且只取消 Jobs `350302`、`356946`。

## Open Questions

- a7 的真实逐元素差异是否落在当前预置上限内尚未得到服务器事实；如果不落在，需要区分同角色基线和跨角色差异后再标定。
- A800 接管窗口尚未出现；历史续训的服务器 Job、release、launch 和运行产物仍未建立。

## Next Actions

1. 观察 a7 的旁车增长和每条 JSON 发布；出现 `_FAILED`、`_COMPLETE` 或返回根级 `try_lock` 时读取完整错误与比较诊断。
2. 若 a7 因实现错误失败，在同一科学契约内修复、重新审查相关窄范围并建立新 attempt。若只是预置逐元素上限不足，使用 a7 的同角色重复与跨角色事实标定，并用下一 attempt 做正式验收。
3. 门禁通过后，把 Job `366071` 的动态命令替换为隔离根的 PDB-centric-1 正式训练入口；再次核对 Job、锁、代码、checkpoint 和命令 SHA 后才触发。保留 `after_lock_366071`。
4. A800 子任务接管成功后，由主任务接管其 Job，部署历史续训提交 `5b4aa5f52d699d904fa82e97e22099047e607e4c`，核对字面 `last.ckpt` 后再放行 `pre_lock`。
5. 只在明确事件发生时更新执行记录与 handoff；稳定状态使用多个独立 `Start-Sleep -Seconds 300` 组成 60 或 90 分钟静默等待，不用 heartbeat。

## Files To Reopen

- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `tmp/find1_optimization_dynamics/optimization_trace.py`
- `tmp/find1_optimization_dynamics/compare_optimization_traces.py`
- `tmp/find1_optimization_dynamics/README.md`
- `训练与运行/sh/Find_1.sh`
- `CLAUDE/memory/handoffs/2026-09-01-find1-h100-held-and-gates.md`
