# Handoff: Stage1 三个训练脚本的学习版式已经统一

Date: 2026-07-26

## Current State

`训练与运行/sh/Find_0.sh` 是用户亲自调整并确认的版式金标准。`Find_1.sh` 与 `unet_c1.sh` 已按该文件统一章节顺序、分隔线、空行组织、行内注释风格、产物路径说明和正式启动布局。

本轮没有修改 `Find_0.sh`，没有恢复运行目录碰撞检查，也没有修改 README、Hydra YAML、通用提交器、release/launch 或四锁运行器。

## Completed

- `Find_1.sh` 保留双 H100、batch 6、两个 density cube chunk 参数、学习率 `5e-5`、CPC1 patience 3，以及 CPC2 的 `0.005` warmup。
- `unet_c1.sh` 保留单 H100、batch 6、20 个 DataLoader worker、学习率 `1e-4`、单阶段正式训练和按设备数决定 DDP 未使用参数检查的行为。
- 两个脚本都显式说明 `experiment_group`、`tag`、`POCKET_RUN_STAMP` 与最终 `logs/` 目录的关系。
- 三个脚本均通过 Git Bash `bash -n`。
- `git diff --check` 通过。
- 三个脚本的通用设置区逐行一致，章节顺序一致。
- 本地 Hydra 成功组合 `Find_1/CPC1`、`Find_1/CPC2` 和 `unet_c1`；解析后的 Dataset、设备数、批量、学习率、warmup 与调度器字段符合脚本契约。

## Decisions

- 代码量的增加直接占用用户的理解资源；不得只为假设性异常在训练主线中增加代码。
- 正式运行器每次执行都会产生唯一 `POCKET_RUN_STAMP`，三个训练脚本不重复增加目录碰撞检查。
- 本轮属于小型工作区修改，不重组、不暂存、不提交 Git。

## Next Actions

- 由用户直接阅读三个脚本和本轮差异。
- 若以后需要纳入 Git 历史，应先保留用户已有的 README 与 `Find_0.sh` 暂存边界，再单独处理本轮两个未暂存脚本。

## Files To Reopen

- `训练与运行/sh/Find_0.sh`
- `训练与运行/sh/Find_1.sh`
- `训练与运行/sh/unet_c1.sh`
- `CLAUDE/memory/learnings/decision-2026-07-26-code-volume-is-human-comprehension-cost.md`
