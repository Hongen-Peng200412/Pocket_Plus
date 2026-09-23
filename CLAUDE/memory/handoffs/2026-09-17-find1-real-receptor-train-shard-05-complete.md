# Handoff: Find_1 真实受体训练第 5 片完成

Date: 2026-09-17

## Current State

- Job `378587` 在 hnode02 保留 H100×1、32 CPU allocation。第 5 片正式流程和全量验收均已完成，Job 当前停在 `try_lock_378587`；`after_lock_378587` 保留，`pre_lock_378587` 与 `kill_lock_378587` 不存在。
- 用户第 6 片已获严格后继授权，其身份门控已经通过，但新的 release 和正式运行尚未启动。第 7 片及以后不在授权范围。

## Completed

- 用户第 5 片固定为 CLI `shard-index=4`，含 275 个唯一 PDB，成员序列 SHA-256 为 `70ff0eb03dbd8bcb9ce1cc99a69a7afb2191dc68d041cb93f9d2be13c8e120b2`。
- 275/275 个成员均完成 probability、冻结阈值 F2 blobs、centered 和冻结 Gaussian score-only，并具备 `score/selected` 与三个完成标记。
- 全量验收逐 PDB 精确复算正式 blobs 与 Gaussian 分数，并核对候选顺序、四组 offsets、来源体素映射、48³ 裁块和超大 blob 提示契约。共核对 9,638 个来源 blob、9,387 个 centered 候选、1,348,302 个归档来源体素、5 个超框候选和 7,059 个最终选择。
- 最终报告为 `/storage/penghongen/tmp/find1_real_receptor_train_shard_05_20260916/final_verification.json`，SHA-256 为 `2626a2e26193182ee85a1da007328d674d1f290c055fb6237d78aada5ba95537`。

## Decisions

- 第 5 片沿用固定 checkpoint、真实受体、F2 语义阈值、`objective_beta=1` Gaussian 参数、`forward_min_voxels=8` 和 `--continue-on-blob-exceed`；没有调参、评估或过滤 PDB。
- 第 6 片只有在上述最终报告通过且 `try_lock_378587` 存在后才能启动；该前置条件已经满足。
- 第 6 片不得创建新的 `kill_lock_378587`，不得触碰 `after_lock_378587`，不得使用 `scancel`，也不得扩大到第 7 片。

## Next Actions

1. 从当前唯一最新的 `Learn/CUMULATIVE` 端点完成只接受用户片号 6 的最小正式入口、测试与 README 审查，并冻结新 release。
2. 记录新 release 与动态命令身份；门控通过后原子替换 Job `378587` 的动态命令，再删除 `try_lock_378587` 启动第 6 片。
3. 首批 probability 产物和 GPU 活动稳定后，按连续 300 秒睡眠组成 60 或 90 分钟静默守护；完成后对 275 个成员执行同口径全量验收。

## Files To Reopen

- `文档/exec_plan/2026-09-16_Find_1真实受体训练分片推理总日志.md`
- `文档/exec_plan/2026-09-17_Find_1真实受体训练第06片推理.md`
- `训练与运行/sh/infer/find1_real_receptor_train_shard_06.sh`
- `CLAUDE/memory/handoffs/2026-09-17-find1-real-receptor-train-shard-05-complete.md`
