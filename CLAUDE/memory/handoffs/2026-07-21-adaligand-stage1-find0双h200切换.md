# Handoff: AdaLigand Stage1 Find_0 双 H200 切换

Date: 2026-07-21

## Current State

当前正式资源为 unet_c1 Job 321107（1×H100、micro6/accum8/GBS48）、Find_1 Job 321540（2×H100、micro6/accum4/GBS48）和 Find_0 Job 321743（2×H200、micro8/accum3/GBS48、24 CPU）。Find_2 已在 step53 撤回。

Find_1 W&B run `qqmuqyxk` 至少到 global step119，unet 至少到 step746，两者近 30 分钟明确错误匹配为 0。Find_0 Job 321743 的两张 H200 在派发前均为 1 MiB/0%；Dataset smoke 已完成，双卡五步 memory trial 已开始，尚无明确错误。旧单 H200 Find_0 Job 321718 已精确 `scancel`。

heartbeat `adaligand-stage1` 为每 15 分钟启动监控。当前已经达到 2 张 H200 上限，不再提交 H200 探针；待 321743 正式 W&B online 并推进若干 step 后，heartbeat 应降回每 3 小时。

## Completed

- 修复 `tmp/adaligand_stage1_h200_acquire_once.sh` 的探针识别：仅当 job name 匹配且对应 `pre_lock` 仍存在时才视为候选。已派发训练但保留 probe job name 的 allocation 不再阻断后续双卡探测。
- 参数 2 捕获双干净 H200 Job 321743；先用正式 dispatcher 派发 Find_0 devices2/micro8/workers10，再精确取消旧单卡 321718。
- 核对 321743 为 `RUNNING`、`after_lock` 存在、无 `pre_lock/try_lock/kill_lock`，Dataset smoke 输出和 memory-trial resolved config 已生成，错误匹配 0。
- ExecPlan 和 heartbeat 已更新为当前双 H200 状态，heartbeat 不再运行 acquire 脚本。

## Decisions

- 321743 是当前唯一 Find_0 allocation。不得继续申请 H200，也不得因显存百分比停止；只有真实 OOM、进程、数据或数值错误才在该 allocation 内按 lock 纪律恢复。
- 321743 正式 CPC1 W&B 和若干 optimizer step 稳定后，将 heartbeat 从 15 分钟降回 3 小时。
- 外部 AlphaFold、Python 和 Job 继续完全禁止干预。

## Next Actions

1. 下一次 heartbeat 检查 321743 五步 smoke 的 `exit_code`、`gpu_peak.tsv` 和正式 CPC1 online/首 step。
2. 同时紧凑检查 321107/321540 的 Job、锁、stderr、W&B 与阶段结果。
3. 321743 稳定后更新 ExecPlan/handoff，并把 heartbeat 降为每 3 小时。
4. 持续验收五段训练的 validation30、LR-drop4/1、BEST 和 strict model-only CPC2 谱系。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_h200_acquire_once.sh`
- `tmp/adaligand_stage1_dispatch_replanned_find.sh`
- `tmp/stage1_formal_find_chain.sh`

