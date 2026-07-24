# Handoff: AdaLigand Stage1 heartbeat 纠偏与 Find_0 单 H200

Date: 2026-07-21

## Current State

当前五段训练使用三个 producer：unet_c1 Job 321107（1×H100、micro6/accum8/GBS48）、Find_1 Job 321540（2×H100、micro6/accum4/GBS48）和 Find_0 Job 321718（1×H200、micro8/accum6/GBS48）。Find_2 已在 global step 53 撤回，不再训练。

Find_1 W&B run `qqmuqyxk` 已 online 并至少推进到 global step 86，stderr 无明确错误。Find_0 Job 321718 获得的 H200 在派发前为 1 MiB/0%，当前 Dataset smoke 已完成并进入五步 memory trial，随后会自动启动正式 CPC1→CPC2。

实际激活的 heartbeat 只有 `adaligand-stage1`，目标线程 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf`。周期已从 3 小时改为 15 分钟，提示词只引用当前 Job、val30、五段目标和 H200 单候选状态机。用户看到的旧 Job 321106/321108、旧 W&B、val25 文本是历史快照，不是磁盘中的当前 automation。

## Completed

- 运行 `/home/penghongen/My_Project/tmp/adaligand_stage1_h200_acquire_once.sh 1` 捕获干净单 H200 Job 321718；立即用正式 dispatcher 派发 Find_0 devices1/micro8/workers10。
- 核对 321718 为 `RUNNING`、`after_lock` 存在、无 `pre_lock/try_lock/kill_lock`；Dataset smoke 输出已生成，memory trial resolved config 已开始生成，错误匹配 0。
- heartbeat 更新为每 15 分钟，以低输出方式尝试双 H200 空档；不再使用旧的 3 小时资源探测节奏。
- ExecPlan 已写入 Find_1 online/step86、Find_0 Job 321718 和 heartbeat 纠偏证据。

## Decisions

- 有单卡 Find_0 时，heartbeat 每次只调用 H200 acquire 脚本参数 2，并且任一时刻最多一个 24-CPU 探针。
- 双卡候选只有在两张卡均物理干净并输出 `CLEAN_HELD` 时才保留。先派发双 H200 Find_0（micro8/accum3），再直接 `scancel` 当前单卡 Find_0 精确 Job；脏卡、等待或有队列只记录并释放/跳过。
- 不因正式长跑显存百分比或 W&B 网页异常停止；只处理真实 OOM、进程、数据或数值错误。
- validation 当前要求是每 epoch 30 次，不是历史 25 次。

## Next Actions

1. 下一次 heartbeat 先确认 Job 321718 的五步 smoke exit、峰值和正式 CPC1 online/首 step；只有真实 OOM 才恢复。
2. 同时紧凑检查 321107/321540 的 Job、锁、stderr、W&B 和阶段结果。
3. 调用 H200 acquire 参数 2 抢双卡空档；严格执行单候选和先派发新双卡再取消旧单卡。
4. 持续审计五段训练的 validation30、BEST、LR-drop4/1 和 strict model-only CPC2 初始化谱系。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_h200_acquire_once.sh`
- `tmp/adaligand_stage1_dispatch_replanned_find.sh`
- `CLAUDE/memory/learnings/decision-2026-07-21-训练改动先做合理性核验.md`

