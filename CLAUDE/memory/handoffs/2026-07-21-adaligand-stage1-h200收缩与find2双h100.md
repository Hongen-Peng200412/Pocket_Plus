# Handoff: AdaLigand Stage1 H200 收缩与 Find_2 双 H100 启动

Date: 2026-07-21

## Current State

截至 2026-07-21 17:32+08:00，正式运行中的 Job 只有 unet_c1 `321107`（hnode02，1×H100，micro6/accum8/GBS48）与 Find_2 `321540`（hnode01，2×H100，micro6/accum4/GBS48）。当前 `penghongen` 名下没有 H200 Job；Find_0 等待一张物理干净的 H200，Find_1 在 Find_0 释放 H200 后恢复。

生产配置已统一为 CPC1 `lr=5e-5`、`patience=2`、`val_per_epoch=30`、`warmup_ratio=0.005`；CPC2 继承 LR/validation，但保持 `patience=1`。CPC1/CPC2 分别在第 4/1 次实质 LR 下降后停止，`max_epochs=20`。Find density cube 的 P/real chunk 为 2048/4096。

Find_2 Job 321540 已完成双 H100 五步 smoke，`exit_code=0`，显存峰值 99.273%/99.270%，没有 OOM。正式 CPC1 W&B run `30dq7bql` online 初始化成功、错误匹配 0，已推进到至少 global step 11。按用户规则，正式训练不因显存百分比停止，只有真实 OOM 或训练错误才恢复。

heartbeat `adaligand-stage1` 已恢复为 `ACTIVE`、每 3 小时，目标线程 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf`。

## Completed

- 旧 Find_0 `321106`、Find_1 `321388`、Find_2 `321373` 已自然 `COMPLETED/0:0`；旧双 H200 Find_0 候选 `321539` 因两卡均被外部进程占用而精确取消。
- H200 候选 321555、321562、321563、321568、321569 均已取消或被调度器拒绝；它们只停在 `pre_lock`，没有在脏卡上启动训练。外部 AlphaFold/Python 从未被发送信号。
- 新增本地 `tmp/adaligand_stage1_h200_acquire_once.sh`，服务器副本为 `/home/penghongen/My_Project/tmp/adaligand_stage1_h200_acquire_once.sh`。脚本每次最多提交一个 24-CPU H200 候选，30 秒内未运行或实测脏卡立即精确取消；真实 Job 321607 已验证输出 `DIRTY_RELEASED` 后释放，随后用户名下 H200 队列为空。
- 活 ExecPlan 已更新到最新 LR/patience/val30/chunk2x、Find_2 双 H100 状态、H200 探针证据和资源状态机。

## Decisions

- 无 Find_0 H200 时只运行单卡探针。输出 `CLEAN_HELD` 后，用 `adaligand_stage1_dispatch_replanned_find.sh JOB Find_0 1 8 10` 启动单 H200 Find_0，得到 micro8/accum6/GBS48。
- 单卡 Find_0 正常运行后才允许双卡探针。只有新双卡 allocation 两张都物理干净并已用 `... JOB Find_0 2 8 10` 派发成功，才直接 `scancel` 原单卡 Find_0 精确 Job，得到 micro8/accum3/GBS48。
- 任一时刻最多存在一个 H200 探针；`DIRTY_RELEASED`、`PENDING_RELEASED` 或 `SKIP_QUEUE` 只记录并等待下次 heartbeat，不并行枚举卡索引。
- 绝不触碰外部 AlphaFold、外部 Python 或外部 Job。W&B 网页异常也不是终止健康训练的理由。

## Open Questions

- Find_0 尚未获得干净 H200，Find_1 尚未恢复；三条 Find 的 CPC1→CPC2 和 unet_c1 终态均未完成。
- 七段训练仍需审计每 epoch 恰好 30 次 validation、BEST 可加载、停止原因、W&B/Slurm/本地日志及 CPC2 strict model-only 初始化谱系。
- Find_2 双 H100 的 99.27% smoke 峰值没有 OOM，但余量很小；正式运行只监控真实 OOM，不使用百分比杀任务。

## Next Actions

1. heartbeat 唤醒后先紧凑检查 321107/321540 的 Job、锁、stderr、W&B、GPU 与阶段结果。
2. 当前无 H200 Find_0 时运行 `/home/penghongen/My_Project/tmp/adaligand_stage1_h200_acquire_once.sh 1`；仅按单行结果处理，不做宽扫描。
3. 单卡 Find_0 启动后，后续 heartbeat 才运行参数 `2`；双卡派发成功后精确取消旧单卡 Job，并立即更新 ExecPlan/handoff。
4. Find_0 完成并释放 H200 后恢复 Find_1；持续监控七段训练至 BEST、validation30 和 checkpoint 谱系全部验收。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_h200_acquire_once.sh`
- `tmp/adaligand_stage1_dispatch_replanned_find.sh`
- `tmp/stage1_formal_find_chain.sh`
- `tmp/stage1_formal_train_stage.sh`

