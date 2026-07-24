# Handoff: AdaLigand Stage1 三 producer 首验完成

Date: 2026-07-22

## Current State

三个正式 Job 均健康运行且已完成至少一次正式 validation：

- unet_c1 Job `321107`：1xH100，micro6/accum8/GBS48，W&B `0yy28kmj`，summary step3209，epoch0 validation 2次，最新 val loss 0.2356146574。
- Find_1 CPC1 Job `321540`：2xH100，micro6/accum4/GBS48，W&B `qqmuqyxk`，step1673，epoch0 validation 1次，val loss 0.4114735126。
- Find_0 CPC1 Job `321743`：2xH200，micro8/accum3/GBS48，W&B `w21l8dof`，step1529，epoch0 validation 1次，val loss 0.4041147530。

三个 `after_lock` 均存在，无其它锁；严格 stderr 错误匹配为0，W&B stream 与 GPU 监控持续更新。三个 run 的 LR reduction count 均为0，尚无阶段 `result.env`。

## Completed

- W&B API 正式 validation 审计：unet epoch0 history step `[3944,7894]`；Find_1 `[2191]`；Find_0 `[1752]`。history step 是 W&B 行序号，不是 trainer global step。
- 三个阶段均确认 TOP/last checkpoint：unet TOP score0.2356、Find_1 score0.4115、Find_0 score0.4041。
- 审计 JSON 位于 `tmp/stage1_audits/`，活 ExecPlan 已记录本轮证据。

## Decisions

- 当前 checkpoint 是训练中的 TOP，不提前视为最终 BEST；继续让训练按 LR-drop4/20 epoch 上限自然推进。
- 不因高显存或 W&B 网页节奏停止；本轮没有真实恢复条件。
- 不运行 H200 acquire；heartbeat 继续每6小时。

## Next Actions

1. 累计并审计 epoch0 validation 次数；完整 epoch 必须恰好30次。
2. 持续跟踪三个阶段的 TOP/last、LR reduction count、阶段停止原因与 `result.env`。
3. Find CPC1 完成时核验 CPC2 从同名 CPC1 BEST strict model-only 初始化；CPC2 在第1次实质 LR下降后停止。
4. 五段终态后执行 checkpoint 可加载性、日志/W&B总审计、ExecPlan/handoff收口和脚手架分类。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_heartbeat_probe.sh`
- `tmp/stage1_wandb_validation_audit.py`
- `tmp/stage1_audits/`
