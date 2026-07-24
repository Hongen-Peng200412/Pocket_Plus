# Handoff: AdaLigand Stage1 unet 首个 val30 窗口

Date: 2026-07-22

## Current State

三个正式 Job 均健康运行：unet_c1 Job `321107`（1xH100，micro6/accum8/GBS48，W&B `0yy28kmj`）、Find_1 Job `321540`（2xH100，micro6/accum4/GBS48，W&B `qqmuqyxk`）和 Find_0 Job `321743`（2xH200，micro8/accum3/GBS48，W&B `w21l8dof`）。2026-07-22 01:05+08:00 的最新 summary step 分别为 1316、653、485。

三个 `after_lock` 均存在，无 `try_lock`、`pre_lock` 或 `kill_lock`。当前三份 stderr 的严格错误匹配均为 0，没有阶段 `result.env` 或 BEST。Find_2 仍保持撤回，不再训练。

unet 在 step1316 暂停常规 step summary，但 W&B internal stream 秒级刷新，GPU 利用率 97%–100%。依据 315,903 micro-batch/epoch、`val_per_epoch=30` 和 accum8，首次 validation 在 micro-batch 10,530、optimizer step 约 1316 触发；现状高度一致于完整 validation 正在执行，不是停滞。首个 val metric 尚未落盘，下一轮仍须确认完成结果。

## Completed

- 用当前精确 Job ID 完成 Slurm、锁、stderr、W&B summary/stream、GPU 监控、阶段结果和 checkpoint 交叉巡检。
- 新增 `tmp/adaligand_stage1_heartbeat_probe.sh` 作为只读低输出探针；它不写服务器、不触碰锁或外部任务。
- 把本轮证据写入活 ExecPlan。健康状态下 heartbeat 按用户授权从 3 小时调整为 6 小时。

## Decisions

- 不因 unet summary 在 validation 期间暂时不增长而恢复训练；W&B stream、GPU 活性和零错误共同证明进程仍健康。
- 不调用 H200 acquire，不提交 H200 探针；当前已使用 2 张 H200。
- 只有真实 OOM、进程、数据或数值错误才进入 run-scoped 恢复。

## Next Actions

1. 下次先确认 unet 首次 `val_loss/global/total` 已写入，并审计该次 validation 的产物和错误状态。
2. Find_1/Find_0 接近 step约1316 时做同样的首次 val30 审计。
3. 继续监控 CPC1/unet LR-drop4、CPC2 LR-drop1、BEST 和 Find 同名 CPC1 BEST strict model-only 初始化谱系。
4. 五段完成后审计 checkpoint 可加载性、收口 ExecPlan/handoff、分类临时脚手架并释放 allocation。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_heartbeat_probe.sh`
- `tmp/stage1_wandb_validation_audit.py`
- `tmp/stage1_formal_find_chain.sh`
