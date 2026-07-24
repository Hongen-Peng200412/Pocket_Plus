# Handoff: AdaLigand Stage1 unet 首次 validation 已确认

Date: 2026-07-22

## Current State

三个正式 Job 仍健康运行：unet_c1 `321107`（1xH100，micro6/accum8/GBS48，W&B `0yy28kmj`，summary step2375）、Find_1 `321540`（2xH100，micro6/accum4/GBS48，W&B `qqmuqyxk`，step1274）和 Find_0 `321743`（2xH200，micro8/accum3/GBS48，W&B `w21l8dof`，step1067）。三个 `after_lock` 均存在，无其它锁；严格 stderr 错误匹配均为0。

unet 首个正式 validation 已确认：W&B `pencounkdual-111/AdaLigand_Stage1/0yy28kmj` epoch0 有 1 条 `val_loss/global/total=0.2606383264064789`，history step3944，run 仍 `running`，LR reduction count 最大值0。该 validation 之后已生成：

`/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-unet_c1/unet_c1____job321107_unet_c1_lr5e5_p2_val30_m6_w1/checkpoints/TOP_epoch_00_score_0.2606.ckpt`

以及同目录 `last.ckpt`。Find_1/Find_0 尚未到首个约 step1316 validation。

## Completed

- 完成当前三个 Job、锁、stderr、W&B summary/stream、GPU 活性、stage result 和 checkpoint 的只读巡检。
- 修复 `tmp/stage1_wandb_validation_audit.py`：validation 和 LR history 改为独立 `scan_history`，避免 W&B 联合 key 筛选丢失 validation 行。
- 新增并通过 `tmp/test_stage1_wandb_validation_audit.py`：`1 passed`；审计脚本与测试 `py_compile` 通过；真实 unet 审计 JSON 已写入 `tmp/stage1_audits/unet_c1_0yy28kmj_validation.json`。
- 优化 `tmp/adaligand_stage1_heartbeat_probe.sh` 使用固定 run 根路径，并分别报告 TOP/last checkpoint；远端 `bash -n` 通过。

## Decisions

- 不因首次 validation 后 summary/计算节奏变化而停止健康 unet；当前 W&B、GPU、锁和错误证据一致正常。
- 不加载 checkpoint 做重型验证；普通 helper 只确认路径、大小和 W&B 指标，后续在合适的正式验证环境执行可加载性审计。
- 不调用 H200 acquire，不提交探针；heartbeat 保持每6小时。

## Next Actions

1. 在 Find_1/Find_0 接近 step约1316时确认各自首个 val loss、validation 计数和 artifact。
2. 继续审计每 epoch 恰好30次 validation、CPC1/unet LR-drop4、CPC2 LR-drop1、BEST与同名CPC1 BEST strict model-only初始化。
3. 五段完成后执行 checkpoint 可加载性、停止原因、W&B/Slurm日志总审计，收口 ExecPlan/handoff 并释放 allocation。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_heartbeat_probe.sh`
- `tmp/stage1_wandb_validation_audit.py`
- `tmp/test_stage1_wandb_validation_audit.py`
