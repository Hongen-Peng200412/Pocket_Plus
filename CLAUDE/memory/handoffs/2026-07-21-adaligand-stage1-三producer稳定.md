# Handoff: AdaLigand Stage1 三 producer 稳定

Date: 2026-07-21

## Current State

三个正式 producer 均已 online 并产生 optimizer step：unet_c1 Job 321107（1×H100，W&B `0yy28kmj`，至少 step587）、Find_1 Job 321540（2×H100，W&B `qqmuqyxk`，至少 step173）和 Find_0 Job 321743（2×H200，W&B `w21l8dof`，至少 step20）。三者均为 epoch0。

Find_0 resolved config 为 devices2、micro8、accum3、GBS48、workers10、LR 5e-5、warmup0.005、CPC1 patience2、val30、LR-drop4、max20。双 H200 smoke 已 exit0，无 OOM。

三个 Job 均 `RUNNING`、`after_lock` 存在、无 `try_lock/kill_lock`；近 30 分钟明确错误匹配为 0。Find_2 已撤回。

heartbeat `adaligand-stage1` 已从 15 分钟降回每 3 小时。当前占满 2 张 H200，不再调用 acquire 或提交探针。

## Decisions

- 启动阶段已结束，后续进入长期监控。W&B 网页异常和显存百分比不触发停止，真实 OOM/进程/数据/数值错误才恢复。
- W&B summary 必须按文件 mtime 选择，不能用路径字典序判断最新 run。
- 继续验收五段训练，不恢复 Find_2。

## Next Actions

1. 每 3 小时紧凑检查三个 Job、锁、stderr、最新 W&B summary 和阶段结果。
2. 接近首次 validation 时审计每 epoch 恰好 30 次，而不是沿用历史 step1579/val25 推算；应基于当前 resolved batch 数重新确定窗口。
3. CPC1 完成时核验 BEST、LR-drop4 和停止原因；Find CPC2 必须从同名 CPC1 BEST strict model-only 初始化，并在 LR-drop1 后停止。
4. 关键里程碑更新 ExecPlan/handoff；全部完成后删除 heartbeat。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/stage1_formal_train_stage.sh`
- `tmp/stage1_formal_find_chain.sh`

