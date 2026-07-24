# Handoff: AdaLigand Stage1 validation 窗口纠正

Date: 2026-07-21

## Current State

AdaLigand Stage1 的三个最终 run 继续健康训练：Find_0 Job 321106（1×H100，W&B `wm5e3v72`）、unet_c1 Job 321107（1×H100，W&B `tp6k1gar`）、Find_1 Job 321108（2×H200，W&B `2py4thx4`）。三者共同 `global_batch_size=48`、per-device `batch_size=6`、`val_per_epoch=25`、CPC1/unet `warmup_ratio=0.005`；Find_0/unet accumulation 8，Find_1 accumulation 4。

截至 2026-07-21 11:46+08:00，三个 Job 均为 `RUNNING`，`after_lock` 存在，`try_lock`/`kill_lock` 不存在，正式 stderr 的 OOM/Traceback/NCCL/worker-killed 匹配均为 0。W&B 的 Find_0/Find_1/unet 分别至少推进到 global step 98/98/83。当前任务 heartbeat `adaligand-stage1` 已设为 `ACTIVE`，每 3 小时唤醒并继续监控。

## Completed

- 查明旧 unet run `ohhl54t1` 在 global step 968 没有正式 validation 属正常调度，而不是验证漏跑或 W&B 上传故障。替换动作在约 step 918–919 开始，旧 `setsid` 孤儿继续运行后才停在 step 968。
- 旧 `train.out` 显示 epoch 0 停在 micro-batch 7,749/315,903。Lightning 2.2.5 使用 `val_check_interval=0.04`，首验在 `int(315903×0.04)=12636` 个 micro-batch；micro6/accum8 下对应 global step≈1,579/1,580。
- 旧 resolved config 明确 `val_per_epoch=25`、`check_val_every_n_epoch=1`，且没有截断 validation。W&B API 扫描 2,903 行，max global step=968，validation keys/rows 均为空；本地日志同样无正式 validation loop。启动时的 2-batch sanity validation 不是正式完整 validation。`315903 = 25×12636+3`，因此 Lightning 每个 epoch 恰好触发 25 次验证，不会在 epoch 末额外触发第 26 次。
- 纠正此前从 W&B 相邻 `warmup_lr` 增量得到的错误推算：`log_every_n_steps=3` 表示相邻记录跨 3 个 optimizer step。正确 warmup 约 3,948 step，总预算约 789,600 step，每 epoch 约 39,488 step，而不是 1,316/263,200/13,160。

## Decisions

- 当前三条 run 的首次正式 validation 高关注窗口统一改为 global step≈1,579/1,580；届时交叉检查训练日志、W&B `val_loss/global/total` 和 validation artifact。
- Find_0 双 H100 切换建议已被用户彻底撤销。即使后续有两张空闲 H100，也不监控、不重提、不替换 Job 321106，除非用户再次主动提出。
- W&B 曲线缺失或网页状态异常不单独构成训练停止依据；继续以 Slurm、进程、stderr、GPU 与实际训练进度判断健康。
- 正式训练中的显存百分比只记录，不触发停止；只有真实 OOM 或其它明确阻断错误才进入恢复。
- 用户已授权在本次历史核验和记忆记录完成后休眠；使用绑定当前任务的 heartbeat `adaligand-stage1` 每 3 小时恢复，不创建独立状态分支。

## Next Actions

1. 持续监控 Job 321106/321107/321108 的进度、错误、W&B、锁和 GPU 活性。
2. 在各 run global step≈1,579/1,580 时验证第一次完整 validation 确实发生，并开始审计每 epoch 恰好 25 次。
3. 后续跟踪 BEST、LR reduction、CPC1→CPC2 的 strict model-only 初始化和五段终态。
4. 五段全部完成前保留 allocation 与 `after_lock`；完成后再做脚手架分类、ExecPlan 收口和最终 handoff。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_formal_health_live.sh`
- `tmp/adaligand_stage1_formal_gpu_peak_live.sh`
- `tmp/stage1_wandb_live_probe.py`
- `tmp/stage1_wandb_validation_audit.py`
- `CLAUDE/memory/handoffs/2026-07-21-adaligand-stage1-warmup0005正式重启.md`
