# Handoff: AdaLigand Stage1 Find_2 撤回与 Find_1 双 H100 启动

Date: 2026-07-21

## Current State

正式训练范围已从四 producer/七段收缩回三 producer/五段：unet_c1、Find_0 CPC1→CPC2、Find_1 CPC1→CPC2。Find_2 根据用户新的科学判断停止，不再训练 CPC1 或 CPC2；其源码、smoke、W&B 和日志保留，不做回滚或清理。

当前运行 Job 为 unet_c1 `321107`（1×H100、micro6/accum8/GBS48）和 Find_1 `321540`（2×H100、micro6/accum4/GBS48）。Job 321540 原承载 Find_2，已通过精确 `kill_lock` 返回 exit 143/`try_lock`，保留 `after_lock` 后直接派发 Find_1 CPC1→CPC2，按用户授权跳过 smoke 和科学测试。Find_0 继续等待物理干净的 H200；当前用户名下无 H200 Job。

heartbeat `adaligand-stage1` 为 `ACTIVE`、每 3 小时，目标线程 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf`。下一次先确认 Find_1 W&B online、首 optimizer step 和明确错误状态。

## Completed

- Find_2 online W&B run `30dq7bql` 在 global step 53 保存最后状态：total loss 约 0.5788，pseudo loss 约 0.3025。现有日志、resolved config、GPU 记录和 W&B 均保留。
- 新增 `tmp/adaligand_stage1_dispatch_direct_find1.sh`，服务器副本为 `/home/penghongen/My_Project/tmp/adaligand_stage1_dispatch_direct_find1.sh`。它只用于本次 321540 原地替换，直接运行 Find_1 正式链，不执行 Dataset/memory smoke。
- Find_1 resolved config 已核对为 devices2、micro6、accum4、GBS48、workers10、`lr=5e-5`、warmup0.005、CPC1 patience2、val30、LR-drop4、max20。
- 最多五分钟外部审计结束时，321540 为 `RUNNING`，`after_lock` 存在、无 `try_lock/kill_lock`，GPU 监控文件持续更新；stderr 没有 Traceback、OOM、NCCL error 或 worker-killed。训练仍在模型冷初始化，W&B 状态尚未生成。
- 活 ExecPlan、heartbeat 和项目 learning 已更新。新 learning 为 `CLAUDE/memory/learnings/decision-2026-07-21-训练改动先做合理性核验.md`。

## Decisions

- Find_2 撤回理由：伪原子和真实原子经过 density box 初始化后都可能带入相似密度调制，削弱点云分支只提取密度信息并区分两者的能力；用户同时在 `train_loss/global/pseudo_step` 趋势中看到支持信号。
- Find_2 证据不足以要求删除实现；当前决定是“不再训练”，不是回滚源码或下游枚举。
- 今后用户再次要求修改或重提正式训练时，agent 必须先核查代码语义、当前证据、观察窗口、科学因果和重启代价，明确支持或反对理由；必要时先建议不停止现有训练的只读核验或更小实验。
- W&B 网页或 online 暂时异常仍不是停止健康训练的理由。

## Open Questions

- Find_1 冷初始化后的 W&B online 和首 optimizer step 尚待下一次 heartbeat 确认。
- Find_0 尚未取得干净 H200；H200 仍按单个 24-CPU 探针、先单卡后双卡升级的状态机管理。
- unet_c1、Find_0 CPC1/CPC2、Find_1 CPC1/CPC2 共五段仍需完成 BEST、每 epoch 30 次 validation、LR-drop 停止及 CPC2 strict model-only 初始化审计。

## Next Actions

1. heartbeat 唤醒后紧凑检查 321107/321540 的 Job、锁、stderr、W&B 和阶段结果；先确认 Find_1 online/首 step，不扩大前台审计。
2. 当前无 Find_0 H200 时运行 `/home/penghongen/My_Project/tmp/adaligand_stage1_h200_acquire_once.sh 1`；仅按一行结果处理，不并行枚举卡。
3. Find_0 单卡启动后才探双卡；新双卡均干净并派发后精确取消旧单卡 Job。
4. 持续监控五段训练到全部终态，并在关键里程碑更新 ExecPlan/handoff。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `CLAUDE/memory/learnings/decision-2026-07-21-训练改动先做合理性核验.md`
- `tmp/adaligand_stage1_dispatch_direct_find1.sh`
- `tmp/adaligand_stage1_h200_acquire_once.sh`
- `tmp/stage1_formal_find_chain.sh`
- `tmp/stage1_formal_train_stage.sh`

