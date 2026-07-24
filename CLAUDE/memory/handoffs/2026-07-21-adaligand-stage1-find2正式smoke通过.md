# Handoff: AdaLigand Stage1 Find_2 正式 smoke 通过

Date: 2026-07-21

## Current State

新版 Find_1 Job 321388 与 Find_2 Job 321373 都是 hnode03 上的单 H200、24 CPU、无时限 sbatch allocation，`after_lock` 保留。两条 5 optimizer-step smoke 均成功，正式 CPC1 已于约 14:31+08:00 启动，当前正在第二轮完整 DataModule 冷加载，尚未到 `wandb.init()`；W&B online gate 均在第一次尝试成功。

Find_0 Job 321106 与 unet_c1 Job 321107 继续健康训练，最近在线 W&B step 分别至少 374 与 542。活执行日志是 `talk/Excx_执行stage1端到端训练.md`。

## Completed

- 服务器正式环境发布测试在两个新 allocation 上均为 42 passed；本地主线整合测试为 67 passed。
- Find_1/Find_2 真实 Dataset/DataLoader smoke 都返回 `status=ok`，专用读取 batch 固定为一个 center、bias、context，shape `[3,56,80,80,80]`。这里的 3 不是训练 batch。
- 训练 smoke 的运行时日志均确认 micro-batch 6；共同 GBS48、单卡 accumulation 8。
- Find_1 smoke：loss 0.805482→0.667957，peak 103,682/143,771 MiB=72.116%，utilization 100%，exit 0。
- Find_2 smoke：loss 0.803148→0.666263，peak 99,718/143,771 MiB=69.359%，utilization 100%，exit 0。
- 正式 resolved config：Find_1 Gaussian=true/tune=false；Find_2 Gaussian=true/tune=true；两者均 GBS48/micro6/device1、val25、warmup0.005、stop-after-4、max_epochs20、offline=false、init_from=null。

## Decisions

- 用户本人于约 14:02+08:00 手动 `scancel 321372`，原因是该 allocation 被分到与本任务外 `alphafold3/run_af_json.py --card 4` 相同的物理 H200。agent 没有取消 321372。
- AlphaFold 任务只允许只读查验，绝不能发送信号、修改或干预；不要猜测其所有者。
- 替代 Find_1 Job 321388 的 GPU UUID 为 `GPU-4af0ba5d-b718-0a8d-01bc-670b512c2772`，启动前为 1 MiB/0%、无计算进程；Find_2 Job 321373 的 UUID 为 `GPU-38991406-7c95-fb26-a5ff-768cde15f172`。
- 不因正式训练 W&B 网页异常或显存百分比停止健康任务；只有真实 OOM、代码/数据/数值错误才恢复。

## Open Questions

- 正式 Find_1/Find_2 仍需完成 online W&B 初始化、达到各自至少 50 optimizer step，并密集监控至少 30 分钟。
- 达到稳定门槛后才能更新并恢复每 3 小时 heartbeat `adaligand-stage1`；当前 automation 仍为 PAUSED，旧 prompt 中的 321108/五段范围也需要更新为四 producer/七段。
- 七段训练的最终 BEST、LR-drop 停止、validation25 与 CPC1→CPC2 strict model-only 谱系仍未完成。

## Next Actions

1. 持续只读监控 321388/321373 的 DataModule I/O、stderr、W&B status 与 GPU；不得干预 AlphaFold。
2. 记录两个正式 online W&B run ID，核对首个 optimizer step、GBS48/micro6/accum8、val25 和 warmup0.005。
3. 两条新 run 都达到至少 step50 且密集监控超过 30 分钟后，更新并恢复 `adaligand-stage1` heartbeat。
4. 继续监控 321106/321107，不重提 Find_0 双 H100 分支。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `CLAUDE/memory/handoffs/2026-07-21-adaligand-stage1-find2实现与单h200交接.md`
- `tmp/stage1_formal_train_stage.sh`
- `tmp/stage1_formal_find_chain.sh`
- `src/model/stage1_embed_head.py`
- `src/model/stage1_model.py`
