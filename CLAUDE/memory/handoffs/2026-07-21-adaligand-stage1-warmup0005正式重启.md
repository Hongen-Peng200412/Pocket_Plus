# Handoff: AdaLigand Stage1 warmup 0.005 正式重启

Date: 2026-07-21

## Current State

数据、split、BOX pool、正式发布和真实 Dataset/DataLoader smoke 已完成。三个无时限 allocation 均为 `RUNNING` 且保留 `after_lock`：Find_0 Job 321106（hnode01，1×H100）、unet Job 321107（hnode02，1×H100）、Find_1 Job 321108（hnode03，2×H200）。

最终共同 `global_batch_size=48`，三个 producer 都使用 per-device `batch_size=6`：Find_0 accumulation 8，unet accumulation 8，Find_1 accumulation 4。Find_0 micro6 的 5-step smoke 完成 40 个 micro-batch，exit 0，峰值 99.226%，无 OOM；用户明确要求仍使用 micro6，正式训练不得因显存百分比终止。

`configs/train/stage1_cpc1.yaml` 已把 `val_per_epoch` 从 10 改为 25，把 `warmup_ratio` 从 0.025 改为 0.005。服务器共享配置已非删除式同步；三份在途 resolved config 都确认 GBS48、micro6、val25、warmup0.005、CPC1/unet stop-after-4。CPC2 保持既定无 warmup、stop-after-1。

当前 W&B：Find_0 `wm5e3v72` 至少 step 20；Find_1 `2py4thx4` 至少 step 23；新 unet run 尚未完成 W&B 初始化，PID/PGID 232266 正在读取正式 pool，约 20 分钟时已读约 39.3 GB，stderr 无错误。

## Completed

- 默认 validation 契约改为每 epoch 25 次；四个 Find 配置和 unet_c1 组合配置回归测试 `7 passed`。
- 三个从头训练 producer 统一改用 `warmup_ratio=0.005` 并从头重启。
- 旧 unet `ohhl54t1` 的 setsid 子进程没有随 lock wrapper TERM 退出；已用 run stamp、PID/PGID/SID 精确识别 PGID 183780，在 hnode02 内 KILL。新 PGID 232266 未受影响。
- H100 实时资源核验表明每个节点最多只有一张空卡，无法无排队取得同节点 2×H100；用户确认忽略双卡 Find_0 建议，当前 Find_0 不停止。
- `stage1_formal_find_chain.sh` 经独立核验严格先完成 CPC1、验证真实 BEST，再用同名 CPC1 BEST strict model-only 初始化 CPC2。

## Decisions

- Find_0 保持单 H100/micro6/accum8；不再为双 H100 方案监控或切换，除非用户重新提出。
- 显存百分比只记录 smoke 观测，不是正式长跑停止条件；真实 OOM 才触发恢复。
- W&B 页面状态不是训练终态来源；继续以 Slurm、PID、stderr、GPU 和训练步数交叉判断。
- CPC1→CPC2 严格串行；CPC1 checkpoint 保留，CPC2 只恢复模型权重。

## Next Actions

1. 监控新 unet 完成 W&B 初始化并进入 optimizer step，记录新 run ID。
2. 持续监控 Find_0 `wm5e3v72`、Find_1 `2py4thx4` 的 GPU、错误、W&B 和首次 validation。
3. 首个 epoch 后审计每个 epoch 恰好 25 次 validation；跟踪 warmup、LR reduction、BEST。
4. Find CPC1 完成后验证 CPC2 init source 等于同名 CPC1 BEST，并保留两段 checkpoint。
5. 五段全部完成前不删除 `after_lock`、不取消 allocation。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_formal_health_live.sh`
- `tmp/adaligand_stage1_formal_gpu_peak_live.sh`
- `tmp/stage1_wandb_live_probe.py`
- `tmp/stage1_wandb_validation_audit.py`
- `tmp/stage1_formal_find_chain.sh`
- `tmp/stage1_formal_train_stage.sh`

