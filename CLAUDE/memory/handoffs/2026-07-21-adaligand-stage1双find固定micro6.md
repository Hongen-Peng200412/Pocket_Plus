# Handoff: AdaLigand Stage1 双 Find 固定 micro6

Date: 2026-07-21

## Current State

正式 preparation root 为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`，allocation root 为 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations`。数据清单、四 split、train/validation 三类 BOX pool、发布验收和真实 Dataset/DataLoader smoke 已完成。

三个无时限 sbatch allocation 仍为 `RUNNING`，并全部保留 `after_lock`：Find_1 Job 321108（hnode03，2×H200）、Find_0 Job 321106（hnode01，1×H100）、unet_c1 Job 321107（hnode02，1×H100）。不得在五段训练全部完成前删除 `after_lock` 或取消这些 allocation。

用户最终明确指定两个 Find 都使用 `train.batch_size=6`，共同 `global_batch_size=48`：Find_1 为双卡、accumulation 4；Find_0 为单卡、accumulation 8。unet_c1 保持 batch_size 6、单卡、accumulation 8。

Find_1 已结束仅用于比较的 micro8 trial，并在 Job 321108 内重新派发 micro6 正式 CPC1→CPC2 链，当前正在 cold initialization。Find_0 micro6 的 5-step smoke 正在 Job 321106 内运行，约 10 分钟时已读取约 19.8 GB，stderr 无错误；成功后应立即运行 `tmp/adaligand_stage1_restart_find0_m6_user_final.sh` 启动 micro6 正式链。unet_c1 W&B `ohhl54t1` 已推进到至少 global step 863。

## Completed

- 严格过滤 20,483 PDB/637,140 occurrence；最终资产清单 18,293 PDB/579,688 occurrence，`fallback=false`。
- split 为 train/validation/calibration/held-out = 13,719/200/100/4,274，无 PDB 泄漏。
- BOX pool 已发布并由真实 Dataset/DataLoader 证明 center、bias、context 均可组成 batch。
- Find_1 两个真实 DDP CPU/NCCL validation bug 已最小修复，服务器定向测试 15 passed，并由双 H200 正式 run 通过 sanity validation 和 optimizer step。
- 显存百分比停止脚本已经禁用；它们不能再创建正式训练 `kill_lock`。

## Decisions

- 87%、90%、95% 等显存百分比仅是正式启动前短 smoke 的经验观察值，不是正式训练硬停止线。正式长跑不能仅因显存百分比越线而终止。
- Find_1 旧 run `nj97r8mc` 在 85.098% 被 TERM 属于调度策略误解，不是训练或 W&B 故障；W&B 显示 `crashed` 是外部 TERM 的结果。
- 用户最终决定 Find_0 与 Find_1 都使用 micro6。Find_0 即使 smoke 超过 90% 也不回退到 micro2；只有真实 OOM 才进入最小恢复。
- W&B 在线状态与训练真实状态分离判断。W&B `crashed`、刷新延迟或在线不可用不能单独触发停止；以 Slurm、进程、stderr、GPU 活性和训练日志为准。

## Next Actions

1. 高频监控 Find_0 micro6 smoke；成功后立即运行 `tmp/adaligand_stage1_restart_find0_m6_user_final.sh`。如真实 OOM，保留完整证据并做最小恢复，不按百分比提前回退。
2. 监控 Find_1 micro6 正式冷启动到 W&B 初始化、双 rank rendezvous、sanity validation 和首个 optimizer step；取得新 W&B run ID。
3. 继续监控 unet_c1 `ohhl54t1`，并在第一个 epoch 后审计恰好 25 次 validation。
4. 完成 unet BEST、Find_0/Find_1 CPC1→CPC2、checkpoint 谱系、LR-drop 停止、resolved config/W&B/Slurm 审计后再释放 allocation。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_user_batch_trials_watch.sh`
- `tmp/adaligand_stage1_user_batch_process_probe.sh`
- `tmp/adaligand_stage1_restart_find0_m6_user_final.sh`
- `tmp/adaligand_stage1_restart_find1_m6_user_final.sh`
- `tmp/adaligand_stage1_formal_health_live.sh`
- `tmp/stage1_wandb_live_probe.py`

