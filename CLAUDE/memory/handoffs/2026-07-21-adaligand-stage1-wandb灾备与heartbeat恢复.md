# Handoff: AdaLigand Stage1 W&B 灾备与 heartbeat 恢复

Date: 2026-07-21

## Current State

四条正式 sbatch 于 2026-07-21 15:25+08:00 均为 `RUNNING`：Find_0 Job 321106（hnode01，1×H100）、unet_c1 Job 321107（hnode02，1×H100）、Find_2 Job 321373（hnode03，1×H200）和 Find_1 Job 321388（hnode03，1×H200）。当前正式参数统一为 `global_batch_size=48`、`micro_batch_size=6`、单卡 `gradient_accumulation=8`、`val_per_epoch=25`、`warmup_ratio=0.005`。

四条当前 W&B run 都已 online 成功：Find_0 `wm5e3v72`、unet_c1 `tp6k1gar`、Find_1 `wdej8848`、Find_2 `s7j101zn`。在线 W&B 不可用不是训练终止条件；本地 `.wandb` 文件可作为审计和事后补同步来源。

heartbeat `adaligand-stage1` 已恢复为 `ACTIVE`、每 3 小时，目标线程已复核为 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf`。旧本地 30 秒前台轮询已终止，不影响服务器训练。

## Completed

- 在 `../AdaLigand/与服务器交互/` 新增 `sync_stage1_wandb_remote.bat` 与 `sync_stage1_wandb_remote.ps1`。双击 BAT 会扫描固定正式根 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations` 下全部 `offline-run-*`，用独立本地快照执行 `wandb sync --no-mark-synced`，不改动或删除既有本地/服务器日志。
- 主线独立复验 PowerShell AST 和 BAT `-DryRun`。当前枚举 17 个离线 memory smoke run，全部属于 `Find_0`、`Find_1`、`Find_2`、`unet_c1`；dry-run 明确没有下载或上传。
- 只读吞吐审计量化当前窗口：Find_0/H100 为 `54.553 s/optimizer-step`，Find_1/H200 为 `60.725 s/step`，Find_2/H200 为 `60.711 s/step`。H200 两条表观慢约 11.3%，但不能据此判定 H200 硬件退化或改变调度。
- 没有执行 `scancel`、`kill_lock`、Job 重提或 AlphaFold 操作。外部 `alphafold3/run_af_json.py --card 4` 仍只允许被动只读看见，绝不能干预。

## Decisions

- 用户撤销“新 Find 至少密集监控 30 分钟且各到 step50 才可休眠”的门槛；W&B 曲线已正常绘制后即可由 heartbeat 接管。
- W&B 网页 crashed、online 暂时失败或同步延迟不得终止健康训练。只有真实 OOM、训练进程退出、代码/数据/数值错误才进入恢复。
- 当前 H200/H100 差异最可能混合了 Find_1/2 的 3×3×3 Gaussian scatter、hnode03 两任务共驻资源竞争，以及不匹配的 global-step/原子数/recycle passes 窗口。继续训练，后续 heartbeat 只读积累同一步区间证据。
- 固定一键同步根之外未来新增的 Stage1 run scope 不会自动纳入；若正式运行根发生变化，必须显式更新包装器，不做全服务器宽扫描。

## Open Questions

- 七段训练仍未完成：unet_c1、三个 Find CPC1 的第 4 次实质 LR 下降或 20 epoch 上限，及三个 Find CPC2 的 strict model-only 初始化、第 1 次实质 LR 下降或 20 epoch 上限仍待监控。
- 每个阶段仍需审计每 epoch 恰好 25 次 validation、BEST 可加载、resolved config、停止原因、W&B 状态和 CPC1→CPC2 checkpoint 谱系。
- 当前吞吐结论不是严格卡型基准；后续只有在相同步区间并控制原子数/recycle passes、平均 GPU util/功率/时钟和 DataLoader wait 后才能进一步归因。

## Next Actions

1. heartbeat 唤醒后先读活 ExecPlan和本 handoff，再只读核对 Job 321106/321107/321388/321373、锁、stderr、GPU、W&B 和阶段结果。
2. 不因显存百分比或 W&B 在线异常停止健康训练；未经用户新授权不得 scancel、kill_lock、重提或触碰 AlphaFold。
3. 在首次 validation 窗口及每个阶段结束时审计 validation25、BEST、LR-drop 停止和 CPC1→CPC2 strict model-only 谱系，并更新 ExecPlan/handoff。
4. 七段全部完成前保留各 allocation 的 `after_lock`。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `../AdaLigand/与服务器交互/sync_stage1_wandb_remote.bat`
- `../AdaLigand/与服务器交互/sync_stage1_wandb_remote.ps1`
- `tmp/stage1_formal_train_stage.sh`
- `tmp/stage1_formal_find_chain.sh`
