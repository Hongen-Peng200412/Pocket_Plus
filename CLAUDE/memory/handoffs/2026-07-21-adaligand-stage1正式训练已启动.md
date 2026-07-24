# Handoff: AdaLigand Stage1 正式训练已启动

Date: 2026-07-21

## Current State

正式数据、四个 split、train/validation 三类 BOX pool、全量 BOX 验收、真实 Dataset/DataLoader smoke 和三 producer 显存试跑均已完成。正式 preparation root 是 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`，正式 allocation scope 是 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/`。活 ExecPlan `talk/Excx_执行stage1端到端训练.md` 是恢复与审计的首要本地记录。

三个无时限 sbatch allocation 均为 `RUNNING`，并已在约 05:18+08:00 启动正式链：Find_1 Job 321108（hnode03，2×H200 NVL）、Find_0 Job 321106（hnode01，1×H100 PCIe）、unet_c1 Job 321107（hnode02，1×H100 PCIe）。全部 `after_lock` 保留，五段训练全部结束前不得释放。Find_0 CPC1 与 unet_c1 已在线接入 W&B 并进入 Trainer；Find_1 CPC1 仍处于较慢的双 rank 冷初始化窗口，当前无异常证据，需继续高频监控到 W&B、DDP 和首批训练稳定。

## Completed

- 严格过滤 `cc_contour > 0.6` 且 map resolution `< 7.0 Å` 得到 20,483 PDB/637,140 occurrence；简单资产检查后正式采用 18,293 PDB/579,688 occurrence，2,190 个排除项全部为 exp/sim geometry mismatch，`fallback=false`。
- 唯一 PDB split 为 train/validation/calibration/held-out = 13,719/200/100/4,274，无跨 split 泄漏，validation/calibration 恰好 200/100 且满足既有 80³ eligibility。
- BOX Job 321111 `COMPLETED/0:0`；正式发布 train 13,715 PDB、validation 200 PDB。优化验收 Job 321141 `COMPLETED/0:0`，`box_pool/verification.json` 状态 `ok`。
- 三 producer 真实 smoke 均从正式清单/BOX pool 读取 center、bias、context 并组成 batch。
- 冻结共同 global batch 48：Find_1 双 H200 micro=6/accum=4、峰值 67.811%/68.773%；Find_0 单 H100 micro=4/accum=12、峰值 78.998%；unet_c1 单 H100 micro=6/accum=8、峰值 80.006%。
- 正式 resolved config 已核对共同 GBS、各 micro、DDP unused 参数策略、`max_epochs=20`、`val_per_epoch=25`、CPC1/unet 停止阈值 4、`offline=false`、首段 `init_from=null`。
- Find_0 W&B run：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/s04troim`；unet_c1 W&B run：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/ohhl54t1`。

## Decisions

- 不修改 Pocket_Plus Stage1 科学源码；当前仍是 commit `de6a89f38e48553edfd440c76bf61823229d9205`。Hydra、allocator、调试 warmup 和双卡 DDP unused-parameter 问题全部限于 `tmp/` run-scoped 包装恢复。
- 五段共用 global batch 48。Find CPC2 只能从同名 CPC1 BEST 做 model-only 初始化，且 CPC1 checkpoint 必须保留；正式 Find 链脚本已把这一条件做成失败即停的门禁。
- W&B 默认在线；只有在线初始化在重试后确实反复失败，正式包装才允许离线回退并记录原因。

## Open Questions

- Find_1 冷初始化何时完成，双 rank DDP/W&B/首批训练是否稳定。
- 首个完整 epoch 的 25 次 validation 真实计数、单 epoch 用时和各模型实际 LR reduction 节奏。
- CPC1 结束后 CPC2 的 model-only 日志、BEST 可加载性与最终停止原因。

## Next Actions

1. 高频监控 Find_1 `formal/Find_1_CPC1/launch_1/{train.out,train.err}`，确认 `wandb_status=online_initialized`、两个 rank 启动、NCCL 正常及 GPU 满载。
2. 同时监控 Find_0/unet 的首次 optimizer step、OOM/数据错误、W&B 和 GPU 利用率；稳定后降低轮询频率但不断开监控。
3. 首个完整 epoch 后分别审计本地与 W&B validation 计数恰好 25，并持续记录 LR reduction、BEST 与停止原因。
4. Find CPC1 完成时核对 CPC1 BEST，再盯住同一 allocation 的 CPC2 model-only 初始化、在线 W&B、首个 batch 和最终 BEST。
5. 每个关键里程碑更新活 ExecPlan；五段全部正确终态后做 W&B/配置/checkpoint/日志总审计和临时脚手架分类，再生成最终 handoff。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/stage1_formal_train_stage.sh`
- `tmp/stage1_formal_find_chain.sh`
- `tmp/stage1_formal_unet.sh`
- `tmp/adaligand_stage1_monitor_formal.sh`
- `tmp/stage1_wandb_validation_audit.py`
