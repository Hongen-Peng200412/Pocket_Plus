# Handoff: AdaLigand Stage1 Find 恢复并稳定训练

Date: 2026-07-21

## Current State

正式数据、四个 split、train/validation 三类 BOX pool、全量 BOX 验收和真实 Dataset/DataLoader smoke 均已完成。首要恢复记录是 `talk/Excx_执行stage1端到端训练.md`。正式 preparation root 为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`，allocation root 为 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations`。

三个无时限 sbatch allocation 仍为 `RUNNING`，全部保留 `after_lock`：Find_1 Job 321108（hnode03，2×H200 NVL）、Find_0 Job 321106（hnode01，1×H100 PCIe）、unet_c1 Job 321107（hnode02，1×H100 PCIe）。不得在五段训练完成前删除 `after_lock` 或取消这些 allocation。

- Find_1 CPC1 成功恢复 run 是 `nj97r8mc`，已越过 2/2 sanity validation 并推进到至少 `trainer/global_step=80`；双 H200 长跑峰值当前 79.188%/80.147%，GPU utilization 均采样到 100%，仍低于 85% 门限。
- Find_0 已从不安全的 micro=4 改为 micro=2，正式 freeze 为 GBS48/accumulation24；五步扩大试跑 exit 0、120 micro-batches、峰值 44.179%、GPU utilization 100%。正式恢复 tag 为 `formal2_m2`；W&B online run `pzfktt9t` 已推进到 `trainer/global_step=47`，在旧 micro4 已达 97.468% 的同一长跑比较区域，micro2 正式峰值仅 59.981%，GPU utilization 100%。
- unet_c1 W&B `ohhl54t1` 持续 running，顺序扫描探针已确认至少 `trainer/global_step=338`，正式峰值约 79.091%，未见错误。

## Completed

- 严格过滤得到 20,483 PDB/637,140 occurrence；最终资产可用清单 18,293 PDB/579,688 occurrence，`fallback=false`。
- split 为 train/validation/calibration/held-out = 13,719/200/100/4,274，无 PDB 泄漏。
- BOX Job 321111 完成，正式发布 train 13,715 PDB、validation 200 PDB；Job 321141 的 `verification.json` 为 `ok`。
- Find_1 首次正式 run `c641mktn` 暴露 TorchMetrics CPU AP 通过默认 NCCL 汇总的真实 bug；第二次 run `mb99gwzt` 证明专用 Gloo group 生效，随后暴露 Lightning 对已聚合 CPU scalar 重复 `sync_dist` 的第二边界。
- 最小源码修复只涉及 `src/wrappers/voxel_point_stage1_metrics.py` 的共享 Gloo process group，以及 `src/wrappers/voxel_point_stage1.py` 对已全局聚合 validation payload 关闭 Lightning 二次同步。远端正式 Conda 环境定向测试 `15 passed`；第三次 run `nj97r8mc` 已提供真实双卡端到端复验。
- 两个 Find_1 失败目录原样归档：`formal/Find_1_CPC1_failed_nccl_cpu_metric_20260721T0549/` 与 `formal/Find_1_CPC1_failed_lightning_cpu_log_sync_20260721T0612/`。
- Find_0 micro=4 正式 run `s04troim` 在长跑中升到 97.468%，已用精确 `kill_lock` 停止；孤儿进程组经核对后精确清除，allocation/after_lock 保留。失败证据归档为 `formal/Find_0_CPC1_unsafe_m4_20260721T0548/`。
- 最新 `batch_freeze.env`：共同 GBS48；Find_1 micro6/accum4，Find_0 micro2/accum24，unet micro6/accum8。

## Decisions

- CPU non-binned AP 继续保持精确全局聚合：NCCL DDP 下使用专用 Gloo group，不做 rank 均值替代、不把未分箱状态长期放回 GPU。
- metric/CPC payload 在进入 Lightning logger 前已经完成全局归约，故 logger 不再做重复 `sync_dist`；正式双卡 run 已验证该边界。
- Find_0 采用 micro2/accum24；micro4 试跑结果不能覆盖正式长跑的 97.468% 证据。micro2 已在相同 global step 47 比较点把正式峰值压到 59.981%，但仍需持续监控长期 allocator/样本峰值，发现超 85% 时按精确锁恢复。
- 所有恢复包装和资源调度脚手架只在本地/服务器 `tmp/`；科学源码只保留上述真实 bug 的最小修复和回归测试。

## Next Actions

1. 继续监控 Find_0 W&B run `pzfktt9t` 的长期 allocator/样本峰值、错误和训练推进，确认正式长跑峰值不超过 85%。
2. 持续监控 Find_1 `nj97r8mc`、unet `ohhl54t1` 的 W&B 状态、GPU 利用率、traceback 和训练推进。
3. 首个完整 epoch 后，用本地日志与 `tmp/stage1_wandb_validation_audit.py` 核对每个 epoch 恰好 25 次 validation；记录 LR reduction 和 BEST。
4. Find CPC1 完成时核对 CPC1 BEST，随后确认同一 chain 的 CPC2 `init_from` 精确等于同名 CPC1 BEST 且为 model-only；保留两个阶段 checkpoint。
5. 五段全部正确终态前不释放 allocation；完成后做 resolved config、停止原因、W&B、BEST 可加载性、谱系、日志与脚手架分类总审计，并收口 ExecPlan/最终 handoff。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/adaligand_stage1_find0_formal_recovery_watch.sh`
- `tmp/adaligand_stage1_find1_watch.sh`
- `tmp/stage1_wandb_live_probe.py`
- `tmp/stage1_wandb_validation_audit.py`
- `tmp/stage1_formal_train_stage.sh`
- `tmp/stage1_formal_find_chain.sh`

## URLs

- Find_1 successful recovery: `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/nj97r8mc`
- Find_1 first/second failed evidence: `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/c641mktn`, `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/mb99gwzt`
- Find_0 unsafe micro4 evidence: `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/s04troim`
- Find_0 micro2 recovery: `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/pzfktt9t`
- unet_c1: `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/ohhl54t1`
