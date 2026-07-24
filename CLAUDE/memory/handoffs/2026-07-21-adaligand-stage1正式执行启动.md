# Handoff: AdaLigand Stage1 正式执行启动

Date: 2026-07-21

## Current State

正式 preparation root 已冻结为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`。严格过滤得到 20,483 个唯一 PDB，高于 18,000 诊断信号，当前不回退到全量候选。Job 321104 正在 cnode01 用 48 CPU/48 进程审计这些 PDB 的 Stage1 直接资产，日志为 `/home/penghongen/My_Project/tmp/adaligand_s1_inventory_321104.{out,err}`；完成后同一 job 调用成熟 `freeze_stage1_splits` 并显式传 validation=200、calibration=100。

三个正式 GPU allocation 已全部 RUNNING 并停在 run-scoped `pre_lock`，尚未执行训练：Find_1 Job 321108 为 hnode03 双 H200/24 CPU；Find_0 Job 321106 为 hnode01 单 H100/24 CPU；unet_c1 Job 321107 为 hnode02 单 H100/24 CPU。各自 scope 位于 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/{JOBID}/`，`after_lock` 均保留。

活 ExecPlan 是 `talk/Excx_执行stage1端到端训练.md`；它是恢复和审计的首要本地记录。

## Completed

- 读取 AdaLigand/Pocket_Plus 的治理、计划、ExecPlan、BOX 契约、项目记忆、最新 handoff 和 Stage1 成熟代码入口。
- 只读确认 A–G 正式 run `adaligand_ag_20260711T154658`、662,078 occurrence candidates、22,251 eligible PDB，且先前不存在 keep-list/split/BOX/本次训练。
- 写入并编译检查 run-scoped inventory、split 和 allocation 脚本；通过非删除式同步发送到服务器。
- 处理首次无默认 QOS 的 `Invalid qos specification`，为 CPU job 显式使用 `cpu96`。
- 取消未运行且因 `QOSMaxCpuPerJobLimit` 挂起的自有双 H200 Job 321105，以 24 CPU 重提为 321108 并确认 RUNNING。

## Decisions

- 不改 Stage1 科学代码；validation=200 通过 `tmp/stage1_freeze_split_200.py` 显式调用现有函数参数。
- 三个 producer 先占有正式资源但保持 `pre_lock`，等正式 BOX 和真实读取 smoke 完成后才执行显存 preflight/训练。
- 20,483 严格 PDB 高于异常信号；资产审计未完成前不宣称最终清单冻结，也不提前判断是否因资产缺失触发回退。

## Open Questions

- 资产审计最终缺失/不兼容数，以及是否仍需全量资产回退。
- BOX pool 并行生成的安全并发数、耗时和 short-map/context 统计。
- 三 producer 的共同 global batch、micro-batch、梯度累积和实际峰值显存。

## Next Actions

1. 高频监控 321104 的资产审计错误和吞吐；完成后验证 inventory/split `_COMPLETE`、摘要、PDB/occurrence 计数与 split 无泄漏。
2. 以总活跃 CPU 不超过 192 的 run-scoped 并行包装生成 train/validation BOX pool，并做 manifest/selection/completion 审计。
3. 在 321108/321106/321107 中运行三 role Dataset/DataLoader smoke 和实际 GPU batch preflight，冻结共同 global batch。
4. 按 Find_1、Find_0、unet_c1 启动正式训练，持续审计 W&B、25 次 validation、LR reduction 停止、BEST 和 CPC1→CPC2 model-only 谱系。

## Files To Reopen

- `talk/Excx_执行stage1端到端训练.md`
- `tmp/stage1_prepare_inventory.py`
- `tmp/stage1_freeze_split_200.py`
- `tmp/adaligand_stage1_inventory_split.sbatch`
- `tmp/adaligand_stage1_alloc_h200_2.sbatch`
- `tmp/adaligand_stage1_alloc_h100_1.sbatch`
