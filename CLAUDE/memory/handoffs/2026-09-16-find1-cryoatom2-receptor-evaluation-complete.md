# Handoff: Find_1 CryoAtom2 受体推理与评估完成

Date: 2026-09-16

## Current State

Job `368455` 的双 H100、64 CPU allocation 已完成 `Find_1` 使用 CryoAtom2 最终预测受体的 Stage1 全套校准与测试。Job 已停回 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_368455`；`/home/penghongen/Feedback/Pocket_Plus/allocations/368455/after_lock_368455` 保留，资源未释放。

正式结果根为 `/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts/Find_1`。最终门控报告 `/storage/penghongen/tmp/find1_cryoatom2_final_gate_20260916/result.json` 为 `passed`。

## Completed

- Job 第 14 次执行完成 CryoAtom2 calibration 100 项与 `test_0` 179 项受体适配；第 15 次执行完成正式适配产物门控。
- 第 16 次执行完成两条 calibration 选参、calibration Gaussian score-only 回填和 179-PDB `test_0` 正式评估。
- 第 17 次执行从父 `test_0` 逐 PDB 事实保序派生 149-PDB `test_1`，没有重复模型前向。
- 第 18 次执行完成最终只读门控：两套 `test_0` 逐 PDB evaluation NPZ 共 358 份，calibration scored-centered 100 份；两套 `test_1` JSONL 与清单及父记录精确一致。
- `test_1` basic 的 semantic micro/macro F1 为 `0.5406947426509066/0.49038813561501443`；Gaussian 为 `0.5266402130264328/0.4800930116407475`。

## Decisions

- 固定 checkpoint 为 `TOP_epoch_04_score_0.6654.ckpt`，对应 W&B 已完成步编号 `38622`。
- F1 basic 冻结为语义阈值 `0.7274169921875`、分数阈值 `0.75816810131073`、`min_voxels=20`。
- F2 Gaussian 冻结为语义阈值 `0.527618408203125`、分数阈值 `0.9171097278594971`、`min_voxels=28`、`tau_angstrom=0.75`、`lambda_positive=0.2816`、`lambda_negative=0.0`。
- `test_1` 仅聚合 149 个父 `test_0` 事实；不把它解释为独立测试运行。

## Next Actions

- 未经用户明确授权，不删除 `after_lock_368455`、不使用 `scancel`，也不把 allocation 改作其他任务。
- 本任务没有待补推理、待评估或待门控范围。

## Files To Reopen

- `文档/exec_plan/2026-09-15_Find_1_CryoAtom2受体推理与评估.md`
- `文档/exec_plan/2026-09-15_Find_1不同受体条件推理与评估总日志.md`
- `文档/mapping/计划执行映射.md`
- AdaLigand `收口の结果/Stage1/Find_1(pdb_centric_v2)/使用cryoatom2预测的受体/说明.md`
