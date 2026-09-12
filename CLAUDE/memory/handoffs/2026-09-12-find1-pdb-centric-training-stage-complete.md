# Handoff: Find_1 两条 PDB-centric 训练阶段性完成

Date: 2026-09-12

## Current State

用户已暂时停止 Find_1 PDB-centric-1 与 PDB-centric-2 两条训练，因为现有 checkpoint 已足够支持近期使用。这是用户主动选择的阶段性完成，不是训练失败，也不是永久结束；未来仍可能继续训练。

本 handoff 取代早期 handoff 中“两条训练仍在运行”的当前状态，但不改写那些文件保存的历史事实。Slurm allocation、控制锁及资源后续用途容易变化，故不作为本次长期交接的核心内容。未来恢复时以训练产物、checkpoint 摘要和完整 validation 边界为准。

## PDB-centric-1 Stage Boundary

训练产物根：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_1/Find_1-pdb_centric_1_resume_job366071____run_job371591_20260907T215604_a1_pdb_centric_1_resume`

W&B run：`pencounkdual-111/AdaLigand_Stage1/gg8y6s0i`。

最后一个可恢复 checkpoint：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_1/Find_1-pdb_centric_1_resume_job366071____run_job371591_20260907T215604_a1_pdb_centric_1_resume/checkpoints/last.ckpt`

- 写入时间：2026-09-10 02:09:49 +08:00；
- 大小：1,466,247,723 bytes；
- SHA-256：`4cd34307634747d3afc29f01419f3702e07971eb81698cac77f1f28a6e437637`；
- epoch 1、`global_step=17860`；
- 当前 epoch 已完成 28,575 个训练批次，累计完成 142,884 个训练批次；
- 最近一次验证完整完成 1,250/1,250 个批次，`is_last_batch=true`，调度身份为 `(17860, 1)`；
- 配体体素 PRAUC 为 `0.5427930355072021`；
- 优化器学习率为 `5.0000000000001236e-5`，实际学习率衰减次数为 0；
- 学习率平台期控制状态为 `validation_index=11`、`best=0.5602476596832275`、`num_bad_epochs=3`；
- 候选阈值为 `p_best=0.0732421875`。

近期使用的最佳模型：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_1/Find_1-pdb_centric_1_resume_job366071____run_job371591_20260907T215604_a1_pdb_centric_1_resume/checkpoints/TOP_epoch_00_score_0.5602.ckpt`

它的大小为 1,466,228,590 bytes，SHA-256 为 `bdaf83e3183d368d9569b22fefc84ce1c3bb781be88048f2628010899d707b5a`；同目录 `BEST.ckpt` 与它逐字节一致。

W&B 末次本地摘要记录 `global_step=17981`，比 `last.ckpt` 多 121 个优化器更新步骤。这些参数更新没有进入可恢复 checkpoint，未来不得用 W&B step 17981 代替续训边界。

## PDB-centric-2 Stage Boundary

训练产物根：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume`

W&B run：`pencounkdual-111/AdaLigand_Stage1/0bydo3jg`。

最后一个可恢复 checkpoint：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/last.ckpt`

- 写入时间：2026-09-12 12:15:17 +08:00；
- 大小：1,466,259,093 bytes；
- SHA-256：`195d380c3a1a348d8acb893412793f7698dd23d4b066c446f7af94a5e69e346f`；
- epoch 4、`global_step=42030`；
- 每个 rank 在当前 epoch 已完成 22,715 个训练批次，累计完成 168,119 个训练批次；
- 最近一次验证在每个 rank 上完整完成 543/543 个批次，`is_last_batch=true`，调度身份为 `(42030, 4)`；
- 配体体素 PRAUC 为 `0.6672196984291077`；
- 优化器学习率为 `1.999999999999997e-6`，实际学习率衰减次数为 2；
- 学习率平台期控制状态为 `validation_index=33`、`best=0.6644246578216553`、`num_bad_epochs=2`；
- 候选阈值为 `p_best=0.3681640625`。

近期使用的最佳模型：

`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/TOP_epoch_04_score_0.6673.ckpt`

它的大小为 1,466,258,582 bytes，SHA-256 为 `bc761a3a0a97aa67924eda87609ed3c6ee87acb0611e2023417ce17564234351`；同目录 `BEST.ckpt` 与它逐字节一致。

本次只读核验时，W&B 已记录到 `global_step=43163`，比 `last.ckpt` 多 1,133 个优化器更新步骤。这些参数更新没有进入上述 checkpoint，不属于未来可恢复边界。

## Decisions

- 两条训练均记作“阶段性完成，可续训”，不得记为失败或最终完成全部训练计划。
- 近期推理或比较使用各自明确命名的最佳模型。需要延续优化器、调度器和采样轨迹时，使用各自的 `last.ckpt`；最佳权重文件不能替代完整续训端点。
- 早期恢复过程已经披露并由用户接受的一次性非逐位偏差继续有效。本次暂停后超出 checkpoint 的 W&B 步骤也不可恢复，因此未来续训不得宣称与未中断轨迹逐位一致。
- 不把 allocation、锁或资源接管状态写成长期判断；这些状态由用户另行处置，并且可能很快变化。

## Next Actions

1. 当前不自动重启两条训练，也不继续周期监视。
2. 用户未来决定续训时，先明确选择 PDB-centric-1 或 PDB-centric-2、使用的恢复 checkpoint 与资源规格。
3. 启动前重新计算源 checkpoint 的 SHA-256，并核对模型、优化器、调度器、候选阈值、当前 epoch 训练 batch 位置和完整 validation 身份。
4. PDB-centric-1 从当前 epoch 的 28,575 个训练批次边界继续；PDB-centric-2 从每个 rank 在当前 epoch 的 22,715 个训练批次边界继续。恢复实现必须避免重复执行已经完整结束的验证。
5. 新续训使用新的 release、launch、W&B run 和输出目录，保留本 handoff 记录的 checkpoint 与最佳模型身份。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `CLAUDE/memory/handoffs/2026-09-07-find1-pdb-centric1-a800-resume-running.md`
- `CLAUDE/memory/handoffs/2026-09-09-find1-pdb-centric2-periodic-checkpoint-recovered.md`
- `ops/find1_historical_resume/rebase_checkpoint.py`
- `训练与运行/sh/Find_1_pdb_centric_2_resume.sh`
