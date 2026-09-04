# Handoff: pdb-centric-v1 校准完成并进入 held-out 测试

Date: 2026-09-03

## Current State

Job `367332` 在 A800、24 CPU 上继续运行 pdb-centric-v1，已从 calibration 转入 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts/unet_c1/held_out_test_0` 的 probability 阶段；2026-09-03 07:55 为 15/179。Job `367411` 在 A100、16 CPU 上运行 occurrence-centric held-out probability；07:58 为 121/179。两个 Job 的 `after_lock` 均保留，`try_lock` 均不存在，第二次执行没有新增 traceback。

## Completed

- pdb-centric-v1 的 100 个 calibration PDB 已全部写出 probability `_COMPLETE`。
- 100 个 calibration PDB 已全部写出 `F1_blobs/_COMPLETE`。
- `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts/unet_c1/tuning/F1_semantic.json` 已冻结：`threshold_value=0.46185302734375`，`macro_f_beta=0.392776324846336`。
- 同目录 `F1_basic.json` 已冻结：`score_threshold=0.5705057382583618`、`prefiltered_min_voxel=8`、`min_voxels=26`、`objective_beta=1.0`。
- calibration 的职责是选参数，不额外执行固定测试评估；正式 shell 已使用上述参数进入 held-out 流水线。

## Decisions

- 不重算 occurrence-centric 的既有 calibration 参数。
- 不在三模型比较中启用 Gaussian 或 centered 分数；最终只比较 F1 blobs+basic。
- pdb-centric-v2 继续等待训练 Job `358384` 正常结束，任何当前 TOP checkpoint 都不提前冻结。

## Open Questions

- Job `358384` 仍在运行；截至本记录，文件名最高分仍是 `TOP_epoch_03_score_0.5932.ckpt`，但它不是最终冻结输入。

## Next Actions

1. 继续以 60 或 120 分钟静默周期守护 Job `367332`、`367411`，核验 held-out probability、F1 blobs、最终 evaluation 和错误增量。
2. occurrence 完成后核验 179 个 probability、179 个 F1 blobs、固定参数评估 JSON 与 micro/macro/PRAUC 汇总。
3. pdb-centric-v1 完成后执行同样闭合核验；资源仍由 `after_lock` 保留，不自行释放。
4. Job `358384` 正常结束后再固定 v2 checkpoint 并启动第三套流水线。

## Files To Reopen

- `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`
- `训练与运行/sh/infer/unet_c1_sampling_comparison.sh`
- `CLAUDE/memory/handoffs/2026-09-03-unet-c1-sampling-comparison-running.md`
