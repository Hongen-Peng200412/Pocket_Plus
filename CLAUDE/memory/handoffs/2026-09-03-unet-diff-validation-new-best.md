# Handoff：Job 350305 occurrence-centric unet_diff 刷新验证最佳值

Date: 2026-09-03

## Current State

Job `350305` 是 occurrence-centric `unet_diff` 密度通道消融训练，仍在 `gnode10` 使用 A800×1、CPU×16 正常运行。2026-09-03 20:48 +08:00 完成一轮 827-batch validation，配体体素 PRAUC 刷新到 `0.6319023967`；21:52 时 W&B 已继续推进到 `global_step=41915`。allocation 只有 `/home/penghongen/Feedback/Pocket_Plus/allocations/350305/after_lock_350305`，没有 `pre_lock`、`try_lock` 或 `kill_lock`。本轮只读核验没有操作训练或资源。

## Training Identity

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_fdb8a30fa196/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/350305/unet_diff_job350305_20260822T171529_a1`。
- 正式运行与全部实验产物根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`。
- checkpoint 目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal/checkpoints`。
- W&B：`pencounkdual-111/AdaLigand_Stage1/nf93buae`。
- immutable `launch.json` SHA-256：`f5852fe2c7ec78da3f03250c2d7c47879b5708abac5f5c89fca284a4713611f0`。
- immutable `run_cmd.sh` 与 allocation 当前 `run_cmd_350305.sh` SHA-256：`29c6433271a5f46c8403d5d2a9b9d7de6c3f1ce460b12a9e4c4714b738cedfc5`。
- 运行目录 `config.yaml`、`train.yaml`、`src_snapshot/manifest.json` SHA-256：`b0c162c246980f3d26c7fd76cabdc8222edef2cd5bf303b23cce13c90afd24c3`、`0193469232fbf453c7cd827c7227419b32b2a559687a9bc2121e0fb2fb5b3bb5`、`f47a4c6ce0ab31311d0a0c26584ace79b4b5a28e552bc0b66ba4d8adb0487123`。
- immutable 启动命令：`exec bash "${TASK_PROJECT_ROOT}"/训练与运行/sh/unet_diff.sh`。

## Validation Evidence

- `val_loss/global/total=0.16146185994148254`。
- `val_score/global/voxel_ligand_PRAUC=0.6319023966789246`。
- `val_score/global/receptor_PRAUC=0.7162516713142395`。
- 蛋白主链与核酸主链 macro PRAUC 分别为 `0.4437437057`、`0.4492560625`。
- 相比上一轮 `0.6288676857948303`，配体体素 PRAUC 提高 `0.0030347108840943`，严格超过 plateau 的绝对阈值 `0.003`。
- `last.ckpt` 中 `WarmupPlateauController.best=0.6319023966789246`、`num_bad_epochs=0`、`validation_index=34`，学习率为 `4.000000000000011e-06`。
- `LearningRateReductionStopper.lr_reduction_count=2`；本轮有效改进没有错误推进第三次衰减或停训。

## Checkpoint Evidence

- `TOP_epoch_00_score_0.6319.ckpt` 与 `last.ckpt`：499,688,614 bytes，SHA-256 均为 `269d9c2be071d73191b8a2751cb53d9128bcb789cf4e980aa86fa47f88dead89`。
- `BEST.ckpt` 与 `TOP_epoch_00_score_0.6289.ckpt`：499,688,614 bytes，SHA-256 均为 `ce837c1e728cffe1a99b04848736d02b64e82ecf461fada09c0fc03bc5570f81`。
- `ModelCheckpoint.best_model_path` 已指向 `TOP_epoch_00_score_0.6319.ckpt`，`best_model_score=0.6319023966789246`。磁盘 `BEST.ckpt` 尚停留在上一轮是已知的一次 validation 发布滞后，不表示最佳状态丢失。

## Next Actions

1. 继续把 Job `350305` 纳入固定醒来检查，关注训练推进、下一次 validation、第三次实际学习率衰减、正常停训和 `try_lock`。
2. 无变化轮询不写日志；稳定阶段使用多个独立 300 秒睡眠命令组成 60 或 90 分钟静默等待，不创建 heartbeat。
3. Job `350305` 没有自动释放授权。训练结束后保留 `after_lock` 和任何新生成的 `try_lock`，先核验最终产物并报告。

## Files To Reopen

- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-training-global-watch.md`
