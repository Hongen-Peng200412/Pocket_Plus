# Handoff：Stage1 采样方式三 unet_c1 训练完成并保留 H100

Date: 2026-09-03

## Current State

采样方式三 `unet_c1` 的正式训练 Job `358384` 已于 2026-09-03 15:46 +08:00 正常完成。最终模型端点为 `TOP_epoch_03_score_0.5957.ckpt`；该文件与 `BEST.ckpt`、`last.ckpt` 字节一致。

Job `358384` 的 Slurm allocation 没有释放。同一 allocation 已由另一条既定工作启动采样方式三的完整图推理，`after_lock_358384` 继续存在。用户最新明确要求保留这张 H100 做其他工作；本 handoff 取代 `2026-08-28-stage1-pdb-centric-v2-unet-c1-running.md` 中关于后续释放资源的旧安排。未经新的明确授权，禁止删除该锁、取消 Job 或停止 allocation 内当前工作。

## Completed

- Job `358384` 于 2026-08-28 09:31:26 +08:00 在 `hnode01` 启动，资源为 H100×1、CPU×32，训练使用 30 个 `DataLoader` workers。
- 训练在第二次实际学习率衰减后达到 `stop_after_lr_reductions=2`，按配置正常停止；包装器记录总时长 540,795.94 秒，并明确记录“正式训练完成”和“第 1 次执行成功”。
- W&B run `pencounkdual-111/AdaLigand_Stage1/9dsi7i9w` 于 15:46:17 完成最终同步。摘要为 epoch 3、`trainer/global_step=36349`、学习率 `2e-5`、验证总损失 `0.1856628805398941`、配体体素 PRAUC `0.5956817269325256`、受体 PRAUC `0.6590440273284912`。
- 训练 release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_58b27dd765ed/Pocket_Plus`，launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260828T093205_a1`。
- 训练 `launch.json` 与 immutable `run_cmd.sh` SHA-256 分别为 `e891f5f13519f2c3e2ecc46d5af6938533ac385ce0ac6b528bf7695accb6e116` 和 `c365ce608826031abb21963f5fb51b1b90cea67cc7d83b40fa8e633e40e0240a`。实际训练命令为 `exec bash "${TASK_PROJECT_ROOT}"/训练与运行/sh/unet_c1.sh`。
- 正式运行目录为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`。其中 `config.yaml`、`train.yaml` 与 `src_snapshot/manifest.json` SHA-256 分别为 `0e56a76a8b3bafc0312972567c72835763c9ea732254101322410fefb6d3229d`、`3eb0e9810e34d522f5d421b82f321c09967b9750beab7eebf37b819564487d42` 和 `d3105c67e8da0343b95258d156a93a388323522ab4ffbc002bf7601820884a9a`。
- checkpoint 目录为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/checkpoints`。`BEST.ckpt`、`TOP_epoch_03_score_0.5957.ckpt` 与 `last.ckpt` 均为 499,678,098 bytes，SHA-256 均为 `8e8bd068ebbeb8b49db3ddd40c6c485bad322e75f3308b39298b8cea7bec3a`。
- 第一次执行成功后，另一条工作消费了 `try_lock_358384`。第二次执行使用 release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus` 和 launch `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260903T161305_a2`；对应 `launch.json` 与 `run_cmd.sh` SHA-256 分别为 `19020bf20907c226040f607e55be2400a6df384c2e2ee18553df5cc91d943b91` 和 `67948adeff133468605e6bc1dda25921da4e37d9b0fed06cc36ea1252afdcac3`。
- 第二次执行命令为 `exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/unet_c1_sampling_comparison.sh" pdb_centric_2`。calibration probability 输出根为 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts`；当前推理进展由 `2026-09-03-unet-c1-sampling-comparison-running.md` 管理。

## Decisions

- 采样方式三后续推理显式使用 `TOP_epoch_03_score_0.5957.ckpt`、同一训练运行目录的 `config.yaml` 和 `training_snapshot` 模型代码，不在运行时动态选择 checkpoint。
- Job `358384` 的训练守护已经完成；综合守护只负责防止本任务误释放该 allocation，不接管当前推理的调度和产物验收。
- 用户最新的资源保留指令优先于旧 handoff 中的释放安排。没有新的明确授权时，不执行 `scancel 358384`，不删除 `/home/penghongen/Feedback/Pocket_Plus/allocations/358384/after_lock_358384`，也不停止当前进程。

## Open Questions

- 训练端没有未决问题；停止原因、最终指标、训练时长和 checkpoint 一致性均已核验。
- H100 的下一项用途和最终释放时机等待用户在当前推理完成后另行决定。

## Next Actions

1. 在综合守护的每次醒来检查中确认本任务没有释放 Job `358384`；无变化时不写日志。
2. 采样方式三推理完成、失败或需要资源转换时，由其专用执行记录和 handoff 保存完整事件，本训练 handoff 不重复记录流水账。
3. 只有用户再次明确授权释放或转换该资源时，才根据当时的进程、锁、release、launch 和 Slurm 状态拟定精确动作。

## Files To Reopen

- `ops/stage1_data_preparation/EXECUTION.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-training-global-watch.md`
- `CLAUDE/memory/handoffs/2026-09-03-unet-c1-sampling-comparison-running.md`
- `CLAUDE/memory/projects/pocket-plus.json`
