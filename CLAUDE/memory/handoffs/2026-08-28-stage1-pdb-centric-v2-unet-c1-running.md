# Handoff: Stage1 采样方式三 unet_c1 已稳定训练

Date: 2026-08-28

## Current State

Stage1 采样方式三的 V2 验证产物、Dataset 请求入口、`unet_c1` 配置和正式 shell 已完成实现与验收。正式训练 Job `358384` 于 2026-08-28 09:31:26 +08:00 在 `hnode01` 启动，使用 1 张 H100、32 CPU 和 30 个 `DataLoader` workers。任务已完成十七次完整 validation、处于 epoch 2，并生成对应 TOP 与 last checkpoint；没有发现 CUDA OOM、NCCL、DataLoader 或非有限值错误。`after_lock_358384` 仍存在，不得自动释放。

详细的数据产物与运行证据记录在 `ops/stage1_data_preparation/EXECUTION.md`，活动科学说明位于 `talk/global/global_8.26.md`。

## Completed

- V2 验证文件为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric_v2.npz`，保存全部 200 个 validation PDB、3,305 个 bias BOX 和 3,200 个 context BOX；V1 文件未被覆盖。
- 训练采样参数固定为 `pdb_foreground_box_num=50`、`pdb_foreground_fraction_target=0.7575757575757576`、`pdb_occurrence_foreground_box_cap=1`。生产请求入口重新展开得到 216,739 个 bias 和 219,472 个 context，共 436,211 个训练 BOX；V2 验证共有 6,505 个 BOX。
- Job `358384` 的 launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260828T093205_a1/launch.json`，release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_58b27dd765ed/Pocket_Plus`。
- 正式运行目录为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`，W&B run 为 `pencounkdual-111/AdaLigand_Stage1/9dsi7i9w`。
- 最终配置已核对：`devices=1`、`nnodes=1`、`num_workers=30`、V2 验证路径、`max_epochs=110`、`val_per_epoch=8`、`warmup_ratio=0.005`、每卡 batch 8、全局 batch 48、学习率 `1e-4`，蛋白与核酸主链损失权重均为 `0.05`。
- release 与运行目录冻结快照中的 `src/datasets/stage1_requests.py` SHA-256 均为 `74ffcbbf916d0758b64da084312932db93b1e93b7b2f69648955b62df4bfe514`。
- 稳定性验收时，W&B 摘要从 `trainer/global_step=200` 推进到 `278`；最新训练总损失约为 `0.3253`，受体、配体体素、蛋白主链、核酸主链和配体距离损失均为有限数值。H100 抽样利用率为 89%，显存使用量为 79,848/81,559 MiB。
- 2026-08-28 14:12 +08:00，首次完整 validation 结束：验证总损失为 `0.3082846`，配体体素 PRAUC 为 `0.2585518`，受体 PRAUC 为 `0.2138680`；全部验证损失均为有限数值。`checkpoints/TOP_epoch_00_score_0.2586.ckpt` 与 `checkpoints/last.ckpt` 已生成，任务随后继续推进到 `trainer/global_step=1304`。
- 2026-08-28 18:50 +08:00，第二次完整 validation 结束：验证总损失为 `0.2699645`，配体体素 PRAUC 为 `0.3804999`，受体 PRAUC 为 `0.3469226`；`checkpoints/TOP_epoch_00_score_0.3805.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=2288`。
- 2026-08-28 23:30 +08:00，第三次完整 validation 结束：验证总损失为 `0.2663394`，配体体素 PRAUC 为 `0.4109082`，受体 PRAUC 为 `0.3935394`；`checkpoints/TOP_epoch_00_score_0.4109.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=3686`。
- 第三次 validation 后，`BEST.ckpt` 已刷新为第二次的 `TOP_epoch_00_score_0.3805.ckpt`，确认训练中的稳定别名比最新 TOP 晚一个 validation 回调。最终退出验收必须确认 `BEST.ckpt` 已追上最终最佳模型；当前无需停止训练。
- 2026-08-29 04:05 +08:00，第四次完整 validation 结束：验证总损失为 `0.2514138`，配体体素 PRAUC 为 `0.4482627`，受体 PRAUC 为 `0.4602929`；`checkpoints/TOP_epoch_00_score_0.4483.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=4673`。`BEST.ckpt` 已刷新为第三次的 `TOP_epoch_00_score_0.4109.ckpt`。
- 2026-08-29 08:39 +08:00，第五次完整 validation 结束：验证总损失为 `0.2458982`，配体体素 PRAUC 为 `0.4758326`，受体 PRAUC 为 `0.4872300`；`checkpoints/TOP_epoch_00_score_0.4758.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=5957`。`BEST.ckpt` 已刷新为第四次的 `TOP_epoch_00_score_0.4483.ckpt`。
- 2026-08-29 13:15 +08:00，第六次完整 validation 结束：验证总损失为 `0.2383705`，配体体素 PRAUC 为 `0.4785158`，受体 PRAUC 为 `0.5158617`；`checkpoints/TOP_epoch_00_score_0.4785.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=7307`。`BEST.ckpt` 已刷新为第五次的 `TOP_epoch_00_score_0.4758.ckpt`，未发现显式错误。
- 2026-08-29 17:50 +08:00，第七次完整 validation 结束：验证总损失降至 `0.2311407`，配体体素 PRAUC 为 `0.4713585`，受体 PRAUC 升至 `0.5202941`；`checkpoints/TOP_epoch_00_score_0.4714.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务随后继续推进到 `trainer/global_step=8012`。`BEST.ckpt` 已刷新为第六次的 `TOP_epoch_00_score_0.4785.ckpt`，错误扫描为空。
- 2026-08-29 22:23 +08:00，第八次完整 validation 结束：验证总损失为 `0.2300247`，配体体素 PRAUC 创新高至 `0.5097033`，受体 PRAUC 创新高至 `0.5372400`；`checkpoints/TOP_epoch_00_score_0.5097.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务进入 epoch 1 并继续推进到 `trainer/global_step=9149`。`BEST.ckpt` 暂时保持第六次的 `TOP_epoch_00_score_0.4785.ckpt`，符合已确认的回调时序；错误扫描为空。
- 2026-08-30 03:08 +08:00，第九次完整 validation 结束：验证总损失降至 `0.2178248`，配体体素 PRAUC 创新高至 `0.5327364`，受体 PRAUC 创新高至 `0.5641335`；`checkpoints/TOP_epoch_01_score_0.5327.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=10289`。`BEST.ckpt` 已刷新为第八次的 `TOP_epoch_00_score_0.5097.ckpt`，错误扫描为空。
- 2026-08-30 07:57 +08:00，第十次完整 validation 结束：验证总损失为 `0.2191286`，配体体素 PRAUC 为 `0.5293481`，受体 PRAUC 创新高至 `0.5670031`；`checkpoints/TOP_epoch_01_score_0.5293.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=11366`。`BEST.ckpt` 已刷新为第九次的 `TOP_epoch_01_score_0.5327.ckpt`，错误扫描为空。
- 2026-08-30 12:45 +08:00，第十一次完整 validation 结束：验证总损失为 `0.2243590`，配体体素 PRAUC 小幅创新高至 `0.5349470`，受体 PRAUC 创新高至 `0.5844466`；`checkpoints/TOP_epoch_01_score_0.5349.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=13079`。`BEST.ckpt` 暂时保持第九次的 `TOP_epoch_01_score_0.5327.ckpt`，符合已确认的回调时序；错误扫描为空。
- 2026-08-30 17:31 +08:00，第十二次完整 validation 结束：验证总损失为 `0.2182586`，配体体素 PRAUC 创新高至 `0.5367544`，受体 PRAUC 为 `0.5783800`；`checkpoints/TOP_epoch_01_score_0.5368.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=13742`。`BEST.ckpt` 已刷新为第十一次的 `TOP_epoch_01_score_0.5349.ckpt`，错误扫描为空。
- 2026-08-30 22:14 +08:00，第十三次完整 validation 结束：验证总损失为 `0.2181289`，配体体素 PRAUC 为 `0.5267326`，受体 PRAUC 为 `0.5686539`；`checkpoints/TOP_epoch_01_score_0.5267.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=14846`。`BEST.ckpt` 已刷新为第十二次的 `TOP_epoch_01_score_0.5368.ckpt`，错误扫描为空。
- 2026-08-31 02:57 +08:00，第十四次完整 validation 结束：验证总损失为 `0.2088470`，配体体素 PRAUC 为 `0.5241200`，受体 PRAUC 为 `0.5801458`；`checkpoints/TOP_epoch_01_score_0.5241.ckpt` 与 `checkpoints/last.ckpt` 已更新，validation 对应 `trainer/global_step=15902`。
- 2026-08-31 07:42 +08:00，第十五次完整 validation 结束：验证总损失为 `0.2180344`，配体体素 PRAUC 创新高至 `0.5419699`，受体 PRAUC 创新高至 `0.5948478`；`checkpoints/TOP_epoch_01_score_0.5420.ckpt` 与 `checkpoints/last.ckpt` 已更新，validation 对应 `trainer/global_step=17038`。
- 2026-08-31 12:29 +08:00，第十六次完整 validation 结束：验证总损失降至 `0.2037673`，配体体素 PRAUC 创新高至 `0.5507095`，受体 PRAUC 创新高至 `0.6137356`；`checkpoints/TOP_epoch_01_score_0.5507.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务进入 epoch 2 并继续推进到 `trainer/global_step=18176`。`BEST.ckpt` 已刷新为第十五次的 `TOP_epoch_01_score_0.5420.ckpt`，错误扫描为空。
- 第十四至第十六次 validation 期间，本地 90 分钟睡眠会话因宿主挂起延长；服务器 Job 始终为 `RUNNING`。恢复后通过 Slurm、W&B `scan_history` 和 checkpoint 时间戳补齐了三次验证的权威证据，训练本身未受影响。
- 2026-08-31 17:12 +08:00，第十七次完整 validation 结束：验证总损失为 `0.2076540`，配体体素 PRAUC 为 `0.5448655`，受体 PRAUC 创新高至 `0.6180575`；`checkpoints/TOP_epoch_02_score_0.5449.ckpt` 与 `checkpoints/last.ckpt` 已更新，任务继续推进到 `trainer/global_step=19574`。`BEST.ckpt` 已刷新为第十六次的 `TOP_epoch_01_score_0.5507.ckpt`，错误扫描为空。

## Decisions

- 方式二 V1 与方式三 V2 长期共存。当前 `unet_c1` 只消费 V2；Find、`unet_base` 和 `unet_diff` 继续消费 V1。
- 当前没有理由修改 batch 或 workers。只有出现 CUDA OOM 时才先把每卡 batch 从 8 降为 6并保持全局 batch 48；只有 worker 内存或进程故障时才把 workers 从 30 降为 24。
- 稳定训练采用由多个 300 秒命令组成的 60 或 90 分钟静默守护周期，不使用 heartbeat，不记录无变化轮询。
- `after_hold=1` 是正式资源契约。训练退出后保留 allocation 和 `after_lock_358384`，除非用户另行明确授权，否则不得删除锁或释放资源。

## Open Questions

- 训练最终停止原因、最终最佳 checkpoint 和完整训练时长尚待确认。
- `BEST.ckpt` 在训练中确认晚于最新 TOP checkpoint 一个 validation 回调；需在最终退出时核对别名内容与最终最高分 TOP。

## Next Actions

1. 每 60 或 90 分钟检查 Job 状态、W&B 最新 step、错误信号和 checkpoint；每次等待必须由连续的 300 秒睡眠命令组成。
2. 后续 validation 继续核对验证损失、配体体素 PRAUC、受体 PRAUC、全部损失的有限性，以及 TOP、last 与 BEST checkpoint 的更新时间。
3. 若出现可恢复故障，在同一训练目标内按既定 batch/worker 次序修复，并把科学契约变化、原因和恢复证据写入执行记录与本 handoff。
4. 训练退出后核对 Slurm、W&B、最终 checkpoint 和 `after_lock_358384`，保持 allocation 未释放，再完成文档收口与 goal。

## Files To Reopen

- `ops/stage1_data_preparation/EXECUTION.md`
- `talk/global/global_8.26.md`
- `configs/dataset/stage1_unet_c1.yaml`
- `训练与运行/sh/unet_c1.sh`
- `CLAUDE/memory/projects/pocket-plus.json`
