# Stage1 pdb-centric-1 unet_c1 训练完成交接

Date: 2026-09-02

## Current State

Job `356953` 确实是 `unet_c1 mainchain` 使用采样方式二 `pdb-centric-1` 的正式训练，训练主体已经正常完成。Slurm 仍显示 `RUNNING`，只是因为提交时启用了 `after_hold=1`：训练包装器已创建 `try_lock_356953`，节点上没有 `src/train.py` 或 GPU 计算进程。当前可直接消费的正式模型为运行目录中的 `checkpoints/BEST.ckpt`；complete-map 推理和三种采样方式的统一比较尚未完成。

## Completed

- Slurm 权威资源为 `nvlink` partition、`h200g4` QOS、`gnode09`、1 张 A800 和 24 CPU，提交契约为 `pre_hold=0`、`after_hold=1`。Job 于 2026-08-26 11:09:47 +08:00 启动。
- 启动留证为 `/home/penghongen/Feedback/Pocket_Plus/launches/356953/unet_c1_job356953_20260826T110721_a1/launch.json`；release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_064fd1e46dcc/Pocket_Plus`，内容 SHA-256 为 `064fd1e46dcc58bb7d3702acc7ed36a8498726924de08f51798406e60f5300e4`。
- 正式运行目录为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal`；最终 `config.yaml` 的 SHA-256 为 `3156d81fca04b1ab897e519febf05864c5b9563aca614f1704d5f83e07c90347`；W&B run 为 `pencounkdual-111/AdaLigand_Stage1/6uuzdbgs`。
- 最终配置确认输入只有 `exp_clipnorm_nopost` 一个实验密度通道，训练采样参数为 `25/0.5/25`，验证读取 V1 `validation_selection_pdb_centric.npz`。V1 文件包含 150 个 PDB、3,750 个 bias、3,750 个 context 和 0 个 center BOX；每个 epoch 共有 685,850 个训练 BOX。
- 训练使用每卡 batch 8、全局 batch 48、梯度累积 6、24 个 `DataLoader` workers、学习率 `1e-4`、`max_epochs=70`、`val_per_epoch=12`、BF16 混合精度、`patience=3` 和 `stop_after_lr_reductions=2`。配体区域、配体体素、蛋白主链、核酸主链和配体反距离五项损失均已启用，权重为 `0.1/1.0/0.05/0.05/0.3`。
- 训练保存 20 个 TOP checkpoint。W&B 最终摘要记录 `trainer/global_step=23813`、验证总损失 `0.1979677`、配体体素 PRAUC `0.4891676`、受体 PRAUC `0.5830685`；`last.ckpt` 内部 `global_step=23814`，并记录第二次实际学习率衰减。
- 最高配体体素 PRAUC 为 `0.4918`。`BEST.ckpt` 与 `TOP_epoch_01_score_0.4918.ckpt` 的 SHA-256 均为 `0bb6bd5231c638c13dee5144bfb361b6bd20f5268bd10337a040c8486efa9b5a`，证明最终最佳别名已经闭合；`last.ckpt` 的 SHA-256 为 `ce1408095c49095ea31bd78e78dbdd6ba441e6ae48ea09ecc485f1de5a34d1f0`。
- 标准输出明确记录两次实际学习率衰减触发停止、正式训练完成和第 1 次执行成功。`try_lock_356953` 于 2026-09-01 06:48:18 +08:00 创建；严格错误扫描为空。
- 2026-09-02 的节点抽样确认 A800 仅占用 2/81,920 MiB 显存、GPU 利用率为 0%，没有计算进程。`after_lock_356953` 与 `try_lock_356953` 均未被修改。
- 采样方式二来自提交 `f806f12007e2cef521f71588103edf17318e8b4a`。采样方式三实现提交 `b4529f574e8d1728183a4e72f64c5b80b20a0215` 与独立 Job `358384` 在方式二训练期间启动；Job `358384` 不是 Job `356953` 的续训或替代 checkpoint。

## Decisions

- “Job `356953` 已完成 `unet_c1 + pdb-centric-1` 训练”可以作为已核实事实使用。
- Slurm 的 `RUNNING` 只表示 `after_hold` allocation 仍保留，不得据此把训练状态写成仍在推进。
- 不用方式二 V1 与方式三 V2 各自的 BOX validation PRAUC 直接决定采样策略。最终选择继续使用同一 complete-map 数据与指标契约。
- 本次只读核验和文档收口不触碰 `after_lock_356953`、`try_lock_356953`、任务本身或任何 checkpoint。

## Open Questions

- Job `356953` 的 `BEST.ckpt` 尚未发现 complete-map 推理与评估产物；方式二在统一完整密度图评估上的表现仍未知。
- 采样方式三 Job `358384` 尚未在本交接中做最终训练收口。三种 `unet_c1` 采样方式的最终比较仍等待各自可比产物。
- Job `356953` 的空闲 A800 allocation 何时释放由用户决定；没有明确授权时保持锁和 Job 原状。

## Next Actions

1. 以 Job `356953` 的 `BEST.ckpt` 运行与采样方式一相同的 complete-map 推理与评估流程，并保存 checkpoint、源码快照、推理配置和评估产物之间的可追溯关系。
2. 等采样方式三训练及 complete-map 评估也完成后，在同一测试集合和指标契约下比较三种采样方式，再决定 Find 训练采用哪一种。
3. 只有用户明确要求释放资源时，才按项目锁协议处理 Job `356953` 的 `after_hold`；不得用未经授权的 `scancel` 或删除锁代替决策。

## Files To Reopen

- `talk/global/global_8.26.md`
- `ops/stage1_data_preparation/EXECUTION.md`
- `CLAUDE/memory/projects/pocket-plus.json`
- `CLAUDE/memory/handoffs/2026-08-28-stage1-pdb-centric-v2-unet-c1-running.md`
- 服务器正式运行目录中的 `config.yaml`、`checkpoints/BEST.ckpt` 和 `checkpoints/last.ckpt`
