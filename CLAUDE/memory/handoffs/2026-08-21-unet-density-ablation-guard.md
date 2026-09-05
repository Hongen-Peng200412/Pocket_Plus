# Handoff: U-Net 密度通道消融训练与资源均已收口

Date: 2026-08-21
Updated: 2026-09-05

## Current State

AdaLigand Stage1 的两项 density-only U-Net 消融均已完成端到端修复、测试、安全同步、Slurm 提交和正式训练。Job `350302` 是 `unet_base`，Job `350305` 是 occurrence-centric `unet_diff`；两者均在第 3 次实际学习率衰减后按 `stop_after_lr_reductions=3` 正常结束。Job `350305` 于 2026-09-05 05:01 +08:00 完成训练，用户在最终产物验收后释放其 held allocation；10:26 复核时，该 Job 已不在 Slurm 队列，活动锁与动态命令均已清理。

详细证据和后续关键事件记录在 `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`。

## Completed

- `unet_base` 固定 56 个密度通道；`unet_diff` 固定 `exp_clipnorm_nopost` 与 `diff_clipnorm_nopost` 两个通道。
- 两项任务共同使用每卡批量 8、全局批量 48、梯度累积 6、16 个 `DataLoader` worker、学习率 `1.0e-4` 和五项体素损失。
- 删除 `Stage1Dataset` 对 producer 名称白名单和 `unet_c1` 单通道的硬检查。
- 模拟密度读取改为由启用通道的 `sim_`、`diff_`、`posdiff_` 前缀决定；两个新实验接入现有主链和配体距离辅助监督目标。
- 本地 Dataset 全套 14 项测试通过；新增参数化测试在本地和服务器均为 4 项通过。
- 服务器正式 A–G 数据样本确认输入形状分别为 56 通道和 2 通道，且全部辅助目标存在。
- 安全同步后的本地与服务器关键文件 SHA-256 一致。
- 相关任务文件已从 Git 暂存区撤出；本地改动保持未暂存或未跟踪，没有创建提交。
- Job `350302` 的 release、launch 与最终配置已核验；W&B 本地摘要达到 `trainer/global_step=1154`，训练总损失约 `0.1692`，首次验证总损失约 `0.2399`、配体体素 PRAUC 约 `0.5019`，五项损失均为有限数值，已判定持续稳定训练。
- Job `350305` 的 launch 为 `unet_diff_job350305_20260822T171529_a1`，release 为 `Pocket_Plus_fdb8a30fa196`；最终配置确认 2 个差分密度通道、全局批量 48、每卡批量 8、梯度累积 6、16 个 worker、学习率 `1.0e-4`、结构输出头和五项损失全部开启。W&B 本地摘要连续从 `trainer/global_step=5` 推进到 23、38，五项损失均为有限值，A800 显存约 79.4 GiB，GPU 利用率抽样为 57% 和 73%，没有错误、OOM、NaN 或 `try_lock`；已判定稳定训练。
- Job `350305` 已在 `trainer/global_step=1175` 完成首次验证：验证总损失约 `0.2976`、配体体素 PRAUC 约 `0.2980`、受体 PRAUC 约 `0.2707`，五项验证损失均为有限值并继续训练。同期 Job `350302` 达到 `trainer/global_step=6998`；两项任务均没有错误、OOM、NaN 或 `try_lock`。
- Job `350302` 于 2026-08-31 12:10 达到 3 次实际学习率衰减后按配置正常结束。最终 `trainer/global_step=39508`，验证总损失约 `0.1334`、受体 PRAUC 约 `0.7658`、配体体素 PRAUC 约 `0.6794`；`last.ckpt` 和 TOP checkpoint 已写入，现有最高分 TOP checkpoint 为 `TOP_epoch_00_score_0.6809.ckpt`。调度包装器记录第 1 次执行成功并创建 `/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_350302`；`after_lock_350302` 继续保留资源，两个锁均未被触碰。同期 Job `350305` 继续训练并达到 `trainer/global_step=30827`，没有错误或 `try_lock`。
- Job `350302` 的 Slurm allocation 于 2026-09-01 07:33:50 被账号用户取消；该操作发生在训练正常结束和 checkpoint 写入之后，只释放了 `after_hold` 保留的 A800。调度包装器随后清理 `try_lock_350302`、`after_lock_350302` 与动态命令。守护过程没有执行 `scancel`，也没有删除锁。同期 Job `350305` 继续正常训练，GPU 利用率抽样为 99%，没有错误或 `try_lock`。
- 2026-09-03 14:47 +08:00，Job `350305` 的 W&B run `nf93buae` 已推进到 `global_step=41018`、学习率约 `4e-6`。最近一次 validation 总损失为 `0.1653458`、配体体素 PRAUC 为 `0.6288677`、受体 PRAUC 为 `0.7104262`；运行目录中的 `TOP_epoch_00_score_0.6289.ckpt` 与 `last.ckpt` 已于 11:59 写出。Job 仍为 `RUNNING`，只有 `after_lock_350305`，没有 `try_lock` 或 `kill_lock`。
- Job `350305` 的最终 validation 完整处理 827 batches，验证总损失为 `0.1607799530`、配体体素 PRAUC 为 `0.6330414414`、受体 PRAUC 为 `0.7154738903`。最终 checkpoint 为 epoch 1、`global_step=46282`、`validation_index=38`，优化器学习率为 `8.000000000000022e-07`，实际衰减计数为 3；标准输出明确记录正式训练完成、总时长 `1,165,496.32` 秒和第一次执行成功。
- 最终 `last.ckpt` 与 `TOP_epoch_01_score_0.6330.ckpt` 的 SHA-256 均为 `48ad3b374c32e34edad21a7e9127cf875a5bd42c21327821f84e2b4d45c7dbfd`。全程最高原始配体体素 PRAUC 为 `0.6343953609`；`BEST.ckpt` 与 `TOP_epoch_00_score_0.6344.ckpt` 的 SHA-256 均为 `8777e7a58ffdb6f17914329a3a2bbdc1bb478b345b6da8b570f50108c9c51409`。
- 用户在训练产物验收后释放 Job `350305`。Slurm 主 Job 于 10:14:17 结束，调度包装器随后清理 `try_lock_350305`、`after_lock_350305` 与动态命令；训练产物不受影响。

## Job 350305 可追溯身份与产物

- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_fdb8a30fa196/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/350305/unet_diff_job350305_20260822T171529_a1`。
- 正式训练与全部实验产物根：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`。
- checkpoint 目录：上述产物根的 `checkpoints/`；冻结配置与源码位于 `config.yaml`、`train.yaml` 和 `src_snapshot/`。
- W&B 本地产物：上述产物根的 `wandb/run-20260822_172939-nf93buae/`；在线身份为 `pencounkdual-111/AdaLigand_Stage1/nf93buae`。
- immutable `launch.json` 与 `run_cmd.sh` SHA-256 分别为 `f5852fe2c7ec78da3f03250c2d7c47879b5708abac5f5c89fca284a4713611f0`、`29c6433271a5f46c8403d5d2a9b9d7de6c3f1ce460b12a9e4c4714b738cedfc5`；allocation 当前动态命令与 immutable `run_cmd.sh` 哈希相同。
- `config.yaml`、`train.yaml`、`src_snapshot/manifest.json` SHA-256 分别为 `b0c162c246980f3d26c7fd76cabdc8222edef2cd5bf303b23cce13c90afd24c3`、`0193469232fbf453c7cd827c7227419b32b2a559687a9bc2121e0fb2fb5b3bb5`、`f47a4c6ce0ab31311d0a0c26584ace79b4b5a28e552bc0b66ba4d8adb0487123`。
- Job `350305` 释放后，`/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_350305`、`/home/penghongen/Feedback/Pocket_Plus/allocations/350305/after_lock_350305` 与动态命令均已不存在；该 Job 不再占用 gnode10 资源。

## Decisions

- 必须修改代码，因为旧实现把模拟密度读取和辅助目标生成绑定到固定实验名称；只删除异常检查仍会使 `diff` 通道缺少模拟密度，并使已启用的主链和距离损失缺少目标。
- 修复保持最小范围：不改变模型、损失权重、采样逻辑、Find 原子表或训练制度。
- 提交命令严格沿用用户原来的资源与 QOS，只追加 `--after_hold`；没有添加 `--pre_hold`。
- 排队期间或训练完全稳定后不使用 heartbeat，采用由 300 秒片段组成的 60 或 120 分钟静默等待；训练启动前后、疑似报错或停止等重要事件窗口可以连续检查。
- 普通 validation、epoch 完成和常规 checkpoint 不再写入 handoff；只有任务提交或替换、训练结束、故障修复与重启、资源交接、科学契约变化等阶段事件更新 handoff。
- Job `350305` 已由用户在训练验收后释放；不需要重新提交或恢复 allocation。

## Open Questions

- Jobs `350302`、`350305` 均已完成训练并释放资源，没有待处理锁或训练故障。

## Next Actions

1. 不再把 Jobs `350302`、`350305` 纳入活动训练轮询，也不重新提交这两项消融训练。
2. 保留现有训练产物、release、launch、W&B 和 checkpoint，供后续推理或模型比较使用。
3. 当前全局守护只继续覆盖 Jobs `368455`、`366071` 和 `366277`。

## Files To Reopen

- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `src/datasets/stage1_dataset.py`
- `configs/dataset/stage1_unet_base.yaml`
- `configs/dataset/stage1_unet_diff.yaml`
- `configs/experiment/unet_base.yaml`
- `configs/experiment/unet_diff.yaml`
- `训练与运行/sh/unet_base.sh`
- `训练与运行/sh/unet_diff.sh`
