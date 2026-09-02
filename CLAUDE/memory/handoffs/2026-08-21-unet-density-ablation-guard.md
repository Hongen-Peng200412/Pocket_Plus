# Handoff: U-Net 密度通道消融守护，unet_base 已结束训练

Date: 2026-08-21

## Current State

AdaLigand Stage1 的两项 density-only U-Net 消融已完成端到端修复、测试、安全同步、Slurm 提交和最终运行核验。Job `350302` 是 `unet_base`，使用 `nvlink` partition、`nvlinkg8` QOS、1 张 A800 和 16 CPU；训练已在 `gnode09` 按 `stop_after_lr_reductions=3` 正常结束。该 allocation 随后于 2026-09-01 07:33:50 由账号用户取消，`after_hold` 资源已释放，`try_lock_350302` 与 `after_lock_350302` 已由调度包装器清理。Job `350305` 是 `unet_diff`，使用 `nvlink` partition、`h200g2` QOS、1 张 A800 和 16 CPU，仍在 `gnode10` 正常训练。两项任务都使用 `pre_hold=0`、`after_hold=1`。

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

## Decisions

- 必须修改代码，因为旧实现把模拟密度读取和辅助目标生成绑定到固定实验名称；只删除异常检查仍会使 `diff` 通道缺少模拟密度，并使已启用的主链和距离损失缺少目标。
- 修复保持最小范围：不改变模型、损失权重、采样逻辑、Find 原子表或训练制度。
- 提交命令严格沿用用户原来的资源与 QOS，只追加 `--after_hold`；没有添加 `--pre_hold`。
- 排队期间或训练完全稳定后不使用 heartbeat，采用由 300 秒片段组成的 60 或 120 分钟静默等待；训练启动前后、疑似报错或停止等重要事件窗口可以连续检查。

## Open Questions

- Job `350302` 已完成训练并释放资源，没有待处理锁。
- Job `350305` 当前没有配置或运行阻塞，仍需继续守护其验证、失败、`try_lock` 与结束事件。

## Next Actions

1. Job `350305` 继续采用由连续 300 秒睡眠片段组成的 60 或 120 分钟静默守护周期，不创建 heartbeat。
2. 只在 Job `350305` 失败、进入 `try_lock`、结束或出现其他重要状态变化时更新运行记录和本 handoff。
3. 若 Job `350305` 结束并进入 `try_lock`，保留资源与锁，先诊断并向用户报告，不擅自删除锁或重启。

## Files To Reopen

- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `src/datasets/stage1_dataset.py`
- `configs/dataset/stage1_unet_base.yaml`
- `configs/dataset/stage1_unet_diff.yaml`
- `configs/experiment/unet_base.yaml`
- `configs/experiment/unet_diff.yaml`
- `训练与运行/sh/unet_base.sh`
- `训练与运行/sh/unet_diff.sh`
