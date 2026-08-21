# Handoff: U-Net 密度通道消融已排队并进入守护

Date: 2026-08-21

## Current State

AdaLigand Stage1 的两项 density-only U-Net 消融已完成端到端修复、测试、安全同步和 Slurm 提交。Job `350302` 是 `unet_base`，使用 `nvlink` partition、`nvlinkg8` QOS、1 张 A800 和 16 CPU，已在 `gnode09` 稳定训练；Job `350305` 是 `unet_diff`，使用 `nvlink` partition、`h200g2` QOS、1 张 A800 和 16 CPU，仍因 `Resources` 等待。两项任务都使用 `pre_hold=0`、`after_hold=1`。

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
- Job `350302` 的 release、launch 与最终配置已核验；W&B 本地摘要达到 `trainer/global_step=203`，总损失约 `0.2310`，A800 利用率为 94%，无明确错误，已判定稳定训练。

## Decisions

- 必须修改代码，因为旧实现把模拟密度读取和辅助目标生成绑定到固定实验名称；只删除异常检查仍会使 `diff` 通道缺少模拟密度，并使已启用的主链和距离损失缺少目标。
- 修复保持最小范围：不改变模型、损失权重、采样逻辑、Find 原子表或训练制度。
- 提交命令严格沿用用户原来的资源与 QOS，只追加 `--after_hold`；没有添加 `--pre_hold`。
- 排队期间或训练完全稳定后不使用 heartbeat，采用由 300 秒片段组成的 60 或 120 分钟静默等待；训练启动前后、疑似报错或停止等重要事件窗口可以连续检查。

## Open Questions

- 两项任务何时获得 A800 由 Slurm 优先级决定。
- 获得资源后仍需以 release、launch 和训练目录中的最终 `config.yaml` 核验真实运行配置，并确认正常 step 或明确的长时间内存加载状态。

## Next Actions

1. 以 60 或 120 分钟静默周期监视已稳定训练的 Job `350302` 和仍在排队的 Job `350305`。
2. Job `350305` 启动后核验 release、launch、最终 Hydra 配置、标准输出和标准错误。
3. 确认 Job `350305` 进入正常 step 或可证明的内存加载阶段后，把两项任务共同纳入稳定训练守护。
4. 只在关键事件发生时更新运行记录和本 handoff。
5. 若任务结束并进入 `try_lock`，保留资源与锁，先诊断并向用户报告，不擅自删除锁或重启。

## Files To Reopen

- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `src/datasets/stage1_dataset.py`
- `configs/dataset/stage1_unet_base.yaml`
- `configs/dataset/stage1_unet_diff.yaml`
- `configs/experiment/unet_base.yaml`
- `configs/experiment/unet_diff.yaml`
- `训练与运行/sh/unet_base.sh`
- `训练与运行/sh/unet_diff.sh`
