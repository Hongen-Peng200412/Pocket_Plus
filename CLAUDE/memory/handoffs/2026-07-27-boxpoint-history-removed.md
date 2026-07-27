# Handoff: 删除 BoxPointDataset 历史管线

Date: 2026-07-27

## Current State

Pocket_Plus 当前位于本地实现分支 `codex/remove-boxpoint-history`。共同基点是
`Learn/CUMULATIVE` 的 `b7114cded9eeff3fcd9c7f1d57a417ea6815af36`。
本轮全部改动仍在工作区，没有暂存、提交或推送；用户将先审阅再自行提交。

正在运行的 Job 321107、321540 和 321743 使用服务器上的既有代码快照。本轮只修改
本地 Pocket_Plus，没有连接服务器、改变锁文件或干预这三个训练任务。

## Completed

- 删除 `BoxPointDataset`、专属批次拼装、专属样本构造和
  `BalancedForegroundSampler`。
- 删除 6 份只实例化 `BoxPointDataset` 的数据集配置。
- 删除 40 份直接或递归依赖这些数据集配置的旧实验配置，包括旧
  `trunk`、CPC2 和 CPC3 配置族。
- 删除 3 份只验证旧 BOX 目录契约的测试。
- 从 `src/train.py` 和训练配置中删除旧 BOX 目录前景平衡采样入口。
- 保留当前 `Stage1Dataset` 与推理共同使用的 `box_geometry.py`，并把模块说明改为
  当前调用关系。
- 更新 `src/model/notes_of_network.md` 与模拟密度说明，使阅读入口和字段名称指向
  `Stage1Dataset`、`stage1_collate.py`、`density_input` 与当前辅助监督字段。
- 在 `tests/test_adaligand_stage1_configs.py` 增加约束：六份 Find CPC 配置和
  `unet_c1` 必须实例化 `Stage1Dataset`，组合后的训练配置不得含旧平衡采样字段。

## Verification

- 修改前基线：
  `tests/test_adaligand_stage1_configs.py`、
  `tests/datasets/test_stage1_dataset.py` 和
  `tests/inference/test_stage1_checkpoint.py` 共 33 项通过。
- 修改后：
  Stage1 配置、Dataset、checkpoint 与推理装配共 46 项通过。
- `tests/test_fit_scale_receptor_mask.py` 在
  `MKL_NUM_THREADS=1`、`OMP_NUM_THREADS=1` 下 11 项通过。
- `python -m compileall` 已检查 `src/train.py`、`stage1_dataset.py`、
  `stage1_collate.py` 和 `box_geometry.py`。
- `git diff --check` 通过。
- 完整测试在收集阶段未运行：本机没有项目约定的 `Pocket_Plus_windows` Conda
  环境，基础 Anaconda 缺少 `rootutils`、`lightning`、`torch_cluster`、
  `addict` 和 `wandb`。本轮没有安装依赖。

## Decisions

- Git 历史已经保存被删除代码，不建立可执行的历史归档目录。
- 只删除 BoxPointDataset 及其递归配置族。`configs/experiment/old_1`、
  `old_2`、`组会` 等其他失效历史配置留给下一轮独立审计。
- `box_geometry.py` 和 `density_channel_builder.py` 属于当前
  `Stage1Dataset`/推理共享代码，不能随旧管线删除。
- 当前工作区由用户自行审阅和提交；本轮不重建学习分支。

## Open Questions

- 小数据集试验使用哪个 `box_sample_fraction` 数值，以及启动 Find_0、Find_1、
  `unet_c1` 中的哪些实验，需要在修改训练配置前由用户确定。
- 其他已经失效、但不依赖 BoxPointDataset 的历史配置何时进行下一轮清理。

## Next Actions

1. 用户审阅本轮工作区差异并自行提交。
2. 确认正式启动脚本已经默认使用
   `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation`。
3. 在正式准备目录已成为唯一训练来源后，删除数据集配置、
   `Stage1Dataset`、`stage1_requests.py` 和测试中的 `excluded_pdb_ids` 过滤机制。
4. 按用户指定的 `box_sample_fraction < 1` 和训练参数修改实验配置，通过正式启动
   链路提交新任务。
5. 新任务成功进入正式运行后，本阶段才算完成。

## Files To Reopen

- `src/datasets/stage1_dataset.py`
- `src/datasets/stage1_requests.py`
- `configs/dataset/stage1_find.yaml`
- `configs/dataset/stage1_unet_c1.yaml`
- `configs/experiment/CPC1/Find_1.yaml`
- `configs/experiment/unet_c1.yaml`
- `训练与运行/sh/Find_0.sh`
- `训练与运行/sh/Find_1.sh`
- `训练与运行/sh/unet_c1.sh`
