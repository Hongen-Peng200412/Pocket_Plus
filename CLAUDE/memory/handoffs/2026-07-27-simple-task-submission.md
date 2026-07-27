# Handoff: 通用任务提交器新增 simple 模式

Date: 2026-07-27

## Current State

`训练与运行/submit_task.sh` 现在同时支持完整模式和 `--simple` 模式。未填写
`--simple` 时，原有 release、launch、四锁和 `Feedback/<项目名>` 目录行为保持
不变。填写 `--simple` 时，任务仍由 Slurm 和四锁控制，但不创建 release 或
launch；活动锁、动态命令和 Slurm 标准输出、标准错误统一位于
`${HOME}/SIMPLE_RUN/`。

本轮没有提交 Slurm Job，没有运行训练、模型或数据生产，也没有修改训练脚本、
Python 模型、Dataset 或数据产物。

## Completed

- `--sh 文件名.sh` 继续从 `训练与运行/sh/` 查找。
- `--sh` 的其他写法不再转换为项目相对路径，也不检查是否位于项目中；该字符串
  原样传给任务执行器。
- `--simple` 不调用 release 或 launch 工具，不创建
  `${HOME}/Feedback/<项目名>/releases`、`launches` 或 `allocations`。
- simple 模式的 `pre_lock`、`try_lock`、`after_lock`、`kill_lock` 和
  `run_cmd` 位于 `${HOME}/SIMPLE_RUN/`；stdout 和 stderr 由 Slurm 写入同一目录。
- simple 模式结束时，最后一次任务执行的退出码成为 Slurm Job 的退出码。
- `训练与运行/README.md` 已增加完整命令、数值示例、目录结构和两种 `--sh`
  解释；`与服务器交互/other/readme.md` 已增加简要入口说明。

## Verification

- 三个修改后的 shell 文件通过 Git Bash `bash -n`。
- 用假的 `SBATCH_BIN` 检查了完整模式的 `Find_1.sh` 文件名入口。
- 用假的 `SBATCH_BIN` 检查了用户指定的 simple CPU 命令：任务绝对路径保持
  不变，资源为 `cpu`、`Cpu96`、1 CPU，stdout/stderr 指向
  `${HOME}/SIMPLE_RUN`，没有真实提交。
- 使用临时 `${HOME}` 直接运行 Slurm 包装层，完成一次
  `pre_lock → 任务退出 → try_lock → 删除 after_lock` 生命周期；确认没有生成
  release、launch 或 Feedback 目录，并确认任务退出码 `2` 被包装层保留。
- 临时测试脚本和临时目录已经清理。

## Working Tree Boundary

任务开始前已存在且没有被本轮修改的用户改动：

- `ops/materialize_filtered_stage1_preparation.py`
- `ops/materialize_filtered_stage1_preparation.sh`
- `src/datasets/readme.md`

本轮属于小型工作区修改，没有暂存、提交或重组 Git 历史。

## Files To Reopen

- `训练与运行/submit_task.sh`
- `训练与运行/sbatch/task.sbatch`
- `与服务器交互/other/training_runtime/allocation_runner.sh`
- `训练与运行/README.md`
- `与服务器交互/other/readme.md`
