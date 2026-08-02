# 项目通用任务提交系统与 AdaLigand 移植

## Current State

Pocket_Plus 的项目通用任务提交系统现位于常用工作树 `C:\Users\15919\Desktop\Pocket_Plus` 的 `Learn/CUMULATIVE` 工作区。所有改动按用户要求保持未暂存、未提交，不应自动提交或推送。最初用于隔离实现的 `019fc256-task-runner` 工作树已经在逐文件等价核验和常用工作树定向测试通过后删除；没有独有提交的 `codex/project-generic-task-runner` 临时分支也已删除。

同一套核心入口已经移植到 `C:\Users\15919\Desktop\AdaLigand\训练与运行`。AdaLigand 中新增文件同样保持未暂存；该仓库原有的已暂存和未暂存修改属于其他工作，不得被本任务暂存、提交、移动或丢弃。

## Completed

- `submit_task.sh` 不再区分项目内任务与 `external_task`。只有文件名、项目相对路径和项目内绝对路径都会解析为同一个任务根目录内相对路径。
- 未提供 `--task-root` 时，任务根目录由 `训练与运行` 的上一层推导；显式 `--task-root` 可以选择另一个采用相同目录结构的项目。
- 真正位于任务根目录之外的脚本会被拒绝，防止完整模式创建了 release 却从共享项目目录执行另一个脚本。
- release、launch 和 allocation 运行组件已从 `与服务器交互/other/training_runtime` 移到 `训练与运行/runtime`。
- 通用运行标识改为 `TASK_RUN_STAMP`，launch 证据字段改为 `task_run_stamp`；Pocket_Plus 的三个现有训练入口已同步读取新名称。
- `训练与运行/README.md` 保留原有 13 个编号章节，并补充任务根目录、路径等价、可复制目录和旧提交系统边界。
- 个人 skill `C:\Users\15919\.codex\skills\project-server-interaction` 已把项目根目录下的 `训练与运行` 设为正式任务入口，并把 `与服务器交互/sbatch` 降为旧任务兼容入口。
- AdaLigand 新增冷读 README、通用提交入口、通用 sbatch、三个 runtime 文件和任务脚本目录说明。

## Validation

- Pocket_Plus 定向测试：`8 passed`。
- 同一组任务提交系统测试把待测目录切换到 AdaLigand 后：`5 passed`；覆盖默认任务根目录、绝对与相对任务路径等价、项目外脚本拒绝、完整 release/launch 留证、simple 模式四锁执行。
- Pocket_Plus 全仓测试在临时补充未跟踪 `.project-root` 后得到 `387 passed, 3 failed`。三个失败都来自仓库原有 `tests/test_cpc_v3_configs.py` 所要求的 `configs/experiment/CPC1/trunk_*.yaml` 不存在，与本次改动无关；临时 `.project-root` 已删除。
- Pocket_Plus 与 AdaLigand 的五个 Bash 核心文件均通过 `bash -n`。
- 两个项目的 `submit_task.sh`、`task.sbatch` 和三个 runtime 文件按 UTF-8 文本逐字相同；项目差异只保留在 README 与任务脚本中。
- `git diff --check` 无空白错误。
- `project-server-interaction` 通过 skill 快速校验：`Skill is valid!`。
- 未向 Slurm 提交真实作业，未修改服务器文件或正在运行的任务。
- 2026-08-03 将完整差异迁入常用 `Learn/CUMULATIVE` 工作区后，16 个保留文件与隔离实现统一换行后文本等价，3 个退出文件在两边均不存在；常用工作树定向测试再次得到 `8 passed`。

## Decisions

- 项目外脚本不再拥有绕过冻结副本的正式执行分支。需要运行另一个项目时，应为该项目复制完整的 `训练与运行`，或显式把该项目设为 `--task-root`。
- `训练与运行` 管理正式 Slurm 提交、release、launch 和四锁；`与服务器交互` 继续管理同步、SSH 和旧任务兼容材料。
- 本轮不创建 commit。后续提交者应先检查两个仓库已有工作区内容，再决定如何纳入各自历史。

## Open Questions

- AdaLigand 目前只移植通用系统，没有替用户新增具体训练、推理或数据生产任务脚本。正式使用前，应为具体任务在 `训练与运行/sh` 增加冷读入口并单独核对输入、输出和资源参数。

## Next Actions

1. 用户审阅 Pocket_Plus 与 AdaLigand 两份 `训练与运行/README.md`。
2. 为下一项 AdaLigand 正式任务新增项目专属 `训练与运行/sh/*.sh`。
3. 首次真实提交前，在服务器核对 `rsync`、`sha256sum`、`stat`、`setsid` 和 `stdbuf` 可用，并检查分区与 QOS。
4. 若准备提交这些本地修改，必须显式区分本任务文件和两个仓库中既有的其他修改。

## Files To Reopen

- `训练与运行/submit_task.sh`
- `训练与运行/sbatch/task.sbatch`
- `训练与运行/runtime/allocation_runner.sh`
- `训练与运行/runtime/create_release.sh`
- `训练与运行/runtime/create_launch.sh`
- `训练与运行/README.md`
- `tests/test_project_task_runner.py`
- `C:\Users\15919\Desktop\AdaLigand\训练与运行\README.md`
- `C:\Users\15919\.codex\skills\project-server-interaction\SKILL.md`
