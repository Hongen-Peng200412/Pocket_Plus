# Handoff: 调度保留开关与 Find_0 推理入口说明

Date: 2026-08-03

## Current State

Pocket_Plus 的通用 Slurm 提交系统已经把“首次执行前等待”和“任务结束后保留资源”拆成两个独立开关。默认行为是获得资源后立即执行一次任务，随后自动退出并释放 allocation。所有改动留在工作区，未暂存、未提交，也没有提交真实 Slurm 作业。

`训练与运行/sh/infer/` 的九个 shell 入口已经补齐可直接复制的服务器提交命令和参数注释。Find_0 calibration 的概率图、F1-centered 与 CLG-centered 统一使用两个全局分片；validation 与 train 保持 16 个全局分片。

## Completed

- `训练与运行/submit_task.sh` 新增 `--pre_hold` 与 `--after_hold`，删除旧 `--hold`。
- `--pre_hold` 只创建 `pre_lock` 并延迟第一次执行，不改变任务结束后的默认释放行为。
- `--after_hold` 在每次任务结束后创建 `try_lock`，保留 allocation 供人工续跑；省略时任务结束后立即释放。
- `训练与运行/sbatch/task.sbatch` 和 `训练与运行/runtime/allocation_runner.sh` 已传递并执行两个独立开关。
- `训练与运行/README.md` 和训练脚本中的锁说明已同步为新语义。
- 九个 `训练与运行/sh/infer/*.sh` 均写入完整提交示例；示例不使用数组并发上限 `%N`，并默认展示 `--after_hold`。
- calibration 三个推理脚本统一为 `global_shard_count=2` 与 `--array '0-1'`，两个 GPU 分片各处理 50 个 PDB。
- validation 和 train 脚本继续使用 16 个分片，完整数组为 `--array '0-15'`。
- 所有推理脚本的单进程 Dataset 缓存上限从 500 GiB 改为 100 GiB；推理 manifest 分别记录 calibration 与 validation/train 的分片总数。

## Decisions

- 不保留 `--hold` 兼容别名；旧命令会以“未知参数”退出，调用方必须明确改用 `--pre_hold`。
- 是否在任务结束后保留卡由提交者通过 `--after_hold` 决定，默认释放资源。
- `--after_hold` 只改变 allocation 生命周期，不改变推理输出、分片算法或续跑完成标记。
- 本轮只完善本地入口与说明，不提交服务器任务，不修改已经冻结运行的 release。
- 已经启动的 allocation 使用启动时已经加载的控制器，不会被本地改动热替换。尚未启动且仍携带旧 `--hold` 参数的排队 Job 不属于兼容范围，正式使用前应核对并用新入口重新提交。

## Validation

- Pocket_Plus 的 21 个 `训练与运行/**/*.sh` 与 AdaLigand 对应 shell 文件全部通过 `bash -n`。
- `tests/test_project_task_runner.py` 对 Pocket_Plus 调度目录为 `9 passed`。
- 同一测试把 `PROJECT_TASK_RUNNER_DIRECTORY` 指向 AdaLigand 调度目录后再次得到 `9 passed`。
- 测试覆盖默认自动释放、失败退出码保留、`--pre_hold` 延迟首次执行、`--after_hold` 创建 `try_lock`、两个开关转发和旧 `--hold` 拒绝。

## Next Actions

1. 用户审阅每个推理 shell 顶部的正式提交命令。
2. 真正提交前按当时服务器资源决定是否保留 `--after_hold`；删掉该参数即可在任务结束后自动释放。
3. calibration 的 probability、F1 与 CLG 必须继续共同使用两分片契约，不能只改其中一个脚本。

## Files To Reopen

- `训练与运行/submit_task.sh`
- `训练与运行/sbatch/task.sbatch`
- `训练与运行/runtime/allocation_runner.sh`
- `训练与运行/README.md`
- `训练与运行/sh/infer/README.md`
- `训练与运行/sh/infer/Find_0_calibration_probability.sh`
- `训练与运行/sh/infer/prepare_inference_pdb_lists.sh`
- `tests/test_project_task_runner.py`
