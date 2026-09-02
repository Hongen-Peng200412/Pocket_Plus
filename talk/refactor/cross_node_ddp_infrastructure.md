# 跨节点多卡 DDP 基础设施规格

本文定义 Pocket Plus 正式 `训练与运行` 链路如何在多个 Slurm 节点上启动 PyTorch Distributed Data Parallel（DDP）。DDP 在每张 GPU 上运行一个训练进程，并在每次反向传播时同步梯度。本轮只补齐进程启动、运行留证和失败传播，不改变模型、Dataset、损失、优化器或全局批量定义。

## 当前目标

- 单节点任务继续由 Lightning 在当前节点启动每卡一个训练进程，现有命令和锁语义不变。
- 跨节点任务必须显式传入 `--multi-node-ddp`，并同时满足 GPU 资源、`--nodes` 大于 1、每节点 GPU 数大于 0。
- allocation 控制器只在 Slurm batch 主节点创建 release、launch 和四类锁。每次实际执行时，控制器通过一个 Slurm job step 在每个节点启动一个 `torchrun` agent；每个 agent 再按每节点 GPU 数创建训练进程。
- 第一个分配节点作为 rendezvous 主节点。rendezvous 是全部训练进程交换 rank 和建立 NCCL 进程组时使用的会合地址。
- `kill_lock` 先终止本次 Slurm job step，由 `srun --kill-on-bad-exit` 清理各节点训练进程；allocation 本身继续遵守既有 `try_lock` 与 `after_lock` 语义。

## 正式命令契约

跨两节点、每节点两张 GPU 的任务使用：

```bash
bash 训练与运行/submit_task.sh \
  --sh Find_1.sh \
  --resource a800 \
  --nodes 2 \
  --gpus 2 \
  --cpus 64 \
  --multi-node-ddp
```

`--cpus` 仍表示每个节点上一个 `torchrun` agent 可使用的 CPU 数。训练配置中的 `train.devices` 表示每节点 GPU 数，`train.nnodes` 表示节点数；Lightning 的全局 `world_size` 等于两者乘积。

## 不在本轮范围内

- 不修改 Stage1 的采样、监督、模型结构或数值超参数。
- 不增加弹性重启；一个 rank 失败时，本次 attempt 整体失败并回到既有锁流程。
- 不让 CPU 多节点任务自动复制执行。未提供 `--multi-node-ddp` 时，`--nodes` 必须保持 1，以免多个节点被申请后闲置。
- 不替换或继续维护 `与服务器交互/sbatch/` 中的历史多节点模板。

## 验收标准

- 提交器拒绝不完整或有歧义的多节点参数组合，并把合法拓扑完整传给 `task.sbatch`。
- 模拟 Slurm 双节点时，allocation 只创建一次 release 和 launch，但任务脚本分别收到节点 rank 0 与 1。
- 单节点启动的命令参数与本轮修改前一致。
- 本地测试精确核对每节点 `torchrun` agent 的 static rendezvous 参数；四个真实 Python rank 另行建立 Gloo 进程组并完成一次 all-reduce。两项测试共同覆盖启动参数和全局 rank 通信，但不替代服务器 NCCL smoke。
- `tests/test_project_task_runner.py`、Stage1 配置测试、Shell 语法检查和 `git diff --check` 全部通过。
