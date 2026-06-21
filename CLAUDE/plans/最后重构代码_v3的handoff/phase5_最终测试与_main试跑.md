# Phase 5 handoff：旧 `_main` 试跑记录，已作废

更新时间：2026-06-19 12:58:00 +08:00

## 作废说明

本文件保留为历史记录，但不再代表 `最后重构代码_v3.md` 的最终收口。这里记录的 `_main` 使用了旧入口 `+experiment=CPC/CPC_main`，用户随后明确指出这是计划漂移：最终 `_main` 必须使用 `+experiment=CPC1/trunk_main`，且 CPC1/CPC2/CPC3 应分别落盘 4/4/9 个实体 YAML。

最终有效结果见同目录 `phase6_CPC_v3配置纠偏与最终收口.md`。后续 agent 不要再把本文件里的 `CPC/CPC_main` 当作最终验收入口。

本文件记录 `最后重构代码_v3.md` 的最终服务器验收。阅读顺序建议：先读 `最后重构代码_v3.md`，再读本目录 `00_总览与进度.md`，最后读本文件了解最终收口状态。

## 服务器纪律与实际资源

- 全部实时测试只使用用户指定资源：`/home/penghongen/try_lock_297515` 对应的 Slurm job。
- Slurm job：`297515`。
- 节点：`hnode01`。
- 分区：`h100`。
- 未触碰 `kill_lock_297515`。
- 未删除 `after_lock_297515`。
- 同步代码使用项目级 safe sync：`与服务器交互/sync_code.ps1`，没有使用 clean sync。

## 关键修复：`ModelCheckpoint` state_key 冲突

第一次真实启动 `_main` 时，`train.py` 在构造 `pl.Trainer(...)` 阶段失败：

```text
RuntimeError: Found more than one stateful callback of type `ModelCheckpoint`.
HINT: The `callback.state_key` must be unique among all callbacks in the Trainer.
```

根因：此前同时注册了主 `ModelCheckpoint`、`BEST` 专用 `ModelCheckpoint`，并且默认 `save_every_n_epochs=3` 还会注册周期性 `ModelCheckpoint`。单测只覆盖了调度器逻辑，没有覆盖完整 Trainer callback 列表，所以问题只在真实训练入口暴露。

已修复：

- `src/train.py`
  - 保留唯一的真正 `ModelCheckpoint`，负责 `TOP_*` 与 `last.ckpt`。
  - `BEST.ckpt` 改由 `BestCheckpointAlias` 普通 callback 从主 checkpoint 的 `best_model_path` 复制得到。
  - `PERIODIC_*` 改由 `PeriodicCheckpointSaver` 普通 callback 调用 `trainer.save_checkpoint(...)` 保存。
  - 这样 `pl.Trainer` callback 列表中不再存在多个 stateful `ModelCheckpoint`。
- `tests/test_warmup_plateau_scheduler.py`
  - 新增回归测试 `test_checkpoint_callbacks_have_unique_lightning_state_keys`，直接调用 Lightning 的 `_validate_callbacks_list(...)` 验证主 checkpoint + BEST alias + PERIODIC saver 不再冲突。

## 最终全量测试

在 `try_lock_297515` 对应 job 上执行的命令：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
python -m pytest tests -q
```

最终结果：

```text
277 passed, 8 warnings in 14.59s
```

日志位置：

```text
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.out
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.err
```

说明：

- 修复前曾有一次全量测试结果 `276 passed, 8 warnings`。
- 修复 `ModelCheckpoint` 真实训练入口问题后增加了 1 个回归测试，最终全量测试变为 `277 passed, 8 warnings`。

## `_main` 真实试跑

使用的 `_main` 配置入口：

```text
configs/experiment/CPC/CPC_main.yaml
```

通过 `run_cmd_297515.sh` 写入并由 `try_lock_297515` 触发的训练命令：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
python /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=CPC/CPC_main" train.nnodes=1 train.devices=1
```

真实运行时间点：

- 启动时间：`2026-06-19T02:31:10+0800`。
- 最终观察时间：`2026-06-19T02:43:02+0800`。
- 连续观察时长：约 11 分 52 秒。

观察结论：

- `_main` 已越过 `pl.Trainer(...)` 构造、batch size/accumulate 解析、`trainer.fit(...)` 入口和模型摘要打印。
- 最终观察时 Slurm job `297515` 仍为 `R` 状态，节点为 `hnode01`。
- `/home/penghongen/try_lock_297515` 未重新出现，说明训练没有异常退出回到 lock 暂停态。
- `/home/penghongen/after_lock_297515` 仍存在，资源未释放。
- 没有设置任何“10 分钟后自动停止”的逻辑；观察完成后让训练继续在原卡运行。

运行目录：

```text
/home/penghongen/My_Project/feedback_plus/logs/CPC/CPC_main____job297515
```

Slurm 日志：

```text
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.out
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.err
```

## 当前状态

截至 `2026-06-19 02:43:02 +08:00`：

- `CPC_main` 训练仍在 `job 297515 / hnode01` 上运行。
- 不需要再删除 `try_lock_297515`；它当前不存在。
- 不要为了结束本轮工作去删除 `after_lock_297515`，除非用户明确要求释放资源。
- 如后续要接着使用该 allocation，等待当前训练自然结束后 `_train_core.sh` 会重新生成 `try_lock_297515`。

## 给下一位 agent 的注意事项

- 不要把第一次 `_main` 失败误判为最终失败；那次是 `ModelCheckpoint` state_key 冲突，已经修复并重新通过全量测试。
- 最终有效的 `_main` 启动是 `2026-06-19T02:31:10+0800` 这一轮。
- 如果需要查看训练是否还在跑，只读：
  - `squeue -j 297515`
  - `ls -l /home/penghongen/*_lock_297515 /home/penghongen/run_cmd_297515.sh`
  - `tail /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.out`
- 不要用普通 SSH 会话跑模型训练；仍然遵守 `try_lock/pre_lock/after_lock` 资源纪律。
