# Find_1 完整 checkpoint 续训辅助工具

本目录处理 Find_1 在一次完整 validation 结束后，从 Lightning checkpoint 继续训练时的路径迁移。普通从头训练仍使用 `训练与运行/sh/Find_1.sh`，不会读取本目录。

## 组件与边界

- `rebase_checkpoint.py` 把 `ModelCheckpoint` 回调记录的旧目录迁移到新运行目录，并复制仍由 top-k 状态引用的 checkpoint。
- `src/train.py` 中的 `ResumeSkippingSampler` 跳过恢复 epoch 已处理的样本；`CompletedValidationResumeGuard` 阻止 Lightning 重放刚刚完成的 validation。

迁移只改写 `ModelCheckpoint` 状态中的目录和文件路径。模型参数、优化器、学习率调度器、训练循环进度、其他回调状态与模型保存的候选阈值均保持原值。源 checkpoint 和源 top-k 文件只读，不会被移动、删除或改写。

工具不会为源 checkpoint 补造不存在的参数梯度或随机数生成器状态。若 checkpoint 保存于梯度累积周期中间，尚未执行 `optimizer.step` 的参数梯度会丢失；若源文件不含 Python、NumPy、torch 或 CUDA 随机状态，随机增强、模型内随机分支与 dropout 也不能逐位续接。具体任务必须在启动命令和执行记录中明确这些边界。

## 输入要求

`--source` 必须是 Lightning 2.2.5 保存的完整 checkpoint，并满足以下条件：

- 顶层包含 `loops.fit_loop` 与 `callbacks`。
- `epoch_loop.batch_progress.current` 的 `ready`、`started`、`processed` 和 `completed` 相等且大于 `0`。
- `epoch_loop.val_loop.batch_progress.current` 的四个计数相等且大于 `0`，并且 `is_last_batch=true`。
- 训练进度的 `is_last_batch=false`，当前 epoch 已开始但尚未被 Lightning 标为完成。
- 至少一个回调状态同时包含 `dirpath`、`best_model_path` 和 `best_k_models`；其中引用的文件均可读取。

正式运行应提供 `--expected-source-sha256`，避免同一路径被替换后继续训练错误状态。示例：

```bash
python ops/find1_historical_resume/rebase_checkpoint.py \
    --source /absolute/old-run/checkpoints/last.ckpt \
    --destination /absolute/new-run/checkpoints \
    --expected-source-sha256 <64位SHA-256>
```

## 输出与重跑

目标目录的结构为：

```text
checkpoints/
├── resume_state.ckpt           # 迁移路径后的完整 Lightning checkpoint
├── resume_state_manifest.json  # 源文件、输出文件、恢复边界与复制结果
└── TOP_*.ckpt                  # ModelCheckpoint 状态仍引用的历史 top-k 文件
```

`resume_state.ckpt` 供 `trainer.fit(ckpt_path=...)` 读取。`resume_state_manifest.json` 的顶层字段如下；路径和摘要示例仅展示结构，不代表正式任务身份。

| 字段 | 示例 | 含义 |
| --- | --- | --- |
| `schema_version` | `2` | manifest 结构版本。 |
| `source_checkpoint` | `/old/checkpoints/last.ckpt` | 源 checkpoint 的绝对路径。 |
| `source_sha256` | `<64位SHA-256>` | 源 checkpoint 的文件摘要。 |
| `output_checkpoint` | `/new/checkpoints/resume_state.ckpt` | 迁移后 checkpoint 的绝对路径。 |
| `output_sha256` | `<64位SHA-256>` | 迁移后 checkpoint 的文件摘要。 |
| `random_state_handling` | `not_interpreted_or_modified_by_rebase_tool` | 本工具不解释或修改 checkpoint 中的随机状态；逐位续接能力必须根据源字段另行核验。 |
| `resume_boundary` | `{"global_step": 14287, "epoch": 0, "rank_local_train_batches_completed": 114300, "rank_local_validation_batches_completed": 1250, "validation_is_last_batch": true}` | 已核验的训练位置与完整 validation 边界。 |
| `rebased_callback_keys` | `["ModelCheckpoint{...}"]` | 已迁移路径的 `ModelCheckpoint` 回调键。 |
| `copied_checkpoint_artifacts` | `{"/old/checkpoints/TOP.ckpt": {"destination": "/new/checkpoints/TOP.ckpt", "sha256": "<64位SHA-256>"}}` | 每个源 top-k 文件的目标路径与摘要。 |

工具按单进程运行。复制 top-k 文件时先写入隐藏临时文件，再原子发布；同名目标文件已经存在且 SHA-256 相同时会复用，内容不同时立即报错。由工具生成的 `resume_state.ckpt` 与 `resume_state_manifest.json` 也以原子替换方式发布。失败后可以在确认目标目录仍属于同一次恢复后重跑；正式任务应为每次恢复使用独立的新运行目录。
