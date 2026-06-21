# Phase 4 handoff：train.py 通用化与 small-increment 停训上移

更新时间：2026-06-19 01:49:41 +08:00

## 背景

用户指出：`train.py` 里已有一部分 small-increment 停训能力，但 wrapper 中也保留了 warmup-plateau 的状态保存/恢复和 validation 后推进逻辑。用户希望 wrapper 的这份能力重新上移到 `train.py`，成为通用训练逻辑；不需要兼容 `configs/experiment/CPC` 下的老配置。

## 已完成

- `src/train.py`
  - 新增 `WarmupPlateauController(Callback)`。
  - callback 在 `on_validation_end` 中读取 `pl_module._last_validation_payload[monitor_metric]`，按 `cfg.train.scheduler` 推进 `torch.optim.lr_scheduler.ReduceLROnPlateau`。
  - callback 的 state_dict 保存 plateau scheduler 状态、warmup step、validation index；普通断点恢复由 Lightning callback state 恢复。
  - `LearningRateReductionStopper` 保持在 `train.py`，并在 `WarmupPlateauController` 之后注册，确保先降 LR、再观察实际 LR 下降次数。
  - `cfg.train.scheduler.name == "warmup_plateau"` 时自动注册 `WarmupPlateauController`。

- `src/wrappers/voxel_point_stage1.py`
  - 删除 `_warmup_plateau_scheduler` 和 `_pending_warmup_plateau_state`。
  - 删除 `_step_warmup_plateau_scheduler()`。
  - 删除 checkpoint 中 `warmup_plateau_reduce_on_plateau_state` 的保存/恢复。
  - 新增 `_last_validation_payload`，仅作为 validation 指标快照，供 `train.py` 通用 callback 消费。

- `src/wrappers/voxel_point_stage1_scheduler.py`
  - `configure_stage1_optimizers()` 现在只返回 `(config, candidate_warmup_steps)`。
  - `warmup_plateau` 分支只构建 step 级 warmup scheduler；plateau 部分不再下沉到 wrapper/helper。

- 配置与测试
  - `configs/train/B40_L3.yaml` 和 `configs/model/default.yaml` 的注释已改为指向 `train.py: WarmupPlateauController`。
  - `tests/test_warmup_plateau_scheduler.py` 已重写为测试当前主线的 `WarmupPlateauController`，覆盖 warmup gate、PyTorch patience 语义、state_dict 恢复、缺 monitor metric fail-fast。

## 本地验证

已通过：

```powershell
python -m py_compile src\train.py src\wrappers\voxel_point_stage1.py src\wrappers\voxel_point_stage1_scheduler.py src\modules\lr_schedulers.py
python -m py_compile tests\test_warmup_plateau_scheduler.py
```

本地未通过但原因是环境缺依赖：

```powershell
python -m pytest tests\test_warmup_plateau_scheduler.py -q
# 失败原因: ModuleNotFoundError: No module named 'rootutils'
```

该失败不是项目逻辑结论；必须在服务器 conda 环境中重跑。

## 注意事项

- `src/modules/lr_schedulers.py::WarmupThenReduceLROnPlateau` 目前只剩 `src/wrappers/voxel_point_stage1_old.py` 引用；主线不再使用。暂未删除旧文件，避免扩大本轮改动。
- 不要为了旧 checkpoint 里的 `warmup_plateau_reduce_on_plateau_state` 加兼容恢复逻辑。普通断点恢复应走新的 callback state；跨阶段 model-only 初始化本来不恢复 optimizer/scheduler/global_step。
- 如果服务器测试发现 `WarmupPlateauController` 在 Lightning hook 顺序中读取不到 `_last_validation_payload`，优先检查 wrapper 是否确实进入了 `on_validation_epoch_end()` 并写入 monitor key，不要把 plateau 逻辑重新放回 wrapper。

## 待完成

1. 在服务器 conda 环境运行：
   - `pytest tests/test_warmup_plateau_scheduler.py -q`
   - changed tests
   - full pytest
2. 若测试失败，按当前新结构修测试或代码，不兼容旧 wrapper 下沉逻辑。
3. `_main` 真实训练观察至少 10 分钟无异常退出；观察后让任务继续训练，不写十分钟自动停止逻辑。
4. 最终更新本 handoff，记录服务器测试命令、结果、日志路径、`_main` 任务状态。
