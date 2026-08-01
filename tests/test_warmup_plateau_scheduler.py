from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.connectors.callback_connector import _validate_callbacks_list

from src.train import (
    BestCheckpointAlias,
    LearningRateReductionStopper,
    PeriodicCheckpointSaver,
    WarmupPlateauController,
)


class _FakeTrainer:
    def __init__(self, optimizer: torch.optim.Optimizer, *, global_step: int, epoch: int = 0) -> None:
        """
        提供 WarmupPlateauController 单测所需的最小 Trainer 接口. 

        输入参数:
            - optimizer: torch.optim.Optimizer, 被调度的优化器
            - global_step: int, 当前 optimizer step 计数
            - epoch: int, 当前 epoch index
        """
        self.optimizers = [optimizer]
        self.global_step = int(global_step)
        self.current_epoch = int(epoch)
        self.estimated_stepping_batches = 100
        self.sanity_checking = False
        self.callback_metrics: dict[str, torch.Tensor] = {}


def _build_optimizer(lr: float) -> torch.optim.Optimizer:
    """
    构建单参数 SGD 优化器用于调度器单测. 

    输入参数:
        - lr: float, 初始学习率

    输出:
        - optimizer: torch.optim.Optimizer, 只包含一个参数组的 SGD 优化器
    """
    # torch.nn.Parameter, (), 单测用可训练参数
    param = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([param], lr=lr)


def _build_controller(*, warmup_steps: int, patience: int, threshold: float) -> WarmupPlateauController:
    """
    构建 max 指标方向的 warmup_plateau controller. 

    输入参数:
        - warmup_steps: int, warmup 覆盖的 optimizer step 数
        - patience: int, ReduceLROnPlateau 原生 patience
        - threshold: float, 绝对改进阈值

    输出:
        - controller: WarmupPlateauController, train.py 中实际注册的 callback
    """
    return WarmupPlateauController(
        sched_cfg={
            "name": "warmup_plateau",
            "total_steps": 100,
            "warmup_steps": int(warmup_steps),
            "warmup_ratio": 0.0,
            "warmup_start_factor": 1.0,
            "mode": "max",
            "factor": 0.5,
            "patience": int(patience),
            "threshold": float(threshold),
            "threshold_mode": "abs",
            "cooldown": 0,
            "min_lr": 0.0,
            "eps": 1.0e-8,
            "stop_after_lr_reductions": None,
        },
        monitor_metric="val_score/global/atom_PRAUC",
    )


def _step_validation(controller: WarmupPlateauController, trainer: _FakeTrainer, metric_value: float) -> None:
    """
    模拟一次 wrapper 已经聚合完 payload 后的 validation end. 

    输入参数:
        - controller: WarmupPlateauController, 待测试 callback
        - trainer: _FakeTrainer, 最小 Trainer 替身
        - metric_value: float, 当前 validation 主指标
    """
    module = SimpleNamespace(
        _last_validation_payload={
            "val_score/global/atom_PRAUC": torch.tensor(float(metric_value)),
        }
    )
    controller.on_validation_end(trainer, module)


def test_plateau_waits_until_warmup_is_finished() -> None:
    optimizer = _build_optimizer(lr=1.0)
    trainer = _FakeTrainer(optimizer, global_step=4)
    controller = _build_controller(warmup_steps=5, patience=0, threshold=0.0)

    _step_validation(controller, trainer, 1.0)
    _step_validation(controller, trainer, 0.5)
    assert optimizer.param_groups[0]["lr"] == 1.0

    trainer.global_step = 5
    _step_validation(controller, trainer, 1.0)
    trainer.global_step = 6
    _step_validation(controller, trainer, 0.5)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_plateau_uses_pytorch_native_patience_semantics() -> None:
    optimizer = _build_optimizer(lr=1.0)
    trainer = _FakeTrainer(optimizer, global_step=0)
    controller = _build_controller(warmup_steps=0, patience=1, threshold=0.01)

    _step_validation(controller, trainer, 1.0)
    trainer.global_step = 1
    _step_validation(controller, trainer, 1.005)
    assert optimizer.param_groups[0]["lr"] == 1.0

    trainer.global_step = 2
    _step_validation(controller, trainer, 1.004)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_plateau_state_dict_roundtrip_preserves_bad_epoch_count() -> None:
    optimizer = _build_optimizer(lr=1.0)
    trainer = _FakeTrainer(optimizer, global_step=0)
    controller = _build_controller(warmup_steps=0, patience=1, threshold=0.0)

    _step_validation(controller, trainer, 1.0)
    trainer.global_step = 1
    _step_validation(controller, trainer, 0.9)
    state = controller.state_dict()

    restored_optimizer = _build_optimizer(lr=1.0)
    restored_trainer = _FakeTrainer(restored_optimizer, global_step=2)
    restored_controller = _build_controller(warmup_steps=0, patience=1, threshold=0.0)
    restored_controller.load_state_dict(state)

    _step_validation(restored_controller, restored_trainer, 0.8)
    assert restored_optimizer.param_groups[0]["lr"] == 0.5


def test_plateau_missing_monitor_metric_fails_fast() -> None:
    optimizer = _build_optimizer(lr=1.0)
    trainer = _FakeTrainer(optimizer, global_step=0)
    controller = _build_controller(warmup_steps=0, patience=0, threshold=0.0)
    module = SimpleNamespace(_last_validation_payload={})

    with pytest.raises(RuntimeError, match="monitor metric"):
        controller.on_validation_end(trainer, module)


def test_lr_reduction_stopper_does_not_log_from_validation_end() -> None:
    """学习率下降后直接更新回调状态，不能从 validation end 调用 LightningModule.log。"""

    optimizer = _build_optimizer(lr=1.0)
    trainer = _FakeTrainer(optimizer, global_step=0)
    trainer.should_stop = False
    trainer.is_global_zero = False

    class _ModuleThatRejectsLog:
        def log(self, *_args, **_kwargs) -> None:
            raise AssertionError("validation end 不应调用 LightningModule.log")

    module = _ModuleThatRejectsLog()
    stopper = LearningRateReductionStopper(stop_after_lr_reductions=1)
    stopper.on_train_start(trainer, module)
    optimizer.param_groups[0]["lr"] = 0.5

    stopper.on_validation_end(trainer, module)

    assert stopper.lr_reduction_count == 1
    assert trainer.should_stop is True


def test_checkpoint_callbacks_have_unique_lightning_state_keys(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    main_checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="TOP_epoch_{epoch:02d}_score_{val_score/global/refined_F1:.4f}",
        auto_insert_metric_name=False,
        monitor="val_score/global/refined_F1",
        mode="max",
        save_top_k=10,
        save_last=True,
    )
    callbacks = [
        main_checkpoint,
        BestCheckpointAlias(main_checkpoint, checkpoint_dir / "BEST.ckpt"),
        PeriodicCheckpointSaver(checkpoint_dir, every_n_epochs=3, filename_template="PERIODIC_epoch_{epoch:02d}"),
    ]

    _validate_callbacks_list(callbacks)
