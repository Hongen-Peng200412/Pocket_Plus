"""处理已完整完成 validation 的 Lightning 中途恢复边界。"""

from __future__ import annotations

import lightning as pl
from lightning.pytorch.callbacks import Callback


class CompletedValidationResumeGuard(Callback):
    """让完成 validation 后保存的中途 checkpoint 从下一训练 batch 继续。"""

    def __init__(self) -> None:
        super().__init__()
        self._original_val_check_batch: int | float | None = None

    @staticmethod
    def _progress_counts(progress: object) -> tuple[int, int, int, int]:
        """读取 Lightning progress 的四阶段计数。"""

        return tuple(
            int(getattr(progress, field_name))
            for field_name in ("ready", "started", "processed", "completed")
        )

    @staticmethod
    def _num_validation_batches(trainer: pl.Trainer) -> int:
        """返回当前 rank 一次完整 validation 应处理的 batch 数。"""

        num_batches = trainer.num_val_batches
        if isinstance(num_batches, (list, tuple)):
            return sum(int(value) for value in num_batches)
        return int(num_batches)

    def _restore_validation_schedule(self, trainer: pl.Trainer) -> None:
        """恢复首个训练 batch 前暂时关闭的 validation 调度。"""

        if self._original_val_check_batch is not None:
            trainer.val_check_batch = self._original_val_check_batch
            self._original_val_check_batch = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """验证恢复边界，并屏蔽对已完成 validation 的第一次重复调度。"""

        del pl_module
        epoch_loop = trainer.fit_loop.epoch_loop
        train_progress = epoch_loop.batch_progress
        validation_progress = epoch_loop.val_loop.batch_progress
        train_counts = self._progress_counts(train_progress.current)
        validation_counts = self._progress_counts(validation_progress.current)
        expected_validation_batches = self._num_validation_batches(trainer)
        val_check_batch = trainer.val_check_batch
        boundary_is_complete = (
            bool(epoch_loop.restarting)
            and len(set(train_counts)) == 1
            and train_counts[0] > 0
            and len(set(validation_counts)) == 1
            and validation_counts[0] == expected_validation_batches
            and bool(validation_progress.is_last_batch)
            and not bool(train_progress.is_last_batch)
            and val_check_batch != float("inf")
            and train_counts[0] % int(val_check_batch) == 0
        )
        if not boundary_is_complete:
            raise RuntimeError(
                "历史续训 checkpoint 不是已完整完成 validation 的中途训练边界："
                f"epoch_loop.restarting={epoch_loop.restarting}, "
                f"train_counts={train_counts}, validation_counts={validation_counts}, "
                f"expected_validation_batches={expected_validation_batches}, "
                f"validation_is_last_batch={validation_progress.is_last_batch}, "
                f"train_is_last_batch={train_progress.is_last_batch}, val_check_batch={val_check_batch}。"
            )

        self._original_val_check_batch = val_check_batch
        trainer.val_check_batch = float("inf")
        if bool(getattr(trainer, "is_global_zero", True)):
            print(
                "[Train] 已识别完整 validation 后的中途 checkpoint；"
                "首次恢复迭代将直接处理下一训练 batch，不重复 validation。"
            )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        """首个恢复训练 batch 开始时还原原 validation 调度。"""

        del pl_module, batch, batch_idx
        self._restore_validation_schedule(trainer)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        """异常退出前还原 Trainer 内部调度值。"""

        del pl_module, exception
        self._restore_validation_schedule(trainer)
