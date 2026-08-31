"""验证历史 Find_1 完整续训的最小数据位置恢复能力。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import lightning as pl

from src.train import ResumeSkippingSampler, _resolve_resume_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TrackingSampler(torch.utils.data.Sampler[int]):
    """提供可观察 epoch 的确定性测试 sampler。"""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        begin = self.epoch * 100
        return iter(range(begin, begin + self.size))

    def __len__(self) -> int:
        return self.size


class _ToyResumeModel(pl.LightningModule):
    """记录实际消费样本的最小 Lightning 训练模型。"""

    def __init__(self, consumed: list[int]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.consumed = consumed

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        self.consumed.extend(int(value) for value in batch.flatten().tolist())
        return self.weight * batch.float().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _CheckpointWritten(RuntimeError):
    """表示测试 checkpoint 已经在目标 batch 边界写完。"""


class _SaveAndStopAtBatch(pl.Callback):
    """在指定 batch 开始前保存前序完整进度并结束第一次训练。"""

    def __init__(self, checkpoint_path: Path, batch_idx: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.batch_idx = int(batch_idx)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del pl_module, batch
        if batch_idx != self.batch_idx:
            return
        trainer.save_checkpoint(str(self.checkpoint_path))
        raise _CheckpointWritten


def test_resume_sampler_skips_rank_local_batches_only_in_resume_epoch() -> None:
    """恢复 epoch 跳过既有 batch，后续 epoch 恢复完整确定性顺序。"""

    base_sampler = _TrackingSampler(size=12)
    sampler = ResumeSkippingSampler(
        base_sampler,
        skip_batches=2,
        batch_size=3,
        resume_epoch=0,
    )

    assert list(sampler) == list(range(6, 12))
    assert len(sampler) == 12

    sampler.set_epoch(1)
    assert list(sampler) == list(range(100, 112))
    assert base_sampler.epoch == 1


def test_resume_sampler_preserves_original_loader_length() -> None:
    """DataLoader 长度保留原 epoch 大小，但 iterator 只交付剩余 batch。"""

    sampler = ResumeSkippingSampler(
        _TrackingSampler(size=12),
        skip_batches=2,
        batch_size=3,
        resume_epoch=0,
    )
    loader = torch.utils.data.DataLoader(
        list(range(12)),
        batch_size=3,
        sampler=sampler,
    )

    assert len(loader) == 4
    assert [batch.tolist() for batch in loader] == [[6, 7, 8], [9, 10, 11]]


def test_lightning_mid_epoch_resume_consumes_only_remaining_samples(
    tmp_path: Path,
) -> None:
    """完整 checkpoint 进度与保留原长度的 sampler 共同续完当前 epoch。"""

    dataset = torch.arange(12, dtype=torch.float32)
    checkpoint_path = tmp_path / "mid_epoch.ckpt"
    first_consumed: list[int] = []
    first_trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        callbacks=[_SaveAndStopAtBatch(checkpoint_path, batch_idx=5)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    with pytest.raises(_CheckpointWritten):
        first_trainer.fit(
            _ToyResumeModel(first_consumed),
            train_dataloaders=torch.utils.data.DataLoader(dataset, batch_size=1),
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    completed = checkpoint["loops"]["fit_loop"][
        "epoch_loop.batch_progress"
    ]["current"]["completed"]

    assert first_consumed == list(range(5))
    assert completed == 5

    resume_sampler = ResumeSkippingSampler(
        _TrackingSampler(size=12),
        skip_batches=completed,
        batch_size=1,
        resume_epoch=0,
    )
    resumed_consumed: list[int] = []
    resumed_trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed_trainer.fit(
        _ToyResumeModel(resumed_consumed),
        train_dataloaders=torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=resume_sampler,
        ),
        ckpt_path=str(checkpoint_path),
    )

    assert resumed_consumed == list(range(5, 12))


def test_resume_checkpoint_requires_an_existing_file(tmp_path: Path) -> None:
    """完整续训入口只接受真实 checkpoint 文件。"""

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    assert _resolve_resume_checkpoint(str(checkpoint)) == checkpoint.resolve()

    with pytest.raises(FileNotFoundError, match="resume_from_checkpoint"):
        _resolve_resume_checkpoint(str(tmp_path / "missing.ckpt"))


def test_historical_find1_launcher_freezes_resume_identity() -> None:
    """历史 shell 固定字面 last checkpoint、跳过量与全局裁剪边界。"""

    launcher = (
        PROJECT_ROOT / "训练与运行" / "sh" / "Find_1.sh"
    ).read_text(encoding="utf-8")

    assert "Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt" in launcher
    assert '"train.resume_skip_train_batches=45150"' in launcher
    assert '"train.resume_skip_epoch=0"' in launcher
    assert '"train.gradient_clip_val=0.5"' in launcher
    assert "gradient_clip_mode" not in launcher
    assert 'num_workers="${FIND1_NUM_WORKERS:-$((task_cpu_count - 1))}"' in launcher
    assert "launch_training_python.sh" in launcher
