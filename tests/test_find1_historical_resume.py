"""验证历史 Find_1 完整续训的最小数据位置恢复能力。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from ops.find1_historical_resume.rebase_checkpoint import rebase_checkpoint
from ops.find1_historical_resume.resume_guard import CompletedValidationResumeGuard
from src.train import (
    ResumeSkippingSampler,
    WarmupPlateauController,
    _resolve_resume_checkpoint,
)


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


class _ToyValidationResumeModel(pl.LightningModule):
    """记录恢复后的训练与 validation 执行顺序。"""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.events = events

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        self.events.append(f"train:{int(batch.item())}")
        return self.weight * batch.float().mean()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        del batch, batch_idx
        self.events.append("validation_batch")

    def on_validation_epoch_end(self) -> None:
        self.log("validation_score", self.weight.detach(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _SaveAndStopAfterValidation(pl.Callback):
    """保存完成 validation 的中途 checkpoint，并结束首段训练。"""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        trainer.save_checkpoint(str(self.checkpoint_path))
        raise _CheckpointWritten


class _ValidationSideEffectProbe(pl.Callback):
    """记录 validation end；生产副作用回调均由相同边界触发。"""

    def __init__(self, records: list[tuple[int, int]]) -> None:
        self.records = records
        self._batch_count = 0

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        self._batch_count = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._batch_count += 1

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.records.append((int(trainer.global_step), self._batch_count))


def _sha256_file(path: Path) -> str:
    """计算测试 checkpoint 的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_completed_validation_loops() -> dict[str, object]:
    """返回原始 Find_1 checkpoint 的 Lightning 2.2.5 循环状态结构。"""

    train_current = {"ready": 45150, "started": 45150, "processed": 45150, "completed": 45150}
    validation_current = {"ready": 551, "started": 551, "processed": 551, "completed": 551}
    return {
        "fit_loop": {
            "epoch_loop.batch_progress": {
                "total": dict(train_current),
                "current": dict(train_current),
                "is_last_batch": False,
            },
            "epoch_loop.val_loop.batch_progress": {
                "total": {"ready": 5510, "started": 5510, "processed": 5510, "completed": 5510},
                "current": validation_current,
                "is_last_batch": True,
            },
            "epoch_progress": {
                "total": {"ready": 1, "started": 1, "processed": 0, "completed": 0},
                "current": {"ready": 1, "started": 1, "processed": 0, "completed": 0},
            },
            "epoch_loop.state_dict": {"_batches_that_stepped": 11287},
            "state_dict": {},
        }
    }


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


def test_completed_validation_resume_starts_with_training_and_next_validation_is_full(
    tmp_path: Path,
) -> None:
    """完成 validation 的 checkpoint 不重跑残余 batch，下一次 validation 完整处理 551 batches。"""

    train_data = torch.arange(4, dtype=torch.float32)
    validation_data = torch.arange(551, dtype=torch.float32)
    checkpoint_path = tmp_path / "completed_validation.ckpt"
    first_events: list[str] = []
    first_trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        callbacks=[_SaveAndStopAfterValidation(checkpoint_path)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=0.5,
    )
    with pytest.raises(_CheckpointWritten):
        first_trainer.fit(
            _ToyValidationResumeModel(first_events),
            train_dataloaders=torch.utils.data.DataLoader(train_data, batch_size=1),
            val_dataloaders=torch.utils.data.DataLoader(validation_data, batch_size=1),
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    fit_loop = checkpoint["loops"]["fit_loop"]
    assert fit_loop["epoch_loop.batch_progress"]["current"]["completed"] == 2
    assert fit_loop["epoch_loop.val_loop.batch_progress"]["current"]["completed"] == 551
    assert fit_loop["epoch_loop.val_loop.batch_progress"]["is_last_batch"] is True

    def resume_once() -> tuple[list[str], list[tuple[int, int]]]:
        resumed_events: list[str] = []
        side_effects: list[tuple[int, int]] = []
        resume_sampler = ResumeSkippingSampler(
            _TrackingSampler(size=4),
            skip_batches=2,
            batch_size=1,
            resume_epoch=0,
        )
        resumed_trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            callbacks=[CompletedValidationResumeGuard(), _ValidationSideEffectProbe(side_effects)],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            val_check_interval=0.5,
        )
        resumed_trainer.fit(
            _ToyValidationResumeModel(resumed_events),
            train_dataloaders=torch.utils.data.DataLoader(
                train_data,
                batch_size=1,
                sampler=resume_sampler,
            ),
            val_dataloaders=torch.utils.data.DataLoader(validation_data, batch_size=1),
            ckpt_path=str(checkpoint_path),
        )
        return resumed_events, side_effects

    for _ in range(2):
        resumed_events, side_effects = resume_once()
        assert resumed_events[:2] == ["train:2", "train:3"]
        assert resumed_events.count("validation_batch") == 551
        assert side_effects == [(4, 551)]


@pytest.mark.parametrize(("num_nodes", "devices"), [(1, 2), (2, 1)])
def test_completed_validation_guard_is_topology_independent(num_nodes: int, devices: int) -> None:
    """单节点双卡与两节点各单卡读取同一 rank-local 恢复边界。"""

    train_current = SimpleNamespace(ready=45150, started=45150, processed=45150, completed=45150)
    validation_current = SimpleNamespace(ready=551, started=551, processed=551, completed=551)
    trainer = SimpleNamespace(
        num_nodes=num_nodes,
        num_devices=devices,
        num_val_batches=[551],
        val_check_batch=4515,
        is_global_zero=False,
        fit_loop=SimpleNamespace(
            epoch_progress=SimpleNamespace(current=SimpleNamespace(started=1, completed=0)),
            epoch_loop=SimpleNamespace(
                restarting=True,
                batch_progress=SimpleNamespace(current=train_current, is_last_batch=False),
                val_loop=SimpleNamespace(
                    batch_progress=SimpleNamespace(current=validation_current, is_last_batch=True)
                ),
            ),
        ),
    )
    guard = CompletedValidationResumeGuard()

    guard.on_train_start(trainer, SimpleNamespace())
    assert trainer.val_check_batch == float("inf")
    guard.on_train_batch_start(trainer, SimpleNamespace(), object(), 45150)
    assert trainer.val_check_batch == 4515


def test_original_plateau_state_is_idempotent_at_same_global_step() -> None:
    """旧三元 validation 身份恢复后，同一 global_step 不再推进 plateau。"""

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=5.0e-5)
    trainer = SimpleNamespace(
        optimizers=[optimizer],
        global_step=11287,
        current_epoch=0,
        estimated_stepping_batches=100000,
        sanity_checking=False,
        callback_metrics={},
    )
    controller = WarmupPlateauController(
        sched_cfg={
            "name": "warmup_plateau",
            "total_steps": 100000,
            "warmup_steps": 4515,
            "warmup_ratio": 0.0,
            "warmup_start_factor": 0.33,
            "mode": "max",
            "factor": 0.2,
            "patience": 3,
            "threshold": 0.003,
            "threshold_mode": "abs",
            "cooldown": 0,
            "min_lr": 0.0,
            "eps": 1.0e-8,
            "stop_after_lr_reductions": 3,
        },
        monitor_metric="validation_score",
    )
    controller.load_state_dict(
        {
            "warmup_steps": 4515,
            "plateau_schedulers": [
                {
                    "factor": 0.2,
                    "min_lrs": [0.0],
                    "patience": 3,
                    "verbose": False,
                    "cooldown": 0,
                    "cooldown_counter": 0,
                    "mode": "max",
                    "threshold": 0.003,
                    "threshold_mode": "abs",
                    "eps": 1.0e-8,
                    "last_epoch": 7,
                    "_last_lr": [5.0e-5],
                    "mode_worse": float("-inf"),
                    "best": 0.6182763576507568,
                    "num_bad_epochs": 2,
                }
            ],
            "last_stepped_validation": (11287, 0, 6),
            "validation_index": 7,
        }
    )
    controller.on_fit_start(trainer, SimpleNamespace())
    module = SimpleNamespace(_last_validation_payload={"validation_score": torch.tensor(0.571532)})

    controller.on_validation_end(trainer, module)
    state = controller.state_dict()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-5)
    assert state["plateau_schedulers"][0]["num_bad_epochs"] == 2
    assert state["last_stepped_validation"] == (11287, 0)
    assert state["validation_index"] == 7


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
    assert '"+train.resume_after_completed_validation=true"' in launcher
    assert "d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c" in launcher
    assert '--expected-source-sha256 "${source_resume_sha256}"' in launcher
    assert '"train.ddp_timeout_seconds=86400"' in launcher
    assert '"train.gradient_clip_val=0.5"' in launcher
    assert "gradient_clip_mode" not in launcher
    assert 'num_workers="${FIND1_NUM_WORKERS:-$((task_cpu_count - 1))}"' in launcher
    assert "launch_training_python.sh" in launcher


def test_rebased_checkpoint_restores_real_model_checkpoint_state(tmp_path: Path) -> None:
    """跨目录迁移后，Lightning 应恢复 score、top-k 和 last 的完整状态."""

    source_dir = tmp_path / "old" / "checkpoints"
    destination_dir = tmp_path / "new" / "checkpoints"
    source_dir.mkdir(parents=True)
    first_top = source_dir / "TOP_epoch_00_score_0.5000.ckpt"
    second_top = source_dir / "TOP_epoch_00_score_0.4000.ckpt"
    first_top.write_bytes(b"first-top")
    second_top.write_bytes(b"second-top")
    source_checkpoint = source_dir / "last.ckpt"

    original_callback = ModelCheckpoint(
        dirpath=source_dir,
        monitor="metric",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    callback_state = original_callback.state_dict()
    callback_state.update(
        {
            "best_model_score": torch.tensor(0.5),
            "best_model_path": str(first_top),
            "current_score": torch.tensor(0.5),
            "best_k_models": {
                str(first_top): torch.tensor(0.5),
                str(second_top): torch.tensor(0.4),
            },
            "kth_best_model_path": str(second_top),
            "kth_value": torch.tensor(0.4),
            "last_model_path": str(source_checkpoint),
        }
    )
    plateau_state = {
        "warmup_steps": 4515,
        "plateau_schedulers": [{"_last_lr": [5.0e-5], "num_bad_epochs": 2, "best": 0.6182763576507568}],
        "last_stepped_validation": (11287, 0, 6),
        "validation_index": 7,
    }
    source_payload = {
        "pytorch-lightning_version": "2.2.5",
        "epoch": 0,
        "global_step": 11287,
        "callbacks": {
            original_callback.state_key: callback_state,
            "WarmupPlateauController": plateau_state,
            "LearningRateReductionStopper": {"lr_reduction_count": 0, "last_lrs": [5.0e-5]},
        },
        "loops": _real_completed_validation_loops(),
        "state_dict": {"weight": torch.ones(())},
        "optimizer_states": [{"state": {0: {"step": torch.tensor(11287.0)}}, "param_groups": [{"lr": 5.0e-5}]}],
        "lr_schedulers": [{"last_epoch": 11287, "_last_lr": [5.0e-5]}],
        "voxel_ligand_candidate_class_ids": (1,),
        "voxel_ligand_p_best_by_class": torch.tensor([0.1826171875]),
        "voxel_ligand_p_sampling_by_class": torch.tensor([0.0]),
        "voxel_ligand_best_f1_before_refine_by_class": torch.tensor([0.6359091997146606]),
    }
    torch.save(source_payload, source_checkpoint)
    source_sha256 = _sha256_file(source_checkpoint)
    with pytest.raises(ValueError, match="SHA-256"):
        rebase_checkpoint(
            source_checkpoint,
            tmp_path / "wrong-source",
            expected_source_sha256="0" * 64,
        )

    rebased_checkpoint = rebase_checkpoint(
        source_checkpoint,
        destination_dir,
        expected_source_sha256=source_sha256,
    )
    rebased_payload = torch.load(rebased_checkpoint, map_location="cpu")
    rebased_state = rebased_payload["callbacks"][original_callback.state_key]
    restored_callback = ModelCheckpoint(
        dirpath=destination_dir,
        monitor="metric",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    restored_callback.load_state_dict(rebased_state)

    assert restored_callback.best_model_score.item() == pytest.approx(0.5)
    assert restored_callback.kth_value.item() == pytest.approx(0.4)
    assert Path(restored_callback.best_model_path).parent == destination_dir
    assert Path(restored_callback.kth_best_model_path).parent == destination_dir
    assert Path(restored_callback.last_model_path) == rebased_checkpoint
    assert set(restored_callback.best_k_models) == {
        str(destination_dir / first_top.name),
        str(destination_dir / second_top.name),
    }
    assert (destination_dir / first_top.name).read_bytes() == b"first-top"
    assert (destination_dir / second_top.name).read_bytes() == b"second-top"
    manifest_path = destination_dir / "resume_state_manifest.json"
    assert manifest_path.is_file()

    assert rebased_payload["global_step"] == 11287
    assert rebased_payload["loops"] == source_payload["loops"]
    assert rebased_payload["callbacks"]["WarmupPlateauController"] == plateau_state
    assert rebased_payload["callbacks"]["LearningRateReductionStopper"] == {
        "lr_reduction_count": 0,
        "last_lrs": [5.0e-5],
    }
    torch.testing.assert_close(rebased_payload["state_dict"], source_payload["state_dict"])
    torch.testing.assert_close(rebased_payload["optimizer_states"], source_payload["optimizer_states"])
    torch.testing.assert_close(rebased_payload["lr_schedulers"], source_payload["lr_schedulers"])
    torch.testing.assert_close(
        rebased_payload["voxel_ligand_p_best_by_class"],
        torch.tensor([0.1826171875]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["source_sha256"] == source_sha256
    assert manifest["resume_boundary"] == {
        "global_step": 11287,
        "epoch": 0,
        "rank_local_train_batches_completed": 45150,
        "rank_local_validation_batches_completed": 551,
        "validation_is_last_batch": True,
    }
