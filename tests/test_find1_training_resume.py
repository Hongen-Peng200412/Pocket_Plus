"""验证 Find_1 在完整 validation checkpoint 处恢复训练的状态边界."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from ops.find1_historical_resume.rebase_checkpoint import rebase_checkpoint
from src.train import (
    CompletedValidationResumeGuard,
    DatasetEpochController,
    EpochAwareDistributedSampler,
    ResumeSkippingSampler,
    WarmupPlateauController,
    _resolve_resume_checkpoint,
    _validate_resume_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TrackingSampler(torch.utils.data.Sampler[int]):
    """提供可观察 epoch 的确定性测试 sampler."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(range(self.size))

    def __len__(self) -> int:
        return self.size


class _CheckpointWritten(RuntimeError):
    """表示测试 checkpoint 已在目标 validation 边界写完."""


class _ResumeModel(pl.LightningModule):
    """记录恢复后的训练与 validation 执行顺序."""

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
    """保存目标次完整 validation 后的 checkpoint, 随后结束首段训练."""

    def __init__(self, checkpoint_path: Path, target_validation_count: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.target_validation_count = int(target_validation_count)
        self.completed_validation_count = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.completed_validation_count += 1
        if self.completed_validation_count != self.target_validation_count:
            return
        trainer.save_checkpoint(str(self.checkpoint_path))
        raise _CheckpointWritten


class _ValidationSideEffectProbe(pl.Callback):
    """记录每次 validation end 对应的 global step 与完整 batch 数."""

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
    """计算测试 checkpoint 的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_validation_loops() -> dict[str, object]:
    """返回 Job 366071 `last.ckpt` 对应的 Lightning 2.2.5 循环结构."""

    train_current = {
        "ready": 114300,
        "started": 114300,
        "processed": 114300,
        "completed": 114300,
    }
    validation_current = {
        "ready": 1250,
        "started": 1250,
        "processed": 1250,
        "completed": 1250,
    }
    return {
        "fit_loop": {
            "epoch_loop.batch_progress": {
                "total": dict(train_current),
                "current": dict(train_current),
                "is_last_batch": False,
            },
            "epoch_loop.val_loop.batch_progress": {
                "total": dict(validation_current),
                "current": dict(validation_current),
                "is_last_batch": True,
            },
            "epoch_progress": {
                "total": {"ready": 1, "started": 1, "processed": 0, "completed": 0},
                "current": {"ready": 1, "started": 1, "processed": 0, "completed": 0},
            },
            "epoch_loop.state_dict": {"_batches_that_stepped": 14287},
            "state_dict": {},
        }
    }


def test_resume_sampler_skips_only_the_recorded_epoch() -> None:
    """恢复 epoch 跳过已消费 batch, 下一 epoch 重新提供全部样本."""

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
    assert list(sampler) == list(range(12))
    assert base_sampler.epoch == 1


def test_resume_sampler_is_independent_of_ddp_node_layout() -> None:
    """固定 num_replicas、rank、seed 与恢复参数时, 单节点双卡和双节点各单卡产生相同续训样本索引."""

    dataset = list(range(24))

    def resumed_indices(rank: int) -> list[int]:
        base_sampler = EpochAwareDistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=29,
        )
        sampler = ResumeSkippingSampler(
            base_sampler,
            skip_batches=2,
            batch_size=3,
            resume_epoch=0,
        )
        return list(sampler)

    single_node_two_gpu = [resumed_indices(rank) for rank in range(2)]
    two_node_one_gpu_each = [resumed_indices(rank) for rank in range(2)]

    assert single_node_two_gpu == two_node_one_gpu_each
    assert all(len(rank_indices) == 6 for rank_indices in single_node_two_gpu)


def test_last_scheduled_validation_resume_consumes_epoch_tail_without_replaying_validation(
    tmp_path: Path,
) -> None:
    """最后一次定期 validation 后仍有训练尾段时, 应先消费尾段再执行完整 validation."""

    train_data = torch.arange(13, dtype=torch.float32)
    validation_data = torch.arange(3, dtype=torch.float32)
    checkpoint_path = tmp_path / "completed_epoch_validation.ckpt"
    first_events: list[str] = []
    first_trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        callbacks=[_SaveAndStopAfterValidation(checkpoint_path, target_validation_count=3)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=4,
    )
    with pytest.raises(_CheckpointWritten):
        first_trainer.fit(
            _ResumeModel(first_events),
            train_dataloaders=torch.utils.data.DataLoader(train_data, batch_size=1),
            val_dataloaders=torch.utils.data.DataLoader(validation_data, batch_size=1),
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    fit_loop = payload["loops"]["fit_loop"]
    assert payload["epoch"] == 0
    assert fit_loop["epoch_loop.batch_progress"]["current"]["completed"] == 12
    assert fit_loop["epoch_loop.val_loop.batch_progress"]["current"]["completed"] == 3
    assert fit_loop["epoch_loop.val_loop.batch_progress"]["is_last_batch"] is True
    assert fit_loop["epoch_loop.batch_progress"]["is_last_batch"] is False

    def resume_once() -> tuple[list[str], list[tuple[int, int]]]:
        events: list[str] = []
        side_effects: list[tuple[int, int]] = []
        sampler = ResumeSkippingSampler(
            _TrackingSampler(size=13),
            skip_batches=12,
            batch_size=1,
            resume_epoch=0,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            logger=False,
            callbacks=[
                CompletedValidationResumeGuard(),
                DatasetEpochController(),
                _ValidationSideEffectProbe(side_effects),
            ],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            val_check_interval=4,
        )
        trainer.fit(
            _ResumeModel(events),
            train_dataloaders=torch.utils.data.DataLoader(
                train_data,
                batch_size=1,
                sampler=sampler,
            ),
            val_dataloaders=torch.utils.data.DataLoader(validation_data, batch_size=1),
            ckpt_path=str(checkpoint_path),
        )
        return events, side_effects

    for _ in range(2):
        resumed_events, side_effects = resume_once()
        assert resumed_events[0] == "train:12"
        assert side_effects[0] == (17, 3)
        assert all(validation_batch_count == 3 for _, validation_batch_count in side_effects)


def test_plateau_checkpoint_identity_is_idempotent_at_the_same_step() -> None:
    """旧三元 validation 身份恢复为二元键后, 不重复推进同一训练位置的 plateau."""

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=5.0e-5)
    trainer = SimpleNamespace(
        optimizers=[optimizer],
        global_step=14287,
        current_epoch=0,
        estimated_stepping_batches=100000,
        sanity_checking=False,
        callback_metrics={},
    )
    controller = WarmupPlateauController(
        sched_cfg={
            "name": "warmup_plateau",
            "total_steps": 100000,
            "warmup_steps": 5001,
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
            "stop_after_lr_reductions": 2,
        },
        monitor_metric="validation_score",
    )
    controller.load_state_dict(
        {
            "warmup_steps": 5001,
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
                    "last_epoch": 8,
                    "_last_lr": [5.0e-5],
                    "mode_worse": float("-inf"),
                    "best": 0.5602476596832275,
                    "num_bad_epochs": 0,
                }
            ],
            "last_stepped_validation": (14287, 0, 7),
            "validation_index": 8,
        }
    )
    controller.on_fit_start(trainer, SimpleNamespace())
    module = SimpleNamespace(_last_validation_payload={"validation_score": torch.tensor(0.1)})

    controller.on_validation_end(trainer, module)
    state = controller.state_dict()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-5)
    assert state["plateau_schedulers"][0]["num_bad_epochs"] == 0
    assert state["last_stepped_validation"] == (14287, 0)
    assert state["validation_index"] == 8


def test_job366071_completed_validation_guard_accepts_real_boundary() -> None:
    """恢复门禁接受 Job 366071 的 114300/1250 完整 validation 边界."""

    train_current = SimpleNamespace(
        ready=114300,
        started=114300,
        processed=114300,
        completed=114300,
    )
    validation_current = SimpleNamespace(
        ready=1250,
        started=1250,
        processed=1250,
        completed=1250,
    )
    trainer = SimpleNamespace(
        current_epoch=0,
        num_val_batches=[1250],
        val_check_batch=9525,
        is_global_zero=False,
        fit_loop=SimpleNamespace(
            epoch_loop=SimpleNamespace(
                restarting=True,
                batch_progress=SimpleNamespace(current=train_current, is_last_batch=False),
                val_loop=SimpleNamespace(
                    batch_progress=SimpleNamespace(
                        current=validation_current,
                        is_last_batch=True,
                    )
                ),
            )
        ),
    )
    guard = CompletedValidationResumeGuard()

    guard.on_train_start(trainer, SimpleNamespace())
    assert trainer.val_check_batch == float("inf")
    guard.on_train_batch_start(trainer, SimpleNamespace(), object(), 114300)
    assert trainer.val_check_batch == 9525


def test_resume_checkpoint_requires_an_existing_file(tmp_path: Path) -> None:
    """完整续训入口只接受真实 checkpoint 文件."""

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    assert _resolve_resume_checkpoint(str(checkpoint)) == checkpoint.resolve()

    with pytest.raises(FileNotFoundError, match="resume_from_checkpoint"):
        _resolve_resume_checkpoint(str(tmp_path / "missing.ckpt"))


def test_resume_configuration_requires_checkpoint_and_position_as_one_contract(
    tmp_path: Path,
) -> None:
    """checkpoint、sampler 跳过位置与 validation 门禁必须成套启用."""

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    complete_settings = {
        "resume_skip_train_batches": 114300,
        "resume_skip_epoch": 0,
        "resume_after_completed_validation": True,
    }

    _validate_resume_configuration(complete_settings, checkpoint)
    _validate_resume_configuration({}, None)
    with pytest.raises(ValueError, match="只能与 resume_from_checkpoint"):
        _validate_resume_configuration(complete_settings, None)
    with pytest.raises(ValueError, match="缺少"):
        _validate_resume_configuration({}, checkpoint)
    with pytest.raises(ValueError, match="必须大于 0"):
        _validate_resume_configuration(
            {**complete_settings, "resume_skip_train_batches": 0},
            checkpoint,
        )
    with pytest.raises(ValueError, match="必须启用"):
        _validate_resume_configuration(
            {**complete_settings, "resume_after_completed_validation": False},
            checkpoint,
        )


def test_rebased_checkpoint_preserves_job366071_training_state(tmp_path: Path) -> None:
    """迁移路径时保持 Job 366071 的模型、优化器、学习率调度器与候选阈值状态."""

    source_dir = tmp_path / "old" / "checkpoints"
    destination_dir = tmp_path / "new" / "checkpoints"
    source_dir.mkdir(parents=True)
    top_checkpoint = source_dir / "TOP_epoch_00_score_0.5602.ckpt"
    top_checkpoint.write_bytes(b"top-checkpoint")
    source_checkpoint = source_dir / "last.ckpt"
    model_checkpoint = ModelCheckpoint(
        dirpath=source_dir,
        monitor="metric",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callback_state = model_checkpoint.state_dict()
    callback_state.update(
        {
            "best_model_score": torch.tensor(0.5602476596832275),
            "best_model_path": str(top_checkpoint),
            "current_score": torch.tensor(0.5602476596832275),
            "best_k_models": {
                str(top_checkpoint): torch.tensor(0.5602476596832275),
            },
            "kth_best_model_path": str(top_checkpoint),
            "kth_value": torch.tensor(0.5602476596832275),
            "last_model_path": str(source_checkpoint),
        }
    )
    plateau_state = {
        "warmup_steps": 5001,
        "plateau_schedulers": [
            {
                "_last_lr": [5.0e-5],
                "num_bad_epochs": 0,
                "best": 0.5602476596832275,
            }
        ],
        "last_stepped_validation": (14287, 0, 7),
        "validation_index": 8,
    }
    source_payload = {
        "pytorch-lightning_version": "2.2.5",
        "epoch": 0,
        "global_step": 14287,
        "callbacks": {
            model_checkpoint.state_key: callback_state,
            "WarmupPlateauController": plateau_state,
            "LearningRateReductionStopper": {
                "lr_reduction_count": 0,
                "last_lrs": [5.0e-5],
            },
        },
        "loops": _completed_validation_loops(),
        "state_dict": {"weight": torch.ones(())},
        "optimizer_states": [
            {
                "state": {0: {"step": torch.tensor(14287.0)}},
                "param_groups": [{"lr": 5.0e-5}],
            }
        ],
        "lr_schedulers": [{"last_epoch": 14287, "_last_lr": [5.0e-5]}],
        "voxel_ligand_candidate_class_ids": (1,),
        "voxel_ligand_p_best_by_class": torch.tensor([0.2802734375]),
        "voxel_ligand_p_sampling_by_class": torch.tensor([0.0]),
        "voxel_ligand_best_f1_before_refine_by_class": torch.tensor([0.6188839078]),
    }
    torch.save(source_payload, source_checkpoint)
    source_sha256 = _sha256_file(source_checkpoint)

    rebased_checkpoint = rebase_checkpoint(
        source_checkpoint,
        destination_dir,
        expected_source_sha256=source_sha256,
    )
    rebased_payload = torch.load(rebased_checkpoint, map_location="cpu")

    assert rebased_payload["global_step"] == 14287
    assert rebased_payload["loops"] == source_payload["loops"]
    assert rebased_payload["callbacks"]["WarmupPlateauController"] == plateau_state
    torch.testing.assert_close(rebased_payload["state_dict"], source_payload["state_dict"])
    torch.testing.assert_close(
        rebased_payload["optimizer_states"],
        source_payload["optimizer_states"],
    )
    torch.testing.assert_close(
        rebased_payload["lr_schedulers"],
        source_payload["lr_schedulers"],
    )
    torch.testing.assert_close(
        rebased_payload["voxel_ligand_p_best_by_class"],
        torch.tensor([0.2802734375]),
    )
    manifest = json.loads(
        (destination_dir / "resume_state_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_sha256"] == source_sha256
    assert (
        manifest["random_state_handling"]
        == "not_interpreted_or_modified_by_rebase_tool"
    )
    assert manifest["resume_boundary"] == {
        "global_step": 14287,
        "epoch": 0,
        "rank_local_train_batches_completed": 114300,
        "rank_local_validation_batches_completed": 1250,
        "validation_is_last_batch": True,
    }


def test_resume_overrides_compose_with_hydra() -> None:
    """续训新增字段可在不改写从头训练配置时完成 Hydra 组合."""

    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"),
        version_base=None,
    ):
        config = hydra.compose(
            config_name="base",
            overrides=[
                "+experiment=CPC1/Find_1",
                "resume_from_checkpoint=/tmp/resume_state.ckpt",
                "+train.resume_skip_train_batches=114300",
                "+train.resume_skip_epoch=0",
                "+train.resume_after_completed_validation=true",
                "+train.num_sanity_val_steps=0",
            ],
        )

    assert config.resume_from_checkpoint == "/tmp/resume_state.ckpt"
    assert config.train.resume_skip_train_batches == 114300
    assert config.train.resume_skip_epoch == 0
    assert config.train.resume_after_completed_validation is True
    assert config.train.num_sanity_val_steps == 0
