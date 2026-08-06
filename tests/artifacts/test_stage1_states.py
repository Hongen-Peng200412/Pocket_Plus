"""PDB `_RUNNING`、role `_COMPLETE`、`_BLOB_EXCEED` 与 runner 续跑测试. """

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from src.artifacts.io import atomic_write_json
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import (
    PdbRunningLease,
    mark_blob_exceed,
    mark_role_complete,
    pdb_is_consumable,
)
from src.component_lineage.clg import CLGEnumerationConfig
from src.evaluation.calibration import calibrate_thresholds, publish_threshold_calibration
from src.inference.full_map import FullMapResult, publish_full_map
from src.inference.runner import (
    BlobExceeded,
    ProductionTask,
    Stage1ProductionRunner,
    load_component_runtime_contract,
    make_component_role_producer,
    make_f1_clg_centered_role_producers,
)


def test_running_lease_is_pdb_scoped_and_owner_safe(tmp_path: Path) -> None:
    paths = Stage1ArtifactPaths(tmp_path, "Find_0", "train", "1abc")
    first = PdbRunningLease.acquire(paths, "worker-a")
    assert first is not None
    assert PdbRunningLease.acquire(paths, "worker-b") is None
    assert not pdb_is_consumable(paths, ("probability",))
    first.release()
    second = PdbRunningLease.acquire(paths, "worker-b")
    assert second is not None
    second.release()


def test_complete_and_blob_exceed_are_distinct_terminal_states(tmp_path: Path) -> None:
    paths = Stage1ArtifactPaths(tmp_path, "Find_1", "train", "2def")
    mark_role_complete(paths, "probability")
    assert pdb_is_consumable(paths, ("probability",))
    mark_blob_exceed(paths, n_f1_eligible=201, limit=200)
    assert not pdb_is_consumable(paths, ("probability",))
    assert not paths.role_complete_path("components").exists()


def test_runner_resumes_by_role_and_stops_blob_exceed(tmp_path: Path) -> None:
    calls: list[str] = []

    def probability(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        calls.append(f"{task.pdb_id}:probability")
        mark_role_complete(paths, "probability")

    def components(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
        del paths
        calls.append(f"{task.pdb_id}:components")
        raise BlobExceeded(n_f1_eligible=250, limit=200)

    runner = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={"probability": probability, "components": components},
        owner_token="test-worker",
    )
    task = ProductionTask("unet_c1", "train", "3ghi")
    record = runner.run_task(task, ("probability", "components"))
    assert record.status == "blob_exceed"
    assert record.completed_roles == ("probability",)
    assert calls == ["3ghi:probability", "3ghi:components"]
    assert not Stage1ArtifactPaths(tmp_path, "unet_c1", "train", "3ghi").running_dir.exists()
    second = runner.run_task(task, ("probability", "components"))
    assert second.status == "blob_exceed"
    assert calls == ["3ghi:probability", "3ghi:components"]


def test_runner_can_continue_after_blob_exceed_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    """超限标记保留时，显式开关仍允许补充已有契约角色。"""

    task = ProductionTask("Find_0", "calibration", "overflow")
    paths = Stage1ArtifactPaths(tmp_path, "Find_0", "calibration", "overflow")
    mark_blob_exceed(paths, n_f1_eligible=250, limit=200)

    def components(current_task: ProductionTask, current_paths: Stage1ArtifactPaths) -> None:
        assert current_task == task
        mark_role_complete(current_paths, "components")

    blocked = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={"components": components},
        owner_token="blocked-worker",
    )
    assert blocked.run_task(task, ("components",)).status == "blob_exceed"

    continued = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={"components": components},
        owner_token="continued-worker",
        continue_on_blob_exceed=True,
    )
    record = continued.run_task(task, ("components",))
    assert record.status == "completed"
    assert paths.blob_exceed_path.is_file()
    assert paths.role_complete_path("components").is_file()


def test_two_phase_entries_require_frozen_thresholds_and_keep_role_order(tmp_path: Path) -> None:
    calls: list[str] = []

    def producer_for(role: str):
        def produce(task: ProductionTask, paths: Stage1ArtifactPaths) -> None:
            calls.append(f"{task.split}/{task.pdb_id}:{role}")
            mark_role_complete(paths, role)

        return produce

    runner = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={
            role: producer_for(role)
            for role in ("probability", "components", "F1_centered", "CLG_centered")
        },
        owner_token="phase-worker",
    )
    validation = ProductionTask("Find_0", "validation", "val_a")
    calibration = ProductionTask("Find_0", "calibration", "cal_a")
    train = ProductionTask("Find_0", "train", "train_a")
    with pytest.raises(RuntimeError, match="尚未冻结 calibration 阈值"):
        runner.run_validation_calibration_train((validation,), (calibration,), (train,))

    calibration_marker = Stage1ArtifactPaths(
        tmp_path, "Find_0", "calibration", "unused"
    ).calibration_complete_path
    atomic_write_json(calibration_marker, {"stage1_model_name": "Find_0"})
    mark_role_complete(
        Stage1ArtifactPaths(tmp_path, "Find_0", "calibration", "cal_a"),
        "probability",
    )
    records = runner.run_validation_calibration_train(
        (validation,), (calibration,), (train,)
    )
    assert [record.task.pdb_id for record in records] == ["val_a", "cal_a", "train_a"]
    assert calls == [
        "validation/val_a:probability",
        "validation/val_a:components",
        "validation/val_a:F1_centered",
        "validation/val_a:CLG_centered",
        "calibration/cal_a:components",
        "calibration/cal_a:F1_centered",
        "calibration/cal_a:CLG_centered",
        "train/train_a:probability",
        "train/train_a:components",
        "train/train_a:F1_centered",
        "train/train_a:CLG_centered",
    ]


def test_formal_component_producer_uses_unique_threshold_layers_and_publishes(tmp_path: Path) -> None:
    shape = (80, 80, 80)
    task = ProductionTask("unet_c1", "calibration", "cal_component")
    paths = Stage1ArtifactPaths(tmp_path, task.stage1_model_name, task.split, task.pdb_id)
    probability = np.zeros(shape, dtype=np.float32)
    probability[10, 20, 30] = 0.9
    publish_full_map(
        paths=paths,
        result=FullMapResult(
            probability_map=probability,
            weight_sum=np.ones(shape, dtype=np.float32),
            window_starts_zyx=((0, 0, 0),),
        ),
        origin_xyz=(0.0, 0.0, 0.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    target = np.zeros(shape, dtype=np.bool_)
    target[10, 20, 30] = True
    calibration = calibrate_thresholds([(probability, target)], denominator=8)
    publish_threshold_calibration(
        paths=paths,
        result=calibration,
        min_voxels=1,
        max_voxels=10,
        fitted_metrics={"smoke": True},
    )
    runtime = load_component_runtime_contract(paths)
    assert runtime.threshold_grid_indices_descending == (1,)

    occurrence = np.asarray(
        [np.ravel_multi_index((10, 20, 30), shape)], dtype=np.int64
    )
    component_producer = make_component_role_producer(
        occurrence_voxel_provider=lambda current_task, full_shape: {7: occurrence},
        clg_config=CLGEnumerationConfig(1, 1, 32),
    )
    runner = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={"components": component_producer},
        owner_token="component-worker",
    )
    record = runner.run_task(task, ("components",))
    assert record.status == "completed"
    assert paths.role_complete_path("components").is_file()
    summary = json.loads(paths.component_summary_json.read_text(encoding="utf-8"))
    assert summary["threshold_grid_indices_descending"] == [1]
    assert summary["n_f1_eligible_seeds"] == 1

    class Wrapper:
        def __call__(self, batch):
            batch_size = len(batch["requests"])
            ligand_logits = np.full(shape, -80.0, dtype=np.float32)
            ligand_logits[probability > 0.0] = np.log(
                probability[probability > 0.0]
                / (1.0 - probability[probability > 0.0])
            )
            return {
                "voxel_logits_ligand": np.broadcast_to(
                    ligand_logits, (batch_size, 1, *shape)
                ),
                "voxel_logits_aux": np.zeros(
                    (batch_size, 1, *shape), dtype=np.float32
                ),
                "voxel_features": {
                    "voxel_final": np.broadcast_to(
                        np.zeros((1, 48, 1, 1, 1), dtype=np.float16),
                        (batch_size, 48, *shape),
                    ),
                },
            }

    centered_producers = make_f1_clg_centered_role_producers(
        wrapper_provider=lambda current_task: Wrapper(),
        batch_builder_provider=lambda current_task: lambda requests: {
            "requests": tuple(requests),
            "hardmask": np.zeros((len(requests), *shape), dtype=np.bool_),
            "box_shape_zyx": np.asarray([shape] * len(requests), dtype=np.int64),
            "voxel_size_world": np.ones((len(requests), 3), dtype=np.float32),
        },
        centered_batch_size=12,
    )
    centered_runner = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers=centered_producers,
        owner_token="centered-worker",
    )
    centered_record = centered_runner.run_task(
        task, ("F1_centered", "CLG_centered")
    )
    assert centered_record.completed_roles == ("F1_centered", "CLG_centered")
    assert paths.centered_npz("F1_centered").is_file()
    assert paths.centered_npz("CLG_centered").is_file()
