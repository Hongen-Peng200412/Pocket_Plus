# -*- coding: utf-8 -*-
"""Stage1 V3 请求, NPY/mmap Dataset 与批处理契约测试."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import src.datasets.stage1_dataset as stage1_dataset_module
from ops.stage1_data_preparation.utils import generate_context_starts, sample_bias_starts
from src.datasets.density_channel_builder import ALL_CHANNEL_NAMES
from src.datasets.stage1_collate import Stage1BatchCollator
from src.datasets.stage1_dataset import Stage1Dataset
from src.datasets.stage1_requests import (
    ResolvedStage1Crop,
    Stage1TrainingRequestSet,
    centered_start_from_centroid_zyx,
    centered_start_from_sparse_mask,
    load_split_pdb_ids,
    load_validation_selection,
    resolve_stage1_start,
)


def test_split_pdb_ids_deduplicate_in_first_appearance_order(tmp_path: Path) -> None:
    """候选记录级 split 可含重复 PDB, 推理清单保持首次出现顺序."""

    split_path = tmp_path / "train.json"
    split_path.write_text(
        json.dumps(
            [
                {"pdb_id": "2DEF", "candidate_id": 0},
                {"pdb_id": "1abc", "candidate_id": 0},
                {"pdb_id": "2def", "candidate_id": 1},
            ]
        ),
        encoding="utf-8",
    )

    assert load_split_pdb_ids(split_path) == ("2def", "1abc")


def _write_upstream(
    root: Path,
    pdb_id: str = "1abc",
    shape: tuple[int, int, int] = (80, 80, 80),
) -> None:
    """写入一份最小 V3 NPY, 空间元数据, 受体与标签资产."""

    density_directory = root / "density" / pdb_id
    parse_directory = root / "parse" / pdb_id
    label_directory = root / "labels" / pdb_id
    density_directory.mkdir(parents=True)
    parse_directory.mkdir(parents=True)
    label_directory.mkdir(parents=True)

    voxel_size = np.ones(3, dtype=np.float32)
    origin = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    z, y, x = np.indices(shape, dtype=np.float32)
    exp = (x + 2.0 * y + 3.0 * z)[None]
    sim = (0.5 * x + y + 0.25 * z + 1.0)[None]
    np.save(density_directory / "exp.npy", exp, allow_pickle=False)
    np.save(density_directory / "sim.npy", sim, allow_pickle=False)
    np.savez(
        density_directory / "exp.npz",
        schema_version=np.asarray(2, dtype=np.int32),
        canonical_shape_zyx=np.asarray(shape, dtype=np.int64),
        voxel_size=voxel_size,
        origin=origin,
    )
    np.savez(
        density_directory / "sim.npz",
        schema_version=np.asarray(2, dtype=np.int32),
        voxel_size=voxel_size,
        origin=origin,
    )

    union_mask = np.zeros((1, *shape), dtype=np.bool_)
    union_mask[0, 4, 3, 2] = True
    np.save(density_directory / "union_mask.npy", union_mask, allow_pickle=False)
    np.savez_compressed(
        density_directory / "ligand_area.npz",
        schema_version=np.asarray(3, dtype=np.int32),
        grid_shape_zyx=np.asarray(shape, dtype=np.int64),
        voxel_size_xyz=voxel_size,
        origin_xyz=origin,
    )

    distance = np.full((1, *shape), 100.0, dtype=np.float16)
    distance[0, 4, 3, 2] = np.float16(3.0)
    np.save(density_directory / "ligand_dist.npy", distance, allow_pickle=False)
    np.savez_compressed(
        density_directory / "ligand_dist.npz",
        schema_version=np.asarray(1, dtype=np.uint16),
        grid_shape_zyx=np.asarray(shape, dtype=np.int64),
        voxel_size_xyz=voxel_size,
        origin_xyz=origin,
        distance_unit=np.asarray("angstrom"),
    )

    coords = np.asarray(
        [
            [10.5, 20.5, 30.5],
            [89.5, 99.5, 109.5],
            [94.0, 50.0, 60.0],
            [99.0, 50.0, 60.0],
        ],
        dtype=np.float32,
    )
    feat = np.arange(coords.shape[0] * 49, dtype=np.float32).reshape(coords.shape[0], 49)
    np.savez(
        parse_directory / "receptor_tokens.npz",
        coords=coords,
        feat=feat,
        is_backbone=np.asarray([True, True, True, False], dtype=np.bool_),
        res_type=np.asarray([0, 0, 20, 28], dtype=np.uint8),
        atom_name=np.asarray([b"CA", b"N", b"P", b"CB"], dtype="S4"),
    )
    np.savez(
        label_directory / "atom_labels.npz",
        binding_atom=np.asarray([True, False, True, False]),
    )


def _request(require_targets: bool) -> tuple[ResolvedStage1Crop, ...]:
    return (
        ResolvedStage1Crop(
            pdb_id="1ABC",
            box_start_zyx=(0, 0, 0),
            require_targets=require_targets,
            role="centered",
            occurrence_id=0,
        ),
    )


def _density_config(channels: list[str]) -> dict[str, object]:
    return {
        "clip_percentile": [0.001, 0.999],
        "fit_mask_percentile": 0.003,
        "enabled_channels": channels,
    }


def _write_v3_pool(root: Path) -> Path:
    """写入包含一个 train PDB 与一个 validation PDB 的 V3 请求池."""

    for split_name, pdb_id in (("train", "1abc"), ("validation", "2def")):
        pool_directory = root / split_name
        pool_directory.mkdir(parents=True, exist_ok=True)
        occurrence_ids = np.arange(55, dtype=np.int32)
        centers = np.stack([occurrence_ids] * 3, axis=1)
        np.savez(
            pool_directory / f"{pdb_id}.npz",
            pdb_id=np.asarray(pdb_id),
            occurrence_id=occurrence_ids,
            center_start_zyx=centers,
            bias_start_zyx=np.repeat(centers[:, None, :], 30, axis=1),
            context_start_zyx=np.stack([np.arange(10)] * 3, axis=1).astype(np.int32),
        )
    manifest = {
        "schema_version": 1,
        "splits": {
            "train": [{"pdb_id": "1abc", "path": "train/1abc.npz"}],
            "validation": [{"pdb_id": "2def", "path": "validation/2def.npz"}],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps({"schema_version": 1, "entry_ratio": {"center": 0, "bias": 5, "context": 5}}),
        encoding="utf-8",
    )
    (root / "_COMPLETE").write_text("", encoding="utf-8")
    return root


def test_resolve_stage1_start_clamps_without_padding() -> None:
    assert resolve_stage1_start((-5, 99, 12), (90, 100, 80)) == (0, 20, 0)
    with pytest.raises(ValueError, match="不小于"):
        resolve_stage1_start((0, 0, 0), (79, 80, 80))


def test_centered_start_uses_same_corner_geometry_for_centroid_and_sparse_mask() -> None:
    sparse = np.asarray([[0, 2, 4], [2, 4, 6], [4, 6, 8]], dtype=np.int32)
    centroid = sparse.astype(np.float64).mean(axis=0)
    assert centered_start_from_centroid_zyx(centroid, (120, 130, 140)) == (0, 0, 0)
    assert centered_start_from_sparse_mask(sparse, (120, 130, 140)) == (0, 0, 0)


def test_find_dataset_materializes_mmap_crop_and_separate_backbone_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_upstream(tmp_path)
    monkeypatch.setattr(
        stage1_dataset_module,
        "build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros(
            (56, *exp_raw.shape), dtype=np.float32
        ),
    )
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=_request(require_targets=True),
        mode="val",
        stage1_model_name="Find_1",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        enable_random_rotation=False,
    )

    sample = dataset[0]
    assert sample["density_input"].shape == (56, 80, 80, 80)
    assert sample["atom_global_indices"].tolist() == [0, 1, 2]
    assert sample["atom_feat"].shape == (3, 49)
    assert sample["atom_is_backbone"].tolist() == [True, True, True]
    assert sample["atom_is_in_core_box"].tolist() == [True, True, False]
    assert sample["ligand_area_target"][4, 3, 2].item() is True
    assert sample["protein_mainchain_target"][0, 0, 0].item() == 2
    assert sample["protein_mainchain_target"][79, 79, 79].item() == 1
    assert sample["ligand_inverse_distance_target"][4, 3, 2].item() == pytest.approx(0.25)


def test_unet_dataset_does_not_read_sim_or_return_atom_table(tmp_path: Path) -> None:
    _write_upstream(tmp_path)
    (tmp_path / "density" / "1abc" / "sim.npy").unlink()
    (tmp_path / "density" / "1abc" / "sim.npz").unlink()
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=_request(require_targets=True),
        mode="val",
        stage1_model_name="unet_c1",
        box_pool_root=None,
        density_channel_config=_density_config(["exp_clipnorm_nopost"]),
        enable_random_rotation=False,
    )
    sample = dataset[0]
    assert sample["density_input"].shape == (1, 80, 80, 80)
    assert "atom_feat" not in sample
    assert sample["ligand_area_target"].sum().item() == 1
    assert sample["ligand_inverse_distance_target"][4, 3, 2].item() == pytest.approx(0.25)


def test_source_cache_remains_consistent_under_inference_threads(tmp_path: Path) -> None:
    """推理线程共享 mmap LRU 时, 锁必须保持字节计数和条目映射一致."""

    _write_upstream(tmp_path)
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=_request(require_targets=False),
        mode="full_map",
        stage1_model_name="unet_c1",
        box_pool_root=None,
        density_channel_config=_density_config(["exp_clipnorm_nopost"]),
        enable_random_rotation=False,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        samples = tuple(executor.map(lambda _: dataset[0], range(8)))
    assert len(samples) == 8
    assert dataset._source_cache.current_bytes == sum(
        size for _, size in dataset._source_cache.values.values()
    )


def test_distance_validation_reads_only_the_requested_crop(tmp_path: Path) -> None:
    """裁块外的 Inf 不触发扫描, 裁块内的 Inf 按 V3 数值契约拒绝."""

    shape = (100, 100, 100)
    _write_upstream(tmp_path, shape=shape)
    distance_path = tmp_path / "density" / "1abc" / "ligand_dist.npy"
    distance = np.load(distance_path, mmap_mode="r+")
    distance[0, 99, 99, 99] = np.inf
    distance.flush()
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=_request(require_targets=True),
        mode="val",
        stage1_model_name="unet_c1",
        box_pool_root=None,
        density_channel_config=_density_config(["exp_clipnorm_nopost"]),
        enable_random_rotation=False,
    )
    dataset[0]
    distance[0, 0, 0, 0] = np.inf
    distance.flush()
    dataset._source_cache.values.clear()
    dataset._source_cache.current_bytes = 0
    with pytest.raises(ValueError, match="NaN 或 Inf"):
        dataset[0]


def test_repeated_windows_reuse_mmap_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_upstream(tmp_path)
    read_count: Counter[str] = Counter()
    original_loader = stage1_dataset_module._load_mmap_array

    def counted_loader(path: Path, expected_dtype: np.dtype) -> np.memmap:
        read_count[path.name] += 1
        return original_loader(path, expected_dtype)

    monkeypatch.setattr(stage1_dataset_module, "_load_mmap_array", counted_loader)
    monkeypatch.setattr(
        stage1_dataset_module,
        "build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros(
            (56, *exp_raw.shape), dtype=np.float32
        ),
    )
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=_request(require_targets=False),
        mode="centered",
        stage1_model_name="Find_1",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        cache_max_bytes=64 * 1024 * 1024,
        enable_random_rotation=False,
    )
    first = dataset[0]
    second = dataset[0]
    assert read_count == Counter({"exp.npy": 1, "sim.npy": 1})
    assert torch.equal(first["density_input"], second["density_input"])


def test_collator_keeps_backbone_flag_and_b_plus_one_offsets() -> None:
    common = {
        "pdb_id": "x",
        "request_role": "centered",
        "occurrence_id": None,
        "candidate_index": None,
        "box_start_zyx": torch.zeros(3, dtype=torch.int32),
        "box_shape_zyx": torch.full((3,), 80, dtype=torch.int64),
        "box_origin_world": torch.zeros(3),
        "voxel_size_world": torch.ones(3),
        "density_input": torch.zeros(1, 2, 2, 2),
        "hardmask": torch.zeros(2, 2, 2, dtype=torch.bool),
    }
    atom_fields = {
        "atom_global_indices": torch.empty(0, dtype=torch.int64),
        "atom_feat": torch.empty(0, 49),
        "atom_is_backbone": torch.empty(0, dtype=torch.bool),
        "atom_coord_world": torch.empty(0, 3),
        "atom_coord_local_voxel": torch.empty(0, 3),
        "atom_coord_centered_world": torch.empty(0, 3),
        "atom_is_in_core_box": torch.empty(0, dtype=torch.bool),
    }
    sample0 = {**common, **atom_fields}
    sample1 = {
        **common,
        **atom_fields,
        "atom_global_indices": torch.arange(2),
        "atom_feat": torch.ones(2, 49),
        "atom_is_backbone": torch.tensor([True, False]),
        "atom_coord_world": torch.zeros(2, 3),
        "atom_coord_local_voxel": torch.zeros(2, 3),
        "atom_coord_centered_world": torch.zeros(2, 3),
        "atom_is_in_core_box": torch.ones(2, dtype=torch.bool),
    }
    batch = Stage1BatchCollator()([sample0, sample1])
    assert batch["atom_counts"].tolist() == [0, 2]
    assert batch["atom_offsets"].tolist() == [0, 0, 2]
    assert batch["atom_is_backbone"].tolist() == [True, False]


def test_training_pool_rebuilds_deterministic_zero_five_five_epochs(tmp_path: Path) -> None:
    pool_root = _write_v3_pool(tmp_path)
    source = Stage1TrainingRequestSet(pool_root / "train", seed=7)
    epoch0 = tuple(source.requests)
    assert len(epoch0) == 50 * 10
    assert Counter(request.role for request in epoch0) == {"bias": 250, "context": 250}
    source.set_epoch(1)
    assert tuple(source.requests) != epoch0
    source_again = Stage1TrainingRequestSet(pool_root / "train", seed=7)
    source_again.set_epoch(1)
    assert tuple(source_again.requests) == tuple(source.requests)


def test_validation_selection_expands_zero_one_one_indices(tmp_path: Path) -> None:
    pool_root = _write_v3_pool(tmp_path)
    validation_ids = np.asarray([b"2def"], dtype="S4")
    np.savez(
        pool_root / "validation_selection.npz",
        validation_pdb_id=validation_ids,
        center_pdb_index=np.empty(0, dtype=np.int32),
        center_occurrence_id=np.empty(0, dtype=np.int32),
        bias_pdb_index=np.zeros(1, dtype=np.int32),
        bias_occurrence_id=np.full(1, 3, dtype=np.int32),
        bias_candidate_index=np.zeros(1, dtype=np.int16),
        context_pdb_index=np.zeros(1, dtype=np.int32),
        context_candidate_index=np.zeros(1, dtype=np.int32),
    )
    requests = load_validation_selection(pool_root / "validation_selection.npz", pool_root)
    assert Counter(request.role for request in requests) == {"bias": 1, "context": 1}


def test_context_and_bias_generators_remain_deterministic() -> None:
    coords = np.full((4, 3), 40.0, dtype=np.float32)
    context_first = generate_context_starts(
        coords,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (84, 85, 86),
        np.random.default_rng(9),
        target_count=5,
        max_attempts=8,
        min_core_atoms=0,
    )
    context_second = generate_context_starts(
        coords,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (84, 85, 86),
        np.random.default_rng(9),
        target_count=5,
        max_attempts=8,
        min_core_atoms=0,
    )
    np.testing.assert_array_equal(context_first, context_second)

    sparse = np.asarray([[98, 99, 100], [99, 100, 101], [100, 101, 102]], dtype=np.int32)
    bias_first = sample_bias_starts(
        sparse,
        (200, 200, 200),
        np.random.default_rng(17),
        num_candidates=30,
        voxel_size_world=(0.5, 1.0, 2.0),
        extra_drift_max_angstrom=3.0,
    )
    bias_second = sample_bias_starts(
        sparse,
        (200, 200, 200),
        np.random.default_rng(17),
        num_candidates=30,
        voxel_size_world=(0.5, 1.0, 2.0),
        extra_drift_max_angstrom=3.0,
    )
    np.testing.assert_array_equal(bias_first, bias_second)
    assert np.all((bias_first >= 0) & (bias_first <= 120))
