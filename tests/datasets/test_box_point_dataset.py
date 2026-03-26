from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np

from src.datasets.box_point_dataset import BoxPointDataset


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _build_dataset(
    tmp_path: Path,
    *,
    label_channel_first: bool = True,
    enable_random_rotation: bool = False,
    valid_crop_margin: int = 1,
    atom_buffer_radius: float = 1.0,
) -> BoxPointDataset:
    class_name = "small_molecule"
    sample_name = "abcd_7_0_0_0_C"

    split_path = tmp_path / "split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([sample_name]), encoding="utf-8")

    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    emdb_grid = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
    pdb_feature_grid = np.zeros((2, 4, 4, 4), dtype=np.float32)
    pdb_feature_grid[0, 1, 1, 1] = 5.0
    pdb_feature_grid[1, 2, 2, 2] = -3.0

    label_grid_3d = np.zeros((4, 4, 4), dtype=np.float32)
    label_grid_3d[1, 1, 1] = 1.0
    label_grid = label_grid_3d[None, ...] if label_channel_first else label_grid_3d

    all_data_path = tmp_path / "all_data"
    _write_npz(
        all_data_path / "emdb_BOX" / class_name / f"{sample_name}.npz",
        grid=emdb_grid,
        origin=origin,
        voxel_size=voxel_size,
    )
    _write_npz(
        all_data_path / "pdb_feature_BOX" / class_name / f"{sample_name}.npz",
        grid=pdb_feature_grid,
        origin=origin,
        voxel_size=voxel_size,
    )
    _write_npz(
        all_data_path / "pdb_label_BOX" / class_name / f"{sample_name}.npz",
        grid=label_grid,
        origin=origin,
        voxel_size=voxel_size,
    )

    sample_root_path = tmp_path / "structures"
    atom_coords = np.array(
        [
            [1.5, 1.5, 1.5],  # core, valid
            [0.5, 1.5, 1.5],  # core, but cropped out by margin
            [4.5, 1.5, 1.5],  # buffer only
            [6.0, 1.5, 1.5],  # outside buffer
        ],
        dtype=np.float32,
    )
    atom_features = np.arange(12, dtype=np.float32).reshape(4, 3)
    binding_mask = np.array([True, False, True, True], dtype=bool)
    pocket_class_ids = np.array([0, 1, 1, 2], dtype=np.int64)
    instance_ids = np.array([7, 7, 8, 9], dtype=np.int64)

    _write_npz(sample_root_path / "abcd" / "atoms.npz", coords=atom_coords, features=atom_features)
    _write_npz(
        sample_root_path / "abcd" / "labels.npz",
        binding_mask=binding_mask,
        pocket_class_ids=pocket_class_ids,
        instance_ids=instance_ids,
    )

    return BoxPointDataset(
        all_data_path=str(all_data_path),
        sample_root_path=str(sample_root_path),
        split_file=[str(split_path)],
        mode="train" if enable_random_rotation else "val",
        data_folder_names=["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"],
        class_folder_names=[class_name],
        atom_buffer_radius=atom_buffer_radius,
        valid_crop_margin=valid_crop_margin,
        enable_random_rotation=enable_random_rotation,
    )


def test_box_point_dataset_builds_expected_sample(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)

    sample = dataset[0]

    assert sample["sample_name"] == "abcd_7_0_0_0_C"
    assert sample["pdb_id"] == "abcd"
    assert sample["class_name"] == "small_molecule"
    assert sample["instance_id"] == 7
    assert sample["is_center_box"] is True

    assert tuple(sample["voxel_grid"].shape) == (3, 4, 4, 4)
    assert tuple(sample["voxel_label"].shape) == (4, 4, 4)
    assert tuple(sample["hardmask"].shape) == (4, 4, 4)
    assert tuple(sample["voxel_valid_mask"].shape) == (4, 4, 4)

    emdb_grid = sample["voxel_grid"][0]
    assert torch.isclose(emdb_grid.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(emdb_grid.std(unbiased=False), torch.tensor(1.0), atol=1e-6)

    assert sample["hardmask"][1, 1, 1].item() == 1
    assert sample["hardmask"][0, 0, 0].item() == 0
    assert sample["voxel_label"][1, 1, 1].item() == 1

    assert sample["atom_is_in_core_box"].tolist() == [True, True, False]
    assert sample["atom_valid_mask"].tolist() == [True, False, False]
    assert sample["atom_label"].tolist() == [0, 1, 1]

    assert torch.allclose(
        sample["atom_coord_local_voxel"][0],
        torch.tensor([1.5, 1.5, 1.5], dtype=torch.float32),
    )
    assert torch.allclose(
        sample["atom_coord_centered_world"][0],
        torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32),
    )


def test_load_box_npz_triplet_accepts_both_label_shapes(tmp_path: Path) -> None:
    dataset_4d = _build_dataset(tmp_path / "case_4d", label_channel_first=True)
    dataset_3d = _build_dataset(tmp_path / "case_3d", label_channel_first=False)

    box_data_4d = dataset_4d._load_box_npz_triplet("small_molecule", "abcd_7_0_0_0_C")
    box_data_3d = dataset_3d._load_box_npz_triplet("small_molecule", "abcd_7_0_0_0_C")

    assert tuple(box_data_4d["voxel_label"].shape) == (4, 4, 4)
    assert tuple(box_data_3d["voxel_label"].shape) == (4, 4, 4)
    assert np.array_equal(box_data_4d["voxel_label"], box_data_3d["voxel_label"])


def test_structure_cache_reuses_loaded_data(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)

    first = dataset._load_structure_npz_cached("abcd")
    second = dataset._load_structure_npz_cached("abcd")

    assert first is second
    assert list(dataset.structure_cache.keys()) == ["abcd"]


def _torch_sample_to_numpy(sample: dict) -> dict:
    """将 __getitem__ 返回的 torch sample 还原为 numpy dict, 供 _apply_synced_rotation 测试使用。"""
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy()
        else:
            out[k] = v
    return out


def test_synced_rotation_updates_atom_coordinates_and_masks(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path, valid_crop_margin=1)
    # mode='val' 不启用旋转, dataset[0] 直接返回未旋转的 torch 样本
    torch_sample = dataset[0]
    sample = _torch_sample_to_numpy(torch_sample)

    dataset._sample_rotation_params = lambda: (1, 2, 1)  # type: ignore[method-assign]
    rotated = dataset._apply_synced_rotation(sample)

    assert rotated["box_shape_zyx"].tolist() == [4, 4, 4]
    assert rotated["atom_is_in_core_box"].tolist() == [True, True, False]
    assert rotated["atom_valid_mask"].tolist() == [True, False, False]

    assert np.allclose(rotated["atom_coord_local_voxel"][0], np.array([1.5, 2.5, 1.5], dtype=np.float32))
    assert np.allclose(rotated["atom_coord_local_voxel"][1], np.array([1.5, 3.5, 1.5], dtype=np.float32))

    expected_center = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    assert np.allclose(
        rotated["atom_coord_world"],
        rotated["atom_coord_centered_world"] + expected_center[None, :],
    )


def test_atom_label_with_class_mapping(tmp_path: Path) -> None:
    """class_mapping 同时作用于 voxel 和 atom 标签。"""
    class_name = "small_molecule"
    sample_name = "abcd_7_0_0_0_C"

    split_path = tmp_path / "split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([sample_name]), encoding="utf-8")

    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    emdb_grid = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
    pdb_feature_grid = np.zeros((2, 4, 4, 4), dtype=np.float32)
    pdb_feature_grid[0, 1, 1, 1] = 5.0

    # voxel label: class 2 在 (1,1,1)、其余为 0
    label_grid = np.zeros((1, 4, 4, 4), dtype=np.float32)
    label_grid[0, 1, 1, 1] = 2.0

    all_data_path = tmp_path / "all_data"
    _write_npz(all_data_path / "emdb_BOX" / class_name / f"{sample_name}.npz", grid=emdb_grid, origin=origin, voxel_size=voxel_size)
    _write_npz(all_data_path / "pdb_feature_BOX" / class_name / f"{sample_name}.npz", grid=pdb_feature_grid, origin=origin, voxel_size=voxel_size)
    _write_npz(all_data_path / "pdb_label_BOX" / class_name / f"{sample_name}.npz", grid=label_grid, origin=origin, voxel_size=voxel_size)

    sample_root_path = tmp_path / "structures"
    atom_coords = np.array([[1.5, 1.5, 1.5], [0.5, 1.5, 1.5]], dtype=np.float32)
    atom_features = np.arange(6, dtype=np.float32).reshape(2, 3)
    binding_mask = np.array([True, False], dtype=bool)
    # pocket_class_ids: 原子 0 是 class 2, 原子 1 是 class 0
    pocket_class_ids = np.array([2, 0], dtype=np.int64)
    instance_ids = np.array([7, 7], dtype=np.int64)

    _write_npz(sample_root_path / "abcd" / "atoms.npz", coords=atom_coords, features=atom_features)
    _write_npz(sample_root_path / "abcd" / "labels.npz", binding_mask=binding_mask, pocket_class_ids=pocket_class_ids, instance_ids=instance_ids)

    # class_mapping: [0, 1, 1] —— 把 class 2 合并到 class 1
    dataset = BoxPointDataset(
        all_data_path=str(all_data_path),
        sample_root_path=str(sample_root_path),
        split_file=[str(split_path)],
        mode="val",
        data_folder_names=["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"],
        class_folder_names=[class_name],
        class_mapping=[0, 1, 1],
        atom_buffer_radius=0.0,
        valid_crop_margin=0,
        enable_random_rotation=False,
    )

    sample = dataset[0]

    # atom: pocket_class_ids=[2, 0] → mapping → [1, 0]
    assert sample["atom_label"].tolist() == [1, 0]
    # voxel: label=2 在 (1,1,1) → mapping → 1
    assert sample["voxel_label"][1, 1, 1].item() == 1
    assert sample["voxel_label"][0, 0, 0].item() == 0
