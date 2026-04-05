from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np

from src.datasets.box_point_dataset import BoxPointDataset
from src.datasets.box_sample_builder import build_box_point_numpy_sample, to_torch_sample


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _build_dataset(
    tmp_path: Path,
    *,
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
    label_grid = np.zeros((1, 4, 4, 4), dtype=np.float32)
    label_grid[0, 1, 1, 1] = 2.0

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
            [0.5, 1.5, 1.5],  # core, cropped out by margin
            [4.5, 1.5, 1.5],  # buffer only
            [6.0, 1.5, 1.5],  # outside buffer
        ],
        dtype=np.float32,
    )
    atom_features = np.arange(12, dtype=np.float32).reshape(4, 3)
    binding_mask = np.array([True, False, True, True], dtype=bool)
    pocket_class_ids = np.array([0, 1, 2, 3], dtype=np.int64)
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
        mode="val",
        data_folder_names=["emdb_BOX", "pdb_feature_BOX", "pdb_label_BOX"],
        class_folder_names=[class_name],
        atom_buffer_radius=atom_buffer_radius,
        valid_crop_margin=valid_crop_margin,
        enable_random_rotation=False,
        class_mapping=[0, 1, 1, 1],
    )


def _torch_sample_to_numpy(sample: dict) -> dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = value
    return out


def _build_builder_sample(dataset: BoxPointDataset) -> dict:
    sample_meta = dataset.total_sample[0]
    sample_name = sample_meta["sample_name"]
    class_name = sample_meta["class_name"]
    parsed_name = dataset._parse_sample_name(sample_name)
    box_data = dataset._load_box_npz_triplet(class_name, sample_name)
    structure_data = dataset._load_structure_npz_cached(parsed_name["pdb_id"])

    sample = build_box_point_numpy_sample(
        voxel_grid=box_data["voxel_grid"],
        voxel_label=box_data["voxel_label"],
        atom_coords_world_full=structure_data["atom_coord_world"],
        atom_features_raw_full=structure_data["atom_feature_raw"],
        atom_labels_full=structure_data["pocket_class_ids"],
        box_origin_world=box_data["box_origin_world"],
        voxel_size_world=box_data["voxel_size_world"],
        box_shape_zyx=box_data["box_shape_zyx"],
        atom_buffer_radius=dataset.atom_buffer_radius,
        valid_crop_margin=dataset.valid_crop_margin,
        class_mapping=dataset.class_mapping,
    )
    sample["sample_name"] = sample_name
    sample["pdb_id"] = parsed_name["pdb_id"]
    sample["class_name"] = class_name
    sample["instance_id"] = parsed_name["instance_id"]
    sample["is_center_box"] = parsed_name["is_center_box"]
    return sample


def test_builder_output_fields_match_contract(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    sample = _build_builder_sample(dataset)

    expected_keys = {
        "voxel_grid",
        "voxel_label",
        "hardmask",
        "voxel_valid_mask",
        "box_origin_world",
        "voxel_size_world",
        "box_shape_zyx",
        "atom_coord_world",
        "atom_coord_local_voxel",
        "atom_coord_centered_world",
        "atom_feat",
        "atom_label",
        "atom_is_in_core_box",
        "atom_valid_mask",
        "_selected_idx",
        "sample_name",
        "pdb_id",
        "class_name",
        "instance_id",
        "is_center_box",
    }
    assert set(sample.keys()) == expected_keys

    contract = {
        "voxel_grid": (np.float32, 4),
        "voxel_label": (np.int64, 3),
        "hardmask": (np.int64, 3),
        "voxel_valid_mask": (np.bool_, 3),
        "box_origin_world": (np.float32, 1),
        "voxel_size_world": (np.float32, 1),
        "box_shape_zyx": (np.int64, 1),
        "atom_coord_world": (np.float32, 2),
        "atom_coord_local_voxel": (np.float32, 2),
        "atom_coord_centered_world": (np.float32, 2),
        "atom_feat": (np.float32, 2),
        "atom_label": (np.int64, 1),
        "atom_is_in_core_box": (np.bool_, 1),
        "atom_valid_mask": (np.bool_, 1),
        "_selected_idx": (np.int64, 1),
    }
    for key, (dtype, ndim) in contract.items():
        assert sample[key].dtype == dtype
        assert sample[key].ndim == ndim

    assert sample["_selected_idx"].tolist() == [0, 1, 2]
    assert sample["atom_valid_mask"].tolist() == [True, False, False]


def test_builder_matches_dataset_getitem_tensor_values(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)

    dataset_sample = _torch_sample_to_numpy(dataset[0])
    builder_sample = _build_builder_sample(dataset)

    assert set(builder_sample.keys()) == set(dataset_sample.keys())
    for key, value in builder_sample.items():
        other = dataset_sample[key]
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.floating):
                assert np.allclose(other, value)
            else:
                assert np.array_equal(other, value)
        else:
            assert other == value


def test_to_torch_sample_dtype_consistency_and_sidecar_passthrough(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    sample = _build_builder_sample(dataset)
    sample["custom_meta"] = {"source": "unit-test"}

    torch_sample = to_torch_sample(sample)

    expected_dtypes = {
        "voxel_grid": torch.float32,
        "voxel_label": torch.int64,
        "hardmask": torch.int64,
        "voxel_valid_mask": torch.bool,
        "box_origin_world": torch.float32,
        "voxel_size_world": torch.float32,
        "box_shape_zyx": torch.int64,
        "atom_coord_world": torch.float32,
        "atom_coord_local_voxel": torch.float32,
        "atom_coord_centered_world": torch.float32,
        "atom_feat": torch.float32,
        "atom_label": torch.int64,
        "atom_is_in_core_box": torch.bool,
        "atom_valid_mask": torch.bool,
    }
    for key, dtype in expected_dtypes.items():
        assert isinstance(torch_sample[key], torch.Tensor)
        assert torch_sample[key].dtype == dtype

    assert isinstance(torch_sample["_selected_idx"], np.ndarray)
    assert torch_sample["_selected_idx"].tolist() == [0, 1, 2]
    assert torch_sample["custom_meta"] == {"source": "unit-test"}


def test_builder_handles_empty_selection_and_torch_conversion() -> None:
    voxel_grid = np.zeros((2, 4, 4, 4), dtype=np.float32)
    voxel_label = np.zeros((4, 4, 4), dtype=np.int64)
    atom_coords = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    atom_feat = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    atom_label = np.array([2], dtype=np.int64)

    sample = build_box_point_numpy_sample(
        voxel_grid=voxel_grid,
        voxel_label=voxel_label,
        atom_coords_world_full=atom_coords,
        atom_features_raw_full=atom_feat,
        atom_labels_full=atom_label,
        box_origin_world=np.zeros(3, dtype=np.float32),
        voxel_size_world=np.ones(3, dtype=np.float32),
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
        atom_buffer_radius=0.5,
        valid_crop_margin=1,
        class_mapping=None,
    )

    assert sample["_selected_idx"].shape == (0,)
    assert sample["atom_coord_world"].shape == (0, 3)
    assert sample["atom_coord_local_voxel"].shape == (0, 3)
    assert sample["atom_coord_centered_world"].shape == (0, 3)
    assert sample["atom_feat"].shape == (0, 3)
    assert sample["atom_label"].shape == (0,)
    assert sample["atom_is_in_core_box"].shape == (0,)
    assert sample["atom_valid_mask"].shape == (0,)

    torch_sample = to_torch_sample(sample)
    assert tuple(torch_sample["atom_coord_world"].shape) == (0, 3)
    assert tuple(torch_sample["atom_feat"].shape) == (0, 3)
    assert tuple(torch_sample["atom_label"].shape) == (0,)
