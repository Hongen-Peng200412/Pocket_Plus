from __future__ import annotations

from pathlib import Path

import numpy as np

from src.inference.voxel_gt import load_ligand_gt_from_labels_npz



def test_labels_npz_maps_ligand_coords_distance_to_voxels(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        ligand_candidate_ids=np.array([3], dtype=np.int32),
        ligand_class_ids=np.array([1], dtype=np.int32),
        ligand_coords_3=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )

    result = load_ligand_gt_from_labels_npz(
        labels_npz_path=str(labels_path),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        grid_shape_zyx=(2, 2, 2),
        class_mapping=[0, 1],
        ligand_gt_distance_threshold=0.6,
    )

    assert result["gt_ligand_mask"].sum() == 1
    assert result["gt_instance_label"][0, 0, 0] == 1
    assert result["gt_instance_label"][0, 0, 1] == 0



def test_labels_npz_builds_gt_by_mapped_class(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        ligand_candidate_ids=np.array([3, 4], dtype=np.int32),
        ligand_class_ids=np.array([1, 4], dtype=np.int32),
        ligand_coords_3=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        ligand_coords_4=np.array([[1.5, 1.5, 1.5]], dtype=np.float32),
    )

    result = load_ligand_gt_from_labels_npz(
        labels_npz_path=str(labels_path),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        grid_shape_zyx=(3, 3, 3),
        class_mapping=[0, 1, 0, 0, 2],
        ligand_gt_distance_threshold=0.6,
    )

    assert set(result["gt_ligand_mask_by_class_id"].keys()) == {1, 2}
    assert result["gt_ligand_mask_by_class_id"][1][0, 0, 0]
    assert result["gt_ligand_mask_by_class_id"][2][1, 1, 1]
    assert result["gt_ligand_mask"].sum() == 2
    assert [item["class_id"] for item in result["gt_instance_meta"]] == [1, 2]



def test_labels_npz_missing_ligand_coords_fails(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        ligand_candidate_ids=np.array([3], dtype=np.int32),
        ligand_class_ids=np.array([1], dtype=np.int32),
    )

    try:
        load_ligand_gt_from_labels_npz(
            labels_npz_path=str(labels_path),
            origin=np.zeros(3, dtype=np.float32),
            voxel_size=np.ones(3, dtype=np.float32),
            grid_shape_zyx=(2, 2, 2),
            class_mapping=[0, 1],
            ligand_gt_distance_threshold=0.6,
        )
        assert False, "缺少 ligand_coords_{id} 应抛出 KeyError"
    except KeyError:
        pass



def test_labels_npz_ligand_coords_outside_grid_produces_empty_instance(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        ligand_candidate_ids=np.array([3], dtype=np.int32),
        ligand_class_ids=np.array([1], dtype=np.int32),
        ligand_coords_3=np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
    )

    result = load_ligand_gt_from_labels_npz(
        labels_npz_path=str(labels_path),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        grid_shape_zyx=(2, 2, 2),
        class_mapping=[0, 1],
        ligand_gt_distance_threshold=0.6,
    )

    assert result["gt_ligand_mask"].sum() == 0
    assert result["gt_instance_meta"][0]["voxel_count"] == 0



def test_labels_npz_class_mapping_zero_skips_ligand(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        ligand_candidate_ids=np.array([3], dtype=np.int32),
        ligand_class_ids=np.array([1], dtype=np.int32),
        ligand_coords_3=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )

    result = load_ligand_gt_from_labels_npz(
        labels_npz_path=str(labels_path),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        grid_shape_zyx=(2, 2, 2),
        class_mapping=[0, 0],
        ligand_gt_distance_threshold=0.6,
    )

    assert result["gt_ligand_mask"].sum() == 0
    assert result["gt_instance_meta"] == []
