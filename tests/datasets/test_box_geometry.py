from __future__ import annotations

import numpy as np

from src.datasets.box_geometry import (
    build_atom_coordinates,
    build_atom_features,
    build_atom_valid_mask,
    build_hardmask_from_atom_coordinates,
    build_hardmask_from_world_coordinates,
    build_voxel_valid_mask,
    select_atoms_for_box,
)


def test_select_atoms_for_box_returns_core_and_buffer_atoms() -> None:
    atom_coords_world = np.array(
        [
            [1.5, 1.5, 1.5],  # core, valid
            [0.5, 1.5, 1.5],  # core, but later cropped by margin
            [4.5, 1.5, 1.5],  # buffer only
            [6.0, 1.5, 1.5],  # outside buffer
        ],
        dtype=np.float32,
    )

    selected = select_atoms_for_box(
        atom_coords_world=atom_coords_world,
        box_origin_world=np.zeros(3, dtype=np.float32),
        voxel_size_world=np.ones(3, dtype=np.float32),
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
        buffer_radius=1.0,
    )

    assert selected["selected_idx"].tolist() == [0, 1, 2]
    assert selected["atom_is_in_core_box"].tolist() == [True, True, False]


def test_build_atom_coordinates_returns_all_coordinate_views() -> None:
    atom_coords_world = np.array(
        [
            [1.5, 1.5, 1.5],
            [0.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
        ],
        dtype=np.float32,
    )

    coord_data = build_atom_coordinates(
        atom_coords_world=atom_coords_world,
        selected_idx=np.array([0, 2], dtype=np.int64),
        box_origin_world=np.zeros(3, dtype=np.float32),
        voxel_size_world=np.ones(3, dtype=np.float32),
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
    )

    assert np.allclose(
        coord_data["atom_coord_world"],
        np.array([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5]], dtype=np.float32),
    )
    assert np.allclose(
        coord_data["atom_coord_local_voxel"],
        np.array([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5]], dtype=np.float32),
    )
    assert np.allclose(
        coord_data["atom_coord_centered_world"],
        np.array([[-0.5, -0.5, -0.5], [2.5, -0.5, -0.5]], dtype=np.float32),
    )


def test_build_atom_features_slices_selected_atoms() -> None:
    atom_features_raw = np.arange(12, dtype=np.float64).reshape(4, 3)

    atom_feat = build_atom_features(
        atom_features_raw=atom_features_raw,
        selected_idx=np.array([1, 3], dtype=np.int64),
    )

    assert atom_feat.dtype == np.float32
    assert np.array_equal(
        atom_feat,
        np.array([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]], dtype=np.float32),
    )


def test_build_atom_valid_mask_matches_dataset_margin_behavior() -> None:
    atom_valid_mask = build_atom_valid_mask(
        atom_coord_local_voxel=np.array(
            [
                [1.5, 1.5, 1.5],  # core, valid
                [0.5, 1.5, 1.5],  # core, cropped out
                [4.5, 1.5, 1.5],  # buffer only
            ],
            dtype=np.float32,
        ),
        atom_is_in_core_box=np.array([True, True, False], dtype=bool),
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
        valid_crop_margin=1.0,
    )

    assert atom_valid_mask.tolist() == [True, False, False]


def test_build_voxel_valid_mask_crops_all_faces() -> None:
    voxel_valid_mask = build_voxel_valid_mask(
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
        valid_crop_margin=1,
    )

    assert voxel_valid_mask.shape == (4, 4, 4)
    assert voxel_valid_mask.sum() == 8
    assert voxel_valid_mask[1, 1, 1]
    assert not voxel_valid_mask[0, 1, 1]
    assert not voxel_valid_mask[3, 2, 2]


def test_build_hardmask_from_atom_coordinates_marks_home_voxels() -> None:
    atom_coord_local_voxel = np.array(
        [
            [1.5, 1.5, 1.5],
            [0.5, 1.5, 1.5],
        ],
        dtype=np.float32,
    )
    atom_is_in_core_box = np.array([True, True], dtype=bool)
    box_shape_zyx = np.array([4, 4, 4], dtype=np.int64)

    hardmask = build_hardmask_from_atom_coordinates(
        atom_coord_local_voxel=atom_coord_local_voxel,
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=box_shape_zyx,
    )

    assert hardmask[1, 1, 0] == 1
    assert hardmask[1, 1, 1] == 1
    assert hardmask[0, 0, 0] == 0


def test_build_hardmask_from_atom_coordinates_ignores_buffer_atoms() -> None:
    atom_coord_local_voxel = np.array(
        [
            [1.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
        ],
        dtype=np.float32,
    )
    atom_is_in_core_box = np.array([True, False], dtype=bool)
    box_shape_zyx = np.array([4, 4, 4], dtype=np.int64)

    hardmask = build_hardmask_from_atom_coordinates(
        atom_coord_local_voxel=atom_coord_local_voxel,
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=box_shape_zyx,
    )

    assert hardmask.sum() == 1
    assert hardmask[1, 1, 1] == 1


def test_build_hardmask_from_world_coordinates_handles_empty_atoms() -> None:
    hardmask = build_hardmask_from_world_coordinates(
        atom_coords_world=np.empty((0, 3), dtype=np.float32),
        box_origin_world=np.zeros(3, dtype=np.float32),
        voxel_size_world=np.ones(3, dtype=np.float32),
        box_shape_zyx=np.array([4, 4, 4], dtype=np.int64),
    )

    assert hardmask.shape == (4, 4, 4)
    assert hardmask.sum() == 0
