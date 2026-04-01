from __future__ import annotations

import numpy as np

from src.datasets.box_geometry import (
    build_hardmask_from_atom_coordinates,
    build_hardmask_from_world_coordinates,
)


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
