from __future__ import annotations

import numpy as np

from src.inference.parse_input import split_volume_to_boxes


def test_split_volume_to_boxes_covers_volume_tail() -> None:
    grid = np.zeros((1, 95, 95, 95), dtype=np.float32)
    box_dicts = split_volume_to_boxes(
        grid=grid,
        atom_coords_world=np.zeros((0, 3), dtype=np.float32),
        atom_feat=np.zeros((0, 49), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        window_size=48,
        stride=32,
        atom_buffer_radius=0.0,
        valid_crop_margin=0,
        emdb_channels=1,
    )

    starts = sorted({box_dict["box_position_zyx"][0] for box_dict in box_dicts})
    assert starts == [0, 32, 47]
    assert max(start + 48 for start in starts) == 95
