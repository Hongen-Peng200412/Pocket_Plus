from __future__ import annotations

import torch
import numpy as np

from src.inference.parse_input import prepare_batched_boxes, split_volume_to_boxes


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


def test_split_volume_to_boxes_builds_geometric_hardmask() -> None:
    grid = np.zeros((1, 4, 4, 4), dtype=np.float32)
    atom_coords_world = np.array(
        [
            [0.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
            [4.5, 1.5, 1.5],
        ],
        dtype=np.float32,
    )
    atom_feat = np.zeros((3, 49), dtype=np.float32)

    box_dicts = split_volume_to_boxes(
        grid=grid,
        atom_coords_world=atom_coords_world,
        atom_feat=atom_feat,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        window_size=4,
        stride=4,
        atom_buffer_radius=1.0,
        valid_crop_margin=0,
        emdb_channels=1,
    )

    assert len(box_dicts) == 1
    hardmask = box_dicts[0]["hardmask"].numpy()
    assert hardmask[1, 1, 0] == 1
    assert hardmask[1, 1, 1] == 1
    assert hardmask.sum() == 2


def test_split_volume_to_boxes_prepare_batch_preserves_sidecar_and_indices() -> None:
    grid = np.zeros((1, 6, 4, 4), dtype=np.float32)
    atom_coords_world = np.array(
        [
            [1.5, 1.5, 0.5],
            [1.5, 1.5, 2.5],
            [1.5, 1.5, 4.5],
        ],
        dtype=np.float32,
    )
    atom_feat = np.arange(12, dtype=np.float32).reshape(3, 4)

    box_dicts = split_volume_to_boxes(
        grid=grid,
        atom_coords_world=atom_coords_world,
        atom_feat=atom_feat,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        window_size=4,
        stride=2,
        atom_buffer_radius=0.0,
        valid_crop_margin=0,
        emdb_channels=1,
    )

    assert len(box_dicts) == 2
    original_global_indices = [box["global_atom_indices"].copy() for box in box_dicts]
    original_positions = [box["box_position_zyx"] for box in box_dicts]

    batches = prepare_batched_boxes(box_dicts, batch_size=2, device="cpu")

    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch["atom_global_indices"], torch.Tensor)
    assert batch["atom_global_indices"].dtype == torch.long
    assert batch["atom_global_indices"].tolist() == np.concatenate(original_global_indices).tolist()
    assert [meta["box_position_zyx"] for meta in batch["_box_meta"]] == original_positions

    # prepare_batched_boxes 内部会 pop sidecar，再恢复回原 box_dicts；这里验证不会破坏输入列表
    assert [box["box_position_zyx"] for box in box_dicts] == original_positions
    for box, original_idx in zip(box_dicts, original_global_indices):
        assert np.array_equal(box["global_atom_indices"], original_idx)


def test_split_volume_to_boxes_sample_passthrough_collate_with_empty_atoms() -> None:
    grid = np.zeros((1, 5, 5, 5), dtype=np.float32)
    box_dicts = split_volume_to_boxes(
        grid=grid,
        atom_coords_world=np.zeros((0, 3), dtype=np.float32),
        atom_feat=np.zeros((0, 49), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        window_size=4,
        stride=3,
        atom_buffer_radius=0.0,
        valid_crop_margin=0,
        emdb_channels=1,
    )

    batches = prepare_batched_boxes(box_dicts, batch_size=2, device="cpu")

    assert len(batches) == 4
    for batch in batches:
        assert batch["atom_counts"].tolist() == [0, 0]
        assert batch["atom_offsets"].tolist() == [0, 0]
        assert tuple(batch["atom_coord_world"].shape) == (0, 3)
        assert tuple(batch["atom_feat"].shape) == (0, 49)
        assert batch["atom_global_indices"].tolist() == []
