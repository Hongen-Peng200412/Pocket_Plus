from __future__ import annotations

import numpy as np
import torch

from src.inference.get_pred import (
    _compute_box_slices_for_merge,
    _merge_box_probability_into_full,
    extract_voxel_head_logits,
)


def test_ligand_pred_is_zero_on_hardmask_like_final_step() -> None:
    # np.ndarray, (2,2,2), float32, 合并后的 ligand 概率图
    ligand_pred = np.ones((2, 2, 2), dtype=np.float32)
    # np.ndarray, (2,2,2), float32, 原子落点 hardmask
    hardmask = np.zeros((2, 2, 2), dtype=np.float32)
    hardmask[0, 0, 0] = 1.0

    masked = ligand_pred * (1.0 - hardmask)

    assert masked[0, 0, 0] == 0.0
    assert masked.sum() == 7.0


def test_receptor_pred_is_zero_outside_hardmask_like_final_step() -> None:
    # np.ndarray, (2,2,2), float32, 合并后的 receptor 概率图
    receptor_pred = np.ones((2, 2, 2), dtype=np.float32)
    # np.ndarray, (2,2,2), float32, 原子落点 hardmask
    hardmask = np.zeros((2, 2, 2), dtype=np.float32)
    hardmask[0, 0, 0] = 1.0

    masked = receptor_pred * hardmask

    assert masked[0, 0, 0] == 1.0
    assert masked.sum() == 1.0


def test_mean_merge_with_core_crop_covers_full_volume_edges() -> None:
    value_sum = np.zeros((6, 4, 4), dtype=np.float32)
    weight_sum = np.zeros((6, 4, 4), dtype=np.float32)
    box_prob = np.ones((4, 4, 4), dtype=np.float32)

    _merge_box_probability_into_full(
        value_sum=value_sum,
        weight_sum=weight_sum,
        box_prob=box_prob,
        box_position_zyx=(0, 0, 0),
        full_shape_zyx=(6, 4, 4),
        merge_mode="mean",
        core_offset=1,
        gaussian_sigma_ratio=0.5,
    )
    _merge_box_probability_into_full(
        value_sum=value_sum,
        weight_sum=weight_sum,
        box_prob=box_prob,
        box_position_zyx=(2, 0, 0),
        full_shape_zyx=(6, 4, 4),
        merge_mode="mean",
        core_offset=1,
        gaussian_sigma_ratio=0.5,
    )

    assert np.all(weight_sum > 0)
    assert np.allclose(value_sum / weight_sum, 1.0)


def test_compute_box_slices_keeps_global_boundaries() -> None:
    box_slices, full_slices = _compute_box_slices_for_merge(
        box_position_zyx=(0, 2, 2),
        box_shape_zyx=(4, 4, 4),
        full_shape_zyx=(6, 6, 6),
        core_offset=1,
    )

    assert box_slices[0] == slice(0, 3)
    assert full_slices[0] == slice(0, 3)
    assert box_slices[1] == slice(1, 4)
    assert full_slices[1] == slice(3, 6)


def test_extract_voxel_head_logits_optional_ligand_not_requested() -> None:
    outputs = {
        "voxel_logits_aux": torch.zeros((2, 1, 4, 4, 4), dtype=torch.float32),
        "voxel_logits_ligand": None,
    }

    result = extract_voxel_head_logits(outputs, output_heads=("receptor",))

    assert tuple(result["receptor"].shape) == (2, 1, 4, 4, 4)


def test_extract_voxel_head_logits_requested_none_head_fails() -> None:
    outputs = {
        "voxel_logits_aux": torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32),
        "voxel_logits_ligand": None,
    }

    try:
        extract_voxel_head_logits(outputs, output_heads=("ligand",))
        assert False, "请求 None ligand head 应抛出 ValueError"
    except ValueError:
        pass
