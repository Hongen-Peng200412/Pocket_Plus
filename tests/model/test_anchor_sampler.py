from __future__ import annotations

import torch

from src.model.sparse_refine.anchor_sampler import SparseAnchorSampler, build_anchor_coordinates


def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """
    构造 anchor sampler 测试用 batch. 

    输入参数:
        - batch_size: int, BOX 数量

    输出:
        - batch: dict[str, torch.Tensor], 包含坐标转换所需字段
    """
    return {
        "box_origin_world": torch.zeros(batch_size, 3),
        "voxel_size_world": torch.ones(batch_size, 3),
        "box_shape_zyx": torch.tensor([[4, 4, 4] for _ in range(batch_size)], dtype=torch.long),
    }


def _make_candidate_outputs() -> dict[str, torch.Tensor]:
    """
    构造已按物理 voxel 唯一化且带路由类别的 C 输出. 

    输出:
        - candidate_outputs: dict[str, torch.Tensor], SparseCandidateSetBuilder 风格输出
    """
    return {
        "candidate_voxel_zyx": torch.tensor(
            [
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                [3, 3, 3],
            ],
            dtype=torch.long,
        ),
        "candidate_batch_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "candidate_class": torch.tensor([2, 1, 2, 1], dtype=torch.long),
        "candidate_prob": torch.tensor([0.9, 0.8, 0.7, 0.6]),
    }


def test_anchor_sampler_rejects_max_anchors_length_mismatch() -> None:
    """
    验证 candidate_class_ids 与 max_anchors_per_class 长度不一致时 fail-fast. 
    """
    try:
        SparseAnchorSampler(
            candidate_class_ids=[1, 2],
            max_anchors_per_class=[1],
            mode="topk_nms",
            weight_power=1.0,
            chunk_size=8192,
            nms_radius_voxel=2,
            random_start=False,
        )
    except ValueError as exc:
        assert "长度" in str(exc)
    else:
        raise AssertionError("SparseAnchorSampler should reject length mismatch")


def test_anchor_sampler_inherits_unique_candidate_route_class() -> None:
    """
    验证 sampler 直接消费唯一 C, 并令 P 继承 builder 已确定的路由类别. 
    """
    sampler = SparseAnchorSampler(
        candidate_class_ids=[1, 2],
        max_anchors_per_class=[10, 10],
        mode="topk_nms",
        weight_power=1.0,
        chunk_size=8192,
        nms_radius_voxel=0,
        random_start=False,
    )

    outputs = sampler(_make_candidate_outputs(), _make_batch())

    assert outputs["anchor_source_candidate_index"].tolist() == [1, 0, 3, 2]
    assert outputs["anchor_class"].tolist() == [1, 2, 1, 2]
    assert outputs["anchor_counts"].tolist() == [2, 2]
    assert outputs["anchor_counts_by_class"].tolist() == [[1, 1], [1, 1]]


def test_anchor_coordinates_use_xyz_center_corner_semantics() -> None:
    """
    验证 P 坐标使用 x/y/z 顺序和 voxel center corner 语义. 
    """
    coords = build_anchor_coordinates(
        anchor_voxel_zyx=torch.tensor([[2, 1, 0]], dtype=torch.long),
        anchor_batch_index=torch.tensor([0], dtype=torch.long),
        box_origin_world=torch.tensor([[10.0, 20.0, 30.0]]),
        voxel_size_world=torch.tensor([[2.0, 3.0, 4.0]]),
        box_shape_zyx=torch.tensor([[4, 6, 8]], dtype=torch.long),
    )

    torch.testing.assert_close(coords["anchor_coord_local_voxel"], torch.tensor([[0.5, 1.5, 2.5]]))
    torch.testing.assert_close(coords["anchor_coord_world"], torch.tensor([[11.0, 24.5, 40.0]]))
    torch.testing.assert_close(coords["anchor_coord_centered_world"], torch.tensor([[-7.0, -4.5, 2.0]]))


def test_weighted_fps_respects_per_class_cap_and_first_highest_prob() -> None:
    """
    验证 weighted_fps 遵守 per-class cap 且第一枚 anchor 是最高概率候选. 
    """
    candidate_outputs = {
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0], [0, 0, 3], [3, 3, 3]], dtype=torch.long),
        "candidate_batch_index": torch.tensor([0, 0, 0], dtype=torch.long),
        "candidate_class": torch.tensor([1, 1, 1], dtype=torch.long),
        "candidate_prob": torch.tensor([0.1, 0.9, 0.2]),
    }
    sampler = SparseAnchorSampler(
        candidate_class_ids=[1],
        max_anchors_per_class=[2],
        mode="weighted_fps",
        weight_power=1.0,
        chunk_size=2,
        nms_radius_voxel=0,
        random_start=False,
    )

    outputs = sampler(candidate_outputs, _make_batch(batch_size=1))

    assert outputs["anchor_source_candidate_index"].shape[0] == 2
    assert int(outputs["anchor_source_candidate_index"][0].item()) == 1
    assert outputs["anchor_counts_by_class"].tolist() == [[2]]


def test_topk_nms_selects_local_maxima() -> None:
    """
    验证 topk_nms 只保留局部最大候选. 
    """
    candidate_outputs = {
        "candidate_voxel_zyx": torch.tensor([[1, 1, 1], [1, 1, 2], [3, 3, 3]], dtype=torch.long),
        "candidate_batch_index": torch.tensor([0, 0, 0], dtype=torch.long),
        "candidate_class": torch.tensor([1, 1, 1], dtype=torch.long),
        "candidate_prob": torch.tensor([0.9, 0.4, 0.8]),
    }
    sampler = SparseAnchorSampler(
        candidate_class_ids=[1],
        max_anchors_per_class=[10],
        mode="topk_nms",
        weight_power=1.0,
        chunk_size=8192,
        nms_radius_voxel=1,
        random_start=False,
    )

    outputs = sampler(candidate_outputs, _make_batch(batch_size=1))

    assert outputs["anchor_source_candidate_index"].tolist() == [0, 2]


def test_binary_anchor_sampler_config_shape() -> None:
    """
    验证二分类 candidate_class_ids=[1] 时 counts_by_class 维度为 (B,1). 
    """
    sampler = SparseAnchorSampler(
        candidate_class_ids=[1],
        max_anchors_per_class=[1024],
        mode="topk_nms",
        weight_power=1.0,
        chunk_size=8192,
        nms_radius_voxel=0,
        random_start=False,
    )
    candidate_outputs = {
        "candidate_voxel_zyx": torch.tensor([[0, 0, 0]], dtype=torch.long),
        "candidate_batch_index": torch.tensor([0], dtype=torch.long),
        "candidate_class": torch.tensor([1], dtype=torch.long),
        "candidate_prob": torch.tensor([0.9]),
    }

    outputs = sampler(candidate_outputs, _make_batch(batch_size=2))

    assert tuple(outputs["anchor_counts_by_class"].shape) == (2, 1)
    assert outputs["anchor_counts"].tolist() == [1, 0]
