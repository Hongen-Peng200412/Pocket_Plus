from __future__ import annotations

import pytest
import torch

from src.model.sparse_refine.candidate_set import SparseCandidateSetBuilder


def _make_builder(
    candidate_class_ids: list[int],
    warmup_topc_per_class: list[int],
    adaptive_expand_factor: list[float],
    max_candidate_voxels_per_class: list[int],
    min_candidate_voxels_per_class: list[int],
    selection_mode: str,
) -> SparseCandidateSetBuilder:
    """
    构造当前 SparseCandidateSetBuilder。

    输入参数:
        - candidate_class_ids: list[int], 候选前景类别 ID
        - warmup_topc_per_class: list[int], warmup fixed top-C
        - adaptive_expand_factor: list[float], adaptive 扩张倍数
        - max_candidate_voxels_per_class: list[int], 每类候选上限
        - min_candidate_voxels_per_class: list[int], adaptive 每类候选下限
        - selection_mode: str, 候选选择模式

    输出:
        - builder: SparseCandidateSetBuilder, 测试实例
    """
    return SparseCandidateSetBuilder(
        candidate_class_ids=candidate_class_ids,
        warmup_topc_per_class=warmup_topc_per_class,
        adaptive_expand_factor=adaptive_expand_factor,
        max_candidate_voxels_per_class=max_candidate_voxels_per_class,
        min_candidate_voxels_per_class=min_candidate_voxels_per_class,
        selection_mode=selection_mode,
    )


def test_binary_sigmoid_candidate_generation_uses_all_voxels() -> None:
    """
    验证单通道 sigmoid 路径直接在全 BOX 体素上生成 C。
    """
    builder = _make_builder([1], [2], [2.0], [10], [0], "adaptive_threshold")
    logits = torch.tensor([[[[[0.0, 2.0], [3.0, -1.0]]]]], requires_grad=True)

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=None,
        p_sampling_by_class=None,
        use_fixed_warmup=True,
    )

    assert output["candidate_logits"].shape == (2, 1)
    assert output["candidate_logits"].requires_grad is False
    assert output["candidate_counts"].tolist() == [2]
    assert output["candidate_class"].tolist() == [1, 1]
    assert [tuple(row) for row in output["candidate_voxel_zyx"].tolist()] == [(0, 1, 0), (0, 0, 1)]
    assert "candidate_coord_world" not in output
    assert "candidate_coord_local_voxel" not in output


def test_binary_rejects_non_one_class_id() -> None:
    """
    验证单通道 sigmoid 路径拒绝非 1 的前景类别 ID。
    """
    builder = _make_builder([2], [1], [2.0], [10], [0], "adaptive_threshold")
    logits = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(ValueError, match="单通道"):
        builder(
            voxel_logits_ligand=logits,
            p_best_by_class=None,
            p_sampling_by_class=None,
            use_fixed_warmup=True,
        )


def test_multiclass_softmax_candidate_generation_routes_same_voxel_once() -> None:
    """
    验证多通道 softmax 路径将多类提名冲突 voxel 合并为一条随机路由记录。
    """
    builder = _make_builder([1, 2], [1, 1], [2.0, 2.0], [10, 10], [0, 0], "adaptive_threshold")
    logits = torch.zeros(1, 3, 1, 1, 2)
    logits[:, 1, 0, 0, 0] = 5.0
    logits[:, 2, 0, 0, 0] = 5.0

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=None,
        p_sampling_by_class=None,
        use_fixed_warmup=True,
    )

    assert output["candidate_voxel_zyx"].tolist() == [[0, 0, 0]]
    assert output["candidate_class"].tolist()[0] in {1, 2}
    assert output["candidate_counts"].tolist() == [1]
    assert int(output["candidate_counts_by_class"].sum().item()) == 1
    assert output["candidate_target_counts_by_class"].tolist() == [[1, 1]]


def test_adaptive_threshold_applies_min_candidate_floor() -> None:
    """
    验证 adaptive_threshold 先按 p_best 估计规模，再用 min_candidate_voxels_per_class 抬下限。
    """
    builder = _make_builder([1], [1], [2.0], [4], [3], "adaptive_threshold")
    logits = torch.logit(torch.tensor([[[[[0.1, 0.2, 0.3, 0.4, 0.5]]]]]), eps=1e-6)

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=torch.tensor([0.99]),
        p_sampling_by_class=None,
        use_fixed_warmup=False,
    )

    assert output["candidate_counts"].tolist() == [3]
    assert output["candidate_target_counts_by_class"].tolist() == [[0]]
    assert [tuple(row) for row in output["candidate_voxel_zyx"].tolist()] == [(0, 0, 4), (0, 0, 3), (0, 0, 2)]


def test_adaptive_threshold_requires_finite_p_best() -> None:
    """
    验证 adaptive_threshold 非 warmup 阶段要求完整有限的 p_best。
    """
    builder = _make_builder([1], [1], [2.0], [10], [0], "adaptive_threshold")
    logits = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(RuntimeError, match="p_best_by_class"):
        builder(
            voxel_logits_ligand=logits,
            p_best_by_class=torch.tensor([float("nan")]),
            p_sampling_by_class=None,
            use_fixed_warmup=False,
        )


def test_recorded_threshold_uses_cached_p_sampling() -> None:
    """
    验证 recorded_threshold 使用全局 p_sampling 阈值筛选候选。
    """
    builder = _make_builder([1], [3], [2.0], [10], [0], "recorded_threshold")
    logits = torch.logit(torch.tensor([[[[[0.2, 0.8, 0.9]]]]]), eps=1e-6)

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=None,
        p_sampling_by_class=torch.tensor([0.75]),
        use_fixed_warmup=False,
    )

    assert output["candidate_counts"].tolist() == [2]
    torch.testing.assert_close(output["candidate_p_sampling_by_class"], torch.tensor([[0.75]]))


def test_topk_mode_uses_max_candidate_count_without_threshold_cache() -> None:
    """
    验证正式 topk 模式使用 max_candidate_voxels_per_class，且不读取阈值缓存。
    """
    builder = _make_builder([1], [1], [2.0], [2], [0], "topk")
    logits = torch.logit(torch.tensor([[[[[0.2, 0.8, 0.9]]]]]), eps=1e-6)

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=None,
        p_sampling_by_class=None,
        use_fixed_warmup=False,
    )

    assert output["candidate_counts"].tolist() == [2]
    assert [tuple(row) for row in output["candidate_voxel_zyx"].tolist()] == [(0, 0, 2), (0, 0, 1)]
    assert output["candidate_target_counts_by_class"].tolist() == [[2]]
    torch.testing.assert_close(output["candidate_p_sampling_by_class"], torch.tensor([[0.8]]))


def test_topk_mode_zero_count_keeps_empty_sampling_boundary_nan() -> None:
    """
    验证正式 topk 模式请求 0 个候选时保留空 C 与 nan 截断概率。
    """
    builder = _make_builder([1], [1], [2.0], [0], [0], "topk")
    logits = torch.zeros(1, 1, 1, 1, 2)

    output = builder(
        voxel_logits_ligand=logits,
        p_best_by_class=None,
        p_sampling_by_class=None,
        use_fixed_warmup=False,
    )

    assert output["candidate_counts"].tolist() == [0]
    assert output["candidate_target_counts_by_class"].tolist() == [[0]]
    assert bool(torch.isnan(output["candidate_p_sampling_by_class"]).all())
