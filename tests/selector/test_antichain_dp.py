"""精确反链 partition、MAP 与 online oracle 的小树穷举测试. """

from __future__ import annotations

import itertools

import numpy as np
import torch

from src.selector.structured.antichain_dp import (
    build_candidate_tree_closure,
    exact_antichain_log_partition,
    exact_antichain_map,
)
from src.selector.structured.oracle import build_online_oracle, compute_candidate_max_iou


def _valid_antichains() -> list[tuple[int, ...]]:
    """返回测试树 0→{1,2} 的全部非空反链. """
    values: list[tuple[int, ...]] = []
    for bits in itertools.product((0, 1), repeat=3):
        selected = tuple(index for index, flag in enumerate(bits) if flag)
        if selected and not (0 in selected and (1 in selected or 2 in selected)):
            values.append(selected)
    return values


def test_log_partition_map_and_gradient_match_bruteforce() -> None:
    """logsumexp/max DP 应与全部非空反链的直接枚举逐值一致. """
    logits = torch.tensor([0.2, 0.8, -0.4], dtype=torch.float64, requires_grad=True)
    lambda_count = 0.05
    parent_index = (-1, 0, 0)
    candidate_by_node = (0, 1, 2)
    antichains = _valid_antichains()
    energies = torch.stack(
        [logits[list(selected)].sum() - lambda_count * len(selected) for selected in antichains]
    )
    expected_partition = torch.logsumexp(energies, dim=0)
    actual_partition = exact_antichain_log_partition(
        logits, parent_index, candidate_by_node, lambda_count
    )
    torch.testing.assert_close(actual_partition, expected_partition)

    expected_gradient = torch.autograd.grad(expected_partition, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual_partition, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)

    best_score, selected = exact_antichain_map(logits.detach(), parent_index, candidate_by_node, lambda_count)
    expected_index = int(torch.argmax(energies.detach()).item())
    torch.testing.assert_close(best_score, energies.detach()[expected_index])
    assert tuple(selected.tolist()) == antichains[expected_index]


def test_minimal_closure_keeps_only_paths_to_candidate_lca() -> None:
    """闭包应保留 candidate 间路径中的非 candidate 传递节点, 不带入 LCA 以上祖先. """
    closure = build_candidate_tree_closure(
        node_id=(0, 1, 2, 3, 4, 5),
        parent_node_id=(-1, 0, 0, 1, 1, 2),
        candidate_node_id=(3, 4),
    )
    assert closure.node_id == (1, 3, 4)
    assert closure.parent_index == (-1, 0, 0)
    assert closure.candidate_index_by_node == (-1, 0, 1)


def test_overlap_uses_occurrence_table_index_and_oracle_can_choose_empty() -> None:
    """overlap index 应直接索引 GT 计数; 全部 q 不足计数惩罚时 oracle 选择空集. """
    q = compute_candidate_max_iou(
        candidate_occurrence_offsets=(0, 2, 2),
        overlap_occurrence_index=(1, 0),
        intersection_voxel_count=(5, 2),
        candidate_voxel_count=(10, 4),
        occurrence_voxel_count=(8, 10),
    )
    np.testing.assert_allclose(q, np.asarray([1.0 / 3.0, 0.0], dtype=np.float32))
    oracle = build_online_oracle(
        candidate_max_iou=torch.tensor([0.01, 0.02]),
        parent_index=(-1, 0),
        candidate_index_by_node=(0, 1),
        lambda_count=0.05,
    )
    assert not bool(oracle.is_valid.item())
    assert oracle.selected_candidate_index.numel() == 0
    assert float(oracle.best_score) == 0.0
