"""Selector selection.npz 的门控、精确 MAP 与字段契约测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.component_lineage import ComponentForest, ComponentNode, ComponentTree
from src.selector.inference import load_selected_nodes_for_pdb, produce_selection_for_pdb


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """保存一个测试 NPZ；测试数据无需模拟正式原子发布。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def test_selection_schema_uses_gate_and_clg_local_candidate_indices(tmp_path: Path) -> None:
    """门控通过时取非空精确 MAP，失败时写空段且不混用 forest node ID。"""
    scores_path = tmp_path / "scores.npz"
    forest_path = tmp_path / "forest.npz"
    clg_path = tmp_path / "clg.npz"
    selection_path = tmp_path / "selection.npz"
    _save_npz(
        scores_path,
        CLG_id=np.asarray([5, 6], dtype=np.int32),
        CLG_logit=np.asarray([2.0, -2.0], dtype=np.float32),
        CLG_valid_probability=np.asarray([0.9, 0.2], dtype=np.float32),
        candidate_offsets=np.asarray([0, 3, 4], dtype=np.int64),
        predicted_max_iou=np.asarray([0.2, 0.8, 0.7, 0.9], dtype=np.float32),
        selection_logit=np.asarray([0.1, 2.0, 1.5, 5.0], dtype=np.float32),
    )
    def node(tree_id: int, node_id: int, values: list[int]) -> ComponentNode:
        """构造用于 schema/恢复测试的最小合法组件节点。"""
        return ComponentNode(
            tree_id=tree_id,
            node_id=node_id,
            threshold_grid_index=100,
            threshold_value=100 / 32768,
            voxel_global_linear_index=np.asarray(values, dtype=np.int64),
            bbox_min_zyx=np.asarray([0, 0, 0], dtype=np.int32),
            bbox_max_zyx=np.asarray([1, 1, 1], dtype=np.int32),
            centroid_zyx=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
            probability_mean=0.5,
            probability_max=0.8,
            candidate_eligible=True,
            ineligible_reason_code=0,
        )

    root = node(0, 10, [0, 1, 2])
    left = node(0, 11, [0])
    right = node(0, 12, [1])
    left.parent = root
    right.parent = root
    root.children.extend([left, right])
    other = node(1, 20, [5])
    forest_arrays = ComponentForest(
        (ComponentTree(0, (root, left, right)), ComponentTree(1, (other,)))
    ).to_arrays()
    _save_npz(forest_path, **forest_arrays)
    _save_npz(
        clg_path,
        CLG_id=np.asarray([5, 6], dtype=np.int32),
        tree_id=np.asarray([0, 1], dtype=np.int32),
        candidate_offsets=np.asarray([0, 3, 4], dtype=np.int64),
        candidate_node_id=np.asarray([10, 11, 12, 20], dtype=np.int32),
    )

    produce_selection_for_pdb(
        scores_path=scores_path,
        forest_path=forest_path,
        clg_path=clg_path,
        selection_path=selection_path,
        tau_g=0.5,
        lambda_count=0.05,
    )

    with np.load(selection_path, allow_pickle=False) as selection:
        assert set(selection.files) == {
            "CLG_id",
            "CLG_gate_pass",
            "selected_candidate_offsets",
            "selected_candidate_index",
        }
        np.testing.assert_array_equal(selection["CLG_id"], [5, 6])
        np.testing.assert_array_equal(selection["CLG_gate_pass"], [True, False])
        np.testing.assert_array_equal(selection["selected_candidate_offsets"], [0, 2, 2])
        # 11 与 12 是 sibling，可同时进入反链；保存的是本 CLG 局部下标 1/2。
        np.testing.assert_array_equal(selection["selected_candidate_index"], [1, 2])
        assert selection["selected_candidate_index"].dtype == np.int16

    selected_nodes = load_selected_nodes_for_pdb(selection_path, forest_path, clg_path)
    assert [(value.tree_id, value.node_id) for value in selected_nodes] == [(0, 11), (0, 12)]
