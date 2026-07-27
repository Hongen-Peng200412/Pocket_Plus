"""组件森林数值往返、树关系和 WorkingTree D(g) 测试. """

from __future__ import annotations

import numpy as np
import pytest

import src.component_lineage.forest as forest_module
from src.component_lineage.forest import build_component_forest
from src.component_lineage.structures import ComponentForest, WorkingTree


def _merge_forest() -> ComponentForest:
    probability = np.zeros((5, 5, 5), dtype=np.float32)
    probability[2, 2, 1] = 0.9
    probability[2, 2, 2] = 0.6
    probability[2, 2, 3] = 0.9
    forest, _ = build_component_forest(
        probability_map=probability,
        threshold_grid_indices=(8, 5),
        denominator=10,
        min_voxels=1,
        max_voxels=100,
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        box_shape_zyx=(5, 5, 5),
    )
    return forest


def test_forest_builds_unique_parent_and_numeric_roundtrip() -> None:
    forest = _merge_forest()
    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.root.threshold_grid_index == 5
    assert len(tree.root.children) == 2
    assert all(child.parent is tree.root for child in tree.root.children)
    assert tree.lca(tree.root.children[0], tree.root.children[1]) is tree.root

    arrays = forest.to_arrays()
    restored = ComponentForest.from_arrays(arrays)
    for field, value in arrays.items():
        assert np.array_equal(restored.to_arrays()[field], value)


def test_working_tree_D_removes_ancestors_and_subtree_only() -> None:
    tree = _merge_forest().trees[0]
    left, right = tree.root.children
    working = WorkingTree(tree)
    original_voxels = {node.node_id: node.voxel_global_linear_index.copy() for node in tree.nodes}
    removed = working.delete_D(left)
    assert {node.node_id for node in removed} == {left.node_id, tree.root.node_id}
    assert not working.is_active(left)
    assert not working.is_active(tree.root)
    assert working.is_active(right)
    for node in tree.nodes:
        assert np.array_equal(node.voxel_global_linear_index, original_voxels[node.node_id])


def test_forest_rejects_high_component_partly_missing_from_low_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high_labels = np.zeros((3, 3, 3), dtype=np.int32)
    high_labels[1, 1, 1:3] = 1
    low_labels = np.zeros_like(high_labels)
    low_labels[1, 1, 2] = 1
    label_results = iter(((high_labels, 1), (low_labels, 1)))
    monkeypatch.setattr(
        forest_module.ndimage,
        "label",
        lambda *_args, **_kwargs: next(label_results),
    )

    with pytest.raises(RuntimeError, match="阈值单调性被破坏"):
        build_component_forest(
            probability_map=np.ones((3, 3, 3), dtype=np.float32),
            threshold_grid_indices=(8, 5),
            denominator=10,
            min_voxels=1,
            max_voxels=27,
            resolve_box_start=lambda centroid, shape: (0, 0, 0),
            box_shape_zyx=(3, 3, 3),
        )


def test_empty_forest_keeps_numeric_n_by_three_shapes() -> None:
    probability = np.zeros((80, 80, 80), dtype=np.float32)
    forest, _ = build_component_forest(
        probability_map=probability,
        threshold_grid_indices=(10,),
        denominator=10,
        min_voxels=1,
        max_voxels=100,
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
    )
    arrays = forest.to_arrays()
    assert arrays["bbox_min_zyx"].shape == (0, 3)
    assert arrays["centroid_zyx"].shape == (0, 3)
    assert ComponentForest.from_arrays(arrays).nodes == ()
