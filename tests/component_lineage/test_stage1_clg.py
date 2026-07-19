"""CLG split/merge 原子事件、cap 与统一 D(g) 测试。"""

from __future__ import annotations

import numpy as np

from src.artifacts.paths import Stage1ArtifactPaths
from src.component_lineage.clg import CLGEnumerationConfig, enumerate_clgs
from src.component_lineage.forest import build_component_forest, publish_component_artifacts
from src.component_lineage.overlap import build_candidate_occurrence_overlap
from src.component_lineage.structures import (
    ComponentForest,
    ComponentNode,
    ComponentTree,
    clgs_from_arrays,
)


def _forest():
    probability = np.zeros((5, 5, 5), dtype=np.float32)
    probability[2, 2, 1] = 0.9
    probability[2, 2, 2] = 0.6
    probability[2, 2, 3] = 0.9
    return build_component_forest(
        probability_map=probability,
        threshold_grid_indices=(8, 5),
        denominator=10,
        min_voxels=1,
        max_voxels=100,
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        box_shape_zyx=(5, 5, 5),
    )[0]


def test_merge_event_is_atomic_and_oldest_is_unique() -> None:
    forest = _forest()
    before = forest.to_arrays()
    result = enumerate_clgs(
        forest,
        f1_threshold_grid_index=8,
        config=CLGEnumerationConfig(1, 1, 3),
    )
    assert len(result.clgs) == 2
    assert len(result.clgs[0].candidate_nodes) == 3
    assert result.clgs[0].oldest_node is forest.trees[0].root
    assert len(result.clgs[1].candidate_nodes) == 1
    assert result.summary["n_CLG_rejected_by_node_cap"] == 0
    restored = clgs_from_arrays(result.arrays, forest)
    assert [len(clg.candidate_nodes) for clg in restored] == [3, 1]
    for field, value in before.items():
        assert np.array_equal(forest.to_arrays()[field], value)


def test_node_cap_rejects_whole_attempt_without_allocating_id() -> None:
    forest = _forest()
    result = enumerate_clgs(
        forest,
        f1_threshold_grid_index=8,
        config=CLGEnumerationConfig(1, 1, 2),
    )
    assert result.summary["n_CLG_rejected_by_node_cap"] == 1
    assert len(result.clgs) == 1
    assert result.clgs[0].CLG_id == 0
    assert len(result.clgs[0].candidate_nodes) == 1


def test_overlap_uses_local_occurrence_index_not_external_identity() -> None:
    forest = _forest()
    result = enumerate_clgs(
        forest,
        f1_threshold_grid_index=8,
        config=CLGEnumerationConfig(1, 1, 3),
    )
    left = forest.trees[0].root.children[0].voxel_global_linear_index
    right = forest.trees[0].root.children[1].voxel_global_linear_index
    overlap = build_candidate_occurrence_overlap(
        result.clgs,
        occurrence_voxel_indices={99: right, 10: left},
    )
    assert overlap["occurrence_id"].tolist() == [10, 99]
    assert "overlap_occurrence_id" not in overlap
    assert set(overlap["overlap_occurrence_index"].tolist()) <= {0, 1}
    assert overlap["candidate_occurrence_offsets"].shape == (
        sum(len(clg.candidate_nodes) for clg in result.clgs) + 1,
    )


def _wide_merge_forest(n_children: int) -> ComponentForest:
    root_voxels = np.arange(n_children, dtype=np.int64)
    root = ComponentNode(
        tree_id=0,
        node_id=0,
        threshold_grid_index=5,
        threshold_value=0.5,
        voxel_global_linear_index=root_voxels,
        bbox_min_zyx=np.zeros(3, dtype=np.int32),
        bbox_max_zyx=np.asarray([0, 0, n_children - 1], dtype=np.int32),
        centroid_zyx=np.asarray([0.0, 0.0, (n_children - 1) / 2], dtype=np.float32),
        probability_mean=0.6,
        probability_max=0.9,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )
    nodes = [root]
    for child_index in range(n_children):
        child = ComponentNode(
            tree_id=0,
            node_id=child_index + 1,
            threshold_grid_index=8,
            threshold_value=0.8,
            voxel_global_linear_index=np.asarray([child_index], dtype=np.int64),
            bbox_min_zyx=np.asarray([0, 0, child_index], dtype=np.int32),
            bbox_max_zyx=np.asarray([0, 0, child_index], dtype=np.int32),
            centroid_zyx=np.asarray([0.0, 0.0, child_index], dtype=np.float32),
            probability_mean=0.9 - child_index * 1e-4,
            probability_max=0.9,
            candidate_eligible=True,
            ineligible_reason_code=0,
            parent=root,
        )
        root.children.append(child)
        nodes.append(child)
    return ComponentForest((ComponentTree(0, nodes),))


def test_formal_depth_caps_allow_exact_32_64_and_reject_overflow() -> None:
    depth1_exact = enumerate_clgs(
        _wide_merge_forest(31), 8, CLGEnumerationConfig(1, 1, 32)
    )
    assert len(depth1_exact.clgs[0].candidate_nodes) == 32
    depth1_over = enumerate_clgs(
        _wide_merge_forest(32), 8, CLGEnumerationConfig(1, 1, 32)
    )
    assert depth1_over.summary["n_CLG_rejected_by_node_cap"] == 1

    depth2_exact = enumerate_clgs(
        _wide_merge_forest(63), 8, CLGEnumerationConfig(2, 2, 64)
    )
    assert len(depth2_exact.clgs[0].candidate_nodes) == 64
    depth2_over = enumerate_clgs(
        _wide_merge_forest(64), 8, CLGEnumerationConfig(2, 2, 64)
    )
    assert depth2_over.summary["n_CLG_rejected_by_node_cap"] == 1


def test_component_bundle_publishes_only_after_numeric_roundtrip(tmp_path) -> None:
    forest = _forest()
    result = enumerate_clgs(
        forest, 8, CLGEnumerationConfig(max_split_events=1, max_merge_events=1, max_nodes_per_CLG=3)
    )
    overlap = build_candidate_occurrence_overlap(
        result.clgs,
        occurrence_voxel_indices={17: forest.trees[0].root.voxel_global_linear_index},
    )
    paths = Stage1ArtifactPaths(tmp_path, "Find_0", "train", "1abc")
    publish_component_artifacts(
        paths=paths,
        forest=forest,
        clg_result=result,
        overlap_arrays=overlap,
        forest_summary={"connectivity": 26},
    )
    assert paths.forest_npz.is_file()
    assert paths.clg_npz.is_file()
    assert paths.overlap_npz.is_file()
    assert paths.component_summary_json.is_file()
    assert paths.role_complete_path("components").is_file()


def test_seed_order_is_f1_first_then_lower_layers_by_probability_mean() -> None:
    specs = (
        (8, 0.7),
        (8, 0.9),
        (5, 0.4),
        (5, 0.99),
        (10, 1.0),
    )
    trees = []
    for tree_id, (threshold_index, probability_mean) in enumerate(specs):
        node = ComponentNode(
            tree_id=tree_id,
            node_id=0,
            threshold_grid_index=threshold_index,
            threshold_value=threshold_index / 10.0,
            voxel_global_linear_index=np.asarray([tree_id], dtype=np.int64),
            bbox_min_zyx=np.asarray([0, 0, tree_id], dtype=np.int32),
            bbox_max_zyx=np.asarray([0, 0, tree_id], dtype=np.int32),
            centroid_zyx=np.asarray([0.0, 0.0, tree_id], dtype=np.float32),
            probability_mean=probability_mean,
            probability_max=probability_mean,
            candidate_eligible=True,
            ineligible_reason_code=0,
        )
        trees.append(ComponentTree(tree_id, (node,)))
    result = enumerate_clgs(
        ComponentForest(tuple(trees)),
        f1_threshold_grid_index=8,
        config=CLGEnumerationConfig(0, 0, 1),
    )
    assert [clg.seed_node.threshold_grid_index for clg in result.clgs] == [8, 8, 5, 5]
    assert [clg.seed_node.probability_mean for clg in result.clgs] == [0.9, 0.7, 0.99, 0.4]
    assert all(clg.seed_node.threshold_grid_index != 10 for clg in result.clgs)
