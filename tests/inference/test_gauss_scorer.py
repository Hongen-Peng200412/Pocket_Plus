from pathlib import Path

import numpy as np
import pytest

from src.artifacts import load_npz_strict
from src.component_lineage import ComponentForest, ComponentNode, ComponentTree
from src.inference.Gauss_Scorer import (
    GaussScorerParameters,
    add_gauss_fields,
    publish_gauss_fields,
    score_f1_centered_nodes,
)


def _forest_arrays() -> dict[str, np.ndarray]:
    eligible = ComponentNode(
        tree_id=0,
        node_id=0,
        threshold_grid_index=100,
        threshold_value=100 / 32768,
        voxel_global_linear_index=np.asarray([0, 1], dtype=np.int64),
        bbox_min_zyx=np.asarray([0, 0, 0], dtype=np.int32),
        bbox_max_zyx=np.asarray([0, 0, 1], dtype=np.int32),
        centroid_zyx=np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
        probability_mean=0.4,
        probability_max=0.8,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )
    ineligible = ComponentNode(
        tree_id=1,
        node_id=0,
        threshold_grid_index=90,
        threshold_value=90 / 32768,
        voxel_global_linear_index=np.asarray([7], dtype=np.int64),
        bbox_min_zyx=np.asarray([1, 1, 1], dtype=np.int32),
        bbox_max_zyx=np.asarray([1, 1, 1], dtype=np.int32),
        centroid_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        probability_mean=0.2,
        probability_max=0.2,
        candidate_eligible=False,
        ineligible_reason_code=1,
    )
    return ComponentForest((ComponentTree(0, (eligible,)), ComponentTree(1, (ineligible,)))).to_arrays()


def _centered_arrays() -> dict[str, np.ndarray]:
    return {
        "centered_box_index": np.asarray([0], dtype=np.int32),
        "box_start_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
        "box_shape_zyx": np.asarray([[80, 80, 80]], dtype=np.uint8),
        "box_origin_world": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "voxel_size_world": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "source_tree_id": np.asarray([0], dtype=np.int32),
        "source_node_id": np.asarray([0], dtype=np.int32),
        "source_threshold_grid_index": np.asarray([100], dtype=np.int32),
        "source_threshold_value": np.asarray([100 / 32768], dtype=np.float32),
        "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int16),
        "centered_probability": np.asarray([0.8, 0.7], dtype=np.float32),
        "voxel_final": np.ones((2, 1), dtype=np.float16),
        "voxel_aux_offsets": np.asarray([0, 0], dtype=np.int64),
        "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
        "voxel_aux_probability": np.empty(0, dtype=np.float32),
        "P_offsets": np.asarray([0, 0], dtype=np.int64),
        "P_coord_local_xyz": np.empty((0, 3), dtype=np.float32),
        "P_probability": np.empty(0, dtype=np.float32),
        "P_feat_L2": np.empty((0, 1), dtype=np.float16),
        "P_feat_L3": np.empty((0, 1), dtype=np.float16),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_global_index": np.asarray([10, 11], dtype=np.int64),
        "A_coord_local_xyz": np.asarray([[0.5, 0.5, 0.5], [20.0, 20.0, 20.0]], dtype=np.float32),
        "A_coord_centered_world": np.asarray([[-39.5, -39.5, -39.5], [-20.0, -20.0, -20.0]], dtype=np.float32),
        "A_probability": np.asarray([0.9, 0.1], dtype=np.float32),
        "A_feat_L0": np.zeros((2, 49), dtype=np.float32),
        "A_feat_L1": np.zeros((2, 1), dtype=np.float16),
        "A_feat_L2": np.zeros((2, 1), dtype=np.float16),
        "A_feat_L3": np.zeros((2, 1), dtype=np.float16),
    }


def test_score_uses_sum_weighting_and_fixed_distance_cutoff() -> None:
    parameters = GaussScorerParameters(
        lambda_positive=2.0,
        lambda_negative=1.0,
        tau_angstrom=1.0,
        gauss_score_min=2.0,
    )
    scores, selected = score_f1_centered_nodes(
        forest_arrays=_forest_arrays(),
        centered_arrays=_centered_arrays(),
        full_shape_zyx=(2, 2, 2),
        parameters=parameters,
    )

    assert scores[0] == pytest.approx(0.4 + 2.0 * 0.9 - 0.1)
    assert bool(selected[0])
    assert np.isnan(scores[1])
    assert not bool(selected[1])


def test_add_and_publish_preserve_core_fields(tmp_path: Path) -> None:
    forest = _forest_arrays()
    scores = np.asarray([2.1, np.nan], dtype=np.float32)
    selected = np.asarray([True, False], dtype=np.bool_)
    updated = add_gauss_fields(forest, scores, selected)
    ComponentForest.from_arrays(updated)
    for field, value in forest.items():
        assert np.array_equal(updated[field], value)

    path = tmp_path / "forest.npz"
    np.savez_compressed(path, **forest)
    publish_gauss_fields(path, scores, selected)
    restored = load_npz_strict(path)
    assert np.array_equal(restored["gauss_score"], scores, equal_nan=True)
    assert np.array_equal(restored["gauss_selected"], selected)
    publish_gauss_fields(path, scores, selected)


def test_forest_rejects_partial_or_inconsistent_gauss_fields() -> None:
    forest = _forest_arrays()
    with pytest.raises(KeyError, match="必须同时存在"):
        ComponentForest.from_arrays(
            {**forest, "gauss_score": np.asarray([1.0, np.nan], dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="candidate_eligible"):
        ComponentForest.from_arrays(
            {
                **forest,
                "gauss_score": np.asarray([1.0, 2.0], dtype=np.float32),
                "gauss_selected": np.asarray([False, True], dtype=np.bool_),
            }
        )
