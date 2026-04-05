from __future__ import annotations

import numpy as np
import pytest

from src.inference.postprocess import merge_box_atom_results, point_semantic_segment


def test_merge_box_atom_results_supports_logit_mean_and_prob_mean() -> None:
    box_results = [
        {
            "global_atom_indices": np.array([0], dtype=np.int64),
            "atom_logits": np.array([8.0], dtype=np.float32),
            "atom_is_in_core": np.array([True], dtype=bool),
            "atom_coord_local_voxel": np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
            "box_shape_zyx": np.array([4, 4, 4], dtype=np.int64),
            "box_spatial_weight": np.array([1.0], dtype=np.float32),
            "box_confidence_weight": 1.0,
        },
        {
            "global_atom_indices": np.array([0], dtype=np.int64),
            "atom_logits": np.array([0.0], dtype=np.float32),
            "atom_is_in_core": np.array([True], dtype=bool),
            "atom_coord_local_voxel": np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
            "box_shape_zyx": np.array([4, 4, 4], dtype=np.int64),
            "box_spatial_weight": np.array([1.0], dtype=np.float32),
            "box_confidence_weight": 1.0,
        },
    ]

    atom_probs_logit_mean = merge_box_atom_results(
        box_results=box_results,
        total_atom_count=1,
        core_decay_mode="none",
        core_offset=0,
        merge_mode="logit_mean",
        voxel_size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        window_size=4,
        box_spatial_weight_sigma_ratio=0.5,
    )
    atom_probs_prob_mean = merge_box_atom_results(
        box_results=box_results,
        total_atom_count=1,
        core_decay_mode="none",
        core_offset=0,
        merge_mode="prob_mean",
        voxel_size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        window_size=4,
        box_spatial_weight_sigma_ratio=0.5,
    )

    assert atom_probs_logit_mean.shape == (1,)
    assert atom_probs_prob_mean.shape == (1,)
    assert atom_probs_logit_mean[0] > atom_probs_prob_mean[0]
    assert np.isclose(atom_probs_logit_mean[0], 0.98201376, atol=1e-6)
    assert np.isclose(atom_probs_prob_mean[0], 0.74983233, atol=1e-6)


def test_point_semantic_segment_threshold_mode_keeps_all_positive_atoms() -> None:
    atom_probs = np.array([0.95, 0.91, 0.88, 0.92, 0.20], dtype=np.float32)
    atom_coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.00, 0.12, 0.00],
            [4.00, 4.00, 4.00],
            [8.00, 8.00, 8.00],
        ],
        dtype=np.float32,
    )

    pred_atom_coords = point_semantic_segment(
        atom_probs=atom_probs,
        atom_coords=atom_coords,
        threshold=0.5,
        semantic_segment_method="threshold",
        dbscan_eps=0.25,
        dbscan_min_samples=2,
    )

    assert pred_atom_coords.shape == (4, 3)
    assert np.allclose(pred_atom_coords, atom_coords[:4])


def test_point_semantic_segment_dbscan_mode_filters_isolated_atoms() -> None:
    atom_probs = np.array([0.95, 0.91, 0.88, 0.92, 0.20], dtype=np.float32)
    atom_coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.00, 0.12, 0.00],
            [4.00, 4.00, 4.00],
            [8.00, 8.00, 8.00],
        ],
        dtype=np.float32,
    )

    pred_atom_coords = point_semantic_segment(
        atom_probs=atom_probs,
        atom_coords=atom_coords,
        threshold=0.5,
        semantic_segment_method="dbscan",
        dbscan_eps=0.25,
        dbscan_min_samples=2,
    )

    assert pred_atom_coords.shape == (3, 3)
    assert np.allclose(pred_atom_coords, atom_coords[:3])


def test_point_semantic_segment_rejects_unknown_method() -> None:
    atom_probs = np.array([0.9], dtype=np.float32)
    atom_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="semantic_segment_method"):
        point_semantic_segment(
            atom_probs=atom_probs,
            atom_coords=atom_coords,
            threshold=0.5,
            semantic_segment_method="unknown",
            dbscan_eps=2.0,
            dbscan_min_samples=3,
        )
