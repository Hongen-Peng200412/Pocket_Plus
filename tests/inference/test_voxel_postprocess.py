from __future__ import annotations

import numpy as np

from src.inference.voxel_postprocess import postprocess_ligand_probability_map


def _run_postprocess(prob: np.ndarray, **kwargs):
    params = {
        "ligand_pred": prob,
        "origin": np.zeros(3, dtype=np.float32),
        "voxel_size": np.ones(3, dtype=np.float32),
        "threshold": 0.5,
        "min_component_voxels": 2,
        "filter_strength": "basic",
        "connectivity_policy": "7_none",
        "sigma_nearby": 1.0,
        "kernel_nearby": 3,
        "receptor_pred": None,
        "sigma_response": 1.0,
        "kernel_response": 3,
        "score_add": 1.0,
        "score_minus": 1.0,
        "voxel_score_min": 0.0,
        "instance_score_min": 0.0,
        "merge_dist": 0.0,
    }
    params.update(kwargs)
    if params["filter_strength"] == "advanced" and params["receptor_pred"] is None:
        params["receptor_pred"] = np.zeros_like(prob)
    return postprocess_ligand_probability_map(**params)


def test_basic_removes_small_components() -> None:
    prob = np.zeros((5, 5, 5), dtype=np.float32)
    prob[1:3, 1, 1] = 0.9
    prob[4, 4, 4] = 0.9

    result = _run_postprocess(prob)

    assert result.binary_mask_filtered.sum() == 2
    assert len(result.candidates) == 1


def test_connectivity_policy_7_none_keeps_single_label_after_score_gap() -> None:
    prob = np.zeros((5, 5, 5), dtype=np.float32)
    prob[2, 1:4, 2] = 0.9
    prob[2, 2, 2] = 0.51

    result = _run_postprocess(
        prob,
        filter_strength="advanced",
        connectivity_policy="7_none",
        voxel_score_min=1.1,
        min_component_voxels=1,
    )

    instance_ids = [int(v) for v in np.unique(result.instance_label_filtered) if int(v) > 0]
    assert len(instance_ids) <= 1


def test_connectivity_policy_7_7_currently_matches_7_none_after_score_gap() -> None:
    prob = np.zeros((5, 5, 5), dtype=np.float32)
    prob[2, 1, 2] = 0.9
    prob[2, 2, 2] = 0.51
    prob[2, 3, 2] = 0.9

    result = _run_postprocess(
        prob,
        filter_strength="advanced",
        connectivity_policy="7_7",
        sigma_nearby=0.1,
        voxel_score_min=1.2,
        min_component_voxels=1,
    )

    instance_ids = [int(v) for v in np.unique(result.instance_label_filtered) if int(v) > 0]
    assert len(instance_ids) == 1


def test_instance_score_min_removes_low_mean_instance() -> None:
    prob = np.zeros((5, 5, 5), dtype=np.float32)
    prob[1:3, 1, 1] = 0.6
    prob[3:5, 3, 3] = 0.95

    result = _run_postprocess(
        prob,
        filter_strength="advanced",
        connectivity_policy="7_7",
        sigma_nearby=0.1,
        min_component_voxels=1,
        instance_score_min=1.5,
    )

    assert len(result.candidates) == 1
