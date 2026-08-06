import numpy as np
import pytest

from ops.Gauss_Scorer.tune import (
    PdbFacts,
    build_refinement_grid,
    enumerate_configs,
    evaluate_config,
)


def test_enumerate_configs_adds_exactly_one_baseline() -> None:
    grid = {
        "lambda_positive": [0.1, 0.2],
        "lambda_negative": [0.1, 0.2],
        "tau_angstrom": [1.0, 2.0],
        "gauss_score_min": [0.2, 0.4],
        "distance_cutoff_angstrom": 5.0,
    }
    configs = enumerate_configs(grid)
    assert len(configs) == 17
    assert configs[0] == {"config_index": 0, "kind": "baseline"}
    assert [item["config_index"] for item in configs] == list(range(17))


def test_refinement_grid_has_exactly_375_positive_configs_without_baseline() -> None:
    grid = build_refinement_grid(
        {
            "lambda_positive": 0.1,
            "lambda_negative": 0.001,
            "tau_angstrom": 1.0,
            "gauss_score_min": 1.0,
            "distance_cutoff_angstrom": 5.0,
        }
    )
    configs = enumerate_configs(grid)
    assert len(configs) == 375
    assert all(item["kind"] == "gauss" for item in configs)
    assert grid["lambda_positive"] == pytest.approx([0.08, 0.09, 0.1, 0.11, 0.12])
    assert grid["lambda_negative"] == pytest.approx(
        [0.0008, 0.0009, 0.001, 0.0011, 0.0012]
    )
    assert grid["gauss_score_min"] == pytest.approx(
        [0.3 + 0.1 * index for index in range(15)]
    )


def test_evaluate_config_uses_selected_node_union_and_reports_objective() -> None:
    facts = PdbFacts(
        pdb_id="demo",
        forest_arrays={},
        candidate_rows=np.asarray([0, 1], dtype=np.int64),
        candidate_voxels=(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([5, 6], dtype=np.int64),
        ),
        probability_mean=np.asarray([0.8, 0.2], dtype=np.float64),
        positive_terms_by_tau={1.0: np.asarray([1.0, 0.0], dtype=np.float64)},
        negative_terms_by_tau={1.0: np.asarray([0.0, 1.0], dtype=np.float64)},
        occurrence_voxels=(np.asarray([0, 1], dtype=np.int64),),
        intersections=np.asarray([[2], [0]], dtype=np.int64),
        pred_sizes=np.asarray([2, 2], dtype=np.int64),
        gt_sizes=np.asarray([2], dtype=np.int64),
        target_union=np.asarray([0, 1], dtype=np.int64),
    )
    config = {
        "config_index": 1,
        "kind": "gauss",
        "lambda_positive": 1.0,
        "lambda_negative": 1.0,
        "tau_angstrom": 1.0,
        "gauss_score_min": 0.5,
        "distance_cutoff_angstrom": 5.0,
    }
    result = evaluate_config((facts,), config)
    metrics = result["metrics"]
    assert metrics["semantic_dice_micro"] == 1.0
    assert metrics["coverage_f1_0p3"] == 1.0
    assert metrics["one_to_one_f1_0p3"] == 1.0
    assert metrics["top3_success_ratio_0p3"] == 1.0
    assert "top3_success_0p3_ratio" not in metrics
    assert metrics["objective"] == 3.0
