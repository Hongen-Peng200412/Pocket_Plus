from __future__ import annotations

import numpy as np

from src.inference.voxel_evaluator import build_instance_overlap_stats, evaluate_global_instance_matching, evaluate_voxel_mask


def test_voxel_dice_matches_formula() -> None:
    pred = np.array([1, 1, 0, 0], dtype=bool).reshape(2, 2, 1)
    gt = np.array([1, 0, 1, 0], dtype=bool).reshape(2, 2, 1)

    metrics = evaluate_voxel_mask(pred, gt)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["voxel_dice"] == 0.5


def test_instance_precision_alpha_containment() -> None:
    pred = np.zeros((4, 4, 1), dtype=np.int32)
    gt = np.zeros((4, 4, 1), dtype=np.int32)
    pred[0:2, 0:2, 0] = 1
    gt[0:2, 0:2, 0] = 1

    metrics = evaluate_global_instance_matching(pred, gt, coverage_thresholds=(0.6,))

    assert metrics["tp_cov06"] == 1
    assert metrics["num_pred_instances"] == 1


def test_instance_recall_beta_containment() -> None:
    pred = np.zeros((4, 4, 1), dtype=np.int32)
    gt = np.zeros((4, 4, 1), dtype=np.int32)
    pred[0:2, 0:2, 0] = 1
    gt[0:2, 0:2, 0] = 1

    metrics = evaluate_global_instance_matching(pred, gt, coverage_thresholds=(0.6,))

    assert metrics["tp_cov06"] == 1
    assert metrics["num_gt_instances"] == 1


def test_instance_metrics_empty_prediction() -> None:
    pred = np.zeros((4, 4, 1), dtype=np.int32)
    gt = np.zeros((4, 4, 1), dtype=np.int32)
    gt[0:2, 0:2, 0] = 1

    metrics = evaluate_global_instance_matching(pred, gt, coverage_thresholds=(0.5,))
    stats = build_instance_overlap_stats(pred, gt)

    assert metrics["num_pred_instances"] == 0
    assert metrics["num_gt_instances"] == 1
    assert metrics["tp_cov05"] == 0
    assert stats["pred_cover_matrix"].shape == (0, 1)
