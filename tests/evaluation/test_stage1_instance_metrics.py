"""旧定义 coverage、固定 Hungarian 与 top-K 指标测试. """

from __future__ import annotations

import numpy as np

from src.evaluation.instance_metrics import (
    aggregate_instance_counts,
    evaluate_instance_overlap_counts,
    evaluate_instance_masks,
    evaluate_topk_overlap_counts,
    evaluate_topk_success,
)


def test_hungarian_pairing_is_fixed_before_thresholding() -> None:
    # 连续最优配对选择两个 0.5 交叉边; 在 tau=0.55 时不会改做含一个 0.6 边的最大基数匹配. 
    counts = evaluate_instance_overlap_counts(
        intersections=np.asarray([[60, 50], [50, 0]], dtype=np.int64),
        pred_sizes=np.asarray([100, 100], dtype=np.int64),
        gt_sizes=np.asarray([100, 100], dtype=np.int64),
        coverage_thresholds=(0.55,),
    )
    assert counts.coverage_pred_hit.tolist() == [1]
    assert counts.coverage_gt_hit.tolist() == [1]
    assert counts.one_to_one_tp.tolist() == [0]


def test_coverage_allows_many_to_one_but_one_to_one_does_not() -> None:
    gt = np.zeros((1, 1, 6), dtype=np.bool_)
    gt[..., :4] = True
    pred_a = np.zeros_like(gt)
    pred_b = np.zeros_like(gt)
    pred_a[..., :2] = True
    pred_b[..., 2:4] = True
    counts = evaluate_instance_masks([pred_a, pred_b], [gt], coverage_thresholds=(0.3, 0.5))
    assert counts.coverage_pred_hit.tolist() == [2, 2]
    assert counts.coverage_gt_hit.tolist() == [1, 1]
    assert counts.one_to_one_tp.tolist() == [1, 1]
    total = aggregate_instance_counts([counts, counts])
    assert total.n_pred == 4 and total.n_gt == 2
    assert total.one_to_one_tp.tolist() == [2, 2]


def test_topk_uses_quality_score_and_no_hungarian() -> None:
    gt = np.zeros((1, 1, 5), dtype=np.bool_)
    gt[..., 0] = True
    misses = [np.roll(gt, shift, axis=2) for shift in (1, 2, 3)]
    result = evaluate_topk_success(
        pred_masks=[*misses, gt],
        gt_masks=[gt],
        candidate_scores=[0.9, 0.8, 0.7, 0.6],
        topk_values=(3, 4),
        coverage_thresholds=(0.5,),
    )
    assert result["top3_success_0p5"] == 0
    assert result["top4_success_0p5"] == 1
    sparse_result = evaluate_topk_overlap_counts(
        intersections=np.asarray([[0], [0], [0], [1]], dtype=np.int64),
        pred_sizes=np.ones(4, dtype=np.int64),
        gt_sizes=np.ones(1, dtype=np.int64),
        candidate_scores=[0.9, 0.8, 0.7, 0.6],
        topk_values=(3, 4),
        coverage_thresholds=(0.5,),
    )
    assert sparse_result == result
